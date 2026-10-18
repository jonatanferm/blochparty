#!/bin/bash
# From the laptop: run lesion_design.py's arm-L comparison (G and every design in <designs-dir>, seeds 0-9) on a
# self-deleting Runpod pod, driven by the Mac mini (docs/lessons/gpu.md):
#
#   validation/design/pod/launch.sh <tag> <designs-dir>      # <designs-dir>/<name>/arm_C.pt, one per design
#
# Creates a secure L40 (else a secure RTX 4090) with the Runpod API key in its environment (read from
# ~/.runpod/config.toml by Python, never printed), starts a GPU logger and a 2-hour dead-man delete on it, authorizes
# a throwaway key of the mini, copies the code and the designs, builds the venv, and hands over to the mini
# (mini_pipeline.sh: copy the slabs, run pod_job.sh, fetch the results to ~/data/<tag>/ on the mini). The pod deletes
# itself once the mini has the results; if setup fails here, this script removes it.
set -euo pipefail
TAG=$1
DESIGNS=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
W=$(mktemp -d)
read -r ID IP PORT < <("$ROOT/.venv/bin/python" - "$TAG" <<'EOF'
import json, os, subprocess, sys, time, tomllib
key = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))["apikey"]
for gpu in ("NVIDIA L40", "NVIDIA L40", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"):
    r = subprocess.run(["runpodctl", "pod", "create", "--name", f"bp-{sys.argv[1]}", "--gpu-id", gpu, "--cloud-type", "SECURE",
        "--image", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "--min-cuda-version", "12.8", "--ports", "22/tcp",
        "--container-disk-in-gb", "40", "--env", json.dumps({"BP_RUNPOD_KEY": key}), "--wait", "--wait-timeout", "10m"],
        capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        if "id" in d and d.get("ssh", {}).get("ip"):
            print(f"{gpu}, ${d.get('costPerHr')}/h, {d.get('vcpuCount')} vCPUs", file=sys.stderr)
            print(d["id"], d["ssh"]["ip"], d["ssh"]["port"])
            break
    except ValueError:
        pass
    print(f"{gpu}: not available", file=sys.stderr)
    time.sleep(20)
else:
    sys.exit("no pod")
EOF
)
echo "pod $ID at $IP:$PORT; ssh config $W/ssh_config"
trap 'echo "setup failed: removing pod $ID"; runpodctl pod remove "$ID" > /dev/null 2>&1 || true' ERR
cat > "$W/ssh_config" <<EOF
Host pod
  HostName $IP
  Port $PORT
  User root
  IdentityFile ~/.runpod/ssh/runpodctl-ssh-key
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $W/known_hosts
  ServerAliveInterval 30
EOF
POD="ssh -F $W/ssh_config pod"
for i in $(seq 1 12); do $POD true 2> /dev/null && break; sleep 10; done
code=$($POD 'eval "$(tr "\0" "\n" < /proc/1/environ | grep -E "^(BP_RUNPOD_KEY|RUNPOD_POD_ID)=" | sed "s/^/export /")"
  curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $BP_RUNPOD_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
  setsid bash -c "exec nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 >> /root/gpu_util.csv 2>&1" < /dev/null > /dev/null 2>&1 & disown
  setsid bash -c "sleep 7200; curl -s -o /dev/null -w \"deadman delete: %{http_code}\" -X DELETE -H \"Authorization: Bearer $BP_RUNPOD_KEY\" https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID > /root/deadman.log" < /dev/null > /dev/null 2>&1 & disown')
[ "$code" = 200 ] || { echo "the pod cannot reach the Runpod API with its key ($code)"; false; }
echo "key works from the pod; logger and 2-hour dead-man started"
ssh mini 'test -f ~/.ssh/runpod_tmp || ssh-keygen -q -t ed25519 -N "" -C "mini-to-runpod-temporary" -f ~/.ssh/runpod_tmp; cat ~/.ssh/runpod_tmp.pub' | $POD "cat >> /root/.ssh/authorized_keys"
cd "$ROOT"
rsync -az -e "ssh -F $W/ssh_config" --exclude .git --exclude .venv --exclude __pycache__ --exclude outputs --exclude .claude \
  --exclude validation/real/data --exclude validation/anatomy/data --exclude '*.eml' --exclude '*.dmg' --exclude '*.pkg' ./ pod:/root/blochparty/
rsync -az -e "ssh -F $W/ssh_config" "$DESIGNS/" pod:/root/designs/
scp -q -F "$W/ssh_config" "$HERE/pod_job.sh" pod:/root/pod_job.sh
$POD 'cd /root/blochparty && uv venv -q --system-site-packages .venv && uv pip install -q -p .venv -e . --no-deps && uv pip install -q -p .venv numpy scipy &&
  .venv/bin/python -c "import torch; assert torch.cuda.is_available()" && echo "designs: $(ls /root/designs | paste -sd " " -)"'
ssh mini "mkdir -p ~/data/$TAG/designs" && rsync -az "$DESIGNS/" "mini:data/$TAG/designs/"
scp -q "$HERE/mini_pipeline.sh" mini:runpod_pipeline.sh
ssh mini "nohup bash ~/runpod_pipeline.sh $IP $PORT $TAG < /dev/null > ~/runpod_pipeline.log 2>&1 &"
trap - ERR
echo "handed to the mini: ssh mini cat runpod_pipeline.log (results in the mini's ~/data/$TAG/)"
