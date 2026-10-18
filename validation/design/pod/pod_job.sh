#!/bin/bash
# On the pod (started by mini_pipeline.sh): fill arm L's caches for /root/designs/*/arm_C.pt, run arms G and L for
# seeds 0-9 in five parallel jobs, mark them done, wait up to 30 min for the mini to fetch the results, then delete
# this pod through the Runpod API (launch.sh passed the key at creation as BP_RUNPOD_KEY).
TAG=$1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(BP_RUNPOD_KEY|RUNPOD_POD_ID)=' | sed 's/^/export /')"
cd /root/blochparty
q=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max)
t=$((q / 5))
[ "$t" -lt 1 ] && t=1
D=$(ls -d /root/designs/*/arm_C.pt | paste -sd, -)
COMMON="--slabs /root/slabs --fraction 0.5 --front rss --sampling gauss --snr 20 --epochs 100 --starts 15:60:70 --mask-seed 0 --fixed-designs $D"
echo "start $(date -u), quota $q CPUs, $t threads per job, designs $D"
OMP_NUM_THREADS=$q .venv/bin/python validation/design/lesion_design.py $COMMON --out /root/${TAG}_cache --arms L --cache-only > /root/${TAG}_cache.log 2>&1
rc=$?
echo "caches $(date -u), exit $rc"
export OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t
for b in 0 2 4 6 8; do
  (time .venv/bin/python validation/design/lesion_design.py $COMMON --out /root/${TAG}_seed$b --arms G,L --seeds 2 --seed $b
    echo "job exit $?") > /root/${TAG}_seed$b.log 2>&1 &
done
wait
echo "jobs done $(date -u)"
touch /root/job_done
for i in $(seq 1 60); do [ -f /root/fetched ] && break; sleep 30; done
echo "fetched: $([ -f /root/fetched ] && echo yes || echo no), deleting $(date -u)"
curl -s -o /dev/null -w "delete: %{http_code}\n" -X DELETE -H "Authorization: Bearer $BP_RUNPOD_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
