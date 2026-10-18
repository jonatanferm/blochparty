#!/bin/bash
# On the mini (started by launch.sh): copy the slabs to the pod, start pod_job.sh, wait for it, fetch the results to
# ~/data/<tag>/ and, once all five results.json are here, tell the pod (which then deletes itself). The temporary key
# goes in every case.
H=$1
P=$2
TAG=$3
SSH="ssh -i $HOME/.ssh/runpod_tmp -p $P -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/runpod_tmp_known -o ConnectTimeout=30 -o ServerAliveInterval=30"
OUT=$HOME/data/$TAG
trap 'rm -f ~/.ssh/runpod_tmp ~/.ssh/runpod_tmp.pub ~/.ssh/runpod_tmp_known; echo "temporary key removed $(date)"' EXIT
cd ~/data/slabs || exit 1
FILES=(--include=index.json "--include=slab_*.pt" --include=cache_snr_unit.pt --include=cache_gre32_15_60_70_gauss0.pt "--exclude=*")
want=$(ls index.json slab_*.pt cache_snr_unit.pt cache_gre32_15_60_70_gauss0.pt | wc -l | tr -d ' ')
echo "copy $want files $(date)"
rsync -a -e "$SSH" "${FILES[@]}" ./ root@$H:/root/slabs/ || exit 1
n=$($SSH root@$H 'ls /root/slabs | wc -l' | tr -d ' ')
[ "$n" = "$want" ] || { echo "the pod has $n of $want slab files"; exit 1; }
$SSH root@$H "setsid nohup bash /root/pod_job.sh $TAG < /dev/null > /root/pod_job.out 2>&1 &"
echo "job started $(date)"
for i in $(seq 1 240); do $SSH root@$H test -f /root/job_done && break; sleep 30; done
mkdir -p "$OUT"
for try in 1 2 3; do
  rsync -a -e "$SSH" --include="${TAG}_seed*/" --include=results.json --include="${TAG}_*.log" --include=gpu_util.csv \
    --include=pod_job.out --exclude='*' root@$H:/root/ "$OUT/" && break
  sleep 20
done
ok=$(ls "$OUT"/${TAG}_seed*/results.json 2> /dev/null | wc -l | tr -d ' ')
echo "results: $ok of 5 $(date)"
[ "$ok" = 5 ] && $SSH root@$H touch /root/fetched && echo "pod told"
