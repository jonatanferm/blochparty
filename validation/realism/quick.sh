#!/bin/bash
# Quick realism iteration (~10 min on the Mac mini): 8 heads (the even seeds of 200-215, spread over the prepared
# subjects as the full runs pick them), slices 2, 4, 6, 8 spins per voxel (--spins 2,2,2), the realistic preset with
# the IXI residual and the real echo train; score.py --quick (real vs simulated only, on those slices).
# Less robust than a full run (16 heads x slices 1-7, 32 spins: ~5 h): 24 simulated images (wide CIs) and more
# sub-voxel grain; compare quick runs with each other (2026-10-13: q27 = r27's heads, q29 = v14 heads).
#
#   validation/realism/quick.sh <tag> [prepared dir under validation/anatomy/data/ixi] [extra simulate_heads args]
#
# Writes $OUT/<tag>/ (simulations), $OUT/<tag>.log, $OUT/score_<tag>/ and prints the summary. OUT defaults to ~/data.
set -e
cd "$(dirname "$0")/../.."
UV="${UV:-$HOME/.local/bin/uv}"
OUT="${OUT:-$HOME/data}"
D=validation/anatomy/data/ixi
tag=$1; shift
if [ -n "$1" ] && [ -d "$D/$1" ]; then rm -rf $D/prepared && cp -R $D/$1 $D/prepared; echo "prepared = $1"; shift; fi
subs=($(ls $D/prepared | sed 's/IXI0*\([0-9]*\).npz/\1/' | sort -n)); n=${#subs[@]}
t0=$(date +%s); echo "$tag started $(date)"
for s in $(seq 200 2 214); do
  i=$((s - 200)); sub=${subs[$(( (i * n) / 16 ))]}
  $UV run python -W ignore validation/realism/simulate_heads.py --seeds $s-$s --slices 2-6:2 --preset realistic \
    --quality real_fine --spins 2,2,2 --source ixi --subject $sub --ordering linear_full --train header --residual \
    --out "$OUT/$tag" "$@" >> "$OUT/$tag.log" 2>&1
done
t1=$(date +%s); echo "$tag simulated: $(ls "$OUT/$tag" | wc -l | tr -d ' ') heads in $(( (t1 - t0) / 60 )) min"
$UV run python -W ignore validation/realism/score.py --sim "$OUT/$tag" --slices 2-6:2 --quick --out "$OUT/score_$tag" \
  > "$OUT/score_$tag.log" 2>&1
t2=$(date +%s); echo "$tag scored in $(( (t2 - t1) / 60 )) min (total $(( (t2 - t0) / 60 )) min) $(date)"
$UV run python -c "
import json; r = json.load(open('$OUT/score_$tag/score.json'))[0]
print('$tag CNN AUC %.3f [%.3f, %.3f], histogram %.3f, W1 %.4f' % (r['cnn_auc'], *r['cnn_ci'], r['histogram_auc'], r['intensity']['w1']))"
