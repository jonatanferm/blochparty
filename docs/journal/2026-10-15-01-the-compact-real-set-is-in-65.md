# 2026-10-15 — The compact real set is in: 65 scans; r34 relaunched

- The compacting stream took the whole validation archive overnight: **65 3 T AXT2 scans** (the archive holds 101
  AXT2 volumes, 36 at 1.5 T dropped after their headers), 1.9 GB in `~/data/fastmri_c` on the mini (the raw set
  would have been ~35 GB); the real-vs-real control on the 65 reads 0.511. v17 (the 0.5 mm residual) is complete:
  62 heads, no failures. The first r34 attempt behind the stream produced nothing: simulate_heads' protocol manifest
  still defaulted to the deleted `fastmri_brain`; the defaults now follow `realism.FASTMRI = ~/data/fastmri_c`
  (simulate_heads `--manifest`, generate.py's spec). r34 relaunched (62 v16 heads, 2 x 3 x 3, the drawn receive,
  one head per process, ~8 h), scored against the 65.
- **r34** (62 v16 heads x slices 1-7 at 2 x 3 x 3 spins, the elliptical normalised receive drawn per head, one head per
  process, against the 65 compact scans): CNN AUC **0.996**, W1 0.0165, controls real-vs-real 0.511 / sim-vs-sim
  0.558. gap.py: the levels over white matter all within noise (q50 1.235 vs 1.249, q75 1.94 vs 1.96, CSF 5.17 vs
  5.02); the grain in flat white matter 0.012 against the real 0.010 (AUC 0.75; 2 x 3 x 3 took it from 0.015 to
  0.012, not to 0.010: the full 4 x 4 x 4 of r27 does); the scalp's bright decile 4.35 against 3.87 (AUC 0.62: the
  65-scan population's scalp is dimmer than the 34's 4.06 / the 21's 4.40, so the receive draw calibrated on the 34
  now sits 12 % high) and the posterior scalp 0.51 vs 0.44 (0.67). At 62 heads against 65 scans the CNN resolves
  cues of this size; the two it has left are named and cheap: spins (4 x 4 x 4, ~2.8 x the cost per slice) and the
  receive draw's centre against the larger real set. The transfer test, not this number, is M7's metric.
- **Next: the transfer number made robust, and the lesion test.** The receive draw re-centred on the 65-scan set
  (`--receive-scale 1.1,1.45`, the generator's `scanner.receive_scale`): q51 on v17 heads reads scalp over WM 4.09
  against the real 3.90 (4.35 before), CSF 4.98 vs 5.02, posterior / anterior scalp 0.50 / 0.45 vs 0.44 / 0.44.
  transfer.py now holds out the last tenth of each synthetic set, takes a second set (`--sim2`) as an ablation, and
  scores a lesion test: on held-out synthetic samples with lesions, each net's NMSE inside the lesion pixels against
  the rest of the brain and the lesion's reconstructed level (does a net trained without lesions flatten them?).
  Tonight on the mini (`night2.sh`): 60 v17 heads x 3 slices at 2 x 4 x 4 spins with 1-4 lesions each (`gen_les`)
  and the same heads without (`gen_nol`), one head per process with a retry, then three seeds of the transfer test
  against the 65 real scans (real-only now trains on ~300 slices); the raw samples are deleted afterwards.
