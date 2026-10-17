# 2026-10-10 — Wansapura 1999 GM/WM T2 swap in the human 3 T comparison

- `validation/anatomy_brainweb.py` propagated Wansapura et al. 1999 (JMRI 9:531, PMID 10232510)
  3 T means through the BrainWeb protocols with T2 WM 110 / GM 80 ms, the abstract's order. Every
  other 3 T measurement has GM > WM (Stanisz 2005 99 / 69 ms), and the abstract also mislabels its
  FLASH T2* values as "T2"; found while building the tissue database (`tissuedb` branch). Now
  WM 80 / GM 110 ms, with the source string saying the order is assigned by physics.
- Full text not read: Wiley serves a bot-verification page (not bypassed), ResearchGate's table
  page returns 403, Europe PMC has no open copy, Bojorquez 2017's review does not quote the values.
- Re-run on the Mac mini (7.5 s): human 3 T GM/WM T2 0.688959 -> 1.561440, PD 0.920534 ->
  1.168634, T1 0.744074 -> 0.745675 (T2' = 1/(1/T2* - 1/T2) moves with T2). Every other result
  in `validation_results.json` is identical. Independent check: the closed-form spin-echo ratio
  (before x exp(2 TE (1/80 - 1/110 ms))) gives 1.561440 and 1.168634 to 6 digits; Ernst x
  exp(-TE/T2*) gives T1 0.745521 (0.02 %, finite T2' sampling).
- RESULTS.md also carried a stale T1 ratio (0.709695): the 2026-10-08 T2* update had moved it to
  0.744074 without a re-run. No core code changed, so no snapshot.
- Commits: 049efdf (script, RESULTS), b7cf67a (LESSONS).
- After the tissuedb merge (#7) its LESSONS line saying the script uses Wansapura's abstract order
  was corrected to the new assignment.
