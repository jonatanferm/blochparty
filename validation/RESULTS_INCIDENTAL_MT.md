# Incidental magnetization transfer in multi-slice imaging

    uv run python validation/incidental_mt.py

Against Shin, Gu & Yang, *Incidental magnetization transfer contrast by fat saturation
preparation pulses in multi-slice Look-Locker EPI* (MRM 2009, [PMC2779549](https://pmc.ncbi.nlm.nih.gov/articles/PMC2779549)):
at 3 T, white and grey matter, IR Look-Locker EPI (TR 400 ms, 16°, 5 mm slices) with fat
saturation, the relative steady state falls from ~94–98 % (1 slice) to ~70–72 % (9 slices),
about **30 % loss** (and a 20 % T1 underestimate). Setup and tissue parameters: the script's
docstring. The paper gives neither the fat-saturation pulse's duration nor its flip angle, so they are
scanned. "Bound pool: no" is the same run without MT: direct saturation of the water only.

| tissue | fat sat | bound pool | 1 sl. | 3 sl. | 5 sl. | 7 sl. | 9 sl. | loss 1→9 |
|---|---|---|---|---|---|---|---|---|
| WM | 10 ms, 90° | yes | 99.2 % | 97.6 % | 96.1 % | 94.6 % | 93.2 % | 6.1 % |
| WM | 10 ms, 110° | yes | 98.9 % | 96.5 % | 94.4 % | 92.3 % | 90.4 % | 8.6 % |
| WM | 10 ms, 180° | yes | 97.1 % | 91.7 % | 86.8 % | 82.5 % | 78.6 % | 19.0 % |
| WM | 20 ms, 110° | yes | 99.4 % | 98.1 % | 97.0 % | 95.9 % | 94.8 % | 4.6 % |
| WM | 5 ms, 110° | yes | 96.3 % | 89.9 % | 84.4 % | 79.6 % | 75.3 % | 21.8 % |
| WM | 10 ms, 90° | no | 99.7 % | 99.3 % | 98.9 % | 98.5 % | 98.1 % | 1.6 % |
| WM | 10 ms, 110° | no | 99.6 % | 99.0 % | 98.4 % | 97.8 % | 97.2 % | 2.4 % |
| WM | 10 ms, 180° | no | 99.0 % | 97.5 % | 96.0 % | 94.6 % | 93.2 % | 5.9 % |
| WM | 20 ms, 110° | no | 99.8 % | 99.5 % | 99.2 % | 99.0 % | 98.7 % | 1.2 % |
| WM | 5 ms, 110° | no | 97.5 % | 93.7 % | 90.1 % | 86.8 % | 83.8 % | 14.1 % |
| GM | 10 ms, 90° | yes | 99.3 % | 97.9 % | 96.6 % | 95.4 % | 94.2 % | 5.2 % |
| GM | 10 ms, 110° | yes | 99.0 % | 97.0 % | 95.2 % | 93.4 % | 91.7 % | 7.4 % |
| GM | 10 ms, 180° | yes | 97.5 % | 92.8 % | 88.6 % | 84.7 % | 81.2 % | 16.7 % |
| GM | 20 ms, 110° | yes | 99.5 % | 98.4 % | 97.4 % | 96.5 % | 95.6 % | 3.9 % |
| GM | 5 ms, 110° | yes | 95.9 % | 89.0 % | 83.0 % | 77.9 % | 73.3 % | 23.6 % |
| GM | 10 ms, 90° | no | 99.7 % | 99.3 % | 98.8 % | 98.4 % | 97.9 % | 1.8 % |
| GM | 10 ms, 110° | no | 99.6 % | 98.9 % | 98.3 % | 97.6 % | 97.0 % | 2.6 % |
| GM | 10 ms, 180° | no | 99.0 % | 97.3 % | 95.7 % | 94.1 % | 92.6 % | 6.5 % |
| GM | 20 ms, 110° | no | 99.8 % | 99.5 % | 99.2 % | 98.9 % | 98.5 % | 1.3 % |
| GM | 5 ms, 110° | no | 96.8 % | 91.6 % | 86.9 % | 82.7 % | 78.9 % | 18.5 % |

Shorter pulses (3 ms / 2.5 ms, 110°) saturate the water directly: 59 % / 40 % already with
1 slice, against the measured 94–98 %. So the paper's pulses were long and narrowband, and its
loss is MT, as its authors concluded.

## Reading

- **The mechanism is there**, with nothing tuned. The loss grows about linearly with the
  number of slices, and the single-slice value stays at 97–99 %. GM loses a bit less than WM
  (a smaller bound pool, but a longer T1). For long pulses MT is 2/3–3/4 of the loss. The
  same runs without the bound pool lose 1–6 %.
- **The magnitude depends on the unknown RF power** of the fat-saturation pulse (W ∝ ω1²):
  6 % (10 ms, 90°) and 9 % (110°) up to 19 % (180°). The measured ~30 % needs about twice the
  power of a 10 ms 180° Gaussian. That is plausible for vendor fat-saturation pulses (high
  flip, short, played for every slice) but not checkable without the pulse.
- **The lineshape cutoff barely matters here**: at 434 Hz the 1 kHz cut (8.6 % loss at
  10 ms 110°) and a 1 ppm cut like BMCTool's (10.3 %) differ by less than the unknown pulse.
- **Verdict:** a plausibility check, not a validation. To make it one, we need a dataset with
  its fat-saturation pulse (a Pulseq multi-slice acquisition), or a scan of our own.
