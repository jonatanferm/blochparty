# Lessons: Pulseq and sequences

- **Track k from the excitation centre, not the TR start**: integrating from the TR start
  showed a fake ±350 /m slice-axis residual at every echo and briefly pointed at the importer.

- **Pulseq files can contain dummy TRs without ADC**: the NIST TSE file plays 2 × (1 dummy +
  128 lines); split TRs by timing and count readout TRs separately.

- **Slices are positioned by RF frequency offset.** The NIST IR-SE slice is at x = +57 mm
  (15.2 kHz / 266.7 kHz/m); a phantom at x = 0 only sees far sidelobes (tiny, meaningless
  signal). Compute f/G from the `.seq` (`nist_irse.slice_position`). With blochparty's
  default "physical" reading, frequency f excites at +f/G; "mrzero" mirrors it.

- **Pulseq v1.2 files don't mark RF use**, so pypulseq's `calculate_kspace` treats the 180° as
  an excitation and puts spin-echo readouts off-centre (545–1050 /m instead of ±256).
  Don't rely on it for old files; the Bloch simulation doesn't need it. pypulseq 1.5 reads
  v1.2 files with default rasters (warnings only).

- **Real RF is sampled on a 1 µs raster: ~80 % of all segments.** Merging 4 RF samples
  (preserving integrals) is 3× faster and changes results ≤ 0.4 %; merging 10 changes them
  5 %. (`nist_irse.coarsen_rf`)

- **Two gyromagnetic constants exist**: CODATA 42.577478518 MHz/T (`constants.GAMMA_BAR`, use
  for physics) and pypulseq's 42.576 MHz/T (`PULSEQ_GAMMA_BAR`, only for hardware limits so
  gradient designs equal pypulseq's).

- **A block pulse without a frequency offset is one segment** in `read_pulseq` (shaped pulses
  on the RF raster are stepped per sample; constant pieces are not cut). Relaxation is then
  split around one rotation: harmless for a 100 µs hard pulse, but a 1 s CW saturation pulse
  came out 0.6 off in Mz against BMCTool. Opt-in `max_rf_step=` cuts it (defaults stay
  bit-identical).

- **The antiparallel-spoke delay estimator is biased by unequal axis delays** (2026-10-08,
  benchmark 6): the partner at theta + pi is shifted along the spoke by the same d(theta) but
  sideways by the opposite amount ((dx - dy) sin cos), so the pair samples two parallel lines; an
  off-centre object then adds a phase tilt to their cross-power. Exact for dx = dy or a centred
  symmetric object; 0.4 us off at |dx - dy| = 5 us for an ellipse 2 cm off centre. Check an
  estimator on simulated data of an object like the measured one before trusting it, and vary
  the object (rotating it flipped the sign of the bias).

- **Modern vendor delay compensation leaves sub-microsecond residuals** (Cima.X: 0.2 / 0.65 us,
  < 0.07 samples at 10 us): a "gradient delay" benchmark on such data checks the timing model and
  the estimator, not a visible artefact.

- **A GIRF is defined in a sampling frame** (2026-10-08, benchmark 7): the thin-slice output is
  the mean gradient between ADC samples, the authors' input point samples of the waveform; the
  two frames are half a dwell apart, so delays (and the phase of H) differ by 2.5 us at 5 us
  dwell. State the frame with any GIRF or delay, and convert before applying a published one.

- **Thin-slice B0 terms are drift-limited**: the slice pair's common-mode phase picks up the drift
  left after reference correction (~10 Hz here) and slice-position asymmetry (a DC term of
  -8 to -14 Hz per mT/m from ~0.25 mm), so check a B0 eddy term with both blip polarities (only the
  odd part is eddy current) and as accumulated phase, not per-sample frequency (noise ~70 Hz).

- **Unit trap in `GIRF` B0 rows**: taps are Hz per (Hz/m) because blochparty gradients are Hz/m;
  a B0 transfer function measured in Hz per (mT/m) must be divided by gamma_bar (here that slip
  inflated the B0 term by 4e4 before it was caught).

- **Pulseq-CEST trains reset the RF phase per pulse** (2026-10-08, benchmarks 8-9): each pulse's
  phase advances by 2 pi f t_p (only while RF is on), so in the frame of the offset f, pulse k
  carries -2 pi f k t_d and, with no spoiling between pulses, that matters near water. Simulate
  each offset in its RF frame (rf * exp(+i 2 pi f t), spins at dB0 - f): exact, slow envelopes,
  100 us steps instead of 1 us. To batch all offsets as spins, move the per-pulse phase step into
  the gaps as a rotation (a gap gradient with spin positions x = dpsi / (2 pi G t_gap)): exact to
  1e-5, 58 offsets in 15 s.

- **A .seq's ppm scale is its own**: definitions say B0 3 T while the Prisma fit / Vida run at
  2.894 T; offsets are played in Hz (ppm x 127.73), pool shifts must be placed in Hz at the real
  frequency (raw header lFrequency).

- **pypulseq rounds on both sides** (pypulseq 1.5.0, found writing M4's exporter, 2026-10-09):
  its writer formats amplitudes as `{:12g}`, six significant digits (a 1034070 Hz/m gradient
  becomes 1.03407e+06, ~5e-7), and `Sequence.read` defaults to `remove_duplicates=True`, which
  rounds the event libraries before merging them (amplitudes to six digits, timings to 1 us).
  A lossless round trip needs both a full-precision writer and `remove_duplicates=False`. Every
  scanner file in `validation/real/data` (up to 1.1e8 segments) reads bit-identically either way.

- **Pulseq's `[GRADIENTS]` and `[TRAP]` share one id space**: the block table's gradient ids do not
  say which section they mean, so separate counters make a block pick up the wrong event
  (pypulseq then reports events longer than their block).

- **pypulseq's shape reader runs forever if the file ends right after the last shape sample**: its
  line reader returns -1 at end of file, and the sample loop stops only at a blank or comment
  line. Its own files always follow the shapes with `[SIGNATURE]`; write one (md5 of the content
  above it, the preceding newline belonging to the signature).

- **Values in us or ns do not always survive x / 1e-6 * 1e-6**: 60 * 1e-6 is not 6e-05. Writing the
  decimal that the reader's multiplication maps back to the same double (try x / unit, a few
  roundings and neighbours) makes a re-exported file bit-identical; segment edges computed from
  absolute times can still differ in the last bit (1.7e-18 s on a scanner file).

- **A round TE puts a GRE half a dwell off the gradient raster** (M5 example, 2026-10-09):
  `gre_cartesian` measures TE to the centre of ADC sample n/2, (n/2 + 1/2) dwell after the ADC
  starts, so with a 10 us dwell TE 2.5 ms or 20 ms leaves a fill block 5 us off the 10 us raster
  and `write_pulseq` refuses it; TE 2.505 / 20.005 ms writes and passes `check_timing`. A
  scanner rounds TE itself; the builder does not (changing that would move existing sequences).

- **Other slices' pulses in a multi-slice simulation** (2026-10-10, `tse_2d(slice_order=...)`):
  a far slice's phase ramp (~28 kHz at 7 cm) aliases on a coarse raster. A first fix played the
  envelopes without the ramp and shifted them by `b0_offset = -G z_j`: 0.8 % off in one test but
  2-3 % off at the slab edges, where spins sit inside a neighbour's band and the shifted frame
  does not reproduce its coherent pathways. What works: the neighbours as their real pulses, and
  every farther pulse as one segment of the same RF energy that only the bound pool sees
  (`Sequence.bound_only`, `rf_freq` at G (z_j - z)): 0.71 % against every pulse exact, 3x a single
  slice on the GPU instead of 6x. 180 deg pulses still nutate water a few kHz outside their band,
  so "band-limited" holds for the far slices, not the neighbours.

- **Hiding a pulse from the water by a frame offset breaks float32** (2026-10-10): the first
  bound-only pulses put the water 1 MHz off resonance for the pulse (an integer number of turns,
  exact in float64). In float32 each 100 us pulse leaves ~1e-3 rad of phase (100 turns at 2^-24),
  harmless for WM (T2 69 ms forgets it) but CSF (T2 2 s) accumulated it over 16 slices' pulses:
  Metal vs core 0.45 % in signal and 2 % in Mz, 3 % in Mz on Shepp-Logan at TR 0.8 s. A flag the
  kernel reads (water precesses, the bound pool saturates) gives 6e-6 (CSF) and 7e-7 (WM). Express
  "this does not act on X" as structure, not as a large number that cancels.

- **Count segments by kind before optimising a sequence**: delays padded in pieces and gradient
  ramps of pulses that only matter through their energy made up half of a multi-slice TSE
  (42k -> 23k segments after merging delays and dropping those ramps).
