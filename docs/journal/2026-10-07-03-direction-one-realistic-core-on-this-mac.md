# 2026-10-07 — direction: one realistic core, on this Mac

- The owner: realism of the simulator core comes before agentic sequence design or datasets,
  and for now one engine, the fastest on this Mac. Decision: the Metal GPU kernel is the main
  engine and the PyTorch core the reference; the fused CPU kernel is frozen (new physics
  raises there); GPU gradients, the hybrid runner and other engines come later (CLAUDE.md).
- Measured CPU+GPU concurrency on the 48³ GRE: 1.15–1.5× over the faster device (ideal
  1.7–2.3×; shared power budget, bandwidth, one core driving the GPU). Parked.
- Next: moving spins, starting with free diffusion as exact Brownian segments drawn from
  a counter-based RNG (Philox) shared by the core and the GPU.
- **Diffusion** (037492a core; Metal next commit): spins as exact Brownian walkers. Per segment of
  constant gradient the displacement and the bridge integral ∫r dt are jointly Gaussian, so the
  phase is exact without sub-stepping. The normals come from Philox4x32-10 (Random123 known
  answers) keyed on (seed, spin, segment): the core and the GPU draw the same walk and agree spin
  by spin (1e-5), and merged free runs equal segment stepping. Stejskal–Tanner, isotropic and an
  anisotropic tensor: unbiased (2M walkers × 20 seeds on the GPU: 0.54296 ± 0.00009 vs
  0.54297). Speed: 2M walkers through a PGSE in 0.07 s on the GPU, 30× the core.
- **Multi-pathway diffusion**: a CPMG train with crushers and 150° refocusing (spin and
  stimulated echoes, each with its own diffusion history) against extended phase graphs with
  diffusion (Weigel et al. 2010, implemented independently in the test): 1M GPU walkers agree
  with EPG to 1e-4 on all 8 echoes (e.g. 0.9029/0.9248/0.8496 vs 0.9028/0.9247/0.8495), while
  diffusion moves them by up to 0.23. Passed on the first run.
- **Restricted diffusion** (`restriction.py`, `Phantom.restriction`: planes, cylinders,
  spheres per spin; core and Metal). Walls need sub-steps (rms ≤ size / 10, counts from Python
  so the GPU matches) and reflections. First reflection rule, the radial mirror: signals ~1 %
  high and an equilibrium 11 % depleted at the wall (a KS test caught it). Specular reflection
  fixed it: KS within the 5 % level; sphere long-time limit to 3e-4, cylinder 0.3 % (finite
  pulse), planes vs Tanner–Stejskal's series at Δ = 10/25/200 ms within 3e-3, free along the
  cylinder. GPU = core to 1e-5 (same walk, reflections included). Metal's 31-buffer limit was
  hit; the scalars are now one parameter array (plain kernels unchanged at 0.39 s).
- **Flow and motion, positions** (`motion.py`, `Phantom.motion`; core and Metal): r(t) =
  A(t)(r0 + u(t) + v t + d w(t)) + b(t), evaluated at each segment's mid-time (exact for the
  phase of a constant velocity), on top of the diffusion walk in the tissue frame. Checks:
  phase contrast 2πGvτ² exact; a constant rigid transform equals the moved phantom; a jump
  between k-space halves equals the static k-spaces stitched (Fourier shift); inflow in a
  spoiled GRE vs the time-of-flight recursion within 1 % at 0/0.05/0.15 m/s; GPU = core with
  flow, periodic displacement, rotation and diffusion at once. Test bugs on the way: the textbook
  ideal-slice inflow formula was the wrong reference (the sinc's edges tip less, so the static
  signal is 36 % higher); the reference now uses the pulse's own complex profile. The flow
  column must be long enough to feed the slice for the whole run, and the profile must be read
  before the spoiler. Not yet: fields sampled at the moving position (coils, B1⁺, scanner B0),
  and motion with gradient nonlinearity or concomitant fields (raise).
- **Fields where the spin is** (`maps.py`, `simulate(maps=FieldMaps(...))`; core and Metal): grids of
  the scanner B0, B1⁺ and complex coil sensitivities, trilinear; static spins sample once (merged
  free runs kept), moving spins every segment (coils at every ADC sample on the GPU, direct
  reduction). Exactness of trilinear for linear fields gives sharp checks: static linear maps =
  per-spin fields (1e-12); a spin moving through a linear B0 map = a static gradient (1e-12);
  a translated object under mapped coils = the moved static object; GPU = core 1e-6 with flow,
  rigid motion, diffusion and maps together. Plain kernels unchanged (0.38 s).
- **Receive chain** (`receive.py`): oversampled readouts (segments split at the finer sample
  times), a centred windowed-sinc decimation filter (even length for even os, odd for odd: no
  delay), thermal noise with channel covariance, coherent interference tones, spikes, gain
  drift, quantisation; `ReceiveChain.acquire` in the scanner's order. Checks: measured |H| =
  designed |H| to 1e-5 at os 2/3/4 including the band-edge roll-off (−20 % at 0.9 of
  Nyquist), zero phase; noise power through the filter = σ² Σh²; quantisation error step²/12
  and clipping; a tone lands in the image column of its frequency. Test mistake: I expected a
  zipper line, but a tone coherent with the TR lands at one phase-encode point; real zippers are
  lines because interference is incoherent.
- **Transmit chain** (`transmit.py`): the RF amplifier per pulse: Rapp AM/AM (gain, saturation,
  sharpness), Saleh-type AM/PM, droop with delivered energy; ideal by default. Checks: linear and
  saturation limits; a hard pulse's flip and phase after the amplifier predicted exactly
  (Mxy = i e^{iψ} sin α in blochparty's own RF phase); droop law.
- **Loop coils** (`loops.py`): quasi-static Biot–Savart loop arrays: receive conj(B1−) by
  reciprocity, B1⁺ of the driven array, the sample-loss resistance matrix ω²∫σ A_i·A_j dV as noise
  covariance (Roemer), `field_maps` into `FieldMaps`. Checks: on-axis field (1e-4), far-field
  dipole, off-axis vector potential vs elliptic integrals (1e-4), resistance symmetric, positive
  definite, identical coils fully correlated, neighbours more than far ones; mapped coils equal
  direct ones through a whole simulation (2e-3, trilinear on 1 mm).
