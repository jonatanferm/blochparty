# Full-wave coil fields: results

## mariepy against closed forms and quasi-static loops (`crosscheck_loops.py`)

One loop (radius 5 cm, 3 cm above the top) over a sphere (radius 7 cm, sigma 0.55 S/m,
eps_r 52), 5 mm voxels.

| Check | Result |
|---|---|
| B1+ and receive maps vs Biot–Savart (`loops.py`), 0.01–0.1 T | 0.3 % |
| same at 3 T | 14–16 % apart (wavelength effects, expected) |
| interior E vs Mie series, 3 T, piecewise-linear basis | 5.2 % (5 mm), 2.5 % (2.5 mm) |
| same, piecewise-constant basis | 26 % / 10 % (not used) |
| port resistance vs absorbed power | < 1 % |
| coaxial-loop noise resistance vs quasi-static w^2 int sigma A^2 | -8 / -6 % at 0.1 T, +6 / +9 % at 1.5 T, +48 / +51 % at 3 T (5 / 2.5 mm) |

The receive phase convention was fixed by Faraday's law (`tests/phantoms/test_loops.py`), after which
mariepy's `conj(plus) / 2` and `loops.receive_sensitivities` agree in the quasi-static limit.

## Import (`blochparty.em`, `tests/phantoms/test_em.py`)

- A clockwise field of amplitude 1 (as a mariepy phasor) gives |B1+| = 1 with the rotating-frame
  phase of the demodulated lab field, for three phases; a counterclockwise one gives 0.
- Real (quasi-static) loop fields written as a mariepy file reproduce `loops.field_maps` (B1+
  relative to a reference point, coil sensitivities) to 1e-5, including the (x, y, z) -> (z, y, x)
  reordering.
- Sampling inside the body is exact; outside it continues the edge value.

## BrainWeb head at 3 T (`run_brainweb.py`, `brainweb_demo.py`)

BrainWeb 04 at 5 mm (30 992 voxels, 127.73 MHz), Gabriel 1996 properties via mariepy's table.

Quadrature birdcage (infinitely long, axis through the head centre; drive picked by
`quadrature_drive`): the co-rotating mode gives mean |B1+| 1.09 uT per uT of free-space drive,
the counter-rotating 0.17 uT (it would be ~0 without dielectric effects). Over the head |B1+| /
median spans 0.90–1.37 (1st–99th percentile), cv 0.11; the centre of the brain-height slice is
1.38x the median: the central brightening seen at 3 T.

Eight loops (radius 41 mm, on an ellipse 2 cm outside the head at z = 20 mm; 1 A per port, the
others open): sample-loss resistance 1.27–1.75 Ohm per loop; the noise matrix from E at voxel
centres is 2–3 % below the exact absorbed power and is rescaled to it. Noise correlation |rho|
between channels: mean 0.22, max 0.40.

`brainweb_demo.py` (T1w spoiled GRE, 30 deg, 1.8 M spins, 37 s on the M4 GPU): over the 2 mm slab at
z = 19 mm, |B1+| / median spans 0.87–1.38 (cv 0.13); 30 deg is above the Ernst angle, so the
centre loses T1w signal. Registration check: the birdcage / ideal image ratio correlates −0.995
with the sampled map, −0.93 with its transpose and −0.97 with its mirror. Images (kept local,
BrainWeb terms): ideal; birdcage B1+; |B1+| map; eight loops with correlated noise, RSS and
Roemer combination (noise-whitened, prescan-normalised sensitivities), bright at the periphery as
expected of a surface array.

Not yet checked against measurement: no open dataset with B1+ maps and noise covariance of a
known array on a known head (a measured SNR / g-factor phantom map would be the test; REALISM R4).
