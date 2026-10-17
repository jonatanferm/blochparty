# Full-wave coil fields (mariepy)

Full-wave B1+, receive maps and noise covariance come from
[mariepy](https://github.com/pulserver/mariepy), a Python port of MARIE 3.0 (volume integral
equations, GPL-3). It is **not** a blochparty dependency: it runs in its own environment, writes
map files (`.npz`, no pickle) and blochparty reads them with `blochparty.em`.

## Setup (once)

mariepy is not on PyPI. Built and run here from commit `1b0dc435` (2026-10-06), after reading
its source:

```bash
git clone https://github.com/pulserver/mariepy <dir>/src && cd <dir>
uv venv && uv pip install -e src -e ~/projects/blochparty
```

## Scripts

| Script | Environment | What |
|---|---|---|
| `crosscheck_loops.py` | mariepy | one loop over a conducting sphere: mariepy vs `loops.py` (quasi-static) at low field and 3 T; fixes the phasor conventions |
| `run_brainweb.py` | mariepy | BrainWeb 04 at 5 mm, 3 T: quadrature birdcage (`body.npz`), eight loops (`head8.npz`) and their sample-loss resistance (`head8_noise.npz`), written to `validation/anatomy/data/em/` (git-ignored: BrainWeb may not be redistributed) |
| `brainweb_demo.py` | blochparty | a T1w GRE slab with those fields and correlated noise vs ideal fields |

```bash
<dir>/.venv/bin/python validation/em/run_brainweb.py --tissues <dir>/src/examples/brainweb_tissues.csv
uv run python validation/em/brainweb_demo.py
```

## Conventions (`src/blochparty/em.py`)

mariepy stores `plus = mu0 (Hx + j Hy)` and `minus = mu0 (Hx - j Hy)` per channel, phasors with
exp(+j w t), on a grid indexed (x, y, z). With B0 along +z protons precess clockwise seen from +z,
so the transmit B1+ in blochparty's rotating frame is `conj(minus) / 2` and the receive weight on
Mxy is `conj(plus) / 2` (`tests/phantoms/test_em.py` checks the first against a rotating field and both
against `loops.py` in the quasi-static limit; `tests/phantoms/test_loops.py` fixes the receive weight by
Faraday's law). `field_maps` turns them into `FieldMaps` ((z, y, x), |B1+| relative to its median
over the body, coils scaled to max 1, values outside the body copied from the nearest inside
voxel). The noise covariance is proportional to `read_noise_resistance(...)`.

## Results

See `RESULTS.md`.
