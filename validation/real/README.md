# Realism benchmark against real scanners

Simulators are usually validated against closed forms and against each other. This folder
validates blochparty against **real raw data**: the same Pulseq `.seq` file the scanner ran,
simulated on a digital model of the scanned object whose properties come from **independent
references**, compared with what the scanner measured. See `docs/REALISM.md` §4–5 (step R3).

```bash
uv run python validation/real/fetch_data.py   # ~120 MB into validation/real/data/ (git-ignored)
uv run python validation/real/nist_irse.py    # benchmark 1, prints the table
```

## Data

Tong G, Gaspar AS, Qian E, Ravi KS, Vaughan JT, Nunes RG, Geethanath S. *Open-source magnetic
resonance imaging acquisition: data and documentation for two validated pulse sequences.* Data
in Brief 2022;42:108105. Mendeley Data doi:10.17632/8458pz722c.5, **CC BY 4.0**. Used here: the
main site's 3 T Siemens Prisma Fit acquisitions of the ISMRM/NIST system phantom (20-channel
receive): IRSE T1 mapping (10 inversion times) and multi-echo TSE T2 mapping, with their `.seq`
files (PyPulseq 1.2) and raw data.

## Benchmark 1: NIST T1 plane, inversion-recovery spin echo (`nist_irse.py`)

- **Object**: the 14 NiCl2 spheres as 3D balls. Positions from the authors' ROI masks, radii
  from the images (geometry only). T1 from the CaliberMRI/NIST references at 3 T and 20 °C,
  phantom version 1 *and* 2 (the dataset doesn't record the serial); T2, which no table gives
  for these spheres, from T1 via the 3 T NiCl2 relaxivity ratio (`nist.py` lists all sources).
  Nothing is fitted to this data.
- **Sequence**: the scanner's `.seq` files, read with `read_pulseq`: sinc pulses on a 1 µs
  raster, slice at x = +57 mm (from the excitation's frequency offset), crushers, spoilers.
- **Metric (shape)**: per sphere, ROI-mean magnitude across the 10 inversion times after one
  least-squares scale per sphere (absorbing receiver gain, coil sensitivity, PD and the fixed
  TE's T2 weighting), as a relative RMS error. It isolates the inversion-recovery physics.
- **Baseline**: the inversion-recovery formula the authors fitted, without their free offset.
- **Fitted models** are fitted on 5 inversion times and scored on the other 5.

### Pitfalls found on the way

- The slice is 57 mm off isocentre, set by an RF frequency offset; a phantom at x = 0 sees only
  far sidelobes of the pulses (tiny, meaningless signal).
- pypulseq's `calculate_kspace` on these v1.2 files puts the readout off-centre: the old format
  doesn't mark the 180° as refocusing. blochparty doesn't depend on it.
- Too few spins across the slice leave crusher dephasing unresolved: speckled spheres and
  shallow nulls (the magnitude of a few random phasors is not zero). Spin density is a
  convergence parameter here, reported with the results.
