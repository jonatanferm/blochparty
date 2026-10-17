"""Full-wave B1+, receive maps and noise covariance of a BrainWeb head at 3 T with mariepy.

Runs in mariepy's environment (GPL-3; a port of MARIE 3.0, github.com/pulserver/mariepy), with
blochparty installed there to load the anatomy; see validation/em/README.md. Writes, in
blochparty's anatomy frame (x right, y anterior, z superior, metres; B0 along +z):

* ``body.npz``: a quadrature birdcage (two linear modes, mariepy map format: plus / minus per
  channel, exp(+j w t), T per unit drive), infinitely long, axis through the head's centre;
* ``head8.npz``: eight loops on an elliptic cylinder 2 cm outside the head at brain height,
  1 A into each port with the others open;
* ``head8_noise.npz``: the sample-loss resistance matrix R_ij = int sigma E_i . conj(E_j) dV
  [Ohm] of the eight loops (thermal noise covariance is proportional to it), from the E fields at
  voxel centres, rescaled so its diagonal equals mariepy's exact absorbed power 2 P_i / |I|^2.

All in the piecewise-linear basis (validation: interior E vs Mie 5 % at 5 mm; the constant basis
gets losses wrong at 3 T and below).

    .venv-mariepy/bin/python validation/em/run_brainweb.py --tissues <mariepy>/examples/brainweb_tissues.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from mariepy import fields, incident, maps, tissue
from mariepy.constants import Medium
from mariepy.solver import solve, solve_incident
from mariepy.wire import WireCoil

import blochparty as bp

ROOT = Path(__file__).resolve().parents[1] / "anatomy" / "data"
# blochparty's BrainWeb classes -> labels of mariepy's brainweb_tissues.csv (Gabriel 1996 via IFAC).
# Assumptions where the table has no class: marrow as fat, vessels as muscle (blood: sigma ~1.2 S/m,
# between grey matter and CSF at 128 MHz).
LABELS = {"background": 0, "csf": 1, "gm": 2, "wm": 3, "fat": 4, "muscle": 5, "skin": 5, "skull": 7,
          "marrow": 4, "vessels": 5, "connective": 9, "dura": 9}


def say(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def head_fractions(step_m):
    head = bp.load_brainweb(4, ROOT, voxel=(step_m,) * 3, dtype=torch.float64)
    n = max(LABELS.values()) + 1
    out = torch.zeros(*head.shape[::-1], n, dtype=torch.float64)  # (x, y, z, labels)
    for name, frac in head.fractions.items():
        out[..., LABELS[name]] += frac.permute(2, 1, 0)
    return out, head.origin


def ring_loops(body, count, clearance, height):
    """Loops on an elliptic cylinder around the head (axis z) at ``height``, facing inwards."""
    pts = body.coordinates()[:, body.mask].T.double()
    sel = pts[(pts[:, 2] - height).abs() < 0.03]
    centre = (sel.min(0).values + sel.max(0).values) / 2
    semi = (sel.max(0).values - sel.min(0).values)[:2] / 2 + clearance
    a = 2 * math.pi * torch.arange(count, dtype=torch.float64) / count
    centres = torch.stack([centre[0] + semi[0] * torch.cos(a), centre[1] + semi[1] * torch.sin(a),
                           torch.full_like(a, height)], 1)
    normals = torch.stack([torch.cos(a) / semi[0], torch.sin(a) / semi[1], torch.zeros_like(a)], 1)
    normals = -normals / normals.norm(dim=1, keepdim=True)
    radius = 0.45 * 2 * math.pi * float(semi.mean()) / count
    return centres, normals, radius, centre


def write(path, name, drives, body, medium, channels, drive_unit):
    plus, minus = fields.circular_components(medium, drives)
    maps.write(path, plus, minus, body.mask, coil=name, channels=channels, frequency_hz=medium.frequency,
               drive_unit=drive_unit, origin=body.origin, resolution=body.resolution,
               frame="blochparty anatomy frame: x right, y anterior, z superior, metres; B0 along +z",
               bodies=["BrainWeb subject 04 fuzzy model (do not redistribute)"],
               data_licence="BrainWeb's terms of use (https://brainweb.bic.mni.mcgill.ca/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissues", type=Path, required=True, help="mariepy's examples/brainweb_tissues.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "em")
    ap.add_argument("--field", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=5e-3)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    medium = Medium(args.field)
    fr, origin = head_fractions(args.step)
    built = tissue.mix(fr, tissue.read_table(args.tissues), medium, args.step, origin=tuple(origin))
    body = built.body
    say(f"head: {body.n_voxels} voxels of {args.step * 1e3:.0f} mm, grid {tuple(body.shape)}, "
        f"{medium.frequency / 1e6:.2f} MHz")
    centres, normals, radius, centre = ring_loops(body, 8, 0.02, 0.02)
    say("birdcage: two modes")
    modes = [incident.birdcage(body, medium, angle=a, axis=(float(centre[0]), float(centre[1])), linear=True)
             for a in (0.0, math.pi / 2)]
    drives = solve_incident(body, medium, torch.stack([e for e, _ in modes]), torch.stack([h for _, h in modes]),
                            tol=args.tol, linear=True)
    write(args.out / "body.npz", "body", drives, body, medium, ["body-I", "body-Q"],
          "the birdcage mode whose flux density on its axis in free space is 1 uT")
    say(f"birdcage done ({time.time() - t0:.0f} s); head8: loop radius {radius * 1e3:.0f} mm")
    coil = WireCoil.loops(centres, normals, radius, max(12, math.ceil(2 * math.pi * radius / (1.5 * args.step))))
    res = solve(body, coil, medium, tol=args.tol, linear=True)
    loops = fields.combine(res.fields, res.impedance)  # 1 A into each port, others open
    write(args.out / "head8.npz", "head8", loops, body, medium, [f"head8-{i + 1}" for i in range(8)],
          "1 A into the channel's port, every other port open")
    e = fields.at_centres(loops.electric).to(torch.complex128)  # (C, 3, n1, n2, n3)
    w = (body.conductivity * args.step**3 * body.mask).to(torch.complex128)
    r = torch.einsum("cxijk,dxijk,ijk->cd", e, e.conj(), w)
    exact = 2 * fields.absorbed_power(res.operator, loops).reshape(-1).double()  # |I| = 1 A
    scale = torch.sqrt(exact / r.diagonal().real)
    r = r * scale[:, None] * scale[None, :]
    np.savez(args.out / "head8_noise.npz", resistance=r.numpy(), centre_sampled_diagonal_ratio=(1 / scale**2).numpy(),
             metadata=json.dumps({"coil": "head8", "frequency_hz": medium.frequency, "unit": "Ohm",
                                  "definition": "R_ij = int sigma E_i . conj(E_j) dV, 1 A per port"}))
    say(f"head8 done ({time.time() - t0:.0f} s); R diagonal {np.round(r.diagonal().real.numpy(), 3)} Ohm; "
        f"centre-sampled / exact diagonal {np.round((1 / scale**2).numpy(), 3)}")


if __name__ == "__main__":
    main()
