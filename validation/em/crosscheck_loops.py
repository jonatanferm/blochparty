"""Full-wave (mariepy, a port of MARIE 3.0) against blochparty's quasi-static loops for one loop
over a conducting sphere, at a field low enough for the quasi-static limit and at 3 T.

Runs in mariepy's own environment (GPL-3, not a blochparty dependency), with blochparty
installed there too; see validation/em/README.md.

Checks: B1+ (transmit) and conj(B1-) (receive) maps of the loop, and its sample-loss resistance:
full wave R = int sigma |E|^2 dV / |I|^2 (P = R |I|^2 / 2) against the quasi-static
w^2 int sigma |A|^2 dV, which is exact when no surface charge forms (a loop coaxial with the
sphere: E is azimuthal). The receive map fixes the conventions (mariepy: exp(+j w t),
plus = Bx + j By, minus = Bx - j By).
"""


import torch
from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.fields import circular_components, combine
from mariepy.solver import solve
from mariepy.wire import WireCoil

from blochparty import loops as qs

R_SPHERE, H, SIGMA, EPS = 0.07, 0.005, 0.55, 52.0
LOOP_R, GAP = 0.05, 0.03  # loop radius, distance of its plane above the sphere's top


def run(field_t, offset_x=0.0):
    medium = Medium(field_t)
    body = VoxelBody.sphere(R_SPHERE, H, EPS, SIGMA, padding=1)
    centre = torch.tensor([[offset_x, 0.0, R_SPHERE + GAP]], dtype=torch.float64)
    centre = centre + torch.tensor(body.coordinates()[:, body.mask].T.mean(0))[None]  # sphere centre
    coil = WireCoil.loops(centre, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64), LOOP_R, 96)
    res = solve(body, coil, medium, tol=1e-7)
    drive = combine(res.fields, res.impedance)  # 1 A into the port
    plus, minus = circular_components(medium, drive)  # (1, n1, n2, n3) [T/A]
    m = body.mask
    pts = body.coordinates()[:, m].T.to(torch.float64)  # (P, 3) x y z [m]
    e = drive.electric[0]  # (3, n1, n2, n3)
    r_fw = float((body.conductivity * e.abs().square().sum(0))[m].sum() * H**3)
    lp = qs.Loop(tuple(centre[0].tolist()), (0.0, 0.0, 1.0), LOOP_R, n_seg=256)
    B, A = qs.fields([lp], pts)
    b1p = qs.b1_plus(B)  # (P,)
    rx = qs.receive_sensitivities(B)[:, 0]  # conj(B1-)
    r_qs = float(qs.resistance(A, torch.full((len(pts),), SIGMA, dtype=torch.float64),
                               torch.full((len(pts),), H**3, dtype=torch.float64), medium.frequency)[0, 0])
    fw_p, fw_rx = plus[0][m].to(torch.complex128) / 2, torch.conj(minus[0][m].to(torch.complex128)) / 2
    def rel(a, b):
        return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())
    print(f"{field_t:6.3f} T ({medium.frequency / 1e6:6.2f} MHz), loop offset {offset_x * 1e3:4.0f} mm: "
          f"|B1+| rel. diff {rel(fw_p.abs(), b1p.abs()):.4f}, receive rel. diff {rel(fw_rx, rx.to(torch.complex128)):.4f} "
          f"(conj convention {rel(torch.conj(fw_rx), rx.to(torch.complex128)):.4f}); "
          f"R full wave {r_fw:.4e} Ohm vs quasi-static {r_qs:.4e} ({100 * (r_fw / r_qs - 1):+.2f} %)", flush=True)


if __name__ == "__main__":
    for b0 in (0.01, 0.1, 1.5, 3.0):
        run(b0)
    for b0 in (0.01, 3.0):
        run(b0, offset_x=0.04)
