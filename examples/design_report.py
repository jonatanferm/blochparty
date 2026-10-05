"""A design report, as an agent would read it: timing, every hardware / safety constraint
with its margin and specific violations, PSF, and per-tissue signal, CNR and SNR efficiency
of an 8-coil 2D GRE of Shepp-Logan (noise from the coil noise model, by pseudo-replicas
through the RSS reconstruction).

    uv run python examples/design_report.py      # -> outputs/design_report.json
"""

import json
import time
from pathlib import Path

import blochparty as bp
from blochparty.design import Hardware, design_report
from blochparty.phantom import LABELS

N, FOV = 64, 0.22

maps = bp.shepp_logan_maps(N)
phantom = bp.phantom_from_maps(maps["pd"], maps["t1"], maps["t2"], FOV)
coils = bp.birdcage_sensitivities(phantom.pos, n_coils=8)
masks = {t: maps["label"] == LABELS.index(t) for t in ("wm", "gm", "csf", "fat")}
seq = bp.gre_2d(N, FOV, flip_deg=16.0, tr=10e-3, te=3e-3, dwell=10e-6, n_dummy=200)

t0 = time.perf_counter()
report = design_report(seq, phantom, hardware=Hardware(max_duration=2.0), coils=coils, masks=masks,
                       contrast_pairs=[("wm", "gm"), ("gm", "csf")], kspace_sigma=0.05)
print(f"report in {time.perf_counter() - t0:.2f} s; ok = {report['ok']}")
for line in report["violations"]:
    print("  -", line)
for name, c in report["constraints"].items():
    margin = "no limit" if c["margin_pct"] is None else f"margin {c['margin_pct']:+.1f} %"
    print(f"  {name:11s} {c['value']:10.4g} {c['unit']:12s} {margin}")
print("PSF:", {k: round(v, 3) for k, v in report["psf"].items() if k.startswith(("fwhm", "side"))})
print("CNR:", {k: round(v["cnr"], 1) for k, v in report["image"]["contrasts"].items()})

out = Path("outputs") / "design_report.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(f"wrote {out}")
