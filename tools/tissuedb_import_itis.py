"""Import the IT'IS Foundation tissue database into src/blochparty/tissuedb/measurements.csv.

Two IT'IS tables, served as JSON cell dictionaries ({"B16": "Brain (Grey Matter)", "BZ16": 1260.8, ...}):

* the material parameter database v4.2 (doi:10.13099/VIP21000-04-2), file
  materialparameterdatabasecurrent20240604.xls: column B tissue (rows 4..), relaxation times in ms as
  (mean, SD across studies, number of studies, min, max) in BZ..CD (T1 1.5 T), CE..CI (T2 1.5 T),
  CJ..CN (T1 3 T), CO..CS (T2 3 T); and the Gabriel et al. (1996) four-term Cole-Cole parameters in
  AB..AO (ef, del1, tau1 [ps], alf1, del2, tau2 [ns], alf2, sig [S/m], del3, tau3 [us], alf3, del4,
  tau4 [ms], alf4), evaluated here at the Larmor frequencies of 1.5, 3 and 7 T;
* water content v5.0 (doi:10.13099/VIP21000-05-0), file watercontent20250821.xls: columns B..G =
  name, mean [% by mass], SD, n, min, max.

Many IT'IS tissues are proxies for others (Eye Retina = Brain GM, Eye Iris = Muscle, ...): their
value tuples are copies. A tuple that appears under more than one IT'IS name is kept only for the
name listed in OWNERS (whose own data it is) and dropped everywhere else; a duplicated tuple without
an owner is dropped for all. Every IT'IS row and block is logged, taken or skipped with the reason,
in src/blochparty/tissuedb/itis_import_log.csv, with the SHA-256 of the two downloads.

    uv run python tools/tissuedb_import_itis.py            # download and rewrite the IT'IS rows
    uv run python tools/tissuedb_import_itis.py --relax a.json --water b.json   # from saved files
"""
from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "src" / "blochparty" / "tissuedb"
BASE = "https://itis.swiss/_resources/vendor/oetiker/spreadbutler/fcgi/sb.fcgi/fetch?file="
RELAX_FILE, WATER_FILE = "materialparameterdatabasecurrent20240604.xls", "watercontent20250821.xls"
RELAX_KEY, WATER_KEY = "itis_v4_2", "itis_water_v5_0"
GAMMA_BAR = 42.577478461e6  # Hz/T, CODATA 2018 (blochparty.constants.GAMMA_BAR)
EPS0 = 8.8541878128e-12
FIELDS = (1.5, 3.0, 7.0)

# IT'IS name -> (tissue id, region). Names not listed are skipped (lumens: air or gut contents).
TISSUES = {
    "Adrenal Gland": ("adrenal_gland", ""), "Bile": ("gallbladder.bile", ""), "Blood": ("blood", ""),
    "Blood Plasma": ("blood.plasma", ""), "Blood Serum": ("blood.serum", ""),
    "Blood Vessel Wall": ("vessel_wall", ""), "Bone (Cancellous)": ("bone.cancellous", ""),
    "Bone (Cortical)": ("bone.cortical", ""), "Bone Marrow (Red)": ("bone_marrow.red", ""),
    "Bone Marrow (Yellow)": ("bone_marrow.yellow", ""), "Brain": ("brain", "whole brain"),
    "Brain (Grey Matter)": ("brain.gm", ""), "Brain (White Matter)": ("brain.wm", ""),
    "Breast Fat": ("breast.fat", ""), "Breast Gland": ("breast.fibroglandular", ""),
    "Bronchi": ("bronchi", "bronchial wall"), "Cartilage": ("cartilage", ""),
    "Cerebellum": ("brain.cerebellum", ""), "Cerebrospinal Fluid": ("csf", ""), "Cervix": ("uterus.cervix", ""),
    "Commissura Anterior": ("brain.wm", "commissura anterior"),
    "Commissura Posterior": ("brain.wm", "commissura posterior"), "Connective Tissue": ("connective", ""),
    "Diaphragm": ("muscle.skeletal", "diaphragm"), "Ductus Deferens": ("ductus_deferens", ""),
    "Dura": ("dura", ""), "Epididymis": ("epididymis", ""), "Esophagus": ("esophagus", ""),
    "Extracellular Fluids": ("extracellular_fluid", ""), "Eye (Aqueous Humor)": ("eye.aqueous", ""),
    "Eye (Choroid)": ("eye.choroid", ""), "Eye (Ciliary Body)": ("eye.ciliary_body", ""),
    "Eye (Cornea)": ("eye.cornea", ""), "Eye (Iris)": ("eye.iris", ""), "Eye (Lens)": ("eye.lens", ""),
    "Eye (Retina)": ("eye.retina", ""), "Eye (Sclera)": ("eye.sclera", ""),
    "Eye (Vitreous Humor)": ("eye.vitreous", ""), "Eye Lens (Cortex)": ("eye.lens", "lens cortex"),
    "Eye Lens (Nucleus)": ("eye.lens", "lens nucleus"), "Fat": ("fat", ""),
    "Fat (Average Infiltrated)": ("fat", "average infiltrated"), "Fat (Not Infiltrated)": ("fat", "not infiltrated"),
    "Gallbladder": ("gallbladder", "wall"), "Heart Lumen": ("blood", "heart lumen"),
    "Heart Muscle": ("heart.myocardium", ""), "Hippocampus": ("brain.gm.hippocampus", ""),
    "Hypophysis": ("pituitary", ""), "Hypothalamus": ("brain.gm.deep.hypothalamus", ""),
    "Intervertebral Disc": ("intervertebral_disc", ""), "Kidney": ("kidney", ""),
    "Kidney (Cortex)": ("kidney.cortex", ""), "Kidney (Medulla)": ("kidney.medulla", ""),
    "Large Intestine": ("colon", "wall"), "Larynx": ("larynx", ""), "Liver": ("liver", ""),
    "Lung": ("lung", ""), "Lung (Deflated)": ("lung", "deflated"), "Lung (Inflated)": ("lung", "inflated"),
    "Lymph": ("lymph", ""), "Lymphnode": ("lymph_node", ""), "Mandible": ("bone.cortical", "mandible"),
    "Medulla Oblongata": ("brain.brainstem", "medulla oblongata"), "Meniscus": ("meniscus", ""),
    "Midbrain": ("brain.brainstem", "midbrain"), "Mucous Membrane": ("mucosa", ""),
    "Muscle": ("muscle.skeletal", ""), "Nerve": ("nerve", ""), "Ovary": ("ovary", ""),
    "Pancreas": ("pancreas", ""), "Penis": ("penis", ""), "Pineal Body": ("pineal", ""),
    "Placenta": ("placenta", ""), "Pons": ("brain.brainstem", "pons"), "Prostate": ("prostate", ""),
    "Salivary Gland": ("salivary_gland", ""), "SAT (Subcutaneous Fat)": ("fat.subcutaneous", ""),
    "Scalp": ("scalp", ""), "Seminal vesicle": ("seminal_vesicle", ""), "Skin": ("skin", ""),
    "Skull": ("bone.cortical", "skull"), "Skull (Cancellous)": ("bone.cancellous", "skull"),
    "Skull (Cortical)": ("bone.cortical", "skull (cortical)"), "Small Intestine": ("small_intestine", "wall"),
    "Spinal Cord": ("spinal_cord", ""), "Spleen": ("spleen", ""), "Stomach": ("stomach", "wall"),
    "Tendon\\Ligament": ("tendon", "tendon/ligament"), "Testis": ("testis", ""),
    "Thalamus": ("brain.gm.deep.thalamus", ""), "Thymus": ("thymus", ""), "Thyroid Gland": ("thyroid", ""),
    "Tongue": ("tongue", ""), "Tooth": ("tooth", ""), "Tooth (Dentine)": ("tooth.dentin", ""),
    "Tooth (Enamel)": ("tooth.enamel", ""), "Trachea": ("trachea", "wall"),
    "Ureter\\Urethra": ("urethra", "ureter/urethra"), "Urinary Bladder Wall": ("bladder.wall", ""),
    "Urine": ("bladder.urine", ""), "Uterus": ("uterus", ""), "Vagina": ("vagina", ""),
    "Vertebrae": ("bone", "vertebrae"), "Water": ("water", ""),
}
# Names whose values are their own when other names carry a copy, per table (checked against the
# duplicate groups in the 2024-06-04 / 2025-08-21 files; a group without an owner is dropped entirely).
# Dielectric owners are tissues with their own parameter set in Gabriel et al. 1996 (Part III).
OWNERS = {
    "relax": {"Blood", "Brain (Grey Matter)", "Brain (White Matter)", "Cartilage", "Cerebrospinal Fluid",
              "Muscle", "Prostate", "Breast Fat", "Pons"},
    "dielectric": {"Blood", "Brain (Grey Matter)", "Brain (White Matter)", "Brain", "Cartilage",
                   "Cerebrospinal Fluid", "Muscle", "Bone (Cortical)", "Bone (Cancellous)", "Fat", "Skin",
                   "Blood Vessel Wall", "Trachea", "Thyroid Gland", "Tendon\\Ligament", "Kidney",
                   "Lung (Inflated)", "Eye Lens (Nucleus)", "Nerve", "Stomach", "Large Intestine", "Testis"},
    "water": {"Blood", "Brain (Grey Matter)", "Brain (White Matter)", "Bone (Cortical)", "Bone (Cancellous)",
              "Fat", "Cartilage", "Muscle", "Skin"},
}
# Blocks that are not proxies but are not usable as labelled.
EXCLUDE = {("Lung", "water_content_mass"): "27.2 % (also Lung (Inflated)) is not a plausible mass fraction "
                                           "of lung tissue (deflated lung: 80.3 %); an air-diluted value",
           ("Lung (Inflated)", "water_content_mass"): "as Lung"}
# Caveats appended to the note of an IT'IS name's relaxation rows.
CAVEATS = {"Breast Fat": "; caution: T1 0.89-0.99 s is far above adipose T1 (0.37-0.42 s at 3 T in Edden et al. 2010), "
                         "so these IT'IS values probably describe breast tissue rather than pure fat",
           "Lung": "; lung parenchyma T2 depends strongly on echo spacing (susceptibility of alveolar air)"}
RELAX = {("t1", 1.5): "BZ", ("t2", 1.5): "CE", ("t1", 3.0): "CJ", ("t2", 3.0): "CO"}
COLE = ["AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO"]
HEADER = ["tissue", "region", "property", "field_T", "value", "sd", "sd_kind", "n", "unit", "species", "state",
          "temperature_C", "method", "source", "location", "note"]


def col(name: str, offset: int = 0) -> str:
    n = 0
    for ch in name:
        n = n * 26 + ord(ch) - 64
    n += offset
    out = ""
    while n:
        n, m = divmod(n - 1, 26)
        out = chr(65 + m) + out
    return out


def fetch(path, name):
    raw = Path(path).read_bytes() if path else urllib.request.urlopen(BASE + name, timeout=60).read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def cole_cole(p, f):
    """Gabriel et al. 1996 (Phys Med Biol 41:2271, Eq. 3): relative permittivity and conductivity [S/m]."""
    ef, d1, t1, a1, d2, t2, a2, sig, d3, t3, a3, d4, t4, a4 = p
    w = 2 * math.pi * f
    eps = complex(ef) + sig / (1j * w * EPS0)
    for de, tau, alpha in ((d1, t1 * 1e-12, a1), (d2, t2 * 1e-9, a2), (d3, t3 * 1e-6, a3), (d4, t4 * 1e-3, a4)):
        eps += de / (1 + (1j * w * tau) ** (1 - alpha))
    return eps.real, -eps.imag * w * EPS0


def fmt(x):
    return "" if x is None else format(float(x), ".6g")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--relax", help="saved JSON of " + RELAX_FILE)
    ap.add_argument("--water", help="saved JSON of " + WATER_FILE)
    args = ap.parse_args(argv)
    relax, relax_sha = fetch(args.relax, RELAX_FILE)
    water, water_sha = fetch(args.water, WATER_FILE)

    def rows_of(table):
        return [r for r in range(4, 1000) if isinstance(table.get(f"B{r}"), str) and table[f"B{r}"].strip()]

    # blocks: (itis name, xls row, property, field or None, tuple, table key)
    blocks = []
    for r in rows_of(relax):
        name = relax[f"B{r}"]
        for (prop, field), c0 in RELAX.items():
            blocks.append((name, r, prop, field, tuple(relax.get(f"{col(c0, k)}{r}") for k in range(5)), "relax"))
        blocks.append((name, r, "dielectric", None, tuple(relax.get(f"{c}{r}") for c in COLE), "relax"))
    for r in rows_of(water):
        blocks.append((water[f"B{r}"], r, "water_content_mass", None,
                       tuple(water.get(f"{c}{r}") for c in "CDEFG"), "water"))
    seen = defaultdict(list)
    for name, r, prop, field, values, _ in blocks:
        if any(isinstance(v, (int, float)) and v != 0 for v in values):
            seen[prop, field, values].append(name)

    out, log = [], []
    for name, r, prop, field, values, table in blocks:
        def skip(reason):
            log.append(dict(itis_name=name, xls_row=r, block=prop + ("" if field is None else f" {field:g} T"),
                            decision="skip", reason=reason, tissue=TISSUES.get(name, ("", ""))[0]))
        if name not in TISSUES:
            skip("not mapped (lumen content or air)")
            continue
        if not any(isinstance(v, (int, float)) and v != 0 for v in values):
            skip("no data")
            continue
        if prop != "dielectric" and not (isinstance(values[2], (int, float)) and values[2] >= 1):
            skip("number of studies/samples is 0")
            continue
        sharers = seen[prop, field, values]
        owners = OWNERS["dielectric" if prop == "dielectric" else table]
        if len(sharers) > 1 and name not in owners:
            owner = [s for s in sharers if s in owners]
            skip(("proxy: copy of " + owner[0]) if owner else
                 "duplicate of " + ", ".join(s for s in sharers if s != name) + " with no owner")
            continue
        if (name, prop) in EXCLUDE:
            skip(EXCLUDE[name, prop])
            continue
        tissue, region = TISSUES[name]
        region = region or name
        base = dict(tissue=tissue, region=region, species="human", state="compilation", temperature_C="")
        if prop in ("t1", "t2"):
            mean, sd, n, lo, hi = values
            c0 = RELAX[prop, field]
            out.append(base | dict(property=prop, field_T=fmt(field), value=fmt(mean / 1000),
                       sd=fmt(sd / 1000) if n >= 2 else "", sd_kind="across_studies" if n >= 2 else "", n=int(n),
                       unit="s", method="literature compilation", source=RELAX_KEY,
                       location=f"{RELAX_FILE} row {r} ({name}), cols {c0}-{col(c0, 4)}",
                       note=f"IT'IS '{name}': mean {mean:.6g} ms" + (f", SD across studies {sd:.4g} ms" if n >= 2 else "")
                       + f", {int(n)} {'study' if n == 1 else 'studies'}, range {lo:g}-{hi:g} ms" + CAVEATS.get(name, "")))
        elif prop == "dielectric":
            if any(not isinstance(v, (int, float)) for v in values):
                skip("incomplete Cole-Cole parameters")
                continue
            for field_T in FIELDS:
                f = GAMMA_BAR * field_T
                eps, sigma = cole_cole(values, f)
                for p, v, unit in (("permittivity_rel", eps, "1"), ("conductivity", sigma, "S/m")):
                    out.append(base | dict(property=p, field_T=fmt(field_T), value=format(v, ".4g"), sd="",
                               sd_kind="", n="", unit=unit, method="Gabriel 4-term Cole-Cole model, parameters from IT'IS",
                               source=RELAX_KEY, location=f"{RELAX_FILE} row {r} ({name}), cols AB-AO",
                               note=f"derived: Cole-Cole model (Gabriel et al. 1996, Phys Med Biol 41:2271) evaluated at "
                                    f"the Larmor frequency {f / 1e6:.2f} MHz; parameters ef..alf4 = "
                                    + " ".join(format(x, "g") for x in values)))
        else:
            mean, sd, n, lo, hi = values
            out.append(base | dict(property=prop, field_T="", value=fmt(mean / 100),
                       sd=fmt(sd / 100) if n >= 2 and sd else "", sd_kind="across_studies" if n >= 2 and sd else "",
                       n=int(n), unit="1", method="literature compilation (mass fraction)", source=WATER_KEY,
                       location=f"{WATER_FILE} row {r} ({name}), cols C-G",
                       note=f"IT'IS '{name}': {mean:.6g} % by mass" + (f", SD {sd:.4g} %" if n >= 2 and sd else "")
                       + f", n = {int(n)}, range {lo:g}-{hi:g} %"))
        log.append(dict(itis_name=name, xls_row=r, block=prop + ("" if field is None else f" {field:g} T"),
                        decision="take", reason="", tissue=tissue))

    path = DB / "measurements.csv"
    kept = []
    if path.exists():
        with path.open(newline="") as fh:
            kept = [row for row in csv.DictReader(fh) if row["source"] not in (RELAX_KEY, WATER_KEY)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(kept + out)
    with (DB / "itis_import_log.csv").open("w", newline="") as fh:
        fh.write(f"# {BASE}{RELAX_FILE} sha256={relax_sha}\n# {BASE}{WATER_FILE} sha256={water_sha}\n")
        w = csv.DictWriter(fh, ["itis_name", "xls_row", "block", "decision", "reason", "tissue"], lineterminator="\n")
        w.writeheader()
        w.writerows(log)
    taken = sum(e["decision"] == "take" for e in log)
    print(f"{len(out)} rows from {taken} IT'IS blocks ({len(log) - taken} skipped) -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
