"""Markdown summary of the tissue database: row and source counts per tissue / property / field, and
the candidate tissue_table(field) next to the current anatomy.DEFAULT_TISSUES[field] for every
BrainWeb class (what would change if the candidate became the default).

    uv run python tools/tissuedb_report.py [--field 3.0] [--counts]
"""
import argparse
from collections import defaultdict
from dataclasses import fields

from blochparty.anatomy import DEFAULT_TISSUES, TissueProperties
from blochparty.tissuedb import load, tissue_table

UNITS = {"pd": (1, ""), "t1": (1e3, "ms"), "t2": (1e3, "ms"), "t2star": (1e3, "ms"), "chi": (1e6, "ppm"),
         "bound_fraction": (1, ""), "exchange_rate": (1, "/s"), "bound_t2": (1e6, "us"),
         "diffusivity": (1e9, "um^2/ms"), "fat_fraction": (1, "")}


def fmt(d, name):
    if d is None:
        return "-"
    scale, _ = UNITS[name]
    text = f"{d.mean * scale:.4g}" + (f" +- {d.sd * scale:.2g}" if d.sd else "")
    return text + (" (a)" if d.assumed else "")


def counts(db):
    rows = defaultdict(int)
    srcs = defaultdict(set)
    for m in db.measurements:
        key = (m.tissue, m.property, "" if m.field_T is None else f"{m.field_T:g}")
        rows[key] += 1
        srcs[key].add(m.source)
    print(f"{len(db.measurements)} rows, {len(db.sources)} sources, {len({m.tissue for m in db.measurements})} tissues\n")
    print("| tissue | property | field [T] | rows | sources |\n|---|---|---|---|---|")
    for key in sorted(rows):
        print(f"| {key[0]} | {key[1]} | {key[2]} | {rows[key]} | {len(srcs[key])} |")


def compare(db, field_T):
    cand, base = tissue_table(field_T, db=db), DEFAULT_TISSUES[field_T]
    names = [f.name for f in fields(TissueProperties)]
    print(f"\n### tissue_table({field_T:g}) vs DEFAULT_TISSUES[{field_T:g}]\n")
    print("Each cell: candidate / current; (a) = assumed; units: " +
          ", ".join(f"{n} {UNITS[n][1]}" for n in names if UNITS[n][1]) + ". Unchanged cells show one value.\n")
    print("| class | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for cls in sorted(base):
        c, b = cand.get(cls), base[cls]
        cells = []
        for n in names:
            x, y = (fmt(getattr(c, n), n) if c else "-"), fmt(getattr(b, n), n)
            same = c is not None and x == y  # same mean, SD and flag (the citation may differ)
            cells.append(y if same else f"**{x}** / {y}")
        print(f"| {cls} | " + " | ".join(cells) + " |")
    print("\nCitations of the changed candidate values:\n")
    for cls in sorted(cand):
        for n in names:
            d, e = getattr(cand[cls], n), getattr(base[cls], n) if cls in base else None
            if d is not None and fmt(d, n) != fmt(e, n):
                print(f"- {cls}.{n}: {d.citation}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", type=float, default=3.0)
    ap.add_argument("--counts", action="store_true")
    args = ap.parse_args()
    db = load()
    if args.counts:
        counts(db)
    compare(db, args.field)


if __name__ == "__main__":
    main()
