"""A curated, sourced table of MR tissue properties (README.md in this directory).

Three CSV files ship with the package: ``measurements.csv`` (one published value per row, long
format, SI units), ``sources.csv`` (every source key with its DOI or URL) and ``tissues.csv`` (the
controlled vocabulary of dotted tissue ids and their BrainWeb class). ``load()`` reads them,
``query()`` selects rows, ``pooled()`` combines them into an ``anatomy.Distribution`` and
``tissue_table()`` builds a candidate {BrainWeb class: TissueProperties} table from the database,
falling back to ``anatomy.DEFAULT_TISSUES`` per field. ``DEFAULT_TISSUES`` itself is not built from
here: switching is a decision, and ``tissue_table`` returns a candidate.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

from ..anatomy import DEFAULT_TISSUES, Distribution, TissueProperties

ROOT = Path(__file__).resolve().parent

# property -> SI unit
PROPERTIES = {"t1": "s", "t2": "s", "t2star": "s", "bound_t2": "s", "md": "m^2/s", "fa": "1",
              "bound_fraction": "1", "exchange_rate": "1/s", "water_content_mass": "1", "pd": "1",
              "fat_fraction": "1", "chi": "ppm", "chi_qsm": "ppm", "permittivity_rel": "1", "conductivity": "S/m"}
FRACTIONS = {"fa", "bound_fraction", "water_content_mass", "pd", "fat_fraction"}
FIELD_DEPENDENT = {"t1", "t2", "t2star", "permittivity_rel", "conductivity"}  # field_T required, no field fallback
FIELD_PREFERRED = {"bound_fraction", "exchange_rate", "bound_t2"}  # other field allowed, marked assumed
SD_KINDS = {"between_subject", "within_subject", "specimen", "across_studies", "se", "unknown"}
STATES = {"in_vivo", "ex_vivo", "in_vitro", "compilation", "physical"}
KINDS = {"article", "abstract", "review", "database"}
# TissueProperties field -> database property
TABLE_PROPERTIES = {"pd": "pd", "t1": "t1", "t2": "t2", "chi": "chi", "t2star": "t2star",
                    "bound_fraction": "bound_fraction", "exchange_rate": "exchange_rate", "bound_t2": "bound_t2",
                    "diffusivity": "md", "fat_fraction": "fat_fraction"}
_SCALE = {"chi": 1e-6}  # database ppm -> TissueProperties SI


@dataclass(frozen=True)
class Measurement:
    tissue: str
    region: str
    property: str
    field_T: float | None
    value: float
    sd: float | None
    sd_kind: str
    n: int | None
    unit: str
    species: str
    state: str
    temperature_C: float | None
    method: str
    source: str
    location: str
    note: str


@dataclass(frozen=True)
class Source:
    key: str
    authors: str
    year: int
    title: str
    venue: str
    doi: str
    url: str
    kind: str
    note: str


@dataclass(frozen=True)
class Tissue:
    id: str
    name: str
    parent: str
    brainweb_class: str
    description: str


@dataclass(frozen=True)
class Database:
    measurements: tuple[Measurement, ...]
    sources: dict[str, Source]
    tissues: dict[str, Tissue]

    def descendants(self, tissue: str) -> list[str]:
        """Tissue ids below ``tissue`` in the hierarchy (not including it)."""
        out, frontier = [], [tissue]
        while frontier:
            kids = [t.id for t in self.tissues.values() if t.parent in frontier]
            out += kids
            frontier = kids
        return out


def _float(text):
    return float(text) if text.strip() else None


def _read(root, name):
    with (Path(root) / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


@lru_cache(maxsize=4)
def load(root: str | None = None) -> Database:
    """Read the three CSV files (from ``root``, default: the package directory)."""
    root = root or ROOT
    rows = _read(root, "measurements.csv")
    sources = {r["key"]: Source(r["key"], r["authors"], int(r["year"]), r["title"], r["venue"], r["doi"],
                                r["url"], r["kind"], r["note"]) for r in _read(root, "sources.csv")}
    tissues = {r["id"]: Tissue(r["id"], r["name"], r["parent"], r["brainweb_class"], r["description"])
               for r in _read(root, "tissues.csv")}
    meas = tuple(Measurement(r["tissue"], r["region"], r["property"], _float(r["field_T"]), float(r["value"]),
                             _float(r["sd"]), r["sd_kind"], int(r["n"]) if r["n"].strip() else None, r["unit"],
                             r["species"], r["state"], _float(r["temperature_C"]), r["method"], r["source"],
                             r["location"], r["note"]) for r in rows)
    return Database(meas, sources, tissues)


def _same_field(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def query(tissue: str, property: str, field_T: float | None = None, *, db: Database | None = None,
          descendants: bool = False, **filters) -> list[Measurement]:
    """Rows of ``tissue`` (and its descendants if asked) and ``property``; ``field_T`` keeps rows at that
    field (rows without a field are kept for field-independent properties); ``filters`` match any other
    column exactly, e.g. ``species="human", state="in_vivo", source="stanisz2005"``."""
    db = db or load()
    if property not in PROPERTIES:
        raise ValueError(f"unknown property {property!r}; one of {sorted(PROPERTIES)}")
    bad = set(filters) - {f.name for f in fields(Measurement)}
    if bad:
        raise ValueError(f"unknown filter columns {sorted(bad)}")
    ids = {tissue, *db.descendants(tissue)} if descendants else {tissue}
    out = []
    for m in db.measurements:
        if m.tissue not in ids or m.property != property:
            continue
        if field_T is not None and not _same_field(m.field_T, field_T) and \
                not (m.field_T is None and property not in FIELD_DEPENDENT | FIELD_PREFERRED):
            continue
        if all(getattr(m, k) == v for k, v in filters.items()):
            out.append(m)
    return out


def _tier(m: Measurement, property: str, field_T):
    """0 = human in vivo at the requested field (the only tier that is not an assumption)."""
    field_ok = field_T is None or _same_field(m.field_T, field_T) or property not in FIELD_DEPENDENT | FIELD_PREFERRED
    if not field_ok and property in FIELD_DEPENDENT:
        return None
    human = m.species == "human"
    order = [(True, "in_vivo"), (True, "compilation"), (True, "ex_vivo"), (True, "in_vitro"), (None, "physical"),
             (False, "in_vivo"), (False, "ex_vivo"), (False, "in_vitro")]
    for i, (h, state) in enumerate(order):
        if m.state == state and (h is None or h == human):
            return i + (0 if field_ok else len(order))
    return None


def _source_sd(m: Measurement):
    if m.sd is None:
        return None
    if m.sd_kind == "se":
        return m.sd * math.sqrt(m.n) if m.n else None
    return m.sd


def combine(rows: list[Measurement]) -> tuple[float, float | None, list[str]]:
    """The pooling rule: one value per source (mean of its rows, i.e. of its regions; within-source SD
    = RMS of its row SDs; weight = its largest n, else 1), then the n-weighted mean over sources and
    SD = sqrt(weighted mean within-source SD^2 + weighted between-source variance). SD is None for a
    single source without SDs. Returns (mean, sd, source keys)."""
    by_source = {}
    for m in rows:
        by_source.setdefault(m.source, []).append(m)
    vals, sds, weights = [], [], []
    for group in by_source.values():
        vals.append(sum(m.value for m in group) / len(group))
        s = [x for x in map(_source_sd, group) if x is not None]
        sds.append(math.sqrt(sum(x * x for x in s) / len(s)) if s else None)
        weights.append(max((m.n for m in group if m.n), default=1))
    total = sum(weights)
    mean = sum(w * v for w, v in zip(weights, vals)) / total
    between = sum(w * (v - mean) ** 2 for w, v in zip(weights, vals)) / total if len(vals) > 1 else 0.
    known = [(w, s) for w, s in zip(weights, sds) if s is not None]
    within = sum(w * s * s for w, s in known) / sum(w for w, _ in known) if known else 0.
    sd = math.sqrt(within + between) if known or len(vals) > 1 else None
    return mean, sd, list(by_source)


_TIER_NAMES = ["human in vivo", "human compilation", "human ex vivo", "human in vitro", "physical",
               "animal in vivo", "animal ex vivo", "animal in vitro"]


def pooled(tissue: str, property: str, field_T: float | None = None, *, db: Database | None = None,
           descendants: bool = False, **filters) -> Distribution | None:
    """Pool the best available rows into a Distribution (None if there are none): human in vivo at
    ``field_T`` if any (assumed=False); otherwise the first non-empty tier of human compilation, ex
    vivo, in vitro, physical, then animal (assumed=True). T1, T2, T2* and dielectric values never fall
    back to another field; MT parameters do (assumed=True); other properties do not depend on field."""
    db = db or load()
    rows = query(tissue, property, db=db, descendants=descendants, **filters)
    tiers = {}
    for m in rows:
        t = _tier(m, property, field_T)
        if t is not None:
            tiers.setdefault(t, []).append(m)
    if not tiers:
        return None
    best = min(tiers)
    mean, sd, keys = combine(tiers[best])
    unit = PROPERTIES[property]
    scale = _SCALE.get(property, 1.)
    tier = _TIER_NAMES[best % len(_TIER_NAMES)] + ("" if best < len(_TIER_NAMES) else ", other field")
    where = ", ".join(f"{k} ({'; '.join(sorted({m.location for m in tiers[best] if m.source == k}))})" for k in keys)
    citation = (f"tissuedb pooled {tissue} {property}" + (f" at {field_T:g} T" if field_T is not None else "")
                + f" [{unit}] from {len(keys)} source(s), tier {tier}: {where}")
    return Distribution(mean * scale, None if sd is None else sd * scale, citation, best != 0)


def tissue_table(field_T: float, *, db: Database | None = None) -> dict[str, TissueProperties]:
    """A candidate {BrainWeb class: TissueProperties} at ``field_T`` from the database (tissues.csv
    names one tissue per class). Each property is pooled from that tissue's rows, else from its
    descendants' (assumed=True), else the current ``DEFAULT_TISSUES[field_T]`` entry is used. A class
    is left out when a required property (pd, t1, t2, chi) has neither. T2* above T2 is clipped to T2
    (assumed=True). A signalling class without MT or diffusion data gets bound_fraction 0 and
    diffusivity 0, marked as assumptions, so ``to_phantom(mt=True, diffusion=True)`` runs."""
    db = db or load()
    base_table = DEFAULT_TISSUES.get(field_T, {})
    out = {}
    for t in sorted(db.tissues.values(), key=lambda t: t.id):
        cls = t.brainweb_class
        if not cls or cls in out:
            continue
        base = base_table.get(cls)
        values = {}
        for name, prop in TABLE_PROPERTIES.items():
            d = pooled(t.id, prop, field_T, db=db)
            if d is None and db.descendants(t.id):
                d = pooled(t.id, prop, field_T, db=db, descendants=True)
                if d is not None:
                    d = replace(d, citation=d.citation + "; pooled over descendant tissues", assumed=True)
            if d is None and base is not None:
                d = getattr(base, name)
            values[name] = d
        if any(values[k] is None for k in ("pd", "t1", "t2", "chi")):
            continue
        # to_phantom(mt=True / diffusion=True) needs these for every signalling class: say so explicitly
        if values["pd"].mean > 0:
            for key, what in (("bound_fraction", "no MT data, no semi-solid pool"), ("diffusivity", "no diffusion data, D = 0")):
                if values[key] is None:
                    values[key] = Distribution(0., None, f"Assumption: tissuedb has {what} for {t.id}", True)
        star, t2 = values["t2star"], values["t2"]
        if star is not None and star.mean > t2.mean:
            values["t2star"] = Distribution(t2.mean, None, "clipped to T2: " + star.citation, True)
        out[cls] = TissueProperties(**values)
    return out


__all__ = ["Database", "Measurement", "Source", "Tissue", "PROPERTIES", "load", "query", "pooled", "combine",
           "tissue_table"]
