"""The tissue-property database (src/blochparty/tissuedb): schema, plausibility checks that catch
transcription and unit errors, the round trip to the values anatomy.DEFAULT_TISSUES uses, and a
candidate table that assembles a phantom. The independent checks of a literature table are these
cross-checks; the values themselves keep their source and location per row."""
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import fields

import pytest
import torch

from blochparty import tissuedb
from blochparty.anatomy import DEFAULT_TISSUES, Anatomy, Distribution, TissueProperties, to_phantom
from blochparty.tissuedb import FIELD_DEPENDENT, FRACTIONS, PROPERTIES, load, pooled, query, tissue_table

DB = load()
CLASSES = ("background", "csf", "gm", "wm", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")
HEADER = ["tissue", "region", "property", "field_T", "value", "sd", "sd_kind", "n", "unit", "species", "state",
          "temperature_C", "method", "source", "location", "note"]


def test_schema():
    with (tissuedb.ROOT / "measurements.csv").open(newline="") as fh:
        assert next(csv.reader(fh)) == HEADER
    for key, s in DB.sources.items():
        assert s.doi or s.url, key
        assert s.kind in tissuedb.KINDS, key
        assert not s.doi.startswith("http"), key
    for t in DB.tissues.values():
        assert not t.parent or t.parent in DB.tissues, t.id
    classes = [t.brainweb_class for t in DB.tissues.values() if t.brainweb_class]
    assert sorted(classes) == sorted(CLASSES)  # exactly one tissue per BrainWeb class
    seen = set()
    for m in DB.measurements:
        where = f"{m.source} {m.tissue} {m.property} {m.field_T} {m.region}"
        assert m.source in DB.sources, where
        assert m.tissue in DB.tissues, where
        assert m.property in PROPERTIES and m.unit == PROPERTIES[m.property], where
        assert math.isfinite(m.value), where
        assert m.state in tissuedb.STATES, where
        assert m.location, where
        assert (m.sd is None) == (m.sd_kind == ""), where
        if m.sd is not None:
            assert m.sd >= 0 and m.sd_kind in tissuedb.SD_KINDS, where
        if m.property in FRACTIONS:
            assert 0 <= m.value <= 1, where
        if m.property in ("t1", "t2", "t2star", "bound_t2", "md", "exchange_rate", "conductivity"):
            assert m.value > 0 or (m.property == "md" and m.value >= 0), where
        if m.property in FIELD_DEPENDENT:
            assert m.field_T is not None and m.field_T > 0, where
        if m.n is not None:
            assert m.n >= 1, where
        key = (m.tissue, m.region, m.property, m.field_T, m.value, m.source, m.method, m.temperature_C)
        assert key not in seen, f"duplicate row: {where}"
        seen.add(key)
    used = {m.source for m in DB.measurements}
    assert used == set(DB.sources), set(DB.sources) ^ used


def test_units_by_magnitude():
    """Catches ms-for-s, %-for-fraction and mm^2/s-for-m^2/s: every value in its plausible SI range."""
    ranges = {"t1": (.05, 8.), "t2": (2e-5, 3.), "t2star": (5e-5, 3.), "bound_t2": (1e-6, 1e-3),
              "md": (0, 4e-9), "exchange_rate": (.1, 200), "chi": (-15, 1), "chi_qsm": (-5, 5),
              "permittivity_rel": (1, 200), "conductivity": (0, 3)}
    for m in DB.measurements:
        if m.property in ranges:
            lo, hi = ranges[m.property]
            assert lo <= m.value <= hi, (m.source, m.tissue, m.property, m.value)


def _tissue_fields():
    out = defaultdict(set)
    for m in DB.measurements:
        if m.field_T is not None:
            out[m.tissue].add(m.field_T)
    return out


# Known tensions between sources, not transcription errors: each pair is checked to stay the only one.
ORDERING_EXCEPTIONS = {
    # Peters 2006 WM T2* (71.7 ms, voxels outside a GM mask, CSF contamination per the authors) vs
    # Ma 2013 MRF WM T2 (65 ms; MRF T2 runs low)
    ("brain.wm", 1.5),
}


def test_relaxation_ordering_pooled():
    """T2 < T1 and T2* <= T2 per tissue and field, on the pooled values (all tiers)."""
    checked = 0
    for tissue, strengths in _tissue_fields().items():
        for b in strengths:
            t1, t2, star = (pooled(tissue, p, b) for p in ("t1", "t2", "t2star"))
            if t1 and t2:
                assert t2.mean < t1.mean, (tissue, b, t1.mean, t2.mean)
                checked += 1
            if t2 and star and (tissue, b) not in ORDERING_EXCEPTIONS:
                assert star.mean <= t2.mean * 1.05, (tissue, b, star.mean, t2.mean)  # 5 %: different subjects
    assert checked > 40


def test_t1_rises_with_field_within_a_source():
    groups = defaultdict(dict)
    for m in DB.measurements:
        # a compilation pools different studies at each field, so it is not one measurement series
        if m.property == "t1" and m.state not in ("physical", "compilation"):
            groups[m.source, m.tissue, m.region, m.method, m.temperature_C].setdefault(m.field_T, []).append(m.value)
    pairs = 0
    exceptions = {"csf"}  # CSF T1 does not depend on field (Rooney et al. 2007)
    for (source, tissue, *_), by_field in groups.items():
        if len(by_field) < 2 or tissue in exceptions:
            continue
        strengths = sorted(by_field)
        means = [statistics.fmean(by_field[b]) for b in strengths]
        for (b0, m0), (b1, m1) in zip(zip(strengths, means), zip(strengths[1:], means[1:])):
            assert m1 > m0 * 0.97, (source, tissue, b0, m0, b1, m1)
            pairs += 1
    assert pairs > 20


# (tissue, property, field): why the sources may differ by more than a factor of two
AGREEMENT_EXCEPTIONS = {
    ("eye.lens", "t1", 3.): "Ma 2019 lens nucleus 0.58 s (abstract, sequence not stated) vs Lie 2021 VFA 1.26 s",
}


def test_human_in_vivo_sources_agree_within_a_factor_of_two():
    """Fractions below 0.2 (fat in muscle) are compared by absolute difference (0.1) instead."""
    by = defaultdict(lambda: defaultdict(list))
    for m in DB.measurements:
        if m.species == "human" and m.state == "in_vivo" and m.property not in ("chi_qsm",):
            by[m.tissue, m.property, m.field_T][m.source].append(m.value)
    compared = 0
    for key, sources in by.items():
        if len(sources) < 2:
            continue
        means = {s: statistics.fmean(v) for s, v in sources.items()}
        if min(means.values()) <= 0 or key in AGREEMENT_EXCEPTIONS:
            continue
        compared += 1
        if key[1] in FRACTIONS and max(means.values()) < .2:
            assert max(means.values()) - min(means.values()) <= .1, (key, means)
        else:
            assert max(means.values()) / min(means.values()) <= 2, (key, means)
    assert compared > 20


# Rows that reproduce what anatomy.py's tables use: (class, field, TissueProperties field) -> query.
ROUND_TRIP = {
    ("wm", 1.5, "t1"): ("brain.wm", "t1", "stanisz2005"), ("wm", 1.5, "t2"): ("brain.wm", "t2", "stanisz2005"),
    ("wm", 3., "t1"): ("brain.wm", "t1", "stanisz2005"), ("wm", 3., "t2"): ("brain.wm", "t2", "stanisz2005"),
    ("gm", 1.5, "t1"): ("brain.gm", "t1", "stanisz2005"), ("gm", 1.5, "t2"): ("brain.gm", "t2", "stanisz2005"),
    ("gm", 3., "t1"): ("brain.gm", "t1", "stanisz2005"), ("gm", 3., "t2"): ("brain.gm", "t2", "stanisz2005"),
    ("muscle", 1.5, "t1"): ("muscle.skeletal", "t1", "stanisz2005"),
    ("muscle", 1.5, "t2"): ("muscle.skeletal", "t2", "stanisz2005"),
    ("muscle", 3., "t1"): ("muscle.skeletal", "t1", "stanisz2005"),
    ("muscle", 3., "t2"): ("muscle.skeletal", "t2", "stanisz2005"),
    ("vessels", 1.5, "t1"): ("blood", "t1", "stanisz2005"), ("vessels", 1.5, "t2"): ("blood", "t2", "stanisz2005"),
    ("vessels", 3., "t1"): ("blood", "t1", "stanisz2005"), ("vessels", 3., "t2"): ("blood", "t2", "stanisz2005"),
    ("csf", 1.5, "t1"): ("csf", "t1", "spijkerman2018"), ("csf", 3., "t1"): ("csf", "t1", "spijkerman2018"),
    ("wm", 3., "t2star"): ("brain.wm", "t2star", "shin2025"), ("gm", 3., "t2star"): ("brain.gm", "t2star", "shin2025"),
    ("wm", 7., "t1"): ("brain.wm", "t1", "caan2019"), ("wm", 7., "t2star"): ("brain.wm", "t2star", "caan2019"),
    ("gm", 7., "t1"): ("brain.gm", "t1", "caan2019"), ("gm", 7., "t2star"): ("brain.gm", "t2star", "caan2019"),
    ("fat", 3., "t1"): ("fat.subcutaneous", "t1", "debazelaire2004"),
    ("fat", 3., "t2"): ("fat.subcutaneous", "t2", "debazelaire2004"),
    ("fat", 1.5, "t1"): ("fat.subcutaneous", "t1", "itis_v4_2"), ("fat", 1.5, "t2"): ("fat.subcutaneous", "t2", "itis_v4_2"),
    ("wm", 3., "bound_fraction"): ("brain.wm", "bound_fraction", "yarnykh2016"),
    ("gm", 3., "bound_fraction"): ("brain.gm", "bound_fraction", "yarnykh2016"),
    ("wm", 3., "exchange_rate"): ("brain.wm", "exchange_rate", "yarnykh2012"),
    ("wm", 3., "bound_t2"): ("brain.wm", "bound_t2", "yarnykh2012"),
    ("gm", 3., "exchange_rate"): ("brain.gm", "exchange_rate", "yarnykh2012"),
    ("gm", 3., "bound_t2"): ("brain.gm", "bound_t2", "yarnykh2012"),
    ("muscle", 3., "bound_fraction"): ("muscle.skeletal", "bound_fraction", "sinclair2010"),
    ("muscle", 3., "bound_t2"): ("muscle.skeletal", "bound_t2", "sinclair2010"),
    ("wm", 3., "diffusivity"): ("brain.wm", "md", "helenius2002"),
    ("gm", 3., "diffusivity"): ("brain.gm", "md", "helenius2002"),
    ("vessels", 3., "diffusivity"): ("blood", "md", "funck2018"),
    ("fat", 3., "diffusivity"): ("fat.subcutaneous", "md", "weidlich2018"),
    ("csf", 3., "diffusivity"): ("water", "md", "holz2000"),
    ("fat", 3., "bound_fraction"): ("fat", "bound_fraction", "chen2010"),
    ("csf", 3., "bound_fraction"): ("csf", "bound_fraction", "smith2005"),
}


@pytest.mark.parametrize("key", sorted(ROUND_TRIP, key=str))
def test_round_trip_to_default_tissues(key):
    cls, strength, attr = key
    tissue, prop, source = ROUND_TRIP[key]
    rows = query(tissue, prop, None if prop not in FIELD_DEPENDENT else strength, source=source)
    if prop == "md" and source == "holz2000":
        rows = [m for m in rows if m.temperature_C == 37]
    if prop in FIELD_DEPENDENT and source not in ("itis_v4_2",):
        rows = [m for m in rows if m.region not in ("peripheral (subarachnoid) CSF",)]
    assert len(rows) == 1, rows
    used = getattr(DEFAULT_TISSUES[strength][cls], attr)
    rel = 5e-4 if source == "itis_v4_2" else 1e-12  # anatomy.py rounds the IT'IS means to 4 digits
    assert rows[0].value == pytest.approx(used.mean, rel=rel, abs=1e-15)
    if used.sd is not None:
        assert rows[0].sd == pytest.approx(used.sd, rel=max(rel, 2e-3 if source == "itis_v4_2" else 0))


def test_li2014_muscle_md_reproduces_default():
    """DEFAULT_TISSUES muscle diffusivity = mean and SD of the 8 thigh-muscle means of Li et al. Table 2."""
    rows = query("muscle.skeletal", "md", source="li2014")
    assert len(rows) == 8
    values = [m.value for m in rows]
    used = DEFAULT_TISSUES[3.]["muscle"].diffusivity
    assert statistics.fmean(values) == pytest.approx(used.mean, abs=.005e-9)
    assert statistics.stdev(values) == pytest.approx(used.sd, abs=.005e-9)


def test_cole_cole_reproduces_gabriel_values():
    """The import's Cole-Cole evaluation against Gabriel-model values as tabulated by IFAC
    (Andreuccetti et al., niremf.ifac.cnr.it, an independent implementation): 128 MHz."""
    def value(tissue, prop, strength=3.):
        rows = query(tissue, prop, strength, source="itis_v4_2", region={"brain.gm": "Brain (Grey Matter)",
                     "brain.wm": "Brain (White Matter)", "csf": "Cerebrospinal Fluid", "muscle.skeletal": "Muscle"}[tissue])
        assert len(rows) == 1
        return rows[0].value
    expected = {"brain.gm": (73.5, .59), "brain.wm": (52.5, .34), "csf": (84.0, 2.14), "muscle.skeletal": (63.5, .72)}
    for tissue, (eps, sigma) in expected.items():
        # 127.7 MHz (3 T) vs 128 MHz: < 0.3 % in eps and sigma for these tissues
        assert value(tissue, "permittivity_rel") == pytest.approx(eps, rel=.01), tissue
        assert value(tissue, "conductivity") == pytest.approx(sigma, rel=.02), tissue


def test_pooling_rule():
    def row(source, value, sd=None, n=None, state="in_vivo", species="human", region="r"):
        return tissuedb.Measurement("t", region, "t1", 3., value, sd, "between_subject" if sd else "", n, "s",
                                    species, state, None, "", source, "Table 1", "")
    # one value per source: its regions averaged first
    mean, sd, keys = tissuedb.combine([row("a", 1.0, .1, 10, region="x"), row("a", 2.0, .1, 10, region="y"),
                                       row("b", 3.0, .2, 30)])
    assert keys == ["a", "b"]
    assert mean == pytest.approx((10 * 1.5 + 30 * 3.0) / 40)
    within = (10 * .1**2 + 30 * .2**2) / 40
    between = (10 * (1.5 - mean) ** 2 + 30 * (3.0 - mean) ** 2) / 40
    assert sd == pytest.approx(math.sqrt(within + between))
    assert tissuedb.combine([row("a", 1.0)])[1] is None  # one source, no SD: unreported, not zero


def test_pooled_prefers_human_in_vivo_and_flags_fallbacks():
    d = pooled("brain.wm", "t1", 3.)
    assert isinstance(d, Distribution) and not d.assumed and "human in vivo" in d.citation
    assert "stanisz2005" not in d.citation  # ex vivo animal rows are a fallback only
    ex = pooled("brain.wm", "t1", 3., species="bovine")
    assert ex.assumed and ex.mean == pytest.approx(1.084)
    assert pooled("brain.wm", "t1", 12.3) is None  # T1 never borrows another field
    assert pooled("water", "md").assumed  # physical, not human in vivo


def test_default_tissues_unchanged_by_import():
    before = repr(DEFAULT_TISSUES)
    tissue_table(3.)
    assert repr(DEFAULT_TISSUES) == before
    assert DEFAULT_TISSUES[3]["wm"].t1.mean == 1.084 and DEFAULT_TISSUES[3]["fat"].t1.mean == .382


@pytest.mark.parametrize("strength", (1.5, 3.))
def test_tissue_table_assembles_a_phantom(strength):
    table = tissue_table(strength)
    assert set(table) == set(CLASSES)
    for name, p in table.items():
        assert isinstance(p, TissueProperties)
        for f in fields(p):
            d = getattr(p, f.name)
            assert d is None or d.citation
    a = Anatomy({k: torch.full((2, 2, 2), 1 / 12, dtype=torch.float64) for k in CLASSES}, (.001,) * 3, (0, 0, 0))
    ph = to_phantom(a, {strength: table}, strength, mt=True, diffusion=True, chi_to_b0=False, t2star="off")
    assert ph.bound is not None and (ph.diffusion >= 0).all() and len(ph) > 0
    assert torch.isfinite(ph.t1).all() and (ph.t2 <= ph.t1).all()
