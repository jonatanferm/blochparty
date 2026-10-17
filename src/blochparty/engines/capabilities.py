"""What each engine accepts, forward and with gradients: one table and one check.

A case's features (``features``) are named after what the inputs use; ``SUPPORTED`` lists, per engine and
direction, the features it runs, and ``REFUSED`` the combinations it refuses although it runs each part alone.
``check`` raises ``NotImplementedError`` for anything else, with one wording for every engine. The capability
matrix of ``docs/ENGINES.md`` is the same table cell by cell: ``tests/engines/test_engines.py`` checks the
declaration against every cell, and runs every engine on every cell it can.
"""

from __future__ import annotations

FEATURES = {
    "b1": "B1+ (Phantom.b1)",
    "maps": "field maps (FieldMaps)",
    "bound": "a bound pool (Phantom.bound)",
    "generalized": "the generalized Bloch bound pool",
    "pools": "free exchanging pools (Phantom.pools)",
    "diffusion": "diffusion (Phantom.diffusion)",
    "motion": "motion (Phantom.motion)",
    "scanner": "the scanner field model (scanner=)",
    "bound_only": "bound-pool-only RF segments (Sequence.bound_only)",
    "m_init": "an initial magnetization (m_init)",
}
ENGINES = ("core", "metal", "cuda", "chunked")
_ALL = frozenset(FEATURES)
_GPU_GRAD = _ALL - {"generalized", "pools", "diffusion", "motion", "bound_only"}
SUPPORTED = {
    ("core", "forward"): _ALL, ("core", "gradients"): _ALL,
    ("metal", "forward"): _ALL, ("metal", "gradients"): _GPU_GRAD,
    ("cuda", "forward"): _ALL, ("cuda", "gradients"): _GPU_GRAD,
    ("chunked", "forward"): frozenset({"b1"}), ("chunked", "gradients"): frozenset({"b1"}),
}
_GPU_REFUSED = (frozenset({"pools", "generalized"}), frozenset({"bound_only", "pools"}),
                frozenset({"bound_only", "generalized"}))
REFUSED = {"core": (frozenset({"bound_only", "generalized"}),), "metal": _GPU_REFUSED, "cuda": _GPU_REFUSED,
           "chunked": ()}


def features(phantom, seq, m_init=None, scanner=None, maps=None) -> frozenset:
    """The ``FEATURES`` a case uses."""
    bound_only = getattr(seq, "bound_only", None)
    present = {
        "b1": getattr(phantom, "b1", None) is not None,
        "maps": maps is not None,
        "bound": phantom.bound is not None,
        "generalized": phantom.bound is not None and phantom.bound.model == "generalized",
        "pools": bool(getattr(phantom, "pools", None)),
        "diffusion": getattr(phantom, "diffusion", None) is not None,
        "motion": getattr(phantom, "motion", None) is not None,
        "scanner": scanner is not None,
        "bound_only": bound_only is not None and any(bound_only),
        "m_init": m_init is not None,
    }
    return frozenset(k for k, v in present.items() if v)


def unsupported(engine: str, used: frozenset, grad: bool = False) -> list[str]:
    """The features (and refused combinations) of ``used`` that ``engine`` does not run, as ``FEATURES`` names."""
    missing = set(used - SUPPORTED[(engine, "forward")])
    if grad:
        missing |= used - SUPPORTED[(engine, "gradients")]
    out = sorted(missing)
    for combo in REFUSED[engine]:
        if combo <= used and not combo & missing:
            out.append(" with ".join(sorted(combo)))
    return out


def check(engine: str, used: frozenset, grad: bool = False) -> None:
    """Raise ``NotImplementedError`` if ``engine`` does not run a case with features ``used`` (gradients: ``grad``)."""
    bad = unsupported(engine, used, grad)
    if bad:
        what = "; ".join(" with ".join(FEATURES[k] for k in b.split(" with ")) for b in bad)
        kind = "gradients through" if grad else "simulating with"
        alt = ("blochparty.simulate (the core), or the GPU under torch.no_grad()" if grad and engine in ("metal", "cuda")
               else "blochparty.simulate (the core)")
        raise NotImplementedError(f"{engine} engine: {kind} {what} is not implemented; use {alt} (docs/ENGINES.md)")
