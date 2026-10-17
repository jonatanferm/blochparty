"""Training data (ROADMAP M7): a spec and a seed -> simulated samples with their truth, written with provenance.

    from blochparty import generate
    spec = generate.default_spec()                  # IXI heads, the realism preset (``spec`` documents the keys)
    spec["anatomy"]["lesions"] = {"count": [2, 4], "region": "periventricular", "size_mm": [5, 12]}
    for s in generate.samples(generate.merged(spec), 7, root="validation/anatomy/data", table=table,
                              protocols=generate.FALLBACK_PROTOCOLS, floors={...}, covs=None):
        generate.write(s, Path("out"))

``spec`` (the specification and its checks), ``heads`` (the preset's anatomy per seed), ``protocols`` (the paired
real protocols and their TSE), ``sample`` (simulation, noise, truth, files). ``validation/realism/generate.py`` is the
command line around it, with the real scans' noise floors.
"""

from .sample import samples, write
from .spec import DEFAULT_SPEC, FALLBACK_PROTOCOLS, default_spec, merged, span

__all__ = ["DEFAULT_SPEC", "FALLBACK_PROTOCOLS", "default_spec", "merged", "samples", "span", "write"]
