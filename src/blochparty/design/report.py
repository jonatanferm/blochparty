"""``design_report``: a sequence's hardware checks, metrics and estimation precision in one table."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import Tensor

from ..phantom import Phantom
from ..sequence import Sequence
from .hardware import Hardware, check_hardware, violations
from .metrics import (
    _default_simulate,
    contrast,
    image_noise_sigma,
    magnitude_image,
    point_spread_function,
    psf_metrics,
    snr,
    tissue_mean,
)

# --- design report -------------------------------------------------------------------------


def _num(x):
    if torch.is_tensor(x):
        return float(x.detach()) if x.numel() == 1 else [float(v) for v in x.detach().flatten()]
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_num(v) for v in x]
    return str(x)


def design_report(
    seq: Sequence,
    phantom: Phantom | None = None,
    *,
    hardware: Hardware | None = None,
    coils: Tensor | None = None,
    masks: dict[str, Tensor] | None = None,
    contrast_pairs: list[tuple[str, str]] = (),
    noise_sigma: float | None = None,
    kspace_sigma: float | None = None,
    noise_cov: Tensor | None = None,
    psf: bool = True,
    psf_tissue: tuple[float, float] = (1.0, 0.1),
    simulate=None,
    max_items: int = 10,
) -> dict:
    """Machine-readable summary of ``seq``: timing, every constraint of ``hardware`` with
    its margin and specific violations, and (for Cartesian sequences) the PSF and, given a
    ``phantom`` and label ``masks`` on the image grid, per-tissue signal, contrast, CNR and
    SNR efficiency. Noise: image-domain ``noise_sigma``, or ``kspace_sigma`` (+
    ``noise_cov``) propagated by pseudo-replicas through the reconstruction; without either,
    noise-dependent metrics are omitted. Only plain JSON types are returned. ``simulate``: the core
    (float64) unless given, e.g. ``blochparty.metal.simulate_metal`` for large phantoms.
    """
    hw = hardware or Hardware()
    simulate = simulate or _default_simulate()
    with torch.no_grad():
        duration = float(seq.duration)
        report = {
            "timing": {
                "duration_s": duration,
                "n_segments": len(seq),
                "n_rf_segments": sum(seq.is_rf),
                "n_adc": seq.n_adc,
                "n_spoils": sum(seq.spoil),
                **{f"{k}_s": _num(seq.meta[k]) for k in ("tr", "te", "dwell") if k in seq.meta},
            },
            "hardware": asdict(hw),
            "constraints": {},
            "violations": [],
        }
        for c in check_hardware(seq, hw):
            v = c.values.detach().double()
            peak = float(v.max()) if v.numel() else 0.0
            items = violations(c, seq, max_items)
            entry = {
                "value": peak,
                "limit": c.limit,
                "unit": c.unit,
                "margin_pct": None if c.limit is None else 100.0 * (1.0 - peak / c.limit),
                "ok": c.limit is None or peak <= c.limit,
                "n_violating": 0 if c.limit is None else int((v > c.limit).sum()),
                "violations": items,
            }
            if v.numel():
                entry["at_segment"] = int(c.segments[int(v.argmax())])
            report["constraints"][c.name] = entry
            if items:  # one line per violated constraint: how often, and the worst place
                n_ranges = len(items) - 1 + items[-1].get("n_more", 1)
                where = (
                    ""
                    if c.where == "sequence"
                    else (
                        f" at {entry['n_violating']} {'boundaries' if c.where == 'boundaries' else 'segments'}"
                        f" ({n_ranges} ranges); worst"
                    )
                )
                worst = items[0]["message"][len(f"{c.name} exceeded ") :]
                report["violations"].append(f"{c.name} exceeded{where} {worst}" if where else items[0]["message"])
        report["ok"] = not report["violations"]

        cartesian = "kspace_shape" in seq.meta
        if cartesian and psf:
            p = point_spread_function(seq, t1=psf_tissue[0], t2=psf_tissue[1], simulate=simulate)
            report["psf"] = {k: float(v) for k, v in psf_metrics(p).items()}
            report["psf"]["tissue_t1_t2_s"] = list(psf_tissue)
        if cartesian and phantom is not None and masks:
            sig = simulate(phantom, seq, coils).signal
            img = magnitude_image(sig, seq)
            sigma = noise_sigma
            if sigma is None and kspace_sigma is not None:
                any_mask = torch.stack(list(masks.values())).any(0)
                recon = lambda s: magnitude_image(s, seq)
                sigma = float(
                    image_noise_sigma(
                        sig, recon, kspace_sigma, noise_cov, any_mask, generator=torch.Generator().manual_seed(0)
                    )
                )
            metrics = {"noise_sigma": sigma, "tissues": {}, "contrasts": {}}
            for name, m in masks.items():
                if m.any():
                    t = {"mean_signal": float(tissue_mean(img, m)), "n_pixels": int(m.sum())}
                    if sigma:
                        t["snr"] = float(snr(img, m, sigma))
                        t["snr_efficiency_per_sqrt_s"] = t["snr"] / math.sqrt(duration)
                    metrics["tissues"][name] = t
            for a, b in contrast_pairs:
                c = float(contrast(img, masks[a], masks[b]))
                entry = {"contrast": c}
                if sigma:
                    entry["cnr"] = c / sigma
                    entry["cnr_efficiency_per_sqrt_s"] = c / sigma / math.sqrt(duration)
                metrics["contrasts"][f"{a}-{b}"] = entry
            report["image"] = metrics
    return report
