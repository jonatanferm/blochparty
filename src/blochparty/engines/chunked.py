"""Spin-chunked, memory-bounded backward pass.

Backprop through ``simulate`` keeps intermediates for every spin at every segment,
O(N_spins x N_segments) memory. Spins are independent and the signal is a plain sum
over spins, so the gradient decomposes over spin chunks:

1. forward once without grad, chunk by chunk (sum signals, concatenate ``m``);
2. in backward, recompute each chunk with grad enabled and pull the incoming
   dL/d(signal) and that chunk's dL/dm back through it (a vector-Jacobian product).

Sequence gradients are summed over chunks; per-spin phantom and coil gradients are
scattered into full-size buffers. Peak memory scales with ``spin_chunk``, not N, at
the cost of one extra forward pass. ``checkpoint_every`` still bounds the segment
direction inside each chunk.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ..result import SimResult
from .core import simulate
from ..phantom import Phantom
from ..sequence import Sequence


class _Static(NamedTuple):  # everything that is not a differentiable tensor
    is_rf: list[bool]
    adc: list[bool]
    spoil: list[bool]
    meta: dict
    spin_chunk: int
    checkpoint_every: int | None


def _simulate_chunk(st: _Static, seq_t, spin_t, s: slice) -> SimResult:
    """Plain ``simulate`` on spins ``s``; ``seq_t`` = (dt, rf, grad, adc_phase, b0_offset or None),
    ``spin_t`` = (pos, pd, t1, t2, db0, coils or None, b1 or None), full size or pre-sliced."""
    *seq_t, b0_offset = seq_t
    seq = Sequence(*seq_t, st.is_rf, st.adc, st.spoil, st.meta, b0_offset)
    pos, pd, t1, t2, db0, coils, b1 = (None if t is None else t[s] for t in spin_t)
    return simulate(Phantom(pos, pd, t1, t2, db0, b1), seq, coils, checkpoint_every=st.checkpoint_every)


class _ChunkedSimulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, st: _Static, dt, rf, grad, adc_phase, b0_offset, pos, pd, t1, t2, db0, coils, b1):
        ctx.set_materialize_grads(False)
        ctx.st = st
        ctx.save_for_backward(dt, rf, grad, adc_phase, b0_offset, pos, pd, t1, t2, db0, coils, b1)
        n = pd.shape[0]
        signal, ms = None, []
        for a in range(0, n, st.spin_chunk):
            res = _simulate_chunk(st, (dt, rf, grad, adc_phase, b0_offset), (pos, pd, t1, t2, db0, coils, b1),
                                  slice(a, a + st.spin_chunk))
            signal = res.signal if signal is None else signal + res.signal
            ms.append(res.m)
        return signal, torch.cat(ms)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_signal, grad_m):
        st = ctx.st
        saved = ctx.saved_tensors
        need = ctx.needs_input_grad[1:]
        n_seq = 5
        grads: list[Tensor | None] = [None] * len(saved)
        if grad_signal is None and grad_m is None:
            return None, *grads
        # Sequence tensors are shared by all chunks: one set of leaves, grads summed.
        seq_leaves = [None if t is None else t.detach().requires_grad_(r)
                      for t, r in zip(saved[:n_seq], need[:n_seq])]
        # Per-spin tensors: full-size buffers that each chunk scatters its slice into.
        for i in range(n_seq, len(saved)):
            if need[i]:
                grads[i] = torch.zeros(saved[i].shape, dtype=saved[i].dtype, device=saved[i].device)

        n = saved[6].shape[0]
        for a in range(0, n, st.spin_chunk):
            s = slice(a, a + st.spin_chunk)
            spin_leaves = [None if t is None else t[s].detach().requires_grad_(r)
                           for t, r in zip(saved[n_seq:], need[n_seq:])]
            with torch.enable_grad():
                res = _simulate_chunk(st, seq_leaves, spin_leaves, slice(None))
            outs, gouts = [], []
            for out, g in ((res.signal, grad_signal), (res.m, None if grad_m is None else grad_m[s])):
                if g is not None and out.requires_grad:
                    outs.append(out)
                    gouts.append(g)
            leaves = seq_leaves + spin_leaves
            idx = [i for i, t in enumerate(leaves) if t is not None and t.requires_grad]
            if not outs or not idx:
                continue
            vjp = torch.autograd.grad(outs, [leaves[i] for i in idx], gouts, allow_unused=True)
            for i, g in zip(idx, vjp):
                if g is None:
                    continue
                if i < n_seq:
                    grads[i] = g if grads[i] is None else grads[i] + g
                else:
                    grads[i][s] = g
            del res, outs, vjp  # free this chunk's graph before building the next
        return None, *grads


def simulate_chunked(
    phantom: Phantom,
    seq: Sequence,
    coils: Tensor | None = None,
    *,
    spin_chunk: int = 4096,
    checkpoint_every: int | None = None,
    m_init: Tensor | None = None,
    scanner=None,
    maps=None,
) -> SimResult:
    """Drop-in replacement for ``simulate`` whose backward pass needs memory for
    ``spin_chunk`` spins instead of all of them.

    The forward runs without a graph; backward recomputes the simulation one spin
    chunk at a time (about one extra forward pass in total). Gradients reach every
    tensor ``simulate`` would differentiate: the sequence (``dt``, ``rf``, ``grad``,
    ``adc_phase``, ``b0_offset``), the phantom (``pos``, ``pd``, ``t1``, ``t2``, ``db0``, ``b1``) and ``coils``,
    and through them whatever produced those tensors. Higher-order gradients are
    not supported. ``checkpoint_every`` is passed on to ``simulate`` inside each chunk.
    ``m_init``, ``scanner`` and ``maps`` are not supported here and raise (``docs/ENGINES.md``).

    Example::

        from blochparty.chunked import simulate_chunked

        maps = bp.shepp_logan_maps(64)
        t1 = maps["t1"].requires_grad_()
        phantom = bp.phantom_from_maps(maps["pd"], t1, maps["t2"], fov=0.22)
        flip = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
        seq = bp.gre_2d(64, flip_deg=flip, n_dummy=200)
        kspace = simulate_chunked(phantom, seq, spin_chunk=512).signal
        kspace.abs().sum().backward()  # fills flip.grad and t1.grad
    """
    if getattr(seq, "bound_only", None) is not None and any(seq.bound_only):
        raise NotImplementedError("simulate_chunked: bound-pool-only RF segments (Sequence.bound_only); use simulate")
    if getattr(phantom, "motion", None) is not None:
        raise NotImplementedError("motion (Phantom.motion) is in blochparty.simulate and the Metal kernel")
    if getattr(phantom, "diffusion", None) is not None:
        raise NotImplementedError("diffusion (Phantom.diffusion) is in blochparty.simulate and the Metal kernel")
    if phantom.bound is not None or phantom.pools:
        raise NotImplementedError("exchange (Phantom.bound / pools) is not implemented here yet; use blochparty.simulate")
    for name, value in (("m_init", m_init), ("scanner", scanner), ("maps", maps)):
        if value is not None:
            raise NotImplementedError(f"{name} is not implemented in simulate_chunked; use blochparty.simulate")

    if spin_chunk < 1:
        raise ValueError(f"spin_chunk must be positive, got {spin_chunk}")
    st = _Static(list(seq.is_rf), list(seq.adc), list(seq.spoil), seq.meta, spin_chunk, checkpoint_every)
    signal, m = _ChunkedSimulate.apply(st, seq.dt, seq.rf, seq.grad, seq.adc_phase, seq.b0_offset, phantom.pos,
                                       phantom.pd, phantom.t1, phantom.t2, phantom.db0, coils, phantom.b1)
    return SimResult(signal, m)
