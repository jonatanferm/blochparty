"""The CUDA programs: the shared device code (``../shaders/*.metal``) turned into CUDA C++ by a few text rules
(``_shared_source``), and one program per kernel (``kernel_source``): ``prelude.cuh`` (Metal's vector types and
math for CUDA), the shared code, ``common.cuh`` (warp reductions, the coil sum) and that kernel's template."""

from __future__ import annotations

import functools
import re

from .. import source
from ..driver import ADJOINT_COILS, MAX_POOLS

# ---------------------------------------------------------------- the CUDA source

# Metal's built-in vector types and functions that the shared device code uses, for CUDA. Metal's names are
# renamed in the shared text (float3 -> bp_float3, ...), so CUDA's own vector types are left alone.
_PRELUDE = source("cuda/prelude.cuh")

# ---- CUDA-only parts: warp reductions, the coil sums of one column (an ADC sample or a NUFFT grid column)
# into the block buffer, and its flush (as metal's flush: sums over the warps, one atomic per value).
_COMMON = source("cuda/common.cuh")

# The forward kernel, as metal's bloch_chunk (@MMA@ 0) and bloch_chunk_mma (@MMA@ 1: the zero-padded
# sensitivity layout, NUFFT readouts and reused pulses); placeholders as metal._KERNELS.
_KERNEL = source("cuda/kernels.cu.in")

# the other kernels, each as its Metal namesake: pulse maps, the in-order segment reduction, the adjoint's replay
# and reverse sweep
_PULSE = source("cuda/pulse.cu.in")
_SEG_REDUCE = source("cuda/seg_reduce.cu")
_REPLAY = source("cuda/replay.cu.in")
_ADJOINT = source("cuda/adjoint.cu.in")


@functools.lru_cache(maxsize=None)
def _shared_source() -> str:
    """The Metal shader's device functions (forward and adjoint) as CUDA C++: without its Metal-only flushes
    and kernels; address spaces dropped (NVRTC's -default-device makes every function a device function);
    vector types renamed onto ``_PRELUDE``'s; precise math onto CUDA's (correctly rounded to within 2 ulp
    without fast math, which is not used)."""
    src = source("shaders/common.metal")
    a, b = src.index("inline void flush(threadgroup"), src.index("// Many coils:")
    src = src[:a] + src[b:]
    src = src[:src.index("inline void flush_mma(")]
    adj = source("shaders/adjoint_common.metal")
    adj = adj[:adj.index("kernel void seg_reduce(")]
    text = (src + adj).replace("#include <metal_stdlib>", "").replace("using namespace metal;", "")
    text = re.sub(r"\b(thread|threadgroup|device)\s+(?=[A-Za-z_])", "", text)
    text = re.sub(r"\bconstant\s+(?=(?:float|int|uint)\b)", "const ", text)
    text = re.sub(r"\bprecise::(sin|cos|exp|log)\(", r"\1f(", text)
    # the scalar math of the shared code in float (CUDA's double versions run at 1/64 rate on consumer GPUs);
    # floor is also called on a vector (map_sample), the prelude's overload
    text = re.sub(r"\b(sin|cos|exp|log|sqrt|fabs|ceil)\(", r"\1f(", text)
    text = re.sub(r"\bfloor\(pos\)", "floorf(pos)", text)
    text = re.sub(r"\bmulhi\(", "__umulhi(", text)
    for t in ("float2", "float3", "int3", "uint2", "uint4"):
        text = re.sub(rf"\b{t}\b", f"bp_{t}", text)
    for define in ("MAX_TERMS", "NU_HEAD", "RBLOCK", "TG_FLOATS", "MAX_SG", "BM_NC"):
        assert f"#define {define}" in text, define
    return text

_VARIANTS = {"": (0, 0), "_full": (1, 0), "_gen": (2, 0), **{f"_bm{k}": (3, k) for k in range(1, MAX_POOLS + 1)}}


def _variant(suffix: str) -> tuple[int, bool, int]:
    """(V, DIFF, NP) of a kernel suffix (``_full_diff``, ``_bm2``, ...), as metal._lib()."""
    diff = suffix.endswith("_diff")
    v, np_ = _VARIANTS[suffix[: -len("_diff")] if diff else suffix]
    return v, diff, np_


@functools.lru_cache(maxsize=None)
def kernel_source(name: str) -> str:
    """The CUDA program of one kernel (metal's kernel names): the prelude, the shared device code, the
    CUDA-only helpers and that kernel."""
    m = re.fullmatch(r"(bloch_chunk_mma|bloch_chunk|bloch_replay|bloch_adjoint|pulse_maps|seg_reduce)(\w*)", name)
    if m is None:
        raise AttributeError(f"no CUDA kernel {name!r}")
    family, rest = m[1], m[2]
    body = _PRELUDE + _shared_source() + _COMMON
    if family == "seg_reduce":
        if rest:
            raise AttributeError(f"no CUDA kernel {name!r}")
        return body + _SEG_REDUCE
    if family == "bloch_adjoint":
        mc = re.fullmatch(r"(\w*)_c(\d+)", rest)
        if mc is None or int(mc[2]) not in ADJOINT_COILS or mc[1] not in ("", "_full"):
            raise AttributeError(f"no CUDA kernel {name!r}")
        v, nc = _VARIANTS[mc[1]][0], mc[2]
        return body + _ADJOINT.replace("@NAME@", name).replace("@V@", str(v)).replace("@NC@", nc)
    if family in ("bloch_replay", "pulse_maps"):
        if rest not in ("", "_full"):
            raise AttributeError(f"no CUDA kernel {name!r}")
        text = _REPLAY if family == "bloch_replay" else _PULSE
        return body + text.replace("@NAME@", name).replace("@V@", str(_VARIANTS[rest][0]))
    try:
        v, diff, np_ = _variant(rest)
    except KeyError:
        raise AttributeError(f"no CUDA kernel {name!r}") from None
    mma = family == "bloch_chunk_mma"
    return body + (_KERNEL.replace("@NAME@", name).replace("@MMA@", "1" if mma else "0").replace("@V@", str(v))
                   .replace("@DIFF@", "true" if diff else "false").replace("@NP@", str(np_)))
