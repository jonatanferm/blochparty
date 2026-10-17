"""NVRTC and the CUDA driver API through ctypes: compile a kernel's program for the device's architecture
(cached on disk, ``BLOCHPARTY_CUDA_CACHE``), load it, launch it on PyTorch's current stream."""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import sys
from pathlib import Path

import torch

from .translate import kernel_source

# ---------------------------------------------------------------- NVRTC and the driver API


def _load_nvrtc() -> ctypes.CDLL:
    """libnvrtc: PyTorch's (already loaded, or the nvidia-cuda-nvrtc wheel next to it), else a CUDA toolkit."""
    major = (torch.version.cuda or "12").split(".")[0]
    names = [f"libnvrtc.so.{major}", "libnvrtc.so"]
    for root in sys.path:
        names += sorted(glob.glob(os.path.join(root, "nvidia", "**", f"libnvrtc.so.{major}*"), recursive=True))
    names += sorted(glob.glob(f"/usr/local/cuda*/lib64/libnvrtc.so.{major}*"))
    for name in names:
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    raise OSError("NVRTC (libnvrtc) not found: install the nvidia-cuda-nvrtc wheel or a CUDA toolkit")


class _Driver:
    """NVRTC and the CUDA driver API through ctypes; modules per device."""

    def __init__(self):
        self.nvrtc = _load_nvrtc()
        self.nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._nvrtc_check(self.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
        self.nvrtc_version = (major.value, minor.value)
        self.functions: dict[tuple[int, str], ctypes.c_void_p] = {}
        self.modules: list[ctypes.c_void_p] = []

    def _nvrtc_check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NVRTC error {code}: {self.nvrtc.nvrtcGetErrorString(code).decode()}")

    def _cuda_check(self, code: int) -> None:
        if code != 0:
            msg = ctypes.c_char_p()
            self.cuda.cuGetErrorString(code, ctypes.byref(msg))
            raise RuntimeError(f"CUDA error {code}: {(msg.value or b'?').decode()}")

    def compile(self, name: str, arch: str) -> bytes:
        """The cubin of kernel ``name`` for ``arch`` (e.g. "sm_89"), from the disk cache or NVRTC."""
        source = kernel_source(name)
        options = [f"--gpu-architecture={arch}", "-default-device", "--std=c++17"]
        key = hashlib.sha256("\0".join([source, *options, str(self.nvrtc_version)]).encode()).hexdigest()[:32]
        cache = Path(os.environ.get("BLOCHPARTY_CUDA_CACHE", Path.home() / ".cache" / "blochparty" / "cuda"))
        path = cache / f"{name}-{arch}-{key}.cubin"
        if path.exists():
            return path.read_bytes()
        prog = ctypes.c_void_p()
        self._nvrtc_check(self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), f"{name}.cu".encode(),
                                                        0, None, None))
        try:
            opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            code = self.nvrtc.nvrtcCompileProgram(prog, len(options), opts)
            size = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log)
            if code != 0:
                raise RuntimeError(f"NVRTC could not compile {name}:\n{log.value.decode(errors='replace')}")
            self._nvrtc_check(self.nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
            cubin = ctypes.create_string_buffer(size.value)
            self._nvrtc_check(self.nvrtc.nvrtcGetCUBIN(prog, cubin))
            data = cubin.raw
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        try:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:
            pass
        return data

    def function(self, name: str) -> ctypes.c_void_p:
        dev = torch.cuda.current_device()
        key = (dev, name)
        if key not in self.functions:
            torch.empty(1, device=f"cuda:{dev}")  # the primary context exists and is current
            major, minor = torch.cuda.get_device_capability(dev)
            cubin = self.compile(name, f"sm_{major}{minor}")
            module, func = ctypes.c_void_p(), ctypes.c_void_p()
            self._cuda_check(self.cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(cubin)))
            self._cuda_check(self.cuda.cuModuleGetFunction(ctypes.byref(func), module, name.encode()))
            self.modules.append(module)
            self.functions[key] = func
        return self.functions[key]

    def attribute(self, name: str, attr: int) -> int:
        """cuFuncGetAttribute (3: local memory bytes per thread, 4: registers per thread)."""
        value = ctypes.c_int()
        self._cuda_check(self.cuda.cuFuncGetAttribute(ctypes.byref(value), attr, self.function(name)))
        return value.value

    def launch(self, name: str, args, threads: int, group_size: int = 256) -> None:
        func = self.function(name)
        holders = []
        for a in args:
            if not isinstance(a, torch.Tensor) or not a.is_cuda:
                raise TypeError(f"{name}: every argument must be a CUDA tensor")
            holders.append(ctypes.c_void_p(a.data_ptr()))
        params = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        grid = max(1, -(-threads // group_size))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._cuda_check(self.cuda.cuLaunchKernel(func, grid, 1, 1, group_size, 1, 1, 0, stream, params, None))
