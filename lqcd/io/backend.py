from typing import Literal

_BACKEND = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        set_backend("numpy")
    return _BACKEND


def set_backend(backend: Literal["numpy", "cupy"]):
    global _BACKEND
    if not isinstance(backend, str):
        backend = backend.__name__
    backend = backend.lower()
    assert backend in ["numpy", "cupy"]
    if backend == "numpy":
        import numpy
        _BACKEND = numpy
    elif backend == "cupy":
        import cupy
        _BACKEND = cupy
    elif backend == "torch":
        import torch
        torch.set_default_device("cuda")
        _BACKEND = torch
    else:
        raise ValueError('Backend must be "numpy", "cupy" or "torch"')
