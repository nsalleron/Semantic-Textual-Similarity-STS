"""Device detection utilities.

Selects the best available compute device for inference, preferring CUDA
(NVIDIA GPU), then MPS (Apple Silicon), then CPU. Import of ``torch`` is done
lazily so that lightweight parts of the app (e.g. ``/health``) do not require
the heavy ML stack to be installed.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return the best available device as a string: ``cuda``, ``mps`` or ``cpu``.

    The result is cached because the underlying hardware does not change during
    the lifetime of the process.
    """
    try:
        import torch
    except ImportError:
        # torch is not installed (e.g. running only the lightweight endpoints).
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    # ``torch.backends.mps`` only exists on recent torch builds.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"
