"""Repo-wide pytest configuration.

Two environment accommodations for this CPU-environment Windows machine, both
set before numpy/torch import:

1. ``MKL_THREADING_LAYER=SEQUENTIAL`` (+ ``KMP_DUPLICATE_LIB_OK`` fallback):
   torch and numpy/MKL each ship their own ``libiomp5md.dll``. When a test
   calls a numpy LAPACK solver while torch is resident, the duplicate OpenMP
   runtime aborts the process (OMP Error #15). Forcing MKL to its sequential
   layer keeps it off OpenMP entirely; this is deterministic and does not
   change numerical results.
2. The plan's gate runs ``python -m pytest tests -q -W error``. The
   command-line ``-W error`` is applied after ``pytest.ini`` and therefore
   overrides the ini ``ignore`` entries, so the scoped triton CUDA-discovery
   ignores are installed inside each test's active warnings context instead.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

import pytest

_TRITON_CUDA_MESSAGES = (
    "Failed to find CUDA",
    "Failed to find cuobjdump",
    "Failed to find nvdisasm",
)


@pytest.fixture(autouse=True)
def _ignore_triton_cuda_discovery_warnings():
    """Silence only triton's CPU-only-build CUDA discovery messages per test."""
    for message in _TRITON_CUDA_MESSAGES:
        warnings.filterwarnings("ignore", message=message, category=UserWarning)
    yield
