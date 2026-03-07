"""Lazy array wrapper using dask for large file support."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

LAZY_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB


def file_size_bytes(path: str) -> int:
    """Return the size of *path* in bytes."""
    return os.path.getsize(path)


def is_lazy(arr) -> bool:
    """Return True if *arr* is a LazyArray."""
    return isinstance(arr, LazyArray)


class LazyArray:
    """Numpy-compatible wrapper around a dask array.

    Provides .ndim, .shape, .dtype, .size, .nbytes properties and
    defers computation until explicitly materialized or when numpy
    needs the data via ``__array__()``.
    """

    def __init__(self, dask_array):
        self._dask = dask_array
        self._cached: Optional[np.ndarray] = None

    @property
    def ndim(self) -> int:
        return self._dask.ndim

    @property
    def shape(self):
        return self._dask.shape

    @property
    def dtype(self):
        return self._dask.dtype

    @property
    def size(self) -> int:
        return int(np.prod(self._dask.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def materialize(self) -> np.ndarray:
        """Compute and cache the full ndarray."""
        if self._cached is None:
            self._cached = np.asarray(self._dask.compute())
        return self._cached

    def __array__(self, dtype=None):
        arr = self.materialize()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __getitem__(self, key):
        if self._cached is not None:
            return self._cached[key]
        result = self._dask[key]
        if hasattr(result, 'compute'):
            return np.asarray(result.compute())
        return result

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        status = 'materialized' if self._cached is not None else 'lazy'
        return f"LazyArray(shape={self.shape}, dtype={self.dtype}, {status})"
