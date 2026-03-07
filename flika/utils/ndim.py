"""Helpers for making 2-D/3-D image filters work on higher-dimensional data."""
from __future__ import annotations

import functools
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def per_plane(func=None, *, parallel=False):
    """Decorator that applies a 2-D or 3-D filter to each plane of a 4-D+ stack.

    The decorated function must accept an ``ndarray`` as its first positional
    argument and return an ``ndarray`` of the same spatial shape.

    For arrays with ``ndim <= 3`` the function is called directly.
    For ``ndim > 3`` the function is called on each 3-D sub-stack obtained by
    iterating over the extra (trailing) dimensions.

    Parameters
    ----------
    parallel : bool
        If True and settings allow, use ThreadPoolExecutor for 4-D+ data.
        NumPy releases the GIL so threads provide real parallelism for
        array operations.

    Example::

        @per_plane
        def my_filter(image_3d, sigma=1.0):
            ...
            return filtered_image_3d

        @per_plane(parallel=True)
        def my_fast_filter(image_3d, sigma=1.0):
            ...
    """
    # Handle both @per_plane and @per_plane(parallel=True) syntax
    if func is not None:
        # Called as @per_plane without arguments
        return _wrap(func, False)
    # Called as @per_plane(parallel=True)
    def decorator(fn):
        return _wrap(fn, parallel)
    return decorator


def _wrap(func, parallel):
    @functools.wraps(func)
    def wrapper(image, *args, **kwargs):
        if image.ndim <= 3:
            return func(image, *args, **kwargs)
        # image.ndim > 3  →  iterate over extra trailing dims
        original_shape = image.shape
        # Reshape to (T, X, Y, -1) collapsing all trailing dims
        n_extra = int(np.prod(original_shape[3:]))
        reshaped = image.reshape(original_shape[:3] + (n_extra,))

        def process_plane(i):
            return func(reshaped[..., i], *args, **kwargs)

        use_parallel = parallel
        if use_parallel:
            try:
                from .. import global_vars as g
                use_parallel = g.settings.get('multiprocessing', True)
                n_workers = g.settings.get('nCores', 4)
            except Exception:
                n_workers = 4

        if use_parallel and n_extra > 1:
            with ThreadPoolExecutor(max_workers=min(n_workers, n_extra)) as executor:
                results = list(executor.map(process_plane, range(n_extra)))
        else:
            results = [process_plane(i) for i in range(n_extra)]

        out = np.stack(results, axis=-1).reshape(original_shape)
        return out
    return wrapper
