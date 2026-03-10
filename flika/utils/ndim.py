"""Helpers for making 2-D/3-D image filters work on higher-dimensional data."""
from __future__ import annotations

import functools

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def per_plane(func=None, *, parallel=False, expects_2d=False):
    """Decorator that applies a 2-D or 3-D filter to each plane of a higher-dim stack.

    The decorated function must accept an ``ndarray`` as its first positional
    argument and return an ``ndarray`` of the same spatial shape.

    When ``expects_2d=False`` (default, backward-compatible):
        - 2D or 3D input -> pass through directly
        - 4D+ input -> iterate over extra trailing dimensions

    When ``expects_2d=True``:
        - 2D input -> pass through directly
        - 3D input (T,Y,X) -> iterate over frames, apply func to each 2D frame
        - 4D+ input -> reshape to (T,Y,X,-1) then iterate over both T and
          trailing dims, applying func to each 2D slice

    Parameters
    ----------
    parallel : bool
        If True and settings allow, use ThreadPoolExecutor for multi-plane data.
        NumPy releases the GIL so threads provide real parallelism for
        array operations.
    expects_2d : bool
        If True, the decorated function expects 2D input.  3D stacks are
        automatically split into individual frames before calling the
        function.

    Example::

        @per_plane
        def my_filter(image_3d, sigma=1.0):
            ...
            return filtered_image_3d

        @per_plane(parallel=True)
        def my_fast_filter(image_3d, sigma=1.0):
            ...

        @per_plane(expects_2d=True)
        def my_2d_filter(image_2d, kernel):
            # image_2d is guaranteed to be 2D
            return fftconvolve(image_2d, kernel, mode='same')
    """
    # Handle both @per_plane and @per_plane(...) syntax
    if func is not None:
        # Called as @per_plane without arguments
        return _wrap(func, False, False)
    # Called as @per_plane(parallel=True) or @per_plane(expects_2d=True)
    def decorator(fn):
        return _wrap(fn, parallel, expects_2d)

    return decorator


def _get_parallel_settings(parallel):
    """Resolve parallel flag and worker count from global settings."""
    use_parallel = parallel
    n_workers = 4
    if use_parallel:
        try:
            import flika.global_vars as g

            use_parallel = g.settings.get("multiprocessing", True)
            n_workers = g.settings.get("nCores", 4)
        except Exception:
            pass
    return use_parallel, n_workers


def _run_over_slices(func, slices, args, kwargs, parallel, n_workers):
    """Apply *func* to each element of *slices*, optionally in parallel."""

    def process(s):
        return func(s, *args, **kwargs)

    if parallel and len(slices) > 1:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(slices))) as executor:
            return list(executor.map(process, slices))
    return [process(s) for s in slices]


def _wrap(func, parallel, expects_2d):
    @functools.wraps(func)
    def wrapper(image, *args, **kwargs):
        if expects_2d:
            return _dispatch_expects_2d(func, image, args, kwargs, parallel)
        else:
            return _dispatch_default(func, image, args, kwargs, parallel)

    return wrapper


def _dispatch_default(func, image, args, kwargs, parallel):
    """Original behavior: pass through for ndim <= 3, iterate for 4D+."""
    if image.ndim <= 3:
        return func(image, *args, **kwargs)
    # image.ndim > 3  ->  iterate over extra trailing dims
    original_shape = image.shape
    n_extra = int(np.prod(original_shape[3:]))
    reshaped = image.reshape(original_shape[:3] + (n_extra,))

    use_parallel, n_workers = _get_parallel_settings(parallel)
    slices = [reshaped[..., i] for i in range(n_extra)]
    results = _run_over_slices(func, slices, args, kwargs, use_parallel, n_workers)

    out = np.stack(results, axis=-1).reshape(original_shape)
    return out


def _dispatch_expects_2d(func, image, args, kwargs, parallel):
    """When expects_2d=True: split 3D into frames, 4D+ into frames x trailing."""
    if image.ndim == 2:
        return func(image, *args, **kwargs)

    if image.ndim == 3:
        # (T, Y, X) -> iterate over T
        use_parallel, n_workers = _get_parallel_settings(parallel)
        slices = [image[t] for t in range(image.shape[0])]
        results = _run_over_slices(func, slices, args, kwargs, use_parallel, n_workers)
        return np.stack(results, axis=0)

    # ndim > 3 -> reshape to (T, Y, X, -1), iterate over T and trailing
    original_shape = image.shape
    n_extra = int(np.prod(original_shape[3:]))
    reshaped = image.reshape(original_shape[:3] + (n_extra,))
    n_frames = original_shape[0]

    use_parallel, n_workers = _get_parallel_settings(parallel)
    slices = [reshaped[t, :, :, e] for t in range(n_frames) for e in range(n_extra)]
    results = _run_over_slices(func, slices, args, kwargs, use_parallel, n_workers)

    # Reconstruct: results has n_frames * n_extra items
    out = np.empty(original_shape, dtype=results[0].dtype)
    idx = 0
    for t in range(n_frames):
        for e in range(n_extra):
            out_view = out.reshape(original_shape[:3] + (n_extra,))
            out_view[t, :, :, e] = results[idx]
            idx += 1
    return out
