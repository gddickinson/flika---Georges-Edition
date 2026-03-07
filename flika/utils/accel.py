"""Unified device detection and GPU acceleration helpers.

Provides functions to detect available compute devices (CUDA, MPS, OpenCL, CuPy),
select the best device based on user settings, and decide whether to use GPU
acceleration for a given operation.

Example::

    from flika.utils.accel import detect_devices, get_torch_device

    info = detect_devices()
    print(info.status_report())

    device = get_torch_device('Auto')  # returns best available torch.device
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

_cached_info: Optional['AccelerationInfo'] = None


@dataclass
class DeviceInfo:
    """Description of a single compute device."""
    name: str           # e.g. 'CUDA:0', 'MPS', 'CPU'
    backend: str        # 'torch-cuda', 'torch-mps', 'cupy', 'opencl', 'cpu'
    memory_mb: int = 0  # total memory in MB (0 = unknown)
    usable: bool = True
    detail: str = ''    # version or extra info


@dataclass
class AccelerationInfo:
    """Aggregated acceleration capability report."""
    devices: List[DeviceInfo] = field(default_factory=list)
    torch_version: str = ''
    cupy_version: str = ''
    numba_version: str = ''
    best_device: str = 'CPU'  # recommended device name

    def status_report(self) -> str:
        """Generate a monospace status table."""
        lines = []
        lines.append(f"{'Device':<20} {'Backend':<14} {'Memory':<12} {'Status':<10} {'Detail'}")
        lines.append("-" * 80)
        for d in self.devices:
            mem = f"{d.memory_mb} MB" if d.memory_mb > 0 else "N/A"
            status = "OK" if d.usable else "Error"
            lines.append(f"{d.name:<20} {d.backend:<14} {mem:<12} {status:<10} {d.detail}")

        lines.append("")
        lines.append(f"PyTorch:  {self.torch_version or 'not installed'}")
        lines.append(f"CuPy:     {self.cupy_version or 'not installed'}")
        lines.append(f"Numba:    {self.numba_version or 'not installed'}")
        lines.append("")
        lines.append(f"Recommended device: {self.best_device}")
        return "\n".join(lines)


def detect_devices(force_refresh=False) -> AccelerationInfo:
    """Probe available compute devices.

    Results are cached for the session unless *force_refresh* is True.
    Follows the cellpose pattern of actually allocating a test tensor
    rather than just checking `is_available()`.
    """
    global _cached_info
    if _cached_info is not None and not force_refresh:
        return _cached_info

    info = AccelerationInfo()
    info.devices.append(DeviceInfo(name='CPU', backend='cpu', usable=True, detail='always available'))

    # Check numba
    try:
        import numba
        info.numba_version = numba.__version__
    except ImportError:
        pass

    # Check PyTorch
    try:
        import torch
        info.torch_version = torch.__version__

        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem_mb = torch.cuda.get_device_properties(i).total_mem // (1024 * 1024)
                # Test usability by allocating a small tensor
                usable = True
                detail = name
                try:
                    t = torch.zeros(16, device=f'cuda:{i}')
                    del t
                except Exception as e:
                    usable = False
                    detail = str(e)
                info.devices.append(DeviceInfo(
                    name=f'CUDA:{i}', backend='torch-cuda',
                    memory_mb=mem_mb, usable=usable, detail=detail))

        # MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            usable = True
            detail = 'Apple Silicon GPU'
            try:
                t = torch.zeros(16, device='mps')
                del t
            except Exception as e:
                usable = False
                detail = str(e)
            info.devices.append(DeviceInfo(
                name='MPS', backend='torch-mps',
                usable=usable, detail=detail))

    except ImportError:
        pass

    # Check CuPy
    try:
        import cupy
        info.cupy_version = cupy.__version__
        try:
            a = cupy.zeros(16)
            del a
            mem_mb = cupy.cuda.Device(0).mem_info[1] // (1024 * 1024)
            info.devices.append(DeviceInfo(
                name='CuPy:0', backend='cupy',
                memory_mb=mem_mb, usable=True, detail=f'cupy {cupy.__version__}'))
        except Exception as e:
            info.devices.append(DeviceInfo(
                name='CuPy:0', backend='cupy', usable=False, detail=str(e)))
    except ImportError:
        pass

    # Determine best device
    for d in info.devices:
        if d.usable and d.backend.startswith('torch-cuda'):
            info.best_device = d.name
            break
    else:
        for d in info.devices:
            if d.usable and d.backend == 'torch-mps':
                info.best_device = d.name
                break

    _cached_info = info
    return info


def get_torch_device(preference: str = 'Auto'):
    """Return a ``torch.device`` based on user preference.

    Parameters
    ----------
    preference : str
        'Auto', 'CPU', 'CUDA', or 'MPS'.

    Returns
    -------
    torch.device or None if torch is not installed.
    """
    try:
        import torch
    except ImportError:
        return None

    if preference == 'CPU':
        return torch.device('cpu')

    if preference == 'CUDA':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    if preference == 'MPS':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # Auto: pick best available
    info = detect_devices()
    for d in info.devices:
        if d.usable and d.backend == 'torch-cuda':
            return torch.device('cuda')
    for d in info.devices:
        if d.usable and d.backend == 'torch-mps':
            return torch.device('mps')
    return torch.device('cpu')


def should_use_gpu(arr, min_bytes: int = 10 * 1024 * 1024) -> bool:
    """Check whether GPU acceleration should be used for an array.

    Returns True if:
    - The acceleration_device setting is not 'CPU'
    - The array is larger than *min_bytes* (default 10 MB)
    - A usable GPU device is detected

    Parameters
    ----------
    arr : ndarray
        The array to check.
    min_bytes : int
        Minimum array size in bytes to justify GPU transfer overhead.
    """
    try:
        from .. import global_vars as g
        pref = g.settings.get('acceleration_device', 'Auto')
    except Exception:
        pref = 'Auto'

    if pref == 'CPU':
        return False

    if arr.nbytes < min_bytes:
        return False

    info = detect_devices()
    return any(d.usable and d.backend != 'cpu' for d in info.devices)
