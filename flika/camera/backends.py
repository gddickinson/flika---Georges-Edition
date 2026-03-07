"""Camera backend abstraction and implementations.

Provides:
  - ``CameraBackend`` ABC with property-based interface
  - ``OpenCVBackend`` for USB cameras and webcams (cv2.VideoCapture)
  - ``MicroManagerBackend`` for scientific cameras via pymmcore-plus

All heavy imports (cv2, pymmcore_plus) are deferred.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logger import logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CameraInfo:
    """Descriptor for a discovered camera."""
    id: Any                   # Backend-specific identifier
    name: str                 # Human-readable name
    backend: str              # 'opencv', 'micromanager', etc.
    serial: str = ''
    model: str = ''


@dataclass
class CameraProperty:
    """A single camera property with its current value and constraints."""
    name: str
    value: Any
    prop_type: str            # 'int', 'float', 'enum', 'bool', 'string'
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    choices: Optional[List[str]] = None   # for enum type
    read_only: bool = False
    category: str = 'General'


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class CameraBackend(ABC):
    """Abstract base class for camera backends.

    All methods that access the camera hardware should be safe to call
    from the acquisition thread.
    """

    @staticmethod
    @abstractmethod
    def list_cameras() -> List[CameraInfo]:
        """Discover available cameras."""

    @abstractmethod
    def open(self, camera_id: Any) -> None:
        """Open a camera connection."""

    @abstractmethod
    def close(self) -> None:
        """Close the camera connection."""

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if camera is connected and ready."""

    @abstractmethod
    def get_properties(self) -> List[CameraProperty]:
        """Return all available camera properties."""

    @abstractmethod
    def get_property(self, name: str) -> Any:
        """Get a single property value."""

    @abstractmethod
    def set_property(self, name: str, value: Any) -> None:
        """Set a single property value."""

    @abstractmethod
    def snap(self) -> Tuple[np.ndarray, float]:
        """Capture a single frame.

        Returns (frame_array, timestamp_seconds).
        """

    @abstractmethod
    def start_continuous(self) -> None:
        """Start continuous acquisition mode."""

    @abstractmethod
    def stop_continuous(self) -> None:
        """Stop continuous acquisition."""

    @abstractmethod
    def grab_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Grab the latest frame from continuous acquisition.

        Returns (frame_array, timestamp_seconds) or None if no frame ready.
        """

    @abstractmethod
    def sensor_size(self) -> Tuple[int, int]:
        """Return (width, height) of the sensor in pixels."""

    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Return current (width, height) accounting for ROI/binning."""

    def backend_name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# OpenCV backend
# ---------------------------------------------------------------------------

class OpenCVBackend(CameraBackend):
    """Camera backend using OpenCV VideoCapture.

    Works with most USB cameras, webcams, and some industrial cameras
    via DirectShow (Windows) or V4L2 (Linux).
    """

    def __init__(self):
        self._cap = None
        self._camera_id = None
        self._continuous = False

    @staticmethod
    def list_cameras() -> List[CameraInfo]:
        try:
            import cv2
        except ImportError:
            return []

        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend_name = cap.getBackendName()
                cameras.append(CameraInfo(
                    id=i, name=f'Camera {i} ({w}x{h}, {backend_name})',
                    backend='opencv', model=backend_name,
                ))
                cap.release()
            else:
                cap.release()
                break
        return cameras

    def open(self, camera_id: Any) -> None:
        import cv2
        self.close()
        self._cap = cv2.VideoCapture(int(camera_id))
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Could not open camera {camera_id}")
        self._camera_id = camera_id
        logger.info("Opened OpenCV camera %s", camera_id)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._camera_id = None
            self._continuous = False

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_properties(self) -> List[CameraProperty]:
        if not self.is_open():
            return []
        import cv2

        props = []
        # Exposure
        exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        props.append(CameraProperty(
            'exposure', exp, 'float', min_val=-13, max_val=0,
            step=1, category='Exposure'))

        # Auto exposure
        auto_exp = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        props.append(CameraProperty(
            'auto_exposure', bool(auto_exp == 3.0), 'bool',
            category='Exposure'))

        # Gain
        gain = self._cap.get(cv2.CAP_PROP_GAIN)
        props.append(CameraProperty(
            'gain', gain, 'float', min_val=0, max_val=255,
            step=1, category='Gain'))

        # Brightness
        bright = self._cap.get(cv2.CAP_PROP_BRIGHTNESS)
        props.append(CameraProperty(
            'brightness', bright, 'float', min_val=0, max_val=255,
            step=1, category='Image'))

        # Contrast
        contrast = self._cap.get(cv2.CAP_PROP_CONTRAST)
        props.append(CameraProperty(
            'contrast', contrast, 'float', min_val=0, max_val=255,
            step=1, category='Image'))

        # FPS
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        props.append(CameraProperty(
            'fps', fps, 'float', min_val=1, max_val=120,
            step=1, category='Acquisition'))

        # Resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        props.append(CameraProperty(
            'width', w, 'int', min_val=1, max_val=4096,
            category='Resolution'))
        props.append(CameraProperty(
            'height', h, 'int', min_val=1, max_val=4096,
            category='Resolution'))

        return props

    def get_property(self, name: str) -> Any:
        if not self.is_open():
            return None
        import cv2
        prop_map = {
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'fps': cv2.CAP_PROP_FPS,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        }
        cv_prop = prop_map.get(name)
        if cv_prop is not None:
            return self._cap.get(cv_prop)
        return None

    def set_property(self, name: str, value: Any) -> None:
        if not self.is_open():
            return
        import cv2
        prop_map = {
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'fps': cv2.CAP_PROP_FPS,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        }
        cv_prop = prop_map.get(name)
        if cv_prop is not None:
            if name == 'auto_exposure':
                self._cap.set(cv_prop, 3.0 if value else 1.0)
            else:
                self._cap.set(cv_prop, float(value))

    def snap(self) -> Tuple[np.ndarray, float]:
        if not self.is_open():
            raise RuntimeError("Camera not open")
        ret, frame = self._cap.read()
        ts = time.time()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        # Convert BGR to RGB if color, or to grayscale
        if frame.ndim == 3:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, ts

    def start_continuous(self) -> None:
        self._continuous = True

    def stop_continuous(self) -> None:
        self._continuous = False

    def grab_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if not self.is_open():
            return None
        ret, frame = self._cap.read()
        ts = time.time()
        if not ret:
            return None
        if frame.ndim == 3:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, ts

    def sensor_size(self) -> Tuple[int, int]:
        if not self.is_open():
            return (0, 0)
        import cv2
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def frame_size(self) -> Tuple[int, int]:
        return self.sensor_size()


# ---------------------------------------------------------------------------
# Micro-Manager backend
# ---------------------------------------------------------------------------

class MicroManagerBackend(CameraBackend):
    """Camera backend using Micro-Manager via pymmcore-plus.

    Supports 300+ scientific camera models through Micro-Manager
    device adapters. Requires Micro-Manager installation.
    """

    def __init__(self):
        self._mmc = None
        self._camera_label = ''
        self._continuous = False

    @staticmethod
    def list_cameras() -> List[CameraInfo]:
        try:
            from pymmcore_plus import CMMCorePlus
        except ImportError:
            return []

        cameras = []
        try:
            mmc = CMMCorePlus.instance()
            if not mmc.getDeviceAdapterSearchPaths():
                return []
            # List loaded camera devices
            for label in mmc.getLoadedDevicesOfType(2):  # CameraDevice
                model = mmc.getDeviceLibrary(label)
                cameras.append(CameraInfo(
                    id=label, name=f'{label} ({model})',
                    backend='micromanager', model=model,
                ))
        except Exception as e:
            logger.debug("Micro-Manager camera listing failed: %s", e)
        return cameras

    def open(self, camera_id: Any) -> None:
        from pymmcore_plus import CMMCorePlus
        self._mmc = CMMCorePlus.instance()
        self._camera_label = str(camera_id)
        self._mmc.setCameraDevice(self._camera_label)
        logger.info("Opened Micro-Manager camera: %s", camera_id)

    def close(self) -> None:
        if self._mmc is not None and self._continuous:
            try:
                self._mmc.stopSequenceAcquisition()
            except Exception:
                pass
        self._continuous = False
        self._mmc = None

    def is_open(self) -> bool:
        return self._mmc is not None

    def get_properties(self) -> List[CameraProperty]:
        if not self.is_open():
            return []
        props = []
        try:
            dev = self._camera_label
            for prop_name in self._mmc.getDevicePropertyNames(dev):
                try:
                    val = self._mmc.getProperty(dev, prop_name)
                    read_only = self._mmc.isPropertyReadOnly(dev, prop_name)

                    # Determine type
                    if self._mmc.hasPropertyLimits(dev, prop_name):
                        lo = self._mmc.getPropertyLowerLimit(dev, prop_name)
                        hi = self._mmc.getPropertyUpperLimit(dev, prop_name)
                        try:
                            fval = float(val)
                            props.append(CameraProperty(
                                prop_name, fval, 'float',
                                min_val=lo, max_val=hi,
                                read_only=read_only, category='Device'))
                        except ValueError:
                            props.append(CameraProperty(
                                prop_name, val, 'string',
                                read_only=read_only, category='Device'))
                    elif self._mmc.getAllowedPropertyValues(dev, prop_name):
                        choices = list(self._mmc.getAllowedPropertyValues(
                            dev, prop_name))
                        props.append(CameraProperty(
                            prop_name, val, 'enum', choices=choices,
                            read_only=read_only, category='Device'))
                    else:
                        props.append(CameraProperty(
                            prop_name, val, 'string',
                            read_only=read_only, category='Device'))
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Error reading MM properties: %s", e)
        return props

    def get_property(self, name: str) -> Any:
        if not self.is_open():
            return None
        try:
            return self._mmc.getProperty(self._camera_label, name)
        except Exception:
            # Try core properties
            if name == 'exposure':
                return self._mmc.getExposure()
            return None

    def set_property(self, name: str, value: Any) -> None:
        if not self.is_open():
            return
        try:
            if name == 'exposure':
                self._mmc.setExposure(float(value))
            else:
                self._mmc.setProperty(self._camera_label, name, str(value))
        except Exception as e:
            logger.warning("Failed to set property %s: %s", name, e)

    def snap(self) -> Tuple[np.ndarray, float]:
        if not self.is_open():
            raise RuntimeError("Camera not open")
        self._mmc.snapImage()
        ts = time.time()
        frame = self._mmc.getImage()
        return frame, ts

    def start_continuous(self) -> None:
        if not self.is_open():
            return
        self._mmc.startContinuousSequenceAcquisition(0)
        self._continuous = True

    def stop_continuous(self) -> None:
        if not self.is_open():
            return
        try:
            self._mmc.stopSequenceAcquisition()
        except Exception:
            pass
        self._continuous = False

    def grab_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if not self.is_open():
            return None
        try:
            if self._mmc.getRemainingImageCount() > 0:
                frame = self._mmc.getLastImage()
                return frame, time.time()
        except Exception:
            pass
        return None

    def sensor_size(self) -> Tuple[int, int]:
        if not self.is_open():
            return (0, 0)
        try:
            return (self._mmc.getImageWidth(), self._mmc.getImageHeight())
        except Exception:
            return (0, 0)

    def frame_size(self) -> Tuple[int, int]:
        return self.sensor_size()


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def discover_cameras() -> List[CameraInfo]:
    """Discover cameras across all available backends."""
    cameras = []
    for backend_cls in [OpenCVBackend, MicroManagerBackend]:
        try:
            cameras.extend(backend_cls.list_cameras())
        except Exception as e:
            logger.debug("Backend %s discovery failed: %s",
                         backend_cls.__name__, e)
    return cameras


def create_backend(camera_info: CameraInfo) -> CameraBackend:
    """Create and return the appropriate backend for a camera."""
    if camera_info.backend == 'opencv':
        return OpenCVBackend()
    elif camera_info.backend == 'micromanager':
        return MicroManagerBackend()
    raise ValueError(f"Unknown backend: {camera_info.backend}")
