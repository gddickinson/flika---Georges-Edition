"""Pluggable file-format registry.

Usage::

    from flika.io import registry

    # Read any supported format
    array, metadata = registry.read('image.tif')

    # Register a custom handler
    @registry.register('.xyz')
    class XyzHandler(FormatHandler):
        ...
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class FormatHandler(ABC):
    """Base class for file-format handlers."""

    extensions: List[str] = []  # e.g. ['.tif', '.tiff']

    @abstractmethod
    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        """Read *path* and return ``(array, metadata)``."""

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Write *data* to *path*.  Optional — raise ``NotImplementedError`` if
        the handler is read-only."""
        raise NotImplementedError(f"{type(self).__name__} does not support writing")

    def can_read(self, path: str) -> bool:
        """Return True if this handler can read *path*."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.extensions


class FormatRegistry:
    """Central registry mapping file extensions to handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, FormatHandler] = {}

    def add(self, handler: FormatHandler) -> None:
        """Register *handler* for all its declared extensions."""
        for ext in handler.extensions:
            self._handlers[ext.lower()] = handler

    def register(self, *extensions: str):
        """Class decorator that instantiates the handler and registers it.

        Example::

            @registry.register('.xyz')
            class XyzHandler(FormatHandler):
                extensions = ['.xyz']
                def read(self, path): ...
        """
        def decorator(cls):
            instance = cls()
            if extensions:
                instance.extensions = list(extensions)
            self.add(instance)
            return cls
        return decorator

    def get_handler(self, path: str) -> Optional[FormatHandler]:
        """Return the handler for *path*, or ``None``.

        Checks compound extensions (longest match first), then falls back
        to single extension, then tries ``can_read()`` for directory-based formats.
        """
        path_lower = path.lower()
        # Check compound extensions (longest first)
        for ext in sorted(self._handlers.keys(), key=len, reverse=True):
            if path_lower.endswith(ext):
                return self._handlers[ext]
        # Fallback: try can_read() for directory-based formats
        if os.path.isdir(path):
            for handler in dict.fromkeys(self._handlers.values()):
                if handler.can_read(path):
                    return handler
        return None

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        """Read *path* using the appropriate handler."""
        handler = self.get_handler(path)
        if handler is None:
            raise ValueError(f"No handler registered for {os.path.splitext(path)[1]!r}")
        return handler.read(path)

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Write *data* to *path* using the appropriate handler."""
        handler = self.get_handler(path)
        if handler is None:
            raise ValueError(f"No handler registered for {os.path.splitext(path)[1]!r}")
        handler.write(path, data, metadata)

    @property
    def supported_extensions(self) -> List[str]:
        return sorted(self._handlers.keys())


# ---------------------------------------------------------------------------
# Singleton registry with built-in handlers
# ---------------------------------------------------------------------------
registry = FormatRegistry()


class TiffHandler(FormatHandler):
    extensions = ['.tif', '.tiff', '.stk', '.ome']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        import tifffile
        from flika.io.lazy import file_size_bytes, LAZY_THRESHOLD_BYTES, LazyArray
        with tifffile.TiffFile(path) as tif:
            metadata: dict = {}
            if tif.imagej_metadata:
                metadata['imagej'] = tif.imagej_metadata
            # Use lazy loading for large files if dask is available
            if file_size_bytes(path) > LAZY_THRESHOLD_BYTES:
                try:
                    import dask.array as da
                    store = tif.aszarr()
                    data = LazyArray(da.from_zarr(store))
                    return data, metadata
                except ImportError:
                    pass
            data = tif.asarray()
        return data, metadata

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        import tifffile
        tifffile.imwrite(path, data)


class NumpyHandler(FormatHandler):
    extensions = ['.npy']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        return np.load(path), {}

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        np.save(path, data)


class ImageHandler(FormatHandler):
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        from skimage.io import imread
        return imread(path), {}

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        from skimage.io import imsave
        imsave(path, data)


class HDF5Handler(FormatHandler):
    extensions = ['.h5', '.hdf5', '.hdf']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to read HDF5 files.  Install with: pip install h5py")
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            data = np.array(f[keys[0]])
            metadata = dict(f[keys[0]].attrs) if f[keys[0]].attrs else {}
        return data, metadata

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to write HDF5 files.  Install with: pip install h5py")
        with h5py.File(path, 'w') as f:
            ds = f.create_dataset('data', data=data)
            if metadata:
                for k, v in metadata.items():
                    try:
                        ds.attrs[k] = v
                    except TypeError:
                        pass


class ZarrHandler(FormatHandler):
    extensions = ['.zarr']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required to read Zarr files.  Install with: pip install zarr")
        z = zarr.open(path, mode='r')
        return np.array(z), dict(z.attrs)

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required to write Zarr files.  Install with: pip install zarr")
        z = zarr.open(path, mode='w', shape=data.shape, dtype=data.dtype)
        z[:] = data
        if metadata:
            z.attrs.update(metadata)


class ImarisHandler(FormatHandler):
    """Read/write Imaris .ims files (HDF5-based)."""
    extensions = ['.ims']

    def read(self, path: str) -> Tuple[np.ndarray, dict]:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for Imaris files.  Install with: pip install h5py")
        metadata: dict = {}
        with h5py.File(path, 'r') as f:
            # Navigate the Imaris resolution pyramid — read highest resolution
            ds = f.get('DataSet')
            if ds is None:
                raise ValueError("Not a valid Imaris file: missing DataSet group")
            res0 = ds.get('ResolutionLevel 0')
            if res0 is None:
                raise ValueError("Not a valid Imaris file: missing ResolutionLevel 0")

            # Discover time points and channels
            time_keys = sorted([k for k in res0.keys() if k.startswith('TimePoint')],
                               key=lambda k: int(k.split()[-1]))
            if not time_keys:
                raise ValueError("No TimePoint data found")

            # Read first timepoint to discover channels and shape
            tp0 = res0[time_keys[0]]
            ch_keys = sorted([k for k in tp0.keys() if k.startswith('Channel')],
                             key=lambda k: int(k.split()[-1]))
            sample = np.array(tp0[ch_keys[0]]['Data'])  # (Z, Y, X)

            nt = len(time_keys)
            nc = len(ch_keys)
            nz, ny, nx = sample.shape

            # Read DataSetInfo for metadata
            dsi = f.get('DataSetInfo')
            if dsi is not None:
                img_info = dsi.get('Image')
                if img_info is not None:
                    for attr_name in img_info.attrs:
                        val = img_info.attrs[attr_name]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        metadata[attr_name] = val

            # Assemble 5D array (T, C, Z, Y, X)
            data = np.zeros((nt, nc, nz, ny, nx), dtype=sample.dtype)
            for ti, tk in enumerate(time_keys):
                tp = res0[tk]
                for ci, ck in enumerate(ch_keys):
                    data[ti, ci] = np.array(tp[ck]['Data'])

        metadata['original_axes'] = 'TCZYX'
        # Squeeze single-channel
        if nc == 1:
            data = data[:, 0]  # (T, Z, Y, X)
            metadata['original_axes'] = 'TZYX'
        return data, metadata

    def write(self, path: str, data: np.ndarray, metadata: Optional[dict] = None) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for Imaris files.  Install with: pip install h5py")

        if metadata is None:
            metadata = {}

        # Normalise to 5D: (T, C, Z, Y, X)
        if data.ndim == 3:
            data = data[np.newaxis, np.newaxis, ...]  # (1,1,Z,Y,X)
        elif data.ndim == 4:
            data = data[:, np.newaxis, ...]  # (T,1,Z,Y,X)
        elif data.ndim == 5:
            pass
        else:
            raise ValueError(f"Cannot write {data.ndim}D array as Imaris file")

        nt, nc, nz, ny, nx = data.shape
        dx = float(metadata.get('dx', 1.0))
        dz = float(metadata.get('dz', 1.0))
        unit = metadata.get('Unit', 'um')

        def _h5str(s):
            return np.bytes_(str(s))

        with h5py.File(path, 'w') as f:
            f.attrs['ImarisDataSet'] = _h5str('ImarisDataSet')
            f.attrs['ImarisVersion'] = _h5str('5.5.0')
            f.attrs['DataSetInfoDirectoryName'] = _h5str('DataSetInfo')
            f.attrs['ThumbnailDirectoryName'] = _h5str('Thumbnail')
            f.attrs['DataSetDirectoryName'] = _h5str('DataSet')

            # DataSetInfo
            dsi = f.create_group('DataSetInfo')
            imaris_grp = dsi.create_group('Imaris')
            imaris_grp.attrs['Version'] = _h5str('8.0')

            img_grp = dsi.create_group('Image')
            img_grp.attrs['X'] = _h5str(nx)
            img_grp.attrs['Y'] = _h5str(ny)
            img_grp.attrs['Z'] = _h5str(nz)
            img_grp.attrs['NumberOfChannels'] = _h5str(nc)
            img_grp.attrs['Noc'] = _h5str(nc)
            img_grp.attrs['Unit'] = _h5str(unit)
            img_grp.attrs['ExtMin0'] = _h5str(0)
            img_grp.attrs['ExtMin1'] = _h5str(0)
            img_grp.attrs['ExtMin2'] = _h5str(0)
            img_grp.attrs['ExtMax0'] = _h5str(nx * dx)
            img_grp.attrs['ExtMax1'] = _h5str(ny * dx)
            img_grp.attrs['ExtMax2'] = _h5str(nz * dz)
            img_grp.attrs['RecordingDate'] = _h5str('2000-01-01 00:00:00.000')

            ti_grp = dsi.create_group('TimeInfo')
            ti_grp.attrs['DatasetTimePoints'] = _h5str(nt)
            ti_grp.attrs['FileTimePoints'] = _h5str(nt)

            for c in range(nc):
                ch_grp = dsi.create_group(f'Channel {c}')
                ch_grp.attrs['ColorMode'] = _h5str('BaseColor')
                ch_grp.attrs['ColorOpacity'] = _h5str(1.0)
                ch_grp.attrs['GammaCorrection'] = _h5str(1.0)
                ch_grp.attrs['ColorRange'] = _h5str(f'0 {float(np.max(data[:, c]))}')
                ch_grp.attrs['Name'] = _h5str(f'Channel {c}')

            # Thumbnail (minimal)
            thumb_grp = f.create_group('Thumbnail')
            thumb = np.zeros((128, 128, 4), dtype=np.uint8)
            thumb_grp.create_dataset('Data', data=thumb, compression='gzip')

            # DataSet — single resolution level
            ds = f.create_group('DataSet')
            res0 = ds.create_group('ResolutionLevel 0')

            for t in range(nt):
                tp = res0.create_group(f'TimePoint {t}')
                for c in range(nc):
                    ch = tp.create_group(f'Channel {c}')
                    vol = data[t, c]
                    ch.create_dataset('Data', data=vol, compression='gzip',
                                      chunks=True)
                    # Histogram
                    hist, _ = np.histogram(vol, bins=256)
                    ch.create_dataset('Histogram', data=hist.astype(np.uint64))
                    ch.attrs['ImageSizeX'] = _h5str(nx)
                    ch.attrs['ImageSizeY'] = _h5str(ny)
                    ch.attrs['ImageSizeZ'] = _h5str(nz)


class OMEZarrHandler(FormatHandler):
    """Read/write OME-Zarr (NGFF) datasets."""
    extensions = ['.ome.zarr']

    def _build_ome_axes(self, ndim, metadata):
        """Build NGFF axes metadata from array dimensions."""
        default_maps = {
            2: [{'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
            3: [{'name': 't', 'type': 'time'}, {'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
            4: [{'name': 't', 'type': 'time'}, {'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
            5: [{'name': 't', 'type': 'time'}, {'name': 'c', 'type': 'channel'}, {'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}],
        }
        return default_maps.get(ndim, default_maps[3])

    def _build_ome_transforms(self, ndim, metadata):
        """Build coordinate transforms from metadata pixel sizes."""
        scale = [1.0] * ndim
        if 'pixel_size_x' in metadata:
            scale[-1] = float(metadata['pixel_size_x'])
        if 'pixel_size_y' in metadata:
            scale[-2] = float(metadata['pixel_size_y'])
        if ndim >= 4 and 'pixel_size_z' in metadata:
            scale[-3] = float(metadata['pixel_size_z'])
        return [{'type': 'scale', 'scale': scale}]

    def can_read(self, path):
        if path.endswith('.ome.zarr'):
            return True
        if os.path.isdir(path) and os.path.exists(os.path.join(path, '.zattrs')):
            try:
                import json
                with open(os.path.join(path, '.zattrs')) as f:
                    attrs = json.load(f)
                return 'multiscales' in attrs
            except Exception:
                return False
        return False

    def read(self, path):
        try:
            import ome_zarr.io
            import ome_zarr.reader
        except ImportError:
            raise ImportError("ome-zarr is required.  Install with: pip install ome-zarr")
        loc = ome_zarr.io.parse_url(path, mode='r')
        if loc is None:
            raise ValueError(f"Cannot parse OME-Zarr at {path}")
        reader = ome_zarr.reader.Reader(loc)
        nodes = list(reader())
        if not nodes:
            raise ValueError(f"No data found in OME-Zarr at {path}")
        node = nodes[0]
        # Level 0 = highest resolution
        data = node.data[0]
        if hasattr(data, 'compute'):
            data = np.asarray(data)
        metadata = {}
        if hasattr(node, 'metadata') and node.metadata:
            metadata.update(node.metadata)
        return data, metadata

    def write(self, path, data, metadata=None):
        try:
            import ome_zarr.writer
            import zarr
        except ImportError:
            raise ImportError("ome-zarr is required.  Install with: pip install ome-zarr")
        if metadata is None:
            metadata = {}
        store = zarr.DirectoryStore(path)
        root = zarr.group(store, overwrite=True)
        axes = self._build_ome_axes(data.ndim, metadata)
        transforms = self._build_ome_transforms(data.ndim, metadata)
        ome_zarr.writer.write_image(
            image=data,
            group=root,
            axes=axes,
            coordinate_transformations=[transforms],
            storage_options=dict(chunks=True),
        )


class BioFormatsHandler(FormatHandler):
    """Read microscopy formats via aicsimageio (CZI, LIF, OIB, OIF, VSI)."""
    extensions = ['.czi', '.lif', '.oib', '.oif', '.vsi']

    def read(self, path):
        try:
            from aicsimageio import AICSImage
        except ImportError:
            raise ImportError("aicsimageio is required.  Install with: pip install 'aicsimageio[bioformats]'")
        img = AICSImage(path)
        data = img.get_image_dask_data("TCZYX").compute()
        data = np.squeeze(data)
        metadata = {}
        if img.physical_pixel_sizes.X is not None:
            metadata['pixel_size_x'] = float(img.physical_pixel_sizes.X)
        if img.physical_pixel_sizes.Y is not None:
            metadata['pixel_size_y'] = float(img.physical_pixel_sizes.Y)
        if img.physical_pixel_sizes.Z is not None:
            metadata['pixel_size_z'] = float(img.physical_pixel_sizes.Z)
        if hasattr(img, 'channel_names') and img.channel_names:
            metadata['channel_names'] = list(img.channel_names)
        return data, metadata

    def write(self, path, data, metadata=None):
        raise NotImplementedError("BioFormatsHandler is read-only")


# Register all built-in handlers
for _cls in (TiffHandler, NumpyHandler, ImageHandler, HDF5Handler, ZarrHandler, ImarisHandler, OMEZarrHandler, BioFormatsHandler):
    registry.add(_cls())
