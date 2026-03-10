"""Tests for the file-format registry (flika.io.registry)."""
import os
import tempfile

import numpy as np
import pytest

from flika.io.registry import FormatRegistry, FormatHandler, registry, TiffHandler, NumpyHandler, OMEZarrHandler, BioFormatsHandler


class TestFormatRegistry:
    def test_supported_extensions(self):
        exts = registry.supported_extensions
        assert '.tif' in exts
        assert '.npy' in exts
        assert '.png' in exts

    def test_get_handler_tif(self):
        h = registry.get_handler('image.tif')
        assert isinstance(h, TiffHandler)

    def test_get_handler_npy(self):
        h = registry.get_handler('data.npy')
        assert isinstance(h, NumpyHandler)

    def test_get_handler_unknown(self):
        h = registry.get_handler('file.xyz_unknown')
        assert h is None

    def test_read_raises_for_unknown(self):
        with pytest.raises(ValueError, match="No handler"):
            registry.read('file.xyz_unknown')

    def test_roundtrip_npy(self):
        data = np.random.rand(10, 10).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            registry.write(path, data)
            loaded, meta = registry.read(path)
            np.testing.assert_array_almost_equal(data, loaded)
        finally:
            os.unlink(path)

    def test_custom_handler_registration(self):
        r = FormatRegistry()

        class DummyHandler(FormatHandler):
            extensions = ['.dummy']
            def read(self, path):
                return np.zeros((2, 2)), {}

        r.add(DummyHandler())
        h = r.get_handler('test.dummy')
        assert h is not None
        arr, _ = r.read('test.dummy')
        assert arr.shape == (2, 2)

    def test_register_decorator(self):
        r = FormatRegistry()

        @r.register('.custom')
        class CustomHandler(FormatHandler):
            extensions = ['.custom']
            def read(self, path):
                return np.ones((3, 3)), {'source': 'custom'}

        h = r.get_handler('file.custom')
        assert h is not None


class TestOMEZarrHandler:
    def test_extensions(self):
        h = OMEZarrHandler()
        assert '.ome.zarr' in h.extensions

    def test_compound_extension_lookup(self):
        h = registry.get_handler('test.ome.zarr')
        assert isinstance(h, OMEZarrHandler)

    def test_can_read_suffix(self):
        h = OMEZarrHandler()
        assert h.can_read('/tmp/test.ome.zarr')
        assert not h.can_read('/tmp/test.zarr')

    def test_roundtrip(self):
        ome_zarr = pytest.importorskip('ome_zarr')
        zarr = pytest.importorskip('zarr')
        data = np.random.rand(5, 10, 10).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.ome.zarr')
            h = OMEZarrHandler()
            h.write(path, data)
            loaded, meta = h.read(path)
            assert loaded.shape == data.shape
            np.testing.assert_array_almost_equal(data, loaded)


class TestBioFormatsHandler:
    def test_extensions(self):
        h = BioFormatsHandler()
        assert '.czi' in h.extensions
        assert '.lif' in h.extensions

    def test_write_raises(self):
        h = BioFormatsHandler()
        with pytest.raises(NotImplementedError):
            h.write('test.czi', np.zeros((5, 5)))

    def test_handler_lookup(self):
        h = registry.get_handler('test.czi')
        assert isinstance(h, BioFormatsHandler)


class TestLazyArray:
    def test_shape_without_materialize(self):
        da = pytest.importorskip('dask.array')
        from ..io.lazy import LazyArray
        arr = da.ones((10, 20, 30), dtype=np.float32)
        lazy = LazyArray(arr)
        assert lazy.shape == (10, 20, 30)
        assert lazy.ndim == 3
        assert lazy.dtype == np.float32
        assert lazy._cached is None  # not yet materialized

    def test_materialize(self):
        da = pytest.importorskip('dask.array')
        from ..io.lazy import LazyArray
        arr = da.ones((5, 5), dtype=np.float64)
        lazy = LazyArray(arr)
        result = lazy.materialize()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.ones((5, 5)))

    def test_numpy_protocol(self):
        da = pytest.importorskip('dask.array')
        from ..io.lazy import LazyArray
        arr = da.ones((3, 3), dtype=np.float32)
        lazy = LazyArray(arr)
        result = np.asarray(lazy)
        assert isinstance(result, np.ndarray)

    def test_getitem(self):
        da = pytest.importorskip('dask.array')
        from ..io.lazy import LazyArray
        arr = da.arange(100, dtype=np.int32).reshape((10, 10))
        lazy = LazyArray(arr)
        row = lazy[0]
        assert isinstance(row, np.ndarray)
        np.testing.assert_array_equal(row, np.arange(10, dtype=np.int32))
