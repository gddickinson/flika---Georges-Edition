"""Array interoperability protocols for flika Windows."""
import numpy as np

__all__ = ['add_array_protocol']


def add_array_protocol(window_class):
    """Add __array__, to_xarray, to_dask methods to Window class.

    Call this during initialization to monkey-patch Window.
    """

    def __array__(self, dtype=None):
        """Support np.asarray(window)."""
        if dtype is not None:
            return np.asarray(self.image, dtype=dtype)
        return np.asarray(self.image)

    def to_xarray(self, dims=None, coords=None):
        """Convert to xarray.DataArray with coordinates.

        Parameters:
            dims: dimension names (auto-detected if None)
            coords: coordinate arrays
        Returns:
            xarray.DataArray
        """
        import xarray as xr
        from .. import global_vars as g

        if dims is None:
            ndim = self.image.ndim
            if ndim == 2:
                dims = ('y', 'x')
            elif ndim == 3:
                dims = ('t', 'y', 'x')
            elif ndim == 4:
                if self.metadata.get('is_rgb', False):
                    dims = ('t', 'y', 'x', 'c')
                else:
                    dims = ('t', 'y', 'x', 'z')
            else:
                dims = tuple(f'd{i}' for i in range(ndim))

        if coords is None:
            coords = {}
            px = g.settings.get('pixel_size', 1.0)
            fi = g.settings.get('frame_interval', 1.0)
            for i, d in enumerate(dims):
                n = self.image.shape[i]
                if d == 't':
                    coords[d] = np.arange(n) * fi
                elif d in ('x', 'y'):
                    coords[d] = np.arange(n) * px
                else:
                    coords[d] = np.arange(n)

        return xr.DataArray(
            self.image,
            dims=dims,
            coords=coords,
            attrs=dict(self.metadata) if self.metadata else {},
            name=self.name
        )

    @classmethod
    def from_xarray(cls, da, name=None):
        """Create Window from xarray.DataArray."""
        name = name or (da.name if da.name else 'xarray import')
        return cls(da.values, name=name, metadata=dict(da.attrs))

    def to_dask(self, chunks='auto'):
        """Return as dask array.

        Parameters:
            chunks: chunk specification for dask
        Returns:
            dask.array.Array
        """
        import dask.array as da
        return da.from_array(self.image, chunks=chunks)

    @classmethod
    def from_dask(cls, darr, name='dask import'):
        """Create Window from dask array (computes immediately)."""
        return cls(np.asarray(darr), name=name)

    # Monkey-patch the class
    window_class.__array__ = __array__
    window_class.to_xarray = to_xarray
    window_class.from_xarray = from_xarray
    window_class.to_dask = to_dask
    window_class.from_dask = from_dask
