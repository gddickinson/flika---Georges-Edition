"""Enhanced OME format I/O."""
import numpy as np
import flika.global_vars as g
from flika.logger import logger

__all__ = ['to_ome_tiff', 'to_ome_zarr']


def to_ome_tiff(window=None, path=None, pixel_size=None, channel_names=None):
    """Export with OME-XML metadata.

    Parameters:
        window: flika Window (defaults to g.win)
        path: output file path (prompts if None)
        pixel_size: physical pixel size in microns
        channel_names: list of channel names
    """
    import tifffile
    from flika.utils.misc import save_file_gui

    w = window or g.win
    if w is None:
        g.alert('No window selected')
        return

    if path is None:
        path = save_file_gui('Save OME-TIFF', filetypes='*.ome.tiff *.ome.tif')
        if not path:
            return

    px = pixel_size or g.settings.get('pixel_size', 1.0)

    metadata = {'axes': 'TYX' if w.image.ndim == 3 else 'YX'}
    if px:
        metadata['PhysicalSizeX'] = px
        metadata['PhysicalSizeY'] = px
        metadata['PhysicalSizeXUnit'] = 'nm'
        metadata['PhysicalSizeYUnit'] = 'nm'
    if channel_names:
        metadata['Channel'] = {'Name': channel_names}

    tifffile.imwrite(path, w.image, ome=True, metadata=metadata)
    logger.info(f'Saved OME-TIFF: {path}')
    if g.m is not None:
        g.m.statusBar().showMessage(f'Saved {path}')


def to_ome_zarr(window=None, path=None, pixel_size=None, channel_names=None):
    """Export with OME-NGFF (Zarr) format with multiscale pyramid.

    Parameters:
        window: flika Window (defaults to g.win)
        path: output directory path
        pixel_size: physical pixel size
        channel_names: list of channel names
    """
    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.writer import write_image
    except ImportError:
        g.alert('OME-Zarr export requires: pip install ome-zarr zarr')
        return

    from flika.utils.misc import save_file_gui

    w = window or g.win
    if w is None:
        g.alert('No window selected')
        return

    if path is None:
        path = save_file_gui('Save OME-Zarr', filetypes='*.zarr')
        if not path:
            return

    store = parse_url(path, mode='w').store
    root = zarr.group(store=store)

    px = pixel_size or g.settings.get('pixel_size', 1.0)

    # Build coordinate transforms
    axes = []
    transforms = []
    if w.image.ndim >= 3:
        axes.append({'name': 't', 'type': 'time'})
        fi = g.settings.get('frame_interval', 1.0)
        transforms.append({'type': 'scale', 'scale': [fi] + [px] * (w.image.ndim - 1)})

    write_image(
        image=w.image,
        group=root,
        axes=['t', 'y', 'x'] if w.image.ndim == 3 else ['y', 'x'],
        storage_options=dict(chunks=True),
    )

    logger.info(f'Saved OME-Zarr: {path}')
    if g.m is not None:
        g.m.statusBar().showMessage(f'Saved {path}')
