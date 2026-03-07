"""Bridge between flika and napari viewer."""
import numpy as np
from .. import global_vars as g
from ..logger import logger

__all__ = ['to_napari', 'from_napari', 'is_napari_available']


def is_napari_available():
    try:
        import napari
        return True
    except ImportError:
        return False


def to_napari(window=None, viewer=None):
    """Send a flika Window to napari.

    Parameters:
        window: flika Window (defaults to g.win)
        viewer: napari Viewer (creates one if None)
    Returns:
        napari.Viewer
    """
    import napari
    w = window or g.win
    if w is None:
        g.alert('No window selected')
        return None
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            viewer = napari.Viewer()

    scale = [1.0] * w.image.ndim
    # Set spatial scale from settings
    if w.image.ndim >= 2:
        px = g.settings.get('pixel_size', 1.0)
        scale[-1] = px
        scale[-2] = px

    viewer.add_image(
        w.image,
        name=w.name,
        scale=tuple(scale),
        metadata=w.metadata
    )
    logger.info(f'Sent "{w.name}" to napari')
    return viewer


def from_napari(layer, name=None):
    """Import a napari image layer into flika.

    Parameters:
        layer: napari Image layer
        name: optional name (defaults to layer.name)
    Returns:
        flika Window
    """
    from ..window import Window
    data = np.asarray(layer.data)
    name = name or layer.name
    metadata = dict(layer.metadata) if hasattr(layer, 'metadata') else {}
    w = Window(data, name=name, metadata=metadata)
    logger.info(f'Imported "{name}" from napari')
    return w
