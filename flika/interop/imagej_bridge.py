"""Bridge between flika and ImageJ/Fiji via pyimagej."""
import numpy as np
from .. import global_vars as g
from ..logger import logger

__all__ = ['get_imagej', 'to_imagej', 'from_imagej', 'run_imagej_command', 'is_imagej_available']

_ij_instance = None


def is_imagej_available():
    try:
        import imagej
        return True
    except ImportError:
        return False


def get_imagej(fiji_path=None):
    """Get or create an ImageJ2 gateway."""
    global _ij_instance
    if _ij_instance is not None:
        return _ij_instance
    import imagej
    mode = 'headless' if getattr(g, 'headless', False) else 'interactive'
    _ij_instance = imagej.init(fiji_path or 'sc.fiji:fiji', mode=mode)
    logger.info(f'Initialized ImageJ2 in {mode} mode')
    return _ij_instance


def to_imagej(window=None):
    """Send flika Window to ImageJ."""
    ij = get_imagej()
    w = window or g.win
    if w is None:
        g.alert('No window selected')
        return
    dataset = ij.py.to_dataset(w.image)
    dataset.setName(w.name)
    if not getattr(g, 'headless', False):
        ij.ui().show(dataset)
    logger.info(f'Sent "{w.name}" to ImageJ')
    return dataset


def from_imagej(dataset=None):
    """Import active ImageJ image into flika."""
    from ..window import Window
    ij = get_imagej()
    if dataset is None:
        dataset = ij.py.active_dataset()
    if dataset is None:
        g.alert('No active dataset in ImageJ')
        return None
    data = np.asarray(ij.py.from_java(dataset))
    name = str(dataset.getName()) if hasattr(dataset, 'getName') else 'ImageJ Import'
    w = Window(data, name=name)
    logger.info(f'Imported "{name}" from ImageJ')
    return w


def run_imagej_command(command, args=None, window=None):
    """Run an ImageJ command/plugin on a flika Window.

    Parameters:
        command (str): ImageJ command name
        args (dict): Arguments for the command
        window: flika Window (defaults to g.win)
    Returns:
        flika Window with result
    """
    from ..window import Window
    ij = get_imagej()
    w = window or g.win
    if w is None:
        g.alert('No window selected')
        return None
    dataset = ij.py.to_dataset(w.image)
    result = ij.py.run_plugin(command, args or {}, imp=dataset)
    result_data = np.asarray(ij.py.from_java(result.getOutput('output')))
    new_w = Window(result_data, name=f'{w.name} - {command}')
    logger.info(f'Ran ImageJ command "{command}"')
    return new_w
