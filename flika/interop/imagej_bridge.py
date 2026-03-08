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


def get_imagej(fiji_path=None, mode=None):
    """Get or create an ImageJ2 gateway.

    On macOS, PyImageJ cannot use plain ``interactive`` mode when a Qt
    event loop already owns the main thread (which is always the case
    inside flika).  We therefore fall back through several modes:

    1. ``headless`` -- if ``g.headless`` is set.
    2. ``interactive:force`` -- on macOS when Qt is running (bypasses
       the CoreFoundation main-thread check).
    3. ``interactive`` -- on other platforms.

    Parameters
    ----------
    fiji_path : str, optional
        Path to a local Fiji installation, or a Maven coordinate
        (default ``'sc.fiji:fiji'``).
    mode : str, optional
        Explicit PyImageJ init mode.  When *None* the mode is chosen
        automatically as described above.
    """
    global _ij_instance
    if _ij_instance is not None:
        return _ij_instance

    import imagej
    import sys

    if mode is None:
        if getattr(g, 'headless', False):
            mode = 'headless'
        elif sys.platform == 'darwin':
            # macOS: Qt already owns the main thread, so plain
            # 'interactive' raises an EnvironmentError.  Use the
            # force flag so PyImageJ skips the CoreFoundation check.
            mode = 'interactive:force'
        else:
            mode = 'interactive'

    endpoint = fiji_path or 'sc.fiji:fiji'

    try:
        _ij_instance = imagej.init(endpoint, mode=mode)
    except (EnvironmentError, OSError) as exc:
        # If interactive:force also fails, fall back to headless so the
        # user can still transfer data programmatically.
        if 'headless' not in mode:
            logger.warning('ImageJ interactive init failed (%s); '
                           'falling back to headless mode.', exc)
            mode = 'headless'
            _ij_instance = imagej.init(endpoint, mode=mode)
        else:
            raise

    logger.info('Initialized ImageJ2 in %s mode', mode)
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
