# -*- coding: utf-8 -*-
from .logger import logger
logger.debug("Started 'reading flika.py'")
import sys, os
import platform
import argparse
import warnings
logger.debug("Started 'reading flika.py, importing numpy'")
import numpy as np
logger.debug("Completed 'reading flika.py, importing numpy'")
from .version import __version__
from .app.application import FlikaApplication


# for development purposes, add this if flika is not in your site-packages
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))



try:
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
except AttributeError:
    pass  # removed in NumPy 2.0+

def parse_arguments(argv):
    '''Parses command line arguments for valid flika args

    Arguments:
        argv: Arguments passed to program

    Returns:
        A tuple of (namespace, positional_args)
    '''
    parser = argparse.ArgumentParser(
        prog='flika',
        description='An interactive image processing program for biologists.',
    )
    parser.add_argument('files', nargs='*', help='Data files to load')
    parser.add_argument('-x', '--execute', action='store_true', dest='script',
                        help='Open file in script editor and run')
    parser.add_argument('-t', '--test', action='store_true', dest='test',
                        help='Run test suite')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Increase the verbosity level')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (headless mode)')
    parser.add_argument('--script-file', type=str, dest='script_file',
                        help='Execute a Python script in headless mode and exit')
    parser.add_argument('--batch', type=str, dest='batch_dir',
                        help='Process all files in directory using --script-file')
    parser.add_argument('--version', action='version', version=str(__version__))

    args = parser.parse_args(argv)
    if args.script and len(args.files) != 1:
        parser.error("Must provide exactly one script with -x/--execute")

    return args, args.files

def ipython_qt_event_loop_setup():
    try:
        __IPYTHON__
    except NameError:
        return #  If __IPYTHON__ is not defined, we are not in ipython
    else:
        logger.info("Starting flika inside IPython")
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic("gui qt")

def load_files(files):
    from .process.file_ import open_file
    for f in files:
        open_file(f)

def start_flika(files=None, headless=False):
    """Run a flika session, beginning the event loop.

    Parameters:
        files (list): An optional list of data files to load.
        headless (bool): If True, run without GUI (for scripting/batch).

    Returns:
        A flika application object with optional files loaded

    """
    if files is None:
        files = []
    logger.debug("Started 'flika.start_flika()'")
    logger.info('Starting flika' + (' (headless)' if headless else ''))

    from . import global_vars as g
    g.headless = headless
    if headless:
        g.settings['show_windows'] = False

    fa = FlikaApplication(headless=headless)
    load_files(files)
    if not headless:
        fa.start()
        ipython_qt_event_loop_setup()
    logger.debug("Completed 'flika.start_flika()'")
    return fa


def start_flika_headless():
    """Convenience function: start flika in headless mode for scripting.

    Returns:
        The global_vars module (g) for easy access to windows and settings.

    Example::

        from flika.flika import start_flika_headless
        g = start_flika_headless()
        from flika.process.file_ import open_file
        from flika.process.filters import gaussian_blur
        w = open_file('input.tif')
        gaussian_blur(sigma=2.0)
        g.win.save('output.tif')
    """
    from . import global_vars as g
    start_flika(headless=True)
    return g


def exec_():
    args, files = parse_arguments(sys.argv[1:])

    if args.headless or args.script_file:
        fa = start_flika(files=files, headless=True)

        if args.script_file:
            logger.info(f'Executing script: {args.script_file}')
            with open(args.script_file, 'r') as f:
                exec(f.read(), {'__builtins__': __builtins__, '__file__': args.script_file})

            if args.batch_dir:
                from .batch import BatchProcessor
                from . import global_vars as g
                input_files = BatchProcessor.collect_files(args.batch_dir)
                logger.info(f'Batch processing {len(input_files)} files from {args.batch_dir}')

        return 0

    fa = start_flika(files)
    return fa.app.exec_()

def post_install():
    if platform.system() == 'Windows':
        logger.info("Creating start menu shortcut...")
        import winshell
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'favicon.ico')
        flika_exe = os.path.join(sys.exec_prefix, 'Scripts', 'flika.exe')
        link_path = os.path.join(winshell.programs(), "flika.lnk")
        with winshell.shortcut(link_path) as link:
            link.path = flika_exe
            link.description = "flika"
            link.icon_location = (icon_path, 0)
        link_path = os.path.join(winshell.desktop(), "flika.lnk")
        with winshell.shortcut(link_path) as link:
            link.path = flika_exe
            link.description = "flika"
            link.icon_location = (icon_path, 0)

if __name__ == '__main__':
    start_flika(sys.argv[1:])

logger.debug("Completed 'reading flika.py'")
"""
def exec_(args=sys.argv):
    opt, args = parse_arguments(args[1:])

    if opt.verbose:
        logger.setLevel("INFO")

    logger.info("Input arguments: %s", sys.argv)

    start_flika(files=args)


def run(args=sys.argv):
    ''' open flika without running exec_. For debugging purposes
    '''
    opt, args = parse_arguments(args[1:])

    if opt.verbose:
        logger.setLevel("INFO")

    fa = FlikaApplication()
    fa.show()

    load_files(files=args)

    if 'PYCHARM_HOSTED' not in os.environ and 'SPYDER_SHELL_ID' not in os.environ:
        return fa.app.exec_()
"""
