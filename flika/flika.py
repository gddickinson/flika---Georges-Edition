# -*- coding: utf-8 -*-
"""
Main module for the flika application.
"""

# Standard library imports
import argparse
import os
import pathlib
import platform
import sys
import warnings
from typing import Any

# Set Jupyter to use platformdirs (fixes deprecation warning)
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

# Third-party imports
import numpy as np

import flika.utils.misc
from flika.app.application import FlikaApplication

# Local application imports
from flika.logger import logger
from flika.version import __version__

# Filter out known warnings
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)


def parse_arguments(argv: list[str]) -> tuple[Any, list[str]]:
    """Parses command line arguments for valid flika args

    Arguments:
        argv: Arguments passed to program

    Returns:
        A tuple of (namespace, positional_args)
    """
    parser = argparse.ArgumentParser(
        prog="flika",
        description="An interactive image processing program for biologists.",
    )
    parser.add_argument("files", nargs="*", help="Data files to load")
    parser.add_argument(
        "-x",
        "--execute",
        action="store_true",
        dest="script",
        help="Open file in script editor and run",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        dest="test",
        help="Run test suite",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase the verbosity level",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (headless mode)",
    )
    parser.add_argument(
        "--script-file",
        type=str,
        dest="script_file",
        help="Execute a Python script in headless mode and exit",
    )
    parser.add_argument(
        "--batch",
        type=str,
        dest="batch_dir",
        help="Process all files in directory using --script-file",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=str(__version__),
    )

    args = parser.parse_args(argv)
    if args.script and len(args.files) != 1:
        parser.error("Must provide exactly one script with -x/--execute")

    return args, args.files


def ipython_qt_event_loop_setup() -> None:
    """Set up the IPython Qt event loop if running inside IPython."""
    if flika.utils.misc.inside_ipython():
        print("Starting flika inside IPython")
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("gui", "qt")


def load_files(files: list[str]) -> None:
    from flika.process.file_ import open_file

    for f in files:
        open_file(f)


def start_flika(files: list[str] | None = None, headless: bool = False) -> FlikaApplication:
    """Run a flika session, beginning the event loop.

    Parameters:
        files: An optional list of data files to load.
        headless: If True, run without GUI (for scripting/batch).

    Returns:
        A flika application object with optional files loaded
    """
    if files is None:
        files = []
    logger.debug("Started 'flika.start_flika()'")
    logger.info("Starting flika" + (" (headless)" if headless else ""))

    import flika.global_vars as g

    g.headless = headless
    if headless:
        g.settings["show_windows"] = False

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
    import flika.global_vars as g

    start_flika(headless=True)
    return g


def exec_() -> int:
    """Execute the flika application."""
    args, files = parse_arguments(sys.argv[1:])

    if args.headless or args.script_file:
        fa = start_flika(files=files, headless=True)

        if args.script_file:
            logger.info(f"Executing script: {args.script_file}")
            with open(args.script_file, "r") as f:
                exec(f.read(), {"__builtins__": __builtins__, "__file__": args.script_file})

        return 0

    fa = start_flika(files)
    return fa.app.exec_()


def post_install() -> None:
    if platform.system() == "Windows":
        print("Creating start menu shortcut...")
        try:
            import importlib.resources as pkg_resources

            import winshell

            from flika import images

            # Use importlib.resources instead of os.path
            with pkg_resources.path(images, "favicon.ico") as icon_path:
                # Use Path for path manipulation
                flika_exe = pathlib.Path(sys.exec_prefix) / "Scripts" / "flika.exe"
                link_path = pathlib.Path(winshell.programs()) / "flika.lnk"
                with winshell.shortcut(str(link_path)) as link:
                    link.path = str(flika_exe)
                    link.description = "flika"
                    link.icon_location = (str(icon_path), 0)
                link_path = pathlib.Path(winshell.desktop()) / "flika.lnk"
                with winshell.shortcut(str(link_path)) as link:
                    link.path = str(flika_exe)
                    link.description = "flika"
                    link.icon_location = (str(icon_path), 0)
        except ImportError:
            print("winshell package not found. Shortcuts not created.")


if __name__ == "__main__":
    start_flika(sys.argv[1:])
