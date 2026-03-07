# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/file_.py'")
import pyqtgraph as pg
import pyqtgraph.exporters
import time
import os.path
import sys
import numpy as np
from qtpy import uic, QtGui, QtCore, QtWidgets
import shutil, subprocess
import datetime
import json
import re
import pathlib

from .. import global_vars as g
from ..utils.BaseProcess import BaseDialog
from ..window import Window
from ..utils.misc import open_file_gui, save_file_gui
import tifffile

__all__ = ['save_file', 'save_points', 'save_rois', 'save_movie_gui', 'open_file', 'open_file_from_gui', 'open_image_sequence_from_gui', 'open_points', 'close']

########################################################################################################################
######################                  SAVING FILES                                         ###########################
########################################################################################################################



def save_file(filename=None):
    """save_file(filename=None)
    Save the image in the currentWindow to a .tif file.

    Parameters:
        filename (str): The image or movie will be saved as  'filename'.tif.

    """
    if filename is None or filename is False:
        filetypes = 'TIFF (*.tif);;HDF5 (*.h5);;NumPy (*.npy);;All Files (*.*)'
        prompt = 'Save File'
        filename = save_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
    if os.path.dirname(filename) == '':  # if the user didn't specify a directory
        directory = os.path.normpath(os.path.dirname(g.settings['filename']))
        filename = os.path.join(directory, filename)
    g.status_msg(f'Saving {os.path.basename(filename)}')
    # Save the full 4D volume when available, otherwise the 3D image
    A = g.win.volume if g.win.volume is not None else g.win.image
    if A.dtype == bool:
        A = A.astype(np.uint8)
    metadata = g.win.metadata
    is_rgb = metadata.get('is_rgb', False)
    try:
        metadata_str = json.dumps(metadata, default=JSONhandler)
    except TypeError as e:
        msg = f"Error saving metadata.\n{e}\nContinuing to save file"
        g.alert(msg)
        metadata_str = '{}'
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.tif', '.tiff', '.stk', '.ome', ''):
        if len(A.shape) == 4 and not is_rgb:
            A = np.transpose(A, (0, 2, 1, 3))  # (T,X,Y,Z) -> (T,Y,X,Z) for FIJI compat
        elif len(A.shape) == 3 and not is_rgb:
            A = np.transpose(A, (0, 2, 1))  # This keeps the x and the y the same as in FIJI
        elif len(A.shape) == 2:
            A = np.transpose(A, (1, 0))
        tifffile.imwrite(filename, A,
                         description=metadata_str)
    else:
        # Try the format registry for non-TIFF formats
        from ..io.registry import registry as io_registry
        try:
            io_registry.write(str(filename), A, metadata)
        except (ValueError, NotImplementedError) as e:
            g.alert(f"Cannot save to '{ext}' format: {e}")
            return None
    g.status_msg(f'Successfully saved {os.path.basename(filename)}')
    return filename

def save_points(filename=None):
    """save_points(filename=None)
    Saves the points in the current window to a text file

    Parameters:
        filename (str): Address to save the points to, with .txt


    """

    if filename is None:
        filetypes = '*.txt'
        prompt = 'Save Points'
        filename = save_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
    g.status_msg(f'Saving Points in {os.path.basename(filename)}')
    p_out = []
    p_in = g.win.scatterPoints
    for t in np.arange(len(p_in)):
        for p in p_in[t]:
            p_out.append(np.array([t, p[0], p[1]]))
    p_out = np.array(p_out)
    np.savetxt(filename, p_out)
    g.status_msg(f'Successfully saved {os.path.basename(filename)}')
    return filename


def save_movie_gui():
    rateSpin = pg.SpinBox(value=50, bounds=[1, 1000], suffix='fps', int=True, step=1)
    rateDialog = BaseDialog([{'string': 'Framerate', 'object': rateSpin}], 'Save Movie', 'Set the framerate')
    rateDialog.accepted.connect(lambda: save_movie(rateSpin.value()))
    g.dialogs.append(rateDialog)
    rateDialog.show()

def save_rois( filename=None):
    g.currentWindow.save_rois(filename)


def save_movie(rate, filename=None):
    """save_movie(rate, filename)
    Saves the currentWindow video as a .mp4 movie by joining .jpg frames together

    Parameters:
        rate (int): framerate
        filename (str): Address to save the movie to, with .mp4

    Notes:
        Once you've exported all of the frames you wanted, open a command line and run the following:
        ffmpeg -r 100 -i %03d.jpg output.mp4
        -r: framerate
        -i: input files.
        %03d: The files have to be numbered 001.jpg, 002.jpg... etc.

    """


    ## Check if ffmpeg is installed
    if os.name == 'nt':  # If we are running windows
        try:
            subprocess.call(["ffmpeg"])
        except FileNotFoundError as e:
            if e.errno == os.errno.ENOENT:
                # handle file not found error.
                # I used http://ffmpeg.org/releases/ffmpeg-2.8.4.tar.bz2 originally
                g.alert("The program FFmpeg is required to export movies. \
                \n\nFor instructions on how to install, go here: http://www.wikihow.com/Install-FFmpeg-on-Windows")
                return None
            else:
                # Something else went wrong while trying to run `wget`
                raise

    filetypes = "Movies (*.mp4)"
    prompt = "Save movie to .mp4 file"
    filename = save_file_gui(prompt, filetypes=filetypes)
    if filename is None:
        return None

    win = g.win
    A = win.image
    if len(A.shape) < 3:
        g.alert('Movie not the right shape for saving.')
        return None
    try:
        exporter = pg.exporters.ImageExporter(win.imageview.view)
    except TypeError:
        exporter = pg.exporters.ImageExporter.ImageExporter(win.imageview.view)

    nFrames = len(A)
    tmpdir = os.path.join(os.path.dirname(g.settings.settings_file), 'tmp')
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.mkdir(tmpdir)
    win.top_left_label.hide()
    for i in np.arange(0, nFrames):
        win.setIndex(i)
        exporter.export(os.path.join(tmpdir, f'{i:03}.jpg'))
        QtWidgets.QApplication.processEvents()
    win.top_left_label.show()
    olddir = os.getcwd()
    os.chdir(tmpdir)
    subprocess.call(
        ['ffmpeg', '-r', '%d' % rate, '-i', '%03d.jpg', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 'output.mp4'])
    os.rename('output.mp4', filename)
    os.chdir(olddir)
    g.status_msg(f'Successfully saved movie as {os.path.basename(filename)}.')










########################################################################################################################
######################                         OPENING FILES                                 ###########################
########################################################################################################################

def open_image_sequence_from_gui():
    open_image_sequence(None, True)

def open_file_from_gui():
    open_file(None, True)


def open_image_sequence(filename=None, from_gui=False):
    """ open_image_sequencefilename(filename=None)
    Opens an image sequence (.tif, .png) into a new_window.

    Parameters:
        filename (str): Address of the first of a series of files that will be stitched together into a movie.
                            If no filename is provided, the last opened file is used.
    Returns:
        new_window

    """
    if filename is None:
        if from_gui:
            filetypes = 'Image Files (*.tif *.tiff *.png);;All Files (*.*)'
            prompt = 'Open File'
            filename = open_file_gui(prompt, filetypes=filetypes)
            if filename is None:
                return None
        else:
            filename = g.settings['filename']
            if filename is None:
                g.alert('No filename selected')
                return None
    logger.info(f"Filename: {filename}")
    g.status_msg(f'Loading {os.path.basename(filename)}')
    t = time.time()
    metadata = dict()

    filename = pathlib.Path(filename)
    assert filename.is_file()
    directory = filename.parents[0]
    ext = filename.suffix
    all_image_filenames = [p for p in directory.iterdir() if p.suffix == ext]

    all_images = []
    if ext in ['.tif', '.stk', '.tiff', '.ome']:
        for f in all_image_filenames:
            results = open_tiff(f, metadata)
            if results is None:
                return None
            else:
                A, metadata = results
                all_images.append(A)
        all_images = np.array(all_images)
    elif ext in ['.png']:
        pass

    append_recent_file(str(filename))  # make first in recent file menu
    msg = f'{filename.parts[-1]} successfully loaded ({time.time() - t} s)'
    g.status_msg(msg)
    g.settings['filename'] = str(filename)
    commands = [f"open_image_sequence('{str(filename)}')"]
    new_window = Window(all_images, filename.parts[-1], str(filename), commands, metadata)
    return new_window


def open_file(filename=None, from_gui=False):
    """ open_file(filename=None)
    Opens an image or movie file (.tif, .stk, .nd2) into a new_window.

    Parameters:
        filename (str): Address of file to open. If no filename is provided, the last opened file is used.
    Returns:
        new_window

    """
    if filename is None:
        if from_gui:
            filetypes = 'Image Files (*.tif *.stk *.tiff *.nd2 *.h5 *.hdf5 *.npy *.czi *.lif *.oib *.oif *.vsi *.ims);;All Files (*.*)'
            prompt = 'Open File'
            filename = open_file_gui(prompt, filetypes=filetypes)
            if filename is None:
                return None
        else:
            filename = g.settings['filename']
            if filename is None:
                g.alert('No filename selected')
                return None
    logger.info(f"Filename: {filename}")
    g.status_msg(f'Loading {os.path.basename(str(filename))}')
    t = time.time()
    metadata = dict()
    # Check for directory-based formats (e.g., .ome.zarr)
    if os.path.isdir(str(filename)):
        from ..io.registry import registry as io_registry
        handler = io_registry.get_handler(str(filename))
        if handler is not None:
            A, meta = handler.read(str(filename))
            metadata.update(meta)
            append_recent_file(str(filename))
            msg = f'{os.path.basename(str(filename))} successfully loaded ({time.time() - t} s)'
            g.status_msg(msg)
            g.settings['filename'] = str(filename)
            command = f"open_file('{filename}')"
            commands = [command]
            from ..app.macro_recorder import macro_recorder
            macro_recorder.record(command)
            new_window = Window(A, os.path.basename(str(filename)), str(filename), commands, metadata)
            return new_window
    ext = os.path.splitext(str(filename))[1]
    if ext in ['.tif', '.stk', '.tiff', '.ome']:
        results = open_tiff(str(filename), metadata)
        if results is None:
            return None
        else:
            A, metadata = results
    elif ext == '.nd2':
        import nd2reader
        nd2 = nd2reader.ND2Reader(str(filename))
        axes = nd2.axes
        mx = nd2.metadata['width']
        my = nd2.metadata['height']
        mt = nd2.metadata['total_images_per_channel']
        A = np.zeros((mt, mx, my))
        percent = 0
        for frame in range(mt):
            A[frame] = nd2[frame].T
            if percent < int(100 * float(frame) / mt):
                percent = int(100 * float(frame) / mt)
                g.status_msg(f'Loading file {percent}%')
                QtWidgets.QApplication.processEvents()
        metadata = nd2.metadata
    elif ext == '.py':
        from ..app.script_editor import ScriptEditor
        ScriptEditor.importScript(filename)
        return
    elif ext == '.whl':
        # first, remove trailing (1) or (2)
        newfilename = re.sub(r' \([^)]*\)', '', filename)
        try:
            os.rename(filename, newfilename)
        except FileExistsError:
            pass
        filename = newfilename
        result = subprocess.call([sys.executable, '-m', 'pip', 'install', f'{filename}'])
        if result == 0:
            g.alert(f'Successfully installed {filename}')
        else:
            g.alert(f'Install of {filename} failed')
        return
    elif ext == '.jpg' or ext == '.png':
        import skimage.io
        A = skimage.io.imread(filename)
        if len(A.shape) == 3:
            perm = get_permutation_tuple(['y', 'x', 'c'], ['x', 'y', 'c'])
            A = np.transpose(A, perm)
            metadata['is_rgb'] = True

    else:
        # Try the format registry as a fallback
        from ..io.registry import registry as io_registry
        try:
            A, meta = io_registry.read(str(filename))
            metadata.update(meta)
        except (ValueError, Exception):
            msg = f"Could not open.  Filetype for '{filename}' not recognized"
            g.alert(msg)
            if filename in g.settings['recent_files']:
                g.settings['recent_files'].remove(filename)
            return

    append_recent_file(str(filename))  # make first in recent file menu
    msg = f'{os.path.basename(str(filename))} successfully loaded ({time.time() - t} s)'
    g.status_msg(msg)
    g.settings['filename'] = str(filename)
    command = f"open_file('{filename}')"
    commands = [command]
    # Record for macro playback
    from ..app.macro_recorder import macro_recorder
    macro_recorder.record(command)
    new_window = Window(A, os.path.basename(str(filename)), filename, commands, metadata)
    return new_window

def _map_axes_to_canonical(axes, shape):
    """Map tifffile axes labels to flika's canonical axis order.

    Returns (target_axes, is_rgb).

    Canonical orders:
      2D: [width, height]
      3D grayscale movie: [time-like, width, height]
      3D RGB still: [width, height, channel-like]
      4D RGB movie: [time-like, width, height, channel-like]
      4D volume: [time-like, width, height, depth]
      5D: [time-like, width, height, depth, channel-like]
    """
    time_like = {'time', 'series', 'other'}
    channel_like = {'channel', 'sample'}
    spatial = {'width', 'height'}
    depth_like = {'depth'}

    axis_set = set(axes)
    ndim = len(axes)

    # Check for channel-like axes and whether they qualify as RGB (size <= 4)
    ch_axis = axis_set & channel_like
    has_channel = bool(ch_axis)
    ch_size = 0
    if has_channel:
        ch_name = list(ch_axis)[0]
        ch_idx = axes.index(ch_name)
        ch_size = shape[ch_idx]

    is_rgb = False

    # Override from settings
    override = g.settings.get('default_axis_order', 'Auto')
    if override != 'Auto':
        axis_map = {
            'TXYZ': (['time', 'width', 'height', 'depth'], False),
            'TXYC': (['time', 'width', 'height', 'channel'], True),
            'TZXY': (['depth', 'width', 'height', 'time'], False),
        }
        if override in axis_map and ndim == 4:
            return axis_map[override]

    if ndim == 2:
        # 2D still image
        target_axes = ['width', 'height']
    elif ndim == 3:
        if has_channel and ch_size <= 4:
            # Still image in color
            target_axes = ['width', 'height', ch_name]
            is_rgb = True
        elif axis_set & time_like:
            # Movie in grayscale
            t_name = list(axis_set & time_like)[0]
            target_axes = [t_name, 'width', 'height']
        elif axis_set & depth_like:
            # Z-stack
            target_axes = ['depth', 'width', 'height']
        else:
            # Fallback: treat first non-spatial axis as time
            non_spatial = [a for a in axes if a not in spatial]
            if non_spatial:
                target_axes = [non_spatial[0], 'width', 'height']
            else:
                target_axes = list(axes)
    elif ndim == 4:
        if has_channel and ch_size <= 4 and (axis_set & time_like):
            # Color movie (T, W, H, C)
            t_name = list(axis_set & time_like)[0]
            target_axes = [t_name, 'width', 'height', ch_name]
            is_rgb = True
        elif has_channel and ch_size <= 4 and (axis_set & depth_like):
            # Z-stack with color (Z, W, H, C)
            target_axes = ['depth', 'width', 'height', ch_name]
            is_rgb = True
        elif (axis_set & time_like) and (axis_set & depth_like):
            # 4D volume (T, W, H, Z)
            t_name = list(axis_set & time_like)[0]
            target_axes = [t_name, 'width', 'height', 'depth']
        elif (axis_set & time_like) and has_channel:
            # Color movie with many channels
            t_name = list(axis_set & time_like)[0]
            target_axes = [t_name, 'width', 'height', ch_name]
            is_rgb = ch_size <= 4
        else:
            # Fallback for 4D
            non_spatial = [a for a in axes if a not in spatial]
            target_axes = non_spatial[:1] + ['width', 'height'] + non_spatial[1:]
    elif ndim == 5:
        t_name = list(axis_set & time_like)[0] if (axis_set & time_like) else axes[0]
        ch_name_5 = list(ch_axis)[0] if ch_axis else axes[-1]
        target_axes = [t_name, 'width', 'height', 'depth', ch_name_5]
        is_rgb = has_channel and ch_size <= 4
    else:
        # ndim > 5 or unexpected — pass through
        target_axes = list(axes)

    return target_axes, is_rgb


def open_tiff(filename, metadata):
    try:
        Tiff = tifffile.TiffFile(str(filename))
    except Exception as s:
        g.alert(f"Unable to open {filename}. {s}")
        return None
    metadata = get_metadata_tiff(Tiff)
    A = Tiff.asarray()
    Tiff.close()
    _axes_labels = {
        'X': 'width', 'Y': 'height', 'Z': 'depth',
        'S': 'sample', 'T': 'time', 'C': 'channel',
        'I': 'series', 'Q': 'other',
    }
    axes = [_axes_labels.get(ax, ax) for ax in Tiff.series[0].axes]
    # print("Original Axes = {}".format(Tiff.series[0].axes)) #sample means RBGA, plane means frame, width means X, height means Y
    try:
        assert len(axes) == len(A.shape)
    except AssertionError:
        msg = 'Tiff could not be loaded because the number of axes in the array does not match the number of axes found by tifffile.py\n'
        msg += f"Shape of array: {A.shape}\n"
        msg += f"Axes found by tifffile.py: {axes}\n"
        g.alert(msg)
        return None
    target_axes, is_rgb = _map_axes_to_canonical(axes, A.shape)
    metadata['is_rgb'] = is_rgb
    metadata['flika_axes'] = ''.join(a[0].upper() for a in target_axes)

    perm = get_permutation_tuple(axes, target_axes)
    A = np.transpose(A, perm)
    if target_axes[-1] in ['channel', 'sample', 'series'] and A.shape[-1] == 2:
        B = np.zeros(A.shape[:-1])
        B = np.expand_dims(B, len(B.shape))
        A = np.append(A, B, len(A.shape) - 1)  # add a column of zeros to the last dimension.
        # if A.ndim == 4 and axes[3] == 'sample' and A.shape[3] == 1:
        #    A = np.squeeze(A)  # this gets rid of the meaningless 4th dimention in .stk files
    return [A, metadata]


def open_points(filename=None):
    """open_points(filename=None)
    Opens a specified text file and displays the points from that file into the currentWindow

    Parameters:
        filename (str): Address of file to open. If no filename is provided, the last opened file is used.

    Note:
        Any existing points on a currentWindow will persist when another points file is opened and displayed

    """
    if g.win is None:
        g.alert('Points cannot be loaded if no window is selected. Open a file and click on a window.')
        return None
    if filename is None:
        filetypes = '*.txt'
        prompt = 'Load Points'
        filename = open_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
    msg = f'Loading points from {os.path.basename(filename)}'
    g.status_msg(msg)
    try:
        pts = np.loadtxt(filename)
    except UnicodeDecodeError:
        g.alert('This points file contains text that cannot be read. No points loaded.')
        return None
    if len(pts) == 0:
        g.alert('This points file is empty. No points loaded.')
        return None
    nCols = pts.shape[1]
    pointSize = g.settings['point_size']
    pointColor = QtGui.QColor(g.settings['point_color'])
    if nCols == 3:
        for pt in pts:
            t = int(pt[0])
            if g.win.mt == 1:
                t = 0
            g.win.scatterPoints[t].append([pt[1],pt[2], pointColor, pointSize])
        if g.settings['show_all_points']:
            pts = []
            for t in np.arange(g.win.mt):
                pts.extend(g.win.scatterPoints[t])
            pointSizes = [pt[3] for pt in pts]
            brushes = [pg.mkBrush(*pt[2].getRgb()) for pt in pts]
            g.win.scatterPlot.setData(pos=pts, size=pointSizes, brush=brushes)
        else:
            t = g.win.currentIndex
            g.win.scatterPlot.setData(pos=g.win.scatterPoints[t])
            g.win.updateindex()
    elif nCols == 2:
        t = 0
        for pt in pts:
            g.win.scatterPoints[t].append([pt[0], pt[1], pointColor, pointSize])
        t = g.win.currentIndex
        g.win.scatterPlot.setData(pos=g.win.scatterPoints[t])

    g.status_msg(f'Successfully loaded {os.path.basename(filename)}')


def open_spt_results(filename=None):
    """open_spt_results(filename=None)

    Load SPT localization / tracking results from a file and attach them
    to the current window.  Supports:

    - **Flika CSV** — ``frame, x, y, intensity [, track_id]``
    - **Flika plugin CSV** — ``track_number, frame, x, y [, intensities]``
    - **ThunderSTORM CSV** — ``id, frame, x [nm], y [nm], intensity [photon], …``
    - **Flika plugin JSON** — ``{tracks: [...], txy_pts: [...]}``
    - **Headerless TXT** — space/tab-delimited ``[frame, x, y]`` (detect_puffs)
    - **Generic CSV** — auto-detect column names

    The loaded data is stored in ``window.metadata['spt']`` as a
    :class:`~flika.spt.particle_data.ParticleData` and legacy keys are
    populated.  If the Results Table is open it is refreshed automatically.

    Parameters
    ----------
    filename : str, optional
        Path to the file.  If *None*, a file dialog is shown.
    """
    if g.win is None:
        g.alert('SPT results cannot be loaded if no window is selected. '
                'Open an image and select a window first.')
        return None

    if filename is None:
        filetypes = ('SPT Results (*.csv *.txt *.json);;'
                     'CSV Files (*.csv);;'
                     'JSON Files (*.json);;'
                     'Text Files (*.txt);;'
                     'All Files (*)')
        filename = open_file_gui('Open SPT Results', filetypes=filetypes)
        if filename is None:
            return None

    g.status_msg(f'Loading SPT results from {os.path.basename(filename)}...')

    try:
        pdata = _load_spt_file(filename)
    except Exception as exc:
        logger.exception("Failed to load SPT results from %s", filename)
        g.alert(f'Failed to load SPT results:\n{exc}')
        return None

    if pdata is None or pdata.n_localizations == 0:
        g.alert('No localization data found in the selected file.')
        return None

    # Attach to current window
    w = g.win
    if 'spt' not in w.metadata:
        w.metadata['spt'] = {}
    spt = w.metadata['spt']

    spt['particle_data'] = pdata
    legacy = pdata.to_spt_dict()
    spt['localizations'] = legacy['localizations']
    spt['tracks'] = legacy['tracks']
    spt['tracks_dict'] = legacy['tracks_dict']
    spt['detection_method'] = 'loaded'
    spt['detection_params'] = {'source_file': filename}
    if pdata.n_tracks > 0:
        spt['linking_method'] = 'loaded'
        spt['linking_params'] = {'source_file': filename}

    n_locs = pdata.n_localizations
    n_tracks = pdata.n_tracks
    msg = f'Loaded {n_locs} localizations'
    if n_tracks > 0:
        msg += f' in {n_tracks} tracks'
    msg += f' from {os.path.basename(filename)}'
    g.status_msg(msg)
    logger.info(msg)

    # Display particles on window
    locs = legacy['localizations']
    if len(locs) > 0 and hasattr(w, 'scatterPoints'):
        from qtpy.QtGui import QColor
        color = QColor(0, 255, 0, 180)
        for frame_pts in w.scatterPoints:
            frame_pts.clear()
        for det in locs:
            frame = int(det[0])
            if 0 <= frame < len(w.scatterPoints):
                # Loaded data: [:,1]=dim1 (pyqtgraph x), [:,2]=dim2 (pyqtgraph y)
                w.scatterPoints[frame].append([det[1], det[2], color, 5])
        if hasattr(w, 'updateindex'):
            w.updateindex()

    # Refresh results table if open
    try:
        from ..viewers.results_table import ResultsTableWidget
        table = ResultsTableWidget._instance
        if table is not None and table.isVisible():
            table.set_particle_data(pdata)
    except Exception:
        pass

    return pdata


def _load_spt_file(path):
    """Auto-detect format and load SPT results into a ParticleData.

    Returns a :class:`~flika.spt.particle_data.ParticleData` or raises
    on failure.
    """
    import pandas as pd
    from ..spt.particle_data import ParticleData

    ext = os.path.splitext(path)[1].lower()

    # --- JSON ---
    if ext == '.json':
        return _load_spt_json(path)

    # --- TXT (headerless numeric) ---
    if ext == '.txt':
        return _load_spt_txt(path)

    # --- CSV (with header) ---
    if ext == '.csv':
        return _load_spt_csv(path)

    # Fallback: try CSV, then TXT
    try:
        return _load_spt_csv(path)
    except Exception:
        pass
    try:
        return _load_spt_txt(path)
    except Exception:
        pass
    raise ValueError(f"Cannot determine format of {path}")


def _load_spt_csv(path):
    """Load CSV with auto-detection of ThunderSTORM, flika, or plugin format."""
    import pandas as pd
    from ..spt.particle_data import ParticleData

    df = pd.read_csv(path)
    if df.empty:
        return ParticleData()

    cols_lower = {c: c.strip().lower() for c in df.columns}

    # --- ThunderSTORM format ---
    ts_cols = [c for c in df.columns if '[nm]' in c]
    if ts_cols:
        return _load_thunderstorm_csv(df)

    # Normalise column names
    rename = {}
    for orig, low in cols_lower.items():
        if low in ('track_number', 'trackid', 'track_id', 'particle'):
            rename[orig] = 'track_id'
        elif low == 'x':
            rename[orig] = 'x'
        elif low == 'y':
            rename[orig] = 'y'
        elif low == 'frame':
            rename[orig] = 'frame'
        elif low in ('intensity', 'intensities'):
            rename[orig] = 'intensity'
        elif low == 'id':
            rename[orig] = 'id'
        elif low == 'sigma':
            rename[orig] = 'sigma_x'
        elif low in ('sigma_x', 'sigma1'):
            rename[orig] = 'sigma_x'
        elif low in ('sigma_y', 'sigma2'):
            rename[orig] = 'sigma_y'
        elif low in ('background', 'offset', 'bkgstd'):
            rename[orig] = 'background'
        elif low in ('uncertainty', 'precision'):
            rename[orig] = 'uncertainty'

    df = df.rename(columns=rename)

    # Ensure required columns
    if 'frame' not in df.columns:
        # If no frame column but we have an index-like column, use it
        if 'id' in df.columns and 'x' in df.columns:
            df['frame'] = 0
        else:
            raise ValueError("CSV has no 'frame' column")

    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV has no 'x' and 'y' columns")

    df['frame'] = df['frame'].fillna(0).astype('int64')

    if 'intensity' not in df.columns:
        df['intensity'] = 0.0

    if 'id' not in df.columns:
        df.insert(0, 'id', np.arange(len(df), dtype=np.int64))
    else:
        # Drop rows with NaN id (corrupt/incomplete rows), then cast
        df = df.dropna(subset=['id'])
        df['id'] = df['id'].astype('int64')

    if 'track_id' not in df.columns:
        df['track_id'] = np.int64(-1)
    else:
        df['track_id'] = df['track_id'].fillna(-1).astype('int64')

    return ParticleData(df)


def _load_thunderstorm_csv(df):
    """Convert a ThunderSTORM DataFrame (nm coords, 1-based frames) to ParticleData."""
    from ..spt.particle_data import ParticleData

    import pandas as pd
    result = pd.DataFrame()

    # Try to get pixel size from column data — default 108 nm/px
    # Let user keep nm values for now; store raw pixel coordinates
    # We need to ask for pixel size or use a default
    pixel_size = 108.0  # default

    if 'id' in df.columns:
        result['id'] = df['id'].astype('int64')
    else:
        result['id'] = np.arange(len(df), dtype=np.int64)

    result['frame'] = df['frame'].astype('int64') - 1  # 1-based → 0-based

    if 'x [nm]' in df.columns:
        result['x'] = df['x [nm]'] / pixel_size
    elif 'x' in df.columns:
        result['x'] = df['x']

    if 'y [nm]' in df.columns:
        result['y'] = df['y [nm]'] / pixel_size
    elif 'y' in df.columns:
        result['y'] = df['y']

    if 'intensity [photon]' in df.columns:
        result['intensity'] = df['intensity [photon]']
    elif 'intensity' in df.columns:
        result['intensity'] = df['intensity']
    else:
        result['intensity'] = 0.0

    # Optional ThunderSTORM columns
    if 'sigma [nm]' in df.columns:
        result['sigma_x'] = df['sigma [nm]'] / pixel_size
        result['sigma_y'] = result['sigma_x']
    if 'sigma1 [nm]' in df.columns:
        result['sigma_x'] = df['sigma1 [nm]'] / pixel_size
    if 'sigma2 [nm]' in df.columns:
        result['sigma_y'] = df['sigma2 [nm]'] / pixel_size
    if 'uncertainty [nm]' in df.columns:
        result['uncertainty'] = df['uncertainty [nm]'] / pixel_size
    if 'offset [photon]' in df.columns:
        result['background'] = df['offset [photon]']
    elif 'bkgstd [photon]' in df.columns:
        result['background'] = df['bkgstd [photon]']

    result['track_id'] = np.int64(-1)

    return ParticleData(result)


def _load_spt_json(path):
    """Load JSON in flika plugin or generic format."""
    import pandas as pd
    from ..spt.particle_data import ParticleData

    with open(path, 'r') as f:
        data = json.load(f)

    # --- Flika pynsight plugin format: {tracks: [[idx, ...], ...], txy_pts: [[t,x,y], ...]} ---
    if 'txy_pts' in data:
        txy = np.array(data['txy_pts'], dtype=np.float64)
        df = pd.DataFrame({
            'id': np.arange(len(txy), dtype=np.int64),
            'frame': txy[:, 0].astype('int64'),
            'x': txy[:, 1],
            'y': txy[:, 2],
            'intensity': np.zeros(len(txy)),
        })
        df['track_id'] = np.int64(-1)

        tracks_raw = data.get('tracks', [])
        if tracks_raw:
            for tid, indices in enumerate(tracks_raw):
                if len(indices) > 1:  # only linked tracks
                    valid = [i for i in indices if 0 <= i < len(df)]
                    df.loc[valid, 'track_id'] = np.int64(tid)

        return ParticleData(df)

    # --- Generic JSON: {tracks: [{track_id, points: [{frame, x, y}, ...]}, ...]} ---
    if 'tracks' in data and isinstance(data['tracks'], list):
        rows = []
        for track in data['tracks']:
            if isinstance(track, dict):
                tid = track.get('track_id', track.get('id', 0))
                for pt in track.get('points', []):
                    rows.append({
                        'frame': int(pt['frame']),
                        'x': float(pt['x']),
                        'y': float(pt['y']),
                        'intensity': float(pt.get('intensity', 0)),
                        'track_id': int(tid),
                    })
        if rows:
            df = pd.DataFrame(rows)
            df['id'] = np.arange(len(df), dtype=np.int64)
            return ParticleData(df)

    # --- Generic JSON: {localizations: [{frame, x, y}, ...]} ---
    if 'localizations' in data:
        rows = []
        for pt in data['localizations']:
            rows.append({
                'frame': int(pt['frame']),
                'x': float(pt['x']),
                'y': float(pt['y']),
                'intensity': float(pt.get('intensity', 0)),
            })
        if rows:
            df = pd.DataFrame(rows)
            df['id'] = np.arange(len(df), dtype=np.int64)
            df['track_id'] = np.int64(-1)
            return ParticleData(df)

    raise ValueError("JSON file does not contain recognized SPT data "
                     "(expected 'txy_pts', 'tracks', or 'localizations' key)")


def _load_spt_txt(path):
    """Load headerless space/tab-delimited text file.

    Expected formats:
    - 3 columns: ``[frame, x, y]``
    - 4 columns: ``[frame, x, y, intensity]``
    - 5 columns: ``[frame, x, y, intensity, track_id]``
    """
    import pandas as pd
    from ..spt.particle_data import ParticleData

    pts = np.loadtxt(path)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    n_cols = pts.shape[1]
    if n_cols < 3:
        raise ValueError(f"Text file has only {n_cols} columns, need at least 3 "
                         "(frame, x, y)")

    df = pd.DataFrame()
    df['id'] = np.arange(len(pts), dtype=np.int64)
    df['frame'] = pts[:, 0].astype('int64')
    df['x'] = pts[:, 1]
    df['y'] = pts[:, 2]

    if n_cols >= 4:
        df['intensity'] = pts[:, 3]
    else:
        df['intensity'] = 0.0

    if n_cols >= 5:
        df['track_id'] = pts[:, 4].astype('int64')
    else:
        df['track_id'] = np.int64(-1)

    return ParticleData(df)



########################################################################################################################
######################                INTERNAL HELPER FUNCTIONS                              ###########################
########################################################################################################################
def get_permutation_tuple(src, dst):
    """get_permtation_tuple(src, dst)

    Parameters:
        src (list): The original ordering of the axes in the tiff.
        dst (list): The desired ordering of the axes in the tiff.

    Returns:
        result (tuple): The required permutation so the axes are ordered as desired.
    """
    result = []
    for i in dst:
        result.append(src.index(i))
    result = tuple(result)
    return result

def append_recent_file(fname):
    fname = os.path.abspath(fname)
    if fname in g.settings['recent_files']:
        g.settings['recent_files'].remove(fname)
    if os.path.exists(fname):
        g.settings['recent_files'].append(fname)
        if len(g.settings['recent_files']) > 8:
            g.settings['recent_files'] = g.settings['recent_files'][-8:]
    return fname


def get_metadata_tiff(Tiff):
    metadata = {}
    page0 = Tiff.pages[0]
    if hasattr(page0, 'is_micromanager') and page0.is_micromanager:
        imagej_tags_unpacked = {}
        if hasattr(page0, 'imagej_tags'):
            imagej_tags = page0.imagej_tags
            imagej_tags['info']
            imagej_tags_unpacked = json.loads(imagej_tags['info'])
        micromanager_metadata = page0.tags['micromanager_metadata']
        metadata = {**micromanager_metadata.value, **imagej_tags_unpacked}
        if 'Frames' in metadata and metadata['Frames'] > 1:
            timestamps = [c.tags['micromanager_metadata'].value['ElapsedTime-ms'] for c in Tiff.pages]
            metadata['timestamps'] = timestamps
            metadata['timestamp_units'] = 'ms'
        keys_to_remove = ['NextFrame', 'ImageNumber', 'Frame', 'FrameIndex']
        for key in keys_to_remove:
            metadata.pop(key, None)
    else:
        try:
            metadata = page0.image_description
            metadata = txt2dict(metadata)
        except AttributeError:
            metadata = dict()
    if hasattr(page0, 'is_rgb'):
        metadata['is_rgb'] = page0.is_rgb
    else:
        metadata['is_rgb'] = (
            page0.photometric == 2  # RGB
            and page0.samplesperpixel >= 3
        )
    return metadata



def txt2dict(metadata):
    meta = dict()
    try:
        metadata = json.loads(metadata.decode('utf-8'))
        return metadata
    except ValueError:  # if the metadata isn't in JSON
        pass
    for line in metadata.splitlines():
        line = re.split('[:=]', line.decode())
        if len(line) == 1:
            meta[line[0]] = ''
        else:
            meta[line[0].lstrip().rstrip()] = line[1].lstrip().rstrip()
    return meta


def JSONhandler(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        json.JSONEncoder().default(obj)


def close(windows=None):
    """close(window=None)
    Will close a window or a set of windows.

    Parameters:
        'all' (str): closes all windows
        windows (list): closes each window in the list
        Window: closes individual window
        (None): closes current window

    """
    if isinstance(windows, str):
        if windows == 'all':
            windows = [window for window in g.windows]
            for window in windows:
                window.close()
    elif isinstance(windows,list):
        for window in windows:
            if isinstance(window,Window):
                window.close()
    elif isinstance(windows,Window):
        windows.close()
    elif windows is None:
        if g.win is not None:
            g.win.close()


########################################################################################################################
######################             OLD FUNCTIONS THAT MIGHT BE USEFUL SOMEDAY                ###########################
########################################################################################################################
"""

def save_roi_traces(filename):
    g.status_msg('Saving traces to {}'.format(os.path.basename(filename)))
    to_save = [roi.getTrace() for roi in g.win.rois]
    np.savetxt(filename, np.transpose(to_save), header='\t'.join(['ROI %d' % i for i in range(len(to_save))]), fmt='%.4f', delimiter='\t', comments='')
    g.settings['filename'] = filename
    g.status_msg('Successfully saved traces to {}'.format(os.path.basename(filename)))

def load_metadata(filename=None):
    '''This function loads the .txt file corresponding to a file into a dictionary
    The .txt is a file which includes database connection information'''
    meta=dict()
    if filename is None:
        filename=os.path.splitext(g.settings['filename'])[0]+'.txt'
    BOM = codecs.BOM_UTF8.decode('utf8')
    if not os.path.isfile(filename):
        g.alert("'"+filename+"' is not a file.")
        return dict()
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.lstrip(BOM)
            line=line.split('=')
            meta[line[0].lstrip().rstrip()]=line[1].lstrip().rstrip()
    for k in meta.keys():
        if meta[k].isdigit():
            meta[k]=int(meta[k])
        else:
            try:
                meta[k]=float(meta[k])
            except ValueError:
                pass
    return meta

def save_metadata(meta):
    filename=os.path.splitext(g.settings['filename'])[0]+'.txt'
    f=open(filename, 'w')
    text=''
    for item in meta.items():
        text+="{}={}\n".format(item[0],item[1])
    f.write(text)
    f.close()


def save_current_frame(filename):
    "" save_current_frame(filename)
    Save the current single frame image of the currentWindow to a .tif file.

    Parameters:
        | filename (str) -- Address to save the frame to.
    ""
    if os.path.dirname(filename)=='': #if the user didn't specify a directory
        directory=os.path.normpath(os.path.dirname(g.settings['filename']))
        filename=os.path.join(directory,filename)
    g.status_msg('Saving {}'.format(os.path.basename(filename)))
    A=np.average(g.win.image, 0)#.astype(g.settings['internal_data_type'])
    metadata=json.dumps(g.win.metadata)
    if len(A.shape)==3:
        A = A[g.win.currentIndex]
        A=np.transpose(A,(0,2,1)) # This keeps the x and the y the same as in FIJI
    elif len(A.shape)==2:
        A=np.transpose(A,(1,0))
    tifffile.imsave(filename, A, description=metadata) #http://stackoverflow.com/questions/20529187/what-is-the-best-way-to-save-image-metadata-alongside-a-tif-with-python
    g.status_msg('Successfully saved {}'.format(os.path.basename(filename)))

def make_recent_menu():
    g.m.menuRecent_Files.clear()
    if len(g.settings['recent_files']) == 0:
        no_recent = QtWidgets.QAction("No Recent Files", g.m)
        no_recent.setEnabled(False)
        g.m.menuRecent_Files.addAction(no_recent)
        return
    def openFun(f):
        return lambda: open_file(append_recent_file(f))
    for fname in g.settings['recent_files'][:10]:
        if os.path.exists(fname):
            g.m.menuRecent_Files.addAction(QtWidgets.QAction(fname, g.m, triggered=openFun(fname)))

"""

logger.debug("Completed 'reading process/file_.py'")
