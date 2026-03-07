# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/export.py'")
import numpy as np
import os
import tempfile
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess_noPriorWindow, BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox
from ..utils.misc import save_file_gui

__all__ = ['video_exporter']


class Video_Exporter(BaseProcess_noPriorWindow):
    """video_exporter()

    Export a 3D image stack (time series) as a video file.

    Supports H.264 (mp4) and MJPEG (avi) codecs via imageio or OpenCV.
    Frames are normalized to uint8 for video encoding.

    Parameters:
        window: The source window containing a 3D image stack.
        framerate (int): Frames per second (1-120, default 30).
        start_frame (int): First frame to export (default 0).
        end_frame (int): Last frame to export (default 0, meaning last frame).
        codec (str): Video codec - 'H.264 (mp4)' or 'MJPEG (avi)'.

    Returns:
        None
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window = WindowSelector()
        framerate = SliderLabel(0)
        framerate.setRange(1, 120)
        framerate.setValue(30)
        start_frame = SliderLabel(0)
        start_frame.setRange(0, 99999)
        start_frame.setValue(0)
        end_frame = SliderLabel(0)
        end_frame.setRange(0, 99999)
        end_frame.setValue(0)
        codec = ComboBox()
        codec.addItems(['H.264 (mp4)', 'MJPEG (avi)'])
        self.items.append({'name': 'window', 'string': 'Window', 'object': window})
        self.items.append({'name': 'framerate', 'string': 'Frame Rate (fps)', 'object': framerate})
        self.items.append({'name': 'start_frame', 'string': 'Start Frame', 'object': start_frame})
        self.items.append({'name': 'end_frame', 'string': 'End Frame', 'object': end_frame})
        self.items.append({'name': 'codec', 'string': 'Codec', 'object': codec})
        super().gui()
        # Replace the OK button with an Export button
        self.ui.bbox.clear()
        export_btn = QtWidgets.QPushButton('Export')
        export_btn.clicked.connect(self.export_video)
        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.clicked.connect(self.ui.reject)
        self.ui.bbox.addButton(export_btn, QtWidgets.QDialogButtonBox.AcceptRole)
        self.ui.bbox.addButton(cancel_btn, QtWidgets.QDialogButtonBox.RejectRole)

    def export_video(self):
        """Export the selected window's image stack as a video file."""
        window = self.getValue('window')
        framerate = int(self.getValue('framerate'))
        start = int(self.getValue('start_frame'))
        end = int(self.getValue('end_frame'))
        codec = self.getValue('codec')

        if window is None:
            g.alert("No window selected. Please select a window first.")
            return

        image = window.image
        if image.ndim < 3:
            g.alert("Video export requires a 3D image stack (time series).")
            return

        nframes = image.shape[0]
        if end == 0 or end >= nframes:
            end = nframes - 1
        if start < 0:
            start = 0
        if start > end:
            g.alert("Start frame must be less than or equal to end frame.")
            return

        # Determine file extension and get save path
        if 'mp4' in codec:
            ext = 'mp4'
            filetypes = '*.mp4'
        else:
            ext = 'avi'
            filetypes = '*.avi'

        filepath = save_file_gui("Save Video As", filetypes=filetypes)
        if filepath is None or filepath == '':
            return

        # Ensure correct extension
        if not filepath.endswith('.' + ext):
            filepath = filepath + '.' + ext

        g.status_msg('Exporting video...')
        QtWidgets.QApplication.processEvents()

        try:
            self._export_with_imageio(image, filepath, start, end, framerate, codec)
        except ImportError:
            logger.info("imageio not available, falling back to OpenCV for video export.")
            try:
                self._export_with_cv2(image, filepath, start, end, framerate, codec)
            except ImportError:
                g.alert("Video export requires either 'imageio' or 'opencv-python'. "
                        "Install one with: pip install imageio[ffmpeg] or pip install opencv-python")
                return
            except Exception as e:
                g.alert("Video export failed (cv2): {}".format(str(e)))
                logger.error("Video export error (cv2): {}".format(str(e)))
                return
        except Exception as e:
            g.alert("Video export failed (imageio): {}".format(str(e)))
            logger.error("Video export error (imageio): {}".format(str(e)))
            return

        g.status_msg('Video exported successfully to {}'.format(filepath))

    def _normalize_frame(self, frame):
        """Normalize a single frame to uint8 (0-255)."""
        frame = frame.astype(np.float64)
        fmin = frame.min()
        fmax = frame.max()
        if fmax - fmin == 0:
            return np.zeros(frame.shape, dtype=np.uint8)
        frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        return frame

    def _export_with_imageio(self, image, filepath, start, end, framerate, codec):
        """Export video using imageio."""
        import imageio
        writer = imageio.get_writer(filepath, fps=framerate)
        for i in range(start, end + 1):
            frame = image[i]
            frame = self._normalize_frame(frame)
            # imageio expects (H, W) or (H, W, 3); grayscale is fine
            writer.append_data(frame)
        writer.close()

    def _export_with_cv2(self, image, filepath, start, end, framerate, codec):
        """Export video using OpenCV as fallback."""
        import cv2
        h, w = image.shape[1], image.shape[2]
        if 'mp4' in codec:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), isColor=False)
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer for '{}'".format(filepath))
        for i in range(start, end + 1):
            frame = image[i]
            frame = self._normalize_frame(frame)
            writer.write(frame)
        writer.release()

    def __call__(self):
        """Open the video export dialog."""
        self.gui()


video_exporter = Video_Exporter()

logger.debug("Completed 'reading process/export.py'")
