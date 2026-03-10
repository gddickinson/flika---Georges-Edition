# -*- coding: utf-8 -*-
"""Per-track 6-panel analysis window for Single Particle Tracking.

Ported from the locsAndTracksPlotter plugin.  Opens a dedicated widget
with six synchronized panels showing kinematic, spatial, and intensity
features for a single selected track.

Panels
------
1. Intensity vs frame
2. Distance from origin vs frame
3. 2D trajectory (origin-centered, aspect=1)
4. Nearest-neighbor distance vs frame
5. Instantaneous velocity vs frame
6. Rolling intensity variance vs frame
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from flika.logger import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROLLING_WINDOW = 5          # default rolling-variance window size
_ARROW_SPACING = 5           # place direction arrows every N points
_ARROW_SIZE = 8              # arrow head size in pixels
_PANEL_TITLES = [
    "Intensity vs Frame",
    "Distance from Origin",
    "2D Trajectory",
    "Nearest-Neighbor Distance",
    "Instantaneous Velocity",
    "Intensity Variance (rolling)",
]

# Plot pen / brush colours
_PEN_MAIN = pg.mkPen('#4fc3f7', width=2)
_PEN_VARIANCE = pg.mkPen('#ab47bc', width=2)
_PEN_VELOCITY = pg.mkPen('#66bb6a', width=2)
_PEN_NN = pg.mkPen('#ffa726', width=2)
_PEN_DIST = pg.mkPen('#ef5350', width=2)
_PEN_TRAJ = pg.mkPen('#4fc3f7', width=2)
_PEN_FRAME_LINE = pg.mkPen('#ffffff', width=1, style=QtCore.Qt.DashLine)

_BRUSH_START = pg.mkBrush('#4caf50')
_BRUSH_END = pg.mkBrush('#f44336')


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rolling_variance(values, window=_ROLLING_WINDOW):
    """Compute a centered rolling variance over *values*.

    Returns an array the same length as *values*; edges are computed with
    the available (smaller) window.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([])
    out = np.empty(n, dtype=np.float64)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.var(values[lo:hi])
    return out


def _extract_track_positions(locs_array, track_indices):
    """Return (frames, xy, intensities) arrays for a single track.

    Parameters
    ----------
    locs_array : ndarray, shape (N, 3+)
        Columns ``[frame, x, y, intensity?, ...]``.
    track_indices : list of int
        Row indices into *locs_array* belonging to this track.

    Returns
    -------
    frames : ndarray (M,)
    xy : ndarray (M, 2)
    intensities : ndarray (M,) or None
    """
    locs = np.asarray(locs_array, dtype=np.float64)
    rows = locs[track_indices]
    # Sort by frame
    order = np.argsort(rows[:, 0])
    rows = rows[order]

    frames = rows[:, 0].astype(int)
    xy = rows[:, 1:3]
    intensities = rows[:, 3] if rows.shape[1] > 3 else None
    return frames, xy, intensities


# ---------------------------------------------------------------------------
# TrackDetailWindow
# ---------------------------------------------------------------------------

class TrackDetailWindow(QtWidgets.QWidget):
    """6-panel analysis window for a single selected track.

    Panel 1: Intensity vs time (line plot)
    Panel 2: Distance from origin vs time
    Panel 3: 2D trajectory (origin-centered, aspect ratio = 1)
    Panel 4: Nearest neighbor distance vs time
    Panel 5: Instantaneous velocity vs time
    Panel 6: Intensity variance (rolling window)
    """

    def __init__(self, source_window=None, parent=None):
        super().__init__(parent)
        self.source_window = source_window
        self.setWindowTitle("Track Detail")
        self.resize(1000, 700)

        # State
        self._track_id = None
        self._locs_array = None
        self._tracks = None          # list of lists of indices
        self._features_dict = None   # optional precomputed features per track
        self._frames = None          # ndarray of frame numbers for current track
        self._xy = None              # (M, 2) positions for current track
        self._intensities = None     # (M,) or None
        self._frame_lines = []       # InfiniteLine items for frame indicator

        # ----- Top toolbar -----
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 0)

        self.prev_btn = QtWidgets.QPushButton("<< Prev")
        self.prev_btn.setToolTip("Previous track")
        self.prev_btn.clicked.connect(self._on_prev)
        toolbar.addWidget(self.prev_btn)

        toolbar.addWidget(QtWidgets.QLabel("Track:"))
        self.track_combo = QtWidgets.QComboBox()
        self.track_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents)
        self.track_combo.currentIndexChanged.connect(self._on_combo_changed)
        toolbar.addWidget(self.track_combo)

        self.next_btn = QtWidgets.QPushButton("Next >>")
        self.next_btn.setToolTip("Next track")
        self.next_btn.clicked.connect(self._on_next)
        toolbar.addWidget(self.next_btn)

        toolbar.addStretch()

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setTextFormat(QtCore.Qt.RichText)
        self.summary_label.setWordWrap(True)
        toolbar.addWidget(self.summary_label)

        # ----- Plot panels (3 x 2 grid) -----
        self.panels = []
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        for idx in range(6):
            pw = pg.PlotWidget(title=_PANEL_TITLES[idx])
            pw.showGrid(x=True, y=True, alpha=0.25)
            row, col = divmod(idx, 2)
            grid.addWidget(pw, row, col)
            self.panels.append(pw)

        # Axis labels for time-based panels
        for i in (0, 1, 3, 4, 5):
            self.panels[i].setLabel('bottom', 'Frame')

        self.panels[0].setLabel('left', 'Intensity')
        self.panels[1].setLabel('left', 'Distance (px)')
        self.panels[2].setLabel('bottom', 'X (px)')
        self.panels[2].setLabel('left', 'Y (px)')
        self.panels[3].setLabel('left', 'NN Distance (px)')
        self.panels[4].setLabel('left', 'Velocity (px/frame)')
        self.panels[5].setLabel('left', 'Variance')

        # Lock aspect ratio for the trajectory panel
        self.panels[2].getViewBox().setAspectLocked(True)

        # ----- Assemble layout -----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addLayout(toolbar)
        main_layout.addLayout(grid, stretch=1)

        # Connect to source window frame changes
        if self.source_window is not None:
            try:
                self.source_window.sigTimeChanged.connect(
                    self.set_frame_indicator)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, locs_array, tracks, features_dict=None):
        """Supply the full localisation / track data.

        Parameters
        ----------
        locs_array : ndarray, shape (N, 3+)
            Columns ``[frame, x, y, intensity?, ...]``.
        tracks : list of list of int
            Each element is a list of row indices into *locs_array*.
        features_dict : dict, optional
            Mapping ``{track_id: {feature_name: value}}``.
        """
        self._locs_array = np.asarray(locs_array, dtype=np.float64)
        self._tracks = tracks
        self._features_dict = features_dict or {}

        # Populate combo box
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for tid in range(len(tracks)):
            n_pts = len(tracks[tid])
            self.track_combo.addItem(f"{tid}  ({n_pts} pts)", userData=tid)
        self.track_combo.blockSignals(False)

        if len(tracks) > 0:
            self.track_combo.setCurrentIndex(0)
            self._select_track(0)

    def set_track(self, track_id, locs_array=None, tracks=None,
                  features_dict=None):
        """Display panels for a specific track.

        If *locs_array* and *tracks* are provided they replace the
        currently stored data.  Otherwise the previously loaded data is
        used.

        Parameters
        ----------
        track_id : int
        locs_array : ndarray, optional
        tracks : list of list of int, optional
        features_dict : dict, optional
        """
        if locs_array is not None:
            self._locs_array = np.asarray(locs_array, dtype=np.float64)
        if tracks is not None:
            self._tracks = tracks
        if features_dict is not None:
            self._features_dict = features_dict

        if self._locs_array is None or self._tracks is None:
            logger.warning("TrackDetailWindow.set_track called without data")
            return
        if track_id < 0 or track_id >= len(self._tracks):
            logger.warning("track_id %d out of range (0..%d)",
                           track_id, len(self._tracks) - 1)
            return

        self._select_track(track_id)

    def set_frame_indicator(self, frame):
        """Draw / move a vertical line on every time-based panel.

        Parameters
        ----------
        frame : int
            Current frame number.
        """
        # Remove old lines
        for pw, line in self._frame_lines:
            try:
                pw.removeItem(line)
            except Exception:
                pass
        self._frame_lines.clear()

        # Time-based panels: 0, 1, 3, 4, 5
        for idx in (0, 1, 3, 4, 5):
            pw = self.panels[idx]
            line = pg.InfiniteLine(
                pos=frame, angle=90, pen=_PEN_FRAME_LINE, movable=False)
            pw.addItem(line)
            self._frame_lines.append((pw, line))

    # ------------------------------------------------------------------
    # Navigation slots
    # ------------------------------------------------------------------

    def _on_prev(self):
        idx = self.track_combo.currentIndex()
        if idx > 0:
            self.track_combo.setCurrentIndex(idx - 1)

    def _on_next(self):
        idx = self.track_combo.currentIndex()
        if idx < self.track_combo.count() - 1:
            self.track_combo.setCurrentIndex(idx + 1)

    def _on_combo_changed(self, index):
        if index < 0:
            return
        tid = self.track_combo.itemData(index)
        if tid is not None:
            self._select_track(tid)

    # ------------------------------------------------------------------
    # Internal: select and draw
    # ------------------------------------------------------------------

    def _select_track(self, track_id):
        """Extract data for *track_id* and refresh all panels."""
        self._track_id = track_id
        indices = self._tracks[track_id]
        self._frames, self._xy, self._intensities = _extract_track_positions(
            self._locs_array, indices)

        # If no explicit intensity column, try to pull from source window
        if self._intensities is None and self.source_window is not None:
            self._intensities = self._extract_intensity_from_window()

        self.setWindowTitle(f"Track Detail  --  Track {track_id}")

        # Sync combo without triggering infinite loop
        combo_tid = self.track_combo.itemData(self.track_combo.currentIndex())
        if combo_tid != track_id:
            self.track_combo.blockSignals(True)
            for i in range(self.track_combo.count()):
                if self.track_combo.itemData(i) == track_id:
                    self.track_combo.setCurrentIndex(i)
                    break
            self.track_combo.blockSignals(False)

        # Update all panels
        self._update_panel_1()
        self._update_panel_2()
        self._update_panel_3()
        self._update_panel_4()
        self._update_panel_5()
        self._update_panel_6()
        self._update_summary()

        # Redraw frame indicator at current source frame
        if self.source_window is not None:
            try:
                self.set_frame_indicator(self.source_window.currentIndex)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Panel update methods
    # ------------------------------------------------------------------

    def _update_panel_1(self):
        """Panel 1: Intensity vs frame."""
        pw = self.panels[0]
        pw.clear()

        if self._intensities is None or len(self._intensities) == 0:
            pw.setTitle("Intensity vs Frame  (no data)")
            return

        pw.setTitle("Intensity vs Frame")
        pw.plot(self._frames, self._intensities, pen=_PEN_MAIN)

    def _update_panel_2(self):
        """Panel 2: Distance from origin vs frame."""
        pw = self.panels[1]
        pw.clear()

        if self._xy is None or len(self._xy) < 1:
            return

        try:
            from flika.spt.features.kinematic import distance_from_origin
            dists = distance_from_origin(self._xy)
        except ImportError:
            logger.debug("spt.features.kinematic not available; "
                         "computing distance inline")
            origin = self._xy[0]
            dists = np.sqrt(np.sum((self._xy - origin) ** 2, axis=1))

        pw.plot(self._frames, dists, pen=_PEN_DIST)

    def _update_panel_3(self):
        """Panel 3: 2D trajectory, origin-centered, equal aspect."""
        pw = self.panels[2]
        pw.clear()

        if self._xy is None or len(self._xy) < 2:
            return

        # Centre at origin (first point)
        origin = self._xy[0].copy()
        xs = self._xy[:, 0] - origin[0]
        ys = self._xy[:, 1] - origin[1]

        # Main trajectory line
        pw.plot(xs, ys, pen=_PEN_TRAJ)

        # Start marker (green circle)
        pw.plot([xs[0]], [ys[0]], pen=None,
                symbol='o', symbolSize=10, symbolBrush=_BRUSH_START)

        # End marker (red circle)
        pw.plot([xs[-1]], [ys[-1]], pen=None,
                symbol='o', symbolSize=10, symbolBrush=_BRUSH_END)

        # Direction arrows along the path
        n = len(xs)
        if n > _ARROW_SPACING:
            arrow_indices = list(range(_ARROW_SPACING, n - 1, _ARROW_SPACING))
            for ai in arrow_indices:
                dx = xs[ai] - xs[ai - 1]
                dy = ys[ai] - ys[ai - 1]
                length = np.sqrt(dx * dx + dy * dy)
                if length < 1e-9:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                arrow = pg.ArrowItem(
                    pos=(xs[ai], ys[ai]),
                    angle=-angle + 180,  # pyqtgraph arrow convention
                    tipAngle=30,
                    headLen=_ARROW_SIZE,
                    tailLen=0,
                    pen=pg.mkPen('#4fc3f7'),
                    brush=pg.mkBrush('#4fc3f7'),
                )
                pw.addItem(arrow)

    def _update_panel_4(self):
        """Panel 4: Nearest-neighbor distance vs frame.

        For each frame in the track, compute the NN distance to the
        closest localization in *all* localizations on that frame.
        """
        pw = self.panels[3]
        pw.clear()

        if (self._locs_array is None or self._frames is None
                or len(self._frames) == 0):
            return

        try:
            from flika.spt.features.spatial import nearest_neighbor_distances
        except ImportError:
            logger.debug("spt.features.spatial not available; skipping "
                         "panel 4")
            pw.setTitle("Nearest-Neighbor Distance  (unavailable)")
            return

        locs = self._locs_array
        nn_dists = np.full(len(self._frames), np.nan)

        for i, (fr, x, y) in enumerate(
                zip(self._frames, self._xy[:, 0], self._xy[:, 1])):
            # All localizations on this frame
            frame_mask = locs[:, 0] == fr
            frame_xy = locs[frame_mask][:, 1:3]
            if len(frame_xy) < 2:
                continue
            # Find this point in the frame set
            dists = np.sqrt(np.sum((frame_xy - np.array([x, y])) ** 2,
                                   axis=1))
            # Nearest that is not self (distance > 0)
            dists[dists == 0] = np.inf
            nn_dists[i] = np.min(dists)

        valid = ~np.isnan(nn_dists)
        if np.any(valid):
            pw.plot(self._frames[valid], nn_dists[valid], pen=_PEN_NN)

    def _update_panel_5(self):
        """Panel 5: Instantaneous velocity vs frame."""
        pw = self.panels[4]
        pw.clear()

        if self._xy is None or len(self._xy) < 2:
            return

        try:
            from flika.spt.features.kinematic import velocity_analysis
            result = velocity_analysis(self._xy, dt=1.0)
            velocities = result['instantaneous_velocities']
        except ImportError:
            logger.debug("spt.features.kinematic not available; "
                         "computing velocity inline")
            steps = np.diff(self._xy, axis=0)
            velocities = np.sqrt(np.sum(steps ** 2, axis=1))

        if len(velocities) == 0:
            return

        # Velocity is defined between consecutive frames; plot at the
        # midpoint frame or simply at frames[1:]
        frames_vel = self._frames[1:]
        if len(frames_vel) != len(velocities):
            # Defensive: trim to shorter
            m = min(len(frames_vel), len(velocities))
            frames_vel = frames_vel[:m]
            velocities = velocities[:m]

        pw.plot(frames_vel, velocities, pen=_PEN_VELOCITY)

    def _update_panel_6(self):
        """Panel 6: Rolling intensity variance vs frame."""
        pw = self.panels[5]
        pw.clear()

        if self._intensities is None or len(self._intensities) < 2:
            pw.setTitle("Intensity Variance  (no data)")
            return

        pw.setTitle("Intensity Variance (rolling)")
        variance = _rolling_variance(self._intensities, window=_ROLLING_WINDOW)
        pw.plot(self._frames, variance, pen=_PEN_VARIANCE)

    # ------------------------------------------------------------------
    # Summary label
    # ------------------------------------------------------------------

    def _update_summary(self):
        """Build a short HTML summary of the track's aggregate features."""
        tid = self._track_id
        parts = [f"<b>Track {tid}</b>"]
        n_pts = len(self._frames) if self._frames is not None else 0
        parts.append(f"&nbsp; pts={n_pts}")

        if self._frames is not None and len(self._frames) > 0:
            parts.append(f"frames {int(self._frames[0])}-"
                         f"{int(self._frames[-1])}")

        # Try precomputed features
        feat = self._features_dict.get(tid, {}) if self._features_dict else {}

        if 'radius_gyration' in feat:
            parts.append(f"Rg={feat['radius_gyration']:.2f}")
        if 'diffusion_coefficient' in feat:
            parts.append(f"D={feat['diffusion_coefficient']:.4f}")
        if 'anomalous_exponent' in feat:
            parts.append(f"alpha={feat['anomalous_exponent']:.2f}")
        if 'classification' in feat:
            parts.append(f"class={feat['classification']}")

        # If no precomputed features, compute a few on the fly
        if not feat and self._xy is not None and len(self._xy) >= 3:
            try:
                from flika.spt.features.geometric import radius_of_gyration
                rg_result = radius_of_gyration(self._xy)
                parts.append(f"Rg={rg_result['rg']:.2f}")
            except Exception:
                pass
            try:
                from flika.spt.features.kinematic import msd_analysis
                msd_result = msd_analysis(self._xy)
                parts.append(
                    f"D={msd_result['diffusion_coefficient']:.4f}")
                parts.append(
                    f"alpha={msd_result['anomalous_exponent']:.2f}")
            except Exception:
                pass

        self.summary_label.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))

    # ------------------------------------------------------------------
    # Intensity extraction from source window
    # ------------------------------------------------------------------

    def _extract_intensity_from_window(self):
        """Pull intensity values from the source window's image data.

        Uses the track's (frame, x, y) to sample pixel values.

        Returns
        -------
        ndarray (M,) or None
        """
        win = self.source_window
        if win is None:
            return None

        try:
            image = win.image
        except Exception:
            return None

        if image is None:
            return None

        ndim = image.ndim
        n_pts = len(self._frames)
        intensities = np.empty(n_pts, dtype=np.float64)

        for i in range(n_pts):
            fr = int(self._frames[i])
            ix = int(round(self._xy[i, 0]))
            iy = int(round(self._xy[i, 1]))
            try:
                if ndim >= 3:
                    # 3-D or 4-D stack: first axis is time
                    frame_img = image[fr]
                elif ndim == 2:
                    frame_img = image
                else:
                    intensities[i] = np.nan
                    continue

                # Bounds check
                if (0 <= ix < frame_img.shape[0]
                        and 0 <= iy < frame_img.shape[1]):
                    val = frame_img[ix, iy]
                    # Handle RGB by taking mean
                    if hasattr(val, '__len__'):
                        val = float(np.mean(val))
                    intensities[i] = float(val)
                else:
                    intensities[i] = np.nan
            except Exception:
                intensities[i] = np.nan

        if np.all(np.isnan(intensities)):
            return None
        return intensities

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Disconnect source-window signals on close."""
        if self.source_window is not None:
            try:
                self.source_window.sigTimeChanged.disconnect(
                    self.set_frame_indicator)
            except (TypeError, RuntimeError):
                pass
        # Remove frame indicator lines
        for pw, line in self._frame_lines:
            try:
                pw.removeItem(line)
            except Exception:
                pass
        self._frame_lines.clear()
        event.accept()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_track_detail(window=None, track_id=None):
    """Open a :class:`TrackDetailWindow` for the given window's SPT data.

    The window must have SPT metadata stored under
    ``window.metadata['spt']`` with at least ``'localizations'`` (ndarray)
    and ``'tracks'`` (list of lists of indices).

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        Source image window.  Defaults to ``g.win``.
    track_id : int, optional
        Track to display initially.  Defaults to the first track.

    Returns
    -------
    TrackDetailWindow
    """
    import flika.global_vars as g

    if window is None:
        window = g.win
    if window is None:
        logger.error("No active window for track detail viewer")
        raise RuntimeError("No active window.  Open an image first.")

    spt = window.metadata.get('spt', {})
    locs = spt.get('localizations', None)
    tracks = spt.get('tracks', None)

    if locs is None or tracks is None:
        raise ValueError(
            "The active window does not contain SPT data.  "
            "Run particle detection and linking first, or load tracks "
            "via the Track Overlay panel.")

    locs = np.asarray(locs, dtype=np.float64)
    features = spt.get('features_by_track', {})

    detail = TrackDetailWindow(source_window=window)
    detail.set_data(locs, tracks, features_dict=features)

    if track_id is not None and 0 <= track_id < len(tracks):
        detail.set_track(track_id)

    detail.show()
    detail.raise_()

    logger.info("Opened TrackDetailWindow for window '%s' (%d tracks)",
                getattr(window, 'name', '?'), len(tracks))
    return detail
