# -*- coding: utf-8 -*-
"""Calcium imaging analysis.

Provides dF/F computation, calcium transient event detection, event
statistics, and trace smoothing.
"""
from flika.logger import logger
logger.debug("Started 'reading process/calcium.py'")
import numpy as np
from scipy import ndimage, signal
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
import flika.global_vars as g
from flika.utils.BaseProcess import BaseProcess
from flika.utils.custom_widgets import WindowSelector, SliderLabel, CheckBox, ComboBox, MissingWindowError

__all__ = ['calcium_analysis']


# ---------------------------------------------------------------------------
# Pure analysis functions (no Qt dependencies)
# ---------------------------------------------------------------------------

def compute_dff(trace, baseline_frames=None, baseline_method='mean',
                percentile=10):
    """Compute Delta-F/F0 (dF/F) for a fluorescence trace.

    Parameters
    ----------
    trace : 1-D array
        Raw fluorescence intensity over time.
    baseline_frames : slice or array of int, optional
        Frame indices for baseline calculation.  If None, uses the first
        10% of frames.
    baseline_method : str
        'mean', 'median', or 'percentile'.
    percentile : float
        Percentile for baseline_method='percentile'.

    Returns
    -------
    ndarray
        dF/F0 trace: (F - F0) / F0.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)

    if baseline_frames is None:
        n_baseline = max(1, n // 10)
        baseline_frames = slice(0, n_baseline)

    baseline = trace[baseline_frames]

    if baseline_method == 'median':
        F0 = np.median(baseline)
    elif baseline_method == 'percentile':
        F0 = np.percentile(baseline, percentile)
    else:  # mean
        F0 = np.mean(baseline)

    if abs(F0) < 1e-12:
        return np.zeros_like(trace)

    return (trace - F0) / F0


def compute_dff_image(stack, baseline_frames=None, baseline_method='mean'):
    """Compute pixel-wise dF/F0 for an entire image stack.

    Parameters
    ----------
    stack : 3-D array (T, Y, X)
    baseline_frames : slice, optional
    baseline_method : str

    Returns
    -------
    ndarray (T, Y, X)
        dF/F0 image stack.
    """
    stack = np.asarray(stack, dtype=np.float64)
    T = stack.shape[0]

    if baseline_frames is None:
        n_baseline = max(1, T // 10)
        baseline_frames = slice(0, n_baseline)

    baseline_region = stack[baseline_frames]

    if baseline_method == 'median':
        F0 = np.median(baseline_region, axis=0)
    else:
        F0 = np.mean(baseline_region, axis=0)

    F0 = np.where(np.abs(F0) > 1e-12, F0, 1e-12)
    return (stack - F0[np.newaxis, :, :]) / F0[np.newaxis, :, :]


def detect_calcium_events(trace, threshold=2.0, min_duration=3,
                          min_interval=5):
    """Detect calcium transient events in a dF/F trace.

    Uses threshold crossing with minimum duration and inter-event interval
    constraints.

    Parameters
    ----------
    trace : 1-D array
        dF/F0 trace.
    threshold : float
        Detection threshold in units of baseline standard deviation.
        Events are detected where trace > threshold * baseline_std.
    min_duration : int
        Minimum event duration in frames.
    min_interval : int
        Minimum inter-event interval in frames.

    Returns
    -------
    list of dict
        Each dict has keys: start, peak, end, amplitude, duration,
        rise_time, decay_time, area.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    if n < min_duration:
        return []

    # Use the first 10% as "quiet" baseline for noise estimation
    n_quiet = max(3, n // 10)
    baseline_std = np.std(trace[:n_quiet])
    if baseline_std < 1e-12:
        baseline_std = np.std(trace)
    if baseline_std < 1e-12:
        return []

    thresh_val = threshold * baseline_std
    above = trace > thresh_val

    # Label connected regions above threshold
    labeled, n_events = ndimage.label(above)
    events = []

    for i in range(1, n_events + 1):
        indices = np.where(labeled == i)[0]
        if len(indices) < min_duration:
            continue

        start = int(indices[0])
        end = int(indices[-1])
        duration = end - start + 1

        # Check inter-event interval
        if events and (start - events[-1]['end']) < min_interval:
            continue

        peak_idx = start + int(np.argmax(trace[start:end + 1]))
        amplitude = float(trace[peak_idx])

        # Rise time: start to peak
        rise_time = peak_idx - start

        # Decay time: peak to end
        decay_time = end - peak_idx

        # Area under curve
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        area = float(_trapz(trace[start:end + 1]))

        events.append({
            'start': start,
            'peak': peak_idx,
            'end': end,
            'amplitude': amplitude,
            'duration': duration,
            'rise_time': rise_time,
            'decay_time': decay_time,
            'area': area,
        })

    return events


def compute_calcium_stats(events):
    """Compute summary statistics from detected calcium events.

    Parameters
    ----------
    events : list of dict
        Output from :func:`detect_calcium_events`.

    Returns
    -------
    dict
        Keys: n_events, mean_amplitude, mean_duration, mean_rise_time,
        mean_decay_time, mean_area, frequency (events per frame),
        mean_interval.
    """
    if not events:
        return {
            'n_events': 0,
            'mean_amplitude': np.nan,
            'mean_duration': np.nan,
            'mean_rise_time': np.nan,
            'mean_decay_time': np.nan,
            'mean_area': np.nan,
            'frequency': 0.0,
            'mean_interval': np.nan,
        }

    amps = [e['amplitude'] for e in events]
    durs = [e['duration'] for e in events]
    rises = [e['rise_time'] for e in events]
    decays = [e['decay_time'] for e in events]
    areas = [e['area'] for e in events]

    # Inter-event intervals
    if len(events) > 1:
        intervals = [events[i + 1]['start'] - events[i]['end']
                     for i in range(len(events) - 1)]
        mean_interval = float(np.mean(intervals))
    else:
        mean_interval = np.nan

    # Frequency: events per total time span
    total_span = events[-1]['end'] - events[0]['start'] + 1
    freq = len(events) / total_span if total_span > 0 else 0.0

    return {
        'n_events': len(events),
        'mean_amplitude': float(np.mean(amps)),
        'mean_duration': float(np.mean(durs)),
        'mean_rise_time': float(np.mean(rises)),
        'mean_decay_time': float(np.mean(decays)),
        'mean_area': float(np.mean(areas)),
        'frequency': float(freq),
        'mean_interval': mean_interval,
    }


def smooth_trace(trace, method='savgol', window_length=11, polyorder=3):
    """Smooth a fluorescence trace.

    Parameters
    ----------
    trace : 1-D array
    method : str
        'savgol' (Savitzky-Golay), 'median', or 'gaussian'.
    window_length : int
        Window size (must be odd for savgol).
    polyorder : int
        Polynomial order for Savitzky-Golay filter.

    Returns
    -------
    ndarray
        Smoothed trace.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) < window_length:
        return trace.copy()

    if method == 'savgol':
        # Ensure odd window
        if window_length % 2 == 0:
            window_length += 1
        return signal.savgol_filter(trace, window_length, polyorder)
    elif method == 'median':
        return ndimage.median_filter(trace, size=window_length)
    elif method == 'gaussian':
        sigma = window_length / 6.0
        return ndimage.gaussian_filter1d(trace, sigma)
    else:
        return trace.copy()


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class CalciumAnalysis(BaseProcess):
    """calcium_analysis(window, baseline_frames=10, baseline_method='mean', threshold=2.0, min_duration=3, smooth=False, smooth_method='savgol', use_roi=True, keepSourceWindow=True)

    Calcium imaging analysis pipeline.

    Computes dF/F0, detects calcium transients, and reports event statistics.
    Creates a dF/F image, intensity trace plot with detected events, and
    summary statistics.

    Parameters:
        window (Window): Calcium imaging time-series stack.
        baseline_frames (int): Number of initial frames for F0 baseline.
        baseline_method (str): 'mean', 'median', or 'percentile'.
        threshold (float): Event detection threshold (x baseline std).
        min_duration (int): Minimum event duration (frames).
        smooth (bool): Smooth trace before event detection.
        smooth_method (str): 'savgol', 'median', or 'gaussian'.
        use_roi (bool): Extract trace from active ROI.
    Returns:
        Window -- dF/F0 image stack.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window = WindowSelector()
        baseline_frames = SliderLabel(decimals=0)
        baseline_frames.setRange(1, 1000)
        baseline_frames.setValue(10)
        baseline_method = ComboBox()
        baseline_method.addItem('mean')
        baseline_method.addItem('median')
        baseline_method.addItem('percentile')
        threshold = SliderLabel(decimals=1)
        threshold.setRange(0.5, 20.0)
        threshold.setValue(2.0)
        min_duration = SliderLabel(decimals=0)
        min_duration.setRange(1, 100)
        min_duration.setValue(3)
        smooth = CheckBox()
        smooth.setChecked(False)
        smooth_method = ComboBox()
        smooth_method.addItem('savgol')
        smooth_method.addItem('median')
        smooth_method.addItem('gaussian')
        use_roi = CheckBox()
        use_roi.setChecked(True)

        self.items.append({'name': 'window', 'string': 'Window', 'object': window})
        self.items.append({'name': 'baseline_frames', 'string': 'Baseline frames', 'object': baseline_frames})
        self.items.append({'name': 'baseline_method', 'string': 'Baseline method', 'object': baseline_method})
        self.items.append({'name': 'threshold', 'string': 'Threshold (x std)', 'object': threshold})
        self.items.append({'name': 'min_duration', 'string': 'Min duration (frames)', 'object': min_duration})
        self.items.append({'name': 'smooth', 'string': 'Smooth trace', 'object': smooth})
        self.items.append({'name': 'smooth_method', 'string': 'Smooth method', 'object': smooth_method})
        self.items.append({'name': 'use_roi', 'string': 'Use ROI', 'object': use_roi})
        super().gui()

    def __call__(self, window, baseline_frames=10, baseline_method='mean',
                 threshold=2.0, min_duration=3, smooth=False,
                 smooth_method='savgol', use_roi=True, keepSourceWindow=True):
        if window is None:
            raise MissingWindowError("A window must be selected for calcium analysis.")

        img = window.image
        if img.ndim < 3:
            g.alert("Calcium analysis requires a time-series (3-D) stack.")
            return

        n_frames = img.shape[0]

        # Build ROI mask
        mask = None
        if use_roi:
            if hasattr(window, 'currentROI') and window.currentROI is not None:
                roi = window.currentROI
                yy, xx = roi.getMask()
                if yy.size > 0:
                    mask = np.zeros(img.shape[1:3], dtype=bool)
                    valid = (
                        (yy >= 0) & (yy < img.shape[1]) &
                        (xx >= 0) & (xx < img.shape[2])
                    )
                    mask[yy[valid], xx[valid]] = True

        # Compute dF/F image stack
        bl_slice = slice(0, int(baseline_frames))
        dff_stack = compute_dff_image(img, baseline_frames=bl_slice,
                                      baseline_method=baseline_method)

        # Extract trace
        if mask is not None:
            raw_trace = np.array([np.mean(img[i][mask]) for i in range(n_frames)])
        else:
            raw_trace = np.array([np.mean(img[i]) for i in range(n_frames)])

        # Compute dF/F trace
        dff_trace = compute_dff(raw_trace, baseline_frames=bl_slice,
                                baseline_method=baseline_method)

        # Optionally smooth
        if smooth:
            analysis_trace = smooth_trace(dff_trace, method=smooth_method)
        else:
            analysis_trace = dff_trace

        # Detect events
        events = detect_calcium_events(analysis_trace, threshold=threshold,
                                       min_duration=int(min_duration))
        stats = compute_calcium_stats(events)

        # Results
        name = window.name
        results = {
            'baseline_frames': int(baseline_frames),
            'baseline_method': baseline_method,
            'threshold': float(threshold),
            'raw_trace': raw_trace,
            'dff_trace': dff_trace,
            'smoothed_trace': analysis_trace if smooth else None,
            'events': events,
            'n_events': stats['n_events'],
            'mean_amplitude': stats['mean_amplitude'],
            'mean_duration': stats['mean_duration'],
            'mean_rise_time': stats['mean_rise_time'],
            'mean_decay_time': stats['mean_decay_time'],
            'mean_area': stats['mean_area'],
            'frequency': stats['frequency'],
            'mean_interval': stats['mean_interval'],
        }

        print("=" * 55)
        print("Calcium Analysis: {}".format(name))
        print("=" * 55)
        print("  Baseline frames    = {}".format(baseline_frames))
        print("  Baseline method    = {}".format(baseline_method))
        print("  Detection thresh   = {}".format(threshold))
        print("  Events detected    = {}".format(stats['n_events']))
        print("  Mean amplitude     = {:.4f}".format(stats['mean_amplitude']))
        print("  Mean duration      = {:.1f} frames".format(stats['mean_duration']))
        print("  Mean rise time     = {:.1f} frames".format(stats['mean_rise_time']))
        print("  Mean decay time    = {:.1f} frames".format(stats['mean_decay_time']))
        print("  Mean area          = {:.4f}".format(stats['mean_area']))
        print("  Frequency          = {:.4f} events/frame".format(stats['frequency']))
        print("=" * 55)

        window.metadata['calcium'] = results

        if g.m is not None:
            g.m.statusBar().showMessage(
                "Calcium: {} events, amp={:.3f}, dur={:.1f}f".format(
                    stats['n_events'], stats['mean_amplitude'], stats['mean_duration']))

        # Create dF/F window
        from flika.window import Window
        dff_window = Window(dff_stack.astype(np.float32),
                            name='dF/F: {}'.format(name))

        # Trace plot with events
        self._trace_widget = QtWidgets.QWidget()
        self._trace_widget.setWindowTitle('Calcium Trace: {}'.format(name))
        self._trace_widget.resize(700, 400)
        layout = QtWidgets.QVBoxLayout(self._trace_widget)

        pw = pg.PlotWidget()
        pw.setLabel('bottom', 'Frame')
        pw.setLabel('left', 'dF/F0')
        pw.setTitle('Calcium Trace: {}'.format(name))
        layout.addWidget(pw)

        frames = np.arange(n_frames)
        pw.plot(frames, dff_trace, pen=pg.mkPen('w', width=1), name='dF/F')

        if smooth:
            pw.plot(frames, analysis_trace,
                    pen=pg.mkPen('c', width=1.5), name='Smoothed')

        # Mark events
        for evt in events:
            region = pg.LinearRegionItem(
                [evt['start'], evt['end']],
                movable=False,
                brush=pg.mkBrush(255, 100, 100, 50))
            pw.addItem(region)
            # Peak marker
            pw.plot([evt['peak']], [evt['amplitude']],
                    pen=None, symbol='d', symbolSize=8,
                    symbolBrush=pg.mkBrush('r'))

        # Threshold line
        n_quiet = max(3, n_frames // 10)
        baseline_std = np.std(dff_trace[:n_quiet])
        thresh_line = pg.InfiniteLine(
            pos=threshold * baseline_std, angle=0,
            pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine),
            label='threshold={}x'.format(threshold),
            labelOpts={'position': 0.9, 'color': 'y'})
        pw.addItem(thresh_line)

        # Stats text
        stats_text = (
            "Events: {}\n"
            "Mean amp: {:.3f}\n"
            "Mean dur: {:.1f}f".format(
                stats['n_events'], stats['mean_amplitude'], stats['mean_duration']))
        text_item = pg.TextItem(stats_text, anchor=(1, 0), color='w')
        text_item.setPos(n_frames, np.max(dff_trace))
        pw.addItem(text_item)

        self._trace_widget.show()

        return dff_window

    def get_init_settings_dict(self):
        return {
            'baseline_frames': 10,
            'baseline_method': 'mean',
            'threshold': 2.0,
            'min_duration': 3,
            'smooth': False,
            'smooth_method': 'savgol',
            'use_roi': True,
        }


calcium_analysis = CalciumAnalysis()

logger.debug("Completed 'reading process/calcium.py'")
