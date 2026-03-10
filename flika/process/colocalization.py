# -*- coding: utf-8 -*-
"""Colocalization analysis: Pearson, Manders, Costes, Li ICQ."""
from flika.logger import logger
logger.debug("Started 'reading process/colocalization.py'")
import numpy as np
from scipy import stats
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
import flika.global_vars as g
from flika.utils.BaseProcess import BaseProcess
from flika.utils.custom_widgets import WindowSelector, CheckBox, MissingWindowError

__all__ = ['colocalization']


# ---------------------------------------------------------------------------
# Pure analysis functions (ROI-mask aware)
# ---------------------------------------------------------------------------

def pearson_correlation(ch1, ch2, mask=None):
    """Compute Pearson's correlation coefficient between two channels.

    Parameters
    ----------
    ch1, ch2 : 2-D numpy arrays of the same shape.
    mask : 2-D bool array (True = include pixel), optional.

    Returns
    -------
    float in [-1, 1].  Returns NaN when there are fewer than two valid
    pixels or when a channel has zero variance.
    """
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    if mask is not None:
        ch1 = ch1[mask]
        ch2 = ch2[mask]
    else:
        ch1 = ch1.ravel()
        ch2 = ch2.ravel()

    if ch1.size < 2:
        return np.nan

    cc = np.corrcoef(ch1, ch2)
    return float(cc[0, 1])


def manders_coefficients(ch1, ch2, thresh1, thresh2, mask=None):
    """Compute Manders colocalization coefficients M1 and M2.

    M1 = sum(ch1[ch2 > thresh2]) / sum(ch1)
        Fraction of ch1 intensity that colocalizes with ch2.
    M2 = sum(ch2[ch1 > thresh1]) / sum(ch2)
        Fraction of ch2 intensity that colocalizes with ch1.

    Parameters
    ----------
    ch1, ch2 : 2-D numpy arrays of the same shape.
    thresh1 : float -- intensity threshold for ch1.
    thresh2 : float -- intensity threshold for ch2.
    mask : 2-D bool array (True = include pixel), optional.

    Returns
    -------
    (M1, M2) : tuple of floats in [0, 1].
    """
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    if mask is not None:
        ch1 = ch1[mask]
        ch2 = ch2[mask]
    else:
        ch1 = ch1.ravel()
        ch2 = ch2.ravel()

    sum_ch1 = np.sum(ch1)
    sum_ch2 = np.sum(ch2)

    if sum_ch1 == 0.0:
        M1 = 0.0
    else:
        M1 = float(np.sum(ch1[ch2 > thresh2]) / sum_ch1)

    if sum_ch2 == 0.0:
        M2 = 0.0
    else:
        M2 = float(np.sum(ch2[ch1 > thresh1]) / sum_ch2)

    return (M1, M2)


def costes_threshold(ch1, ch2, mask=None):
    """Automatic threshold determination using the Costes method.

    A binary search is performed on ch1 intensities to find the threshold
    at which the Pearson correlation of pixels *above* the threshold
    approaches zero.  The corresponding ch2 threshold is derived from the
    linear regression of ch2 on ch1.

    Parameters
    ----------
    ch1, ch2 : 2-D numpy arrays of the same shape.
    mask : 2-D bool array (True = include pixel), optional.

    Returns
    -------
    (thresh1, thresh2, pearson_above, pearson_below) : tuple of floats.
    """
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    if mask is not None:
        ch1_flat = ch1[mask]
        ch2_flat = ch2[mask]
    else:
        ch1_flat = ch1.ravel()
        ch2_flat = ch2.ravel()

    if ch1_flat.size < 2:
        return (0.0, 0.0, np.nan, np.nan)

    # Linear regression: ch2 = slope * ch1 + intercept
    slope, intercept, _, _, _ = stats.linregress(ch1_flat, ch2_flat)

    lo = float(ch1_flat.min())
    hi = float(ch1_flat.max())

    # Binary search for threshold where Pearson of above-threshold pixels ~ 0
    max_iterations = 64
    best_thresh1 = lo
    best_r_above = 1.0

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        above = ch1_flat > mid
        n_above = np.sum(above)

        if n_above < 2:
            hi = mid
            continue

        r_above = np.corrcoef(ch1_flat[above], ch2_flat[above])[0, 1]
        if np.isnan(r_above):
            hi = mid
            continue

        if abs(r_above) < abs(best_r_above):
            best_r_above = r_above
            best_thresh1 = mid

        if r_above > 0:
            lo = mid
        else:
            hi = mid

        # Convergence check
        if (hi - lo) < 1e-6 * (ch1_flat.max() - ch1_flat.min() + 1e-12):
            break

    thresh1 = best_thresh1
    thresh2 = slope * thresh1 + intercept

    # Compute final Pearson values above and below the threshold
    above = ch1_flat > thresh1
    below = ~above

    if np.sum(above) >= 2:
        pearson_above = float(np.corrcoef(ch1_flat[above], ch2_flat[above])[0, 1])
    else:
        pearson_above = np.nan

    if np.sum(below) >= 2:
        pearson_below = float(np.corrcoef(ch1_flat[below], ch2_flat[below])[0, 1])
    else:
        pearson_below = np.nan

    return (float(thresh1), float(thresh2), pearson_above, pearson_below)


def li_icq(ch1, ch2, mask=None):
    """Compute Li's Intensity Correlation Quotient (ICQ).

    ICQ = (fraction of pixels where (ch1 - mean_ch1)*(ch2 - mean_ch2) > 0) - 0.5

    Parameters
    ----------
    ch1, ch2 : 2-D numpy arrays of the same shape.
    mask : 2-D bool array (True = include pixel), optional.

    Returns
    -------
    float in [-0.5, 0.5].  Returns NaN when there are no valid pixels.
    """
    ch1 = ch1.astype(np.float64)
    ch2 = ch2.astype(np.float64)
    if mask is not None:
        ch1 = ch1[mask]
        ch2 = ch2[mask]
    else:
        ch1 = ch1.ravel()
        ch2 = ch2.ravel()

    n = ch1.size
    if n == 0:
        return np.nan

    product = (ch1 - np.mean(ch1)) * (ch2 - np.mean(ch2))
    fraction_positive = np.sum(product > 0) / n
    return float(fraction_positive - 0.5)


# ---------------------------------------------------------------------------
# Scatter-plot widget
# ---------------------------------------------------------------------------

class ColocalizationScatterWidget(QtWidgets.QWidget):
    """Interactive scatter plot showing pixel intensity correlation between
    two channels, with regression line and colocalization statistics overlay.
    """

    MAX_POINTS = 50000

    def __init__(self, ch1_data, ch2_data, name1, name2, mask=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Colocalization: {} vs {}'.format(name1, name2))
        self.resize(550, 500)

        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self._build_plot(ch1_data, ch2_data, name1, name2, mask)
        self.show()

    def _build_plot(self, ch1_data, ch2_data, name1, name2, mask):
        ch1 = ch1_data.astype(np.float64)
        ch2 = ch2_data.astype(np.float64)

        if mask is not None:
            ch1_flat = ch1[mask]
            ch2_flat = ch2[mask]
        else:
            ch1_flat = ch1.ravel()
            ch2_flat = ch2.ravel()

        # Downsample for performance
        n = ch1_flat.size
        if n > self.MAX_POINTS:
            idx = np.random.default_rng(0).choice(n, self.MAX_POINTS, replace=False)
            ch1_plot = ch1_flat[idx]
            ch2_plot = ch2_flat[idx]
        else:
            ch1_plot = ch1_flat
            ch2_plot = ch2_flat

        pw = self.plot_widget
        pw.setLabel('bottom', name1)
        pw.setLabel('left', name2)
        pw.setTitle('{} vs {}'.format(name1, name2))

        # Scatter
        scatter = pg.ScatterPlotItem(
            ch1_plot, ch2_plot,
            pen=None,
            brush=pg.mkBrush(100, 100, 255, 40),
            size=3,
        )
        pw.addItem(scatter)

        # Regression line
        if ch1_flat.size >= 2:
            slope, intercept, _, _, _ = stats.linregress(ch1_flat, ch2_flat)
            x_range = np.array([ch1_flat.min(), ch1_flat.max()])
            y_fit = slope * x_range + intercept
            line = pg.PlotDataItem(x_range, y_fit, pen=pg.mkPen('r', width=2))
            pw.addItem(line)

        # Compute statistics using full (non-downsampled) data
        r = pearson_correlation(ch1_data, ch2_data, mask)
        t1, t2, _, _ = costes_threshold(ch1_data, ch2_data, mask)
        M1, M2 = manders_coefficients(ch1_data, ch2_data, t1, t2, mask)
        icq = li_icq(ch1_data, ch2_data, mask)

        # Text overlay
        stats_text = (
            "Pearson r = {:.4f}\n"
            "Manders M1 = {:.4f}\n"
            "Manders M2 = {:.4f}\n"
            "Li ICQ = {:.4f}".format(r, M1, M2, icq)
        )
        text_item = pg.TextItem(stats_text, anchor=(0, 0), color='w')
        text_item.setPos(ch1_flat.min(), ch2_flat.max())
        pw.addItem(text_item)


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class Colocalization(BaseProcess):
    """colocalization(window1, window2, use_roi=False, keepSourceWindow=True)

    Quantitative colocalization analysis between two image channels.

    Computes Pearson correlation, Manders coefficients (with automatic
    Costes thresholds), and Li's Intensity Correlation Quotient.  Results
    are printed to the console, shown on the status bar, stored in
    ``window1.metadata['colocalization']``, and displayed in an interactive
    scatter-plot widget.

    Parameters:
        window1 (Window): First channel window.
        window2 (Window): Second channel window.
        use_roi (bool): If True, restrict analysis to the active ROI mask.
    Returns:
        None
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        use_roi = CheckBox()
        use_roi.setChecked(False)
        self.items.append({'name': 'window1', 'string': 'Channel 1', 'object': window1})
        self.items.append({'name': 'window2', 'string': 'Channel 2', 'object': window2})
        self.items.append({'name': 'use_roi', 'string': 'Use ROI', 'object': use_roi})
        super().gui()

    def __call__(self, window1, window2, use_roi=False, keepSourceWindow=True):
        if window1 is None or window2 is None:
            raise MissingWindowError("Two windows must be selected for colocalization analysis.")

        # Extract 2-D image data (current frame if 3-D stack)
        img1 = window1.image
        if img1.ndim == 3:
            img1 = img1[window1.currentIndex]
        img2 = window2.image
        if img2.ndim == 3:
            img2 = img2[window2.currentIndex]

        if img1.shape != img2.shape:
            g.alert("Channel images must have the same dimensions. "
                    "Got {} and {}.".format(img1.shape, img2.shape))
            return

        # Build boolean mask from ROI if requested
        mask = None
        if use_roi:
            roi_source = window1
            if hasattr(roi_source, 'currentROI') and roi_source.currentROI is not None:
                roi = roi_source.currentROI
                yy, xx = roi.getMask()
                if yy.size > 0:
                    mask = np.zeros(img1.shape[:2], dtype=bool)
                    valid = (
                        (yy >= 0) & (yy < img1.shape[0]) &
                        (xx >= 0) & (xx < img1.shape[1])
                    )
                    mask[yy[valid], xx[valid]] = True
            else:
                if g.m is not None:
                    g.m.statusBar().showMessage('No ROI found; analysing full image.')

        # --- Compute metrics ---
        r = pearson_correlation(img1, img2, mask)
        t1, t2, r_above, r_below = costes_threshold(img1, img2, mask)
        M1, M2 = manders_coefficients(img1, img2, t1, t2, mask)
        icq = li_icq(img1, img2, mask)

        # --- Report results ---
        name1 = window1.name
        name2 = window2.name

        results = {
            'pearson_r': r,
            'costes_thresh1': t1,
            'costes_thresh2': t2,
            'costes_pearson_above': r_above,
            'costes_pearson_below': r_below,
            'manders_M1': M1,
            'manders_M2': M2,
            'li_icq': icq,
            'channel1': name1,
            'channel2': name2,
        }

        print("=" * 50)
        print("Colocalization Analysis: {} vs {}".format(name1, name2))
        print("=" * 50)
        print("  Pearson r          = {:.4f}".format(r))
        print("  Costes thresh ch1  = {:.4f}".format(t1))
        print("  Costes thresh ch2  = {:.4f}".format(t2))
        print("  Costes r (above)   = {:.4f}".format(r_above))
        print("  Costes r (below)   = {:.4f}".format(r_below))
        print("  Manders M1         = {:.4f}".format(M1))
        print("  Manders M2         = {:.4f}".format(M2))
        print("  Li ICQ             = {:.4f}".format(icq))
        print("=" * 50)

        # Store in metadata
        window1.metadata['colocalization'] = results

        # Status bar summary
        if g.m is not None:
            g.m.statusBar().showMessage(
                'Colocalization: Pearson r={:.3f}, M1={:.3f}, M2={:.3f}, ICQ={:.3f}'.format(
                    r, M1, M2, icq))

        # Open scatter widget
        self._scatter = ColocalizationScatterWidget(
            img1, img2, name1, name2, mask=mask
        )

        return None

    def get_init_settings_dict(self):
        return {'use_roi': False}


colocalization = Colocalization()

logger.debug("Completed 'reading process/colocalization.py'")
