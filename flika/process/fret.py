# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/fret.py'")
import numpy as np
from scipy import ndimage
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, SliderLabel

__all__ = ['fret_analysis']


# ---------------------------------------------------------------------------
# Pure analysis functions (no Qt dependencies)
# ---------------------------------------------------------------------------

def compute_apparent_fret(donor, acceptor, background_donor=0.0,
                          background_acceptor=0.0):
    """Compute pixel-wise apparent FRET efficiency.

    apparent_E = Ia / (Ia + Id)

    Parameters
    ----------
    donor : ndarray
        Donor channel image (2-D or 3-D stack).
    acceptor : ndarray
        Acceptor channel image (same shape as *donor*).
    background_donor : float
        Background value to subtract from donor.
    background_acceptor : float
        Background value to subtract from acceptor.

    Returns
    -------
    ndarray (float64)
        FRET efficiency image, same shape as input.  NaN where both
        channels are zero after background subtraction.
    """
    d = np.asarray(donor, dtype=np.float64) - background_donor
    a = np.asarray(acceptor, dtype=np.float64) - background_acceptor
    d = np.maximum(d, 0.0)
    a = np.maximum(a, 0.0)
    total = d + a
    with np.errstate(divide='ignore', invalid='ignore'):
        E = np.where(total > 0, a / total, np.nan)
    return E


def compute_corrected_fret(donor, acceptor, background_donor=0.0,
                           background_acceptor=0.0,
                           bleedthrough=0.0, direct_excitation=0.0,
                           gamma=1.0):
    """Compute corrected FRET efficiency (3-cube method).

    Applies bleedthrough, direct excitation, and gamma corrections::

        Fc = Ia - bleedthrough * Id - direct_excitation * Ia
        E  = Fc / (Fc + gamma * Id)

    Parameters
    ----------
    donor : ndarray
        Donor channel image.
    acceptor : ndarray
        Acceptor channel image.
    background_donor, background_acceptor : float
        Background values.
    bleedthrough : float
        Donor bleedthrough into acceptor channel (typically 0.0-0.5).
    direct_excitation : float
        Direct excitation of acceptor by donor excitation light.
    gamma : float
        Detection efficiency ratio (acceptor / donor).

    Returns
    -------
    ndarray (float64)
        Corrected FRET efficiency.
    """
    d = np.asarray(donor, dtype=np.float64) - background_donor
    a = np.asarray(acceptor, dtype=np.float64) - background_acceptor
    d = np.maximum(d, 0.0)
    a = np.maximum(a, 0.0)

    Fc = a - bleedthrough * d - direct_excitation * a
    denom = Fc + gamma * d
    with np.errstate(divide='ignore', invalid='ignore'):
        E = np.where(denom > 0, Fc / denom, np.nan)
    return E


def compute_stoichiometry(donor, acceptor, background_donor=0.0,
                          background_acceptor=0.0, gamma=1.0):
    """Compute FRET stoichiometry ratio.

    S = (Ia + Id) / (Ia + gamma * Id)

    Values near 1 indicate donor-only, near 0 indicate acceptor-only,
    and ~0.5 indicates 1:1 stoichiometry.

    Parameters
    ----------
    donor, acceptor : ndarray
    background_donor, background_acceptor : float
    gamma : float

    Returns
    -------
    ndarray (float64)
    """
    d = np.asarray(donor, dtype=np.float64) - background_donor
    a = np.asarray(acceptor, dtype=np.float64) - background_acceptor
    d = np.maximum(d, 0.0)
    a = np.maximum(a, 0.0)

    numer = a + d
    denom = a + gamma * d
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(denom > 0, numer / denom, np.nan)
    return S


def fret_histogram(E, bins=100, range_=(0, 1)):
    """Compute histogram of FRET efficiency values.

    Parameters
    ----------
    E : ndarray
        FRET efficiency image (may contain NaN).
    bins : int
    range_ : tuple of float

    Returns
    -------
    (counts, bin_edges) : tuple of ndarray
    """
    valid = E[np.isfinite(E)]
    if valid.size == 0:
        return np.zeros(bins), np.linspace(range_[0], range_[1], bins + 1)
    return np.histogram(valid, bins=bins, range=range_)


def compute_fret_stats(E, mask=None):
    """Compute summary statistics for a FRET efficiency image.

    Parameters
    ----------
    E : ndarray
        FRET efficiency image.
    mask : ndarray of bool, optional
        ROI mask.

    Returns
    -------
    dict with keys: mean_E, median_E, std_E, n_pixels.
    """
    if mask is not None:
        vals = E[mask & np.isfinite(E)]
    else:
        vals = E[np.isfinite(E)]

    if vals.size == 0:
        return {'mean_E': np.nan, 'median_E': np.nan,
                'std_E': np.nan, 'n_pixels': 0}

    return {
        'mean_E': float(np.mean(vals)),
        'median_E': float(np.median(vals)),
        'std_E': float(np.std(vals)),
        'n_pixels': int(vals.size),
    }


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class FRETAnalysis(BaseProcess):
    """fret_analysis(donor_window, acceptor_window, bleedthrough=0.0, direct_excitation=0.0, gamma=1.0, use_roi=False, keepSourceWindow=True)

    FRET (Forster Resonance Energy Transfer) analysis.

    Computes apparent or corrected FRET efficiency from donor and
    acceptor channel images.  Creates a FRET efficiency map, histogram,
    and summary statistics.

    Parameters:
        donor_window (Window): Donor channel window.
        acceptor_window (Window): Acceptor channel window.
        bleedthrough (float): Donor bleedthrough coefficient (0-1).
        direct_excitation (float): Direct acceptor excitation coefficient (0-1).
        gamma (float): Detection efficiency ratio.
        use_roi (bool): Restrict analysis to active ROI.
    Returns:
        Window -- FRET efficiency image.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        donor_window = WindowSelector()
        acceptor_window = WindowSelector()
        bleedthrough = SliderLabel(decimals=3)
        bleedthrough.setRange(0.0, 1.0)
        bleedthrough.setValue(0.0)
        direct_excitation = SliderLabel(decimals=3)
        direct_excitation.setRange(0.0, 1.0)
        direct_excitation.setValue(0.0)
        gamma = SliderLabel(decimals=3)
        gamma.setRange(0.01, 10.0)
        gamma.setValue(1.0)
        use_roi = CheckBox()
        use_roi.setChecked(False)
        self.items.append({'name': 'donor_window', 'string': 'Donor Channel', 'object': donor_window})
        self.items.append({'name': 'acceptor_window', 'string': 'Acceptor Channel', 'object': acceptor_window})
        self.items.append({'name': 'bleedthrough', 'string': 'Bleedthrough', 'object': bleedthrough})
        self.items.append({'name': 'direct_excitation', 'string': 'Direct Excitation', 'object': direct_excitation})
        self.items.append({'name': 'gamma', 'string': 'Gamma', 'object': gamma})
        self.items.append({'name': 'use_roi', 'string': 'Use ROI', 'object': use_roi})
        super().gui()

    def __call__(self, donor_window, acceptor_window, bleedthrough=0.0,
                 direct_excitation=0.0, gamma=1.0, use_roi=False,
                 keepSourceWindow=True):
        if donor_window is None or acceptor_window is None:
            raise MissingWindowError("Both donor and acceptor windows must be selected.")

        donor = donor_window.image.astype(np.float64)
        acceptor = acceptor_window.image.astype(np.float64)

        # Handle stacks: use current frame for 3-D
        if donor.ndim == 3:
            donor_2d = donor[donor_window.currentIndex]
        else:
            donor_2d = donor
        if acceptor.ndim == 3:
            acceptor_2d = acceptor[acceptor_window.currentIndex]
        else:
            acceptor_2d = acceptor

        if donor_2d.shape != acceptor_2d.shape:
            g.alert("Donor and acceptor images must have the same dimensions. "
                    f"Got {donor_2d.shape} and {acceptor_2d.shape}.")
            return

        # Build ROI mask
        mask = None
        if use_roi:
            if hasattr(donor_window, 'currentROI') and donor_window.currentROI is not None:
                roi = donor_window.currentROI
                yy, xx = roi.getMask()
                if yy.size > 0:
                    mask = np.zeros(donor_2d.shape[:2], dtype=bool)
                    valid = (
                        (yy >= 0) & (yy < donor_2d.shape[0]) &
                        (xx >= 0) & (xx < donor_2d.shape[1])
                    )
                    mask[yy[valid], xx[valid]] = True

        # Compute corrected FRET
        corrected = bleedthrough > 0 or direct_excitation > 0 or gamma != 1.0
        if corrected:
            E = compute_corrected_fret(
                donor_2d, acceptor_2d,
                bleedthrough=bleedthrough,
                direct_excitation=direct_excitation,
                gamma=gamma)
        else:
            E = compute_apparent_fret(donor_2d, acceptor_2d)

        # Statistics
        stats = compute_fret_stats(E, mask)

        # Stoichiometry
        S = compute_stoichiometry(donor_2d, acceptor_2d, gamma=gamma)
        stoich_stats = compute_fret_stats(S, mask)

        # Results
        name_d = donor_window.name
        name_a = acceptor_window.name
        results = {
            'donor': name_d,
            'acceptor': name_a,
            'corrected': corrected,
            'bleedthrough': bleedthrough,
            'direct_excitation': direct_excitation,
            'gamma': gamma,
            **stats,
            'stoichiometry_mean': stoich_stats['mean_E'],
        }

        print("=" * 55)
        print(f"FRET Analysis: {name_d} / {name_a}")
        print("=" * 55)
        print(f"  Method             = {'corrected' if corrected else 'apparent'}")
        if corrected:
            print(f"  Bleedthrough       = {bleedthrough:.4f}")
            print(f"  Direct excitation  = {direct_excitation:.4f}")
            print(f"  Gamma              = {gamma:.4f}")
        print(f"  Mean FRET E        = {stats['mean_E']:.4f}")
        print(f"  Median FRET E      = {stats['median_E']:.4f}")
        print(f"  Std FRET E         = {stats['std_E']:.4f}")
        print(f"  N pixels           = {stats['n_pixels']}")
        print(f"  Mean Stoichiometry = {stoich_stats['mean_E']:.4f}")
        print("=" * 55)

        donor_window.metadata['fret'] = results

        g.status_msg(
            f"FRET: E={stats['mean_E']:.3f} +/- {stats['std_E']:.3f}, "
            f"n={stats['n_pixels']}"
        )

        # Create FRET efficiency image window
        from ..window import Window
        fret_window = Window(E, name=f'FRET E: {name_d}/{name_a}')

        # Create histogram
        self._hist_widget = QtWidgets.QWidget()
        self._hist_widget.setWindowTitle(f'FRET Histogram: {name_d}/{name_a}')
        self._hist_widget.resize(500, 350)
        layout = QtWidgets.QVBoxLayout(self._hist_widget)
        pw = pg.PlotWidget()
        pw.setLabel('bottom', 'FRET Efficiency')
        pw.setLabel('left', 'Count')
        pw.setTitle('FRET Efficiency Distribution')
        layout.addWidget(pw)

        counts, edges = fret_histogram(E)
        centers = (edges[:-1] + edges[1:]) / 2
        pw.plot(centers, counts, stepMode=False,
                fillLevel=0, fillOutline=True,
                brush=pg.mkBrush(100, 200, 100, 120),
                pen=pg.mkPen('g', width=1))

        # Mean line
        if np.isfinite(stats['mean_E']):
            mean_line = pg.InfiniteLine(
                pos=stats['mean_E'], angle=90,
                pen=pg.mkPen('r', width=2),
                label=f"mean={stats['mean_E']:.3f}",
                labelOpts={'position': 0.9, 'color': 'r'})
            pw.addItem(mean_line)

        self._hist_widget.show()

        return fret_window

    def get_init_settings_dict(self):
        return {
            'bleedthrough': 0.0,
            'direct_excitation': 0.0,
            'gamma': 1.0,
            'use_roi': False,
        }


fret_analysis = FRETAnalysis()

logger.debug("Completed 'reading process/fret.py'")
