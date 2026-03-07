# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/spt.py'")

import numpy as np
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, ComboBox, SliderLabel


__all__ = ['spt_analysis', 'detect_particles', 'link_particles_process']


class SPT_Analysis(BaseProcess):
    """spt_analysis(keepSourceWindow=True)

    Opens the SPT Analysis control panel for particle detection,
    linking, feature computation, and batch processing.
    """
    def gui(self):
        if g.win is None:
            g.alert("No window selected.")
            return
        self.__call__()

    def __call__(self, keepSourceWindow=True):
        # Lazy import to avoid circular deps
        from ..viewers.spt_control_panel import SPTControlPanel
        panel = SPTControlPanel.instance(g.m)
        from qtpy import QtCore
        g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
        panel.show()
        panel.raise_()
        g.status_msg('SPT Analysis panel opened.')

spt_analysis = SPT_Analysis()


class Detect_Particles(BaseProcess):
    """detect_particles(method, psf_sigma, alpha, min_intensity, filter_type, detector_type, fitter_type, threshold, fit_radius, initial_sigma, photons_per_adu, baseline, keepSourceWindow=True)

    Detect particles in the current window using statistical methods.
    Results stored in window.metadata['spt']['localizations'].

    Parameters
    ----------
    method : str
        Detection method. One of 'U-Track', 'ThunderSTORM', or 'AI Localizer'.
    psf_sigma : float
        Estimated PSF standard deviation in pixels (U-Track).
    alpha : float
        Significance level for statistical detection threshold (U-Track).
    min_intensity : float
        Minimum intensity for a valid detection (U-Track).
    filter_type : str
        ThunderSTORM image filter type.
    detector_type : str
        ThunderSTORM candidate detector type.
    fitter_type : str
        ThunderSTORM sub-pixel fitter type.
    threshold : str
        ThunderSTORM detection threshold expression.
    fit_radius : int
        ThunderSTORM fitting ROI half-size in pixels.
    initial_sigma : float
        ThunderSTORM initial PSF sigma estimate.
    photons_per_adu : float
        ThunderSTORM camera photons per ADU.
    baseline : float
        ThunderSTORM camera baseline offset.
    keepSourceWindow : bool
        Whether to keep the source window after processing.
    """
    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItems(['U-Track', 'ThunderSTORM', 'AI Localizer'])
        psf_sigma = SliderLabel(1)
        psf_sigma.setRange(0.5, 10.0)
        psf_sigma.setValue(1.5)
        alpha = SliderLabel(3)
        alpha.setRange(0.001, 0.5)
        alpha.setValue(0.05)
        min_intensity = SliderLabel(1)
        min_intensity.setRange(0, 10000)
        min_intensity.setValue(0)
        # ThunderSTORM parameters
        filter_type = ComboBox()
        filter_type.addItems(['wavelet', 'gaussian', 'dog', 'lowered_gaussian',
                              'median', 'box', 'none'])
        detector_type = ComboBox()
        detector_type.addItems(['local_max', 'nms', 'centroid', 'grid'])
        fitter_type = ComboBox()
        fitter_type.addItems(['gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
                              'elliptical_gaussian_mle', 'phasor',
                              'radial_symmetry', 'centroid', 'multi_emitter'])
        threshold = ComboBox()
        threshold.setEditable(True)
        threshold.addItems(['std(Wave.F1)', '2*std(Wave.F1)', '3*std(Wave.F1)'])
        fit_radius = SliderLabel(0)
        fit_radius.setRange(2, 10)
        fit_radius.setValue(3)
        initial_sigma = SliderLabel(2)
        initial_sigma.setRange(0.5, 5.0)
        initial_sigma.setValue(1.3)
        photons_per_adu = SliderLabel(2)
        photons_per_adu.setRange(0.01, 100)
        photons_per_adu.setValue(3.6)
        baseline = SliderLabel(1)
        baseline.setRange(0, 10000)
        baseline.setValue(100)

        self.items.append({'name': 'method', 'string': 'Detection Method', 'object': method})
        self.items.append({'name': 'psf_sigma', 'string': 'PSF Sigma (px)', 'object': psf_sigma})
        self.items.append({'name': 'alpha', 'string': 'Significance Level', 'object': alpha})
        self.items.append({'name': 'min_intensity', 'string': 'Min Intensity', 'object': min_intensity})
        self.items.append({'name': 'filter_type', 'string': 'TS Filter Type', 'object': filter_type})
        self.items.append({'name': 'detector_type', 'string': 'TS Detector Type', 'object': detector_type})
        self.items.append({'name': 'fitter_type', 'string': 'TS Fitter Type', 'object': fitter_type})
        self.items.append({'name': 'threshold', 'string': 'TS Threshold', 'object': threshold})
        self.items.append({'name': 'fit_radius', 'string': 'TS Fit Radius (px)', 'object': fit_radius})
        self.items.append({'name': 'initial_sigma', 'string': 'TS Initial Sigma', 'object': initial_sigma})
        self.items.append({'name': 'photons_per_adu', 'string': 'TS Photons/ADU', 'object': photons_per_adu})
        self.items.append({'name': 'baseline', 'string': 'TS Camera Baseline', 'object': baseline})
        super().gui()

    def __call__(self, method='U-Track', psf_sigma=1.5, alpha=0.05,
                 min_intensity=0, filter_type='wavelet',
                 detector_type='local_max', fitter_type='gaussian_lsq',
                 threshold='std(Wave.F1)', fit_radius=3,
                 initial_sigma=1.3, photons_per_adu=3.6, baseline=100,
                 keepSourceWindow=True):
        if g.win is None:
            raise MissingWindowError("No window selected.")

        w = g.win
        data = w.image

        g.status_msg(f'Detecting particles ({method})...')

        if method == 'U-Track':
            from ..spt.detection.utrack_detector import UTrackDetector
            detector = UTrackDetector(psf_sigma=psf_sigma, alpha=alpha,
                                      min_intensity=min_intensity)
            if data.ndim == 2:
                data = data[np.newaxis]
            locs = detector.detect_stack(data)
        elif method == 'ThunderSTORM':
            from ..spt.detection.thunderstorm import ThunderSTORMDetector
            roi_size = fit_radius * 2 + 1
            detector = ThunderSTORMDetector(
                filter_type=filter_type,
                detector_type=detector_type,
                fitter_type=fitter_type,
                threshold=threshold,
                roi_size=roi_size,
                camera_params={
                    'photons_per_adu': photons_per_adu,
                    'baseline': baseline,
                },
                fitter_params={
                    'initial_sigma': initial_sigma,
                },
            )
            if data.ndim == 2:
                data = data[np.newaxis]
            locs = detector.detect_stack(data)
        else:
            g.alert(f"Detection method '{method}' not implemented yet.")
            return

        # Normalize coordinates: detectors output x=col(dim2),
        # y=row(dim1) assuming row-major images, but flika stores
        # images as (dim1, dim2).  Swap so [:,1]=dim1, [:,2]=dim2.
        if len(locs) > 0:
            locs[:, 1], locs[:, 2] = locs[:, 2].copy(), locs[:, 1].copy()

        # Store results
        if 'spt' not in w.metadata:
            w.metadata['spt'] = {}
        w.metadata['spt']['localizations'] = locs

        # Build ParticleData
        from ..spt.particle_data import ParticleData
        pdata = ParticleData.from_numpy(locs)
        pdata._detection_params = {'method': method}
        w.metadata['spt']['particle_data'] = pdata

        n_detections = len(locs)
        n_frames = len(np.unique(locs[:, 0])) if n_detections > 0 else 0
        g.status_msg(
            f'Detection complete: {n_detections} particles in {n_frames} frames.')

        # Display as scatter points on window
        if n_detections > 0:
            self._display_detections(w, locs)

        return locs

    def _display_detections(self, window, locs):
        """Show detections as scatter points on the window.

        After coordinate normalization, locs[:,1]=dim1 (pyqtgraph x)
        and locs[:,2]=dim2 (pyqtgraph y).
        """
        from qtpy.QtGui import QColor
        # Clear existing scatter
        for frame_pts in window.scatterPoints:
            frame_pts.clear()

        color = QColor(0, 255, 0, 180)  # green
        for det in locs:
            frame = int(det[0])
            if 0 <= frame < len(window.scatterPoints):
                window.scatterPoints[frame].append([det[1], det[2], color, 5])

        # Refresh display
        if hasattr(window, 'updateindex'):
            window.updateindex()

detect_particles = Detect_Particles()


class Link_Particles_Process(BaseProcess):
    """link_particles_process(method, max_distance, max_gap, min_track_length, motion_model, num_tracking_rounds, velocity_persistence, merge_split, link_type, adaptive_stop, adaptive_step, keepSourceWindow=True)

    Link detected particles into tracks. Requires prior detection
    (results in window.metadata['spt']['localizations']).
    Results stored in window.metadata['spt']['tracks'].

    Parameters
    ----------
    method : str
        Linking method. One of 'Greedy', 'U-Track LAP', or 'Trackpy'.
    max_distance : float
        Maximum distance (in pixels) for linking particles between frames.
    max_gap : int
        Maximum number of frames a particle can disappear and reappear.
    min_track_length : int
        Minimum number of frames a track must span to be kept.
    motion_model : str
        U-Track LAP motion model: 'brownian', 'linear', 'confined', 'mixed'.
    num_tracking_rounds : int
        U-Track LAP forward-reverse-forward rounds.
    velocity_persistence : float
        U-Track LAP velocity persistence (0-1).
    merge_split : bool
        U-Track LAP merge/split detection.
    link_type : str
        Trackpy link type: 'standard', 'adaptive', 'velocityPredict',
        'adaptive + velocityPredict'.
    adaptive_stop : float
        Trackpy adaptive minimum search range (0 = auto).
    adaptive_step : float
        Trackpy adaptive reduction factor.
    keepSourceWindow : bool
        Whether to keep the source window after processing.
    """
    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItems(['Greedy', 'U-Track LAP', 'Trackpy'])
        max_distance = SliderLabel(1)
        max_distance.setRange(1, 50)
        max_distance.setValue(5)
        max_gap = SliderLabel(0)
        max_gap.setRange(0, 50)
        max_gap.setValue(1)
        min_length = SliderLabel(0)
        min_length.setRange(1, 100)
        min_length.setValue(3)
        # U-Track LAP params
        motion_model = ComboBox()
        motion_model.addItems(['brownian', 'linear', 'confined', 'mixed'])
        motion_model.setCurrentIndex(3)  # mixed
        num_rounds = SliderLabel(0)
        num_rounds.setRange(1, 5)
        num_rounds.setValue(3)
        vel_persist = SliderLabel(2)
        vel_persist.setRange(0, 1)
        vel_persist.setValue(0.8)
        merge_split = CheckBox()
        merge_split.setChecked(True)
        # Trackpy params
        link_type = ComboBox()
        link_type.addItems(['standard', 'adaptive', 'velocityPredict',
                            'adaptive + velocityPredict'])
        adaptive_stop = SliderLabel(1)
        adaptive_stop.setRange(0, 20)
        adaptive_stop.setValue(0)
        adaptive_step = SliderLabel(2)
        adaptive_step.setRange(0.5, 0.99)
        adaptive_step.setValue(0.95)

        self.items.append({'name': 'method', 'string': 'Linking Method', 'object': method})
        self.items.append({'name': 'max_distance', 'string': 'Max Distance (px)', 'object': max_distance})
        self.items.append({'name': 'max_gap', 'string': 'Max Gap (frames)', 'object': max_gap})
        self.items.append({'name': 'min_track_length', 'string': 'Min Track Length', 'object': min_length})
        self.items.append({'name': 'motion_model', 'string': 'Motion Model (U-Track)', 'object': motion_model})
        self.items.append({'name': 'num_tracking_rounds', 'string': 'Tracking Rounds (U-Track)', 'object': num_rounds})
        self.items.append({'name': 'velocity_persistence', 'string': 'Velocity Persistence (U-Track)', 'object': vel_persist})
        self.items.append({'name': 'merge_split', 'string': 'Merge/Split (U-Track)', 'object': merge_split})
        self.items.append({'name': 'link_type', 'string': 'Link Type (Trackpy)', 'object': link_type})
        self.items.append({'name': 'adaptive_stop', 'string': 'Adaptive Stop (Trackpy)', 'object': adaptive_stop})
        self.items.append({'name': 'adaptive_step', 'string': 'Adaptive Step (Trackpy)', 'object': adaptive_step})
        super().gui()

    def __call__(self, method='Greedy', max_distance=5, max_gap=1,
                 min_track_length=3, motion_model='mixed',
                 num_tracking_rounds=3, velocity_persistence=0.8,
                 merge_split=True, link_type='standard',
                 adaptive_stop=0, adaptive_step=0.95,
                 keepSourceWindow=True):
        if g.win is None:
            raise MissingWindowError("No window selected.")

        w = g.win
        spt_data = w.metadata.get('spt', {})
        locs = spt_data.get('localizations')

        if locs is None or len(locs) == 0:
            g.alert("No localizations found. Run Detect Particles first.")
            return

        g.status_msg(f'Linking particles ({method})...')

        if method == 'Greedy':
            from ..spt.linking.greedy_linker import link_particles, tracks_to_dict
            tracks, stats = link_particles(locs, max_distance=max_distance,
                                           max_gap=max_gap,
                                           min_track_length=min_track_length)
        elif method == 'U-Track LAP':
            from ..spt.linking.utrack_linker import UTrackLinker
            linker = UTrackLinker(max_distance=max_distance, max_gap=max_gap,
                                  min_track_length=min_track_length,
                                  motion_model=motion_model,
                                  num_tracking_rounds=num_tracking_rounds)
            tracks, stats = linker.link(locs)
        elif method == 'Trackpy':
            try:
                from ..spt.linking.trackpy_linker import link_with_trackpy
                a_stop = adaptive_stop if adaptive_stop > 0 else None
                tracks, stats = link_with_trackpy(
                    locs, search_range=max_distance,
                    memory=max_gap, min_track_length=min_track_length,
                    link_type=link_type, adaptive_stop=a_stop,
                    adaptive_step=adaptive_step)
            except ImportError:
                g.alert("Trackpy is not installed. Use: pip install trackpy")
                return
        else:
            g.alert(f"Unknown linking method: {method}")
            return

        # Store results
        w.metadata['spt']['tracks'] = tracks
        w.metadata['spt']['linking_stats'] = stats

        # Convert to dict for TrackOverlay compatibility
        from ..spt.linking.greedy_linker import tracks_to_dict
        tracks_dict = tracks_to_dict(locs, tracks)
        w.metadata['spt']['tracks_dict'] = tracks_dict

        # Update ParticleData if it exists
        pdata = w.metadata['spt'].get('particle_data')
        if pdata is not None:
            pdata.set_tracks(tracks, linking_params={'method': method})

        n_tracks = stats.get('num_tracks', len(tracks))
        g.status_msg(
            f'Linking complete: {n_tracks} tracks '
            f'(mean length {stats.get("mean_track_length", 0):.1f}).')

        # Show tracks on window using TrackOverlay
        self._display_tracks(w, tracks_dict)

        return tracks, stats

    def _display_tracks(self, window, tracks_dict):
        """Display tracks using TrackOverlay."""
        try:
            from ..viewers.track_overlay import TrackOverlay, show_track_overlay
            overlay, panel = show_track_overlay(window)
            overlay.load_tracks_from_dict(tracks_dict)
        except Exception as e:
            logger.warning(f"Could not display tracks: {e}")

link_particles_process = Link_Particles_Process()


logger.debug("Completed 'reading process/spt.py'")
