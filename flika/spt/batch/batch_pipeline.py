"""Multi-file SPT batch analysis pipeline.

Orchestrates the three phases of single-particle tracking analysis:

1. **Detection** -- locate particles in each frame.
2. **Linking** -- connect localisations across frames into tracks.
3. **Analysis** -- compute features, classify motion, and perform
   directional autocorrelation.

The pipeline can be run on individual flika :class:`~flika.window.Window`
objects or on lists of TIFF files for batch processing.
"""
import os
import time
import numpy as np
import pandas as pd
from ...logger import logger


class SPTParams:
    """Container for all SPT pipeline parameters.

    Provides sensible defaults for typical membrane protein tracking
    experiments (TIRF, 108 nm/px, 20 Hz).

    Args:
        detector: Detection method -- ``'utrack'`` or ``'thunderstorm'``.
        linker: Linking method -- ``'greedy'``, ``'utrack_lap'``, or
            ``'trackpy'``.
        max_distance: Maximum linking distance in pixels.
        max_gap: Maximum frame gap for gap closing.
        min_track_length: Minimum number of localisations in a track.
        psf_sigma: Expected PSF width in pixels.
        alpha: Detection significance threshold (u-track style).
        pixel_size: Nanometres per pixel (for physical unit conversion).
        frame_interval: Seconds between frames.
        enable_features: Compute per-track feature vectors.
        enable_classification: Classify tracks (requires trained model).
        enable_autocorrelation: Compute directional autocorrelation.
        classification_model_path: Path to a saved SPTClassifier model.
        thunderstorm_filter: Filter type for ThunderSTORM detector
            (``'wavelet'``, ``'gaussian'``, ``'dog'``, ``'lowered_gaussian'``).
        thunderstorm_fitter: Fitter type for ThunderSTORM detector
            (``'gaussian_lsq'``, ``'gaussian_mle'``, ``'radial_symmetry'``,
            ``'phasor'``, ``'centroid'``).
        threshold: Detection threshold in noise standard deviations.
        autocorrelation_intervals: Number of lag intervals for
            autocorrelation analysis.
        autocorrelation_min_length: Minimum track length for
            autocorrelation.
    """

    def __init__(self, detector='utrack', linker='greedy',
                 max_distance=5.0, max_gap=1, min_track_length=3,
                 psf_sigma=1.5, alpha=0.05,
                 pixel_size=108.0, frame_interval=0.05,
                 enable_features=True, enable_classification=False,
                 enable_autocorrelation=False,
                 classification_model_path=None,
                 thunderstorm_filter='wavelet',
                 thunderstorm_fitter='gaussian_lsq',
                 threshold=1.5,
                 autocorrelation_intervals=20,
                 autocorrelation_min_length=10):
        self.detector = detector
        self.linker = linker
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        self.psf_sigma = psf_sigma
        self.alpha = alpha
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.enable_features = enable_features
        self.enable_classification = enable_classification
        self.enable_autocorrelation = enable_autocorrelation
        self.classification_model_path = classification_model_path
        self.thunderstorm_filter = thunderstorm_filter
        self.thunderstorm_fitter = thunderstorm_fitter
        self.threshold = threshold
        self.autocorrelation_intervals = autocorrelation_intervals
        self.autocorrelation_min_length = autocorrelation_min_length

    def to_dict(self):
        """Serialise parameters to a plain dict."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d):
        """Create an SPTParams instance from a dict."""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__init__.__code__.co_varnames})

    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in self.to_dict().items())
        return f'SPTParams({items})'


class SPTBatchPipeline:
    """Multi-file SPT analysis pipeline.

    Runs detection, linking, and (optionally) feature computation,
    classification, and autocorrelation on one or more image stacks.

    Args:
        params: An :class:`SPTParams` instance.  Defaults are used if
            ``None``.
    """

    def __init__(self, params=None):
        self.params = params or SPTParams()
        self._classifier = None  # lazily loaded

    # ------------------------------------------------------------------
    # Internal: detector factory
    # ------------------------------------------------------------------

    def _make_detector(self):
        """Instantiate the detector specified by params."""
        p = self.params
        if p.detector == 'thunderstorm':
            from ..detection.thunderstorm import ThunderSTORMDetector
            roi = max(3, int(p.psf_sigma * 3) | 1)
            # Build filter_params based on filter type
            filt = p.thunderstorm_filter
            if filt in ('gaussian', 'lowered_gaussian'):
                f_params = {'sigma': p.psf_sigma}
            elif filt == 'dog':
                f_params = {'sigma1': p.psf_sigma,
                            'sigma2': p.psf_sigma * 1.6}
            elif filt == 'wavelet':
                f_params = {'scale': max(2, int(p.psf_sigma + 0.5))}
            else:
                f_params = {}
            return ThunderSTORMDetector(
                filter_type=filt,
                fitter_type=p.thunderstorm_fitter,
                threshold=p.threshold,
                roi_size=roi,
                filter_params=f_params,
                fitter_params={'initial_sigma': p.psf_sigma},
            )
        else:  # 'utrack' (default)
            from ..detection.utrack_detector import UTrackDetector
            return UTrackDetector(
                psf_sigma=p.psf_sigma,
                alpha=p.alpha,
            )

    # ------------------------------------------------------------------
    # Internal: linker dispatch
    # ------------------------------------------------------------------

    def _link(self, locs):
        """Link localisations into tracks using the configured linker.

        Args:
            locs: (N, 3+) array ``[frame, x, y, ...]``.

        Returns:
            ``(tracks, stats)`` tuple.
        """
        p = self.params

        if p.linker == 'trackpy':
            from ..linking.trackpy_linker import link_with_trackpy
            return link_with_trackpy(
                locs, search_range=p.max_distance, memory=p.max_gap,
                min_track_length=p.min_track_length)

        elif p.linker == 'utrack_lap':
            from ..linking.utrack_linker import UTrackLinker
            linker = UTrackLinker(
                max_distance=p.max_distance, max_gap=p.max_gap,
                min_track_length=p.min_track_length)
            return linker.link(locs)

        else:  # 'greedy' (default)
            from ..linking.greedy_linker import link_particles
            return link_particles(
                locs, max_distance=p.max_distance, max_gap=p.max_gap,
                min_track_length=p.min_track_length)

    # ------------------------------------------------------------------
    # Internal: detection
    # ------------------------------------------------------------------

    def _detect(self, data, callback=None):
        """Detect particles in image data.

        Args:
            data: 2D or 3D numpy array (single frame or stack).
            callback: Optional progress callback ``callback(frame_idx)``.

        Returns:
            Localisation array:
                - (N, 4) ``[frame, x, y, intensity]`` for utrack detector.
                - (N, 8) ``[frame, x, y, intensity, sigma_x, sigma_y,
                  background, uncertainty]`` for ThunderSTORM detector.
        """
        detector = self._make_detector()

        if data.ndim == 2:
            data = data[np.newaxis]

        return detector.detect_stack(data, callback=callback)

    # ------------------------------------------------------------------
    # Internal: features
    # ------------------------------------------------------------------

    def _compute_features(self, locs, tracks):
        """Compute per-track feature vectors.

        Args:
            locs: Localisation array.
            tracks: List of track index lists.

        Returns:
            pandas DataFrame with one row per track.
        """
        from ..features.feature_calculator import FeatureCalculator
        calc = FeatureCalculator(
            pixel_size=self.params.pixel_size,
            frame_interval=self.params.frame_interval)
        return calc.compute_all(locs, tracks)

    # ------------------------------------------------------------------
    # Internal: classification
    # ------------------------------------------------------------------

    def _classify(self, features_df):
        """Classify tracks using a trained SVM model.

        Args:
            features_df: DataFrame of per-track features.

        Returns:
            1D array of integer labels, or ``None`` if classification
            is not configured or fails.
        """
        if not self.params.enable_classification:
            return None

        model_path = self.params.classification_model_path
        if model_path is None:
            logger.warning("Classification enabled but no model path set")
            return None

        if not os.path.isfile(model_path):
            logger.warning("Classification model not found: %s", model_path)
            return None

        try:
            if self._classifier is None:
                from ..classification.svm_classifier import SPTClassifier
                self._classifier = SPTClassifier()
                self._classifier.load(model_path)

            labels = self._classifier.predict(features_df)
            return labels
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal: autocorrelation
    # ------------------------------------------------------------------

    def _autocorrelation(self, tracks_dict):
        """Compute directional autocorrelation.

        Args:
            tracks_dict: ``{track_id: (N, 3) array [frame, x, y]}``.

        Returns:
            Autocorrelation result dict, or ``None`` if disabled.
        """
        if not self.params.enable_autocorrelation:
            return None

        try:
            from ..features.autocorrelation import AutocorrelationAnalyzer
            analyzer = AutocorrelationAnalyzer(
                n_intervals=self.params.autocorrelation_intervals,
                min_track_length=self.params.autocorrelation_min_length)
            return analyzer.compute(tracks_dict)
        except Exception as exc:
            logger.error("Autocorrelation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API: single window
    # ------------------------------------------------------------------

    def run_on_window(self, window, callback=None):
        """Run the full pipeline on a loaded flika Window.

        Results are stored in ``window.metadata['spt']`` and also
        returned directly.

        Args:
            window: A flika :class:`~flika.window.Window` object with
                a loaded image stack.
            callback: Optional callable ``callback(phase, progress, msg)``
                where *phase* is ``'detect'``, ``'link'``, or
                ``'analyze'``, *progress* is 0-100, and *msg* is a
                status string.

        Returns:
            dict with keys:

            - **localizations**: (M, K) array of localisations.
            - **tracks**: list of track index lists.
            - **link_stats**: dict of linking statistics.
            - **features**: DataFrame of per-track features (or None).
            - **classification**: array of labels (or None).
            - **autocorrelation**: autocorrelation result dict (or None).
            - **params**: parameter dict.
            - **elapsed**: total processing time in seconds.
        """
        t0 = time.time()
        data = np.asarray(window.image, dtype=np.float64)

        if data.ndim == 2:
            data = data[np.newaxis]

        def _det_cb(frame_idx):
            if callback:
                pct = int(100 * frame_idx / max(data.shape[0], 1))
                callback('detect', pct,
                         f"Detecting frame {frame_idx}/{data.shape[0]}")

        # Phase 1: Detection
        logger.info("SPT pipeline: detecting in %s", window.name if hasattr(window, 'name') else 'window')
        locs = self._detect(data, callback=_det_cb)
        if callback:
            callback('detect', 100, f"Detection complete: {len(locs)} localisations")

        # Phase 2: Linking
        if callback:
            callback('link', 0, "Linking localisations...")
        tracks, link_stats = self._link(locs)
        if callback:
            callback('link', 100, f"Linking complete: {link_stats.get('num_tracks', 0)} tracks")

        # Phase 3: Analysis
        features_df = None
        classification = None
        autocorr = None

        if self.params.enable_features and tracks:
            if callback:
                callback('analyze', 0, "Computing features...")
            features_df = self._compute_features(locs, tracks)

            if self.params.enable_classification and features_df is not None:
                if callback:
                    callback('analyze', 50, "Classifying tracks...")
                classification = self._classify(features_df)

        if self.params.enable_autocorrelation and tracks:
            if callback:
                callback('analyze', 75, "Computing autocorrelation...")
            from ..linking.greedy_linker import tracks_to_dict
            td = tracks_to_dict(locs, tracks)
            autocorr = self._autocorrelation(td)

        if callback:
            callback('analyze', 100, "Analysis complete")

        elapsed = time.time() - t0

        result = {
            'localizations': locs,
            'tracks': tracks,
            'link_stats': link_stats,
            'features': features_df,
            'classification': classification,
            'autocorrelation': autocorr,
            'params': self.params.to_dict(),
            'elapsed': elapsed,
        }

        # Store in window metadata
        if not hasattr(window, 'metadata'):
            window.metadata = {}
        window.metadata['spt'] = result

        logger.info("SPT pipeline complete on window: %d locs, %d tracks, "
                     "%.1f s", len(locs), len(tracks), elapsed)
        return result

    # ------------------------------------------------------------------
    # Public API: single file
    # ------------------------------------------------------------------

    def run_on_file(self, file_path, output_dir=None):
        """Run the pipeline on a single TIFF file.

        Args:
            file_path: Path to a TIFF stack.
            output_dir: If provided, results are saved as CSV files
                in this directory.

        Returns:
            dict with the same structure as :meth:`run_on_window` plus
            an additional *file_path* key.
        """
        t0 = time.time()
        file_path = os.path.abspath(file_path)
        basename = os.path.splitext(os.path.basename(file_path))[0]

        logger.info("SPT pipeline: processing %s", file_path)

        # Load image data
        try:
            import tifffile
            data = tifffile.imread(file_path).astype(np.float64)
        except ImportError:
            raise ImportError(
                "tifffile is required for batch file processing. "
                "Install it with: pip install tifffile")
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)
            return {
                'file_path': file_path,
                'error': str(exc),
                'localizations': np.empty((0, 4)),
                'tracks': [],
                'link_stats': {},
                'features': None,
                'classification': None,
                'autocorrelation': None,
                'params': self.params.to_dict(),
                'elapsed': time.time() - t0,
            }

        if data.ndim == 2:
            data = data[np.newaxis]

        # Phase 1: Detection
        locs = self._detect(data)

        # Phase 2: Linking
        tracks, link_stats = self._link(locs)

        # Phase 3: Analysis
        features_df = None
        classification = None
        autocorr = None

        if self.params.enable_features and tracks:
            features_df = self._compute_features(locs, tracks)

            if self.params.enable_classification and features_df is not None:
                classification = self._classify(features_df)
                if classification is not None and features_df is not None:
                    features_df = features_df.copy()
                    features_df['classification'] = classification

        if self.params.enable_autocorrelation and tracks:
            from ..linking.greedy_linker import tracks_to_dict
            td = tracks_to_dict(locs, tracks)
            autocorr = self._autocorrelation(td)

        elapsed = time.time() - t0

        # Save results if output_dir is specified
        if output_dir is not None:
            self._save_results(output_dir, basename, locs, tracks,
                               link_stats, features_df, autocorr)

        result = {
            'file_path': file_path,
            'localizations': locs,
            'tracks': tracks,
            'link_stats': link_stats,
            'features': features_df,
            'classification': classification,
            'autocorrelation': autocorr,
            'params': self.params.to_dict(),
            'elapsed': elapsed,
        }

        logger.info("SPT pipeline complete: %s -> %d locs, %d tracks, %.1f s",
                     basename, len(locs), len(tracks), elapsed)
        return result

    # ------------------------------------------------------------------
    # Public API: batch
    # ------------------------------------------------------------------

    def run_on_files(self, file_list, output_dir, callback=None):
        """Run the pipeline on multiple files.

        Args:
            file_list: List of paths to TIFF files.
            output_dir: Directory to save all results.
            callback: Optional callable
                ``callback(file_idx, total, filename, result)`` invoked
                after each file is processed.

        Returns:
            list of result dicts (one per file).
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        total = len(file_list)

        logger.info("SPT batch: processing %d files -> %s",
                     total, output_dir)

        for i, path in enumerate(file_list):
            fname = os.path.basename(path)
            logger.info("SPT batch [%d/%d]: %s", i + 1, total, fname)

            try:
                result = self.run_on_file(path, output_dir=output_dir)
            except Exception as exc:
                logger.error("SPT batch: failed on %s: %s", fname, exc)
                result = {
                    'file_path': path,
                    'error': str(exc),
                    'localizations': np.empty((0, 4)),
                    'tracks': [],
                    'link_stats': {},
                    'features': None,
                    'classification': None,
                    'autocorrelation': None,
                    'params': self.params.to_dict(),
                    'elapsed': 0.0,
                }

            results.append(result)

            if callback is not None:
                callback(i, total, fname, result)

        # Write batch summary
        self._write_batch_summary(output_dir, results)

        logger.info("SPT batch complete: %d files processed", total)
        return results

    # ------------------------------------------------------------------
    # Internal: result saving
    # ------------------------------------------------------------------

    def _save_results(self, output_dir, basename, locs, tracks,
                      link_stats, features_df, autocorr):
        """Save pipeline results to files."""
        os.makedirs(output_dir, exist_ok=True)

        from ..io.spt_formats import (write_localizations,
                                      write_features,
                                      write_analysis_metadata)
        from ..linking.greedy_linker import tracks_to_array

        # Localisations CSV
        locs_path = os.path.join(output_dir, f'{basename}_locs.csv')
        try:
            locs_with_tracks = tracks_to_array(locs, tracks)
            # Build DataFrame depending on number of columns
            n_cols = locs_with_tracks.shape[1] if len(locs_with_tracks) > 0 else 0
            if n_cols >= 7:
                df = pd.DataFrame(locs_with_tracks,
                                  columns=['frame', 'x', 'y', 'intensity',
                                           'sigma', 'uncertainty', 'track_id'])
            elif n_cols >= 5:
                df = pd.DataFrame(locs_with_tracks,
                                  columns=['frame', 'x', 'y', 'intensity',
                                           'track_id'])
            elif n_cols >= 4:
                df = pd.DataFrame(locs_with_tracks,
                                  columns=['frame', 'x', 'y', 'track_id'])
            else:
                df = pd.DataFrame(locs_with_tracks[:, :3],
                                  columns=['frame', 'x', 'y'])
            write_localizations(df, locs_path, format='flika')
        except Exception as exc:
            logger.error("Failed to save localisations: %s", exc)

        # Features CSV
        if features_df is not None and not features_df.empty:
            feat_path = os.path.join(output_dir, f'{basename}_features.csv')
            try:
                write_features(features_df, feat_path)
            except Exception as exc:
                logger.error("Failed to save features: %s", exc)

        # Metadata JSON
        meta_path = os.path.join(output_dir, f'{basename}_metadata.json')
        try:
            stats = dict(link_stats)
            stats['n_localizations'] = len(locs)
            if autocorr is not None:
                stats['autocorrelation_n_tracks'] = autocorr.get(
                    'n_tracks_used', 0)
            write_analysis_metadata(meta_path, self.params.to_dict(), stats)
        except Exception as exc:
            logger.error("Failed to save metadata: %s", exc)

    def _write_batch_summary(self, output_dir, results):
        """Write a batch summary CSV."""
        rows = []
        for r in results:
            row = {
                'file': os.path.basename(r.get('file_path', '')),
                'n_localizations': len(r.get('localizations', [])),
                'n_tracks': r.get('link_stats', {}).get('num_tracks', 0),
                'mean_track_length': r.get('link_stats', {}).get(
                    'mean_track_length', 0),
                'linking_efficiency': r.get('link_stats', {}).get(
                    'linking_efficiency', 0),
                'elapsed_s': r.get('elapsed', 0),
                'error': r.get('error', ''),
            }
            rows.append(row)

        summary_path = os.path.join(output_dir, 'batch_summary.csv')
        try:
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            logger.info("Batch summary written to %s", summary_path)
        except Exception as exc:
            logger.error("Failed to write batch summary: %s", exc)
