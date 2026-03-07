"""QThread worker for background SPT batch processing.

Provides a :class:`BatchWorker` that runs an :class:`SPTBatchPipeline`
on a list of files in a background thread, emitting Qt signals for
progress updates, per-file completion, final results, and errors.

Usage::

    worker = BatchWorker(pipeline, file_list, output_dir)
    worker.sig_progress.connect(on_progress)
    worker.sig_file_done.connect(on_file_done)
    worker.sig_all_done.connect(on_all_done)
    worker.sig_error.connect(on_error)
    worker.start()
"""
import os
import traceback

from qtpy.QtCore import QThread, Signal
from ...logger import logger


class BatchWorker(QThread):
    """Background worker for SPT batch file processing.

    Wraps :meth:`SPTBatchPipeline.run_on_file` in a QThread, emitting
    progress signals suitable for connecting to a GUI progress bar or
    log widget.

    Signals:
        sig_progress(int, str):
            Emitted periodically with ``(percent_complete, message)``.
        sig_file_done(str, dict):
            Emitted after each file finishes with
            ``(file_path, result_dict)``.
        sig_all_done(list):
            Emitted once all files are processed with a list of all
            result dicts.
        sig_error(str):
            Emitted when a fatal error occurs (the worker stops).

    Args:
        pipeline: An :class:`~flika.spt.batch.batch_pipeline.SPTBatchPipeline`
            instance.
        file_list: List of file paths to process.
        output_dir: Directory in which to save per-file results.
        parent: Optional parent QObject.
    """

    sig_progress = Signal(int, str)
    sig_file_done = Signal(str, dict)
    sig_all_done = Signal(list)
    sig_error = Signal(str)

    def __init__(self, pipeline, file_list, output_dir, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.file_list = list(file_list)
        self.output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the batch run.

        The worker checks this flag between files.  The currently
        running file will complete before the worker stops.
        """
        self._cancelled = True
        logger.info("BatchWorker: cancellation requested")

    @property
    def is_cancelled(self):
        """Whether cancellation has been requested."""
        return self._cancelled

    def run(self):
        """Process all files (runs in the background thread)."""
        results = []
        total = len(self.file_list)

        if total == 0:
            self.sig_progress.emit(100, "No files to process")
            self.sig_all_done.emit(results)
            return

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("BatchWorker: starting batch of %d files -> %s",
                     total, self.output_dir)

        try:
            for i, path in enumerate(self.file_list):
                if self._cancelled:
                    logger.info("BatchWorker: cancelled at file %d/%d",
                                i + 1, total)
                    self.sig_progress.emit(
                        int(100 * i / total),
                        f"Cancelled after {i} of {total} files")
                    break

                filename = os.path.basename(path)
                pct = int(100 * i / total)
                self.sig_progress.emit(
                    pct, f"Processing {filename} ({i + 1}/{total})...")

                try:
                    result = self.pipeline.run_on_file(
                        path, output_dir=self.output_dir)
                except Exception as exc:
                    logger.error("BatchWorker: error on %s: %s",
                                 filename, exc)
                    result = {
                        'file_path': path,
                        'error': str(exc),
                        'localizations': [],
                        'tracks': [],
                        'link_stats': {},
                        'features': None,
                        'classification': None,
                        'autocorrelation': None,
                        'params': self.pipeline.params.to_dict(),
                        'elapsed': 0.0,
                    }

                results.append(result)
                self.sig_file_done.emit(path, result)

                n_tracks = result.get('link_stats', {}).get('num_tracks', 0)
                elapsed = result.get('elapsed', 0.0)
                logger.info("BatchWorker: %s done (%d tracks, %.1f s)",
                            filename, n_tracks, elapsed)

            # Write batch summary
            if results:
                self.pipeline._write_batch_summary(self.output_dir, results)

            self.sig_progress.emit(100, f"Complete: {len(results)} files")
            self.sig_all_done.emit(results)

            logger.info("BatchWorker: batch complete, %d files processed",
                        len(results))

        except Exception as exc:
            error_msg = f"Fatal batch error: {exc}\n{traceback.format_exc()}"
            logger.error("BatchWorker: %s", error_msg)
            self.sig_error.emit(error_msg)


class SingleFileWorker(QThread):
    """Background worker for processing a single file or window.

    Lighter-weight alternative to :class:`BatchWorker` for running the
    SPT pipeline on a single dataset without the batch overhead.

    Signals:
        sig_progress(int, str):
            Progress updates ``(percent, message)``.
        sig_done(dict):
            Emitted with the result dict on completion.
        sig_error(str):
            Emitted on fatal error.

    Args:
        pipeline: SPTBatchPipeline instance.
        data: 2D or 3D numpy array, or a flika Window object.
        parent: Optional parent QObject.
    """

    sig_progress = Signal(int, str)
    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, pipeline, data, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.data = data

    def run(self):
        """Run the pipeline on the provided data."""
        try:
            import numpy as np

            # Check if data is a Window-like object
            if hasattr(self.data, 'image'):
                def _cb(phase, pct, msg):
                    self.sig_progress.emit(pct, f"[{phase}] {msg}")

                result = self.pipeline.run_on_window(self.data, callback=_cb)
            else:
                # Raw numpy array
                arr = np.asarray(self.data, dtype=np.float64)
                if arr.ndim == 2:
                    arr = arr[np.newaxis]

                self.sig_progress.emit(0, "Detecting particles...")
                locs = self.pipeline._detect(arr)

                self.sig_progress.emit(40, "Linking tracks...")
                tracks, stats = self.pipeline._link(locs)

                features = None
                classification = None
                autocorr = None

                if self.pipeline.params.enable_features and tracks:
                    self.sig_progress.emit(60, "Computing features...")
                    features = self.pipeline._compute_features(locs, tracks)

                    if self.pipeline.params.enable_classification and features is not None:
                        self.sig_progress.emit(75, "Classifying...")
                        classification = self.pipeline._classify(features)

                if self.pipeline.params.enable_autocorrelation and tracks:
                    self.sig_progress.emit(85, "Autocorrelation...")
                    from ..linking.greedy_linker import tracks_to_dict
                    td = tracks_to_dict(locs, tracks)
                    autocorr = self.pipeline._autocorrelation(td)

                result = {
                    'localizations': locs,
                    'tracks': tracks,
                    'link_stats': stats,
                    'features': features,
                    'classification': classification,
                    'autocorrelation': autocorr,
                    'params': self.pipeline.params.to_dict(),
                }

            self.sig_progress.emit(100, "Complete")
            self.sig_done.emit(result)

        except Exception as exc:
            error_msg = f"Processing error: {exc}\n{traceback.format_exc()}"
            logger.error("SingleFileWorker: %s", error_msg)
            self.sig_error.emit(error_msg)
