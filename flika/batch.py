"""Batch processing framework for flika."""
import os
import numpy as np
from . import global_vars as g
from .logger import logger

__all__ = ['BatchProcessor']


class BatchProcessor:
    """Run a processing pipeline on multiple files.

    Example:
        def my_pipeline(input_path, output_dir):
            from flika.process.file_ import open_file
            from flika.process.filters import gaussian_blur
            w = open_file(input_path)
            gaussian_blur(sigma=2.0)
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            g.win.save(output_path)
            g.win.close()

        bp = BatchProcessor(my_pipeline, input_files, output_dir)
        bp.run()
    """

    def __init__(self, pipeline_fn, input_files=None, output_dir=None):
        self.pipeline_fn = pipeline_fn
        self.input_files = input_files or []
        self.output_dir = output_dir
        self.results = []
        self.errors = []

    def run(self, parallel=False, n_workers=None):
        """Run the pipeline on all input files.

        Parameters:
            parallel: if True, use multiprocessing (not recommended for GUI ops)
            n_workers: number of workers for parallel mode
        """
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        total = len(self.input_files)
        self.results = []
        self.errors = []

        for i, filepath in enumerate(self.input_files):
            logger.info(f'Batch processing [{i+1}/{total}]: {os.path.basename(filepath)}')
            if g.m is not None:
                g.m.statusBar().showMessage(f'Batch [{i+1}/{total}]: {os.path.basename(filepath)}')

            try:
                result = self.pipeline_fn(filepath, self.output_dir)
                self.results.append({'file': filepath, 'status': 'success', 'result': result})
            except Exception as e:
                logger.error(f'Batch error on {filepath}: {e}')
                self.errors.append({'file': filepath, 'error': str(e)})
                self.results.append({'file': filepath, 'status': 'error', 'error': str(e)})

        n_ok = sum(1 for r in self.results if r['status'] == 'success')
        n_err = len(self.errors)
        msg = f'Batch complete: {n_ok}/{total} succeeded'
        if n_err:
            msg += f', {n_err} errors'
        logger.info(msg)
        if g.m is not None:
            g.m.statusBar().showMessage(msg)

        return self.results

    def run_from_macro(self, macro_file):
        """Replay a recorded macro on each input file.

        The macro should use 'CURRENT_FILE' as a placeholder for the input path.

        Parameters:
            macro_file: path to a .py macro script
        """
        with open(macro_file, 'r') as f:
            macro_code = f.read()

        def macro_pipeline(filepath, output_dir):
            code = macro_code.replace('CURRENT_FILE', filepath)
            if output_dir:
                code = code.replace('OUTPUT_DIR', output_dir)
            exec(code, {'__builtins__': __builtins__})

        self.pipeline_fn = macro_pipeline
        return self.run()

    @staticmethod
    def collect_files(directory, extensions=None):
        """Collect files from a directory with given extensions.

        Parameters:
            directory: path to directory
            extensions: list of extensions like ['.tif', '.tiff', '.h5']
        Returns:
            sorted list of file paths
        """
        if extensions is None:
            extensions = ['.tif', '.tiff', '.h5', '.hdf5', '.npy', '.zarr']

        files = []
        for f in sorted(os.listdir(directory)):
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                files.append(os.path.join(directory, f))
        return files
