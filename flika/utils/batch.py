"""Batch processing framework for flika.

Applies a sequence of processing operations to multiple files.

Example::

    from flika.utils.batch import BatchRunner
    from flika.process.filters import gaussian_blur
    from flika.process.math_ import subtract
    from flika.process.file_ import save_file

    runner = BatchRunner()
    runner.add_step(gaussian_blur, sigma=2)
    runner.add_step(subtract, value=100)
    runner.run(['/path/to/file1.tif', '/path/to/file2.tif'],
               output_dir='/path/to/output/')
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from flika.logger import logger


class BatchStep:
    """A single processing step in a batch pipeline."""

    __slots__ = ('func', 'kwargs', 'name')

    def __init__(self, func: Callable, **kwargs: Any):
        self.func = func
        self.kwargs = kwargs
        self.name = getattr(func, '__name__', str(func))

    def __repr__(self):
        args = ', '.join('{}={!r}'.format(k, v) for k, v in self.kwargs.items())
        return 'BatchStep({}, {})'.format(self.name, args)


class BatchRunner:
    """Chain multiple flika processing steps and run them on a list of files.

    Each file is opened -> all steps are applied in order -> the result is
    optionally saved to *output_dir*.
    """

    def __init__(self):
        self.steps: List[BatchStep] = []

    def add_step(self, func: Callable, **kwargs: Any) -> 'BatchRunner':
        """Append a processing step.

        *func* should be a callable flika process (e.g. ``gaussian_blur``).
        *kwargs* are passed to ``func(**kwargs, keepSourceWindow=False)``.
        """
        self.steps.append(BatchStep(func, **kwargs))
        return self  # allow chaining

    def clear(self):
        """Remove all steps."""
        self.steps.clear()

    def run(self, file_paths: List[str], output_dir: Optional[str] = None,
            output_format: str = '.tif') -> List[str]:
        """Execute the pipeline on each file in *file_paths*.

        Parameters
        ----------
        file_paths : list of str
            Input file paths.
        output_dir : str, optional
            Directory to save results.  If ``None``, results are left as
            open flika Windows without saving.
        output_format : str
            Extension for output files (default ``'.tif'``).

        Returns
        -------
        results : list of str
            Paths of saved files (empty strings for files that were not saved).
        """
        import flika.global_vars as g
        from flika.process.file_ import open_file, save_file

        results = []
        total = len(file_paths)

        for idx, fpath in enumerate(file_paths):
            msg = 'Batch [{}/{}]: {}'.format(idx + 1, total, os.path.basename(fpath))
            logger.info(msg)
            if g.m is not None:
                g.m.statusBar().showMessage(msg)

            try:
                open_file(fpath)
            except Exception as e:
                logger.error('Batch: failed to open {}: {}'.format(fpath, e))
                results.append('')
                continue

            for step in self.steps:
                try:
                    step.func(**step.kwargs, keepSourceWindow=False)
                except Exception as e:
                    logger.error('Batch: step {} failed on {}: {}'.format(step.name, fpath, e))
                    break

            if output_dir is not None and g.win is not None:
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(fpath))[0]
                out_path = os.path.join(output_dir, base + '_processed' + output_format)
                try:
                    save_file(out_path)
                    results.append(out_path)
                    logger.info('Batch: saved {}'.format(out_path))
                except Exception as e:
                    logger.error('Batch: failed to save {}: {}'.format(out_path, e))
                    results.append('')
            else:
                results.append('')

        if g.m is not None:
            g.m.statusBar().showMessage('Batch complete: processed {} files'.format(total))
        return results

    def __repr__(self):
        step_str = '\n  '.join(repr(s) for s in self.steps)
        return 'BatchRunner(\n  {}\n)'.format(step_str)
