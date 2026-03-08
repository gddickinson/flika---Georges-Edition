To run the tests, open a terminal, navigate to flika, and type::

	pytest flika/tests/ --ignore=flika/tests/test_macros.py -q

Test files:

- ``test_processes.py`` -- Parametrized smoke tests for all process functions
  across multiple dtypes and shapes (880+ tests).
- ``test_correctness.py`` -- Correctness tests that validate actual output values
  for filters, binary ops, math, stacks, detection, segmentation, watershed,
  colocalization, color, deconvolution, stitching, bleach correction (100+ tests).
- ``test_io_registry.py`` -- Format registry and I/O handler tests.
- ``test_settings_batch.py`` -- Settings and batch processing tests.
- ``test_macro_recorder.py`` -- Macro recorder tests.
- ``test_undo.py`` -- Undo/redo stack tests.
- ``test_spt_*.py`` -- Single-particle tracking module tests.
- ``test_macros.py`` -- Legacy plugin manager tests (excluded by default due to
  pre-existing PyQt6 issues).

