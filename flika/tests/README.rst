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
- ``test_ai.py`` -- Non-GUI AI module tests: annotations (CRUD, YOLO/COCO/VOC
  export/import), detection backend (image prep, NMS), feature extractor, PSF
  simulator, classifier backends, denoiser utilities (65 tests, standalone).
- ``test_ai_gui.py`` -- GUI tests for AI dialogs: pixel classifier, object
  detection, particle localizer, denoiser, SAM, model zoo, annotation overlay,
  paint overlay, menu integration, workers (60 tests, requires Qt app).
- ``test_spt_*.py`` -- Single-particle tracking module tests (detection, linking,
  features, I/O, pipeline).
- ``test_spt_particle_data.py`` -- ParticleData model tests: constructors,
  properties, accessors, mutation, I/O, round-trip (40+ tests, standalone).
- ``test_spt_classifier.py`` -- SVM classifier lifecycle: train, predict,
  save/load, feature extraction, edge cases (15+ tests, standalone).
- ``test_spt_trackpy.py`` -- Trackpy linker adapter: all 4 link types, memory,
  min_track_length, stats (15+ tests, standalone, requires trackpy).
- ``test_spt_io_extended.py`` -- Extended SPT I/O: format detection, JSON tracks,
  ThunderSTORM extras, validation errors, ParticleData wrappers (25+ tests).
- ``test_dynamics.py`` -- FRAP, FRET, calcium, spectral unmixing, morphometry
  pure analysis function tests (50+ tests, standalone).
- ``test_structures.py`` -- Structure detection tests: Frangi vesselness, skeletonize,
  medial axis, skeleton analysis, Hough lines/circles, corner detection, LBP,
  structure tensor (26 tests, requires Qt app).
- ``test_macros.py`` -- Legacy plugin manager tests (excluded by default due to
  pre-existing PyQt6 issues).

