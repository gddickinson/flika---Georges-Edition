# Flika Improvement Plan

## Block A: Performance Quick Wins (~200 lines changed)
1. Remove `self.EEEE` dead assignment (window.py:862)
2. Fix `.copy().astype()` → `.astype()` in flash removal (filters.py:1935,1955)
3. Fix chained `.astype()` in segmentation (~12 locations)
4. Cache scatter brush objects in window.py
5. Pre-allocate array in `getScatterPts()`

## Block B: Comprehensive Correctness Test Suite (~2500 lines, new file)
- `flika/tests/test_correctness.py` — tests validating actual output values for all process functions
- Covers: filters (16 untested), binary ops (13 untested), math (4 untested), stacks (8 untested), background_sub, detection, segmentation, alignment, kymograph, color, colocalization, watershed, ROI correctness

## Block C: Missing Features (~1200 lines, 2-3 new files)
1. Deconvolution (`flika/process/deconvolution.py`) — Richardson-Lucy, Wiener, PSF generation
2. Image Stitching (`flika/process/stitching.py`) — Phase cross-correlation, grid layout, blending
3. Color Space Conversions (extend `flika/process/color.py`) — RGB↔HSV, RGB↔LAB, grayscale
4. Standalone Bleach Correction (extend `flika/process/filters.py`) — exponential, histogram, ratio
5. Keyboard Shortcuts (extend `flika/app/application.py`) — Ctrl+G, Ctrl+T, Space, arrows, etc.
6. Batch Export (extend `flika/process/export.py`) — all windows, image sequences

## Block D: TraceFig & Threading Improvements (~150 lines changed)
1. Replace RedrawPartialThread busy-loop with signal-driven updates
2. Add QMutex to plugin manager for thread-safe loading
3. Debounce settings saves in settings_editor.py

## Implementation Order: A → B → C → D → Tests → Docs
