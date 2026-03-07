# Template: Calcium dF/F
# Description: Compute delta-F/F0 from calcium imaging data
# Category: Calcium Imaging

from flika.process import *
from flika.process.file_ import open_file

# --- Configuration ---
BASELINE_FRAMES = 50   # Number of frames for baseline (F0)
BLUR_SIGMA = 1.0       # Gaussian blur sigma for noise reduction

# Step 1: Optional spatial smoothing
gaussian_blur(BLUR_SIGMA, keepSourceWindow=True)

# Step 2: Compute baseline (average of first N frames)
# ratio(first_frame, nFrames, ratio_type, black_level)
ratio(0, BASELINE_FRAMES, 'average', keepSourceWindow=True)

print("dF/F computation complete.")
print(f"Baseline computed from first {BASELINE_FRAMES} frames.")
