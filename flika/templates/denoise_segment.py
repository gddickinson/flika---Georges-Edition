# Template: Denoise and Segment
# Description: Gaussian blur, threshold, and clean up binary mask
# Category: Segmentation

from flika.process import *
from flika.process.file_ import open_file

# --- Configuration ---
BLUR_SIGMA = 2.0       # Gaussian blur sigma
THRESHOLD = 0.5        # Threshold value
MIN_BLOB_SIZE = 20     # Minimum blob size to keep

# Step 1: Denoise with Gaussian blur
gaussian_blur(BLUR_SIGMA, keepSourceWindow=True)

# Step 2: Threshold
threshold(THRESHOLD, keepSourceWindow=True)

# Step 3: Clean up small blobs
remove_small_blobs(MIN_BLOB_SIZE, keepSourceWindow=True)

print("Denoising and segmentation complete.")
