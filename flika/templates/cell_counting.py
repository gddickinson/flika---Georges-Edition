# Template: Cell Counting
# Description: Threshold and count cells using analyze_particles
# Category: Analysis

from flika.process import *
from flika.process.file_ import open_file

# --- Configuration ---
THRESHOLD_VALUE = 0.5  # Adjust based on your image
MIN_SIZE = 50          # Minimum particle area in pixels

# Step 1: Apply Gaussian blur to reduce noise
gaussian_blur(2, keepSourceWindow=True)

# Step 2: Threshold to create binary mask
threshold(THRESHOLD_VALUE, keepSourceWindow=True)

# Step 3: Remove small objects
remove_small_blobs(MIN_SIZE, keepSourceWindow=True)

# Step 4: Count and measure particles
analyze_particles(min_area=MIN_SIZE, keepSourceWindow=True)

print("Cell counting complete. Check the results window.")
