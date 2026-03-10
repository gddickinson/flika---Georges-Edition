# Template: Drift Correction
# Description: Correct XY drift using phase correlation
# Category: Pre-processing

from flika.process import *
from flika.process.file_ import open_file

# --- Configuration ---
# No parameters needed — motion_correction auto-detects drift

# Step 1: Run motion correction
motion_correction(keepSourceWindow=True)

print("Drift correction complete.")
print("Compare the corrected and original windows to verify.")
