# Template: Batch Processing
# Description: Process multiple files with a pipeline using BatchRunner
# Category: Automation

from flika.utils.batch import BatchRunner
from flika.process import *
from flika.process.file_ import open_file, save_file

# --- Configuration ---
INPUT_DIR = '/path/to/input/files'    # Change to your input directory
OUTPUT_DIR = '/path/to/output/files'  # Change to your output directory
FILE_PATTERN = '*.tif'

# Define your processing pipeline as a function
def my_pipeline(window):
    """Apply processing steps to a single window."""
    gaussian_blur(2, keepSourceWindow=False)
    threshold(0.5, keepSourceWindow=False)
    return window

# Run batch processing
runner = BatchRunner(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    pattern=FILE_PATTERN,
    pipeline=my_pipeline,
)
runner.run()

print("Batch processing complete.")
