# Template: Time Series ROI Analysis
# Description: Extract ROI traces and export to CSV
# Category: Analysis

import numpy as np
from flika.process import *
from flika.process.file_ import open_file
from flika import global_vars as g

# --- Configuration ---
OUTPUT_CSV = 'roi_traces.csv'  # Output file path

# Prerequisites: Open an image and draw ROIs before running this template

if g.win is None:
    print("Error: No window selected. Open an image first.")
elif not g.win.rois:
    print("Error: No ROIs found. Draw ROIs on the image first.")
else:
    # Extract traces from all ROIs
    traces = []
    roi_names = []
    for roi in g.win.rois:
        trace = roi.getTrace()
        traces.append(trace)
        name = getattr(roi, 'name', f'ROI_{len(roi_names)}')
        roi_names.append(name)

    # Stack into array and save
    traces_array = np.column_stack(traces)
    header = ','.join(roi_names)
    np.savetxt(OUTPUT_CSV, traces_array, delimiter=',', header=header, comments='')

    print(f"Exported {len(traces)} ROI traces to {OUTPUT_CSV}")
    print(f"Shape: {traces_array.shape} (frames x ROIs)")
