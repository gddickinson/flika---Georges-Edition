# Interoperability

Flika is designed to work alongside other bioimage analysis tools. This document covers
bridges to napari and ImageJ, export formats for data sharing, and batch processing.

## napari Bridge

[napari](https://napari.org/) is a multi-dimensional image viewer for Python. Flika can
send images to napari and import them back.

### Sending to napari

**File > Interop > Send to napari**

Sends the current window's image as a napari Image layer. If napari is not already running,
a new viewer is launched.

```python
# From the console
g.m._send_to_napari()
```

### Importing from napari

**File > Interop > Import from napari**

Imports the active napari layer as a new flika window. Supports Image, Labels, and
Points layers.

```python
g.m._import_from_napari()
```

### Requirements

```bash
pip install napari[all]
```

## ImageJ Bridge

Flika can exchange images with ImageJ/Fiji through the ImageJ-Python bridge.

### Sending to ImageJ

**File > Interop > ImageJ > Send to ImageJ**

Opens the current image in ImageJ. Requires ImageJ to be installed and accessible.

### Importing from ImageJ

**File > Interop > ImageJ > Import from ImageJ**

Pulls the currently active image from ImageJ into a new flika window.

### Setup

The ImageJ bridge uses `pyimagej` for communication:

```bash
pip install pyimagej
```

You may need to configure the path to your Fiji/ImageJ installation.

## OME-TIFF Export

**File > Interop > Export OME-TIFF**

Exports the current window as an OME-TIFF file with standardized metadata. OME-TIFF is
the recommended format for sharing microscopy data because:

- It embeds metadata (pixel size, channel info, timestamps) in OME-XML
- It is readable by virtually all bioimage analysis tools
- It supports multi-channel, multi-Z, time-series data

```python
g.m._export_ome_tiff()
```

## OME-Zarr Export

**File > Interop > Export OME-Zarr**

Exports as OME-Zarr (NGFF), a cloud-optimized format with:

- Chunked storage for efficient partial reads
- Multiple resolution levels (pyramidal)
- Standardized NGFF metadata
- Ideal for very large datasets and remote access

```python
g.m._export_ome_zarr()
```

### Requirements

```bash
pip install zarr ome-zarr
```

## Array Protocol

Flika windows expose their data as standard numpy arrays, making it trivial to exchange
data with any Python library:

```python
import numpy as np
from flika.window import Window

# Export to any tool that accepts numpy arrays
arr = g.win.image
# Use with scikit-image, scipy, etc.
from skimage.filters import gaussian
result = gaussian(arr[0], sigma=2)

# Import from any source
data = np.load('external_data.npy')
win = Window(data, name='imported')
```

### pandas Integration

```python
import pandas as pd

# Export ROI traces to a DataFrame
traces = {}
for i, roi in enumerate(g.win.rois):
    traces[f'ROI_{i}'] = roi.getTrace()
df = pd.DataFrame(traces)
df.to_csv('roi_traces.csv', index=False)
```

## Batch Processing

**File > Batch Process...** opens a dialog for applying a pipeline to multiple files.

### Workflow

1. **Select Input**: Choose a folder of image files
2. **Define Pipeline**: Select the sequence of operations to apply
3. **Configure Output**: Set the output directory and format
4. **Run**: Processing runs on each file with a progress indicator

### Programmatic Batch Processing

```python
import os
from flika.process.file_ import open_file
from flika.process.filters import gaussian_blur
from flika.utils.misc import save_file_gui
import tifffile

input_dir = '/path/to/input/'
output_dir = '/path/to/output/'

for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        win = open_file(os.path.join(input_dir, filename))
        result = gaussian_blur(2.0)
        tifffile.imwrite(
            os.path.join(output_dir, filename),
            result.image
        )
        result.close()
        win.close()
```

### Headless Batch Processing

For server or cluster environments without a display:

```python
import flika
flika.start_flika(headless=True)

from flika.process.file_ import open_file
from flika.process.filters import gaussian_blur

win = open_file('input.tif')
result = gaussian_blur(2.0)
# Save and close...
```

## Provenance Export

Flika records every operation as structured provenance data. This can be exported for
reproducibility and shared alongside data.

**File > Save > Export Provenance**

The provenance file (JSON) contains:
- Complete operation history with parameters
- Input/output relationships
- Timestamps and software version
- Sufficient information to reproduce the analysis

```python
# Export programmatically
g.m._export_provenance()
```

### Workflow Templates

Recorded workflows can be saved as templates and re-applied:

```python
from flika.app.templates import TemplateManager
tm = TemplateManager.instance()
# Templates are stored in ~/.FLIKA/templates/
```

## See Also

- [File Operations](file_operations.md) -- File format details
- [Getting Started](getting_started.md#headless-mode) -- Headless mode setup
- [API Reference](api_reference.md) -- Scripting reference
