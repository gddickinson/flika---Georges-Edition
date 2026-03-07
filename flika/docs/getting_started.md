# Getting Started

This guide walks you through installing flika, launching the application, and performing
basic image operations.

## Installation

### Prerequisites

- Python 3.12 (recommended via Anaconda/Miniconda)
- A working display for the GUI (or use headless mode for scripting)

### Creating an Environment

```bash
conda create -n flika_env python=3.12
conda activate flika_env
```

### Installing Flika

```bash
pip install flika
```

Or from the source directory:

```bash
cd /path/to/flika
pip install -e .
```

### Key Dependencies

Flika depends on the following packages (installed automatically):

| Package | Version | Purpose |
|---|---|---|
| qtpy | latest | Qt abstraction (bridges PyQt5/PyQt6) |
| PyQt6 | 6.x | GUI toolkit |
| pyqtgraph | 0.14+ | Fast image display and plotting |
| numpy | 2.0+ | Array operations |
| scipy | 1.15+ | Scientific computing |
| scikit-image | latest | Image processing algorithms |
| tifffile | latest | TIFF file I/O |
| dask | latest | Lazy array support for large files |

Optional dependencies for specific features:

| Package | Feature |
|---|---|
| torch | AI tools (denoiser, classifiers) |
| cellpose | Cellpose segmentation |
| stardist | StarDist segmentation |
| segment-anything | SAM interactive segmentation |
| zarr | Zarr/OME-Zarr format support |
| h5py | HDF5 file support |
| bioformats | BioFormats file support |
| napari | napari interoperability |

## Launching Flika

### From Python

```python
import flika
flika.start_flika()
```

### From the Command Line

```bash
python -c "import flika; flika.start_flika()"
```

### Headless Mode (No GUI)

For scripting and batch processing without a display:

```python
import flika
flika.start_flika(headless=True)

from flika.process.file_ import open_file
win = open_file('image.tif')
# ... process and save
```

## Opening Your First Image

### From the GUI

1. Go to **File > Open > Open Image/Movie**
2. Browse to your file and click Open
3. The image opens in a new Window

### From the Console

The flika main window includes an embedded IPython console at the bottom. Type commands
directly:

```python
from flika.process.file_ import open_file
win = open_file('/path/to/my_stack.tif')
```

### Supported Formats

Flika supports TIFF, HDF5, NumPy (.npy), Zarr, OME-Zarr, BioFormats, Imaris (.ims),
and BMP files. See [File Operations](file_operations.md) for full details.

## Basic Navigation

### Image Stacks (Time Series)

- **Scroll wheel** on the image to move through frames
- The **timeline** slider below the image shows the current frame
- The **frame counter** in the status area shows `frame / total`

### Zooming and Panning

- **Scroll wheel** while holding **Ctrl** to zoom in/out
- **Click and drag** with the right mouse button to pan
- **Right-click** on the image for the pyqtgraph context menu (auto-range, export, etc.)

### Adjusting Contrast

- The **histogram** on the right side of each image window controls the lookup table (LUT)
- Drag the yellow lines on the histogram to set the display range
- Click **LUT norm** to auto-normalize the display range
- Right-click the histogram to change the colormap

### 4D Data (Volumes)

When you open a 4D dataset (T, Z, X, Y), additional dimension sliders appear below the image.
Use these to navigate through Z-slices and other dimensions.

- **View > Orthogonal Views** opens XZ and YZ cross-section panels
- **View > 3D Volume Viewer** opens a volume rendering window
- Hold **C** in the main window and click to position the orthogonal crosshair

## Working with Multiple Windows

Each image opens in its own Window. You can have many windows open simultaneously.

- Click a window to make it the **current window** (`g.win`)
- All processing operations act on the current window by default
- `g.windows` contains a list of all open windows

```python
# Access all open windows
for w in g.windows:
    print(w.name, w.image.shape)

# Switch the current window
g.win = g.windows[0]
```

## Your First Processing Operation

Try applying a Gaussian blur:

1. Open an image
2. Go to **Process > Filters > Gaussian Blur**
3. Enter a sigma value (e.g., 2.0) and click OK
4. A new window appears with the filtered result

From the console:

```python
from flika.process.filters import gaussian_blur
result = gaussian_blur(2.0)
```

## Undo and Redo

- **Ctrl+Z** to undo the last operation
- **Ctrl+Shift+Z** to redo
- All processing operations support undo via the Edit menu

## Saving Results

- **File > Save > Save Image** to save the current window as TIFF
- **File > Save > Save Movie (.mp4)** to export a movie
- **File > Save > Export Provenance** to save a record of all operations

See [File Operations](file_operations.md) for all save options.

## Next Steps

- [User Interface](user_interface.md) -- Learn the full interface layout
- [Image Menu](image_menu.md) -- Stack operations, color tools, measurements
- [Process Menu](process_menu.md) -- Filters, binary operations, segmentation
- [ROI Guide](roi_guide.md) -- Drawing and using regions of interest
- [Keyboard Shortcuts](keyboard_shortcuts.md) -- Speed up your workflow
