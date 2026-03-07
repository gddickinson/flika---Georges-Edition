# File Operations

Flika supports a wide range of file formats through its FormatRegistry system. This
document covers opening, saving, and interoperability options.

## Supported File Formats

### Reading

| Format | Extensions | Handler | Notes |
|---|---|---|---|
| TIFF | `.tif`, `.tiff` | TiffHandler | Standard and OME-TIFF. Supports BigTIFF. |
| HDF5 | `.h5`, `.hdf5`, `.hdf` | H5Handler | Reads datasets; prompts for dataset selection. |
| NumPy | `.npy` | NPYHandler | NumPy binary arrays. |
| Zarr | `.zarr` | ZarrHandler | Zarr stores (directory format). |
| OME-Zarr | `.ome.zarr` | OMEZarrHandler | NGFF-compliant OME-Zarr. Supports lazy loading. |
| BioFormats | various | BioFormatsHandler | Requires `python-bioformats`. Reads proprietary formats. |
| Imaris | `.ims` | ImarisHandler | Bitplane Imaris files (HDF5-based). |
| BMP | `.bmp` | BMPHandler | Windows bitmap images. |

### Large File Support

For files too large to fit in memory, flika uses **lazy loading** via dask arrays. The
`LazyArray` class wraps dask arrays to provide transparent on-demand reading. This is
especially useful for OME-Zarr, large TIFF stacks, and HDF5 files.

```python
from flika.process.file_ import open_file
# Large files are automatically loaded lazily when they exceed available memory
win = open_file('huge_stack.ome.zarr')
print(type(win.image))  # May be a LazyArray backed by dask
```

## Opening Files

### From the GUI

**File > Open** provides several options:

| Action | Description |
|---|---|
| **Open Image/Movie** | Opens a single image or movie file. Format is auto-detected. |
| **Open Image Sequence** | Opens a folder of images as a single stack (one file per frame). |
| **Open ROIs** | Loads saved ROI definitions onto the current window. |
| **Open Points** | Loads point coordinates from a file. |
| **Open SPT Results** | Loads SPT tracking results from a previously saved analysis. |

### From the Console

```python
from flika.process.file_ import open_file

# Open a single file
win = open_file('/path/to/image.tif')

# Open with explicit format hints
win = open_file('/path/to/data.h5')
```

### Recent Files

**File > Recent Files** shows a list of recently opened files. Click any entry to re-open
that file. The list is stored in `g.settings['recent_files']`.

### Image Sequences

To open a folder of numbered images (e.g., `frame_001.tif`, `frame_002.tif`, ...):

1. **File > Open > Open Image Sequence**
2. Select the folder containing the images
3. Flika reads all compatible images and assembles them into a single stack

## Saving Files

### From the GUI

**File > Save** provides:

| Action | Description |
|---|---|
| **Save Image** | Saves the current window as a TIFF file. Opens a save dialog. |
| **Save Movie (.mp4)** | Exports the current stack as an MP4 movie file. |
| **Save Points** | Saves point ROI coordinates to a file. |
| **Save All ROIs** | Saves all ROIs on the current window to a file. |
| **Export Provenance** | Saves the full operation history as a JSON file. |

### From the Console

```python
from flika.utils.misc import save_file_gui

# Save via dialog
save_file_gui()

# Save programmatically
import tifffile
tifffile.imwrite('/path/to/output.tif', g.win.image)
```

### Export Formats

For interoperability exports, see **File > Interop**:

| Action | Description |
|---|---|
| **Export OME-TIFF** | Saves with OME-XML metadata for cross-platform compatibility. |
| **Export OME-Zarr** | Saves as NGFF-compliant OME-Zarr with pyramidal resolution levels. |

## Interoperability

### napari Bridge

**File > Interop > Send to napari** sends the current window's image to napari as a layer.
**File > Interop > Import from napari** pulls the active napari layer into a new flika window.

```python
# Requires napari to be installed
# Send current window to napari
g.m._send_to_napari()

# Import from napari
g.m._import_from_napari()
```

### ImageJ Bridge

**File > Interop > ImageJ > Send to ImageJ** sends the current image to a running ImageJ
instance via the ImageJ-Python bridge.
**File > Interop > ImageJ > Import from ImageJ** pulls the current ImageJ image into flika.

### Array Protocol

Flika windows expose their data as numpy arrays, making it easy to exchange data with
any Python tool:

```python
# Get the raw array
arr = g.win.image  # numpy ndarray

# Create a new window from any array
from flika.window import Window
import numpy as np
data = np.random.randn(100, 256, 256)
win = Window(data, name='random_data')
```

## Batch Processing

**File > Batch Process...** opens a batch processing dialog that lets you:

1. Select a folder of input files
2. Choose a processing pipeline (sequence of operations)
3. Configure output format and destination
4. Run the pipeline on all files

This is useful for applying the same analysis to many datasets without manual intervention.

## Provenance

Every operation performed in flika is recorded with full provenance information:

- Function name and parameters
- Input and output window references
- Timestamps
- Software version

Use **File > Save > Export Provenance** to save this history. If
`g.settings['auto_export_provenance']` is `True`, provenance is exported automatically
after each operation.

See also: [API Reference](api_reference.md#macro-recording) for macro recording details.

## See Also

- [Getting Started](getting_started.md) -- Opening your first image
- [Interoperability](interoperability.md) -- Detailed interop guide
- [Image Menu](image_menu.md) -- Stack operations after loading
