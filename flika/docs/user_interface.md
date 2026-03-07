# User Interface Guide

This document describes the layout and components of the flika user interface.

## Main Window

The main flika window is a compact control bar that contains:

- **Menu bar** at the top with all menus
- **Status bar** at the bottom showing messages and alerts
- **Console** -- an embedded IPython console for scripting

The main window is always positioned at the top-left of the screen. Image windows float
independently and can be arranged freely.

## Menu Bar

The menu bar contains the following menus:

| Menu | Purpose |
|---|---|
| **File** | Open, save, recent files, interoperability, batch processing, settings |
| **Edit** | Undo (Ctrl+Z), Redo (Ctrl+Shift+Z) |
| **View** | Orthogonal views, 3D volume viewer, ROI Manager, Metadata Editor, Figure Composer |
| **Image** | Stack operations, alignment, color, measurement, overlays |
| **Process** | Binary, math, filters, segmentation, detection, colocalization, SPT, export |
| **Plugins** | Installed plugins and Plugin Manager |
| **Scripts** | Saved scripts and Script Editor |
| **AI** | AI-assisted tools (denoiser, classifiers, segmentation, model zoo) |
| **Help** | Documentation, updates, dependency checks, GPU status |

For detailed descriptions of each menu, see:
- [Image Menu](image_menu.md)
- [Process Menu](process_menu.md)
- [AI Tools](ai_tools.md)
- [Plugins](plugins.md)

## Image Windows

Each image or stack is displayed in a **Window** object. A Window contains:

### Image Display

The central area shows the current frame of the image using pyqtgraph's ImageView widget.
Images are displayed with adjustable contrast and colormap.

### Histogram / LUT Control

The right panel shows a histogram of pixel values with two draggable lines to set the
display range. This controls contrast without modifying the underlying data.

- Drag the yellow lines to adjust min/max display values
- Right-click the histogram area to select a different colormap
- Click **LUT norm** to auto-normalize the display range

### Timeline

For stacks (3D data), a timeline slider appears below the image:

- Drag the slider or scroll the mouse wheel to change frames
- The plot area below the timeline shows ROI traces when ROIs are active

### Dimension Sliders

For 4D+ data, additional sliders appear for Z, Channel, or other dimensions. These let you
navigate through any dimension independently of the time axis.

### Status Information

Each window tracks:
- `window.name` -- the display name
- `window.image` -- the numpy array (T, X, Y) or (X, Y)
- `window.volume` -- the 4D array if present (T, X, Y, Z)
- `window.metadata` -- a dict with file info, provenance, SPT data, etc.
- `window.rois` -- list of ROIs drawn on this window
- `window.currentIndex` -- current frame index

## ROI Drawing

ROIs are drawn on image windows using the mouse. The current drawing mode is set in
**File > Settings** or via the settings toolbar.

Available ROI types:

| Mode | Description |
|---|---|
| Rectangle | Click and drag to draw a rectangle |
| Ellipse | Click and drag to draw an ellipse |
| Line | Click and drag to draw a line segment |
| Freehand | Click and drag to draw a freeform polygon |
| Point | Click to place a crosshair at a single pixel |
| Rect Line | Click to place linked line segments with adjustable width |
| Center-Surround | Click and drag for concentric inner/outer regions |
| Pencil | Click to paint pixel values directly |

Once drawn, ROIs can be:
- **Moved** by dragging
- **Resized** by dragging handles
- **Deleted** by hovering and pressing Delete
- **Plotted** by clicking -- shows a time trace in the trace window

See [ROI Guide](roi_guide.md) for full details.

## View Menu Features

### Orthogonal Views

**View > Orthogonal Views** opens a panel showing XZ and YZ cross-sections through the
current stack. Hold **C** in the image window and click to position the slice crosshair.
Works with both 3D stacks (slicing through time) and true 4D volumes (slicing through Z).

### 3D Volume Viewer

**View > 3D Volume Viewer** opens an interactive 3D rendering of volumetric data.

### ROI Manager

**View > ROI Manager** opens a dockable panel that lists all ROIs in the active window.
The ROI Manager provides:

- List of all ROIs with name, type, and color
- Buttons: Plot All, Unplot All, Delete Selected
- Export CSV of all ROI traces
- Per-ROI editing: rename, change color
- Automatic updates when the active window changes

### Metadata Editor

**View > Metadata Editor** opens the REMBI-compliant metadata editor. This allows you
to annotate images with standardized metadata fields for reproducibility and data sharing.

### Figure Composer

**View > Figure Composer** opens a publication-quality figure layout tool:

- Arrange multiple images in a grid layout
- Add labels, scale bars, and annotations
- Export as PNG, SVG, or PDF
- Drag and drop images from open windows

## Settings

**File > Settings** opens the settings editor where you can configure:

| Setting | Description |
|---|---|
| Internal Data Type | Default dtype for new images (float64) |
| Multiprocessing | Enable/disable parallel processing |
| Number of Cores | CPU cores for parallel operations |
| Mouse Mode | Default ROI drawing tool |
| ROI Color | Default color for new ROIs |
| Point Color | Default color for point ROIs |
| Point Size | Size of point ROI markers |
| Rectangle Width/Height | Default dimensions for rectangle ROIs |
| Pixel Size | Physical pixel size in nm (for scale bars) |
| Frame Interval | Time between frames in seconds |
| Acceleration Device | Auto/CPU/GPU for accelerated operations |
| GPU Memory Limit | Maximum GPU memory usage (0 = unlimited) |
| Auto Export Provenance | Automatically save provenance on each operation |

Settings are stored in `~/.FLIKA/settings.json` and persist across sessions.

## Console

The embedded IPython console at the bottom of the main window provides direct access to
all flika functionality. Key variables available in the console:

```python
g.win          # Current window
g.windows      # List of all open windows
g.settings     # Settings dict
g.m            # Main application window
```

You can run any processing operation, manipulate arrays directly, and create custom
analysis workflows.

## See Also

- [Keyboard Shortcuts](keyboard_shortcuts.md) -- All keyboard shortcuts
- [Getting Started](getting_started.md) -- First steps with flika
- [API Reference](api_reference.md) -- Scripting and class reference
