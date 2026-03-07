# ROI Guide

Regions of Interest (ROIs) are interactive overlays drawn on image windows. They are used
for measuring pixel values, extracting time traces, creating masks, and defining analysis
regions.

## ROI Types

### Rectangle

Click and drag to draw a rectangular region. Default dimensions can be set in
**File > Settings** (rect_width, rect_height).

- Resizable via corner and edge handles
- Reports mean, std, min, max over the enclosed pixels
- Generates a time trace of mean intensity within the rectangle

### Ellipse

Click and drag to draw an elliptical region.

- Resizable via handles
- The mask is computed as an elliptical region within the bounding box

### Line

Click and drag to draw a straight line between two points.

- Endpoints are draggable
- Used for linescan analysis and kymograph generation
- Reports intensity profile along the line

### Freehand

Click and drag to draw a freeform polygon shape.

- The path follows the mouse cursor
- The shape closes automatically when the mouse is released
- Interior pixels are determined by the polygon boundary

### Point

Click to place a single-pixel crosshair marker.

- Shows the intensity value at that pixel over time
- Configurable color and size in Settings
- Point color: `g.settings['point_color']`
- Point size: `g.settings['point_size']`

### Rect Line

Click to create a multi-segment line with adjustable width.

- Each segment has adjustable width for averaging across the line
- Useful for wide linescan measurements along curved paths
- Supports kymograph generation

### Center-Surround

Click and drag to create concentric inner/outer regions.

- The outer ellipse defines the surround region
- The inner region is defined by the `cs_inner_ratio` setting
- `getTrace()` returns center_mean - surround_mean (useful for DF analysis)
- Shape can be circle, ellipse, or square (configured in Settings)

### Pencil

Click to paint pixel values directly onto the image.

- The painted value is set by `g.settings['pencil_value']`
- Useful for manual mask creation and image annotation
- Modifies the image data directly (use undo if needed)

## Drawing ROIs

### Setting the Drawing Mode

The drawing mode determines what type of ROI is created when you click on an image.
Change the mode via **File > Settings** or from the console:

```python
g.settings['mousemode'] = 'rectangle'  # or 'ellipse', 'line', 'freehand', 'point_roi', etc.
```

### Drawing

1. Set the desired ROI type
2. Click and drag on the image window
3. Release to complete the ROI

### Modifying ROIs

- **Move**: Click inside the ROI and drag
- **Resize**: Drag the handles at corners/edges
- **Delete**: Hover over the ROI and press the **Delete** key
- **Color**: Right-click the ROI or use the ROI Manager to change color

## ROI Manager

Open the ROI Manager via **View > ROI Manager**. It provides a centralized panel for
managing all ROIs on the active window.

### Features

| Button | Action |
|---|---|
| **Plot All** | Shows time traces for all ROIs simultaneously |
| **Unplot All** | Hides all ROI time traces |
| **Delete Selected** | Removes selected ROIs from the window |
| **Export CSV** | Exports all ROI traces as a CSV file |

### Per-ROI Operations

- **Rename**: Double-click the ROI name in the list to edit
- **Change Color**: Click the color swatch to open a color picker
- **Toggle Visibility**: Check/uncheck to show/hide individual ROIs

### Auto-Naming

ROIs are automatically named with a sequential identifier (e.g., `ROI_0`, `ROI_1`).
You can rename them in the ROI Manager for clarity.

## Time Traces

When an ROI is drawn on a stack (3D data), clicking it generates a **time trace** showing
how the mean intensity within the ROI changes over frames.

- Traces are displayed in a separate TraceFig window
- Multiple ROIs can be plotted simultaneously
- Each trace uses the ROI's color for identification

```python
# Get the trace data for an ROI
roi = g.win.rois[0]
trace = roi.getTrace()  # 1D numpy array, one value per frame
```

## Measurements

Use **Image > Measure > Measure** to compute statistics for the current window or within
ROIs:

| Metric | Description |
|---|---|
| Mean | Average pixel intensity |
| Std | Standard deviation |
| Min | Minimum pixel value |
| Max | Maximum pixel value |
| Area | Number of pixels in the ROI |
| Centroid | Center position (x, y) |

## Generating ROIs from Binary Images

**Process > Binary > Generate ROIs** creates ROIs from connected components in a binary
image. Each connected region becomes a separate ROI.

```python
from flika.process.binary import threshold, generate_rois
binary = threshold(value=100)
generate_rois()
```

## Saving and Loading ROIs

### Saving

**File > Save > Save All ROIs** saves all ROIs on the current window to a file.

### Loading

**File > Open > Open ROIs** loads ROIs from a saved file and applies them to the current
window.

## Using ROIs from the Console

```python
# Access all ROIs on the current window
rois = g.win.rois

# Get the first ROI
roi = rois[0]

# Get ROI properties
print(roi.kind)       # 'rectangle', 'ellipse', 'line', etc.
print(roi.name)       # display name
print(roi.pos())      # position

# Get the binary mask for an ROI
mask = roi.getMask()   # 2D boolean array matching image dimensions

# Get the trace
trace = roi.getTrace() # 1D array of mean intensity per frame

# Create an ROI programmatically
from flika.roi import ROI_rectangle
roi = ROI_rectangle(g.win, pos=[10, 10], size=[50, 50])
```

## Linked ROIs

ROIs can be linked so they move together. This is useful when comparing the same region
across multiple windows.

```python
# Link two ROIs
roi1.linkedROIs.add(roi2)
roi2.linkedROIs.add(roi1)
```

## ROI Signals

ROIs emit signals that you can connect to for custom behavior:

```python
# Connect to the window's ROI signals
g.win.sigROICreated.connect(lambda roi: print(f"New ROI: {roi.name}"))
g.win.sigROIRemoved.connect(lambda roi: print(f"Removed ROI: {roi.name}"))
```

## See Also

- [Image Menu](image_menu.md#measure) -- Measurement tools
- [User Interface](user_interface.md#roi-drawing) -- ROI drawing modes
- [Keyboard Shortcuts](keyboard_shortcuts.md) -- ROI-related shortcuts
- [API Reference](api_reference.md) -- ROI class reference
