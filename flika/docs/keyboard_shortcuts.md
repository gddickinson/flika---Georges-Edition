# Keyboard Shortcuts

Complete list of keyboard shortcuts available in flika.

## Global Shortcuts

These work anywhere in the application.

| Shortcut | Action |
|---|---|
| **Ctrl+Z** | Undo last operation |
| **Ctrl+Shift+Z** | Redo last undone operation |

## Image Window Shortcuts

These work when an image window is focused.

### Navigation

| Shortcut | Action |
|---|---|
| **Scroll Wheel** | Move through frames (time axis) |
| **Ctrl+Scroll Wheel** | Zoom in/out |
| **Right-click + Drag** | Pan the view |

### ROI Operations

| Shortcut | Action |
|---|---|
| **Delete** | Delete the ROI under the mouse cursor |
| **Click + Drag** | Draw a new ROI (type depends on current mouse mode) |

### Orthogonal and Volume Views

| Shortcut | Action |
|---|---|
| **Hold C + Click** | Position the orthogonal/volume crosshair at the clicked location |

## Mouse Modes

The current mouse mode determines what happens when you click and drag on an image.
Change the mode in **File > Settings**.

| Mode | Click+Drag Behavior |
|---|---|
| `rectangle` | Draw a rectangular ROI |
| `ellipse` | Draw an elliptical ROI |
| `line` | Draw a line ROI |
| `freehand` | Draw a freehand polygon ROI |
| `point_roi` | Place a point ROI at the click location |
| `rect_line` | Draw a multi-segment line with width |
| `center_surround` | Draw concentric center-surround ROI |
| `pencil` | Paint pixel values at the pencil value |

Set the mode from the console:

```python
g.settings['mousemode'] = 'rectangle'
```

## pyqtgraph Context Menu

Right-click on any image to access the pyqtgraph context menu:

| Item | Action |
|---|---|
| **View All** | Auto-range the view to show the full image |
| **Export...** | Export the current view as PNG, SVG, or other formats |
| **X Axis / Y Axis** | Configure axis behavior (auto-range, invert, etc.) |
| **Mouse Mode** | Switch between Pan and Rectangle zoom modes (pyqtgraph modes, not ROI modes) |

## Histogram/LUT Controls

| Action | Description |
|---|---|
| **Drag yellow lines** | Adjust display range (min/max) on the histogram |
| **Right-click histogram** | Change colormap |
| **Click LUT norm** | Auto-normalize display range to data min/max |

## Timeline Controls

| Action | Description |
|---|---|
| **Drag timeline slider** | Jump to a specific frame |
| **Click in timeline plot** | Jump to the clicked time point |
| **Scroll wheel on image** | Step through frames one at a time |

## Console Shortcuts

The embedded IPython console supports standard IPython shortcuts:

| Shortcut | Action |
|---|---|
| **Tab** | Auto-complete |
| **Shift+Enter** | Execute current line |
| **Up/Down arrows** | Navigate command history |
| **Ctrl+C** | Interrupt running command |
| **Ctrl+L** | Clear console |

## ROI Interaction

| Action | Description |
|---|---|
| **Click inside ROI** | Select the ROI; show its trace |
| **Drag inside ROI** | Move the ROI |
| **Drag ROI handle** | Resize the ROI |
| **Hover + Delete** | Delete the ROI under the cursor |
| **Double-click ROI (in Manager)** | Rename the ROI |

## Tips

### Quick ROI Drawing

To quickly switch between ROI types without opening Settings:

```python
g.settings['mousemode'] = 'line'       # Switch to line mode
g.settings['mousemode'] = 'rectangle'  # Switch back to rectangle
```

### Navigating Large Stacks

For stacks with many frames, use the console to jump directly:

```python
g.win.setIndex(500)  # Jump to frame 500
```

### Keyboard-Driven Workflow

Combine console commands with shortcuts for efficient workflows:

```python
# Process current window and move to next
from flika.process.filters import gaussian_blur
gaussian_blur(2.0)
# Click the next window, repeat
```

## See Also

- [User Interface](user_interface.md) -- Full UI description
- [ROI Guide](roi_guide.md) -- ROI drawing and management
- [API Reference](api_reference.md) -- Console scripting reference
