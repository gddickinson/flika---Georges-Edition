# API Reference

This document covers the key classes and APIs for scripting flika from the embedded
console or external scripts.

## Global Variables (`g`)

The `flika.global_vars` module (imported as `g`) provides access to application state:

```python
from flika import global_vars as g
```

| Variable | Type | Description |
|---|---|---|
| `g.m` | `FlikaApplication` | The main application window |
| `g.win` | `Window` or `None` | The currently selected window |
| `g.windows` | `list[Window]` | All open windows |
| `g.currentWindow` | `Window` or `None` | Alias for `g.win` |
| `g.traceWindows` | `list[TraceFig]` | All open trace figure windows |
| `g.currentTrace` | `TraceFig` or `None` | The currently selected trace |
| `g.settings` | `Settings` | Persistent settings dictionary |
| `g.clipboard` | varies | Internal clipboard |
| `g.menus` | `list[QMenu]` | Image and Process menus |
| `g.dialogs` | `list` | Open dialog references |
| `g.headless` | `bool` | True when running without GUI |

### Settings

`g.settings` behaves like a dictionary. Changes are automatically saved to
`~/.FLIKA/settings.json`.

```python
# Read a setting
dtype = g.settings['internal_data_type']  # 'float64'

# Change a setting
g.settings['roi_color'] = '#00ff00'

# Batch changes (single disk write)
with g.settings.batch():
    g.settings['point_size'] = 8
    g.settings['point_color'] = '#ff0000'
```

Key settings:

| Key | Default | Description |
|---|---|---|
| `internal_data_type` | `'float64'` | Default dtype for images |
| `multiprocessing` | `True` | Enable parallel processing |
| `nCores` | CPU count | Number of worker cores |
| `mousemode` | `'rectangle'` | Current ROI drawing mode |
| `roi_color` | `'#ffff00'` | Default ROI color |
| `point_color` | `'#ff0000'` | Default point ROI color |
| `point_size` | `5` | Point marker size |
| `pixel_size` | `108.0` | Pixel size in nm |
| `frame_interval` | `0.05` | Frame interval in seconds |
| `acceleration_device` | `'Auto'` | GPU/CPU selection |
| `debug_mode` | `False` | Enable verbose logging |

### Alerts and Messages

```python
g.alert("Something happened!")  # Shows a popup dialog
g.status_msg("Processing...")   # Shows in the status bar
```

## Window Class

`flika.window.Window` is the central class. Each image is displayed in a Window.

### Creating Windows

```python
from flika.window import Window
import numpy as np

# From a numpy array
data = np.random.randn(100, 256, 256).astype('float32')
win = Window(data, name='My Image')

# From a file
from flika.process.file_ import open_file
win = open_file('/path/to/image.tif')
```

### Constructor

```python
Window(tif, name='flika', filename='', commands=None, metadata=None)
```

| Parameter | Type | Description |
|---|---|---|
| `tif` | `numpy.ndarray` | Image array: (X, Y), (T, X, Y), or (T, X, Y, Z) |
| `name` | `str` | Display name |
| `filename` | `str` | Source file path |
| `commands` | `list[str]` | Command history |
| `metadata` | `dict` | Metadata dictionary |

### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `image` | `ndarray` | The image array (T, X, Y) or (X, Y) |
| `volume` | `ndarray` or `None` | 4D array (T, X, Y, Z) if present |
| `name` | `str` | Window display name |
| `filename` | `str` | Source file path |
| `metadata` | `dict` | Metadata dictionary |
| `rois` | `list[ROI_Base]` | ROIs drawn on this window |
| `currentIndex` | `int` | Current frame index |
| `mx`, `my` | `int` | Image dimensions (width, height) |
| `mt` | `int` | Number of frames |
| `imageview` | `ImageView` | The pyqtgraph ImageView widget |
| `commands` | `list[str]` | Operation history |

### Key Methods

```python
# Close the window
win.close()

# Navigate frames
win.setIndex(frame_number)

# Toggle views
win.toggleOrthogonalViews()
win.toggleVolumeViewer()
```

### Signals

| Signal | Emitted When |
|---|---|
| `closeSignal` | Window is closed |
| `sigTimeChanged(int)` | Frame index changes |
| `sigSliceChanged` | Z/C/D4 slider changes |
| `sigROICreated(object)` | An ROI is added |
| `sigROIRemoved(object)` | An ROI is removed |
| `gainedFocusSignal` | Window gains focus |
| `lostFocusSignal` | Window loses focus |
| `keyPressSignal(QEvent)` | Key pressed in window |

## BaseProcess

All processing operations inherit from `BaseProcess`. It handles GUI creation, undo/redo,
and macro recording.

### Writing a Custom Operation

```python
from flika.utils.BaseProcess import BaseProcess, SliderLabel
from flika import global_vars as g
import numpy as np

class InvertImage(BaseProcess):
    def __init__(self):
        super().__init__()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = g.win.image.max() - g.win.image
        self.newname = f'{g.win.name} - Inverted'
        return self.end()

    def gui(self):
        self.gui_reset()
        super().gui()

invert = InvertImage()

# Use it
result_window = invert()
```

### BaseProcess Flow

1. `gui()` -- builds and shows the dialog
2. `__call__(**params)` -- executes the operation
3. `start(keepSourceWindow)` -- snapshots state for undo
4. Set `self.newtif` (result array) and `self.newname` (result name)
5. `end()` -- creates the result window, records macro, pushes undo

## ROI Types

All ROI types inherit from `ROI_Base` in `flika.roi`:

| Class | Kind String | Description |
|---|---|---|
| `ROI_rectangle` | `'rectangle'` | Rectangular region |
| `ROI_ellipse` | `'ellipse'` | Elliptical region |
| `ROI_line` | `'line'` | Line segment |
| `ROI_freehand` | `'freehand'` | Freeform polygon |
| `ROI_point` | `'point_roi'` | Single-pixel crosshair |
| `ROI_rect_line` | `'rect_line'` | Multi-segment line with width |
| `ROI_center_surround` | `'center_surround'` | Concentric inner/outer regions |

### Common ROI Methods

```python
roi = g.win.rois[0]

# Get the time trace
trace = roi.getTrace()       # 1D array, length = number of frames

# Get the binary mask
mask = roi.getMask()          # 2D boolean array

# Get position and bounds
pos = roi.pos()
# ROI kind
print(roi.kind)              # 'rectangle', 'ellipse', etc.
```

## Macro Recording

Flika records all operations as macro commands for reproducibility.

```python
from flika.app.macro_recorder import MacroRecorder

recorder = MacroRecorder.instance()

# Get the recorded commands
commands = recorder.commands

# Save macro to file
recorder.save('/path/to/macro.py')

# Replay a macro
exec(open('/path/to/macro.py').read())
```

## Undo/Redo

The undo system uses a stack of `ProcessCommand` objects.

```python
from flika.core.undo import undo_stack

undo_stack.undo()   # Undo last operation
undo_stack.redo()   # Redo last undone operation
undo_stack.clear()  # Clear undo history (frees memory)
```

## Console Scripting Examples

### Batch Apply a Filter

```python
from flika.process.filters import gaussian_blur
for win in g.windows:
    g.win = win
    gaussian_blur(sigma=1.5)
```

### Extract ROI Data

```python
import numpy as np
traces = np.array([roi.getTrace() for roi in g.win.rois])
np.savetxt('traces.csv', traces.T, delimiter=',')
```

### Create a Montage

```python
import numpy as np
from flika.window import Window

images = [w.image[0] for w in g.windows[:4]]  # First frame of 4 windows
row1 = np.hstack(images[:2])
row2 = np.hstack(images[2:])
montage = np.vstack([row1, row2])
Window(montage, name='Montage')
```

### Measure Across Windows

```python
for w in g.windows:
    img = w.image
    print(f"{w.name}: mean={img.mean():.2f}, std={img.std():.2f}, "
          f"shape={img.shape}")
```

## See Also

- [Plugins](plugins.md) -- Writing plugins with BaseProcess
- [ROI Guide](roi_guide.md) -- Interactive ROI usage
- [Keyboard Shortcuts](keyboard_shortcuts.md) -- Keyboard reference
- [Process Menu](process_menu.md) -- All built-in operations
