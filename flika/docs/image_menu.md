# Image Menu

The Image menu contains operations for manipulating image stacks, aligning channels,
working with color, making measurements, and adding overlays.

## Stacks

Stack operations modify or create image stacks.

### Duplicate

Creates an exact copy of the current window. Useful before applying destructive operations.

```python
from flika.process.stacks import duplicate
new_win = duplicate()
```

### Generate Random Image

Creates a window filled with random noise. Useful for testing.

### Generate Phantom Volume

Creates a synthetic 4D volume with geometric features for testing 3D/4D operations.

### Trim Frames

Removes frames from the beginning and/or end of a stack.

```python
from flika.process.stacks import trim
trimmed = trim(firstFrame=10, lastFrame=200)
```

### Remove Frames

Removes specific frames from within a stack (not just the ends).

### Deinterlace

Separates interleaved frames into two stacks. Useful for dual-channel acquisitions
where channels alternate frame-by-frame.

```python
from flika.process.stacks import deinterleave
# keep='Odd' or 'Even'
result = deinterleave(keep='Odd')
```

### Z Project

Projects a stack along the time/Z axis using a specified method.

| Method | Description |
|---|---|
| Average | Mean intensity projection |
| Max | Maximum intensity projection |
| Min | Minimum intensity projection |
| Sum | Sum projection |
| Standard Deviation | Std deviation projection |
| Median | Median projection |

```python
from flika.process.stacks import zproject
projected = zproject(method='Average')
```

### Pixel Binning

Reduces spatial resolution by averaging blocks of pixels. Specify the bin factor (e.g., 2
means 2x2 blocks become single pixels).

```python
from flika.process.stacks import pixel_binning
binned = pixel_binning(binFactor=2)
```

### Frame Binning

Reduces temporal resolution by averaging groups of consecutive frames.

### Resize

Resizes the image to new dimensions using interpolation.

### Concatenate Stacks

Joins two stacks along the time axis. Select the second stack from the dialog.

### Change Data Type

Converts the image array to a different numeric type (uint8, uint16, float32, float64, etc.).

### Shear Transform

Applies a shear transformation to the stack. Useful for correcting oblique acquisition angles.

### Motion Correction

Corrects for sample drift across frames by aligning each frame to a reference. Uses
cross-correlation to compute frame-to-frame shifts.

```python
from flika.process.stacks import motion_correction
corrected = motion_correction()
```

## Alignment

### Channel Alignment

Aligns two channel images that may be spatially offset. Opens a dialog to select the
reference and moving windows, then computes and applies the alignment transform.

```python
from flika.process.alignment import channel_alignment
channel_alignment()
```

## Color

### Split Channels

Splits a multi-channel (RGB) image into separate grayscale windows, one per channel.

### Blend Channels

Combines multiple grayscale windows into a single RGB or multi-channel composite.

### Channel Compositor

Opens the Channel Compositor panel for advanced multi-channel overlay. Features:

- Up to 9 channels with individual colormaps (Red, Green, Blue, Cyan, Magenta, Yellow, Gray, Orange, Hot)
- Per-channel visibility toggle, contrast adjustment, and opacity
- Additive blending for fluorescence-style overlay
- Real-time preview

See also: the Channel Compositor dock in [User Interface](user_interface.md).

## Measure

### Measure

Computes statistics for the current window or within ROIs. Reports mean, standard deviation,
min, max, area, and other metrics.

```python
from flika.process.measure import measure
measure()
```

### Linescan

Extracts intensity values along a line ROI across all frames, producing a 2D kymograph-like
display (position vs. time).

### Kymograph

Generates a kymograph from a line ROI. The kymograph shows how intensity along the line
changes over time, displayed as a 2D image where the x-axis is position along the line
and the y-axis is time.

```python
from flika.process.kymograph import kymograph
kymo = kymograph()
```

## Set Value

Sets all pixels in the current window (or within the current ROI) to a specified value.
Useful for masking or creating synthetic images.

## Overlay

### Background

Opens a dialog to overlay a background image from another window behind the current image.
Adjustable opacity lets you blend the foreground and background.

### Timestamp

Burns a timestamp overlay onto each frame of the stack. Configure the time format,
position, font size, and frame interval.

### Scale Bar

Adds a scale bar overlay to the image. Configure:
- Length in physical units (micrometers)
- Pixel size calibration
- Position, color, and font

### Track Overlay

Visualizes particle tracks from SPT analysis as colored paths overlaid on the image.
Tracks are drawn from `window.metadata['spt']` data. See [SPT Guide](spt_guide.md)
for details on generating track data.

## See Also

- [Process Menu](process_menu.md) -- Filtering, binary operations, segmentation
- [ROI Guide](roi_guide.md) -- How to use ROIs with measurement tools
- [File Operations](file_operations.md) -- Opening and saving images
