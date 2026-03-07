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

Adds a timestamp overlay to each frame of the stack. The frame rate auto-populates from
the `frame_interval` setting (1 / interval = Hz), but can be manually overridden.

| Option | Description |
|---|---|
| **Frame Rate (Hz)** | Auto-populated from `frame_interval` setting; manually adjustable |
| **Font Size (pt)** | Text size, 4-120 points |
| **Text Color** | 8 presets (White, Black, Yellow, Green, Red, Cyan, Magenta) + custom color picker |
| **Background Color** | None, Black, White, Semi-transparent Black/White, or custom |
| **Location** | Lower Right, Lower Left, Upper Right, Upper Left |
| **Bold** | Bold text rendering |
| **Show Frame Number** | Appends `[F123]` frame counter |
| **Custom Format** | Format string with placeholders: `{time}`, `{frame}`, `{ms}`, `{s}`, `{min}`, `{hr}` |
| **Show** | Toggle visibility on/off |

Time is auto-formatted with appropriate units (ms for < 1s, seconds, minutes, hours).

```python
from flika.process.overlay import time_stamp
time_stamp(framerate=20.0, font_size=14, color='Yellow', location='Upper Left')
```

### Scale Bar

Adds a scale bar overlay to the image. The bar width auto-calculates from the `pixel_size`
setting (nm) and rounds to a "nice" number (1, 2, 5, 10, 20, 50, ...).

| Option | Description |
|---|---|
| **Physical Width** | Bar length in physical units; auto-calculated from pixel_size setting |
| **Unit** | Display unit: um, nm, mm, or px |
| **Width (pixels)** | Bar length in image pixels; linked to physical width via pixel_size |
| **Bar Thickness (px)** | Bar height in pixels (1-100) |
| **Font Size (pt)** | Label text size |
| **Bold Label** | Bold label text |
| **Label Color** | 8 presets + custom color picker |
| **Bar Color** | Separate color for the bar itself |
| **Background** | Label background: None, Black, White, Semi-transparent, or custom |
| **Location** | 4-corner placement |
| **Show Label** | Toggle the text label above the bar |
| **Custom Label** | Override auto-generated label text |
| **Horizontal/Vertical Offset** | Fine-tune position from the corner |

Physical width and pixel width are linked: changing one auto-updates the other based
on the pixel_size setting.

```python
from flika.process.overlay import scale_bar
scale_bar(width_um=10.0, width_pixels=93, font_size=12, color='White',
          location='Lower Right', bar_color='White', bar_height=4)
```

### Track Overlay

Visualizes particle tracks from SPT analysis as colored paths overlaid on the image.
Tracks are drawn from `window.metadata['spt']` data. See [SPT Guide](spt_guide.md)
for details on generating track data.

## See Also

- [Process Menu](process_menu.md) -- Filtering, binary operations, segmentation
- [ROI Guide](roi_guide.md) -- How to use ROIs with measurement tools
- [File Operations](file_operations.md) -- Opening and saving images
