# Process Menu

The Process menu contains image processing operations organized into submenus: Binary,
Math, Filters, Image Calculator, Segmentation, Detection, Colocalization, SPT Analysis,
and Export Video.

All operations act on the current window (`g.win`) and produce a new window with the result.
They support undo/redo and are recorded in the macro.

## Binary

Operations for thresholding, morphological processing, and binary image analysis.

### Thresholding

| Operation | Description |
|---|---|
| **Threshold** | Global threshold. Pixels above the value become 1, below become 0. |
| **Adaptive Threshold** | Local threshold using a neighborhood window. Handles uneven illumination. |
| **Hysteresis Threshold** | Dual-threshold: strong edges seed, weak edges extend if connected. |
| **Multi-Otsu Threshold** | Automatic multi-class thresholding using Otsu's method. |
| **Canny Edge Detector** | Edge detection with non-maximum suppression and hysteresis. |

```python
from flika.process.binary import threshold, adaptive_threshold
binary = threshold(value=128)
binary = adaptive_threshold(block_size=15, offset=5)
```

### Logical and Morphological

| Operation | Description |
|---|---|
| **Logically Combine** | AND, OR, XOR, NOT between two binary images. |
| **Remove Small Blobs** | Removes connected components smaller than a specified area. |
| **Remove Small Holes** | Fills holes in binary objects smaller than a specified area. |
| **Binary Erosion** | Shrinks foreground objects by one pixel layer. |
| **Binary Dilation** | Expands foreground objects by one pixel layer. |
| **Generate ROIs** | Creates ROIs from connected components in a binary image. |

```python
from flika.process.binary import remove_small_blobs, binary_erosion
cleaned = remove_small_blobs(min_size=50)
eroded = binary_erosion(iterations=1)
```

### Grayscale Morphology

| Operation | Description |
|---|---|
| **Grayscale Opening** | Erosion followed by dilation (removes bright spots). |
| **Grayscale Closing** | Dilation followed by erosion (fills dark spots). |
| **Morphological Gradient** | Difference between dilation and erosion (edge detection). |
| **H-Maxima** | Suppresses maxima with height less than h. |
| **H-Minima** | Suppresses minima with depth less than h. |
| **Area Opening** | Removes bright regions smaller than a given area. |
| **Area Closing** | Removes dark regions smaller than a given area. |
| **Flood Fill** | Fills a connected region starting from a seed point. |

### Analysis

| Operation | Description |
|---|---|
| **Mask Editor** | Interactive binary mask painting and editing tool. |
| **Analyze Particles** | Measures properties of connected components (area, centroid, eccentricity). |
| **Distance Transform** | Computes distance from each foreground pixel to nearest background pixel. |
| **Watershed Segmentation** | Marker-controlled watershed for separating touching objects. |

## Math

Arithmetic operations on pixel values.

| Operation | Description |
|---|---|
| **Multiply** | Multiply all pixels by a scalar value. |
| **Divide** | Divide all pixels by a scalar value. |
| **Subtract** | Subtract a scalar from all pixels. |
| **Power** | Raise all pixels to a power. |
| **Square Root** | Take the square root of all pixels. |
| **Ratio By Baseline** | Divide each frame by the mean of a baseline period (F/F0). |
| **Absolute Value** | Take the absolute value of all pixels. |
| **Subtract Trace** | Subtract a 1D temporal trace from each pixel. |
| **Divide Trace** | Divide each pixel by a 1D temporal trace. |
| **Histogram Equalize** | Redistribute pixel values for uniform histogram. |
| **Normalize** | Scale pixel values to a specified range (e.g., 0-1). |

```python
from flika.process.math_ import ratio, normalize
df_over_f = ratio(first_frame=0, last_frame=10)
normed = normalize(min_val=0, max_val=1)
```

## Filters

Spatial and temporal filters for smoothing, sharpening, and feature detection.

### Smoothing Filters

| Filter | Description |
|---|---|
| **Gaussian Blur** | Isotropic Gaussian smoothing. Sigma controls blur radius. |
| **Difference of Gaussians** | Band-pass filter: difference between two Gaussian blurs. |
| **Butterworth Filter** | Frequency-domain low/high/band-pass filter. |
| **Mean Filter** | Average over a rectangular kernel. |
| **Median Filter** | Median over a neighborhood (good for salt-and-pepper noise). |
| **Bilateral Filter** | Edge-preserving smoothing using spatial and intensity distance. |
| **TV Denoising** | Total variation denoising (preserves edges while smoothing). |
| **Flash Remover** | Detects and removes flash artifacts from time series. |

```python
from flika.process.filters import gaussian_blur, median_filter
smooth = gaussian_blur(sigma=2.0)
denoised = median_filter(radius=3)
```

### Temporal Filters

| Filter | Description |
|---|---|
| **Fourier Filter** | Frequency-domain filtering along the time axis. |
| **Difference Filter** | Frame-to-frame difference (temporal derivative). |
| **Boxcar Differential** | Sliding window temporal difference filter. |
| **Wavelet Filter** | Wavelet-based temporal denoising. |
| **Variance Filter** | Local variance over a neighborhood. |

### Edge and Feature Detectors

| Filter | Description |
|---|---|
| **Sobel** | First-order edge detection. |
| **Laplacian** | Second-order edge/blob detection. |
| **Gaussian Laplace (LoG)** | Laplacian of Gaussian for multi-scale blob detection. |
| **Gaussian Gradient Magnitude** | Gradient magnitude after Gaussian smoothing. |

### Ridge and Structure Filters

| Filter | Description |
|---|---|
| **Sato Tubeness** | Detects tubular structures (vessels, neurites). |
| **Meijering Neuriteness** | Optimized for neurite-like structures. |
| **Hessian Filter** | Hessian-based feature detection. |
| **Gabor Filter** | Orientation-selective texture filter. |

### Rank Filters

| Filter | Description |
|---|---|
| **Maximum Filter** | Local maximum over a neighborhood. |
| **Minimum Filter** | Local minimum over a neighborhood. |
| **Percentile Filter** | Local percentile over a neighborhood. |

## Image Calculator

Performs pixel-wise arithmetic between two windows. Select two windows and an operation
(Add, Subtract, Multiply, Divide, AND, OR, XOR, Max, Min).

```python
from flika.process.binary import image_calculator
result = image_calculator(operation='Subtract')
```

## Segmentation

Advanced segmentation algorithms for labeling and partitioning images.

| Operation | Description |
|---|---|
| **Connected Components** | Labels connected regions in a binary image. |
| **Region Properties** | Computes properties (area, centroid, etc.) of labeled regions. |
| **Clear Border** | Removes labeled regions touching the image border. |
| **Expand Labels** | Grows labeled regions by a specified number of pixels. |
| **Random Walker** | Probabilistic segmentation from seed labels. |
| **SLIC Superpixels** | Segments image into compact superpixels. |
| **Find Boundaries** | Extracts boundaries between labeled regions. |
| **Find Contours** | Detects contour lines at a specified intensity level. |

## Detection

Spot and feature detection algorithms.

| Operation | Description |
|---|---|
| **Blob Detection (LoG)** | Laplacian of Gaussian blob detection with scale selection. |
| **Blob Detection (DoH)** | Determinant of Hessian blob detection. |
| **Peak Local Max** | Finds local intensity maxima above a threshold. |
| **Template Matching** | Finds locations matching a template image. |
| **Local Maxima** | Detects all local maxima in the image. |

## Colocalization

Quantitative colocalization analysis between two channels.

**Process > Colocalization > Colocalization Analysis** computes:

- **Pearson's correlation coefficient** -- linear correlation between channels
- **Manders' coefficients (M1, M2)** -- fraction of each channel overlapping
- **Costes' automatic threshold** -- statistical threshold for significance
- **Li's ICQ** -- intensity correlation quotient
- **Scatter plot** -- interactive 2D histogram of channel intensities

See also: [Image Menu > Color](image_menu.md#color) for channel splitting.

## SPT Analysis

Single Particle Tracking operations. See [SPT Guide](spt_guide.md) for full documentation.

| Operation | Description |
|---|---|
| **SPT Control Panel** | Opens the 6-tab SPT dock widget. |
| **Detect Particles** | Runs particle detection on the current stack. |
| **Link Particles** | Links detected particles into tracks across frames. |
| **Results Table** | Opens the SPT results table dock. |

## Export Video

**Process > Export Video** exports the current stack as an MP4 video file. Configure
frame rate, resolution, codec, and whether to include overlays.

## 4D Support

Most filters and operations support 4D data automatically via the `per_plane` decorator.
When applied to a 4D volume, the operation runs on each 2D plane independently (optionally
in parallel). This includes Gaussian blur, median filter, threshold, and most other
spatial operations.

## See Also

- [Image Menu](image_menu.md) -- Stack manipulation, measurement, overlays
- [AI Tools](ai_tools.md) -- AI-assisted processing
- [API Reference](api_reference.md) -- Using operations from the console
