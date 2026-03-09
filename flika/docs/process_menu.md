# Process Menu

The Process menu contains image processing operations organized into submenus: Binary,
Math, Filters, Deconvolution, Structures, Segmentation, Detection, Colocalization,
Watershed, Stitching, SPT Analysis, Morphometry, Dynamics, Export, and more.

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

### Bleach Correction

**Process > Filters > Bleach Correction** corrects photobleaching artifacts in
time-series data. Three correction methods are available:

| Method | Description |
|---|---|
| **Exponential Fit** | Fits an exponential decay to the mean intensity trace and divides each frame by the fitted curve. Best for monotonic bleaching. |
| **Histogram Matching** | Adjusts each frame's histogram to match the first frame's distribution. Preserves the overall intensity profile. |
| **Ratio to Mean** | Divides each frame by its mean intensity and multiplies by the global mean. Simple normalization suitable for mild bleaching. |

```python
from flika.process.filters import bleach_correction
corrected = bleach_correction(method='Exponential Fit')
corrected = bleach_correction(method='Histogram Matching')
corrected = bleach_correction(method='Ratio to Mean')
```

## Deconvolution

**Process > Deconvolution** provides image deconvolution to reverse optical blur using
a point spread function (PSF) model.

| Operation | Description |
|---|---|
| **Richardson-Lucy** | Iterative maximum-likelihood deconvolution. Suitable for Poisson noise (fluorescence). Controls: PSF sigma, PSF size, number of iterations. |
| **Wiener Deconvolution** | Frequency-domain deconvolution using a Wiener filter. Fast, single-pass. Controls: PSF sigma, PSF size, noise variance. |

Both operations generate a PSF internally from two supported models:

| PSF Model | Description |
|---|---|
| **Gaussian** | Symmetric 2D Gaussian kernel. Parameterized by sigma and kernel size. |
| **Airy Disk** | Diffraction-limited PSF based on the Airy pattern. More physically accurate for optical microscopy. |

All deconvolution operations support 4D data via the `per_plane` decorator.

```python
from flika.process.deconvolution import richardson_lucy, wiener_deconvolution, generate_psf

# Generate a PSF manually
psf = generate_psf(model='gaussian', sigma=1.5, size=15)

# Richardson-Lucy iterative deconvolution
result = richardson_lucy(psf_sigma=1.5, psf_size=15, iterations=20)

# Wiener frequency-domain deconvolution
result = wiener_deconvolution(psf_sigma=1.5, psf_size=15, noise_var=0.01)
```

## Structures

**Process > Structures** provides structure detection, network analysis, and texture
characterization operations.

### Vesselness and Thinning

| Operation | Description |
|---|---|
| **Frangi Vesselness** | Multi-scale Frangi vesselness filter for detecting tubular structures (vessels, neurites, filaments). Uses Hessian eigenvalues to enhance elongated features. |
| **Skeletonize** | Morphological thinning of a binary image to a 1-pixel-wide skeleton. Preserves topology of the original shape. |
| **Medial Axis** | Computes the medial axis (topological skeleton) of a binary image, along with the distance transform to the boundary. |

```python
from flika.process.structures import frangi_vesselness, skeletonize_process, medial_axis_process
vessels = frangi_vesselness()
skeleton = skeletonize_process()
medial = medial_axis_process()
```

### Network Analysis (Submenu)

**Process > Structures > Network Analysis** contains operations for extracting and
analyzing graph-like structures from skeletonized images.

| Operation | Description |
|---|---|
| **Skeleton Analysis** | Extracts a graph from a skeleton image: identifies branch points, endpoints, traces individual segments, and measures branch lengths. Results are overlaid on the image and reported in a table. |
| **Hough Lines** | Detects straight lines using the probabilistic Hough transform. Returns line segments with start/end coordinates. Configurable threshold, minimum line length, and maximum gap. |
| **Hough Circles** | Detects circles using the Hough gradient method. Returns center coordinates and radii. Configurable radius range and sensitivity. |

```python
from flika.process.structures import skeleton_analysis, hough_lines, hough_circles
graph = skeleton_analysis()
lines = hough_lines()
circles = hough_circles()
```

### Corner and Texture Detection

| Operation | Description |
|---|---|
| **Corner Detection** | Detects corners using Harris or Shi-Tomasi methods. Harris uses the eigenvalue product of the structure tensor; Shi-Tomasi uses the minimum eigenvalue. Both return a corner response map. |
| **LBP Texture** | Computes Local Binary Pattern (LBP) texture descriptors. Encodes local intensity relationships around each pixel into a rotation-invariant code. Useful for texture classification. |
| **Structure Tensor** | Computes the structure tensor for each pixel, yielding coherency (anisotropy measure, 0-1) and orientation (angle) maps. Useful for analyzing fiber alignment and directional structures. |

```python
from flika.process.structures import corner_detection, local_binary_pattern_process, structure_tensor_analysis
corners = corner_detection(method='Harris')
lbp = local_binary_pattern_process()
tensor = structure_tensor_analysis()
```

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

## Watershed

**Process > Watershed** provides marker-controlled watershed segmentation for separating
touching objects, particularly useful after distance transform preprocessing.

| Operation | Description |
|---|---|
| **Distance Transform** | Computes the Euclidean distance from each foreground pixel to the nearest background pixel. Useful for measuring object thickness and as input to watershed. |
| **Watershed Segmentation** | Marker-controlled watershed on a distance or intensity image. Markers are generated from local maxima of the distance transform. Separates touching/overlapping objects into individually labeled regions. |

The typical workflow is:

1. Threshold the image to create a binary mask
2. Apply Distance Transform to get a distance map
3. Run Watershed Segmentation to split touching objects

```python
from flika.process.watershed import distance_transform, watershed_segmentation
dist = distance_transform()
labels = watershed_segmentation()
```

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

## Stitching

**Process > Stitch Images** combines two overlapping images (or stacks) into a single
seamless mosaic using subpixel registration.

| Feature | Description |
|---|---|
| **Phase Cross-Correlation** | Subpixel registration using phase correlation in the Fourier domain. Detects the translational offset between overlapping regions with 1/10 pixel accuracy. |
| **Overlap Percentage** | Configurable overlap fraction (default 10%). Determines the region used for registration. |
| **Linear Blending** | Smooth blending at seams using a linear ramp in the overlap zone. Eliminates visible seam artifacts. |
| **Direction** | Horizontal (side-by-side) or vertical (top-bottom) stitching. |
| **Stack Support** | Works on both 2D images and 3D time-series stacks. Each frame in a stack is stitched with the same computed offset. |

```python
from flika.process.stitching import stitch_images
# Stitch two windows horizontally with 15% overlap
stitched = stitch_images(direction='Horizontal', overlap=15)
```

## Dynamics

**Analyze > Dynamics** contains specialized analysis modules for time-resolved
microscopy experiments.

### FRAP Analysis

Fluorescence Recovery After Photobleaching analysis. Quantifies molecular mobility
from recovery curves.

| Feature | Description |
|---|---|
| **Double Normalization** | Normalizes the recovery curve so pre-bleach baseline = 1.0 and post-bleach minimum = 0.0. |
| **Single Exponential Fit** | Fits `I(t) = A * (1 - exp(-t/tau)) + offset` to extract the recovery time constant and mobile fraction. |
| **Double Exponential Fit** | Two-component recovery for systems with fast and slow mobile populations. |
| **Soumpasis Diffusion Model** | Fits the Soumpasis equation for uniform circular bleach spots to extract the diffusion coefficient. |

Results include: half-time of recovery, mobile fraction, immobile fraction, diffusion
coefficient (Soumpasis), and R-squared goodness of fit.

```python
from flika.process.frap import frap_analysis
frap_analysis(bleach_frame=10, pre_bleach_frames=5, model='Single Exponential')
```

### FRET Analysis

Forster Resonance Energy Transfer analysis between donor and acceptor channels.

| Feature | Description |
|---|---|
| **Apparent FRET Efficiency** | Pixel-wise `E = Ia / (Ia + Id)` after background subtraction. |
| **Corrected FRET** | Applies spectral bleed-through and direct excitation corrections. |
| **Stoichiometry** | Computes the donor fraction `S = (Id + Ia) / total` for each pixel. |
| **Histogram** | Displays distributions of FRET efficiency and stoichiometry values. |

```python
from flika.process.fret import fret_analysis
fret_analysis()
```

### Calcium Analysis

Calcium imaging analysis for detecting and quantifying calcium transients.

| Feature | Description |
|---|---|
| **dF/F Calculation** | Computes Delta-F/F0 from a baseline period. Supports mean, median, or percentile baseline methods. |
| **Event Detection** | Detects calcium transients using threshold crossing on the dF/F trace. Reports event times, amplitudes, durations, and inter-event intervals. |
| **Statistics** | Computes event frequency, mean amplitude, mean duration, area under curve, and peak dF/F for each ROI. |
| **Temporal Smoothing** | Optional Savitzky-Golay or moving average smoothing before event detection. |

```python
from flika.process.calcium import calcium_analysis
calcium_analysis(baseline_frames=50, threshold=3.0)
```

### Spectral Unmixing

Linear spectral unmixing for separating overlapping fluorophore signals.

| Feature | Description |
|---|---|
| **NNLS Unmixing** | Non-negative least squares: enforces physically meaningful (non-negative) abundance values. Solves `data = spectra @ abundances` per pixel. |
| **Least-Squares Unmixing** | Unconstrained linear least squares. Faster but may produce negative values. |
| **PCA Endmember Estimation** | Automatically estimates reference spectra from the data using Principal Component Analysis when reference spectra are not available. |

Output is a multi-channel abundance map with one channel per fluorophore component.

```python
from flika.process.spectral import spectral_unmixing
spectral_unmixing(method='nnls')
```

## Morphometry

**Process > Morphometry** (also accessible via **Analyze > Morphometry**) provides
quantitative shape and texture measurements for labeled regions.

### Region Properties

Computes geometric and intensity-based measurements for each labeled region in the image.

| Property | Description |
|---|---|
| **Area** | Number of pixels in the region. |
| **Perimeter** | Length of the region boundary. |
| **Centroid** | (y, x) coordinates of the region center of mass. |
| **Bounding Box** | Smallest rectangle enclosing the region. |
| **Major/Minor Axis** | Lengths of the major and minor axes of the best-fit ellipse. |
| **Eccentricity** | Ratio of focal distance to major axis length (0 = circle, 1 = line). |
| **Solidity** | Region area divided by convex hull area. |
| **Circularity** | `4 * pi * area / perimeter^2` (1.0 = perfect circle). |
| **Equivalent Diameter** | Diameter of a circle with the same area as the region. |
| **Extent** | Region area divided by bounding box area. |
| **Aspect Ratio** | Major axis / minor axis. |
| **Intensity Stats** | Mean, max, min, and standard deviation of intensity (when an intensity image is provided). |

### Haralick Texture

Computes GLCM (Gray-Level Co-occurrence Matrix) texture features for quantifying
spatial patterns in intensity images.

| Feature | Description |
|---|---|
| **Contrast** | Measures local intensity variation. High for rough textures. |
| **Correlation** | Measures linear dependency of gray levels on neighboring pixels. |
| **Energy** | Sum of squared GLCM elements. High for uniform textures. |
| **Homogeneity** | Measures closeness of GLCM element distribution to the diagonal. High for smooth textures. |

### Hu Moments

Computes the 7 Hu invariant moments for each region. These are rotation-invariant,
scale-invariant, and translation-invariant shape descriptors derived from central
moments. Useful for shape matching and classification.

```python
from flika.process.morphometry import morphometry_analysis
morphometry_analysis()
```

## Color Conversions

**Image > Color** provides color space conversions and grayscale extraction for
RGB images.

### Color Space Conversions

| Conversion | Description |
|---|---|
| **RGB to HSV** | Converts to Hue-Saturation-Value. Hue encodes color, saturation encodes purity, value encodes brightness. Useful for color-based segmentation. |
| **HSV to RGB** | Converts HSV back to RGB color space. |
| **RGB to LAB** | Converts to CIELAB (L*a*b*). Perceptually uniform: equal distances in LAB correspond to equal perceived color differences. |
| **LAB to RGB** | Converts LAB back to RGB color space. |
| **RGB to YCrCb** | Converts to luma (Y) and chroma (Cr, Cb) components. Used in video compression and skin-tone detection. |
| **YCrCb to RGB** | Converts YCrCb back to RGB color space. |

### Grayscale Conversion

Converts an RGB image to single-channel grayscale using one of three methods:

| Method | Description |
|---|---|
| **Luminance** | Weighted sum: `0.2126*R + 0.7152*G + 0.0722*B` (ITU-R BT.709). Matches human brightness perception. |
| **Average** | Simple mean: `(R + G + B) / 3`. |
| **Lightness** | `(max(R,G,B) + min(R,G,B)) / 2`. HSL lightness component. |

```python
from flika.process.color import convert_color_space, grayscale
hsv = convert_color_space(conversion='RGB to HSV')
gray = grayscale(method='Luminance')
```

## SPT Analysis

Single Particle Tracking operations. See [SPT Guide](spt_guide.md) for full documentation.

| Operation | Description |
|---|---|
| **SPT Control Panel** | Opens the 6-tab SPT dock widget. |
| **Detect Particles** | Runs particle detection on the current stack. |
| **Link Particles** | Links detected particles into tracks across frames. |
| **Results Table** | Opens the SPT results table dock. |

## Background Subtraction

**Process > Background Subtraction** removes background signal from image stacks using
one of three methods:

### Manual ROI

Uses the currently drawn ROI(s) to define background regions. The mean intensity within
the ROI is computed and subtracted from each pixel.

### Auto ROI

Automatically detects a background region using the dark-corner algorithm (ported from
the spt_batch_analysis plugin). Divides the image into quadrants, selects the dimmest
corner, and uses a region there as background. The detected region is drawn on the window
for verification.

### Statistical

Computes a background value from each frame's pixel distribution:

| Method | Description |
|---|---|
| **Mean** | Mean of all pixels |
| **Median** | Median of all pixels |
| **Mode** | Most frequent pixel value (histogram peak) |
| **5th Percentile** | Conservative estimate for sparse signals |
| **25th Percentile** | First quartile |

### Options

- **Per-frame** -- computes and subtracts a separate background for each frame
- **Whole-stack** -- computes a single background from the average projection

```python
from flika.process.background_sub import background_subtract
# Auto ROI, per-frame
background_subtract(method='Auto ROI', scope='Per Frame')
# Statistical median, whole stack
background_subtract(method='Statistical', stat_method='Median', scope='Whole Stack')
```

## Batch Export

**Process > Export > Batch Export** exports the current window's image data to disk in
bulk.

| Feature | Description |
|---|---|
| **TIFF** | Exports the full stack as a multi-page TIFF file (float32 or uint16). |
| **PNG** | Exports individual frames as numbered PNG files in a directory. |
| **NPY** | Exports the raw numpy array as a `.npy` file (preserves exact dtype and shape). |
| **Image Sequence** | For 3D stacks, exports each frame as a separate numbered image file. Useful for importing into other software. |

```python
from flika.process.export import batch_export
batch_export(format='TIFF', path='/path/to/output.tif')
batch_export(format='PNG', path='/path/to/output_dir/')
```

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
