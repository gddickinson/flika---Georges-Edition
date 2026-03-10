# Microscopy Analysis

Flika provides dedicated analysis modules for common quantitative microscopy techniques.
Each module is accessible through the **Process** menu and integrates with Flika's
undo/redo, macro recording, and metadata systems.

---

## FRAP Analysis

**Process > Dynamics > FRAP**

Fluorescence Recovery After Photobleaching (FRAP) measures molecular mobility by
monitoring fluorescence recovery in a bleached region over time.

### Workflow

1. **Draw an ROI** on the bleached region in the current window.
2. **Run FRAP analysis** from the menu. Flika extracts the mean intensity time trace
   from the ROI.
3. **Normalization** -- The trace is normalized so that the pre-bleach baseline equals
   1.0 and the first post-bleach frame equals 0.0. This removes intensity variation
   unrelated to recovery.
4. **Curve fitting** -- Choose a recovery model:
   - **Single exponential**: `F(t) = A * (1 - exp(-t / tau))` -- suitable for simple
     free diffusion with a single recovery rate.
   - **Double exponential**: `F(t) = A1 * (1 - exp(-t / tau1)) + A2 * (1 - exp(-t / tau2))`
     -- accounts for a fast and a slow recovery component (e.g., free and bound fractions).
   - **Soumpasis diffusion model**: Analytical solution for 2D diffusion into a circular
     bleach spot, parameterized by the diffusion time tau_D.
5. **Results** -- The fit returns:
   - **Half-time (t_1/2)** -- Time to reach 50% recovery.
   - **Mobile fraction** -- Plateau value of the fit; indicates the fraction of molecules
     free to diffuse.
   - **Immobile fraction** -- `1 - mobile fraction`.
   - **Diffusion coefficient** (Soumpasis model) -- Calculated from `D = r^2 / (4 * tau_D)`
     where r is the bleach spot radius.

### Scripting

```python
from flika.process.frap import frap_analysis
frap_analysis.run(model='single_exp')
```

---

## FRET Analysis

**Process > Dynamics > FRET**

Forster Resonance Energy Transfer (FRET) measures nanometer-scale distances between
donor and acceptor fluorophores based on non-radiative energy transfer efficiency.

### Measurements

- **Apparent FRET efficiency**: `E_app = I_A / (I_D + I_A)` where I_D and I_A are donor
  and acceptor intensities. This is a raw ratio that includes spectral cross-talk.
- **Corrected FRET efficiency**: Accounts for three cross-talk coefficients:
  - **a** -- Donor bleed-through into the acceptor channel.
  - **b** -- Direct excitation of the acceptor by the donor excitation wavelength.
  - **d** -- Detection efficiency ratio between channels.
  The corrected efficiency removes these artifacts to yield a value proportional to the
  true energy transfer rate.
- **Stoichiometry (S)**: `S = (I_D + I_A) / (I_D + I_A + I_AA)` where I_AA is the
  acceptor emission under acceptor excitation. Stoichiometry distinguishes donor-only,
  acceptor-only, and FRET-active populations.

### E-S Histogram

The FRET module generates a 2D histogram of Efficiency vs. Stoichiometry. This plot
is the standard visualization for single-molecule or pixel-level FRET data, allowing
identification of distinct molecular populations by their position in E-S space.

### Scripting

```python
from flika.process.fret import fret_analysis
fret_analysis.run(donor_window=win_donor, acceptor_window=win_acceptor,
                  cross_talk_a=0.05, cross_talk_b=0.02, cross_talk_d=1.0)
```

---

## Calcium Imaging

**Process > Dynamics > Calcium**

Calcium imaging tracks intracellular calcium concentration changes over time using
fluorescent indicators (e.g., GCaMP, Fura-2, Fluo-4).

### dF/F Calculation

The change in fluorescence relative to baseline (dF/F) is the standard metric:

`dF/F = (F(t) - F0) / F0`

**Baseline selection**: Flika computes F0 as the mean intensity over a user-specified
range of frames (typically a quiet period before stimulation). The baseline window is
set in the dialog or via scripting.

### Event Detection

Calcium events (transients) are detected by thresholding the dF/F trace:

- A **threshold** (in units of standard deviations above baseline noise or an absolute
  dF/F value) defines the minimum amplitude for an event.
- Events are segmented by finding contiguous above-threshold regions.
- Adjacent events closer than a configurable **minimum inter-event interval** are merged.

### Event Statistics

For each detected event, the module reports:

| Statistic | Definition |
|---|---|
| **Amplitude** | Peak dF/F value of the event |
| **Duration** | Time from onset to return to baseline |
| **Rise time** | Time from onset to peak |
| **Decay time** | Time from peak to return below threshold |
| **Frequency** | Number of events per unit time |
| **Inter-event interval** | Time between consecutive event onsets |

### Temporal Smoothing

An optional smoothing step (Savitzky-Golay or moving average) can be applied to the
dF/F trace before event detection to reduce noise-driven false positives.

### Scripting

```python
from flika.process.calcium import calcium_analysis
calcium_analysis.run(baseline_frames=(0, 50), threshold=3.0, method='std')
```

---

## Spectral Unmixing

**Process > Dynamics > Spectral Unmixing**

Spectral unmixing separates overlapping fluorophore contributions in multi-channel or
hyperspectral image data, producing individual abundance maps for each fluorophore.

### Reference Spectra

Provide reference emission spectra for each fluorophore in the sample. These can be:

- Measured from single-labeled control samples.
- Loaded from a file (CSV or text).
- Estimated automatically using PCA endmember extraction (see below).

### Unmixing Methods

| Method | Description |
|---|---|
| **NNLS** | Non-Negative Least Squares. Constrains all abundance values to be non-negative, which is physically meaningful for fluorophore concentrations. This is the recommended default. |
| **Least Squares** | Ordinary least squares without non-negativity constraint. Faster but may produce negative abundance values in noisy regions. |

### PCA Endmember Estimation

When reference spectra are not available, the module can estimate endmembers directly
from the data using Principal Component Analysis:

1. PCA is applied to the spectral dimension of the image stack.
2. The user selects the number of components (endmembers).
3. The extracted components serve as reference spectra for unmixing.

This approach works best when the fluorophores have sufficiently distinct emission spectra.

### Scripting

```python
from flika.process.spectral import spectral_unmixing
spectral_unmixing.run(reference_spectra=spectra_array, method='nnls')
```

---

## Morphometry

**Process > Morphometry**

Morphometry extracts quantitative shape and texture measurements from labeled regions
in binary or segmented images.

### Region Properties Table

For each labeled region in the image, Flika computes a table of geometric properties:

| Property | Description |
|---|---|
| **Area** | Number of pixels in the region |
| **Perimeter** | Length of the region boundary |
| **Centroid** | (x, y) center of mass |
| **Bounding box** | Smallest enclosing rectangle |
| **Eccentricity** | Ratio of focal distance to major axis length (0 = circle, 1 = line) |
| **Solidity** | Area divided by convex hull area |
| **Orientation** | Angle of the major axis relative to horizontal |
| **Major/Minor axis** | Lengths of the fitted ellipse axes |

The properties table is displayed in a sortable, exportable dialog and is also stored
in `window.metadata['morphometry']`.

### Haralick Texture Features

Texture features are computed from the Gray-Level Co-occurrence Matrix (GLCM) for each
region:

- **Angular Second Moment (Energy)** -- Measures texture uniformity.
- **Contrast** -- Measures local intensity variation.
- **Correlation** -- Measures linear dependency of gray levels on neighboring pixels.
- **Entropy** -- Measures randomness of the intensity distribution.
- **Homogeneity (Inverse Difference Moment)** -- Measures local homogeneity.

GLCM features are computed at multiple angles (0, 45, 90, 135 degrees) and averaged
for rotational invariance.

### Hu Moments

Seven Hu moment invariants are calculated for each region. These moments are invariant
to translation, rotation, and scale, making them useful for shape classification and
matching across different images:

- **hu[0]--hu[1]** -- Related to the spread and symmetry of the shape.
- **hu[2]--hu[3]** -- Capture triangularity and elongation.
- **hu[4]--hu[6]** -- Higher-order shape descriptors sensitive to skewness and chirality.

Hu moments enable comparing shapes regardless of their position, size, or orientation
in the image.

### Scripting

```python
from flika.process.morphometry import morphometry_analysis
morphometry_analysis.run(compute_texture=True, compute_moments=True)
```
