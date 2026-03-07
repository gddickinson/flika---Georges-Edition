# AI Tools

Flika integrates several AI/ML-powered tools for image analysis, accessible from the
**AI** menu. These tools require additional dependencies (PyTorch, specific model packages)
that are installed on demand.

## AI Menu Overview

| Menu Item | Description |
|---|---|
| **Generate Script** | AI-assisted script generation from natural language |
| **Generate Plugin** | AI-assisted plugin scaffolding |
| **AI Denoiser** | Deep learning image denoising |
| **Pixel Classifier** | Train and apply pixel-level classifiers |
| **Particle Localizer** | AI-based particle detection |
| **Segmentation > Cellpose** | Cell segmentation with Cellpose |
| **Segmentation > StarDist** | Star-convex polygon cell detection |
| **Segmentation > SAM Interactive** | Segment Anything Model interactive segmentation |
| **BioImage.IO Model Zoo** | Browse and load community models |

## AI Denoiser

The denoiser uses deep learning models to remove noise from microscopy images while
preserving structural details.

### Usage

1. **AI > AI Denoiser** to open the dialog
2. Select a denoising model (or use the default)
3. Configure parameters (patch size, overlap)
4. Click Apply

The denoiser processes the current window and creates a new window with the denoised result.

### Supported Models

Models are stored in `~/.FLIKA/models/` and can be downloaded from the Model Zoo.

## Pixel Classifier

Train a machine learning classifier to label pixels based on user-drawn examples.

### Workflow

1. Open the image to classify
2. **AI > Pixel Classifier** to open the classifier dialog
3. Draw ROIs on examples of each class (e.g., "cell", "background", "artifact")
4. Assign class labels to each ROI
5. Click **Train** -- the classifier learns from the labeled pixels
6. Click **Predict** -- applies the trained model to the full image

### Features Used

The pixel classifier extracts multi-scale features at each pixel:
- Gaussian blur at multiple sigmas
- Gradient magnitude
- Laplacian of Gaussian
- Structure tensor eigenvalues

The resulting feature vector is classified using the trained model.

```python
# Access the pixel classifier programmatically
g.m._ai_classify()
```

## Particle Localizer

AI-based particle detection that learns from user annotations.

### Workflow

1. Open a stack with particles
2. **AI > Particle Localizer** to open the dialog
3. Mark example particles using point ROIs
4. Train the localizer
5. Run detection on the full stack

The localizer is complementary to the traditional detection methods in
[Process > Detection](process_menu.md#detection) and integrates with the
[SPT pipeline](spt_guide.md).

## Segmentation

### Cellpose

[Cellpose](https://www.cellpose.org/) is a deep learning model for cell segmentation.

1. **AI > Segmentation > Cellpose**
2. Select a pre-trained model:
   - `cyto` -- general cytoplasm segmentation
   - `cyto2` -- improved cytoplasm model
   - `nuclei` -- nuclear segmentation
   - Custom trained models
3. Set the expected cell diameter
4. Run segmentation

The result is a labeled image where each cell has a unique integer ID.

**Requires:** `pip install cellpose`

### StarDist

[StarDist](https://github.com/stardist/stardist) detects cells using star-convex polygons.
Best suited for round or convex-shaped nuclei.

1. **AI > Segmentation > StarDist**
2. Select a pre-trained model:
   - `2D_versatile_fluo` -- fluorescence microscopy
   - `2D_versatile_he` -- H&E stained histology
   - Custom models
3. Configure probability and NMS thresholds
4. Run detection

**Requires:** `pip install stardist`

### SAM Interactive Segmentation

The [Segment Anything Model (SAM)](https://segment-anything.com/) provides interactive
segmentation guided by user clicks.

### Workflow

1. **AI > Segmentation > SAM Interactive**
2. The SAM dialog opens with the current image
3. **Click** on an object to generate a segmentation mask
4. **Shift+Click** to add positive prompts (include this region)
5. **Ctrl+Click** to add negative prompts (exclude this region)
6. The mask updates in real-time
7. Accept the mask to create a binary window

SAM requires downloading the model weights on first use. Weights are cached in
`~/.FLIKA/models/`.

**Requires:** `pip install segment-anything` and a SAM checkpoint

## BioImage.IO Model Zoo

**AI > BioImage.IO Model Zoo** opens a browser for the
[BioImage Model Zoo](https://bioimage.io/), a community repository of pre-trained
models for bioimage analysis.

### Features

- Browse available models by category (segmentation, denoising, detection)
- Download models directly into flika
- View model documentation and example outputs
- Load and apply models to the current image

Models follow the BioImage.IO model specification and can be shared across tools
(ImageJ, napari, QuPath, etc.).

## Script and Plugin Generation

### Generate Script

**AI > Generate Script** uses an LLM to generate Python scripts from natural language
descriptions. Describe what you want to do (e.g., "apply Gaussian blur then threshold
and count particles") and the AI generates a working flika script.

### Generate Plugin

**AI > Generate Plugin** scaffolds a complete plugin structure from a description,
including `info.xml`, the main module, and menu integration. See [Plugins](plugins.md)
for the plugin architecture.

### API Key Security

The Anthropic API key required for AI script/plugin generation is stored securely via
the system keyring (macOS Keychain / Windows Credential Manager / Linux Secret Service).
The key is **never** stored in plaintext in `settings.json`.

- Set the key in **File > Settings** (API Key field)
- Delete the key with the **Delete API Key** button
- Legacy plaintext keys are automatically migrated to the keyring on first access
- The key is never included in exported settings or provenance files

## GPU Acceleration

AI tools benefit significantly from GPU acceleration. Check your GPU status via
**Help > GPU/Acceleration Status**.

- CUDA GPUs (NVIDIA) provide the best performance for PyTorch-based tools
- MPS (Apple Silicon) is supported for many operations
- CPU fallback is always available

Configure the acceleration device in **File > Settings**:
- `Auto` -- automatically select the best available device
- `CPU` -- force CPU computation
- `CUDA` -- use NVIDIA GPU
- `MPS` -- use Apple Silicon GPU

```python
# Check current device
from flika.utils.accel import get_device
print(get_device())  # e.g., 'cuda:0', 'mps', 'cpu'
```

## See Also

- [Process Menu](process_menu.md#segmentation) -- Traditional segmentation algorithms
- [Process Menu](process_menu.md#detection) -- Traditional detection algorithms
- [SPT Guide](spt_guide.md) -- Single particle tracking pipeline
- [Troubleshooting](troubleshooting.md#gpu-acceleration) -- GPU setup issues
