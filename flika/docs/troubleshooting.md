# Troubleshooting

This guide covers common issues, diagnostic tools, and solutions for flika.

## Dependency Checking

### Core Dependencies

**Help > Check Core Dependencies** verifies that all required packages are installed and
at compatible versions. This checks:

- qtpy, PyQt6 (GUI framework)
- pyqtgraph (image display)
- numpy, scipy (computation)
- scikit-image (image processing)
- tifffile (TIFF I/O)
- dask (lazy loading)

If any dependency is missing or outdated, the dialog reports which packages need updating.

### Plugin Dependencies

**Help > Check Plugin Dependencies** checks requirements for all installed plugins. Each
plugin's `info.xml` declares its dependencies, and this tool verifies they are satisfied.

```bash
# Manually install a missing dependency
pip install missing-package
```

## GPU Acceleration

### Checking GPU Status

**Help > GPU/Acceleration Status** shows:

- Detected devices (CUDA GPUs, Apple MPS, CPU)
- Currently selected device
- Available GPU memory
- PyTorch availability and version

### Common GPU Issues

**No GPU detected:**
- Verify PyTorch is installed with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- On Apple Silicon, MPS is available with PyTorch 2.0+
- Check `torch.cuda.is_available()` or `torch.backends.mps.is_available()`

**Out of GPU memory:**
- Reduce `g.settings['gpu_memory_limit']` to limit usage
- Process smaller image regions
- Use CPU fallback: `g.settings['acceleration_device'] = 'CPU'`

**Wrong device selected:**
```python
# Force CPU
g.settings['acceleration_device'] = 'CPU'

# Force CUDA
g.settings['acceleration_device'] = 'CUDA'

# Auto-detect best device
g.settings['acceleration_device'] = 'Auto'
```

### Device Detection

```python
from flika.utils.accel import get_device
print(get_device())  # Shows which device is active
```

## Common Issues

### Application Won't Start

**Symptom:** Flika crashes or hangs on startup.

**Solutions:**
1. Check Python version: `python --version` (requires 3.12)
2. Verify PyQt6 installation: `python -c "from PyQt6 import QtWidgets"`
3. Reset settings: delete `~/.FLIKA/settings.json` and restart
4. Check logs: `~/.FLIKA/logs/` for error messages
5. Run in debug mode:
   ```python
   import flika
   flika.start_flika()
   # Then in console:
   g.settings['debug_mode'] = True
   ```

### Images Not Displaying

**Symptom:** Window opens but image is blank or all black.

**Solutions:**
1. Check the data range: `print(g.win.image.min(), g.win.image.max())`
2. Click **LUT norm** to auto-adjust contrast
3. Check for NaN values: `import numpy as np; print(np.isnan(g.win.image).any())`
4. Verify the array shape: `print(g.win.image.shape)` -- should be (T, X, Y) or (X, Y)

### Large File Performance

**Symptom:** Opening large files is slow or causes memory errors.

**Solutions:**
1. Use OME-Zarr or HDF5 formats for lazy loading
2. Reduce data type: `change_datatype('float32')` instead of float64
3. Crop the region of interest before processing
4. Enable multiprocessing: `g.settings['multiprocessing'] = True`
5. Increase available memory or use 64-bit Python

### ROIs Not Responding

**Symptom:** Cannot draw or interact with ROIs.

**Solutions:**
1. Check the mouse mode: `print(g.settings['mousemode'])`
2. Reset to rectangle: `g.settings['mousemode'] = 'rectangle'`
3. Ensure the image window is focused (click on it)
4. Check if the window has data: `print(g.win.image is not None)`

### Plugin Loading Errors

**Symptom:** Plugin fails to load or menu items are missing.

**Solutions:**
1. Check **Help > Check Plugin Dependencies** for missing packages
2. Verify plugin directory structure: `ls ~/.FLIKA/plugins/my_plugin/`
3. Ensure `info.xml` exists and is valid
4. Check for import errors in the console:
   ```python
   import importlib
   importlib.import_module('flika.plugins.my_plugin')
   ```
5. Verify the plugin imports `flika` (not a renamed package)

### PyQt6 Compatibility

**Symptom:** Errors mentioning `sip`, `QtWidgets`, or enum values.

Flika uses `qtpy` to bridge PyQt5 and PyQt6. Common issues:

1. **Enum changes:** PyQt6 requires fully-qualified enums (e.g.,
   `Qt.AlignmentFlag.AlignCenter` not `Qt.AlignCenter`). qtpy handles most of these.
2. **Missing `exec_`:** PyQt6 uses `exec()` not `exec_()`. qtpy handles this.
3. **Signal/slot changes:** Some signal signatures changed between PyQt5 and PyQt6.

### Undo Not Working

**Symptom:** Ctrl+Z does nothing or causes an error.

**Solutions:**
1. Verify the undo stack has entries:
   ```python
   from flika.core.undo import undo_stack
   print(len(undo_stack._stack))
   ```
2. Some operations (like direct array manipulation) are not undoable
3. Use `duplicate()` before destructive operations as a manual backup

## Headless Mode Issues

### Display Required Error

**Symptom:** Error about QApplication or display when running headless.

**Solution:**
```python
import flika
flika.start_flika(headless=True)  # Must be called before any other flika imports
```

On Linux servers without a display, set a virtual framebuffer:
```bash
export QT_QPA_PLATFORM=offscreen
python my_script.py
```

Or use Xvfb:
```bash
xvfb-run python my_script.py
```

## API Key Issues

### Key Not Found

If AI features report "No API key found":

1. Open **File > Settings** and enter your Anthropic API key
2. The key is stored in the system keyring (macOS Keychain / Windows Credential Manager)
3. Alternatively, set the environment variable: `export ANTHROPIC_API_KEY=sk-...`

### Removing the API Key

Use the **Delete API Key** button in Settings to securely remove the key from the
system keyring. This is recommended before sharing your machine or environment.

### Legacy Plaintext Keys

If you previously stored an API key in `settings.json`, it will be automatically
migrated to the system keyring on next access and removed from the JSON file.

## Logging

Flika logs to `~/.FLIKA/logs/`. Enable debug logging for detailed diagnostics:

```python
g.settings['debug_mode'] = True
```

When debug mode is enabled, the flika logger is set to DEBUG level, providing verbose
output including module loading, operation parameters, and timing information. When
disabled, the logger reverts to WARNING level.

Log files contain:
- Startup sequence and module loading
- Operation execution with parameters
- Error tracebacks
- Plugin loading status

### Reading Logs

```python
import os
log_dir = os.path.expanduser('~/.FLIKA/logs/')
# List recent log files
print(os.listdir(log_dir))
```

## Reporting Bugs

If you encounter a bug:

1. Check the console for error messages
2. Enable debug mode and reproduce the issue
3. Collect the log file from `~/.FLIKA/logs/`
4. Note your environment:
   ```python
   import sys, numpy, scipy, pyqtgraph
   print(f"Python: {sys.version}")
   print(f"NumPy: {numpy.__version__}")
   print(f"SciPy: {scipy.__version__}")
   print(f"pyqtgraph: {pyqtgraph.__version__}")
   ```
5. Report at the flika GitHub repository or via **Help > Documentation**

## Performance Tips

- Use `float32` instead of `float64` to halve memory usage
- Close unused windows to free memory: `g.windows[0].close()`
- Enable multiprocessing for filter operations on large stacks
- Use pixel binning to reduce data size before analysis
- For 4D data, process one volume at a time when possible

## See Also

- [Getting Started](getting_started.md) -- Installation and setup
- [File Operations](file_operations.md) -- File format support
- [Plugins](plugins.md) -- Plugin troubleshooting
- [AI Tools](ai_tools.md#gpu-acceleration) -- GPU setup for AI features
