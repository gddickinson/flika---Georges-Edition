# Plugins

Flika has a plugin system that allows extending the application with custom processing
operations, visualizations, and analysis tools. Plugins integrate into the menu bar and
have full access to flika's API.

## Plugin Architecture

Each plugin is a Python package stored in `~/.FLIKA/plugins/`. A plugin package contains:

```
~/.FLIKA/plugins/
    my_plugin/
        __init__.py        # Plugin entry point
        info.xml           # Plugin metadata and menu layout
        description.html   # User-facing description (optional)
        my_module.py       # Additional modules
```

## Installing Plugins

### From the Plugin Manager

1. Go to **Plugins > Plugin Manager**
2. Browse the list of available plugins
3. Click **Install** next to the desired plugin
4. Restart flika (or the plugin loads automatically)

### Manual Installation

1. Clone or download the plugin repository
2. Place it in `~/.FLIKA/plugins/`
3. Restart flika

```bash
cd ~/.FLIKA/plugins/
git clone https://github.com/flika-org/my_plugin.git
```

### From a URL

The Plugin Manager can install from a Git URL:

1. **Plugins > Plugin Manager**
2. Enter the repository URL
3. Click Install

## Using Plugins

Installed plugins appear in the **Plugins** menu. Each plugin creates its own submenu
based on the `menu_layout` in `info.xml`.

Plugins can also be called from the console:

```python
from flika.plugins.my_plugin import my_function
result = my_function(param1=10, param2='value')
```

## Writing a Plugin

### Step 1: Create the Package

Create a new directory in `~/.FLIKA/plugins/` with the following structure:

```
my_plugin/
    __init__.py
    info.xml
    description.html
```

### Step 2: Write info.xml

The `info.xml` file defines the plugin's metadata and menu structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<plugin name="My Plugin">
    <directory>my_plugin</directory>
    <version>1.0.0</version>
    <author>Your Name</author>
    <url>https://github.com/your-repo/my_plugin</url>
    <description>Short description of the plugin.</description>
    <dependencies>
        <dependency name="numpy"/>
        <dependency name="scipy"/>
        <dependency name="custom-package"/>
        <dependency name="optional-dep" optional="true"/>
    </dependencies>
    <menu_layout>
        <action location="__init__" function="my_operation.gui">My Operation</action>
        <action location="__init__" function="another_operation.gui">Another Operation</action>
    </menu_layout>
</plugin>
```

#### info.xml Fields

| Field | Required | Description |
|---|---|---|
| `name` (attribute on `<plugin>`) | Yes | Display name in the Plugin Manager |
| `directory` | Yes | Package directory name (must match folder name) |
| `version` | Yes | Version string (e.g. `1.0.0` or `2025.01.15`) |
| `author` | No | Author name |
| `url` | No | Repository URL (for download/update) |
| `description` | No | Short description text |
| `dependencies` | No | List of required pip packages (`<dependency name="..."/>`) |
| `menu_layout` | Yes | Menu actions with `location`, `function`, and display text |

#### Dependency Tags

The `<dependencies>` section lists packages that must be installed for the plugin to work.
Each `<dependency>` has a `name` attribute matching the pip package name. The Plugin Manager
checks these before loading and can auto-install missing dependencies. Use `optional="true"`
for packages that enhance functionality but are not strictly required.

```xml
<dependencies>
    <dependency name="scikit-learn"/>
    <dependency name="tensorflow"/>
    <dependency name="joblib" optional="true"/>
</dependencies>
```

You can verify all plugin dependencies via **Help > Check Plugin Dependencies**.

### Step 3: Write the Plugin Code

The `__init__.py` (or referenced modules) should define functions that are referenced
in the `menu_layout`:

```python
"""My Plugin - Example flika plugin."""
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox
from flika import global_vars as g
from flika.window import Window
import numpy as np


class MyOperation(BaseProcess):
    """Example processing operation."""

    def __init__(self):
        super().__init__()

    def gui(self):
        """Show the GUI dialog."""
        self.gui_reset()
        sigma = SliderLabel(decimals=1)
        sigma.setRange(0.1, 50.0)
        sigma.setValue(2.0)
        self.items.append({
            'name': 'sigma',
            'string': 'Sigma',
            'object': sigma,
        })
        super().gui()

    def __call__(self, sigma=2.0, keepSourceWindow=False):
        """Run the operation."""
        self.start(keepSourceWindow)
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(g.win.image.astype(float), sigma=sigma)
        self.newtif = result
        self.newname = f'{g.win.name} - MyOp(sigma={sigma})'
        return self.end()

my_operation = MyOperation()
```

### Step 4: Using BaseProcess

Inherit from `BaseProcess` to get:

- **GUI dialog generation** from `self.items`
- **Undo/redo support** via the UndoStack
- **Macro recording** for reproducibility
- **Automatic window management** (creating result windows)

Key `BaseProcess` methods:

| Method | Purpose |
|---|---|
| `gui()` | Show the parameter dialog |
| `__call__()` | Execute the operation |
| `start(keepSourceWindow)` | Begin processing (stores state for undo) |
| `end()` | Finish processing (creates new window, records macro) |
| `gui_reset()` | Clear previous GUI items |

#### BaseProcess_noPriorWindow

For operations that create a new window without requiring an existing one (e.g.,
simulations, file generators), inherit from `BaseProcess_noPriorWindow` instead.

**Important:** `BaseProcess_noPriorWindow` plugins MUST still call `self.start()` with
no arguments. This sets `self.command`, which `self.end()` needs to record the macro
and complete the operation. Omitting `self.start()` will cause `self.end()` to fail.

```python
from flika.utils.BaseProcess import BaseProcess_noPriorWindow

class MyGenerator(BaseProcess_noPriorWindow):
    def __call__(self, width=256, height=256):
        self.start()  # Required -- sets self.command
        self.newtif = np.random.rand(height, width).astype(np.float32)
        self.newname = 'Random Image'
        return self.end()
```

### Step 5: GUI Widgets

Flika provides several widgets for plugin GUIs:

| Widget | Purpose |
|---|---|
| `SliderLabel` | Numeric slider with text label |
| `CheckBox` | Boolean toggle |
| `ComboBox` | Dropdown selection |
| `WindowSelector` | Dropdown to select an open window |
| `FileSelector` | File browser dialog |

#### Convenience Methods

BaseProcess provides convenience methods for building plugin GUIs more concisely. Use
these inside `gui()` after calling `self.gui_reset()`:

```python
def gui(self):
    self.gui_reset()
    self.add_slider('name', 'Label', value, lo, hi, decimals=2)
    self.add_slider_odd('name', 'Label', value, lo, hi)
    self.add_checkbox('name', 'Label', checked=False)
    self.add_combo('name', 'Label', ['Option A', 'Option B'], default=0)
    self.add_window('name', 'Label')
    super().gui()
```

| Method | Purpose |
|---|---|
| `add_slider(name, label, value, lo, hi, decimals=2)` | Add a numeric slider with range and precision |
| `add_slider_odd(name, label, value, lo, hi)` | Add a slider constrained to odd values (useful for kernel sizes) |
| `add_checkbox(name, label, checked=False)` | Add a boolean checkbox |
| `add_combo(name, label, options, default=0)` | Add a dropdown with a list of options |
| `add_window(name, label)` | Add a window selector dropdown |

These methods handle creating the widget, setting its range/value, and appending it to
`self.items` automatically.

### Step 6: Test Your Plugin

```python
# Reload a plugin during development
import importlib
import my_plugin
importlib.reload(my_plugin)
```

## AI Plugin Generator

Flika includes an AI-powered plugin generator accessible via **AI > Claude > Generate Plugin**.
This tool can create complete, working plugins from a natural language description.

### Workflow

1. Open **AI > Claude > Generate Plugin**
2. Enter a **plugin name** and a **description** of what the plugin should do
3. Click **Generate** -- the AI creates the full plugin code
4. **Review the code preview** in the dialog to verify correctness
5. Click **Save & Load Plugin** to install it immediately into `~/.FLIKA/plugins/`

The generated plugin is ready to use without restarting flika.

### Structural Validation

Before saving, the generator validates the plugin structure to ensure it follows flika
conventions. It checks for:

- `gui_reset()` call in the `gui()` method
- `super().gui()` call to display the dialog
- `self.start()` and `self.end()` calls in the `__call__` method
- A module-level instance (e.g., `my_operation = MyOperation()`) required for menu binding

### Auto-Fix

The generator can automatically fix common structural issues:

- **Missing module-level instance** -- adds the instance assignment at the end of the file
- **Missing `self.start()`** -- inserts the call at the beginning of `__call__`

### Plugin Guidelines

The AI uses a guidelines file at `~/.FLIKA/plugin_guidelines.md` to inform code generation.
You can edit this file to customize the style, conventions, or patterns the AI follows
when generating plugins for your workflow.

### Code Safety

The standard code safety policy applies to AI-generated plugins. Review the generated
code before loading it, especially if the plugin performs file I/O or network operations.

## Plugin Manager

The Plugin Manager (**Plugins > Plugin Manager**) provides:

- List of installed plugins with version info
- Available plugins from the flika plugin repository
- Install, update, and uninstall operations
- Dependency checking and installation
- **Enable/disable individual plugins** -- toggle plugins on or off without uninstalling them
- **Reload Plugins** button -- reload all plugins without restarting flika
- **Suppress startup messages** -- option to hide plugin loading messages at startup
- **GitHub repository browser** -- browse and install plugins directly from GitHub repositories

### Enable/Disable Plugins

Each installed plugin has an enable/disable toggle in the Plugin Manager. Disabled plugins
remain in `~/.FLIKA/plugins/` but are not loaded at startup and do not appear in the
Plugins menu. Re-enable a plugin and click **Reload Plugins** to activate it without
restarting.

### Reload Plugins

The **Reload Plugins** button re-scans the plugins directory and reloads all enabled
plugins. This is useful during development or after enabling/disabling plugins -- no
restart is needed.

### Suppress Startup Messages

Check **Suppress startup messages** to hide the per-plugin loading messages that appear
in the console when flika starts. Errors and warnings are still shown.

### GitHub Repository Browser

The Plugin Manager includes a GitHub repository browser that lets you search for and
install plugins hosted on GitHub. Enter a search term or repository URL to find plugins,
preview their descriptions, and install them directly.

### Checking Dependencies

**Help > Check Plugin Dependencies** verifies that all installed plugins have their
required dependencies satisfied. Missing packages are reported with install instructions.

## Important Notes

- Plugins must import from `flika`, not renamed packages
- All plugins share the same Python environment as flika
- Plugin settings can be stored using `g.settings` with a namespaced key
- Avoid modifying flika internals directly; use the public API

## Example Plugins

The flika community provides several plugins:

- **detect_puffs** -- Calcium puff detection and analysis
- **loinger_flika** -- Loinger analysis tools
- **pynsight** -- Super-resolution analysis

Browse available plugins at [flika-org.github.io](http://flika-org.github.io).

## See Also

- [API Reference](api_reference.md) -- BaseProcess and Window APIs
- [AI Tools](ai_tools.md#generate-plugin) -- AI-assisted plugin generation
- [User Interface](user_interface.md#menu-bar) -- Where plugins appear in the UI
