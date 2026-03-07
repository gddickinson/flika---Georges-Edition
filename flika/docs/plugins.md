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

### Step 5: GUI Widgets

Flika provides several widgets for plugin GUIs:

| Widget | Purpose |
|---|---|
| `SliderLabel` | Numeric slider with text label |
| `CheckBox` | Boolean toggle |
| `ComboBox` | Dropdown selection |
| `WindowSelector` | Dropdown to select an open window |
| `FileSelector` | File browser dialog |

### Step 6: Test Your Plugin

```python
# Reload a plugin during development
import importlib
import my_plugin
importlib.reload(my_plugin)
```

## Plugin Manager

The Plugin Manager (**Plugins > Plugin Manager**) provides:

- List of installed plugins with version info
- Available plugins from the flika plugin repository
- Install, update, and uninstall operations
- Dependency checking and installation

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
