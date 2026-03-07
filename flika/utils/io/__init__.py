# Backwards-compatibility shim: the vendored tifffile has been replaced by
# the PyPI ``tifffile`` package.  Existing code that does
#     from flika.utils.io import tifffile
# will continue to work transparently.
import tifffile  # noqa: F401
