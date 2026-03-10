"""Extensible file-format registry for flika."""
from flika.io.registry import FormatRegistry, FormatHandler, registry
from flika.io.lazy import LazyArray, is_lazy

__all__ = ['FormatRegistry', 'FormatHandler', 'registry', 'LazyArray', 'is_lazy']
