"""Extensible file-format registry for flika."""
from .registry import FormatRegistry, FormatHandler, registry
from .lazy import LazyArray, is_lazy

__all__ = ['FormatRegistry', 'FormatHandler', 'registry', 'LazyArray', 'is_lazy']
