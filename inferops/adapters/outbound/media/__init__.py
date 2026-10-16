"""Explicit media adapters; importing this package creates no resources."""
from .encoder import OpenCVMediaEncoder
from .spool import CrashSafeSpool, SpoolCapacityError, SpoolCorruptError

__all__ = ["CrashSafeSpool", "OpenCVMediaEncoder", "SpoolCapacityError", "SpoolCorruptError"]

