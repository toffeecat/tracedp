"""Registry owned by post-interruption tracers, not action methods."""

from lh_platform.core.registry import Registry

TRACERS = Registry("tracer")

__all__ = ["TRACERS"]
