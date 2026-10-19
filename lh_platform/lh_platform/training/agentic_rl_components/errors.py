"""Shared fail-closed errors for agentic-RL component adapters."""

from __future__ import annotations


class AgenticRLComponentError(ValueError):
    """Raised when a built-in component lacks trustworthy runtime evidence."""
