"""Public API for hypy.locator.

The package exposes `EventLocator`.
Implementation intentionally avoids mixins/multiple inheritance.
"""

from .event_locator import EventLocator

__all__ = ["EventLocator"]
