"""Public package entry point for hypy.

Keep the top-level import lightweight to avoid circular imports when
importing submodules (e.g., ``hypy.logger``). Exports such as
``EventLocator`` are provided via lazy attribute access.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locator import EventLocator  # pragma: no cover

__all__ = ["EventLocator"]


def __getattr__(name: str):
    if name == "EventLocator":
        from .locator import EventLocator

        return EventLocator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
