"""hypy: arrival-based seismic event localization."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locator import EventLocator

__all__ = ["EventLocator"]


def __getattr__(name: str):
    if name == "EventLocator":
        from .locator import EventLocator

        # Caching imported EventLocator object to avoid expensive lookup
        globals()[name] = EventLocator
        return EventLocator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
