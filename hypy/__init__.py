"""hypy: arrival-based seismic event localization."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .locator import EventLocator
    from .utils.io import read_quakeml

__all__ = ["EventLocator", "read_quakeml"]


def __getattr__(name: str):
    if name == "EventLocator":
        from .locator import EventLocator

        # Caching imported EventLocator object to avoid expensive lookup
        globals()[name] = EventLocator
        return EventLocator

    if name == "read_quakeml":
        from .utils.io import read_quakeml

        # Caching imported read_quakeml object to avoid expensive lookup
        globals()[name] = read_quakeml
        return read_quakeml

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
