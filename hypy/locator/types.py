from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from obspy.core.event import (  # type: ignore[import-untyped]
    Arrival,
    Catalog,
    Event,
    Origin,
)
from obspy.core.event.base import QuantityError  # type: ignore[import-untyped]
from obspy.core.event.origin import (  # type: ignore[import-untyped]
    ConfidenceEllipsoid,
    OriginUncertainty,
)
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

DimLiteral = Literal[4]
ModeLiteral = Literal["table", "ray"]
EngineLiteral = Literal["taup", "pyrocko"]
SolverLiteral = Literal["lsqr", "svd"]

# Re-exports for convenience
__all__ = [
    "Path",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "Literal",
    "Sequence",
    "np",
    "Arrival",
    "Catalog",
    "Event",
    "Origin",
    "QuantityError",
    "ConfidenceEllipsoid",
    "OriginUncertainty",
    "Inventory",
    "DimLiteral",
    "ModeLiteral",
    "EngineLiteral",
]
