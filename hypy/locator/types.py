from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal, Sequence

import numpy as np
from obspy.core.event import Arrival, Catalog, Event, Origin  # type: ignore[import-untyped]
from obspy.core.event.base import QuantityError  # type: ignore[import-untyped]
from obspy.core.event.origin import ConfidenceEllipsoid, OriginUncertainty  # type: ignore[import-untyped]
from obspy.core.inventory import Inventory  # type: ignore[import-untyped]

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
