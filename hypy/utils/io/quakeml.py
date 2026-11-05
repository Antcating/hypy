"""QuakeML reader for the fields the locator uses.

ObsPy's reader runs an XPath query per QuakeML field and builds an object for
every arrival. This walks the file once, keeps origins, magnitudes, and picks,
and skips arrivals.
"""

from xml.etree import ElementTree as ET

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin, Pick, WaveformStreamID


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _text(element: ET.Element, name: str) -> str | None:
    node = _first(element, name)
    if node is None:
        return None
    valued = _first(node, "value")
    text = valued.text if valued is not None else node.text
    if text is None:
        return None
    text = text.strip()
    return text or None


def _utc(text: str | None) -> UTCDateTime | None:
    if text is None:
        return None
    return UTCDateTime(text)


def _float(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text)


def _pick(element: ET.Element) -> Pick:
    waveform = _first(element, "waveformID")
    network = station = ""
    if waveform is not None:
        network = waveform.get("networkCode") or ""
        station = waveform.get("stationCode") or ""
    return Pick(
        resource_id=element.get("publicID"),
        time=_utc(_text(element, "time")),
        waveform_id=WaveformStreamID(network_code=network, station_code=station),
        phase_hint=_text(element, "phaseHint"),
        force_resource_id=False,
    )


def _origin(element: ET.Element) -> Origin:
    return Origin(
        resource_id=element.get("publicID"),
        time=_utc(_text(element, "time")),
        latitude=_float(_text(element, "latitude")),
        longitude=_float(_text(element, "longitude")),
        depth=_float(_text(element, "depth")),
        evaluation_mode=_text(element, "evaluationMode"),
        force_resource_id=False,
    )


def _magnitude(element: ET.Element) -> Magnitude:
    return Magnitude(
        resource_id=element.get("publicID"),
        mag=_float(_text(element, "mag")),
        magnitude_type=_text(element, "type"),
        force_resource_id=False,
    )


def _event(element: ET.Element) -> Event:
    origins = []
    picks = []
    magnitudes = []
    preferred_origin = None
    preferred_magnitude = None
    for child in element:
        name = _local(child.tag)
        if name == "origin":
            origins.append(_origin(child))
        elif name == "pick":
            picks.append(_pick(child))
        elif name == "magnitude":
            magnitudes.append(_magnitude(child))
        elif name == "preferredOriginID" and child.text:
            preferred_origin = child.text.strip()
        elif name == "preferredMagnitudeID" and child.text:
            preferred_magnitude = child.text.strip()
    return Event(
        resource_id=element.get("publicID"),
        preferred_origin_id=preferred_origin,
        preferred_magnitude_id=preferred_magnitude,
        origins=origins,
        magnitudes=magnitudes,
        picks=picks,
        force_resource_id=False,
    )


def read_quakeml(path: str) -> Catalog:
    """Read origins, magnitudes, and picks from a QuakeML file."""
    events = []
    for _, element in ET.iterparse(path):
        name = _local(element.tag)
        if name == "arrival":
            element.clear()
        elif name == "event":
            events.append(_event(element))
            element.clear()
    return Catalog(events=events)
