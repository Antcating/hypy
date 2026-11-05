"""Tests for the lightweight QuakeML reader."""

from pathlib import Path

import obspy

from hypy.utils.io import read_quakeml

QUAKEML = """<?xml version="1.0"?>
<q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2"
           xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
  <eventParameters publicID="smi:local/test">
    <event publicID="smi:local/ev1">
      <preferredOriginID>smi:local/or1</preferredOriginID>
      <preferredMagnitudeID>smi:local/mag1</preferredMagnitudeID>
      <origin publicID="smi:local/or-old">
        <time><value>2020-01-01T00:00:00.000000Z</value></time>
        <latitude><value>1.0</value></latitude>
        <longitude><value>2.0</value></longitude>
        <depth><value>1000.0</value></depth>
      </origin>
      <origin publicID="smi:local/or1">
        <time><value>2025-12-31T19:44:45.661878Z</value></time>
        <latitude><value>30.07</value></latitude>
        <longitude><value>35.14</value></longitude>
        <depth><value>18855.0</value></depth>
        <evaluationMode>manual</evaluationMode>
        <arrival publicID="smi:local/arr1">
          <phase>P</phase>
          <timeResidual>0.1</timeResidual>
        </arrival>
      </origin>
      <magnitude publicID="smi:local/mag1">
        <mag><value>1.98</value></mag>
        <type>Mws</type>
      </magnitude>
      <pick publicID="smi:local/pk1">
        <time><value>2025-12-31T19:44:57.494445Z</value></time>
        <waveformID networkCode="IS" stationCode="RMNI"/>
        <phaseHint>P</phaseHint>
      </pick>
    </event>
  </eventParameters>
</q:quakeml>
"""


def test_read_quakeml_matches_obspy_fields(tmp_path: Path) -> None:
    """Parsed fields match ObsPy on a small catalog with arrivals."""
    path = tmp_path / "catalog.xml"
    path.write_text(QUAKEML)

    catalog = read_quakeml(str(path))
    reference = obspy.read_events(path)

    assert len(catalog) == 1
    event = catalog[0]
    origin = event.preferred_origin()
    ref_origin = reference[0].preferred_origin()
    assert origin is not None and ref_origin is not None
    assert origin.time == ref_origin.time
    assert origin.latitude == ref_origin.latitude
    assert origin.longitude == ref_origin.longitude
    assert origin.depth == ref_origin.depth
    assert origin.evaluation_mode == "manual"
    assert event.origins[-1].resource_id == origin.resource_id
    assert len(origin.arrivals) == 0

    magnitude = event.preferred_magnitude()
    ref_magnitude = reference[0].preferred_magnitude()
    assert magnitude is not None and ref_magnitude is not None
    assert magnitude.mag == ref_magnitude.mag
    assert magnitude.magnitude_type == ref_magnitude.magnitude_type

    pick = event.picks[0]
    ref_pick = reference[0].picks[0]
    assert pick.time == ref_pick.time
    assert pick.phase_hint == ref_pick.phase_hint
    assert pick.waveform_id.network_code == ref_pick.waveform_id.network_code
    assert pick.waveform_id.station_code == ref_pick.waveform_id.station_code
