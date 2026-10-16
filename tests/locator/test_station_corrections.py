"""Tests for station time-correction application."""

import logging

import numpy as np
import pandas as pd
import pytest

from hypy.locator.locations_setup import _usable_arrivals
from hypy.utils.stations.corrections import apply_station_time_corrections


def test_apply_none_is_zeros():
    """``None`` leaves a zero correction vector."""
    np.testing.assert_array_equal(
        apply_station_time_corrections(None, [("NW", "STA1")], ("p", "s")),
        [0.0, 0.0],
    )


def test_apply_broadcasts_then_overrides(caplog, monkeypatch):
    """Station-wide delays broadcast; phase rows override; unknowns warn."""
    monkeypatch.setattr(logging.getLogger("hypy"), "propagate", True)
    table = pd.DataFrame(
        [
            {"network": "NW", "station": "STA1", "phase": np.nan, "delay_s": 0.10},
            {"network": "NW", "station": "STA1", "phase": "S", "delay_s": 0.25},
            {"network": "XX", "station": "MISS", "delay_s": 0.40},
            {"network": "NW", "station": "STA2", "phase": "s", "delay_s": 0.30},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="hypy"):
        corrections = apply_station_time_corrections(
            table, [("NW", "STA1"), ("NW", "STA2")], ("p",)
        )

    np.testing.assert_allclose(corrections, [0.10, 0.0])
    assert "XX" in caplog.text
    assert "MISS" in caplog.text
    assert "s" in caplog.text


def test_apply_normalizes_phase_aliases():
    """``Pn``/``Sg`` map onto the configured ``p``/``s`` slots."""
    table = pd.DataFrame(
        [
            {"network": "NW", "station": "STA1", "phase": "*", "delay_s": 0.10},
            {"network": "NW", "station": "STA1", "phase": "Pn", "delay_s": 0.20},
            {"network": "NW", "station": "STA1", "phase": "Sg", "delay_s": 0.30},
            {"network": "NW", "station": "STA2", "delay_s": 0.05, "extra": 1},
        ]
    )
    corrections = apply_station_time_corrections(
        table, [("NW", "STA1"), ("NW", "STA2")], ("P", "S")
    )
    np.testing.assert_allclose(corrections, [0.20, 0.30, 0.05, 0.05])


def test_apply_rejects_missing_columns():
    """Required table columns are validated."""
    table = pd.DataFrame([{"network": "IU", "station": "ANMO"}])
    with pytest.raises(ValueError, match="missing required columns"):
        apply_station_time_corrections(table, [("IU", "ANMO")], ("p",))


def test_apply_rejects_non_finite_delay():
    """Non-finite delays are rejected."""
    table = pd.DataFrame([{"network": "IU", "station": "ANMO", "delay_s": np.nan}])
    with pytest.raises(ValueError, match="finite float"):
        apply_station_time_corrections(table, [("IU", "ANMO")], ("p",))


def test_apply_rejects_empty_station_codes():
    """Blank network or station codes are rejected."""
    table = pd.DataFrame([{"network": "IU", "station": "  ", "delay_s": 0.1}])
    with pytest.raises(ValueError, match="non-empty"):
        apply_station_time_corrections(table, [("IU", "ANMO")], ("p",))


def test_apply_rejects_unsupported_type():
    """Mappings and other types are rejected."""
    with pytest.raises(ValueError, match="must be None"):
        apply_station_time_corrections({("IU", "ANMO"): 0.1}, [("IU", "ANMO")], ("p",))


def test_apply_rejects_unsupported_phase():
    """Phase names must be P- or S-type."""
    table = pd.DataFrame(
        [{"network": "IU", "station": "ANMO", "phase": "Lg", "delay_s": 0.1}]
    )
    with pytest.raises(ValueError, match="Unsupported phase"):
        apply_station_time_corrections(table, [("IU", "ANMO")], ("p",))


def test_usable_arrivals_subtract_station_time_corrections():
    """Seeding uses delay-corrected observed times."""
    data = type("D", (), {})()
    data.config = type("C", (), {"n_phases": 1})()
    data.n_stations = 1
    data.arrivals = np.array([[5.0]], dtype=float)
    data.arrival_time_errors = np.array([[0.2]], dtype=float)
    data.station_time_corrections = np.array([0.3], dtype=float)

    usable = _usable_arrivals(data, 0)
    assert usable == [(0, 0, 4.7, 0.2)]
