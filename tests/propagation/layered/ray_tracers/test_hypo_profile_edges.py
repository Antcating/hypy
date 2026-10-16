"""Tests for HypoRayTracer interface-preserving layer resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypy.model.layered.tvel import TvelModel
from hypy.propagation.layered.ray_tracer.hypo.tracer import HypoRayTracer


@pytest.fixture
def giimod_like_tvel(tmp_path: Path) -> Path:
    """Minimal piecewise-constant crustal model with off-grid interfaces."""
    path = tmp_path / "giimod_like.tvel"
    path.write_text(
        "\n".join(
            [
                " giimod-like P",
                " giimod-like S",
                "     0.000      4.3600      2.4100      2.5000",
                "     2.600      4.3600      2.4100      2.5000",
                "     2.600      5.5100      3.1000      2.6000",
                "     9.800      5.5100      3.1000      2.7000",
                "     9.800      6.2300      3.6000      2.8000",
                "    35.000      6.2300      3.6000      2.8000",
                "    35.000      8.0400      4.4700      3.3198",
                "    77.500      8.0450      4.4850      3.3455",
            ]
        )
        + "\n"
    )
    return path


def test_profile_edges_preserve_off_grid_interfaces(giimod_like_tvel: Path):
    """Uniform 1 km step must still cut layers at 2.6 km and 9.8 km."""
    model = TvelModel(giimod_like_tvel)
    tracer = HypoRayTracer(
        model=model, max_depth_km=50.0, max_distance_km=300.0, layer_step_km=1.0
    )
    edges = tracer._profile_edges("p")
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(50.0)
    assert any(np.isclose(edges, 2.6))
    assert any(np.isclose(edges, 9.8))
    # Interfaces that already lie on the grid remain unique.
    assert np.sum(np.isclose(edges, 35.0)) == 1


def test_interface_preserving_step1_matches_fine_step(giimod_like_tvel: Path):
    """Coarse step with preserved edges should match a fine uniform
    resample."""
    model = TvelModel(giimod_like_tvel)
    coarse = HypoRayTracer(
        model=model, max_depth_km=50.0, max_distance_km=300.0, layer_step_km=1.0
    )
    fine = HypoRayTracer(
        model=model, max_depth_km=50.0, max_distance_km=300.0, layer_step_km=0.1
    )

    depths = (5.0, 8.0, 15.0)
    distances = (20.0, 35.0, 50.0, 80.0)
    for depth in depths:
        for distance in distances:
            t_coarse, _ = coarse.get_arrival_1d(distance, depth, "p")
            t_fine, _ = fine.get_arrival_1d(distance, depth, "p")
            assert t_coarse == pytest.approx(t_fine, abs=2.0e-3)


def test_missing_interface_helper_still_builds_grid(
    monkeypatch, giimod_like_tvel: Path
):
    """If layer tops are unavailable, fall back to the uniform grid only."""
    model = TvelModel(giimod_like_tvel)
    tracer = HypoRayTracer(
        model=model, max_depth_km=20.0, max_distance_km=100.0, layer_step_km=1.0
    )

    def _raise(*_args, **_kwargs):
        raise ValueError("no tops")

    monkeypatch.setattr(model, "get_layer_top_profile_1d", _raise)
    edges = tracer._profile_edges("p")
    assert np.allclose(edges, np.arange(0.0, 21.0, 1.0))
