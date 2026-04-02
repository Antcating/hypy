import os
from pathlib import Path

import numpy as np
import obspy
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

from hypy.locator.model import VelocityModelHandler


def tvel_file_path(model_name: str) -> Path:
    obspy_path = os.path.dirname(obspy.__file__)
    return Path(os.path.join(obspy_path, "taup", "data", f"{model_name}.tvel"))


def test_velocity_model_init():
    model = "iasp91"
    model = VelocityModelHandler._init_velocity_model(model, tvel_file_path(model))
    assert isinstance(model, TauPyModel)


def test_precompute_velocity_profiles_1d():
    model = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = Path(os.path.join(obspy_path, "taup", "data", f"{model}.tvel"))
    handler = VelocityModelHandler(model_path)

    handler._precompute_velocity_profiles_1d(
        phases=["P"], max_depth=100.0, depth_step=0.2
    )

    # Get velocity directly from TauP for comparison
    depths = np.arange(0, 100.0, 0.2)

    for d in depths:
        v = handler.model.model.s_mod.v_mod.evaluate_below(d, "p")[0]
        v_interp = handler._get_velocity_1d(d, "P")

        assert np.isclose(v, v_interp, rtol=1e-3), (
            f"Velocity mismatch at depth {d}: {v} vs {v_interp}"
        )


def test_precompute_velocity_profiles_1d_sampling():
    model = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = Path(os.path.join(obspy_path, "taup", "data", f"{model}.tvel"))
    handler = VelocityModelHandler(model_path)

    handler._precompute_velocity_profiles_1d(
        phases=["P"], max_depth=100.0, depth_step=0.2
    )

    # Get velocity directly from TauP for comparison
    depths = np.arange(0, 100.0, 0.1)

    for d in depths:
        v = handler.model.model.s_mod.v_mod.evaluate_below(d, "p")[0]
        v_interp = handler._get_velocity_1d(d, "P")

        # The error we be bigger much bigger
        # Because the velocity model is a step function and linear
        # interpolation will not capture the jumps in velocity at layer boundaries.

        assert np.isclose(v, v_interp, rtol=3), (
            f"Bigger than expected velocity mismatch at depth {d}: {v} vs {v_interp}"
        )


def test_get_velocity_1d_no_precompute():
    model = "iasp91"
    obspy_path = os.path.dirname(obspy.__file__)
    model_path = Path(os.path.join(obspy_path, "taup", "data", f"{model}.tvel"))
    handler = VelocityModelHandler(model_path)

    # Get velocity directly from TauP for comparison
    depths = np.arange(0, 100.0, 0.2)

    for d in depths:
        v = handler.model.model.s_mod.v_mod.evaluate_below(d, "p")[0]
        v_direct = handler._get_velocity_1d(d, "P")

        assert np.isclose(v, v_direct, rtol=1e-3), (
            f"Velocity mismatch at depth {d} without precompute: {v} vs {v_direct}"
        )
