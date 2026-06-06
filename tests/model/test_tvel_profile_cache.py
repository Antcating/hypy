import shutil
from pathlib import Path

import numpy as np

from hypy.model.layered.tvel import TvelModel


def _iasp91_tvel_path() -> Path:
    import obspy

    obspy_path = Path(obspy.__file__).parents[0]
    return obspy_path / "taup" / "data" / "iasp91.tvel"


def test_tvel_profile_cache_file_created(tmp_path: Path) -> None:
    source_model = _iasp91_tvel_path()
    target_model = tmp_path / "iasp91.tvel"
    shutil.copy2(source_model, target_model)

    model = TvelModel(target_model)
    model._precompute_velocity_profiles_1d(["P"], 120.0, 2.0)

    cache_dir = target_model.parent / ".cache"
    assert cache_dir.exists(), (
        "Cache directory should be created next to the .tvel file"
    )

    cache_files = list(cache_dir.glob("tvel_profile_*_p_*.npz"))
    assert len(cache_files) == 1

    with np.load(cache_files[0], allow_pickle=True) as loaded:
        assert float(loaded["max_depth"]) == 120.0
        assert float(loaded["depth_step"]) == 2.0
        assert loaded["phase"].tolist() == "p"
        assert loaded["depths"].shape == loaded["v_profile"].shape


def test_tvel_profile_cache_loaded_from_disk(tmp_path: Path, monkeypatch) -> None:
    source_model = _iasp91_tvel_path()
    target_model = tmp_path / "iasp91.tvel"
    shutil.copy2(source_model, target_model)

    model = TvelModel(target_model)
    model._precompute_velocity_profiles_1d(["S"], 80.0, 1.0)

    cache_files = list((target_model.parent / ".cache").glob("tvel_profile_*_s_*.npz"))
    assert len(cache_files) == 1

    model2 = TvelModel(target_model)
    built = {"called": False}

    def fake_build(phase_lower, depths):
        built["called"] = True
        return model2._build_velocity_profile(phase_lower, depths)

    monkeypatch.setattr(model2, "_build_velocity_profile", fake_build)
    model2._precompute_velocity_profiles_1d(["S"], 80.0, 1.0)

    assert built["called"] is False
    assert "s" in model2.velocity_profiles
    depths, velocity = model2.velocity_profiles["s"]
    assert depths.shape == velocity.shape


def test_get_velocity_1d_direct_layer_lookup(tmp_path: Path) -> None:
    source_model = _iasp91_tvel_path()
    target_model = tmp_path / "iasp91.tvel"
    shutil.copy2(source_model, target_model)

    model = TvelModel(target_model)
    for depth, phase in [(0.0, "P"), (0.0, "S"), (100.0, "P"), (100.0, "S")]:
        expected = model.model.model.s_mod.v_mod.evaluate_below(
            depth, phase.lower()[-1]
        )[0]
        assert np.isclose(model.get_velocity_1d(depth, phase), expected)


def test_get_velocity_1d_uses_cached_profile(tmp_path: Path) -> None:
    source_model = _iasp91_tvel_path()
    target_model = tmp_path / "iasp91.tvel"
    shutil.copy2(source_model, target_model)

    model = TvelModel(target_model)
    model._precompute_velocity_profiles_1d(["P"], 600.0, 5.0)

    depth = 123.4
    expected = model.model.model.s_mod.v_mod.evaluate_below(depth, "p")[0]
    assert np.isclose(model.get_velocity_1d(depth, "P"), expected)
