import shutil
from pathlib import Path

import numpy as np

from hypy.model.layered.tvel import TvelModel


def _iasp91_tvel_path() -> Path:
    import obspy

    obspy_path = Path(obspy.__file__).parents[0]
    return obspy_path / "taup" / "data" / "iasp91.tvel"


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
