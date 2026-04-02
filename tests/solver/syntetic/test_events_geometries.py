import zlib

import pytest

from tests.solver.syntetic.utils.common import Scenario, run_scenario


def _seed_all(
    geometry_name, phases, mode, engine, solve_mode, solver, bias, n_stations, n_events
) -> int:
    key = (
        f"{geometry_name}|{''.join(phases)}|{mode}|{engine}|{solve_mode}|{solver}|"
        f"{bias:.6f}|nst{n_stations}"
    )
    return zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF


@pytest.mark.parametrize("solve_mode", ["batch", "single"])
def test_multi_events_unified(
    taup_model,
    relocation_summary,
    geometry_name,
    phases,
    mode,
    engine,
    solver,
    bias,
    n_stations,
    n_events,
    solve_mode,
):
    scn = Scenario(
        geometry_name=geometry_name,
        phases=phases,
        mode=mode,
        engine=engine,
        solve_mode=solve_mode,
        solver=solver,
        bias=bias,
        n_stations=n_stations,
        n_events=n_events,
        seed=_seed_all(
            geometry_name,
            phases,
            mode,
            engine,
            solve_mode,
            solver,
            bias,
            n_stations,
            n_events,
        ),
    )
    run_scenario(scn, taup_model, relocation_summary)
