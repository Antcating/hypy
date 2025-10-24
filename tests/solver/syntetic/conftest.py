import logging
import zlib
import numpy as np
import pytest
from obspy.taup import TauPyModel  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---- Global parametrization spaces ----
PHASE_SETS = (["P"], ["S"], ["P", "S"])
MODES = ["table"]
ENGINES = ["taup", "pyrocko"]
GEOM_NAMES = sorted(["uniform_grid", "ring", "L_shape", "sparse", "cluster"])
SOLVERS = ["lsqr", "svd"]
BIASES = [0.01, 0.1]
N_EVENTS = [10, 50, 100]
N_STATIONS = [8, 16, 25, 50, 100]


@pytest.fixture(autouse=True)
def _seed_per_test(request):
    """
    Deterministic numpy seed per collected test/param id.
    Predictable and order-independent.
    """
    seed = zlib.adler32(request.node.nodeid.encode()) & 0xFFFFFFFF
    np.random.seed(seed)


@pytest.fixture(scope="session")
def taup_model():
    return TauPyModel(model="iasp91")


@pytest.fixture(scope="session")
def relocation_summary():
    """
    Session-wide accumulator and final summary.
    """
    stats = {
        "total": 0,
        "success": 0,
        "sum_geo": 0.0,
        "sum_depth": 0.0,
        "sum_time": 0.0,
        "n_err": 0,
    }
    yield stats

    if stats["n_err"] > 0:
        avg_geo = stats["sum_geo"] / stats["n_err"]
        avg_depth = stats["sum_depth"] / stats["n_err"]
        avg_time = stats["sum_time"] / stats["n_err"]
    else:
        avg_geo = avg_depth = avg_time = float("nan")

    logger.info(
        f"[RELOCATION SUMMARY - ALL TESTS] "
        f"{stats['success']}/{stats['total']} events relocated | "
        f"avg Geo {avg_geo:.3f} km | Depth {avg_depth:.3f} km | Time {avg_time:.3f} s"
    )


# ---- Fixtures for param spaces (kept small & declarative) ----


@pytest.fixture(params=GEOM_NAMES, ids=lambda g: g)
def geometry_name(request):
    return request.param


@pytest.fixture(params=PHASE_SETS, ids=lambda p: "ph=" + "".join(p))
def phases(request):
    return request.param


@pytest.fixture(params=MODES, ids=lambda m: m)
def mode(request):
    return request.param


@pytest.fixture(params=ENGINES, ids=lambda e: e)
def engine(request):
    return request.param


@pytest.fixture(params=SOLVERS, ids=lambda s: f"solver={s}")
def solver(request):
    return request.param


@pytest.fixture(params=BIASES, ids=lambda b: f"bias={b:.2f}s")
def bias(request):
    return request.param


@pytest.fixture(params=N_EVENTS, ids=lambda n: f"nev={n}")
def n_events(request):
    return request.param


@pytest.fixture(params=N_STATIONS, ids=lambda n: f"nst={n}")
def n_stations(request):
    return request.param
