"""Tests for the Pyrocko import guard in layered ray tracers."""

import builtins
import importlib
import sys

import pytest

MODULE_PATH = "hypy.propagation.layered.ray_tracer.pyrocko"
IMPORT_ERROR_MSG = (
    "Pyrocko is not installed or not available. Please install Pyrocko to use "
    "the pyrocko engine."
)
MODULE_SUBMODULE = f"{MODULE_PATH}.tracer"


def test_pyrocko_import_error_on_missing(monkeypatch):
    """Missing Pyrocko should surface a clear import error."""
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyrocko" or name.startswith("pyrocko."):
            raise ImportError(
                "Pyrocko is not installed or not available. Please install Pyrocko "
                "to use the pyrocko engine."
            )
        return original_import(name, globals, locals, fromlist, level)

    sys.modules.pop(MODULE_PATH, None)
    sys.modules.pop(MODULE_SUBMODULE, None)
    sys.modules.pop("pyrocko", None)
    sys.modules.pop("pyrocko.cake", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=IMPORT_ERROR_MSG):
        importlib.import_module(MODULE_PATH)
