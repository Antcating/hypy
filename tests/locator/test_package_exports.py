"""Tests for public package exports in hypy.model."""

from hypy.model import NCGrid, NllGridModel, TvelModel, VelocityModel


def test_model_package_exports_common_backends():
    """Public model package should expose the common concrete backend
    classes."""
    assert VelocityModel.__name__ == "VelocityModel"
    assert TvelModel.__name__ == "TvelModel"
    assert NCGrid.__name__ == "NCGrid"
    assert NllGridModel.__name__ == "NllGridModel"
