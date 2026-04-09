"""Tests for public package exports in hypy.propagation."""

from hypy.propagation import (
    LayeredPropagationEngine,
    PropagationEngine,
    VolumetricPropagationEngine,
)


def test_propagation_package_exports_common_bases():
    """Public propagation package should expose the shared base classes."""
    assert PropagationEngine.__name__ == "PropagationEngine"
    assert LayeredPropagationEngine.__name__ == "LayeredPropagationEngine"
    assert VolumetricPropagationEngine.__name__ == "VolumetricPropagationEngine"
