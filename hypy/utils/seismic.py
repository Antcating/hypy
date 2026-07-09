"""Common utilities for seismic data processing."""


def normalize_phase(phase: str) -> str:
    """Normalize phase name to a single character code."""
    phase_lower = phase.lower()
    if phase_lower.startswith("p"):
        return "p"
    elif phase_lower.startswith("s"):
        return "s"
    else:
        raise ValueError(
            f"Unsupported phase '{phase}'. Only 'P' and 'S' phases are supported."
        )
