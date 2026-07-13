"""Common utilities for seismic data processing."""


def normalize_phase(phase: str) -> str:
    """Normalize a seismic phase name to the canonical ``"p"``/``"s"`` code.

    This is the single source of truth for phase normalization across every
    propagation backend (ray tracers and travel-time grids) and the velocity
    models.  Names are stripped and lower-cased, and any phase whose first
    non-space character is ``p``/``s`` (e.g. ``"P"``, ``"Pn"``, ``"Pg"``,
    ``"s"``, ``"Sg"``) maps to the corresponding vertical phase.  Anything else
    raises :class:`ValueError`.

    Parameters
    ----------
    phase : str
        Phase name to normalize.

    Returns
    -------
    str
        Either ``"p"`` or ``"s"``.

    Raises
    ------
    ValueError
        If ``phase`` is not a recognized P- or S-type phase.
    """
    phase_lower = phase.strip().lower()
    if phase_lower.startswith("p"):
        return "p"
    if phase_lower.startswith("s"):
        return "s"
    raise ValueError(
        f"Unsupported phase '{phase}'. Only 'P' and 'S' phases are supported."
    )
