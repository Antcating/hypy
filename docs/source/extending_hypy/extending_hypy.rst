.. _Extending_hypy:

Extending hypy
==============

``hypy`` is designed to be extended with custom:

* **velocity model classes** — new velocity-model formats or data sources
* **propagation engine classes** — new travel-time / derivative engines

Extension is **composition based**. You implement a class, instantiate it, and
pass the instance into the pipeline. There is no registry to edit and no
plugin entry point to declare:

.. code-block:: text

   your VelocityModel  ──►  your PropagationEngine  ──►  EventLocator

Both extension points subclass a small base class, set a few feature flags, and
implement a handful of methods. ``hypy`` validates that the pieces are
compatible before any inversion runs.

.. toctree::
   :maxdepth: 1

   addon_model
   addon_propagation


Extension points
----------------

Custom velocity model
~~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`hypy.model.VelocityModel`, declare ``feature_flags``, and load
your data in ``__init__``. Pass an instance to your propagation engine:

.. code-block:: python

   model = MyVelocityModel("/path/to/model.custom")
   engine = SomeEngine(model)

See :doc:`_model` for the full contract.

Custom propagation engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`hypy.propagation.base.PropagationEngine` (or a convenience base
such as :class:`hypy.propagation.layered.LayeredPropagationEngine`), declare
``feature_flags`` / ``required_model_features``, and implement
``compute_travel_time_params(...)``. Pass an instance to ``EventLocator``:

.. code-block:: python

   from hypy.locator import EventLocator

   engine = MyPropagationEngine(model)
   locator = EventLocator(engine, phases=("P", "S"), derivative_method="numerical")

See :doc:`_propagation` for the full contract.


The compatibility contract
---------------------------

``hypy`` performs two feature checks, both of which fail fast with a clear
error before any location work begins.

Model / engine compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model exposes ``feature_flags`` and every engine exposes
``required_model_features``. When an engine is constructed,
:meth:`~hypy.propagation.base.PropagationEngine.validate_model` checks that::

   engine.required_model_features  ⊆  model.feature_flags

If a required feature is missing, the engine constructor raises ``ValueError``
naming the missing features.

Derivative-method compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the engine is passed to ``EventLocator``, the locator checks that the
engine advertises the requested derivative capability::

   f"derivative.{derivative_method}"  ∈  engine.feature_flags

For ``derivative_method="numerical"`` the engine must declare
``derivative.numerical``; for ``"analytical"`` it must declare
``derivative.analytical``. A mismatch raises ``ValueError``.

Feature-flag vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~

Feature flags are plain strings. The built-in backends use a dotted naming
convention; custom backends are encouraged to follow it for consistency:

* **Model dimensionality / type:** ``model.profile.1d``, ``model.layered``,
  ``model.volumetric``, ``model.grid.3d``
* **Model format:** ``format.tvel``, ``format.netcdf`` (free-form per format)
* **Phases:** ``phase.p``, ``phase.s``
* **Propagation strategy:** ``propagation.layered``, ``propagation.ray``,
  ``propagation.table``, ``propagation.volumetric``
* **Derivatives:** ``derivative.numerical``, ``derivative.analytical``

A backend may also expose inferred flags at runtime through
``supported_features`` (for example ``dimension.1d`` derived from ``dim``).


Recommended testing
--------------------

For every , add tests that cover:

1. Constructor compatibility — your model loads, and your engine accepts the
   model.
2. Feature-compatibility failures — an engine raises ``ValueError`` when the
   model is missing a required feature.
3. A full ``EventLocator`` run using your injected classes end to end.

See ``examples/custom_backend_contract.py`` for a minimal, self-contained
propagation engine that satisfies the runtime contract.
