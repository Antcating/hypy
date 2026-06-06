Addons and Custom Backends
==========================

This page documents how to extend ``hypy`` with custom:

* model classes (new velocity model formats)
* propagation classes (new travel-time engines)

Extension is constructor-driven: you provide compatible classes directly to
``EventLocator`` at initialization time.

Addons topics
-------------

.. toctree::
   :maxdepth: 1

   addon_model
   addon_propagation


Extension points
----------------

Model class injection
~~~~~~~~~~~~~~~~~~~~~

Pass a custom ``VelocityModel`` subclass through
``velocity_model_backend=...`` when creating ``EventLocator``.

The class must provide:

* ``supported_features`` through the ``VelocityModel`` contract
* compatible constructor input from ``model_path``


Propagation class injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a custom ``PropagationEngine`` subclass through
``engine_backend=...`` when creating ``EventLocator``.

The class must satisfy a strict constructor and runtime contract (see
``addon_propagation``):

* ``default_feature_flags``
* ``default_required_model_features``
* ``dim`` property
* ``compute_travel_time_params(...)``

These contracts are validated before inversion starts.

Compatibility contract
----------------------

``hypy`` validates model/propagation compatibility during locator setup.

* Each model exposes ``supported_features``.
* Each propagation backend exposes ``default_required_model_features``.
* Setup fails early with ``ValueError`` when required model features
   are missing.

Typical feature flags include:

* ``model.profile.1d`` / ``dimension.3d``
* ``model.profile.1d`` / ``model.grid.3d``
* ``propagation.ray`` / ``propagation.table``


How to use from applications
----------------------------

Create your locator with injected classes:

.. code-block:: python

   from hypy import EventLocator
   from my_package.models import MyVelocityModel
   from my_package.backends import MyPropagationBackend

   locator = EventLocator(
       model_path="/path/to/model.custom",
       mode="ray-1d",
       engine="taup",
       derivative_method="numerical",
       velocity_model_backend=MyVelocityModel,
       engine_backend=MyPropagationBackend,
   )

During setup, ``hypy`` validates model and propagation feature compatibility
before running location.


Recommended testing
-------------------

For every addon, add tests that cover:

1. Constructor compatibility for your custom model and backend classes.
2. Feature compatibility failures for missing model features.
3. A full ``EventLocator`` run using your injected classes.

See also: ``examples/custom_backend_contract.py`` for a complete backend
template that satisfies the required constructor and method contract.
