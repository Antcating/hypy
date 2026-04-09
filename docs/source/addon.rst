Addons and Custom Backends
==========================

This page documents how to extend ``hypy`` with custom:

* model backends (new velocity model formats)
* propagation backends (new travel-time engines)

The extension system supports runtime registration and plugin discovery through
Python entry points.

Addons topics
-------------

.. toctree::
   :maxdepth: 1

   addon_model
   addon_propagation


Extension points
----------------

Model backends
~~~~~~~~~~~~~~

Model backends are registered by file extension and must return a
``VelocityModel`` instance.

Public APIs:

* ``register_model_backend``
* ``get_model_backend_factory``
* ``list_registered_model_backends``
* ``load_model_plugins``
* ``create_model``


Propagation backends
~~~~~~~~~~~~~~~~~~~~

Propagation backends are registered by ``(mode, engine)`` and must return a
``PropagationEngine`` instance.

Public APIs:

* ``register_propagation_backend``
* ``get_propagation_backend_factory``
* ``list_registered_propagation_backends``
* ``load_propagation_plugins``
* ``create_propagation_backend``

Compatibility contract
----------------------

``hypy`` validates model/propagation compatibility during propagation backend
creation.

* Each model exposes ``supported_features``.
* Each propagation backend exposes ``required_model_features``.
* Backend creation fails early with ``ValueError`` when required model features
  are missing.

Typical feature flags include:

* ``dimension.1d`` / ``dimension.3d``
* ``model.profile.1d`` / ``model.grid.3d``
* ``propagation.ray`` / ``propagation.table``


Use entry points for install-time discovery
-------------------------------------------

For third-party packages, prefer entry points so registration happens after
installation without app-level wiring.

.. code-block:: toml

   [project.entry-points."hypy.model.backends"]
   ".xyz" = "my_package.models:load_xyz_model"

   [project.entry-points."hypy.propagation.backends"]
   "ray-1d:myengine" = "my_package.backends:create_my_ray_backend"

Optional legacy groups are still supported:

* ``hypy.model.loaders``
* ``hypy.propagation.tracers``


How to use from applications
----------------------------

You can explicitly load plugins at startup:

.. code-block:: python

   from hypy.model import load_model_plugins
   from hypy.propagation import load_propagation_plugins

   load_model_plugins()
   load_propagation_plugins()

Then create your locator normally. During setup, ``hypy`` validates model and
propagation feature compatibility before running location.


Recommended testing
-------------------

For every addon, add tests that cover:

1. Registration (runtime and entry-point load paths).
2. Backend creation.
3. Compatibility failures for missing model features.
