Installation
============

``hypy`` requires Python 3.11 or later. From a source checkout:

.. code-block:: bash

   git clone <repository-url> hypy
   cd hypy
   pip install .

Optional backends (Pyrocko, scikit-fmm, and the 3D ``pyekfmm``
stack) are available as `uv <https://docs.astral.sh/uv/>`_ dependency
groups, e.g. ``uv sync --group 3d``; with plain ``pip``, install the
corresponding packages (``pyrocko``, ``scikit-fmm``, ``netcdf4``,
``nllgrid``, ``pyekfmm``) instead.

See :doc:`starting_guide` for a complete first location run.
