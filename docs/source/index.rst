.. hypy documentation master file

hypy documentation
==================

.. image:: _static/hypy_logo.svg
   :width: 150px
   :height: 150px
   :align: center
   :alt: hypy logo

``hypy`` is a Python package for absolute seismic event location (hypocenter
determination) from P/S phase picks. It performs iterative least-squares inversion 
against a velocity model and reports relocated origins together with uncertainty estimates.

The primary goal of ``hypy`` is to provide a user-friendly interface for seismic event location,
while also being flexible enough to allow easy :doc:`extension <extending_hypy/extending_hypy>`. 
It is designed to be used by seismologists, geophysicists, and researchers 
in related fields who need to locate seismic events accurately and efficiently.


.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quick_start

.. toctree::
   :maxdepth: 2
   :caption: Extending hypy

   extending_hypy/extending_hypy

.. toctree::
   :maxdepth: 3
   :caption: API

   api/hypy

