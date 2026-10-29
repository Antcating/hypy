How the inversion works
=======================

``hypy`` locates events by iterative, damped, weighted least squares,
following the HypoInverse formulation. A run proceeds in four stages:
initialization (seeding), iterative inversion, uncertainty estimation,
and export of ObsPy ``Origin`` objects. This page describes each stage,
the safeguards that keep the iteration stable, and the conditions
under which it stops.

Model and observations
----------------------

A hypocenter is stored in geographic coordinates — latitude and
longitude in degrees, depth :math:`z` in km — together with a separate
origin time :math:`t_0`. The inversion, however, does not solve for
degrees. For each event it solves for a local update

.. math::

   \delta\mathbf{m} = [\delta y, \delta x, \delta z, \delta t_0]^{\mathsf{T}},

where :math:`\delta y` is the northward and :math:`\delta x` the
eastward displacement (both in km), :math:`\delta z` is the depth
change (km), and :math:`\delta t_0` is in s. After a step is accepted,
:math:`\delta y` and :math:`\delta x` are converted back to latitude
and longitude increments (the latter scaled by :math:`\cos(\mathrm{lat})`).

For observation :math:`i` at station :math:`s` and phase
:math:`\phi`, the modeled arrival time is

.. math::

   t_i^{\mathrm{m}} = t_0 + T_i(\mathrm{lat}, \mathrm{lon}, z)
   + \Delta t_{s,\phi},

where :math:`T_i` is the travel time returned by the propagation
engine and :math:`\Delta t_{s,\phi}` an optional station–phase time
correction (positive for a delayed observed arrival).

Initialization
--------------

Two seeding strategies are available.

**Multi-seed heuristic** (default). Candidate hypocenters are built
from, in order: the catalog origin (if ``use_origins=True``), the
receiver-network centroid, an early-P anchor (extrapolated weighted early epicenter), and the station of the
earliest P arrival. The early-P anchor is the exponentially weighted
mean of P-arrival stations, extrapolated away from the centroid so
that events outside the network footprint can be represented.
Remaining candidates (``num_trials``, default 8) are Gaussian
perturbations of these anchors. On a two-dimensional network the
horizontal spread scales with the network radius and depths are
sampled around the anchor depth. An approximately linear array (minor-
to-major eigenvalue ratio of the station coordinates below 0.1) keeps
those perturbations in the station strip, with along-array scatter
equal to the network radius and cross-track scatter equal to the
strip half-width. Each candidate is scored with the inversion engine;
its origin time is the inverse-variance-weighted mean of
observed-minus-predicted times (after station corrections), and the
candidate with the lowest weighted root-mean-square residual (WRMS)
becomes the initial model. A fixed ``seed`` makes candidate generation
reproducible.

**Grid search.** If an ``initial_propagation_engine`` is supplied,
candidate epicenters are distributed over the user-defined search
bounds or an expanded envelope of the receiver network, combined with
the depth grid of that engine. Nodes with insufficient station–phase
coverage are discarded and the rest are scored by the same WRMS
criterion. Sampling the model space reduces sensitivity to the initial
estimate at additional computational cost, which is lowest when the
initial engine uses precomputed travel-time tables.

Forward models
--------------

Engines are described in detail in
:doc:`/extending_hypy/custom_propagation_engines`. In brief: 1D ray
tracing is provided through ObsPy TauP, Pyrocko ``cake``, and
``HypoRayTracer``, a local first-arrival tracer for layered models
that maps the spherical structure to flat layers by the
earth-flattening transform
:math:`z_f = R \ln\!\left(R / (R - z)\right)`,
:math:`v_f = v\,R / (R - z)`. For repeated locations, 1D travel-time
tables (from the ray tracers, or from a scikit-fmm fast-marching
solution) are interpolated bilinearly in depth and distance. For 3D
structures, a ``pyekfmm`` backend solves the eikonal equation on a
local Cartesian grid around each receiver, and travel times at
off-grid hypocenters are obtained by trilinear interpolation.

Linearization
-------------

Around the current hypocenter, each valid observation contributes one
Jacobian row and one residual,

.. math::

   \mathbf{J}_i = \left[ \frac{\partial T_i}{\partial y},
   \frac{\partial T_i}{\partial x},
   \frac{\partial T_i}{\partial z}, 1 \right],
   \qquad
   r_i = t_i^{\mathrm{obs}} - t_i^{\mathrm{m}}.

The spatial derivatives are in s/km, taken with respect to the local
north (:math:`y`) and east (:math:`x`) source coordinates and to
depth, in the same order as :math:`\delta\mathbf{m}` above.

Derivatives are analytical (from takeoff angle and local velocity, for
1D engines) or numerical (finite differences), according to
``derivative_method``. Observations whose travel time cannot be
computed — out of range or outside the model — are excluded from the
row set rather than terminating the run.

Iterative inversion
-------------------

Each iteration solves the damped weighted system

.. math::

   \begin{bmatrix} \mathbf{W}\mathbf{J} \\ \lambda\,\mathbf{I} \end{bmatrix}
   \delta\mathbf{m} =
   \begin{bmatrix} \mathbf{W}\mathbf{r} \\ \mathbf{0} \end{bmatrix},

with ``scipy`` LSQR (``damp``\ :math:`=\lambda`). The weight matrix
:math:`\mathbf{W}` combines pick-time uncertainties, an optional
distance taper, and optional HypoInverse-style residual weighting
that, from a configured iteration onward, progressively down-weights
observations with residuals above a multiple of the floored WRMS;
weights are recomputed each iteration and no pick is permanently
removed. Damping follows a Levenberg–Marquardt schedule: :math:`\lambda`
starts at 1; a trial step is accepted only if it improves the WRMS by
a relative factor greater than :math:`10^{-6}`, upon which
:math:`\lambda` is halved (floor :math:`10^{-8}`); otherwise
:math:`\lambda` is multiplied by five, up to :math:`\lambda_{\max}=10`,
and the step is recomputed from the pre-step state. At most eight
damping trials are attempted per iteration; if none is accepted, the
previous model is retained unchanged.

Safeguards
----------

Several mechanisms keep individual iterations — and individual
events — from destabilizing the solution:

* **Step limiting.** Each update vector is scaled as a whole so that
  its horizontal norm does not exceed 3 km, its depth component 1 km,
  and its origin-time component 1 s.
* **Depth bounds.** Applied steps are clamped to :math:`z \ge 0`. A
  proposed step that would leave the model (above the surface, or
  below the engine's ``max_depth_km``) freezes that event's depth —
  at the seed depth (``unresolved_depth="seed"``, the default), at the
  current depth (``"current"``), or, with ``None``, only after
  ``depth_elimination_iteration`` iterations — and the step is
  re-solved without the depth unknown. Depths fixed by the user
  (``fix_depth_km`` or per-event) are never moved.
* **Search bounds.** If geographic bounds are configured, epicenters
  are clipped back into the box after every trial step.
* **Underdetermined events.** Events left with fewer than
  ``min_picks_per_event`` usable observations (for example because
  stations lie beyond ``max_distance_km``) are marked invalid and
  excluded, without affecting other events; they are reported in the
  ``invalid`` indices returned by ``locate()``.
* **Rollback.** Every damping trial restarts from the pre-iteration
  state, so a rejected step leaves no trace in the solution.

Stopping conditions
-------------------

The iteration terminates when any of the following holds:

1. the WRMS falls below ``conv_threshold`` (default
   :math:`10^{-3}` s);
2. the relative WRMS improvement over the previous iteration falls
   below :math:`10^{-8}`, indicating stagnation;
3. ``max_iterations`` (default 100 in ``locate()``) is reached; or
4. no valid observations remain, in which case the run aborts and the
   affected events are reported as invalid.

Uncertainty estimation
----------------------

Following HypoInverse, an earthquake-level variance scale combines the
velocity-model uncertainty (``model_uncertainty_s``, RDERR, default
0.15 s) with the achieved fit:

.. math::

   s_{\mathrm{eq}}^2 = \mathrm{RDERR}^2 + \gamma\,\mathrm{WRMS}^2,

with :math:`\gamma` an internal scaling coefficient
(``error_coefficient``, fixed at its default of 1; it is not a
``locate()`` argument).
Pick uncertainties define the relative observation covariance. By
default it is diagonal; with ``full_data_covariance=True``, same-phase
observations at nearby stations are correlated through
:math:`C_{d,ij} = s_i s_j\, e^{-d_{ij}/L}` with correlation length
``correlation_length_km``, which reduces the effective number of
independent observations for dense arrays. The whitened Jacobian then
gives the model covariance
:math:`\mathbf{C}_m = s_{\mathrm{eq}}^2\,(\mathbf{J}_w^{\mathsf{T}} \mathbf{J}_w)^{-1}`.
The spatial block :math:`\mathbf{C}_{xyz}` is expressed in local
(north :math:`y`, east :math:`x`, down :math:`z`) coordinates.
Its eigenvalues and eigenvectors are the :math:`1\sigma` principal
standard deviations and axes of the uncertainty ellipsoid.
``hypy`` reports the vertical standard deviation

.. math::

   \mathrm{ERZ} = \sqrt{C_{zz}}

and the horizontal error :math:`\mathrm{ERH}`: the standard deviation
along the long axis of the north–east uncertainty ellipse. That axis
is rotated away from north and east when :math:`C_{yx} \neq 0`, so

.. math::

   \mathrm{ERH}
   = \sqrt{\lambda_{\max}\!\left(\mathbf{C}_{yx}\right)},

where :math:`\lambda_{\max}` is the larger eigenvalue of the north–east
block :math:`\mathbf{C}_{yx}`.
HypoInverse instead takes the largest vertical and horizontal
projections of the three principal axes.
The principal axes, :math:`\mathrm{ERH}`, and :math:`\mathrm{ERZ}` are
exported with each ``Origin`` in the QuakeML output, scaled from these
:math:`1\sigma` lengths to 95% confidence and converted to meters.
