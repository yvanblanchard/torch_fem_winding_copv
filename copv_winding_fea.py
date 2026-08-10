"""Composite Overwrapped Pressure Vessel (COPV).

A filament-wound pressure vessel modeled as a layered `Shell`: a metallic liner
overwrapped with helical carbon-fiber plies and loaded by internal pressure. One
octant of the vessel is meshed and closed with symmetry planes.

Two winding models are available, selected by `GEODESIC_WINDING`:

* `False` - the nominal model: every ply keeps its constant nominal angle and a
  uniform thickness over the whole surface.
* `True` - a manufacturing model: each ply follows a Clairaut geodesic, so its
  winding angle opens up from the nominal cylinder value towards 90 deg at the
  turnaround radius, and its thickness follows the band build-up of the dome.

Script version of copv.ipynb.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import torch
from scipy.integrate import quad

from torchfem import Laminate
from torchfem.data import get_data
from torchfem.io import import_mesh
from torchfem.materials import (
    IsotropicElasticityPlaneStress,
    OrthotropicElasticityPlaneStress,
)
from torchfem.rotations import planar_rotation

torch.set_default_dtype(torch.float64)

# Use the geodesic (Clairaut) winding model instead of constant ply angles
GEODESIC_WINDING = True

# --- Winding / manufacturing constants ---
BAND_WIDTH = 16.0  # mm - width of the deposited fiber band
PLY_THICKNESS = 0.35  # mm - nominal thickness of a single ply
T_R = 0.65  # mm - laminate thickness at r = R (cylindrical part)
MINIMUM_THICKNESS_THRESHOLD = 0.2  # mm - below this a layer is dropped

NOMINAL_ANGLE = 15.0  # deg - winding angle in the cylindrical section


# Material and laminate
#
# An isotropic aluminium liner combined with an orthotropic carbon/epoxy
# overwrap. The liner is much thicker than the individual composite plies, which
# are wound in a +/-15 deg helical pattern.

cfrp = OrthotropicElasticityPlaneStress(
    E_1=176800.0,
    E_2=10300.0,
    nu_12=0.23,
    G_12=4800.0,
    G_13=4800.0,
    G_23=3000.0,
    rho=1.6e-9,
)

alu = IsotropicElasticityPlaneStress(
    E=72000.0,
    nu=0.33,
    rho=2.7e-9,
)

p = torch.deg2rad(torch.tensor(NOMINAL_ANGLE))
m = torch.deg2rad(torch.tensor(-NOMINAL_ANGLE))
nominal_layup = Laminate(
    materials=[alu, cfrp, cfrp, cfrp, cfrp],
    thicknesses=[4.0, 0.5, 0.5, 0.5, 0.5],
    angles=[torch.tensor(0.0), p, m, p, m],
)

nominal_layup.plot()


# Geodesic (Clairaut) winding model
#
# On a surface of revolution a geodesic satisfies Clairaut's relation
#
#     r * sin(alpha) = const = R * sin(alpha_0) = r_0,
#
# so a band laid down at `alpha_0` on the cylinder (r = R) opens up towards the
# axis and becomes purely circumferential (alpha = 90 deg) at the turnaround
# radius `r_0`, the polar opening. The fiber cannot go below `r_0`, so a ply
# covers only `r >= r_0` and vanishes past the turnaround.
#
# The thickness build-up over the dome follows the band model of `thickness.py`:
# a cubic in the transition region `r_0 <= r <= r_0 + 2b` matched to the wound
# band coverage outside it. Both the angle and the thickness depend on `r` alone,
# so all elements at the same axial station share the same values.

# The vessel axis is global x here, so the meridional radius is r = sqrt(y^2+z^2)
mesh_nodes = import_mesh(get_data("copv.vtu"), alu).nodes
R_VESSEL = float(torch.sqrt(mesh_nodes[:, 1] ** 2 + mesh_nodes[:, 2] ** 2).max())


def clairaut_angle(r, alpha_0_deg, R=R_VESSEL):
    """Geodesic winding angle [rad] at radius `r` for a nominal angle at r = R.

    Follows `r sin(alpha) = R sin(alpha_0)`. The argument is clipped so that the
    region inside the turnaround radius (unreachable by the fiber) reports the
    limiting 90 deg rather than producing NaN.
    """
    r = torch.as_tensor(r, dtype=torch.get_default_dtype())
    r_0 = R * np.sin(np.radians(alpha_0_deg))
    return torch.asin(torch.clamp(r_0 / r.clamp(min=1e-12), max=1.0))


def turnaround_radius(alpha_0_deg, R=R_VESSEL):
    """Radius at which the geodesic reaches 90 deg and turns around."""
    return R * np.sin(np.radians(alpha_0_deg))


def check_turnaround(alpha_0_deg, r_min, R=R_VESSEL):
    """Verify the geodesic turns around within the meshed vessel.

    If the polar opening `r_0` lies below the smallest radius present in the
    mesh, the fiber never reaches 90 deg on this geometry and the winding is not
    realizable as modeled.
    """
    r_0 = turnaround_radius(alpha_0_deg, R)
    if r_min > r_0:
        raise ValueError(
            f"Geodesic at alpha_0 = {alpha_0_deg} deg turns around at "
            f"r_0 = {r_0:.3f} mm, below the smallest radius in the mesh "
            f"(r_min = {r_min:.3f} mm): the 90 deg turnaround is never reached "
            "within the vessel."
        )
    return r_0


# --- Thickness build-up, ported from thickness.py ---


def _band_globals(alpha_0_deg, R=R_VESSEL):
    """Geometric quantities of the band model for one winding angle."""
    alpha_0 = np.radians(alpha_0_deg)
    r_0 = R * np.sin(alpha_0)
    return {
        "alpha_0": alpha_0,
        "r_0": r_0,
        "r_b": r_0 + BAND_WIDTH,
        "r_2b": r_0 + 2 * BAND_WIDTH,
        "m_R": 2 * np.pi * R * np.cos(alpha_0) / BAND_WIDTH,
        "m_0": 2 * np.pi * r_0 * np.cos(alpha_0) / BAND_WIDTH,
        "n_R": T_R / (2 * PLY_THICKNESS),
        "R": R,
    }


def _a_vec(g):
    """Cubic coefficients of the thickness in the transition region."""
    r_0, r_b, r_2b = g["r_0"], g["r_b"], g["r_2b"]
    m_R, m_0, n_R = g["m_R"], g["m_0"], g["n_R"]

    def pd(degree):
        return r_2b**degree - r_0**degree

    # Constraints: value at r_0, value and slope at r_2b, and total fiber volume
    A = np.array(
        [
            [1.0, r_0, r_0**2, r_0**3],
            [1.0, r_2b, r_2b**2, r_2b**3],
            [0.0, 1.0, 2 * r_2b, 3 * r_2b**2],
            [
                np.pi * pd(2),
                2 * np.pi / 3 * pd(3),
                np.pi / 2 * pd(4),
                2 * np.pi / 5 * pd(5),
            ],
        ]
    )

    c_0 = T_R * np.pi * g["R"] * np.cos(g["alpha_0"]) / (m_0 * BAND_WIDTH)
    c_1 = (
        m_R
        * n_R
        / np.pi
        * (np.arccos(r_0 / r_2b) - np.arccos(r_b / r_2b))
        * PLY_THICKNESS
    )
    c_2 = (
        m_R
        * n_R
        / np.pi
        * (
            r_0 / (r_2b * np.sqrt(pd(2)))
            - r_b / (r_2b * np.sqrt(r_2b**2 - r_b**2))
        )
        * PLY_THICKNESS
    )
    int_1, _ = quad(lambda r: r * np.arccos(r_0 / r), r_0, r_b)
    int_2, _ = quad(
        lambda r: r * np.arccos(r_0 / r_2b) - r * np.arccos(r_b / r_2b), r_b, r_2b
    )
    c_3 = 2.0 * m_R * n_R * PLY_THICKNESS * (int_1 + int_2)

    return np.linalg.solve(A, np.array([c_0, c_1, c_2, c_3]))


def winding_thickness(r, alpha_0_deg, R=R_VESSEL):
    """Ply thickness [mm] at radius `r` for a band wound at `alpha_0_deg`.

    Piecewise: a cubic for `r <= r_0 + 2b` and the band-coverage law beyond it.
    Returns zero inside the turnaround radius and wherever the build-up falls
    below `MINIMUM_THICKNESS_THRESHOLD`.
    """
    r_np = np.asarray(
        r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r, dtype=float
    )
    g = _band_globals(alpha_0_deg, R)
    r_0, r_b, r_2b = g["r_0"], g["r_b"], g["r_2b"]

    t = np.zeros(r_np.shape)

    # Region 1: cubic transition, only where it stays positive
    t_1 = np.poly1d(np.flip(_a_vec(g)))(r_np)
    t += t_1 * ((t_1 >= 0.0) & (r_np <= r_2b))

    # Region 2: band coverage. Clip the arccos arguments to keep them in [-1, 1]
    safe_r = np.where(r_np > 0.0, r_np, 1e-12)
    arg_1 = np.clip(r_0 / safe_r, None, 1.0)
    arg_2 = np.clip(r_b / safe_r, None, 1.0)
    t_2 = (g["m_R"] * g["n_R"] / np.pi) * (np.arccos(arg_1) - np.arccos(arg_2))
    t += np.nan_to_num(t_2 * PLY_THICKNESS) * (r_np > r_2b)

    # No fiber inside the turnaround radius, and drop negligible build-up
    t[r_np < r_0] = 0.0
    t[t <= MINIMUM_THICKNESS_THRESHOLD] = 0.0

    return torch.as_tensor(t, dtype=torch.get_default_dtype())


def geodesic_path(alpha_0_deg, r_of_x, x_equator, n_points=400, R=R_VESSEL):
    """3-D polyline of one geodesic circuit from the equator to the turnaround.

    Integrates `dphi/ds = tan(alpha) / r` along the meridian, where `s` is
    meridional arc length, and returns points `(x, r cos(phi), r sin(phi))`.
    Stops at the turnaround radius, so the polyline ends where alpha = 90 deg.
    """
    r_0 = turnaround_radius(alpha_0_deg, R)

    # March along the meridian from the equator towards the dome apex, stopping
    # at the turnaround. `r_of_x` is (x, r) with r increasing in x, so the
    # turnaround station is interpolated directly on it.
    x_end = float(np.interp(r_0, r_of_x[1], r_of_x[0]))
    x = np.linspace(x_equator, x_end, n_points)
    r = np.interp(x, r_of_x[0], r_of_x[1])
    r = np.clip(r, r_0, None)

    # Meridional arc length and the winding angle along it
    ds = np.hypot(np.diff(x), np.diff(r))
    alpha = np.arcsin(np.clip(r_0 / np.maximum(r, 1e-12), None, 1.0))

    # dphi = tan(alpha) / r ds, integrated with the midpoint rule
    tan_mid = np.tan(0.5 * (alpha[:-1] + alpha[1:]))
    r_mid = 0.5 * (r[:-1] + r[1:])
    phi = np.concatenate([[0.0], np.cumsum(tan_mid / r_mid * ds)])

    return np.column_stack([x, r * np.cos(phi), r * np.sin(phi)])


# --- Per-element winding fields ---

# Element centroid radius: both the angle and the thickness depend on r only, so
# elements at the same axial station receive identical values.
_elements = import_mesh(get_data("copv.vtu"), alu).elements
_centroids = mesh_nodes[_elements].mean(dim=1)
r_elem = torch.sqrt(_centroids[:, 1] ** 2 + _centroids[:, 2] ** 2)

# Meridian profile r(x) taken from the mesh, sorted and made monotone so that it
# can be interpolated to locate an axial station from a radius (and vice versa).
_x_nodes = mesh_nodes[:, 0].numpy()
_r_nodes = torch.sqrt(mesh_nodes[:, 1] ** 2 + mesh_nodes[:, 2] ** 2).numpy()
_order = np.argsort(_x_nodes)
_x_mer, _unique = np.unique(_x_nodes[_order], return_index=True)
_r_mer = np.maximum.accumulate(_r_nodes[_order][_unique])
meridian = (_x_mer, _r_mer)

if GEODESIC_WINDING:
    # Fail loudly if the geodesic would never turn around inside the vessel
    r_0_nominal = check_turnaround(NOMINAL_ANGLE, float(r_elem.min()))

    alpha_elem = clairaut_angle(r_elem, NOMINAL_ANGLE)
    t_elem = winding_thickness(r_elem, NOMINAL_ANGLE)

    print(
        f"Geodesic winding: alpha_0 = {NOMINAL_ANGLE} deg -> turnaround at "
        f"r_0 = {r_0_nominal:.3f} mm"
    )
    print(
        f"  angle  {torch.rad2deg(alpha_elem).min():5.2f} .. "
        f"{torch.rad2deg(alpha_elem).max():5.2f} deg"
    )
    print(f"  thickness {t_elem.min():5.3f} .. {t_elem.max():5.3f} mm")
    print(f"  elements past turnaround (no fiber): {int((t_elem == 0).sum())}")

    # --- 2D graph of the winding angle and thickness along the mandrel axis ---
    # The mandrel axis is global x for this mesh, so both are plotted against the
    # axial coordinate of each element centroid.
    x_elem = _centroids[:, 0]
    _sort = torch.argsort(x_elem)
    x_sorted = x_elem[_sort].numpy()

    fig, ax_angle = plt.subplots()
    fig.suptitle("Geodesic winding angle and ply thickness", fontsize=14)

    # Left axis: winding angle
    line_angle, = ax_angle.plot(
        x_sorted,
        torch.rad2deg(alpha_elem[_sort]).numpy(),
        "-o",
        color="tab:blue",
        markersize=3,
        label=f"winding angle (alpha_0 = {NOMINAL_ANGLE}°)",
    )
    line_turn = ax_angle.axhline(
        90.0, color="k", ls="--", lw=1, label="turnaround (alpha = 90°)"
    )
    line_nom = ax_angle.axhline(
        NOMINAL_ANGLE,
        color="grey",
        ls=":",
        lw=1,
        label=f"nominal (alpha_0 = {NOMINAL_ANGLE}°)",
    )
    ax_angle.set_xlabel("axial coordinate along mandrel axis -- x (mm)")
    ax_angle.set_ylabel("winding angle -- alpha (deg)", color="tab:blue")
    ax_angle.tick_params(axis="y", labelcolor="tab:blue")
    ax_angle.grid(alpha=0.3)

    # Right axis: ply thickness, which vanishes past the turnaround
    ax_thick = ax_angle.twinx()
    line_thick, = ax_thick.plot(
        x_sorted,
        t_elem[_sort].numpy(),
        "-s",
        color="tab:green",
        markersize=3,
        label="ply thickness",
    )
    line_r0 = ax_thick.axvline(
        float(np.interp(r_0_nominal, _r_mer, _x_mer)),
        color="magenta",
        ls="-.",
        lw=1,
        label=f"turnaround station (r_0 = {r_0_nominal:.1f} mm)",
    )
    ax_thick.set_ylabel("ply thickness -- t (mm)", color="tab:green")
    ax_thick.tick_params(axis="y", labelcolor="tab:green")
    ax_thick.set_ylim(bottom=0.0)

    ax_angle.legend(
        handles=[line_angle, line_thick, line_turn, line_nom, line_r0],
        loc="upper left",
        fontsize=8,
    )

    plt.show()

    # Helical plies alternate +/- the geodesic angle; the liner stays isotropic
    layup = Laminate(
        materials=[alu, cfrp, cfrp, cfrp, cfrp],
        thicknesses=[torch.full_like(r_elem, 4.0)] + 4 * [t_elem],
        angles=[torch.zeros_like(r_elem)] + 2 * [alpha_elem, -alpha_elem],
    )
else:
    layup = nominal_layup


# Geometry, boundary conditions and loading
#
# One octant of the vessel - a dome-capped half cylinder - is imported from a VTU
# file and assigned the laminate. The default `orientation` (global x, the vessel
# axis) projects onto each element as the meridional direction, so ply angles are
# measured from the meridian.
#
# Three symmetry planes close the octant: the cut at x = 0 and the y = 0 and
# z = 0 planes. Each plane fixes the out-of-plane translation and the two
# in-plane rotations. Internal pressure is applied as outward nodal forces, with
# every element contributing p * area / 3 along its outward normal to each of its
# three nodes.

path = get_data("copv.vtu")
copv = import_mesh(path, layup)
nodes = copv.nodes
elements = copv.elements

# Symmetry at x
x_symm = nodes[:, 0] > -0.1
copv.constraints[x_symm, 0] = True
copv.constraints[x_symm, 4] = True
copv.constraints[x_symm, 5] = True
# Symmetry at y
y_symm = nodes[:, 1] < 0.1
copv.constraints[y_symm, 1] = True
copv.constraints[y_symm, 3] = True
copv.constraints[y_symm, 5] = True
# Symmetry at z
z_symm = nodes[:, 2] < 0.1
copv.constraints[z_symm, 2] = True
copv.constraints[z_symm, 3] = True
copv.constraints[z_symm, 4] = True

# Internal pressure acting along the outward element normals
pressure = 10.0  # internal gauge pressure [MPa]
surface = torch.ones(copv.n_nod, dtype=torch.bool)
copv.forces[:, 0:3] = copv.integrate_surface_load(surface, pressure)

# Show
copv.plot(bcs=True)

# --- Burst analysis strength allowables ---
#
# Unidirectional carbon/epoxy ply strengths [MPa], used by the max-stress and
# Tsai-Wu criteria below. `S_12` is the in-plane shear strength.
CFRP_XT = 2500.0  # longitudinal tension (fiber direction)
CFRP_XC = 1500.0  # longitudinal compression
CFRP_YT = 60.0  # transverse tension
CFRP_YC = 200.0  # transverse compression
CFRP_S12 = 90.0  # in-plane shear

ALU_YIELD = 300.0  # liner yield strength [MPa]


# Fiber orientations
#
# The element frame `copv.t` = [dir1, dir2, normal] is built by
# `eval_shape_functions`: `dir1` is the global reference direction (default
# global x) projected onto the element surface, and `dir2` is in-plane and
# perpendicular to it. The actual fiber direction of a ply is the reference axis
# rotated in-plane by its angle,
#
#     fiber = cos(angle) * dir1 + sin(angle) * dir2,
#
# which lies exactly in the element plane since dir1 and dir2 both do. Arrows are
# drawn from each element centroid.

# Populate the element frame without solving (the solve would do this too)
copv.eval_shape_functions(copv.etype.ipoints[0])

dir1 = copv.t[:, 0, :]  # element 0-deg axis (meridional here)
dir2 = copv.t[:, 1, :]  # in-plane, perpendicular to dir1 (hoop)
normal = copv.t[:, 2, :]  # element normal, outward for this mesh


def fiber_direction(angle):
    """In-plane unit vector at `angle` (radians) from the element 0-deg axis.

    Accepts a scalar (same angle everywhere) or a per-element `(n_elem,)` tensor,
    e.g. a true filament-wound path where the angle varies over the dome.
    """
    a = torch.as_tensor(angle, dtype=dir1.dtype)
    if a.dim() == 0:
        a = a.expand(copv.n_elem)
    return torch.cos(a)[:, None] * dir1 + torch.sin(a)[:, None] * dir2


# `plot()` colors the arrow sets red, green, blue in order and draws no legend, so
# the plotter is built here to add matching legend entries.
fiber = fiber_direction(copv.section.angles[1])

pl = pyvista.Plotter()
copv.plot(orientations=torch.stack([dir1, fiber], dim=1), plotter=pl)

legend = [
    ("Reference material direction (0°)", "red"),
    (
        "Fiber direction (geodesic)" if GEODESIC_WINDING else "Fiber direction (+15°)",
        "green",
    ),
]


if GEODESIC_WINDING:
    # One geodesic circuit from the equator to the 90 deg turnaround, using the
    # meridian profile built with the winding fields above
    path_pts = geodesic_path(NOMINAL_ANGLE, meridian, x_equator=float(_x_mer.max()))
    pl.add_mesh(
        pyvista.lines_from_points(path_pts),
        color="magenta",
        line_width=5,
        render_lines_as_tubes=True,
    )
    # Mark where the fiber reaches 90 deg and turns around
    pl.add_mesh(
        pyvista.Sphere(radius=3.0, center=path_pts[-1]),
        color="black",
    )
    legend += [
        ("Geodesic winding path", "magenta"),
        (f"Turnaround (r_0 = {r_0_nominal:.1f} mm)", "black"),
    ]

pl.add_legend(legend, bcolor="white")
pl.show()


# Solve
#
# A single linear solve, since all plies are linear elastic.

u, f, sigma, _, _ = copv.solve(aggregate_integration_points=False)


# Results
#
# The nodal radial expansion is recovered by projecting the displacement onto the
# radial direction.

radius = torch.sqrt(nodes[:, 1] ** 2 + nodes[:, 2] ** 2)
u_radial = (u[:, 1] * nodes[:, 1] + u[:, 2] * nodes[:, 2]) / radius.clamp(min=1e-6)

copv.plot(u=10.0 * u[:, :3], node_property={"Radial expansion [mm]": u_radial})


# Ply thickness distribution
#
# With the geodesic model each helical ply thickens over the dome and vanishes
# past the turnaround, so the total laminate thickness varies over the surface.

copv.plot(
    element_property={
        "Helical ply thickness [mm]": copv.section.thicknesses[1],
        "Total laminate thickness [mm]": copv.section.thickness,
    }
)


# Per-ply stresses
#
# In the element frame the in-plane axes are meridional (x', the winding
# reference) and hoop (y'). The aluminium liner carries most of the hoop load,
# while the helical CFRP plies carry the meridional/axial load.

n_layers = copv.section.n_layers
n_simpson = copv.section.n_simpson

meridional = sigma[:, :, 0, 0].reshape(n_layers, n_simpson, -1)
hoop = sigma[:, :, 1, 1].reshape(n_layers, n_simpson, -1)

if GEODESIC_WINDING:
    labels = ["Al liner"] + 2 * ["CFRP +geo", "CFRP -geo"]
else:
    labels = ["Al liner", "CFRP +15°", "CFRP -15°", "CFRP +15°", "CFRP -15°"]

for k, label in enumerate(labels):
    # Elements where a ply has no thickness carry no load, so their stress is
    # meaningless and must not enter the reported maxima
    present = copv.section.thicknesses[k] > 0.0
    if not bool(present.any()):
        print(f"{label:10s}: not present anywhere")
        continue
    print(
        f"{label:10s}: Max. meridional : {meridional[k][:, present].max():6.2f} MPa,"
        f" Max. hoop : {hoop[k][:, present].max():6.2f} MPa"
    )


# Burst analysis
#
# The stresses above are in the *element* frame (meridional/hoop), because
# `Laminate.vectorize` rotates each ply stiffness into it. Composite failure is
# fiber-direction dependent, so the burst check first rotates the ply stress into
# the material frame (1 = fiber, 2 = transverse),
#
#     sigma_mat = Q sigma_elem Q^T,   Q = planar_rotation(ply angle),
#
# and then evaluates two criteria per ply:
#
# * max-stress: the largest of the individual strength ratios, so the failure
#   mode is identifiable (fiber, transverse, or shear);
# * Tsai-Wu: an interactive quadratic criterion, the usual burst predictor.
#
# Both are reported as a failure index FI, where FI >= 1 means failure, and the
# burst pressure is estimated by linear extrapolation p_burst = p / FI_max. That
# scaling is exact only while the response stays linear, which it is here (a
# single linear solve), but it ignores liner yielding and progressive ply
# failure, so it is a first-order estimate rather than a certified burst load.

# Tsai-Wu coefficients from the ply strengths
F_1 = 1.0 / CFRP_XT - 1.0 / CFRP_XC
F_2 = 1.0 / CFRP_YT - 1.0 / CFRP_YC
F_11 = 1.0 / (CFRP_XT * CFRP_XC)
F_22 = 1.0 / (CFRP_YT * CFRP_YC)
F_66 = 1.0 / CFRP_S12**2
# Interaction term, the common approximation F_12 = -sqrt(F_11 F_22) / 2
F_12 = -0.5 * np.sqrt(F_11 * F_22)

# Element-frame stress components per layer, [n_layers, n_simpson, n_elem]
shear = sigma[:, :, 0, 1].reshape(n_layers, n_simpson, -1)

fi_max_stress = torch.zeros(n_layers, copv.n_elem)
fi_tsai_wu = torch.zeros(n_layers, copv.n_elem)
mode = {}

print(f"\nBurst analysis at p = {pressure:.1f} MPa")

for k, label in enumerate(labels):
    present = copv.section.thicknesses[k] > 0.0

    # Rotate the through-thickness stations into the ply material frame. A ply
    # angle may be a scalar (nominal model) or per-element (geodesic model), so
    # the rotation is broadcast to the mesh either way.
    Q = planar_rotation(copv.section.angles[k])
    if Q.dim() == 2:
        Q = Q.expand(copv.n_elem, 2, 2)
    s_elem = torch.stack(
        [
            torch.stack([meridional[k], shear[k]], dim=-1),
            torch.stack([shear[k], hoop[k]], dim=-1),
        ],
        dim=-2,
    )  # [n_simpson, n_elem, 2, 2]
    s_mat = torch.einsum("eij,sejk,elk->seil", Q, s_elem, Q)

    s_1 = s_mat[..., 0, 0]
    s_2 = s_mat[..., 1, 1]
    t_12 = s_mat[..., 0, 1]

    if isinstance(copv.section.materials[k], IsotropicElasticityPlaneStress):
        # Metallic liner: von Mises against yield, in plane stress
        vm = torch.sqrt(s_1**2 - s_1 * s_2 + s_2**2 + 3.0 * t_12**2)
        fi_ms = vm / ALU_YIELD
        fi_tw = fi_ms  # no composite criterion applies to the liner
        modes = torch.zeros_like(fi_ms)
    else:
        # Max-stress ratios, keeping tension and compression separate
        r_1 = torch.where(s_1 >= 0.0, s_1 / CFRP_XT, -s_1 / CFRP_XC)
        r_2 = torch.where(s_2 >= 0.0, s_2 / CFRP_YT, -s_2 / CFRP_YC)
        r_6 = t_12.abs() / CFRP_S12
        ratios = torch.stack([r_1, r_2, r_6], dim=0)
        fi_ms, modes = ratios.max(dim=0)

        # Tsai-Wu is a quadratic in the stress; its failure index is the factor
        # by which the stress may be scaled, i.e. the root of a X^2 + b X = 1
        a = F_11 * s_1**2 + F_22 * s_2**2 + F_66 * t_12**2 + 2.0 * F_12 * s_1 * s_2
        b = F_1 * s_1 + F_2 * s_2
        # Load factor to failure, then FI = 1 / factor
        disc = torch.sqrt(b**2 + 4.0 * a)
        factor = torch.where(a > 0.0, (-b + disc) / (2.0 * a), torch.inf)
        fi_tw = 1.0 / factor.clamp(min=1e-30)

    # Worst station through the thickness, and no credit for absent plies
    fi_max_stress[k] = torch.where(present, fi_ms.max(dim=0).values, 0.0)
    fi_tsai_wu[k] = torch.where(present, fi_tw.max(dim=0).values, 0.0)

    if not bool(present.any()):
        print(f"  {label:10s}: not present anywhere")
        continue

    # Report the governing failure mode of the critical element
    worst = int(fi_max_stress[k].argmax())
    if isinstance(copv.section.materials[k], IsotropicElasticityPlaneStress):
        mode_name = "von Mises"
    else:
        names = ["fiber", "transverse", "shear"]
        mode_name = names[int(modes[:, worst][fi_ms[:, worst].argmax()])]
    mode[label] = mode_name

    print(
        f"  {label:10s}: FI(max-stress) = {fi_max_stress[k].max():5.3f}"
        f"  FI(Tsai-Wu) = {fi_tsai_wu[k].max():5.3f}"
        f"  mode: {mode_name}"
    )

# Envelope over all plies: the critical index anywhere in the laminate
fi_env_ms = fi_max_stress.max(dim=0).values
fi_env_tw = fi_tsai_wu.max(dim=0).values

fi_crit = float(fi_env_tw.max())
p_burst = pressure / fi_crit if fi_crit > 0.0 else float("inf")
print(f"  {'envelope':10s}: FI(Tsai-Wu) = {fi_crit:5.3f}")
print(f"  estimated burst pressure (linear scaling) = {p_burst:6.2f} MPa")

# The critical ply and the two envelopes, so the burst-critical region is visible
copv.plot(
    element_property={
        "FI max-stress (envelope)": fi_env_ms,
        "FI Tsai-Wu (envelope)": fi_env_tw,
        "FI Tsai-Wu (helical ply 1)": fi_tsai_wu[1],
        "FI von Mises (Al liner)": fi_max_stress[0],
    }
)
