"""
Compaction pressure distribution of a hollow elastic (rubber) roller pressed
against a rigid substrate — flat plate or a cylinder portion of arbitrary radius.

Inputs : outer diameter, inner (bore) diameter, width, applied force,
         rubber constitutive model, substrate curvature.
Outputs: normal contact pressure distribution p(x), contact half width,
         radial deflection of the roller wall, hub approach (crush).

Model
-----
* 2D plane-strain slice, element thickness = roller width b. Valid while
  b >> contact half-width; the free-edge relaxation of a real roller is not
  captured (it lowers p by ~10-20 % over ~1 wall thickness at each end).
* Finite strain + hyperelastic rubber (Neo-Hooke / Mooney-Rivlin / Yeoh)
  through torchfem.materials.HyperelasticPlaneStrain and solve(nlgeom=True),
  total-Lagrangian formulation.
* Contact: torch-fem has no contact module, so a node-to-rigid-surface
  active set (Signorini) is implemented here. Candidate nodes of the outer
  ring are constrained onto the substrate profile through their vertical DOF,
  released where the reaction becomes tensile, and activated where the gap
  becomes negative. The horizontal DOF stays free => the reaction is vertical.
  On a curved substrate this is equivalent to allowing a spurious tangential
  traction p*tan(theta) at the surface inclination theta ~ a/R_sub, i.e. an
  effective friction coefficient of a few percent for a/R_sub < 0.05.
  For a flat plate the formulation is exact frictionless Signorini contact.
* Force control: the bore is displacement driven (rigid shaft, bonded),
  and the approach delta is iterated with a log-secant scheme until the sum
  of the contact reactions equals the applied force F.
* Contact pressure = nodal reaction projected on the substrate normal,
  divided by the deformed tributary length and by the roller width
  => true (Cauchy) pressure in MPa.

Units: mm, N, MPa (consistent).

Requires: torch-fem >= 0.8 (pip install torch-fem), matplotlib.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from torchfem import Planar
from torchfem.materials import HyperelasticPlaneStrain

torch.set_default_dtype(torch.float64)  # Newton on rubber needs float64


# ----------------------------------------------------------------------------
# 1. Rubber: strain energy densities psi(F, params), F is the 3x3 def. gradient
# ----------------------------------------------------------------------------


@dataclass
class RubberModel:
    """Hyperelastic rubber definition."""

    psi: Callable[[Tensor, Tensor], Tensor]
    params: list[float]
    mu0: float  # initial shear modulus [MPa], for reference/Hertz
    nu: float  # (slight) compressibility used for the volumetric term
    name: str = "rubber"

    @property
    def E0(self) -> float:
        """Initial Young's modulus [MPa]."""
        return 2.0 * self.mu0 * (1.0 + self.nu)

    def material(self, n_elem: int) -> HyperelasticPlaneStrain:
        return HyperelasticPlaneStrain(self.psi, self.params).vectorize(n_elem)


def _bulk_modulus(mu0: float, nu: float) -> float:
    return 2.0 * mu0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


def _det3(F: Tensor) -> Tensor:
    """Determinant of a 3x3 tensor by cofactor expansion.

    `torch.linalg.det` must NOT be used here. torch-fem builds the material
    tangent with jacrev(jacrev(psi)), and the second derivative of
    `torch.linalg.det` is routed through an SVD-based formula that divides by
    (s_i^2 - s_j^2). At F = I -- the state at the start of every increment --
    the singular values are all equal, so the whole tangent comes out NaN, the
    global stiffness matrix is exactly singular and Newton "diverges" on the
    very first iteration. The cofactor expansion is a plain polynomial and is
    differentiable to any order everywhere.
    """
    return (
        F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
        - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
        + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0])
    )


def _invariants(F: Tensor):
    """J, I1bar, I2bar of the isochoric right Cauchy-Green tensor."""
    J = _det3(F)
    Jc = torch.clamp(J, min=1.0e-3)  # guard against inversion during Newton
    C = F.transpose(-1, -2) @ F
    I1 = (F * F).sum()
    I2 = 0.5 * (I1**2 - (C * C).sum())
    return J, Jc ** (-2.0 / 3.0) * I1, Jc ** (-4.0 / 3.0) * I2


def neo_hooke(mu: float, nu: float = 0.47) -> RubberModel:
    """psi = mu/2 (I1bar - 3) + kappa/2 (J-1)^2."""

    def psi(F: Tensor, p: Tensor) -> Tensor:
        J, I1b, _ = _invariants(F)
        return 0.5 * p[0] * (I1b - 3.0) + 0.5 * p[1] * (J - 1.0) ** 2

    return RubberModel(psi, [mu, _bulk_modulus(mu, nu)], mu, nu, "Neo-Hooke")


def mooney_rivlin(C10: float, C01: float, nu: float = 0.47) -> RubberModel:
    """psi = C10 (I1bar-3) + C01 (I2bar-3) + kappa/2 (J-1)^2, mu0 = 2(C10+C01)."""
    mu = 2.0 * (C10 + C01)

    def psi(F: Tensor, p: Tensor) -> Tensor:
        J, I1b, I2b = _invariants(F)
        return p[0] * (I1b - 3.0) + p[1] * (I2b - 3.0) + 0.5 * p[2] * (J - 1.0) ** 2

    return RubberModel(
        psi, [C10, C01, _bulk_modulus(mu, nu)], mu, nu, "Mooney-Rivlin"
    )


def yeoh(C10: float, C20: float = 0.0, C30: float = 0.0, nu: float = 0.47):
    """psi = sum Ci0 (I1bar-3)^i + kappa/2 (J-1)^2, mu0 = 2 C10.

    Good default for carbon-black filled rubber in compression:
    C20 < 0 softens the mid-strain range, C30 > 0 gives the final stiffening.
    """
    mu = 2.0 * C10

    def psi(F: Tensor, p: Tensor) -> Tensor:
        J, I1b, _ = _invariants(F)
        x = I1b - 3.0
        return (
            p[0] * x + p[1] * x**2 + p[2] * x**3 + 0.5 * p[3] * (J - 1.0) ** 2
        )

    return RubberModel(
        psi, [C10, C20, C30, _bulk_modulus(mu, nu)], mu, nu, "Yeoh"
    )


def shore_a_to_yeoh(shore_a: float, nu: float = 0.47) -> RubberModel:
    """Rough engineering estimate of a Yeoh fit from Shore A hardness.

    E0 [MPa] from Gent's relation, C10 = E0/6, plus a mild softening/stiffening
    pair. Replace by a fit of your own uniaxial/planar test data when available.
    """
    E0 = 0.0981 * (56.0 + 7.62336 * shore_a) / (0.137505 * (254.0 - 2.54 * shore_a))
    C10 = E0 / 6.0
    return yeoh(C10, -0.05 * C10, 0.02 * C10, nu)


# ----------------------------------------------------------------------------
# 2. Geometry
# ----------------------------------------------------------------------------


@dataclass
class Roller:
    d_out: float  # external diameter [mm]
    d_in: float  # bore diameter [mm]
    width: float  # axial width [mm]

    @property
    def r_out(self) -> float:
        return 0.5 * self.d_out

    @property
    def r_in(self) -> float:
        return 0.5 * self.d_in


@dataclass
class Substrate:
    """Rigid counter-surface.

    kind = "flat"    : plane y = -r_out
    kind = "convex"  : cylinder of radius R seen from outside (mandrel, tube)
                       -> smaller contact patch, higher peak pressure
    kind = "concave" : cylindrical cradle of radius R (R must exceed r_out)
                       -> larger contact patch, lower peak pressure
    """

    kind: Literal["flat", "convex", "concave"] = "flat"
    radius: float = math.inf  # [mm], positive

    def signed_curvature(self) -> float:
        if self.kind == "flat":
            return 0.0
        return (1.0 if self.kind == "convex" else -1.0) / self.radius


@dataclass
class MeshOpts:
    n_r: int = 8  # elements through the wall
    n_theta: int = 160  # elements around the circumference
    radial_bias: float = 0.55  # < 1 refines towards the outer surface
    fine_window: float = math.radians(30.0)  # sector kept at the fine size
    coarse_ratio: float = 6.0  # element size far from contact / fine size
    candidate_window: float = math.radians(70.0)  # contact search sector


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation (xp must be increasing)."""
    i = torch.clamp(torch.searchsorted(xp, x), 1, len(xp) - 1)
    w = (x - xp[i - 1]) / (xp[i] - xp[i - 1])
    return fp[i - 1] + w * (fp[i] - fp[i - 1])


def graded_angles(opts: MeshOpts, samples: int = 4001) -> Tensor:
    """Node angles, fine and uniform in the contact sector, coarse elsewhere.

    The element size is prescribed as a smoothstep between `fine_window` and
    twice that angle, and the node positions follow from the cumulative
    density, so the transition is smooth (no size jump inside the mesh).
    """
    phi = torch.linspace(-math.pi, math.pi, samples)
    x = torch.clamp((phi.abs() - opts.fine_window) / opts.fine_window, 0.0, 1.0)
    h = 1.0 + (opts.coarse_ratio - 1.0) * (3.0 * x**2 - 2.0 * x**3)
    dens = 1.0 / h
    cum = torch.cat(
        [
            torch.zeros(1),
            torch.cumsum(0.5 * (dens[1:] + dens[:-1]) * phi.diff(), dim=0),
        ]
    )
    cum = cum / cum[-1]
    q = torch.linspace(0.0, 1.0, opts.n_theta + 1)[:-1]  # periodic
    return -0.5 * math.pi + _interp(q, cum, phi)


def build_annulus(roller: Roller, opts: MeshOpts):
    """Structured graded quad (Quad1) mesh of the annulus.

    Returns nodes [n_nod, 2], elements [n_elem, 4], inner ring and outer ring
    node indices (outer ring sorted by increasing angle) and their angles.
    The bottom of the roller (contact side) is at theta = -pi/2.
    """
    nr, nt = opts.n_r, opts.n_theta

    t = torch.linspace(0.0, 1.0, nr + 1)
    r = roller.r_in + (roller.r_out - roller.r_in) * t**opts.radial_bias

    theta = graded_angles(opts)

    R, TH = torch.meshgrid(r, theta, indexing="ij")
    nodes = torch.stack([R * torch.cos(TH), R * torch.sin(TH)], dim=-1).reshape(-1, 2)

    i = torch.arange(nr).repeat_interleave(nt)
    j = torch.arange(nt).repeat(nr)
    jp = (j + 1) % nt

    def nid(ii, jj):
        return ii * nt + jj

    # counter-clockwise ordering: (r,theta) is a right-handed frame
    elements = torch.stack(
        [nid(i, j), nid(i + 1, j), nid(i + 1, jp), nid(i, jp)], dim=1
    )

    inner = torch.arange(nt)  # i = 0
    outer = torch.arange(nr * nt, (nr + 1) * nt)  # i = nr
    return nodes, elements, inner, outer, theta


# ----------------------------------------------------------------------------
# 3. Contact model
# ----------------------------------------------------------------------------


@dataclass
class ContactResult:
    delta: float  # hub approach [mm]
    force: float  # resultant normal force [N]
    x: Tensor  # deformed x of the contact nodes [mm]
    pressure: Tensor  # normal contact pressure [MPa]
    half_width: float  # contact half width a [mm]
    p_max: float  # peak pressure [MPa]
    p_mean: float  # F / (2a b) [MPa]
    theta_out: Tensor  # angles of the outer ring [rad]
    u_radial: Tensor  # radial displacement of the outer ring [mm]
    max_flattening: float  # max radial compression of the wall [mm]
    u: Tensor  # full displacement field [n_nod, 2]
    active: Tensor  # boolean mask of contacting nodes (candidate numbering)
    nodes_contact: Tensor  # global node ids of the contact patch, sorted by x
    n_solves: int = 0
    t_fem: float = 0.0  # cumulated wall time inside torch-fem solve() [s]
    t_total: float = 0.0  # wall time of the whole force-controlled run [s]

    @property
    def t_contact(self) -> float:
        """Time spent outside torch-fem: active set, gap search, post-processing."""
        return max(self.t_total - self.t_fem, 0.0)


class RollerContact:
    """Hollow hyperelastic roller pressed on a rigid substrate.

    Frictionless. Only the vertical DOF of a candidate node is tied to the
    substrate; its horizontal DOF stays free, so the contact reaction has no
    tangential component and the roller surface is free to slide and to bulge
    sideways out of the patch. Nothing here models stick/slip, rolling
    resistance or the hysteresis of the rubber -- this is a static normal
    indentation problem. See `tangential_bias()` for the one caveat on a
    curved substrate.
    """

    def __init__(
        self,
        roller: Roller,
        rubber: RubberModel,
        substrate: Substrate = Substrate(),
        mesh: MeshOpts = MeshOpts(),
        n_increments: int = 5,
        verbose: bool = False,
    ):
        if substrate.kind == "concave" and substrate.radius <= 1.05 * roller.r_out:
            raise ValueError("Concave substrate radius must exceed the roller radius.")

        self.roller, self.rubber, self.substrate = roller, rubber, substrate
        self.mesh_opts, self.verbose = mesh, verbose
        self.increments = torch.linspace(0.0, 1.0, n_increments)

        t0 = time.perf_counter()
        nodes, elements, inner, outer, theta = build_annulus(roller, mesh)
        self.nodes, self.elements = nodes, elements
        self.inner, self.outer, self.theta_out = inner, outer, theta

        self.model = Planar(
            nodes,
            elements,
            rubber.material(len(elements)),
            thickness=roller.width,
        )

        # contact candidates: outer ring nodes within the search sector
        d = torch.remainder(theta + 0.5 * math.pi + math.pi, 2 * math.pi) - math.pi
        self.cand_local = d.abs() < mesh.candidate_window  # mask on the outer ring
        self.cand = outer[self.cand_local]
        self.n_solves = 0
        self.t_setup = time.perf_counter() - t0  # mesh + model assembly [s]
        self.t_fem = 0.0  # cumulated time inside torch-fem solve()

    # -- rigid surface -------------------------------------------------------

    def surface(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Absolute height y_s(x) and slope dy_s/dx of the rigid substrate."""
        y0 = -self.roller.r_out  # touching point at x = 0
        if self.substrate.kind == "flat":
            return torch.full_like(x, y0), torch.zeros_like(x)
        R = self.substrate.radius
        xc = torch.clamp(x, -0.98 * R, 0.98 * R)
        root = torch.sqrt(R**2 - xc**2)
        if self.substrate.kind == "convex":  # surface drops away from x = 0
            return y0 + root - R, -xc / root
        return y0 + R - root, xc / root  # concave cradle

    def normal(self, x: Tensor) -> Tensor:
        """Unit outward normal of the substrate (pointing into the roller)."""
        _, sl = self.surface(x)
        n = torch.stack([-sl, torch.ones_like(sl)], dim=-1)
        return n / n.norm(dim=-1, keepdim=True)

    # -- one solve at prescribed approach ------------------------------------

    def solve_penetration(
        self, delta: float, active: Tensor | None = None, max_as_iter: int = 12
    ) -> ContactResult:
        """Active-set contact solve for a prescribed hub approach delta."""
        model, nodes = self.model, self.nodes
        x0, y0 = nodes[:, 0], nodes[:, 1]

        if active is None:  # geometric initial guess (rigid translation)
            ys, _ = self.surface(x0[self.cand])
            active = (y0[self.cand] - delta) < ys
        active = active.clone()

        ux = torch.zeros(len(nodes))  # tangential position of contact nodes
        u = f = None

        for _ in range(max_as_iter):
            if not active.any():  # keep the bottom node to avoid an empty patch
                active[len(active) // 2] = True

            idx = self.cand[active]
            model.constraints[:] = False
            model.displacements[:] = 0.0
            model.constraints[self.inner, :] = True
            model.displacements[self.inner, 1] = -delta
            ys, _ = self.surface(x0[idx] + ux[idx])
            model.constraints[idx, 1] = True
            model.displacements[idx, 1] = ys - y0[idx]

            t0 = time.perf_counter()
            u, f, *_ = self.model.solve(
                increments=self.increments,
                nlgeom=True,
                max_iter=25,
                rtol=1e-7,
                verbose=self.verbose,
            )
            self.t_fem += time.perf_counter() - t0
            self.n_solves += 1
            ux = u[:, 0]

            # release nodes with tensile (pulling) reaction. The small negative
            # threshold prevents chattering of the two nodes at the patch edge.
            R = f[idx, 1]
            keep = R > -1e-3 * float(R.abs().mean())
            # activate candidates that penetrate the substrate
            xd, yd = x0[self.cand] + u[self.cand, 0], y0[self.cand] + u[self.cand, 1]
            ysc, _ = self.surface(xd)
            gap = yd - ysc
            tol = 1e-6 * self.roller.r_out

            new = active.clone()
            new[active.nonzero().ravel()] = keep
            new |= (~active) & (gap < -tol)
            if torch.equal(new, active):
                break
            active = new

        return self._post(delta, u, f, active)

    # -- force controlled solve ---------------------------------------------

    def solve_force(
        self, force: float, rtol: float = 5e-3, max_iter: int = 15
    ) -> ContactResult:
        """Iterate the hub approach until the contact resultant equals `force`."""
        delta = 0.015 * self.roller.r_out
        hist: list[tuple[float, float]] = []
        res = None
        t_start = time.perf_counter()

        for _ in range(max_iter):
            res = self.solve_penetration(delta, res.active if res else None)
            F = res.force
            if self.verbose:
                print(f"  delta = {delta:8.4f} mm -> F = {F:10.3f} N"
                      f"   [{time.perf_counter() - t_start:6.1f} s]")
            if F > 0.0 and abs(F - force) <= rtol * force:
                res.t_total = time.perf_counter() - t_start
                return res

            if F <= 1e-9:  # no contact yet
                delta = max(2.0 * delta, 1e-3 * self.roller.r_out)
                continue

            hist.append((math.log(delta), math.log(F)))
            if len(hist) >= 2 and hist[-1][0] != hist[-2][0]:
                (l0, g0), (l1, g1) = hist[-2], hist[-1]
                slope = (g1 - g0) / (l1 - l0)  # dlnF/dln(delta), ~1.5
                slope = min(max(slope, 0.5), 4.0)
            else:
                slope = 1.5
            ratio = (force / F) ** (1.0 / slope)
            delta *= min(max(ratio, 0.4), 2.5)

        raise RuntimeError("Force control did not converge; check inputs/mesh.")

    # -- post-processing -----------------------------------------------------

    def _post(self, delta, u, f, active) -> ContactResult:
        nodes, outer = self.nodes, self.outer
        Xd = nodes + u  # deformed configuration

        # tributary length on the deformed outer ring (periodic neighbours)
        P = Xd[outer]
        seg = (P.roll(-1, 0) - P).norm(dim=1)
        trib = 0.5 * (seg + seg.roll(1, 0))

        idx_local = self.cand_local.nonzero().ravel()[active]
        idx = outer[idx_local]
        xc = Xd[idx, 0]
        Ry = f[idx, 1]  # vertical reaction (positive = substrate pushes up)
        ny = self.normal(xc)[:, 1]  # projection on the substrate normal
        p = Ry * ny / (trib[idx_local] * self.roller.width)

        order = torch.argsort(xc)
        xc, p, idx = xc[order], p[order], idx[order]

        force = float(Ry.sum())
        a = 0.5 * float(xc[-1] - xc[0]) if len(xc) > 1 else 0.0
        p_mean = force / (2.0 * a * self.roller.width) if a > 0 else float("nan")

        centre = torch.tensor([0.0, -delta])  # displaced roller axis
        r_def = (Xd[outer] - centre).norm(dim=1)
        u_r = r_def - self.roller.r_out

        return ContactResult(
            delta=float(delta),
            force=force,
            x=xc,
            pressure=p,
            half_width=a,
            p_max=float(p.max()) if len(p) else 0.0,
            p_mean=p_mean,
            theta_out=self.theta_out,
            u_radial=u_r,
            max_flattening=float(-u_r.min()),
            u=u,
            active=active,
            nodes_contact=idx,
            n_solves=self.n_solves,
            t_fem=self.t_fem,
        )

    # -- friction ------------------------------------------------------------

    def tangential_bias(self, res: ContactResult) -> float:
        """Effective |t/p| implied by constraining the vertical DOF, not the normal.

        The contact is frictionless: no tangential traction is ever prescribed.
        But a candidate node is tied to the substrate through its *vertical*
        DOF, so its reaction is vertical while the substrate normal is tilted
        by theta = atan(dy_s/dx) on a curved substrate. Decomposing that
        vertical reaction gives a parasitic tangential share tan(theta) --
        exactly what a friction coefficient mu = tan(theta) would produce.

        Returns the largest tan(theta) over the contact patch: 0 on a flat
        plate (the formulation is then exact frictionless Signorini), and a
        few percent while a / R_substrate stays small. Above ~0.05 the patch
        is too wrapped for this simplification and a true normal-direction
        constraint would be needed.
        """
        if len(res.x) == 0 or self.substrate.kind == "flat":
            return 0.0
        _, slope = self.surface(res.x)
        return float(slope.abs().max())

    # -- analytical reference ------------------------------------------------

    def hertz(self, force: float) -> tuple[float, float]:
        """Hertz line contact (small strain, solid cylinder on rigid body).

        Sanity check only: a hollow roller with a thin wall is much more
        compliant than the Hertz half-space assumption, so the FE patch is
        wider and the FE peak pressure lower, increasingly so as d_in -> d_out.
        """
        Ro, nu, E = self.roller.r_out, self.rubber.nu, self.rubber.E0
        inv = 1.0 / Ro + self.substrate.signed_curvature()
        if inv <= 0.0:
            return float("nan"), float("nan")
        Rs = 1.0 / inv
        Es = E / (1.0 - nu**2)
        Fp = force / self.roller.width  # line load [N/mm]
        a = math.sqrt(4.0 * Fp * Rs / (math.pi * Es))
        return a, 2.0 * Fp / (math.pi * a)


# ----------------------------------------------------------------------------
# 4. Post-processing helpers
# ----------------------------------------------------------------------------

# Rainbow ramp blue -> cyan -> green -> yellow -> red for the pressure fields.
# "turbo" keeps the familiar jet hue order but is monotonic in lightness, so
# it does not invent banding artefacts inside the contact patch.
PRESSURE_CMAP = "turbo"


def sample_pressure(res: ContactResult, x: Tensor) -> Tensor:
    """Contact pressure interpolated at arbitrary x, zero outside the patch."""
    if len(res.x) < 2:
        return torch.zeros_like(x)
    p = _interp(x.clamp(float(res.x[0]), float(res.x[-1])), res.x, res.pressure)
    outside = (x < res.x[0]) | (x > res.x[-1])
    return torch.where(outside, torch.zeros_like(p), p)


def nodal_radial_displacement(rc: RollerContact, res: ContactResult) -> Tensor:
    """Radial displacement of every node w.r.t. the displaced roller axis."""
    centre = torch.tensor([0.0, -res.delta])
    r_ref = rc.nodes.norm(dim=1)
    r_def = (rc.nodes + res.u - centre).norm(dim=1)
    return r_def - r_ref


def nodal_pressure(rc: RollerContact, res: ContactResult) -> Tensor:
    """Contact pressure mapped on the mesh nodes (0 away from the patch)."""
    p = torch.zeros(len(rc.nodes))
    p[res.nodes_contact] = res.pressure
    return p


# ----------------------------------------------------------------------------
# 5. 2D plots (matplotlib)
# ----------------------------------------------------------------------------


def plot_pressure(rc: RollerContact, res: ContactResult, force: float, n_z: int = 80):
    """Contact pressure: profile p(x) + footprint map p(x, z) on the substrate."""
    b = rc.roller.width
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) pressure profile across the contact patch
    a_h, p_h = rc.hertz(force)
    ax[0].fill_between(res.x, 0.0, res.pressure, color="crimson", alpha=0.15)
    ax[0].plot(res.x, res.pressure, "o-", ms=3.5, lw=1.6, color="crimson",
               label="FE, hyperelastic, hollow roller")
    if not math.isnan(a_h):
        xs = torch.linspace(-a_h, a_h, 300)
        ax[0].plot(xs, p_h * torch.sqrt(1.0 - (xs / a_h) ** 2), "--", lw=1.4,
                   color="gray", label="Hertz, linear, solid cylinder")
    ax[0].axhline(res.p_mean, ls=":", lw=1.0, color="k",
                  label=f"mean = {res.p_mean:.3f} MPa")
    ax[0].plot([-res.half_width, res.half_width], [0, 0], "|-", color="navy", ms=12,
               lw=1.0)
    ax[0].annotate(f"2a = {2 * res.half_width:.2f} mm",
                   xy=(0.0, 0.04 * res.p_max), ha="center", color="navy", fontsize=9)
    ax[0].set_xlabel("x, transverse to the roller axis [mm]")
    ax[0].set_ylabel("normal contact pressure p [MPa]")
    ax[0].set_title(f"F = {res.force:.0f} N ({res.force / b:.2f} N/mm), "
                    f"p_max = {res.p_max:.3f} MPa")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    # (b) footprint map. Plane strain -> p is invariant along the width; the
    # real roller relaxes over ~1 wall thickness at each free edge.
    pad = max(0.6 * res.half_width, 1.0)
    xg = torch.linspace(float(res.x[0]) - pad, float(res.x[-1]) + pad, 300)
    zg = torch.linspace(-0.5 * b, 0.5 * b, n_z)
    pg = sample_pressure(res, xg).unsqueeze(0).expand(n_z, -1)
    m = ax[1].pcolormesh(xg, zg, pg, cmap=PRESSURE_CMAP, shading="gouraud")
    fig.colorbar(m, ax=ax[1], label="p [MPa]")
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("z, along the roller axis [mm]")
    ax[1].set_title("footprint p(x, z)\nplane strain: uniform along the width",
                    fontsize=10)
    ax[1].set_aspect("equal")

    fig.tight_layout()
    return fig


def plot_pressure_directions(rc: RollerContact, res: ContactResult, force: float):
    """Pressure distribution along the two directions of the contact patch.

    (a) in the section plane, p(x), x transverse to the roller axis.
    (b) along the roller axis, p(z), at a few x stations of the patch.

    The plane-strain slice is by construction invariant along z, so (b) is a
    set of flat plateaus: it is the *model's* answer, not a measurement. A real
    roller of finite width relaxes over roughly one wall thickness at each free
    end, which (b) cannot show. Use it to read the levels and to judge whether
    the plane-strain assumption is acceptable: it needs b >> 2a, and the ratio
    is printed in the panel title.
    """
    b, a = rc.roller.width, res.half_width
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) section plane -----------------------------------------------------
    a_h, p_h = rc.hertz(force)
    ax[0].fill_between(res.x, 0.0, res.pressure, color="crimson", alpha=0.15)
    ax[0].plot(res.x, res.pressure, "o-", ms=3.5, lw=1.6, color="crimson",
               label="FE, hyperelastic, hollow roller")
    if not math.isnan(a_h):
        xs = torch.linspace(-a_h, a_h, 300)
        ax[0].plot(xs, p_h * torch.sqrt(1.0 - (xs / a_h) ** 2), "--", lw=1.4,
                   color="gray", label="Hertz, linear, solid cylinder")
    ax[0].axhline(res.p_mean, ls=":", lw=1.0, color="k",
                  label=f"mean = {res.p_mean:.3f} MPa")
    ax[0].axvline(0.0, ls=":", lw=0.8, color="gray")
    ax[0].set_xlabel("x, in the section plane, transverse to the roller axis [mm]")
    ax[0].set_ylabel("normal contact pressure p [MPa]")
    ax[0].set_title(f"(a) section plane   2a = {2 * a:.2f} mm, "
                    f"p_max = {res.p_max:.3f} MPa")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    # (b) roller axis -------------------------------------------------------
    zg = torch.linspace(-0.5 * b, 0.5 * b, 200)
    stations = [0.0, 0.5 * a, 0.8 * a, 0.95 * a]
    colors = plt.cm.turbo(torch.linspace(0.15, 0.9, len(stations)).numpy())
    for xk, c in zip(stations, colors):
        pk = float(sample_pressure(res, torch.tensor([xk])).item())
        ax[1].plot(zg, torch.full_like(zg, pk), lw=1.8, color=c,
                   label=f"x = {xk:.2f} mm   p = {pk:.3f} MPa")
        ax[1].plot([-0.5 * b, -0.5 * b], [0.0, pk], lw=1.8, color=c)
        ax[1].plot([0.5 * b, 0.5 * b], [0.0, pk], lw=1.8, color=c)
    ax[1].axvspan(-0.5 * b, 0.5 * b, color="0.9", zorder=0)
    ax[1].set_xlim(-0.75 * b, 0.75 * b)
    ax[1].set_ylim(0.0, 1.15 * max(res.p_max, 1e-9))
    ax[1].set_xlabel("z, along the roller axis [mm]")
    ax[1].set_ylabel("normal contact pressure p [MPa]")
    ax[1].set_title(f"(b) roller axis   plane strain, b/2a = {b / (2 * a):.1f}")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8, title="station in the patch", title_fontsize=8)

    fig.tight_layout()
    return fig


def plot_deflection(rc: RollerContact, res: ContactResult):
    """Radial deflection of the outer surface + deformed cross-section."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].plot(torch.rad2deg(res.theta_out), res.u_radial, ".", ms=3, color="navy")
    ax[0].axvline(-90.0, ls=":", color="k", lw=1.0)
    ax[0].set_xlabel("angular position [deg]   (-90 deg = contact)")
    ax[0].set_ylabel("radial displacement of the outer surface [mm]")
    ax[0].set_title(f"hub approach = {res.delta:.3f} mm, "
                    f"max flattening = {res.max_flattening:.3f} mm")
    ax[0].grid(alpha=0.3)

    rc.model.plot(u=res.u, node_property=res.u.norm(dim=1), ax=ax[1], bcs=False)
    xs = torch.linspace(-1.3 * rc.roller.r_out, 1.3 * rc.roller.r_out, 300)
    ys, _ = rc.surface(xs)
    ax[1].plot(xs, ys, "-", lw=2.5, color="k")
    ax[1].set_title("deformed cross-section and rigid substrate")
    ax[1].set_aspect("equal")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 6. 3D view (PyVista)
# ----------------------------------------------------------------------------


def roller_grid_3d(rc: RollerContact, res: ContactResult | None = None, n_z: int = 6,
                   scale: float = 1.0):
    """Extrude the plane-strain mesh into a hexahedral PyVista grid.

    With `res = None` the undeformed mesh is returned and no result field is
    attached (used by `plot_mesh_3d` to inspect the discretisation before any
    computation). Otherwise the deformed shape is built: the plane-strain
    solution is invariant along the roller axis, so the 3D body is a pure
    extrusion of the 2D slice over the width b -- the real deformed shape, not
    an extra 3D computation.
    """
    import numpy as np
    import pyvista as pv

    b = rc.roller.width
    X = rc.nodes.numpy() if res is None else (rc.nodes + scale * res.u).numpy()
    quads = rc.elements.numpy().astype(np.int64)
    n = len(X)

    z = np.linspace(-0.5 * b, 0.5 * b, n_z + 1)
    pts = np.vstack([np.column_stack([X[:, 0], X[:, 1], np.full(n, zk)]) for zk in z])

    # VTK hexahedron: bottom quad (CCW in xy) then the same quad one layer up
    hexa = np.vstack([np.hstack([quads + k * n, quads + (k + 1) * n])
                      for k in range(n_z)])
    cells = np.hstack([np.full((len(hexa), 1), 8, dtype=np.int64), hexa]).ravel()
    ctypes = np.full(len(hexa), pv.CellType.HEXAHEDRON, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    if res is not None:
        grid["|u| [mm]"] = np.tile(res.u.norm(dim=1).numpy(), n_z + 1)
        grid["u_radial [mm]"] = np.tile(
            nodal_radial_displacement(rc, res).numpy(), n_z + 1
        )
        grid["p [MPa]"] = np.tile(nodal_pressure(rc, res).numpy(), n_z + 1)
    return grid


def substrate_mesh_3d(rc: RollerContact, res: ContactResult | None = None,
                      span: float | None = None, n_z: int = 40,
                      thickness: float | None = None):
    """Rigid substrate as a solid slab, coloured by the contact pressure.

    With `res = None` the pressure field is identically zero (mesh preview).
    """
    import numpy as np
    import pyvista as pv

    b = rc.roller.width
    span = span or 1.6 * rc.roller.r_out
    thickness = thickness or 0.25 * rc.roller.r_out

    xs = torch.linspace(-span, span, 240)
    zs = torch.linspace(-0.5 * b, 0.5 * b, n_z)
    ys, _ = rc.surface(xs)
    p = torch.zeros_like(xs) if res is None else sample_pressure(res, xs)

    Xg, Zg = np.meshgrid(xs.numpy(), zs.numpy(), indexing="ij")
    Yg = np.repeat(ys.numpy()[:, None], len(zs), axis=1)
    Pg = np.repeat(p.numpy()[:, None], len(zs), axis=1)

    surf = pv.StructuredGrid(Xg[..., None], Yg[..., None], Zg[..., None])
    surf["p [MPa]"] = Pg.ravel(order="F")  # StructuredGrid points are F-ordered
    try:  # give the substrate some thickness, purely cosmetic
        return surf.extract_surface(algorithm="dataset_surface").extrude(
            (0.0, -thickness, 0.0), capping=True
        )
    except Exception:
        return surf


def mesh_report(rc: RollerContact) -> dict[str, float]:
    """Cheap quality metrics of the undeformed quad mesh."""
    P = rc.nodes[rc.elements]
    x, y = P[..., 0], P[..., 1]
    area = 0.5 * (x * y.roll(-1, 1) - x.roll(-1, 1) * y).sum(1)  # signed, CCW > 0
    edge = (P.roll(-1, 1) - P).norm(dim=2)

    ys, _ = rc.surface(rc.nodes[rc.cand, 0])
    gap = rc.nodes[rc.cand, 1] - ys

    return {
        "n_nodes": len(rc.nodes),
        "n_elements": len(rc.elements),
        "n_candidates": int(rc.cand_local.sum()),
        "min_area": float(area.min()),
        "n_inverted": int((area <= 0.0).sum()),
        "min_edge": float(edge.min()),
        "max_aspect": float((edge.max(1).values / edge.min(1).values).max()),
        "min_gap": float(gap.min()),
    }


def _finish(pl, screenshot: str | None, off_screen: bool):
    """Write the PNG and open the window, in the order PyVista requires.

    An on-screen plotter has no render window until `show()` runs, so calling
    `screenshot()` first raises. `show(screenshot=...)` grabs the frame before
    the window is torn down and works for both cases.
    """
    if off_screen:
        if screenshot:
            pl.screenshot(screenshot)
    else:
        pl.show(screenshot=screenshot)
    return pl


def roller_slice_2d(rc: RollerContact, u: Tensor | None = None):
    """The plane-strain Quad1 slice itself, as a flat PyVista grid at z = 0."""
    import numpy as np
    import pyvista as pv

    X = (rc.nodes if u is None else rc.nodes + u).numpy()
    pts = np.column_stack([X[:, 0], X[:, 1], np.zeros(len(X))])
    quads = rc.elements.numpy().astype(np.int64)
    cells = np.hstack([np.full((len(quads), 1), 4, dtype=np.int64), quads]).ravel()
    ctypes = np.full(len(quads), pv.CellType.QUAD, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ctypes, pts)


def substrate_profile_2d(rc: RollerContact, span: float | None = None, n: int = 400):
    """Section line of the rigid substrate in the z = 0 plane."""
    import numpy as np
    import pyvista as pv

    span = span or 1.6 * rc.roller.r_out
    xs = torch.linspace(-span, span, n)
    ys, _ = rc.surface(xs)
    return pv.lines_from_points(
        np.column_stack([xs.numpy(), ys.numpy(), np.zeros(n)])
    )


def plot_mesh_3d(rc: RollerContact, n_z: int = 6, opacity: float = 0.30,
                 off_screen: bool = False, screenshot: str | None = None):
    """Undeformed mesh preview: roller, shaft, substrate and contact candidates.

    Call this *before* solving to check the discretisation, the position of the
    rigid substrate and which outer-ring nodes may enter contact.

    Left view : the extruded assembly, roller drawn translucent so the bore and
                the substrate stay visible through the wall.
    Right view: the 2D Quad1 slice that is actually solved, zoomed on the
                contact sector, with the candidate nodes as red spheres.
    """
    import pyvista as pv

    q = mesh_report(rc)
    roller, b = rc.roller, rc.roller.width

    pl = pv.Plotter(shape=(1, 2), window_size=(1400, 720), off_screen=off_screen)

    # -- left: extruded assembly --------------------------------------------
    pl.subplot(0, 0)
    pl.add_mesh(roller_grid_3d(rc, None, n_z=n_z), color="#d9534f",
                opacity=opacity, show_edges=True, edge_color="#333333",
                line_width=0.6)
    pl.add_mesh(substrate_mesh_3d(rc), color="#9aa0a6", scalars=None,
                opacity=0.85, smooth_shading=True)
    pl.add_mesh(
        pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                    radius=roller.r_in, height=1.25 * b),
        color="#b8bcc2", opacity=0.85, smooth_shading=True,
    )
    pl.add_text(
        f"{q['n_elements']} Quad1 elements, {q['n_nodes']} nodes\n"
        f"extruded over b = {b:g} mm ({n_z} layers, display only)\n"
        f"min area {q['min_area']:.3f} mm2, inverted {q['n_inverted']}\n"
        f"min edge {q['min_edge']:.3f} mm, max aspect {q['max_aspect']:.1f}",
        position="upper_left", font_size=9,
    )
    pl.camera_position = [
        (2.6 * roller.r_out, 1.8 * roller.r_out, 2.2 * b),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]

    # -- right: the solved 2D slice, zoomed on the contact sector -----------
    pl.subplot(0, 1)
    pl.add_mesh(roller_slice_2d(rc), color="#f2c9c6", show_edges=True,
                edge_color="#333333", line_width=0.8)
    pl.add_mesh(substrate_profile_2d(rc), color="#1f3a5f", line_width=5.0)
    # Real sphere glyphs, not screen-space points: `render_points_as_spheres`
    # needs point-sprite support that several OpenGL back-ends (in particular
    # the off-screen one used for the PNG export) silently ignore, which drops
    # the markers altogether. Lifted out of z = 0 to avoid z-fighting.
    marker = 0.012 * roller.r_out
    pl.add_mesh(
        pv.PolyData(
            torch.cat(
                [rc.nodes[rc.cand], torch.full((len(rc.cand), 1), 4.0 * marker)], 1
            ).numpy()
        ).glyph(geom=pv.Sphere(radius=marker), scale=False, orient=False),
        color="crimson",
    )
    pl.add_text(
        f"solved slice, contact sector\n"
        f"{q['n_candidates']} contact candidates (red)\n"
        f"min initial gap {q['min_gap']:.4f} mm\n"
        f"substrate: {rc.substrate.kind}"
        + ("" if rc.substrate.kind == "flat" else f", R = {rc.substrate.radius:g} mm"),
        position="upper_left", font_size=9,
    )
    # orthographic view framing the fine sector plus the wall thickness
    half = max(roller.r_out * math.sin(rc.mesh_opts.fine_window),
               0.7 * (roller.r_out - roller.r_in))
    pl.enable_parallel_projection()
    pl.camera_position = "xy"
    pl.camera.focal_point = (0.0, -roller.r_out + 0.55 * half, 0.0)
    pl.camera.position = (0.0, -roller.r_out + 0.55 * half, 4.0 * roller.r_out)
    pl.camera.parallel_scale = half

    # views are intentionally NOT linked: the two cameras show different scales
    return _finish(pl, screenshot, off_screen)


def plot_3d(rc: RollerContact, res: ContactResult, force: float, n_z: int = 6,
            scalars: str = "u_radial [mm]", cutaway: bool = False,
            show_undeformed: bool = True, opacity: float = 0.45,
            screenshot: str | None = None, off_screen: bool = False):
    """Interactive 3D view: compressed roller, shaft, substrate + footprint.

    scalars: "u_radial [mm]", "|u| [mm]" or "p [MPa]" on the roller body.
    cutaway: clip the half z > 0 to expose the deformed wall section.
    opacity: transparency of the roller body, so the pressure footprint stays
             visible through the wall.
    """
    import pyvista as pv

    roller, b = rc.roller, rc.roller.width
    grid = roller_grid_3d(rc, res, n_z=n_z)
    body = grid.clip(normal="z", origin=(0.0, 0.0, 0.0)) if cutaway else grid

    pl = pv.Plotter(window_size=(1280, 860), off_screen=off_screen)
    pl.add_mesh(
        body,
        scalars=scalars,
        cmap="coolwarm" if "u" in scalars else PRESSURE_CMAP,
        opacity=opacity,
        show_edges=True,
        edge_color="gray",
        line_width=0.4,
        smooth_shading=True,
        scalar_bar_args={"title": scalars, "vertical": True, "position_x": 0.86,
                         "position_y": 0.30, "height": 0.45, "width": 0.05},
    )

    # rigid substrate, coloured by the contact pressure it receives
    pl.add_mesh(
        substrate_mesh_3d(rc, res),
        scalars="p [MPa]",
        cmap=PRESSURE_CMAP,
        clim=(0.0, max(res.p_max, 1e-6)),
        smooth_shading=True,
        # short title: a long one is centred on the bar and runs off the left edge
        scalar_bar_args={"title": "p [MPa]", "vertical": True,
                         "position_x": 0.06, "position_y": 0.30, "height": 0.45,
                         "width": 0.05},
    )

    # shaft through the bore, at its displaced position
    pl.add_mesh(
        pv.Cylinder(center=(0.0, -res.delta, 0.0), direction=(0.0, 0.0, 1.0),
                    radius=roller.r_in, height=1.25 * b),
        color="#b8bcc2", smooth_shading=True,
    )

    # undeformed roller as a ghost, to visualise the flattening
    if show_undeformed:
        pl.add_mesh(
            pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                        radius=roller.r_out, height=b),
            color="white", opacity=0.12, style="surface",
        )

    # applied force
    arrow_len = 0.9 * roller.r_out
    pl.add_mesh(
        pv.Arrow(start=(0.0, 1.35 * roller.r_out, 0.0), direction=(0.0, -1.0, 0.0),
                 scale=arrow_len, tip_length=0.3, shaft_radius=0.02),
        color="black",
    )
    pl.add_point_labels(
        [[0.0, 1.45 * roller.r_out, 0.0]], [f"F = {force:.0f} N"],
        font_size=14, shape=None, show_points=False,
    )
    pl.add_text(
        f"a = {res.half_width:.2f} mm    p_max = {res.p_max:.3f} MPa    "
        f"crush = {res.delta:.3f} mm",
        position="upper_left", font_size=10,
    )

    pl.add_axes(xlabel="x", ylabel="y", zlabel="z (roller axis)")
    pl.camera_position = [
        (2.6 * roller.r_out, 1.8 * roller.r_out, 2.2 * b),
        (0.0, -0.3 * roller.r_out, 0.0),
        (0.0, 1.0, 0.0),
    ]
    return _finish(pl, screenshot, off_screen)


# ----------------------------------------------------------------------------
# 7. Example
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    roller = Roller(d_out=60.0, d_in=25.0, width=50.0)  # mm
    rubber = shore_a_to_yeoh(60.0)  # Shore A 60 silicone / EPDM
    substrate = Substrate("convex", radius=150.0)  # mandrel of R = 150 mm
    force = 300.0  # N on the roller axis

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    rc = RollerContact(roller, rubber, substrate, MeshOpts(n_r=8, n_theta=160),
                       verbose=True)

    # Mesh check. Written to disk only: the window is not opened, so the run
    # goes straight into the solve. Pass off_screen=False to inspect it live.
    plot_mesh_3d(rc, off_screen=True, screenshot=str(out / "01_mesh_preview.png"))
    print(f"mesh preview  -> {out / '01_mesh_preview.png'}")

    res = rc.solve_force(force)

    a_h, p_h = rc.hertz(force)
    print(f"\n{rubber.name}: mu0 = {rubber.mu0:.3f} MPa, E0 = {rubber.E0:.3f} MPa")
    print(f"applied force        {force:10.2f} N   ({force / roller.width:.2f} N/mm)")
    print(f"resultant (check)    {res.force:10.2f} N")
    print(f"hub approach         {res.delta:10.3f} mm")
    print(f"max wall flattening  {res.max_flattening:10.3f} mm")
    print(f"contact half width   {res.half_width:10.3f} mm   (Hertz {a_h:.3f})")
    print(f"peak pressure        {res.p_max:10.3f} MPa  (Hertz {p_h:.3f})")
    print(f"mean pressure        {res.p_mean:10.3f} MPa")
    print("contact                 frictionless (normal Signorini)"
          f", tangential bias {rc.tangential_bias(res):.3f}")

    print("\ncomputation time")
    print(f"  mesh + model setup {rc.t_setup:10.2f} s")
    print(f"  torch-fem solves   {res.t_fem:10.2f} s   "
          f"({res.n_solves} nonlinear solves, "
          f"{res.t_fem / max(res.n_solves, 1):.2f} s each)")
    print(f"  active set + post  {res.t_contact:10.2f} s")
    print(f"  total              {rc.t_setup + res.t_total:10.2f} s")

    figs = {
        "02_pressure_footprint.png": plot_pressure(rc, res, force),
        #"03_pressure_directions.png": plot_pressure_directions(rc, res, force),
        "03_deflection.png": plot_deflection(rc, res),
    }
    print()
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=150)
        print(f"figure       -> {out / name}")

    # 3D views: the radial-deflection one is written and shown interactively,
    # the pressure cutaway is written only.
    plot_3d(rc, res, force, scalars="p [MPa]", cutaway=True, off_screen=True,
            screenshot=str(out / "04_contact_pressure_3d.png"))
    print(f"3D view      -> {out / '04_contact_pressure_3d.png'}")

    plt.show(block=True)
    plot_3d(rc, res, force, scalars="u_radial [mm]", cutaway=False,
            screenshot=str(out / "05_deformed_roller_3d.png"))
    print(f"3D view      -> {out / '05_deformed_roller_3d.png'}")
