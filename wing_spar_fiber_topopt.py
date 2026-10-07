"""Drone wing spar: concurrent topology + continuous-fiber orientation optimization.

The spar is a flat, tapered composite plate in the X-Y plane (span along X,
chord along Y), clamped at the root (x = L) and loaded at the tip (x = 0) by

* DRAG - an in-plane force along -Y  (membrane / in-plane bending),
* LIFT - an out-of-plane force along +Z (plate bending + torsion).

Both load cases need a `Shell` model (membrane + bending), so this is the shell
counterpart of torch-fem's `optimization/planar/topology+orientation.ipynb`
example: SIMP densities updated with an optimality-criteria step and
element-wise fiber angles updated with the compliance gradient. Two additions
make the result manufacturable as *continuous* fibers:

1. The orientation field is regularized every iteration by filtering the
   director in doubled-angle space (cos 2θ, sin 2θ), so that neighbouring
   elements get similar angles and streamlines do not kink.
2. After the optimization, evenly-spaced streamlines (Jobard-Lefer) are traced
   through the solid region of the optimized orientation field. Each streamline
   is one continuous, uncut fiber path; they are exported as polylines (CSV and
   JSON) and chained into one print sequence for continuous-fiber AM / AFP.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.optimize import bisect
from scipy.spatial import cKDTree
from tqdm import tqdm

from torchfem import Shell
from torchfem.materials import OrthotropicElasticityPlaneStress
from torchfem.mesh import rect_tri
from torchfem.rotations import planar_rotation

torch.set_default_dtype(torch.float64)


# --- Geometry (mm) - tapered planform read from the sketch ---
L = 600.0  # span, tip at x = 0, root at x = L
Y_TIP = (76.0, 204.0)  # (trailing, leading) edge y at the tip   -> chord 128
Y_ROOT = (36.0, 244.0)  # (trailing, leading) edge y at the root -> chord 208
THICKNESS = 6.0  # mm - spar plate thickness

# --- Mesh ---
NX = 96  # elements along the span
NY = 32  # elements along the chord

# --- Loads (N) ---
# "tip": both resultants at the tip edge (as drawn) -> two-bar V truss.
# "distributed": elliptic span loading introduced along the leading and
# trailing edges (where the ribs / skin attach) -> chords + branching ribs.
LOAD_MODE = "distributed"
DRAG = 60.0  # in-plane resultant, -Y
LIFT = 120.0  # out-of-plane resultant, +Z
LIFT_LE_SHARE = 0.6  # share of lift on the leading edge (center of pressure)
W_DRAG = 0.5  # weight of the (normalized) drag compliance
W_LIFT = 0.5  # weight of the (normalized) lift compliance

# --- Optimization ---
P_SIMP = 3.0
VOLFRAC = 0.30
MOVE = 0.1
N_ITER = 150
FILTER_RADIUS = 1.6  # in element sizes, density sensitivity filter
ORI_FILTER_RADIUS = 2.5  # in element sizes, orientation regularization
ORI_MAX_STEP = 0.15  # rad - max fiber angle change per iteration

# --- Fiber paths ---
TOW_SPACING = 5.0  # mm - distance between neighbouring fiber paths
RHO_SOLID = 0.5  # density threshold that defines the printed region
STEP = 0.5  # mm - streamline integration step

OUT = Path(f"wing_spar_output_{LOAD_MODE}")

# Carbon / epoxy (same lamina as copv_winding_fea.py)
cfrp = OrthotropicElasticityPlaneStress(
    E_1=176800.0,
    E_2=10300.0,
    nu_12=0.23,
    G_12=4800.0,
    G_13=4800.0,
    G_23=3000.0,
    rho=1.6e-9,
)


def y_edges(x):
    """Trailing and leading edge y-coordinates at span station x."""
    s = x / L
    y_b = Y_TIP[0] + s * (Y_ROOT[0] - Y_TIP[0])
    y_t = Y_TIP[1] + s * (Y_ROOT[1] - Y_TIP[1])
    return y_b, y_t


# Mesh: structured triangles on the unit square, mapped onto the trapezoid
uv, elements = rect_tri(NX + 1, NY + 1, 1.0, 1.0, variant="zigzag")
x = uv[:, 0] * L
y_b, y_t = y_edges(x)
y = y_b + uv[:, 1] * (y_t - y_b)
nodes = torch.stack([x, y, torch.zeros_like(x)], dim=1)

spar = Shell(nodes, elements, cfrp, thickness=THICKNESS)
n_elem = spar.n_elem
centers = nodes[elements].mean(dim=1)
uv_c = uv[elements].mean(dim=1)


tol = 1e-6
tip = x < tol
root = x > L - tol

# Clamped root
spar.constraints[root, :] = True

# Load vectors for the two load cases, spread over the tip edge
f_drag = torch.zeros_like(spar.forces)
f_lift = torch.zeros_like(spar.forces)
if LOAD_MODE == "tip":
    f_drag[tip, 1] = -DRAG / tip.sum()
    f_lift[tip, 2] = LIFT / tip.sum()
else:
    # Elliptic distribution in span coordinate eta (0 at root, 1 at tip)
    eta = 1.0 - x / L
    q = torch.sqrt(torch.clamp(1.0 - eta**2, min=0.0)) * (~root)
    le = (uv[:, 1] > 1.0 - tol) & ~root
    te = (uv[:, 1] < tol) & ~root
    f_drag[le, 1] = -DRAG * q[le] / q[le].sum()
    f_lift[le, 2] = LIFT_LE_SHARE * LIFT * q[le] / q[le].sum()
    f_lift[te, 2] = (1.0 - LIFT_LE_SHARE) * LIFT * q[te] / q[te].sum()

# Passive solid elements: tip rib (load introduction) and root fitting
h = L / NX
passive = (centers[:, 0] < 1.5 * h) | (centers[:, 0] > L - 1.5 * h)
active = ~passive

# Filters in element-size units on the parametric grid (elements are
# stretched along the chord; the filter acts on neighbours in index space).
scale = torch.tensor([NX, NY], dtype=uv.dtype)
dist = torch.cdist(uv_c * scale, uv_c * scale)
H = torch.clamp(FILTER_RADIUS - dist, min=0.0)
H_ori = torch.clamp(ORI_FILTER_RADIUS - dist, min=0.0)

# Volume constraint (elements have different areas on the trapezoid)
area = 0.5 * torch.linalg.norm(
    torch.linalg.cross(
        nodes[elements[:, 1]] - nodes[elements[:, 0]],
        nodes[elements[:, 2]] - nodes[elements[:, 0]],
    ),
    dim=1,
)
V_0 = VOLFRAC * area.sum()

# Material angles are measured from global x in each element's local frame.
# Flip the sign where the element normal points to -z.
normal_sign = torch.sign(
    torch.linalg.cross(
        nodes[elements[:, 1]] - nodes[elements[:, 0]],
        nodes[elements[:, 2]] - nodes[elements[:, 0]],
    )[:, 2]
)

As0 = spar.As.clone()


def compliance(rho, theta, forces):
    """Compliance of one load case for densities rho and global fiber angles."""
    R = planar_rotation(normal_sign * theta)
    spar.material = cfrp.vectorize(n_elem).rotate(R)
    penal = rho**P_SIMP
    spar.material.C = penal[:, None, None, None, None] * spar.material.C
    spar.As = penal[:, None, None] * As0
    spar.forces = forces
    u, f, _, _, _ = spar.solve(
        method="direct", differentiable_parameters=(rho, theta)
    )
    return 0.5 * torch.inner(u.ravel(), f.ravel())


def smooth_orientation(theta, rho):
    """Density-weighted director filter in doubled-angle space."""
    w = H_ori * rho[None, :]
    c = w @ torch.cos(2 * theta)
    s = w @ torch.sin(2 * theta)
    return 0.5 * torch.atan2(s, c)


def optimize(rho, theta, n_iter=N_ITER):
    rho_min = 1e-3 * torch.ones_like(rho)
    rho_max = torch.ones_like(rho)
    with torch.no_grad():
        C0 = [compliance(rho, theta, f).item() for f in (f_drag, f_lift)]

    history = []
    for _ in tqdm(range(n_iter)):
        # One adjoint per load case (each solve owns its own graph)
        C_d = W_DRAG * compliance(rho, theta, f_drag) / C0[0]
        g_d = torch.autograd.grad(C_d, (rho, theta))
        C_l = W_LIFT * compliance(rho, theta, f_lift) / C0[1]
        g_l = torch.autograd.grad(C_l, (rho, theta))
        C = C_d + C_l
        dC_drho, dC_dtheta = g_d[0] + g_l[0], g_d[1] + g_l[1]

        # Sensitivity filter
        dC_drho = H @ (rho * dC_drho) / H.sum(dim=0) / rho
        dC_drho = torch.clamp(dC_drho, max=-1e-12)

        def make_step(mu):
            upper = torch.min(rho_max, (1 + MOVE) * rho)
            lower = torch.max(rho_min, (1 - MOVE) * rho)
            rho_trial = (-dC_drho / mu) ** 0.5 * rho
            rho_new = torch.max(torch.min(rho_trial, upper), lower)
            rho_new[passive] = 1.0
            return rho_new

        def g(mu):
            return ((make_step(mu) * area).sum() - V_0).item()

        with torch.no_grad():
            mu = bisect(g, 1e-12, 1e6)
            rho.data = make_step(mu)

            # Normalized gradient step on the angle, then regularize
            step = ORI_MAX_STEP * dC_dtheta / dC_dtheta.abs().max()
            theta.data = smooth_orientation(theta - step, rho)

        history.append((C.item(), C_d.item() / W_DRAG, C_l.item() / W_LIFT))
    return np.array(history)


# ---------------------------------------------------------------------------
# Continuous fiber paths: evenly spaced streamlines (Jobard & Lefer, 1997)
# ---------------------------------------------------------------------------


class DirectorField:
    """Fiber director and density interpolated on a regular background grid."""

    def __init__(self, centers, theta, rho, res=1.0):
        xc, yc = centers[:, 0], centers[:, 1]
        self.gx = np.arange(0.0, L + res, res)
        self.gy = np.arange(min(Y_ROOT[0], Y_TIP[0]), max(Y_ROOT[1], Y_TIP[1]) + res, res)
        X, Y = np.meshgrid(self.gx, self.gy, indexing="ij")
        pts = np.column_stack([xc, yc])
        fields = []
        for v in (np.cos(2 * theta), np.sin(2 * theta), rho):
            lin = griddata(pts, v, (X, Y), method="linear")
            near = griddata(pts, v, (X, Y), method="nearest")
            fields.append(np.where(np.isnan(lin), near, lin))
        opts = dict(bounds_error=False, fill_value=0.0)
        self.c2 = RegularGridInterpolator((self.gx, self.gy), fields[0], **opts)
        self.s2 = RegularGridInterpolator((self.gx, self.gy), fields[1], **opts)
        self.rho = RegularGridInterpolator((self.gx, self.gy), fields[2], **opts)

    def inside(self, p):
        if p[0] < 0.0 or p[0] > L:
            return False
        y_b, y_t = y_edges(p[0])
        return y_b <= p[1] <= y_t

    def solid(self, p):
        return self.inside(p) and self.rho(p)[0] >= RHO_SOLID

    def direction(self, p, ref):
        c, s = self.c2(p)[0], self.s2(p)[0]
        if np.hypot(c, s) < 0.05:  # isotropic point, no defined fiber axis
            return None
        a = 0.5 * np.arctan2(s, c)
        d = np.array([np.cos(a), np.sin(a)])
        return d if ref is None or d @ ref >= 0.0 else -d


class PathSet:
    """Accepted streamline points with a spatial hash for distance queries."""

    def __init__(self, cell):
        self.cell = cell
        self.grid = {}

    def _key(self, p):
        return (int(p[0] // self.cell), int(p[1] // self.cell))

    def add(self, path, path_id):
        for p in path:
            self.grid.setdefault(self._key(p), []).append((p, path_id))

    def too_close(self, p, dmin, ignore=None):
        i, j = self._key(p)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for q, pid in self.grid.get((i + di, j + dj), ()):
                    if pid != ignore and np.hypot(*(p - q)) < dmin:
                        return True
        return False


def trace(field, paths, seed, d_test, max_len=5.0 * L):
    """Trace a streamline both ways from seed with RK2 until a stop criterion."""
    halves = []
    for sign in (1.0, -1.0):
        d0 = field.direction(seed, None)
        if d0 is None:
            return None
        ref = sign * d0
        p = seed.copy()
        pts = []
        length = 0.0
        while length < max_len:
            k1 = field.direction(p, ref)
            if k1 is None:
                break
            mid = p + 0.5 * STEP * k1
            k2 = field.direction(mid, k1)
            if k2 is None:
                break
            q = p + STEP * k2
            if not field.solid(q) or paths.too_close(q, d_test):
                break
            # Stop on closed loops
            if len(pts) > 20 and np.hypot(*(q - seed)) < 0.5 * STEP:
                break
            pts.append(q)
            ref, p = k2, q
            length += STEP
        halves.append(pts)
    return np.array(halves[1][::-1] + [seed] + halves[0])


def evenly_spaced_streamlines(field, d_sep, d_test_ratio=0.5, min_len=30.0):
    """Jobard-Lefer: seed new streamlines at d_sep from accepted ones."""
    paths = PathSet(cell=d_sep)
    d_test = d_test_ratio * d_sep
    result = []

    # First seeds: a coarse grid over the solid region, longest first
    xs = np.arange(0.5 * d_sep, L, 4 * d_sep)
    seeds = []
    for xi in xs:
        y_b, y_t = y_edges(xi)
        for yi in np.arange(y_b + 0.5 * d_sep, y_t, 4 * d_sep):
            p = np.array([xi, yi])
            if field.solid(p):
                seeds.append(p)
    queue = list(seeds)

    while queue:
        seed = queue.pop(0)
        if not field.solid(seed) or paths.too_close(seed, d_sep * 0.99):
            continue
        line = trace(field, paths, seed, d_test)
        if line is None or len(line) < 2:
            continue
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
        if seg < min_len:
            continue
        pid = len(result)
        paths.add(line, pid)
        result.append(line)
        # Candidate seeds offset by d_sep on both sides, along the new path
        tang = np.gradient(line, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        normal = np.column_stack([-tang[:, 1], tang[:, 0]])
        stride = max(1, int(d_sep / STEP))
        for k in range(0, len(line), stride):
            for s in (1.0, -1.0):
                queue.insert(0, line[k] + s * d_sep * normal[k])
    return result


def chain_paths(paths):
    """Order and orient paths greedily into one continuous print sequence."""
    remaining = list(range(len(paths)))
    # Start with the path whose endpoint is closest to the root (x = L)
    start = max(remaining, key=lambda i: max(paths[i][0, 0], paths[i][-1, 0]))
    order = []
    cur = paths[start] if paths[start][-1, 0] < paths[start][0, 0] else paths[start][::-1]
    order.append(cur)
    remaining.remove(start)
    travel = 0.0
    while remaining:
        end = cur[-1]
        ends = np.array([[paths[i][0], paths[i][-1]] for i in remaining])
        d = np.linalg.norm(ends - end, axis=2)
        k, side = np.unravel_index(np.argmin(d), d.shape)
        travel += d[k, side]
        i = remaining.pop(k)
        cur = paths[i] if side == 0 else paths[i][::-1]
        order.append(cur)
    return order, travel


def export_paths(paths, ordered):
    rows = []
    for pid, line in enumerate(paths):
        for k, (px, py) in enumerate(line):
            rows.append((pid, k, px, py, 0.0))
    np.savetxt(
        OUT / "fiber_paths.csv",
        np.array(rows),
        delimiter=",",
        header="path_id,point_id,x_mm,y_mm,z_mm",
        fmt=["%d", "%d", "%.4f", "%.4f", "%.4f"],
        comments="",
    )
    data = {
        "units": "mm",
        "tow_spacing": TOW_SPACING,
        "thickness": THICKNESS,
        "paths": [line.round(4).tolist() for line in paths],
        "print_sequence": [line.round(4).tolist() for line in ordered],
    }
    (OUT / "fiber_paths.json").write_text(json.dumps(data))


def main():
    OUT.mkdir(exist_ok=True)
    theta = torch.zeros(n_elem, requires_grad=True)  # fibers along the span
    rho = VOLFRAC * torch.ones(n_elem)
    rho[passive] = 1.0
    rho.requires_grad_(True)

    history = optimize(rho, theta)

    rho_np = rho.detach().numpy()
    th_np = theta.detach().numpy()
    c_np = centers[:, :2].numpy()
    np.savez(OUT / "design.npz", centers=c_np, rho=rho_np, theta=th_np)

    # Optimization history
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(history[:, 0], "-k", label="weighted")
    ax.semilogy(history[:, 1], "--", label="drag (in-plane)")
    ax.semilogy(history[:, 2], ":", label="lift (out-of-plane)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized compliance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=150)

    # Density + element orientations
    tri = elements.numpy()
    xy = nodes[:, :2].numpy()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.tripcolor(xy[:, 0], xy[:, 1], tri, facecolors=rho_np, cmap="gray_r", vmin=0, vmax=1)
    m = rho_np > RHO_SOLID
    ax.quiver(
        c_np[m, 0], c_np[m, 1], np.cos(th_np[m]), np.sin(th_np[m]),
        pivot="middle", headlength=0, headaxislength=0, headwidth=0,
        width=0.0015, color="tab:orange", scale=90,
    )
    ax.set_aspect("equal")
    ax.set_title("Optimized density and fiber orientation")
    fig.tight_layout()
    fig.savefig(OUT / "density_orientation.png", dpi=150)

    # Continuous fiber paths
    field = DirectorField(c_np, th_np, rho_np)
    paths = evenly_spaced_streamlines(field, TOW_SPACING)
    ordered, travel = chain_paths(paths)
    export_paths(paths, ordered)
    fiber_len = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths)
    print(f"{len(paths)} continuous fiber paths, total fiber length {fiber_len:.0f} mm")
    print(f"chained print sequence: {travel:.0f} mm of travel moves between paths")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.tripcolor(xy[:, 0], xy[:, 1], tri, facecolors=rho_np, cmap="Greys", vmin=0, vmax=2.5)
    for line in paths:
        ax.plot(line[:, 0], line[:, 1], "-", color="k", lw=0.8)
    ax.plot([L, L], list(Y_ROOT), color="tab:blue", lw=4, label="root (clamped)")
    ax.annotate("", xy=(5, Y_TIP[1] - 40), xytext=(5, Y_TIP[1] + 20),
                arrowprops=dict(color="tab:blue", width=2))
    ax.text(10, Y_TIP[1] + 10, "DRAG", color="tab:blue")
    ax.plot(5, Y_TIP[0] + 10, "o", ms=10, mfc="none", mec="tab:blue", mew=2)
    ax.plot(5, Y_TIP[0] + 10, ".", color="tab:blue")
    ax.text(10, Y_TIP[0] - 5, "LIFT (+z)", color="tab:blue")
    ax.set_aspect("equal")
    ax.set_title(f"Continuous fiber paths ({len(paths)} tows, spacing {TOW_SPACING} mm)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fiber_paths.png", dpi=200)


if __name__ == "__main__":
    main()
