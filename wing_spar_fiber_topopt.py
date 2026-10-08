"""Drone wing spar: concurrent topology + continuous-fiber orientation optimization.

The spar is a flat composite plate in the X-Y plane (span along X, chord along
Y), its planform, supports and load points read from the reference sketch:

* ROOT - three bolted fittings (top chord, middle, bottom chord) at x = L,
* DRAG - in-plane force along -Y on the leading-edge rib stations,
* LIFT - out-of-plane force along +Z on all rib stations (bending + torsion).

Both load cases need a `Shell` model (membrane + bending), so this is the shell
counterpart of torch-fem's `optimization/planar/topology+orientation.ipynb`
example: densities (density filter + Heaviside projection, SIMP) updated with
an optimality-criteria step and element-wise fiber angles updated with the
compliance gradient, regularized in doubled-angle space (cos 2θ, sin 2θ).

Continuous fiber paths are then generated from the optimized design:

* "skeleton" (default): member centerlines are extracted, joined through
  junctions by the straightest continuation into load paths, and filled with
  parallel tows (one bundle per member, count from the member width).
* "streamline": evenly spaced streamlines of the orientation field.

Paths are exported as polylines (CSV and JSON) and chained into one print
sequence for continuous-fiber AM / AFP.

Usage:
    python wing_spar_fiber_topopt.py               # optimize + paths
    python wing_spar_fiber_topopt.py --paths-only  # paths from saved design
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, griddata
from scipy.optimize import bisect
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label
from skimage.morphology import (
    closing,
    disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)
from tqdm import tqdm

from torchfem import Shell
from torchfem.materials import OrthotropicElasticityPlaneStress
from torchfem.mesh import rect_tri
from torchfem.rotations import planar_rotation

torch.set_default_dtype(torch.float64)


# --- Geometry (mm) - read from the reference sketch (1 px = 0.84 mm) ---
L = 600.0  # span, tip at x = 0, root at x = L
# Leading edge (upper chord, arched) and trailing edge (lower chord)
LE_PTS = ([0.0, 243.0, 394.0, 600.0], [214.0, 243.0, 264.0, 256.0])
TE_PTS = ([0.0, 310.0, 600.0], [84.0, 59.0, 42.0])
THICKNESS = 6.0  # mm - spar plate thickness

# --- Mesh ---
NX = 96  # elements along the span
NY = 32  # elements along the chord

# --- Boundary conditions (from the sketch) ---
# Root: three bolted fittings (top chord, middle member, bottom chord) instead
# of a fully clamped edge. Bands are y-ranges on the root edge.
ROOT_FITTINGS = [(236.0, 256.0), (152.0, 178.0), (42.0, 60.0)]
# Load introduction: rib stations = stubs leaving the planform in the sketch,
# plus the two tip corners. (x, edge) with edge "le" or "te".
STATIONS = [(0.0, "le"), (231.0, "le"), (399.0, "le"),
            (0.0, "te"), (80.0, "te"), (310.0, "te")]
PAD_RADIUS = 9.0  # mm - passive solid pad around each station / fitting

# --- Loads (N) ---
DRAG = 60.0  # in-plane resultant, -Y, on the leading-edge stations
LIFT = 120.0  # out-of-plane resultant, +Z, on all stations
TIP_SHARE = 0.5  # share of each resultant carried by the tip stations
W_DRAG = 0.5  # weight of the (normalized) drag compliance
W_LIFT = 0.5  # weight of the (normalized) lift compliance

# --- Optimization ---
P_SIMP = 3.0
VOLFRAC = 0.22
MOVE = 0.1
N_ITER = 180
FILTER_RADIUS = 2.0  # in element sizes, density filter
BETA_START = 40  # Heaviside projection: beta = 1 until this iteration,
BETA_EVERY = 20  # then doubled every BETA_EVERY iterations
BETA_MAX = 16.0
ORI_FILTER_RADIUS = 1.6  # in element sizes, orientation regularization
ORI_MAX_STEP = 0.15  # rad - max fiber angle change per iteration

# --- Fiber paths ---
TOW_SPACING = 5.0  # mm - distance between neighbouring fiber paths
RHO_SOLID = 0.2  # density threshold that defines the printed region
# Intermediate densities are realized with fewer tows: the local path spacing
# is TOW_SPACING / rho, i.e. fiber volume per unit width is proportional to rho.
STEP = 0.5  # mm - streamline integration step
# "skeleton": tows laid parallel to member centerlines, joined through junctions
#             by the straightest continuation (one fiber bundle per load path).
# "streamline": evenly spaced streamlines of the optimized orientation field.
PATH_METHOD = "skeleton"
RHO_MEMBER = 0.5  # density threshold of the member mask for the skeleton
SPUR_LEN = 15.0  # mm - skeleton branches shorter than this are pruned
MAX_TURN = 55.0  # deg - max fiber direction change when passing a junction
SMOOTH = 4.0  # mm - Gaussian smoothing length of the member centerlines

OUT = Path("wing_spar_output")

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


_le = PchipInterpolator(*LE_PTS)
_te = PchipInterpolator(*TE_PTS)


def y_edges(x):
    """Trailing and leading edge y-coordinates at span station x."""
    if isinstance(x, torch.Tensor):
        xn = x.numpy()
        return torch.as_tensor(_te(xn)), torch.as_tensor(_le(xn))
    return float(_te(x)), float(_le(x))


# Mesh: structured triangles on the unit square, mapped onto the planform
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
root = x > L - tol
le_edge = uv[:, 1] > 1.0 - tol
te_edge = uv[:, 1] < tol
h = L / NX

# Root fittings: clamp the root-edge nodes inside each band
passive = centers[:, 0] < 1.5 * h  # tip rib
for y_lo, y_hi in ROOT_FITTINGS:
    spar.constraints[root & (y >= y_lo - tol) & (y <= y_hi + tol), :] = True
    y_mid = 0.5 * (y_lo + y_hi)
    pad = (centers[:, 0] > L - 2.0 * h) & (
        (centers[:, 1] - y_mid).abs() < 0.5 * (y_hi - y_lo) + 0.5 * h
    )
    passive |= pad

# Loads at the rib stations, each on the nearest edge node
f_drag = torch.zeros_like(spar.forces)
f_lift = torch.zeros_like(spar.forces)
station_xy = []
n_tip = sum(1 for xs, _ in STATIONS if xs == 0.0)
n_rib = len(STATIONS) - n_tip
n_tip_le = sum(1 for xs, e in STATIONS if xs == 0.0 and e == "le")
n_rib_le = sum(1 for xs, e in STATIONS if xs > 0.0 and e == "le")
for xs, edge in STATIONS:
    on_edge = le_edge if edge == "le" else te_edge
    idx = torch.nonzero(on_edge).ravel()
    k = idx[torch.argmin((x[idx] - xs).abs())]
    station_xy.append(nodes[k, :2].tolist())
    tip_station = xs == 0.0
    w_lift = TIP_SHARE / n_tip if tip_station else (1 - TIP_SHARE) / n_rib
    f_lift[k, 2] += w_lift * LIFT
    if edge == "le":
        w_drag = TIP_SHARE / n_tip_le if tip_station else (1 - TIP_SHARE) / n_rib_le
        f_drag[k, 1] -= w_drag * DRAG
    passive |= torch.linalg.norm(centers[:, :2] - nodes[k, :2], dim=1) < PAD_RADIUS
station_xy = np.array(station_xy)
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


def physical_density(x, beta, eta=0.5):
    """Density filter followed by a smoothed Heaviside projection."""
    x_t = (H @ x) / H.sum(dim=1)
    num = torch.tanh(torch.tensor(beta * eta)) + torch.tanh(beta * (x_t - eta))
    den = torch.tanh(torch.tensor(beta * eta)) + torch.tanh(torch.tensor(beta * (1 - eta)))
    return torch.where(passive, torch.ones_like(x), num / den)


def beta_at(it):
    """Projection sharpness continuation: 1, then doubled every BETA_EVERY."""
    if it < BETA_START:
        return 1.0
    return min(BETA_MAX, 2.0 ** (1 + (it - BETA_START) // BETA_EVERY))


def optimize(x, theta, n_iter=N_ITER):
    x_min = 1e-3 * torch.ones_like(x)
    x_max = torch.ones_like(x)
    with torch.no_grad():
        rho = physical_density(x, 1.0)
        C0 = [compliance(rho, theta, f).item() for f in (f_drag, f_lift)]

    history = []
    for it in tqdm(range(n_iter)):
        beta = beta_at(it)

        # One adjoint per load case (each solve owns its own graph)
        rho = physical_density(x, beta)
        C_d = W_DRAG * compliance(rho, theta, f_drag) / C0[0]
        g_d = torch.autograd.grad(C_d, (x, theta))
        rho = physical_density(x, beta)
        C_l = W_LIFT * compliance(rho, theta, f_lift) / C0[1]
        g_l = torch.autograd.grad(C_l, (x, theta))
        C = C_d + C_l
        dC_dx = torch.clamp(g_d[0] + g_l[0], max=-1e-12)
        dC_dtheta = g_d[1] + g_l[1]

        # Volume sensitivity through filter and projection
        rho = physical_density(x, beta)
        dV_dx = torch.autograd.grad((rho * area).sum(), x)[0]
        dV_dx = torch.clamp(dV_dx, min=1e-12)

        with torch.no_grad():

            def make_step(mu):
                upper = torch.min(x_max, x + MOVE)
                lower = torch.max(x_min, x - MOVE)
                x_trial = x * (-dC_dx / (mu * dV_dx)) ** 0.5
                x_new = torch.max(torch.min(x_trial, upper), lower)
                x_new[passive] = 1.0
                return x_new

            def g(mu):
                rho_k = physical_density(make_step(mu), beta)
                return ((rho_k * area).sum() - V_0).item()

            mu = bisect(g, 1e-12, 1e6)
            x.data = make_step(mu)

            # Normalized gradient step on the angle, then regularize
            step = ORI_MAX_STEP * dC_dtheta / dC_dtheta.abs().max()
            theta.data = smooth_orientation(theta - step, physical_density(x, beta))

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
        self.gy = np.arange(min(TE_PTS[1]) - res, max(_le(np.linspace(0, L, 200))) + res, res)
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
        self.res = res
        self.rho_grid = fields[2]

    def inside(self, p):
        if p[0] < 0.0 or p[0] > L:
            return False
        y_b, y_t = y_edges(p[0])
        return y_b <= p[1] <= y_t

    def solid(self, p):
        return self.inside(p) and self.rho(p)[0] >= RHO_SOLID

    def spacing(self, p):
        """Local path spacing realizing the optimized density with tows."""
        r = np.clip(self.rho(p)[0], RHO_SOLID, 1.0)
        return TOW_SPACING / r

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


def trace(field, paths, seed, d_test_ratio, max_len=5.0 * L):
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
            if not field.solid(q) or paths.too_close(q, d_test_ratio * field.spacing(q)):
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
    """Jobard-Lefer: seed new streamlines one local spacing from accepted ones.

    d_sep is the spacing in fully dense regions; it grows as d_sep / rho.
    """
    paths = PathSet(cell=d_sep / RHO_SOLID)
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
    # Densest (main load-carrying) members first
    queue = sorted(seeds, key=lambda p: -field.rho(p)[0])

    while queue:
        seed = queue.pop(0)
        if not field.solid(seed) or paths.too_close(seed, 0.99 * field.spacing(seed)):
            continue
        line = trace(field, paths, seed, d_test_ratio)
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
            sep = field.spacing(line[k])
            for s in (1.0, -1.0):
                queue.insert(0, line[k] + s * sep * normal[k])
    return result


# ---------------------------------------------------------------------------
# Continuous fiber paths along the member skeleton
# ---------------------------------------------------------------------------

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def member_mask(field):
    """Binary member mask on the background grid, cleaned for skeletonization."""
    X, Y = np.meshgrid(field.gx, field.gy, indexing="ij")
    y_b, y_t = _te(X), _le(X)
    mask = (field.rho_grid >= RHO_MEMBER) & (Y >= y_b) & (Y <= y_t)
    mask = closing(mask, disk(2))
    mask = remove_small_holes(mask, max_size=60)
    mask = remove_small_objects(mask, max_size=60)
    return mask


def skeleton_graph(mask):
    """Skeleton split into edges (pixel chains) between junction clusters."""
    skel = skeletonize(mask)
    pad = np.pad(skel, 1)
    nbr = sum(np.roll(np.roll(pad, di, 0), dj, 1) for di, dj in _NB)[1:-1, 1:-1]
    junction = skel & (nbr >= 3)
    j_lab, n_j = label(junction, structure=np.ones((3, 3)))
    j_xy = np.array([np.argwhere(j_lab == k + 1).mean(0) for k in range(n_j)])
    e_lab, n_e = label(skel & ~junction, structure=np.ones((3, 3)))

    edges = []
    for k in range(1, n_e + 1):
        pix = {tuple(p) for p in np.argwhere(e_lab == k)}

        def nbrs(p):
            return [(p[0] + a, p[1] + b) for a, b in _NB if (p[0] + a, p[1] + b) in pix]

        start = next((p for p in pix if len(nbrs(p)) <= 1), next(iter(pix)))
        chain, seen = [start], {start}
        while True:
            nxt = [q for q in nbrs(chain[-1]) if q not in seen]
            if not nxt:
                break
            chain.append(nxt[0])
            seen.add(nxt[0])

        def touching(p):
            js = {j_lab[p[0] + a, p[1] + b] for a, b in _NB
                  if 0 <= p[0] + a < skel.shape[0] and 0 <= p[1] + b < skel.shape[1]}
            js.discard(0)
            return min(js) - 1 if js else None

        ends = (touching(chain[0]), touching(chain[-1]))
        pts = np.array(chain, dtype=float)
        if ends[0] is not None:
            pts = np.vstack([j_xy[ends[0]], pts])
        if ends[1] is not None:
            pts = np.vstack([pts, j_xy[ends[1]]])
        edges.append({"pts": pts, "ends": list(ends)})
    return edges, j_xy


def polyline_length(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum() if len(p) > 1 else 0.0


def end_direction(pts, at_start, span=10):
    """Unit tangent pointing out of the polyline at one end."""
    seg = pts[: span + 1] if at_start else pts[-span - 1 :][::-1]
    d = seg[0] - seg[-1]
    return d / (np.linalg.norm(d) + 1e-12)


def build_strokes(edges):
    """Join edges through junctions by the straightest continuation."""
    # Prune short dangling spurs
    edges = [e for e in edges
             if not ((e["ends"][0] is None or e["ends"][1] is None)
                     and polyline_length(e["pts"]) < SPUR_LEN)]
    # Incident edge-ends per junction
    inc = {}
    for i, e in enumerate(edges):
        for side, j in enumerate(e["ends"]):
            if j is not None:
                inc.setdefault(j, []).append((i, side))
    # Greedy pairing at each junction: most opposite directions first
    link = {}
    cos_max = -np.cos(np.deg2rad(MAX_TURN))
    for j, ends in inc.items():
        dirs = {(i, s): end_direction(edges[i]["pts"], s == 0) for i, s in ends}
        cand = sorted(
            ((dirs[a] @ dirs[b], a, b) for ia, a in enumerate(ends)
             for b in ends[ia + 1 :] if a[0] != b[0]),
            key=lambda t: t[0],
        )
        for c, a, b in cand:
            if c <= cos_max and a not in link and b not in link:
                link[a], link[b] = b, a
    # Walk chains of linked edges
    used, strokes = set(), []
    for i0 in range(len(edges)):
        if i0 in used:
            continue
        # Go to one end of the chain
        i, side = i0, 0
        seen = {i0}
        while (i, side) in link:
            i, s2 = link[(i, side)]
            if i in seen:
                break
            seen.add(i)
            side = 1 - s2
        # Walk forward from (i, side)
        pts, cur, entry = [], i, side
        while cur is not None and cur not in used:
            used.add(cur)
            p = edges[cur]["pts"]
            p = p if entry == 0 else p[::-1]
            pts.append(p if not pts else p[1:])
            nxt = link.get((cur, 1 - entry))
            cur, entry = (nxt if nxt is not None else (None, None))
        strokes.append(np.vstack(pts))
    return strokes


def resample(p, ds=1.0):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    t = np.arange(0.0, s[-1] + 1e-9, ds)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def extend_free_end(p, mask, field, at_start):
    """Extend a free stroke end along its tangent to the member boundary."""
    d = end_direction(p, at_start)
    q = p[0] if at_start else p[-1]
    ext = []
    for _ in range(40):
        q = q + d * field.res
        i = int(round((q[0] - field.gx[0]) / field.res))
        j = int(round((q[1] - field.gy[0]) / field.res))
        if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]) or not mask[i, j]:
            break
        ext.append(q.copy())
    if not ext:
        return p
    ext = np.array(ext)
    return np.vstack([ext[::-1], p]) if at_start else np.vstack([p, ext])


def skeleton_paths(field):
    """Continuous tows: offsets of smoothed member centerlines (load paths)."""
    mask = member_mask(field)
    half_width = distance_transform_edt(mask) * field.res
    edges, _ = skeleton_graph(mask)
    strokes_px = build_strokes(edges)

    to_xy = lambda p: np.column_stack(  # noqa: E731
        [field.gx[0] + p[:, 0] * field.res, field.gy[0] + p[:, 1] * field.res]
    )
    hw = RegularGridInterpolator((field.gx, field.gy), half_width,
                                 bounds_error=False, fill_value=0.0)
    strokes, tows = [], []
    for sp in strokes_px:
        p = resample(to_xy(sp))
        if polyline_length(p) < SPUR_LEN or len(p) < 5:
            continue
        sig = SMOOTH / field.res
        p_s = np.column_stack([gaussian_filter1d(p[:, k], sig, mode="nearest")
                               for k in (0, 1)])
        p_s = extend_free_end(p_s, mask, field, True)
        p_s = extend_free_end(p_s, mask, field, False)
        p_s = resample(p_s)
        strokes.append(p_s)
        # Number of tows from the member width along the stroke
        width = 2.0 * np.median(hw(p_s))
        n = max(1, int(round(width / TOW_SPACING)))
        t = np.gradient(p_s, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        for k in range(n):
            off = (k - 0.5 * (n - 1)) * TOW_SPACING
            tows.append(p_s + off * nrm)
    return tows, strokes


def fiber_alignment(field, strokes):
    """Mean |cos| between optimized fiber angle and the load-path tangent."""
    vals = []
    for p in strokes:
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        a = 0.5 * np.arctan2(field.s2(p), field.c2(p))
        vals.append(np.abs(t[:, 0] * np.cos(a) + t[:, 1] * np.sin(a)))
    return float(np.mean(np.concatenate(vals)))


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


def draw_bcs(ax):
    """Root fittings, rib stations and load symbols."""
    for i, (y_lo, y_hi) in enumerate(ROOT_FITTINGS):
        ax.plot([L + 3, L + 3], [y_lo, y_hi], color="tab:blue", lw=5,
                label="root fittings (clamped)" if i == 0 else None)
    drag = station_xy[[e == "le" for _, e in STATIONS]]
    ax.quiver(drag[:, 0], drag[:, 1] + 28, 0, -1, color="tab:blue",
              scale=12, width=0.004, label="drag (-y)")
    ax.plot(station_xy[:, 0], station_xy[:, 1], "o", ms=9, mfc="none",
            mec="tab:red", mew=2, label="lift (+z) / rib stations")
    ax.plot(station_xy[:, 0], station_xy[:, 1], ".", color="tab:red")


def run_optimization():
    # Random (smoothed) initial fiber field: a uniform 0 deg start is a
    # stationary point for members loaded along the chord.
    torch.manual_seed(0)
    theta0 = (torch.rand(n_elem) - 0.5) * torch.pi
    theta = smooth_orientation(theta0, torch.ones(n_elem)).requires_grad_(True)
    x = VOLFRAC * torch.ones(n_elem)
    x[passive] = 1.0
    x.requires_grad_(True)

    history = optimize(x, theta)
    rho = physical_density(x.detach(), beta_at(N_ITER - 1))

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
    draw_bcs(ax)
    ax.set_title("Optimized density and fiber orientation")
    fig.tight_layout()
    fig.savefig(OUT / "density_orientation.png", dpi=150)


def generate_paths():
    """Continuous fiber paths from the design saved by run_optimization()."""
    design = np.load(OUT / "design.npz")
    c_np, rho_np, th_np = design["centers"], design["rho"], design["theta"]
    tri = elements.numpy()
    xy = nodes[:, :2].numpy()

    field = DirectorField(c_np, th_np, rho_np)
    if PATH_METHOD == "skeleton":
        paths, strokes = skeleton_paths(field)
        align = fiber_alignment(field, strokes)
        print(f"{len(strokes)} load paths; optimized fibers vs. path tangent: "
              f"mean |cos| = {align:.3f}")
    else:
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
    draw_bcs(ax)
    ax.set_aspect("equal")
    ax.set_title(
        f"Continuous fiber paths ({len(paths)} tows, spacing {TOW_SPACING} mm / density)"
    )
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fiber_paths.png", dpi=200)


if __name__ == "__main__":
    import sys

    OUT.mkdir(exist_ok=True)
    if "--paths-only" not in sys.argv:
        run_optimization()
    generate_paths()
