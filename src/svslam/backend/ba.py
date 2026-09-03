"""Windowed bundle adjustment, written from scratch.

Bundle adjustment jointly refines keyframe poses and landmark positions so that
every landmark projects as close as possible to where it was actually seen.
The problem is large but extremely structured, and exploiting that structure is
the whole game.

The normal equations have the arrowhead form::

    | U   W | | dc |     | b_c |
    |       | |    |  = -|     |
    | W^T V | | dp |     | b_p |

``U`` is block-diagonal with 6x6 camera blocks, ``V`` is block-diagonal with 3x3
landmark blocks, and ``W`` is sparse -- a block is non-zero only where camera
``i`` actually observes landmark ``j``.  Solving this directly costs
``O((6C + 3P)^3)``, and ``P`` runs into the thousands, so that is hopeless.

The **Schur complement** marginalises the landmarks out first::

    S      = U - W V^-1 W^T
    S dc   = -(b_c - W V^-1 b_p)
    dp_j   = V_j^-1 (-b_p_j - sum_i W_ij^T dc_i)

``V`` is block-diagonal so ``V^-1`` is 3x3 inverses, and ``S`` is only
``6C x 6C`` -- 120x120 for a 20-keyframe window.  Same answer, orders of
magnitude cheaper.  ``tests/test_schur.py`` checks that "same answer" claim
against a dense solve of the full system.

Levenberg-Marquardt wraps the solve: damping is added to the diagonals of ``U``
and ``V`` before forming ``S``, and the damping is raised or lowered depending
on whether the step actually reduced the cost.  Pure Gauss-Newton diverges here
often enough to be a real problem -- a distant landmark with a near-singular
3x3 block produces a huge step and the window blows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reprojection import reprojection_jacobians, reprojection_residual
from ..se3 import normalise_rotation, se3_exp

__all__ = [
    "BAConfig",
    "BAProblem",
    "BAResult",
    "build_normal_equations",
    "solve_schur",
    "solve_dense",
    "bundle_adjust",
]


@dataclass(frozen=True)
class BAConfig:
    """Levenberg-Marquardt and robustification settings."""

    max_iterations: int = 10
    initial_lambda: float = 1e-4
    lambda_up: float = 10.0
    lambda_down: float = 0.1
    min_lambda: float = 1e-12
    max_lambda: float = 1e8
    #: Huber threshold on the residual norm, in pixels.
    huber_delta: float = 2.0
    #: Stop when the relative cost decrease falls below this.
    cost_tolerance: float = 1e-4
    #: Stop when the parameter step norm falls below this.
    step_tolerance: float = 1e-9
    #: Ridge added to V blocks before inversion, for numerical safety.
    point_ridge: float = 1e-9


@dataclass
class BAProblem:
    """A bundle-adjustment window.

    Parameters
    ----------
    poses_cw:
        ``(C, 4, 4)`` world-to-camera transforms, one per keyframe.
    points:
        ``(P, 3)`` landmark positions in the world frame.
    camera_index, point_index:
        ``(M,)`` integer arrays saying which camera and landmark each
        observation belongs to.
    observations:
        ``(M, 2)`` pixels, or ``(M, 3)`` for stereo observations.
    fixed_cameras:
        Indices held constant.  At least one camera must be fixed, otherwise the
        problem has a seven-dimensional gauge freedom (six of SE(3) plus scale
        for monocular) and ``S`` is singular.
    """

    poses_cw: np.ndarray
    points: np.ndarray
    camera_index: np.ndarray
    point_index: np.ndarray
    observations: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float | None = None
    fixed_cameras: tuple[int, ...] = (0,)
    fixed_points: tuple[int, ...] = ()
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.poses_cw = np.array(self.poses_cw, dtype=float).reshape(-1, 4, 4)
        self.points = np.array(self.points, dtype=float).reshape(-1, 3)
        self.camera_index = np.asarray(self.camera_index, dtype=int).reshape(-1)
        self.point_index = np.asarray(self.point_index, dtype=int).reshape(-1)
        k = 3 if self.baseline is not None else 2
        self.observations = np.asarray(self.observations, dtype=float).reshape(-1, k)

    @property
    def n_cameras(self) -> int:
        return int(self.poses_cw.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_observations(self) -> int:
        return int(self.observations.shape[0])

    @property
    def residual_dim(self) -> int:
        return 3 if self.baseline is not None else 2


@dataclass
class BAResult:
    """Outcome of :func:`bundle_adjust`."""

    poses_cw: np.ndarray
    points: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    converged: bool
    initial_rmse: float
    final_rmse: float
    history: list[float] = field(default_factory=list)


def _residuals_and_jacobians(problem: BAProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked per-observation residual and Jacobian blocks.

    Returns ``(residual (M,k), J_pose (M,k,6), J_point (M,k,3))``.
    """
    m = problem.n_observations
    k = problem.residual_dim
    residual = np.zeros((m, k))
    J_pose = np.zeros((m, k, 6))
    J_point = np.zeros((m, k, 3))
    if m == 0:
        return residual, J_pose, J_point

    for cam in np.unique(problem.camera_index):
        sel = np.flatnonzero(problem.camera_index == cam)
        pts = problem.points[problem.point_index[sel]]
        T = problem.poses_cw[cam]
        residual[sel] = reprojection_residual(
            T, pts, problem.observations[sel],
            problem.fx, problem.fy, problem.cx, problem.cy, problem.baseline,
        )
        jp, jx = reprojection_jacobians(T, pts, problem.fx, problem.fy, problem.baseline)
        J_pose[sel] = jp
        J_point[sel] = jx
    return residual, J_pose, J_point


def _robust_weights(residual: np.ndarray, delta: float, prior: np.ndarray | None) -> np.ndarray:
    """Huber IRLS weights, optionally scaled by caller-supplied priors."""
    norms = np.linalg.norm(residual, axis=1)
    w = np.where(norms <= delta, 1.0, delta / np.maximum(norms, 1e-12))
    if prior is not None:
        w = w * np.asarray(prior, dtype=float).reshape(-1)
    return w


def _huber_cost(residual: np.ndarray, delta: float, prior: np.ndarray | None) -> float:
    """The *actual* Huber cost, ``sum rho(||r||)``.

    This is deliberately not ``sum w ||r||^2``.  That quantity is the IRLS
    surrogate the normal equations are built from, and because ``w`` is
    recomputed at every linearisation it is not a fixed objective -- an LM step
    can lower the true cost while raising the surrogate, and then LM rejects a
    perfectly good step, raises the damping, and stalls.  Accept/reject has to
    test the real cost.
    """
    norms = np.linalg.norm(residual, axis=1)
    rho = np.where(
        norms <= delta, norms ** 2, 2.0 * delta * norms - delta * delta
    )
    if prior is not None:
        rho = rho * np.asarray(prior, dtype=float).reshape(-1)
    return float(np.sum(rho))


def build_normal_equations(
    problem: BAProblem, huber_delta: float = 2.0
) -> dict[str, object]:
    """Assemble the arrowhead normal equations for the current linearisation.

    Returns a dict with ``U`` ``(C, 6, 6)``, ``V`` ``(P, 3, 3)``, ``W``
    (a dict keyed by ``(camera, point)`` holding 6x3 blocks), ``b_c`` ``(C, 6)``,
    ``b_p`` ``(P, 3)``, the robustified ``cost``, and the raw ``rmse``.
    """
    residual, J_pose, J_point = _residuals_and_jacobians(problem)
    w = _robust_weights(residual, huber_delta, problem.weights)

    C, P = problem.n_cameras, problem.n_points
    ci = problem.camera_index
    pi = problem.point_index

    # Accumulate the blocks with scatter-adds rather than a Python loop over
    # observations: a local window has thousands of them and this assembly runs
    # twice per LM iteration.
    U = np.zeros((C, 6, 6))
    V = np.zeros((P, 3, 3))
    b_c = np.zeros((C, 6))
    b_p = np.zeros((P, 3))
    if problem.n_observations:
        wp = w[:, None, None]
        np.add.at(U, ci, wp * np.einsum("nki,nkj->nij", J_pose, J_pose))
        np.add.at(V, pi, wp * np.einsum("nki,nkj->nij", J_point, J_point))
        np.add.at(b_c, ci, w[:, None] * np.einsum("nki,nk->ni", J_pose, residual))
        np.add.at(b_p, pi, w[:, None] * np.einsum("nki,nk->ni", J_point, residual))

    W: dict[tuple[int, int], np.ndarray] = {}
    if problem.n_observations:
        blocks = w[:, None, None] * np.einsum("nki,nkj->nij", J_pose, J_point)
        keys = ci.astype(np.int64) * (P + 1) + pi.astype(np.int64)
        unique, inverse = np.unique(keys, return_inverse=True)
        summed = np.zeros((unique.size, 6, 3))
        np.add.at(summed, inverse, blocks)
        cams = (unique // (P + 1)).astype(int)
        pts = (unique % (P + 1)).astype(int)
        W = {(int(a), int(b)): summed[k] for k, (a, b) in enumerate(zip(cams, pts))}

    sq = np.sum(residual * residual, axis=1)
    cost = _huber_cost(residual, huber_delta, problem.weights)
    rmse = float(np.sqrt(np.mean(sq))) if problem.n_observations else 0.0
    return {
        "U": U, "V": V, "W": W, "b_c": b_c, "b_p": b_p,
        "cost": cost, "rmse": rmse, "residual": residual, "weights": w,
    }


def _free_indices(n: int, fixed: tuple[int, ...]) -> tuple[np.ndarray, dict[int, int]]:
    fixed_set = set(int(f) for f in fixed)
    free = np.array([i for i in range(n) if i not in fixed_set], dtype=int)
    return free, {int(g): k for k, g in enumerate(free)}


def _damped(block: np.ndarray, lam: float) -> np.ndarray:
    """Marquardt damping: scale the diagonal by ``1 + lambda``.

    Scaling rather than adding makes the damping invariant to the units of each
    parameter, which matters here because rotation is in radians and translation
    in metres and their curvatures differ by orders of magnitude.
    """
    out = np.array(block, dtype=float)
    diag = np.diag(out).copy()
    np.fill_diagonal(out, diag * (1.0 + lam) + lam * 1e-12)
    return out


def solve_schur(
    normal: dict[str, object],
    problem: BAProblem,
    lam: float = 0.0,
    point_ridge: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the damped normal equations via the Schur complement.

    Returns ``(delta_cameras (C, 6), delta_points (P, 3))`` with zeros in the
    rows of fixed parameters.
    """
    U = normal["U"]; V = normal["V"]; W = normal["W"]  # type: ignore[assignment]
    b_c = normal["b_c"]; b_p = normal["b_p"]  # type: ignore[assignment]

    free_cams, cam_slot = _free_indices(problem.n_cameras, problem.fixed_cameras)
    free_pts, pt_slot = _free_indices(problem.n_points, problem.fixed_points)
    nc, npt = free_cams.size, free_pts.size

    delta_c = np.zeros((problem.n_cameras, 6))
    delta_p = np.zeros((problem.n_points, 3))
    if nc == 0 and npt == 0:
        return delta_c, delta_p

    # V^-1 for the free landmarks, inverted as one batched call.
    V_inv = np.zeros((problem.n_points, 3, 3))
    if npt:
        blocks = np.array([_damped(V[j], lam) for j in free_pts])
        blocks += np.eye(3) * point_ridge
        V_inv[free_pts] = np.linalg.inv(blocks)

    S = np.zeros((nc * 6, nc * 6))
    rhs = np.zeros(nc * 6)
    for i in free_cams:
        s = cam_slot[int(i)] * 6
        S[s:s + 6, s:s + 6] = _damped(U[i], lam)
        rhs[s:s + 6] = -b_c[i]

    # Coupling blocks, laid out as (free camera, free landmark, 6, 3).  For a
    # keyframe window this array is a few megabytes and lets the whole Schur
    # accumulation happen in two einsums instead of a loop over every
    # (camera, camera, landmark) triple.
    if nc and npt:
        dense_ok = nc * npt <= 4_000_000
        if dense_ok:
            Wmat = np.zeros((nc, npt, 6, 3))
            for (i, j), block in W.items():
                if i in cam_slot and j in pt_slot:
                    Wmat[cam_slot[i], pt_slot[j]] = block
            Y = np.einsum("cpij,pjk->cpik", Wmat, V_inv[free_pts])
            coupling = np.einsum("apij,bpkj->abik", Y, Wmat)
            for a in range(nc):
                sa = a * 6
                rhs[sa:sa + 6] += np.einsum("pij,pj->i", Y[a], b_p[free_pts])
                for b in range(nc):
                    sb = b * 6
                    S[sa:sa + 6, sb:sb + 6] -= coupling[a, b]
        else:  # pragma: no cover - only for very large problems
            by_point: dict[int, list[int]] = {}
            for (i, j) in W:
                if j in pt_slot and i in cam_slot:
                    by_point.setdefault(j, []).append(i)
            for j, cams in by_point.items():
                Vj = V_inv[j]
                for a in cams:
                    Ya = W[(a, j)] @ Vj
                    sa = cam_slot[a] * 6
                    rhs[sa:sa + 6] += Ya @ b_p[j]
                    for b in cams:
                        sb = cam_slot[b] * 6
                        S[sa:sa + 6, sb:sb + 6] -= Ya @ W[(b, j)].T

    if nc:
        # S is symmetric positive definite once damped; Cholesky is both faster
        # and a check that the damping is doing its job.
        try:
            L = np.linalg.cholesky(S)
            dc = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        except np.linalg.LinAlgError:
            dc = np.linalg.lstsq(S, rhs, rcond=None)[0]
        for i in free_cams:
            delta_c[i] = dc[cam_slot[int(i)] * 6: cam_slot[int(i)] * 6 + 6]

    # Back-substitute for the landmarks.
    if npt:
        acc = -b_p[free_pts].copy()
        for (i, j), block in W.items():
            if i in cam_slot and j in pt_slot:
                acc[pt_slot[j]] -= block.T @ delta_c[i]
        delta_p[free_pts] = np.einsum("pij,pj->pi", V_inv[free_pts], acc)
    return delta_c, delta_p


def solve_dense(
    normal: dict[str, object],
    problem: BAProblem,
    lam: float = 0.0,
    point_ridge: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference solve of the same system as one dense linear system.

    Only used to validate :func:`solve_schur` in the tests -- it is far too slow
    for a real window, which is the entire reason the Schur complement exists.
    """
    U = normal["U"]; V = normal["V"]; W = normal["W"]  # type: ignore[assignment]
    b_c = normal["b_c"]; b_p = normal["b_p"]  # type: ignore[assignment]

    free_cams, cam_slot = _free_indices(problem.n_cameras, problem.fixed_cameras)
    free_pts, pt_slot = _free_indices(problem.n_points, problem.fixed_points)
    nc, npt = free_cams.size, free_pts.size
    n = nc * 6 + npt * 3

    H = np.zeros((n, n))
    g = np.zeros(n)
    for i in free_cams:
        s = cam_slot[int(i)] * 6
        H[s:s + 6, s:s + 6] = _damped(U[i], lam)
        g[s:s + 6] = -b_c[i]
    for j in free_pts:
        s = nc * 6 + pt_slot[int(j)] * 3
        H[s:s + 3, s:s + 3] = _damped(V[j], lam) + np.eye(3) * point_ridge
        g[s:s + 3] = -b_p[j]
    for (i, j), block in W.items():
        if i not in cam_slot or j not in pt_slot:
            continue
        si = cam_slot[i] * 6
        sj = nc * 6 + pt_slot[j] * 3
        H[si:si + 6, sj:sj + 3] = block
        H[sj:sj + 3, si:si + 6] = block.T

    sol = np.linalg.solve(H, g) if n else np.zeros(0)
    delta_c = np.zeros((problem.n_cameras, 6))
    delta_p = np.zeros((problem.n_points, 3))
    for i in free_cams:
        delta_c[i] = sol[cam_slot[int(i)] * 6: cam_slot[int(i)] * 6 + 6]
    for j in free_pts:
        s = nc * 6 + pt_slot[int(j)] * 3
        delta_p[j] = sol[s:s + 3]
    return delta_c, delta_p


def _apply(problem: BAProblem, delta_c: np.ndarray, delta_p: np.ndarray) -> BAProblem:
    poses = np.array(problem.poses_cw, dtype=float)
    for i in range(problem.n_cameras):
        if np.any(delta_c[i]):
            poses[i] = se3_exp(delta_c[i]) @ poses[i]
            poses[i, :3, :3] = normalise_rotation(poses[i, :3, :3])
    points = problem.points + delta_p
    updated = BAProblem(
        poses_cw=poses,
        points=points,
        camera_index=problem.camera_index,
        point_index=problem.point_index,
        observations=problem.observations,
        fx=problem.fx, fy=problem.fy, cx=problem.cx, cy=problem.cy,
        baseline=problem.baseline,
        fixed_cameras=problem.fixed_cameras,
        fixed_points=problem.fixed_points,
        weights=problem.weights,
    )
    return updated


def bundle_adjust(problem: BAProblem, config: BAConfig | None = None) -> BAResult:
    """Run Levenberg-Marquardt bundle adjustment on a window.

    The damping schedule is the standard one: accept a step that lowers the
    robust cost and divide ``lambda`` by ten (moving towards Gauss-Newton),
    reject a step that raises it and multiply ``lambda`` by ten (moving towards
    gradient descent with a shorter step).
    """
    config = config or BAConfig()
    current = problem
    normal = build_normal_equations(current, config.huber_delta)
    initial_cost = float(normal["cost"])
    initial_rmse = float(normal["rmse"])
    cost = initial_cost
    lam = config.initial_lambda
    history = [cost]
    converged = False
    iteration = 0

    for iteration in range(1, config.max_iterations + 1):
        delta_c, delta_p = solve_schur(normal, current, lam, config.point_ridge)
        step = float(np.linalg.norm(delta_c)) + float(np.linalg.norm(delta_p))
        if not np.isfinite(step):
            lam = min(lam * config.lambda_up, config.max_lambda)
            continue
        if step < config.step_tolerance:
            converged = True
            break

        candidate = _apply(current, delta_c, delta_p)
        cand_normal = build_normal_equations(candidate, config.huber_delta)
        cand_cost = float(cand_normal["cost"])

        if cand_cost < cost:
            relative = (cost - cand_cost) / max(cost, 1e-16)
            current, normal, cost = candidate, cand_normal, cand_cost
            lam = max(lam * config.lambda_down, config.min_lambda)
            history.append(cost)
            if relative < config.cost_tolerance:
                converged = True
                break
        else:
            lam = min(lam * config.lambda_up, config.max_lambda)
            if lam >= config.max_lambda:
                break

    return BAResult(
        poses_cw=current.poses_cw,
        points=current.points,
        initial_cost=initial_cost,
        final_cost=cost,
        iterations=iteration,
        converged=converged,
        initial_rmse=initial_rmse,
        final_rmse=float(normal["rmse"]),
        history=history,
    )
