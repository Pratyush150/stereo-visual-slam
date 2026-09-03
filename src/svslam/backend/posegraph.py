"""SE(3) pose-graph optimisation with robust kernels.

When a loop closure is detected, the accumulated drift has to be distributed
back over the whole trajectory.  That is a pose-graph problem: nodes are
keyframe poses, edges are relative transform measurements, and the optimiser
finds the poses that best explain all the edges at once.

Formulation
-----------
Node ``i`` holds ``T_i = T_wc(i)``, the camera pose in the world.  An edge
carries a measured relative transform ``Z_ij ~ T_i^-1 T_j`` and an information
matrix.  The error is::

    e_ij = log( Z_ij^-1 . T_i^-1 . T_j )

With right perturbations ``T <- T exp(delta)`` the analytic Jacobians are::

    de/d delta_i = -J_l^-1(e) . Adj(Z_ij^-1)
    de/d delta_j =  J_r^-1(e) = J_l^-1(-e)

which follow from the first-order BCH expansion of
``log(exp(a) exp(e) exp(b))``.  Both are checked against central differences in
``tests/test_posegraph.py``.

Why the robust kernel is not optional
-------------------------------------
An odometry edge that is wrong by 20 cm is a nuisance.  A *loop closure* edge
that is wrong -- because the place recogniser matched two different but
similar-looking stretches of road -- is catastrophic: least squares will happily
fold the map in half to satisfy it, and every pose in between is destroyed.
Huber bounds the influence of such an edge; DCS goes further and scales the
whole edge down as its error grows, so a badly wrong loop closure ends up
contributing almost nothing.  ``tests/test_robust_kernel.py`` injects a false
loop closure and checks that the map survives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..se3 import (
    normalise_rotation,
    se3_adjoint,
    se3_exp,
    se3_inverse,
    se3_left_jacobian_inverse,
    se3_log,
)

try:  # pragma: no cover - optional acceleration
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    sp = None
    spla = None
    _HAVE_SCIPY = False

__all__ = [
    "PoseGraphEdge",
    "PoseGraphConfig",
    "PoseGraphResult",
    "edge_error",
    "edge_jacobians",
    "robust_weight",
    "optimise_pose_graph",
    "have_scipy",
]


def have_scipy() -> bool:
    """True if the sparse linear-algebra path is available.

    Without SciPy the solver falls back to a dense Cholesky.  That is exact and
    fine for the graph sizes here (a few hundred keyframes -> a few thousand
    unknowns); it simply costs more memory and time than the sparse path.
    """
    return _HAVE_SCIPY


@dataclass
class PoseGraphEdge:
    """One relative-pose constraint between two keyframes."""

    i: int
    j: int
    measurement: np.ndarray  # 4x4, T_i^-1 T_j
    information: np.ndarray = field(default_factory=lambda: np.eye(6))
    is_loop: bool = False

    def __post_init__(self) -> None:
        self.measurement = np.array(self.measurement, dtype=float).reshape(4, 4)
        self.information = np.array(self.information, dtype=float).reshape(6, 6)


@dataclass(frozen=True)
class PoseGraphConfig:
    """Solver settings."""

    max_iterations: int = 60
    initial_lambda: float = 1e-6
    lambda_up: float = 10.0
    lambda_down: float = 0.1
    #: ``"huber"``, ``"dcs"`` or ``"none"``.
    kernel: str = "dcs"
    #: Huber threshold on the Mahalanobis norm.
    huber_delta: float = 1.0
    #: DCS free parameter; smaller is more aggressive.
    dcs_phi: float = 1.0
    #: Apply the kernel only to loop edges, trusting sequential odometry edges.
    kernel_on_loops_only: bool = True
    #: Iterations run with the kernel disabled before robust reweighting starts.
    #: A *correct* loop closure has a large residual at the moment it is added --
    #: that residual is the drift it exists to remove.  Switching the kernel on
    #: immediately would therefore down-weight exactly the edges that matter.
    #: These warm-up iterations let the graph absorb the drift first, after which
    #: a correct loop has a small residual and a false one still has a large one,
    #: which is precisely the distinction the kernel can act on.
    warmup_iterations: int = 3
    #: Number of graduated non-convexity stages between the warm-up and the
    #: final kernel scale.  Dropping straight to the final scale can reject a
    #: correct loop closure along with the false one, because a single bad edge
    #: distorts the whole graph and inflates every residual.  Annealing the
    #: kernel scale downwards removes the worst edge first, lets the graph relax,
    #: and only then tightens further.
    anneal_stages: int = 6
    #: LM iterations per annealing stage.
    anneal_iterations: int = 3
    cost_tolerance: float = 1e-9
    step_tolerance: float = 1e-10


@dataclass
class PoseGraphResult:
    """Optimised poses plus convergence diagnostics."""

    poses: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    converged: bool
    edge_errors: np.ndarray
    edge_weights: np.ndarray
    history: list[float] = field(default_factory=list)


def edge_error(pose_i: np.ndarray, pose_j: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """``log(Z^-1 T_i^-1 T_j)`` as a 6-vector ``[rho, phi]``."""
    return se3_log(se3_inverse(measurement) @ se3_inverse(pose_i) @ pose_j)


def edge_jacobians(
    pose_i: np.ndarray, pose_j: np.ndarray, measurement: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobians of :func:`edge_error` w.r.t. right perturbations.

    Returns ``(error, J_i, J_j)`` with both Jacobians 6x6.
    """
    error = edge_error(pose_i, pose_j, measurement)
    J_inv = se3_left_jacobian_inverse(error)
    J_i = -J_inv @ se3_adjoint(se3_inverse(measurement))
    J_j = se3_left_jacobian_inverse(-error)
    return error, J_i, J_j


def robust_weight(
    chi2: float, kernel: str, huber_delta: float, dcs_phi: float
) -> float:
    """Scalar IRLS weight for a squared Mahalanobis error.

    ``huber``
        ``1`` while ``sqrt(chi2) <= delta``, then ``delta / sqrt(chi2)``.
    ``dcs``
        Dynamic Covariance Scaling: ``s = min(1, 2 phi / (phi + chi2))``,
        applied as ``s^2``.  Unlike Huber it drives the weight towards zero
        quadratically, so an edge that is badly wrong is effectively switched
        off rather than merely down-weighted.
    """
    if kernel == "none":
        return 1.0
    chi2 = max(float(chi2), 0.0)
    if kernel == "huber":
        norm = np.sqrt(chi2)
        return 1.0 if norm <= huber_delta else float(huber_delta / max(norm, 1e-12))
    if kernel == "dcs":
        s = min(1.0, 2.0 * dcs_phi / max(dcs_phi + chi2, 1e-12))
        return float(s * s)
    raise ValueError(f"unknown kernel {kernel!r}")


def _solve_linear(H: np.ndarray, g: np.ndarray, use_sparse: bool) -> np.ndarray:
    """Solve ``H x = g``, sparse if SciPy is present and the system is big."""
    if use_sparse and _HAVE_SCIPY and H.shape[0] > 60:
        sparse_H = sp.csc_matrix(H)
        return spla.spsolve(sparse_H, g)
    try:
        L = np.linalg.cholesky(H)
        return np.linalg.solve(L.T, np.linalg.solve(L, g))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, g, rcond=None)[0]


def _build_system(
    poses: np.ndarray,
    edges: list[PoseGraphEdge],
    free_slot: dict[int, int],
    config: PoseGraphConfig,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    n = len(free_slot) * 6
    H = np.zeros((n, n))
    b = np.zeros(n)
    cost = 0.0
    errors = np.zeros(len(edges))
    weights = np.ones(len(edges))

    for k, edge in enumerate(edges):
        e, J_i, J_j = edge_jacobians(poses[edge.i], poses[edge.j], edge.measurement)
        omega = edge.information
        chi2 = float(e @ omega @ e)
        errors[k] = np.sqrt(max(chi2, 0.0))

        if config.kernel_on_loops_only and not edge.is_loop:
            w = 1.0
        else:
            w = robust_weight(chi2, config.kernel, config.huber_delta, config.dcs_phi)
        weights[k] = w
        cost += w * chi2

        W = w * omega
        si = free_slot.get(edge.i)
        sj = free_slot.get(edge.j)
        if si is not None:
            H[si:si + 6, si:si + 6] += J_i.T @ W @ J_i
            b[si:si + 6] += J_i.T @ W @ e
        if sj is not None:
            H[sj:sj + 6, sj:sj + 6] += J_j.T @ W @ J_j
            b[sj:sj + 6] += J_j.T @ W @ e
        if si is not None and sj is not None:
            block = J_i.T @ W @ J_j
            H[si:si + 6, sj:sj + 6] += block
            H[sj:sj + 6, si:si + 6] += block.T

    return H, b, cost, errors, weights


def _run_lm(
    poses: np.ndarray,
    edges: list[PoseGraphEdge],
    free_slot: dict[int, int],
    config: PoseGraphConfig,
    max_iterations: int,
) -> tuple[np.ndarray, float, float, int, bool, np.ndarray, np.ndarray, list[float]]:
    """Levenberg-Marquardt loop for a fixed kernel setting."""
    H, b, cost, errors, weights = _build_system(poses, edges, free_slot, config)
    initial_cost = cost
    history = [cost]
    lam = config.initial_lambda
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        scale = max(np.trace(H) / H.shape[0], 1e-12)
        damped = H + np.eye(H.shape[0]) * (lam * scale + 1e-12)
        delta = _solve_linear(damped, -b, use_sparse=True)
        if not np.all(np.isfinite(delta)):
            lam *= config.lambda_up
            continue
        if float(np.linalg.norm(delta)) < config.step_tolerance:
            converged = True
            break

        candidate = np.array(poses, dtype=float)
        for node, slot in free_slot.items():
            candidate[node] = candidate[node] @ se3_exp(delta[slot:slot + 6])
            candidate[node, :3, :3] = normalise_rotation(candidate[node, :3, :3])

        H_new, b_new, cost_new, errors_new, weights_new = _build_system(
            candidate, edges, free_slot, config
        )
        if cost_new < cost:
            relative = (cost - cost_new) / max(cost, 1e-16)
            poses, H, b, errors, weights = candidate, H_new, b_new, errors_new, weights_new
            cost = cost_new
            lam = max(lam * config.lambda_down, 1e-12)
            history.append(cost)
            if relative < config.cost_tolerance:
                converged = True
                break
        else:
            lam *= config.lambda_up
            if lam > 1e10:
                break

    return poses, initial_cost, cost, iteration, converged, errors, weights, history


def optimise_pose_graph(
    poses: np.ndarray,
    edges: list[PoseGraphEdge],
    config: PoseGraphConfig | None = None,
    fixed: tuple[int, ...] = (0,),
) -> PoseGraphResult:
    """Levenberg-Marquardt pose-graph optimisation, in two phases.

    Phase one runs with the robust kernel disabled so the graph can absorb the
    drift that a newly added loop closure exposes.  Phase two switches the
    kernel on and continues; by then a correct loop closure has a small residual
    and a false one does not, so the kernel down-weights the right edge.

    Parameters
    ----------
    poses:
        ``(N, 4, 4)`` initial ``T_wc`` estimates, typically straight from
        odometry and therefore drifted.
    edges:
        Sequential odometry constraints plus any accepted loop closures.
    fixed:
        Node indices held constant.  At least one is required -- otherwise the
        whole graph can translate and rotate freely and ``H`` is singular.
    """
    config = config or PoseGraphConfig()
    poses = np.array(poses, dtype=float).reshape(-1, 4, 4)
    n_nodes = poses.shape[0]
    fixed_set = {int(f) for f in fixed}
    free = [i for i in range(n_nodes) if i not in fixed_set]
    free_slot = {node: k * 6 for k, node in enumerate(free)}

    if not edges or not free:
        return PoseGraphResult(
            poses, 0.0, 0.0, 0, True, np.zeros(len(edges)), np.ones(len(edges))
        )

    history: list[float] = []
    total_iterations = 0
    initial_cost: float | None = None
    errors = np.zeros(len(edges))
    weights = np.ones(len(edges))
    converged = False

    def with_scale(scale: float) -> PoseGraphConfig:
        """Copy of the config with the kernel widened to ``scale`` (a chi2)."""
        fields = dict(config.__dict__)
        fields["dcs_phi"] = float(scale)
        fields["huber_delta"] = float(np.sqrt(max(scale, 1e-12)))
        return PoseGraphConfig(**fields)

    plain = PoseGraphConfig(**{**config.__dict__, "kernel": "none"})
    warmup = max(int(config.warmup_iterations), 0)
    if warmup and config.kernel != "none":
        poses, initial_cost, _, iters, _, errors, weights, hist = _run_lm(
            poses, edges, free_slot, plain, warmup
        )
        history.extend(hist)
        total_iterations += iters

        # Graduated non-convexity: start wide enough that nothing is rejected,
        # then tighten geometrically towards the configured scale.
        kernelled = [
            k for k, e in enumerate(edges)
            if e.is_loop or not config.kernel_on_loops_only
        ]
        chi2_max = float(np.max(errors[kernelled] ** 2)) if kernelled else 0.0
        final_scale = float(config.dcs_phi if config.kernel == "dcs"
                            else config.huber_delta ** 2)
        start_scale = max(chi2_max, final_scale)
        stages = max(int(config.anneal_stages), 0)
        if stages and start_scale > final_scale * 1.001:
            schedule = np.geomspace(start_scale, final_scale, stages + 1)[1:]
            for scale in schedule:
                poses, _, _, iters, _, errors, weights, hist = _run_lm(
                    poses, edges, free_slot, with_scale(scale),
                    max(int(config.anneal_iterations), 1),
                )
                history.extend(hist)
                total_iterations += iters

    # The annealing stages above are preparation; the final solve gets the full
    # iteration budget, because after a bad edge has been switched off the graph
    # still has to unwind the deformation that edge caused.
    poses, robust_initial, final_cost, iters, converged, errors, weights, hist = _run_lm(
        poses, edges, free_slot, config, config.max_iterations
    )
    history.extend(hist)
    total_iterations += iters
    if initial_cost is None:
        initial_cost = robust_initial

    return PoseGraphResult(
        poses=poses,
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        iterations=total_iterations,
        converged=converged,
        edge_errors=errors,
        edge_weights=weights,
        history=history,
    )
