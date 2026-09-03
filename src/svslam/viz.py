"""Figures for the benchmark report.

Every plot here is produced from data the pipeline actually measured; nothing is
illustrative.  Matplotlib is imported lazily with the Agg backend so the package
still imports on a machine without a display, and so that tests that do not draw
anything do not pay for the import.

Colour is used consistently across every figure: one hue per role, never
recycled, and every series is also distinguished by line style or marker shape so
the figures survive being read in greyscale or by a colourblind reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "SURFACE",
    "INK",
    "INK_MUTED",
    "SERIES",
    "plot_trajectory",
    "plot_loop_comparison",
    "plot_kitti_errors",
    "plot_tracked_features",
    "plot_ba_sparsity",
    "plot_covisibility",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e2e1dd"
#: Categorical slots, in fixed order. Slot 1 ground truth, 2 estimate, 3 variant.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
BAD = "#e34948"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    """Recessive axes and grid; text in ink tokens, never a series colour."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def _optimise_png(path: Path) -> None:
    """Shrink a rendered PNG in place, losslessly where it can.

    These figures are committed, so they should not be megabytes.  Matplotlib
    writes 24-bit RGB; every figure here uses at most a few hundred distinct
    colours, so an adaptive 256-colour palette is visually identical and
    typically two to four times smaller.  If Pillow is unavailable the file is
    simply left as matplotlib wrote it.
    """
    try:  # pragma: no cover - depends on the optional Pillow dependency
        from PIL import Image
    except ImportError:  # pragma: no cover
        return
    try:
        with Image.open(path) as image:
            converted = image.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT)
            converted.save(path, format="PNG", optimize=True)
    except OSError:  # pragma: no cover - never fail a run over a figure
        return


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    _pyplot().close(fig)
    _optimise_png(path)
    return path


def _heading(fig, title: str, subtitle: str = "", top: float = 0.88) -> None:
    """Title and subtitle as figure text, above the axes.

    Placing them on the axes collides with the plot as soon as an equal-aspect
    trajectory makes the axes box tall, and the collision is invisible until you
    look at the rendered PNG.  Figure-level text with a reserved top margin
    cannot overlap the data no matter what shape the trajectory turns out to be.
    """
    fig.subplots_adjust(top=top)
    fig.text(0.0, 1.0, title, ha="left", va="top", fontsize=12,
             fontweight="bold", color=INK, transform=fig.transFigure)
    if subtitle:
        fig.text(0.0, 0.955, subtitle, ha="left", va="top", fontsize=9,
                 color=INK_MUTED, transform=fig.transFigure)


def _topdown(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KITTI camera axes are x right, y down, z forward: top-down is (x, z)."""
    p = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    return p[:, 0, 3], p[:, 2, 3]


def plot_trajectory(
    estimated: np.ndarray,
    ground_truth: np.ndarray | None,
    path: str | Path,
    title: str = "Estimated trajectory vs KITTI ground truth",
    subtitle: str = "",
) -> Path:
    """Top-down trajectory with the start marked."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 5.4), facecolor=SURFACE)
    _style(ax)

    if ground_truth is not None and len(ground_truth):
        gx, gz = _topdown(ground_truth)
        ax.plot(gx, gz, color=SERIES[0], linewidth=2.0, linestyle="--",
                label="Ground truth (OXTS)", zorder=2)
    ex, ez = _topdown(estimated)
    ax.plot(ex, ez, color=SERIES[1], linewidth=2.0, label="Stereo SLAM estimate", zorder=3)
    ax.plot([ex[0]], [ez[0]], marker="o", markersize=9, color=SURFACE,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=4)
    ax.annotate("start", (ex[0], ez[0]), textcoords="offset points", xytext=(10, 6),
                color=INK, fontsize=9, fontweight="bold")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID,
                       framealpha=0.9, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK)
    _heading(fig, title, subtitle, top=0.86)
    return _save(fig, path)


def plot_loop_comparison(
    before: np.ndarray,
    after: np.ndarray,
    ground_truth: np.ndarray,
    path: str | Path,
    ate_before: float | None = None,
    ate_after: float | None = None,
) -> Path:
    """Two panels: the same trajectory before and after loop closure.

    Two panels rather than three overlaid lines, because the point of the figure
    is the gap between the estimate and the ground truth closing, and that gap is
    much easier to see when only two lines share an axis.
    """
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), facecolor=SURFACE,
                             sharex=True, sharey=True)
    gx, gz = _topdown(ground_truth)
    panels = (
        (axes[0], before, "Before loop closure", ate_before, SERIES[1]),
        (axes[1], after, "After loop closure", ate_after, SERIES[2]),
    )
    headings: list[str] = []
    for ax, poses, label, ate, colour in panels:
        _style(ax)
        ax.plot(gx, gz, color=SERIES[0], linewidth=1.8, linestyle="--", label="Ground truth")
        ex, ez = _topdown(poses)
        ax.plot(ex, ez, color=colour, linewidth=1.8, label="Estimate")
        ax.plot([ex[0]], [ez[0]], marker="o", markersize=8, color=SURFACE,
                markeredgecolor=INK, markeredgewidth=1.5)
        heading = label if ate is None else f"{label}  -  ATE {ate:.2f} m"
        headings.append(heading)
        ax.set_xlabel("x (m)")
        # The two panels share axes so the drift is comparable by eye, which
        # rules out the datalim aspect adjustment used elsewhere.
        ax.set_aspect("equal", adjustable="box")
        legend = ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID,
                           framealpha=0.9, fontsize=9, loc="best")
        for text in legend.get_texts():
            text.set_color(INK)
    axes[0].set_ylabel("z (m)")
    fig.subplots_adjust(top=0.84)
    for ax, heading in zip(axes, headings):
        box = ax.get_position()
        fig.text(box.x0, 0.93, heading, ha="left", va="bottom", fontsize=10,
                 fontweight="bold", color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.0, "Trajectory before and after loop closure", ha="left",
             va="top", fontsize=12, fontweight="bold", color=INK,
             transform=fig.transFigure)
    return _save(fig, path)


def plot_kitti_errors(report, path: str | Path) -> Path:
    """The official KITTI plot: error against sub-sequence length."""
    plt = _pyplot()
    lengths = sorted(report.per_length)
    trans = [report.per_length[k]["translation_percent"] for k in lengths]
    rot = [report.per_length[k]["rotation_deg_per_m"] for k in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), facecolor=SURFACE)
    for ax, values, ylabel, heading in (
        (axes[0], trans, "translation error (%)", "Translation error vs sub-sequence length"),
        (axes[1], rot, "rotation error (deg/m)", "Rotation error vs sub-sequence length"),
    ):
        _style(ax)
        ax.plot(lengths, values, color=SERIES[0], linewidth=2.0, marker="o", markersize=6,
                markerfacecolor=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_xlabel("sub-sequence length (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(heading, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylim(bottom=0.0)
        # Direct-label the endpoints only; a number on every point is noise.
        for k in (0, len(lengths) - 1):
            ax.annotate(f"{values[k]:.3g}", (lengths[k], values[k]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=INK)
    return _save(fig, path)


def plot_tracked_features(
    image: np.ndarray,
    points: np.ndarray,
    inliers: np.ndarray,
    path: str | Path,
    title: str = "Tracked features: PnP inliers and rejected matches",
    subtitle: str = "",
    previous_points: np.ndarray | None = None,
) -> Path:
    """Draw a real frame with its inlier and outlier correspondences.

    ``previous_points``, when given, is where each feature was in the earlier
    frame; a short line is drawn from there to its current position, so the
    figure shows the optical flow the pose was solved from and not merely a
    scatter of dots.
    """
    plt = _pyplot()
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    inliers = np.asarray(inliers, dtype=bool).reshape(-1)
    height, width = image.shape[:2]

    head_inches = 0.75
    fig_height = 11.0 * height / width + head_inches
    fig, ax = plt.subplots(figsize=(11.0, fig_height), facecolor=SURFACE)
    # The axes is given the whole figure below the heading, and the image is
    # drawn with aspect="auto" so it fills that box exactly.  The box already
    # has the image's aspect ratio, so nothing is stretched, and there is no
    # matplotlib-inserted margin left over to look like a mistake.
    ax.set_position([0.0, 0.0, 1.0, 1.0 - head_inches / fig_height])
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, aspect="auto")

    if previous_points is not None:
        previous_points = np.asarray(previous_points, dtype=float).reshape(-1, 2)
        for mask, colour in ((inliers, SERIES[2]), (~inliers, BAD)):
            if not np.any(mask):
                continue
            segments = np.stack([previous_points[mask], points[mask]], axis=1)
            for segment in segments:
                ax.plot(segment[:, 0], segment[:, 1], color=colour, linewidth=0.9,
                        alpha=0.9, zorder=2)

    ax.scatter(points[inliers, 0], points[inliers, 1], s=26, facecolors="none",
               edgecolors=SERIES[2], linewidths=1.3, zorder=3,
               label=f"RANSAC inliers ({int(inliers.sum())})")
    ax.scatter(points[~inliers, 0], points[~inliers, 1], s=40, marker="x",
               color=BAD, linewidths=1.5, zorder=4,
               label=f"rejected ({int((~inliers).sum())})")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    legend = ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID,
                       framealpha=0.9, fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.text(0.0, 1.0, title, ha="left", va="top", fontsize=12,
             fontweight="bold", color=INK, transform=fig.transFigure)
    if subtitle:
        fig.text(0.0, 1.0 - 0.30 / fig_height, subtitle, ha="left", va="top",
                 fontsize=9, color=INK_MUTED, transform=fig.transFigure)
    return _save(fig, path)


def plot_ba_sparsity(problem, path: str | Path) -> Path:
    """Sparsity of the bundle-adjustment normal equations.

    The left block is the camera-camera part, the right the camera-landmark
    coupling ``W``.  The point of the figure is how little of ``W`` is filled --
    that emptiness is exactly what the Schur complement exploits.
    """
    plt = _pyplot()
    n_cams = problem.n_cameras
    n_pts = problem.n_points
    dense = np.zeros((n_cams * 6, n_cams * 6 + n_pts * 3), dtype=np.uint8)
    for i in range(n_cams):
        dense[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = 1
    ci = np.asarray(problem.camera_index)
    pi = np.asarray(problem.point_index)
    for i, j in zip(ci, pi):
        dense[i * 6:(i + 1) * 6, n_cams * 6 + j * 3: n_cams * 6 + (j + 1) * 3] = 1

    fill = float(dense.mean())
    fig, ax = plt.subplots(figsize=(10.0, 2.6), facecolor=SURFACE)
    ax.imshow(dense, aspect="auto", cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
    ax.axvline(n_cams * 6, color=BAD, linewidth=1.2)
    ax.set_title(
        f"Bundle-adjustment Jacobian structure  -  {n_cams} keyframes, "
        f"{n_pts} landmarks, {problem.n_observations} observations, "
        f"{100 * fill:.2f}% filled",
        fontsize=10, fontweight="bold", loc="left", color=INK,
    )
    ax.set_xlabel("camera block  |  landmark blocks", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("residual rows", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _save(fig, path)


def plot_covisibility(
    slam_map,
    path: str | Path,
    min_shared: int = 15,
    loop_closures: list | None = None,
) -> Path:
    """The keyframe graph, in space and in keyframe index.

    The left panel is the graph laid over the trajectory.  The right panel plots
    each edge as a point in (keyframe, keyframe) index space, which is where a
    loop closure is actually visible: the odometry backbone is the diagonal, and
    a loop closure is a point far off it, linking a keyframe near the end of the
    sequence to one near the beginning.  In the spatial panel those same edges
    are only a few metres long -- the vehicle really has come back to where it
    started -- so they are almost invisible, which is exactly why the second
    panel is here.
    """
    plt = _pyplot()
    graph = slam_map.covisibility(min_shared)
    ids = slam_map.keyframe_ids()
    centres = {kf_id: slam_map.keyframes[kf_id].centre for kf_id in ids}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), facecolor=SURFACE)
    left, right = axes
    _style(left)
    _style(right)

    drawn: set[tuple[int, int]] = set()
    for a, neighbours in graph.items():
        for b in neighbours:
            key = (min(a, b), max(a, b))
            if key in drawn:
                continue
            drawn.add(key)
            pa, pb = centres[a], centres[b]
            left.plot([pa[0], pb[0]], [pa[2], pb[2]], color=GRID, linewidth=0.9, zorder=1)
    n_edges = len(drawn)

    pairs = [(min(a, b), max(a, b)) for a, b in drawn]
    n_loops = 0
    loop_pairs: list[tuple[int, int]] = []
    if loop_closures:
        for closure in loop_closures:
            a = centres.get(closure.candidate_id)
            b = centres.get(closure.query_id)
            if a is None or b is None:
                continue
            n_loops += 1
            loop_pairs.append(
                (min(closure.candidate_id, closure.query_id),
                 max(closure.candidate_id, closure.query_id))
            )
            left.plot([a[0], b[0]], [a[2], b[2]], color=SERIES[1], linewidth=2.0,
                      zorder=4, solid_capstyle="round")

    xs = [centres[i][0] for i in ids]
    zs = [centres[i][2] for i in ids]
    left.scatter(xs, zs, s=9, color=SERIES[0], zorder=3, label=f"keyframes ({len(ids)})")
    left.plot([], [], color=GRID, linewidth=1.2, label=f"covisibility edge ({n_edges})")
    if n_loops:
        left.plot([], [], color=SERIES[1], linewidth=2.0,
                  label=f"accepted loop closure ({n_loops})")
    left.set_xlabel("x (m)")
    left.set_ylabel("z (m)")
    left.set_aspect("equal", adjustable="datalim")
    legend = left.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID,
                         framealpha=0.9, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK)

    if pairs:
        right.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=4,
                      color=SERIES[0], zorder=2, label="covisibility edge")
    if loop_pairs:
        right.scatter([p[0] for p in loop_pairs], [p[1] for p in loop_pairs], s=48,
                      marker="s", facecolors="none", edgecolors=SERIES[1],
                      linewidths=1.6, zorder=3, label="accepted loop closure")
    right.set_xlabel("keyframe index")
    right.set_ylabel("keyframe index")
    # Lower right: a loop closure links a late keyframe to an early one, which
    # puts it in the upper-left corner of this panel -- exactly where a legend
    # would hide it.
    legend = right.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID,
                          framealpha=0.9, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)

    _heading(
        fig, "Keyframe graph: in space, and in keyframe index",
        f"{len(ids)} keyframes, {n_edges} covisibility edges "
        f"(>= {min_shared} shared landmarks)"
        + (f", {n_loops} accepted loop closures" if n_loops else ""),
        top=0.84,
    )
    return _save(fig, path)
