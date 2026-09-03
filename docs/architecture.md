# Architecture

## Data flow

```
KITTI raw drive or odometry sequence
  |
  |  svslam.dataset.kitti
  |    calibration parsing (P0..P3, R_rect, Tr_velo_cam, Tr_imu_velo)
  |    stereo frame iteration
  |    OXTS -> local metric ground truth
  v
per frame:
  left, right (rectified greyscale)
  |
  |  svslam.frontend.features           ORB + spatial bucketing
  v  (u, v, descriptor) x ~1500
  |
  |  svslam.frontend.stereo             SAD along epipolar rows,
  v                                     LR consistency, depth-uncertainty gate
  (u, v, disparity) -> (X, Y, Z) in the camera frame, ~650 points
  |
  |  svslam.frontend.odometry           match to reference keyframe,
  v                                     RANSAC PnP, motion-only Gauss-Newton
  T_cw for this frame
  |
  +-- not a keyframe: store the pose relative to the reference keyframe, done
  |
  +-- keyframe:
        svslam.map                      link landmarks, triangulate new ones
        svslam.backend.ba               windowed LM + Schur complement
        svslam.loop.detector            bag-of-words query -> RANSAC PnP check
        svslam.backend.posegraph        SE(3) optimisation with a robust kernel
        svslam.map                      propagate the correction to landmarks
```

## Frames and conventions

| symbol | meaning |
|---|---|
| `T_cw` | world point -> camera point. The optimisation variable everywhere. |
| `T_wc` | camera pose in the world. `se3_inverse(T_cw)`. What gets plotted. |
| `xi = [rho, phi]` | tangent vector: translation first, rotation second. |
| increment | left perturbation, `T_cw <- exp(xi) T_cw`, in bundle adjustment and PnP refinement. |
| increment | right perturbation, `T_wc <- T_wc exp(xi)`, in the pose graph. |

The world frame is the first keyframe's camera frame, which is the KITTI
odometry convention: the benchmark's poses are those of the left camera
expressed in the first left camera frame. Camera axes are x right, y down,
z forward, so a top-down plot is the x-z plane.

The two perturbation conventions are deliberate rather than accidental. Bundle
adjustment differentiates a residual that is a function of the camera-frame
point, and the left perturbation makes that derivative
`[I | -[p_c]_x]` with no dependence on the pose itself. The pose graph
differentiates a relative-pose error, and the right perturbation is what makes
that come out as the adjoint expression in
`svslam.backend.posegraph.edge_jacobians`. Both are checked against central
differences in the tests.

## Why the reference keyframe, not the previous frame

Tracking chains error. Matching frame `n` to frame `n-1` and composing gives a
trajectory whose error grows with the number of compositions, at frame rate.
Matching instead to the most recent keyframe means the error grows with the
number of *keyframes*, which is three to four times fewer here, and the wider
baseline conditions the PnP better.

The price is that a keyframe eventually stops overlapping the current view.
That is what the keyframe policy in `svslam.frontend.odometry.KeyframePolicy`
exists to detect, and why it triggers on rotation more eagerly than on
translation.

## Every frame's pose is stored relative to its keyframe

`svslam.pipeline` records, for each frame, the identity of its reference
keyframe and the relative transform to it. The reported trajectory is rebuilt
from those at the end. When a loop closure moves the keyframes, the
non-keyframe poses move with them.

Storing absolute per-frame poses instead produces a trajectory where the
keyframes jump into place after a loop closure and every frame between them
stays where it was. It looks almost right, and it is wrong.

## Bundle adjustment structure

The normal equations are an arrowhead matrix:

```
| U   W | | dc |     | b_c |          U : (C, 6, 6) block diagonal
|       | |    |  = -|     |          V : (P, 3, 3) block diagonal
| W^T V | | dp |     | b_p |          W : sparse, one 6x3 block per observation
```

`svslam.backend.ba.solve_schur` marginalises the landmarks:

```
S = U - W V^-1 W^T          6C x 6C, dense, small
S dc = -(b_c - W V^-1 b_p)
dp_j = V_j^-1 (-b_p_j - sum_i W_ij^T dc_i)
```

For a 7-keyframe window with 3000 landmarks, the direct system is
21042 x 21042 and the reduced one is 42 x 42. `solve_dense` implements the
direct solve anyway, purely so the tests can assert the two agree.

## Loop closure, in three gates

1. **Appearance.** Bag-of-words similarity against keyframes outside a temporal
   exclusion window, thresholded relative to how well the query matches its own
   recent neighbours.
2. **Temporal consistency.** The same region must be proposed by consecutive
   queries.
3. **Geometry.** RANSAC PnP between the candidate keyframe's landmarks, in that
   keyframe's own camera frame, and the query frame's features. Because no
   world frame is involved, a loop can still be verified after the global map
   has drifted by metres.

Only after all three does an edge enter the pose graph, and even then the
robust kernel can switch it off. The counters for each gate are reported, so
"the detector rejected nothing" is visible rather than assumed.

## Landmark association is gated, not assumed

When a keyframe is inserted, a landmark carried over from the previous keyframe
is linked only if it projects into the new keyframe within the same threshold
RANSAC used for its inliers. Linking every descriptor match instead is the most
damaging single mistake available in this pipeline, and it is silent: the poses
stay finite, the trajectory looks plausible, and the local bundle adjustment
starts from a forty-pixel reprojection RMSE and drags keyframes metres out of
place trying to satisfy observations that contradict each other. See
`docs/tuning.md` for what that cost and how it was found.
