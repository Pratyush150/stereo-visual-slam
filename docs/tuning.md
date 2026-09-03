# Tuning notes

Everything here was measured on KITTI raw drive `2011_09_30_drive_0027`
(1106 frames, 694.7 m). The numbers are what changed on this machine, on this
sequence; treat them as directions, not constants.

## The three settings that matter most

### Depth-uncertainty rejection (`stereo.max_relative_depth_sigma`)

`sigma_Z / Z = sigma_d / d` exactly, so this setting *is* the minimum
disparity: `d_min = sigma_d / max_relative_sigma`. At the defaults
(`sigma_d = 0.25 px`, 5%) that is 5 px, and with KITTI's `fx = 707 px`,
`b = 0.537 m` it caps usable depth at about 76 m; the separate 60 m ceiling
tightens it to 6.3 px.

Raising the tolerance admits far points. They look like good matches, they have
low reprojection error, and they carry metres of depth error, so they pull PnP
around. Lowering it below about 3% starves the frontend of points near the
horizon and rotation estimates get noisier.

### Feature bucketing (`features.grid_rows` x `grid_cols`, `max_per_cell`)

Without bucketing, ORB's strongest responses cluster on whatever is most
textured, and the pose is well constrained in one direction only. With the
default 5 x 12 grid the measured normalised spread entropy across keyframes is
around 0.87; the tests demonstrate the same effect on a synthetic clumped set.

`max_per_cell` too low throws away real information in genuinely rich cells.
Too high and the cap never binds, which is the same as no bucketing.

### Keyframe rotation threshold (`keyframes.rotation_threshold`)

The default is 0.12 rad, deliberately much tighter than the 3.0 m translation
threshold. Rotation destroys overlap with the reference keyframe far faster
than driving straight does, and a stale reference during a turn is exactly
where tracking breaks.

## The two bugs that cost twelve percentage points

Worth writing down, because both were invisible in every diagnostic the pipeline
printed at the time.

### Landmark association without a geometric check

The first full-sequence run scored **13.3%** translation error. The second, after
adding a motion plausibility gate, scored **5.5%**. Both were wrong for the same
underlying reason, and the gate had only masked part of it.

When a keyframe was inserted, every descriptor match against the previous
keyframe was turned into a landmark observation. Descriptor matching with a
ratio test and a mutual-best cross-check is good, but it is not that good:
roughly half of those matches were wrong, and each wrong one became a permanent
observation with a reprojection error of tens of pixels.

The symptom, once we instrumented the local bundle adjustment, was unmistakable:
windows were starting at a **34 to 72 pixel** reprojection RMSE and finishing at
32 to 60. Bundle adjustment was not refining a good map, it was arbitrating
between contradictory observations, and it was resolving the contradiction by
moving the newest keyframe **two to four metres**. The per-frame pose estimates
were fine the whole time -- raw frame-to-frame steps matched ground truth to a
few centimetres -- and the damage was being done afterwards, by the optimiser that
was supposed to help.

The fix is three lines: project each candidate landmark into the new keyframe
with the pose PnP just estimated, and keep only the ones within the same
threshold RANSAC already used. Translation error over the first 700 frames went
from 5.5% to **1.59%**, and the pipeline got roughly twice as fast, because the
bundle adjustment windows stopped being full of garbage.

Excluding landmarks that only one keyframe in a window observes helps too. Such
a landmark has three stereo residuals and three unknowns of its own: it is
exactly determined by that one observation and constrains no camera pose at all.
Keeping them inflated the problem several-fold and made the reported RMSE
meaningless.

### A pose the solver was confident about and wrong

Separately, near frame 637 -- the vehicle nearly stationary in a turn -- PnP found
a small consensus set among a handful of matches and returned a pose implying a
12.5 m jump in one 100 ms frame, with a 30 degree rotation error. Per-100 m
windows either side of it were between 0.25 m and 1.7 m of error; that one window
was 15.2 m.

`svslam.frontend.odometry.is_plausible_motion` rejects that: more than 4 m or
0.35 rad between consecutive frames, fall back to the constant-velocity
prediction and force a keyframe so the local map is re-seeded. The gate is loose
on purpose -- 4 m per frame at 10 Hz is 144 km/h -- because its job is to catch the
impossible, not the merely surprising.

Honest postscript: with the association bug fixed, the gate **fires zero times**
on this drive. It is still in the pipeline, because the failure it guards
against is real and the cost of the guard is a norm and an arccos per frame, but
this sequence no longer demonstrates it. The ablation table in the README shows
it making no difference here, and that is what the table is for.

## Loop closure thresholds

Measured on this drive, with a 1000-word vocabulary trained on a held-out slice:

* bag-of-words score for a true revisit: median 0.371
* score for an unrelated pair more than 40 m apart: median 0.327
* top-1 retrieval recall 0.59, top-3 0.71, top-5 0.76

That separation is thin, and it is thin for a real reason: the drive is a
residential street where one stretch looks much like another. So appearance is
used to *rank* candidates, not to decide, and `max_candidates` is 5. The
decision is the RANSAC PnP inlier count, which does not care what a place looks
like.

Dropping the vocabulary to 400 words collapses top-3 recall to 0.35: with only
400 words nearly every word appears in nearly every keyframe, the inverse
document frequency goes to zero, and the score stops discriminating.

## Speed

Roughly 0.30 s per frame at the defaults, dominated by sparse stereo. If you
need it faster, in order of effect per unit of accuracy lost:

1. `bundle_adjustment.window` 7 -> 5.
2. `stereo.max_disparity` 96 -> 64, if the scene has no close structure.
3. `features.max_features` 1500 -> 900.
4. `bundle_adjustment.max_iterations` 10 -> 6; it usually converges in 3 or 4.

Disabling bundle adjustment entirely (`--no-ba`) is much faster and noticeably
worse; the point of measuring both is that you can see by how much.
