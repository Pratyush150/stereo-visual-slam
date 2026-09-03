# The KITTI odometry metrics

## Why not final position error

A trajectory that drifts steadily and one that is perfect for 900 frames and
then jumps can end in the same place. Final position error cannot tell them
apart, and it scales with sequence length, so numbers from different sequences
are not comparable.

The KITTI benchmark measures error *per unit of distance travelled*, over many
sub-sequences of many lengths. That is what makes "1.1% translation error"
mean the same thing on a 700 m residential loop and a 4 km motorway run.

## The definition

For every length in `{100, 200, ..., 800}` metres and every start frame:

1. Walk forward from the start frame until the **ground-truth** path length
   first reaches the target length. That is the end frame. If the sequence ends
   first, the sub-sequence is **skipped**, not truncated.
2. Compute the relative pose error

   ```
   E = (gt_i^-1 gt_j)^-1 . (est_i^-1 est_j)
   ```

3. Translation error is `||trans(E)|| / length`; rotation error is
   `angle(rot(E)) / length`.

Average over every sub-sequence of every length. Report translation as a
percentage and rotation in degrees per metre.

## Details that change the answer

* **Start frames step by a fixed number of frames (10), not a fixed distance.**
  Stepping by distance over-weights the slow parts of a sequence, where the
  vehicle is manoeuvring and errors are largest.
* **Path length is measured on the ground truth**, not the estimate. Measuring
  it on the estimate would let a trajectory with a scale error quietly redefine
  what "100 m" means and flatter itself.
* **Skipping, not truncating.** Truncating the last few sub-sequences to
  whatever is left makes short segments dominate, and short segments have lower
  error, so the whole number comes out optimistic.

`svslam.evaluation.kitti_metrics` implements all three. The test
`test_a_one_percent_scale_error_scores_exactly_one_percent` pins the result to a
value that can be worked out on paper.

## The other two metrics

**ATE** (absolute trajectory error) is the RMS position difference after a
least-squares rigid alignment (Umeyama). It is a single number for "how far
from the truth is the whole map", and it is dominated by whatever the largest
single failure was.

For a **stereo** system the alignment should not solve for scale. Scale is
observed through the baseline, so leaving it at 1 makes any scale error show up
in the number where it belongs. `--align-scale` exists for the monocular case,
where scale is genuinely unobservable, and the fitted scale it reports is then
itself a diagnostic.

**RPE** (relative pose error) over a fixed frame gap measures local drift and
ignores global drift entirely. A trajectory with a large constant offset has
zero RPE. Reading ATE and RPE together separates "the map is bent" from "the
odometry is noisy".
