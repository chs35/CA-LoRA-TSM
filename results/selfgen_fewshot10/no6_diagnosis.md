# No.6 outlier (self-generated x0 campaign)

No.6 = n3+n4+n5, core n4, router; the only m>1 configuration without n1.
CMMD 3.216 (dataset-x0 campaign: 2.068); every other arm improved by 0.24–0.64.
Weather CLIP 22.58 is in line with the other routers (22.2–22.6).

## What the images show

`figures/selfgen_fewshot10/no6_samples.png`: rows are new No.2, new No.3, new No.6 and old No.6, same prompts and seeds.
New No.6 is not black/flat, but scene structure breaks up in most frames: fragmented façades,
smeared objects, colour patches and overlay-like artefacts. The old run had a few texture-collapse frames;
here the distortion is pervasive, which is consistent with a CMMD near the base control (4.08).

## What it is not

- **Not a numeric fault.** No non-finite tensors; `router.bias_moved` is true; the g0 gates have no negative values (`runs/.../bundles/n3+n4+n5-core-n4/bundle_stats.json`).
- **Not a different router operating point.** Latent-dependent gates on 12 held-out val frames
  (`router_gate/no6.json` vs `router_gate/no8.json`) are almost identical for the shared sets.
  - ‖ΔW‖-weighted gate, n3 / n5, at t = 50 / 375 / 750:
    - No.6: 0.22 / 0.49 / 1.43 and 0.38 / 0.52 / 1.45.
    - No.8: 0.24 / 0.58 / 1.53 and 0.45 / 0.58 / 1.55.
  - The core n4 weight is the same in both: 1.46 / 1.54 / 1.75.
  - No.8 adds n1 (0.32 → 0.71) on top and still reaches CMMD 1.361.
- **Not stacking alone.** Both No.6 and No.8 sum the bit-identical shared experts of [125,999]
  (n4_i1=n5_i2, n3_i1=n4_i2=n5_i3, n3_i2=n4_i3=n5_i4), so the same ΔW enters up to ~2.6× at high t in both.

## Working hypothesis (untested)

The distinguishing factor is the absence of n1, the t=81 expert active on all of [0,999].
Without it, the stacked high-t experts act alone, and the result degrades. The same ordering held in the dataset-x0
campaign (No.6 worst there as well), and it got worse with the new selections. The new selections place
the 208 heads in fewer modules (46–76 vs 58–96 per expert), so each module's ΔW is denser.
Testing this needs GPU generation (for example No.6 with the shared copies de-duplicated, or with gates capped). It is not run.
