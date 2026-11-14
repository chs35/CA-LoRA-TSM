# No.6 outlier

No.6 = n3+n4+n5, core n4, router; the only m>1 configuration without n1.
CMMD 3.216; every other router configuration is at 1.345–1.430.
Weather CLIP 22.58 is in line with the other routers (21.86–22.61).

## What the images show

`results/figures/no6_samples.png`: rows are No.2, No.3 and No.6, same prompts and seeds.
No.6 is not black/flat, but scene structure breaks up in most frames: fragmented façades,
smeared objects, colour patches and overlay-like artefacts. The distortion is pervasive,
which is consistent with a CMMD near the base control (4.08).

## What it is not

- **Not a numeric fault.** No non-finite tensors; `router.bias_moved` is true; the g0 gates have no negative values (`bundles/n3+n4+n5-core-n4/bundle_stats.json` of the run, in the uncommitted `runs/` directory).
- **Not a different router operating point.** Latent-dependent gates on 12 held-out val frames
  (`router_gate/no6.json` vs `router_gate/no8.json`) are almost identical for the shared sets.
  - ‖ΔW‖-weighted gate, n3 / n5, at t = 50 / 375 / 750:
    - No.6: 0.22 / 0.49 / 1.43 and 0.38 / 0.52 / 1.45.
    - No.8: 0.24 / 0.58 / 1.53 and 0.45 / 0.58 / 1.54.
  - The core n4 weight is the same in both: 1.46 / 1.54 / 1.75.
  - No.8 adds n1 on top (0.32 / 0.41 / 0.64 at the same t) and still reaches CMMD 1.361.
- **Not stacking alone.** Both No.6 and No.8 sum the bit-identical shared experts of [125,999]
  (n4_i1=n5_i2, n3_i1=n4_i2=n5_i3, n3_i2=n4_i3=n5_i4), so at high t the same ΔW enters about 2.5× in No.6
  and 2.6× in No.8 (1 for the core plus the module-mean n3 and n5 gates at t = 999: 0.77 + 0.77 and 0.80 + 0.81).

## Working hypothesis (untested)

The distinguishing factor is the absence of n1, the t=81 expert active on all of [0,999].
Without it, the stacked high-t experts act alone, and the result degrades.
Testing this needs GPU generation (for example No.6 with the shared copies de-duplicated, or with gates capped). It is not run.
