# Deviations from the design

`docs/design.md` (formerly `docs/draft.md`, research note v2, 2026-09-11) was written before any
measurement. This file lists every point where the executed experiment differs from it, with a
one-line reason. Sources: `runs/sdxl_cityscapes/diagnosis/DECISION.md`, `configs/sdxl_cityscapes.yaml`,
`docs/GPU_RUNBOOK.md`, `docs/RESULTS.md` §8, and the run metadata under `runs/`.

## Evaluation

| Design | Actual | Reason |
|---|---|---|
| §3.4, §8: report CMMD **and mIoU** for every configuration ("always together"; mIoU detects leakage) | **mIoU not computed.** CMMD + per-condition CLIP-Score only (`miou: null` in every eval.json) | User decision 2026-09-14, same as Improving-CA-LoRA: no label generator / segmentation pipeline in this project. |
| §8 screening: one seed, CMMD + low-fraction mIoU on all of §5.1, then the top 2–3 fully | Every configuration fully evaluated once (900 images: 500 in-domain + 4 weather × 100) | Without mIoU there is no screening metric to rank by; a full generation run was ~4 h, affordable for all 12. |
| §8 noise floor: baseline with 3 segmentation-training seeds | Not run | No mIoU. |
| §6.5, §8 final comparison: 3 seeds for the top configurations and No.1 | **One generation seed (0) everywhere**; seed-to-seed variance unmeasured | User decision 2026-09-16 (one seed); seed replicates were left as an open item and never run. |
| §8: 50-step sampler is the reference for the n=5 62-step-interval argument | 25 steps, Euler, guidance 5.0, 1024 px (the Improving-CA-LoRA protocol) | Keeps every number comparable with the upstream reference rows and the reused base control. |
| — | Base SDXL control not regenerated: `control_base/eval.json` reused from Improving-CA-LoRA | Identical protocol, code and seed (checked 2026-09-16); saves ~4 h. |

## Configurations

| Design | Actual | Reason |
|---|---|---|
| §6: minimum set No.1, 2, 3 + C1; No.4, 5, 8, C2 if time; No.6, 7, 9 as follow-up | **No.1–9 and C1 all trained and evaluated** | User decision 2026-09-16: train a router for every m > 1 configuration. |
| §5.1 C2: high-interval expert disabled in "the final configuration" | Two C2 variants: **C2 (No.2)** = No.2 with n4_i3 [500,999] off (hard routing; that interval falls back to the base model) and **C2 (No.3)** = No.3 with n4_i3 off, router retrained (n1 covers [500,999]) | No single "final" configuration emerged (No.2 and C1 lead on different metrics); the two variants answer "what does the [500,999] expert do" with and without a fallback expert. |
| §5.3 robustness sweep (lowest-interval t ∈ {41, 81, 121, 201} for No.1 and the final configuration) | **Not run** | Out of GPU time (~30 h estimated); it is the only experiment that tests the original goal directly, see README Limitations. |
| §7: n=1 expert freshly trained for 4K steps | n1_i0 = the **step-4000 checkpoint of Improving-CA-LoRA's 10K t=81 style run**, imported | Same selection, data, constant LR and seed as a fresh 4K run; saves ~1.2 h. Not bit-identical: that run took CA-drift snapshots every 500 steps (merge/unmerge into fp32 weights), recorded in `experts/n1_i0/IMPORTED.json`. |
| §7: 13 LoRAs × 4K = 52K fostering steps | **8 unique experts** (7 trained + n1_i0 imported) + 4 for C1; the other 5 nominal experts are copies | Experts with the same interval, selection and seed would be identical runs (`scripts/share_expert.py`, `SHARED_FROM.json`). Report as "13 nominal, 8 trained", 32K unique steps (+16K for C1). |
| — | C1's n4_i0 was trained again although it equals n4_i0 | Kept as a reproducibility reference (max weight difference 0.04, `docs/RESULTS.md` §8). |
| §7: router = nn.Linear + nn.Embedding (batch unspecified) | Router FC(mean_hw z_t) + E(t), one **raw** gate per projection matrix and context set (no activation, init 1.0), 1000 steps, **batch 8**, lr 1e-3 | Per-matrix gates are needed because experts sit on different heads (TSM-style); batch 1 gave an L_Diff gradient dominated by the (image, t) draw, so the batch-1 No.3 run was stopped at step 450 (user decision 2026-09-16). |
| §2, §3.2: Improving-CA-LoRA 10K results only as a reference | As designed: shown in a separate "reference" block and as hollow points | — |

## Diagnosis and interval design (§4, §3.2, §3.3)

| Design | Actual | Reason |
|---|---|---|
| §4: 13 measurement points 1, 41, 61, 81, 125, 201, 250, 375, 500, 625, 750, 875, 999 | **20 points**: 1, 41, 61, 81, 121, 125, 201, 250, 301, 375, 401, 481, 500, 601, 701, 750, 801, 901, 981, 999. **625 and 875 not measured** | 13 points (1, 41, 81, 121, 201, 301, 401, 481, 601, 701, 801, 901, 981) were reused from the Improving-CA-LoRA sweep (identical protocol); only 61, 125, 250, 375, 500, 750, 999 were new (76 min each). The design itself made 625/875 conditional on an unclear block structure (`docs/GPU_RUNBOOK.md` stage 1); they were not added. |
| §4: seed repeats at t = 81 and 875 | **Not run.** Split-half Jaccard (two halves of the 256 images, same seed) is reported instead | No reason is recorded in the run logs or runbook; they were not scheduled. Split-half measures sampling stability, not seed stability. |
| §4: halve the samples if split-half J ≥ 0.9 at 81 and 999 | Full 256 images × 4 prompt augs at every t | Split-half J at 81 was 0.66 (999: 0.73), below the 0.9 bar. |
| §3.2: choose (A) one representative t per interval or (B) interval-aggregate CA after §4 | **(A) representative**: n1 [81], n3 [81, 375, 750], n4 [81, 201, 375, 750], n5 [41, 81, 201, 375, 750] | User decision 2026-09-15, by eye from the figures (no score); intervals containing 81 keep 81, other intervals take a mid-interval t. (B) is implemented (`interval_select.mode: aggregate`) but was not trained. |
| §3.3: final boundaries decided from the Jaccard block structure; halving is only the starting point | **Halving boundaries kept**: interval starts 62 (n5), 125 (n4, n5), 250, 500 (all m > 1) | The matrix shows a smooth band, not blocks; none of the visible breaks (1 \| 41, 500 \| 601, 801 \| 901) suggested a better cut, and intervals containing 81 were fixed (user, 2026-09-15). |
| §3.3: halving approximates a uniform split in log-SNR | Recorded log-SNR of the starts: 62 → 2.74, 125 → 1.84, 250 → 0.73, 500 → −0.96 (ends 7.07 / −5.36). The widths in log-SNR are 4.3, 0.9, 1.1, 1.7, 4.4, **not uniform** | Reported as measured; the boundaries are stored in log-SNR in every selection/expert file as designed, so they can be carried to another backbone. |
| §3.4: if high-t selections are unstable, use random-sliced or no LoRA on that interval | [500,999] kept **CA-sliced at t = 750** (split-half J 0.48) | Neutral choice; the risk was covered by C2 instead. |
| §1: no CA-drift analysis, no Improving-CA-LoRA score as a mechanism | As designed (`train.stability_every: 0`) | — |
| §3.3: no LoRA on the text encoders | As designed | — |

## Other

| Item | Note |
|---|---|
| Viewpoint axis | Only the imported n1_i0 viewpoint expert and selections exist (`runs/sdxl_cityscapes_viewpoint/`); no viewpoint configuration was assembled or evaluated. Everything reported uses style-axis CA. |
| Router gate analysis | Added after the fact (`scripts/router_gate_vs_t.py`, README "Router gate vs t"); not part of the design. Done for No.3 only. |
| File name | `docs/draft.md` was renamed to `docs/design.md` (2026-10-29); references in the repository were updated. Historical documents keep their wording otherwise. |
