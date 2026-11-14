# Self-generated x0 vs dataset x0 (10-image campaigns)

Same arms, steps, training subset, precision and evaluation protocol; the only difference is the x0 the CA
selection was measured on: the base model's own samples from c (`sdxl_cityscapes_selfgen_fewshot10`, 2026-11-08/09)
vs the first 256 Cityscapes train frames (`sdxl_cityscapes_fewshot10`, 2026-11-04/05). Sources:
`results/selfgen_fewshot10/summary.csv`, `results/fewshot10/summary.csv`. One generation seed; weather SEM ≈ 0.12.

| config | CMMD selfgen | CMMD dataset | Δ | weather selfgen | weather dataset | Δ | in-domain selfgen | in-domain dataset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No.1 | 1.336 | 1.650 | -0.314 | 20.00 | 23.43 | -3.42 | 20.48 | 22.62 |
| No.2 | 1.482 | 1.835 | -0.354 | 25.85 | 26.07 | -0.22 | 24.41 | 23.86 |
| C1 | 1.182 | 1.425 | -0.243 | 20.71 | 21.04 | -0.33 | 21.67 | 21.78 |
| No.3 | 1.362 | 1.644 | -0.281 | 22.29 | 23.28 | -0.99 | 21.22 | 21.45 |
| No.4 | 1.345 | 1.665 | -0.321 | 22.16 | 23.64 | -1.48 | 21.12 | 22.08 |
| No.5 | 1.349 | 1.614 | -0.266 | 22.41 | 23.50 | -1.09 | 21.33 | 21.89 |
| No.6 | 3.216 | 2.068 | +1.149 | 22.58 | 24.41 | -1.84 | 21.79 | 22.86 |
| No.7 | 1.348 | 1.690 | -0.342 | 22.61 | 23.56 | -0.95 | 21.69 | 22.14 |
| No.8 | 1.361 | 1.847 | -0.485 | 21.86 | 21.87 | -0.00 | 21.22 | 21.71 |
| No.9 | 1.430 | 1.752 | -0.322 | 22.27 | 21.93 | +0.34 | 21.60 | 21.71 |
| C2 (No.2) | 1.787 | 2.424 | -0.637 | 26.31 | 29.30 | -2.99 | 24.70 | 27.01 |
| C2 (No.3) | 1.327 | 1.565 | -0.238 | 23.51 | 24.43 | -0.91 | 21.91 | 23.46 |

## Selection overlap at the representative t (top-2 % style heads, 208 units)

| expert (t) | shared units | Jaccard |
|---|---:|---:|
| n5_i0 (t=41) | 53 | 0.146 |
| n4_i0 (t=81) | 54 | 0.149 |
| n4_i1 (t=201) | 34 | 0.089 |
| n4_i2 (t=375) | 39 | 0.103 |
| n4_i3 (t=750) | 49 | 0.134 |
