| Config | Experts | Routing | Active / nominal heads | CMMD ↓ | In-domain CLIP | Weather CLIP ↑ | foggy | night | rainy | snowy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| No.1 | n1 (CA-LoRA, t=81), 4K | single | 2% / 2% | 1.133 | 22.57 | 23.07 ± 0.13 | 25.94 | 20.49 | 23.33 | 22.51 |
| No.2 | n4 | hard | 2% / 8% | 1.514 | 24.58 | **26.71 ± 0.11** | 27.34 | 25.03 | 27.35 | 27.13 |
| No.3 | n1+n4, core n4 | router | 4% / 10% | 1.457 | 22.71 | 23.41 ± 0.15 | 25.05 | 21.36 | 24.00 | 23.23 |
| No.4 | n1+n3+n4, core n4 | router | 6% / 16% | 1.501 | 22.99 | 23.27 ± 0.15 | 25.00 | 21.15 | 23.70 | 23.22 |
| No.5 | n1+n4+n5, core n4 | router | 6% / 20% | 1.476 | 22.72 | 23.10 ± 0.15 | 24.80 | 20.82 | 23.74 | 23.04 |
| No.6 | n3+n4+n5, core n4 | router | 6% / 24% | 1.832 | 24.29 | 25.67 ± 0.12 | 27.11 | 23.54 | 26.46 | 25.57 |
| No.7 | n1+n4+n5, core n5 | router | 6% / 20% | 1.464 | 22.74 | 23.15 ± 0.15 | 25.00 | 20.97 | 23.65 | 23.00 |
| No.8 | n1+n3+n4+n5, core n4 | router | 8% / 26% | 1.580 | 22.51 | 22.84 ± 0.15 | 24.75 | 20.68 | 23.13 | 22.81 |
| No.9 | n1+n3+n4+n5, core n5 | router | 8% / 26% | 1.551 | 22.78 | 23.04 ± 0.14 | 24.85 | 20.90 | 23.34 | 23.08 |
| C1 | n4 intervals, every selection at t=81 | hard | 2% / 8% | **0.966** | 22.33 | 21.59 ± 0.14 | 24.94 | 19.62 | 22.69 | 19.09 |
| C2 (No.2) | No.2 with n4_i3 [500,999] off | hard | 2% / 8% (0% on [500,999]) | 2.459 | 27.29 | 29.95 ± 0.09 | 31.62 | 28.31 | 30.18 | 29.70 |
| C2 (No.3) | No.3 with n4_i3 [500,999] off, router retrained | router | 4% / 10% (2% on [500,999]) | 1.129 | 22.88 | 23.60 ± 0.13 | 25.74 | 21.84 | 24.12 | 22.69 |

Reference only — Improving-CA-LoRA, single LoRA trained 10K steps; not a direct comparison (design §5.1):

| Config | Experts | Routing | Active / nominal heads | CMMD ↓ | In-domain CLIP | Weather CLIP ↑ | foggy | night | rainy | snowy |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base SDXL | no LoRA (0%) | — | 0% / 0% | 4.078 | 28.81 | 31.68 ± 0.08 | 32.86 | 30.26 | 32.03 | 31.56 |
| 10K t=81 | single LoRA, style CA t=81, 10K | single LoRA | 2% / 2% | 0.638 | 22.08 | 21.00 ± 0.14 | 23.79 | 20.45 | 20.75 | 19.02 |
| 10K t=41 | single LoRA, style CA t=41, 10K | single LoRA | 2% / 2% | 0.751 | 23.18 | 23.16 ± 0.13 | 24.73 | 23.21 | 22.37 | 22.36 |
| 10K t=1 | single LoRA, style CA t=1, 10K | single LoRA | 2% / 2% | 0.752 | 21.62 | 19.96 ± 0.12 | 23.15 | 18.08 | 20.07 | 18.55 |
| 10K multi-t | single LoRA, style CA top-3 aggregate, 10K | single LoRA | 2% / 2% | 0.711 | 22.64 | 20.69 ± 0.11 | 23.15 | 19.70 | 20.80 | 19.11 |
| 10K random 2% | single LoRA, random 2% heads, 10K | single LoRA | 2% / 2% | 0.681 | 22.99 | 23.76 ± 0.12 | 25.88 | 22.68 | 25.04 | 21.46 |
| 10K full attention | LoRA on all attention, 10K | single LoRA | 100% / 100% | 0.596 | 21.84 | 20.65 ± 0.12 | 24.31 | 18.87 | 21.53 | 17.87 |
