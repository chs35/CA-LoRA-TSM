# Results manifest

sha256 of every file in `results/` and `figures/` (this file excluded). Regenerate with
`python scripts/make_results.py` (and `scripts/router_gate_vs_t.py` for `router_gate/`).

Provenance: `evals/<arm>.json` are verbatim copies of `runs/<run>/<arm>/eval.json`;
`reference_10k/<arm>.json` are verbatim copies of Improving-CA-LoRA
`results/sdxl_cityscapes/<arm>/eval.json` (identical to its own MANIFEST);
`diagnosis_jaccard.json` is extracted from `runs/sdxl_cityscapes/diagnosis/diagnosis.json`
(source sha256 inside the file). Checkpoints, measurements and generated images stay in the
gitignored `runs/` and are not part of this repository.

| File | Bytes | sha256 |
|---|---:|---|
| `figures/jaccard_matrix.png` | 294,570 | `0c88d4308f2df613adb9484fa8923df48ba6ffce6f67f2c090c44fb44ebd2507` |
| `figures/router_gate_vs_t.png` | 144,063 | `6ce92830b92225315003312c60b16ed13ee2a7d6cb838190a81ed78e585f3fb8` |
| `figures/tradeoff_scatter.png` | 137,019 | `ad7dcfb141018dcb40d9b7237ba420625fafa5f90578dab23c64128757231e69` |
| `results/diagnosis_jaccard.json` | 14,052 | `a74d46f5f36ced2f55c672a4d3c1de2662695014119a6def7e75ef58bee8251b` |
| `results/evals/c1.json` | 4,068 | `cee1ff7df07481f520a4c6a58bebc184fe2705eef235ba4aee8234e9326e485c` |
| `results/evals/c2.json` | 9,390 | `eda4ec422180d7a7a232c4f29bd89ae559fad47305c4535f4f69685c4775ef22` |
| `results/evals/c2n2.json` | 4,145 | `c0d216f90ade13eb952191e51b1258e394e974a5845c147a5cc6f8906b01796d` |
| `results/evals/no1.json` | 2,765 | `644136b6ac4a38e5b318fac94714d4f4b26d83ae5f6a5bff0e65187f1abf75dd` |
| `results/evals/no2.json` | 4,116 | `1d5fd8c2bd8dfe939146810582048bb7322d6cddd52bf9c7de67113b95743ec9` |
| `results/evals/no3.json` | 9,354 | `c9cca871fd0bb214f1af95ea6f7a3e4b060954a9de1b5d156102b6be8b743be1` |
| `results/evals/no4.json` | 16,288 | `aad320d793166edcf7410dd4ce1f0b8a7fbd4d0de241165a8e0fb61924f8edca` |
| `results/evals/no5.json` | 17,690 | `abfaf15b11b654ff7e45aaeba9bf44f138bdfbc6589a0d0be8c57d355d4de295` |
| `results/evals/no6.json` | 18,591 | `a824ca3c44b171a4269bdfd212975b0224d29b78f953114e52e44a0ec22f9567` |
| `results/evals/no7.json` | 17,624 | `0e7b4d2782c297bc60c9a0bef76d4b02a6c7ca467281daaaebb25bc3c9101ad2` |
| `results/evals/no8.json` | 19,066 | `0dc32e5a3539be07456d285568db0250dd8b79d205d74c834956aaf504ff4596` |
| `results/evals/no9.json` | 18,994 | `b6b78fb88b52f478a09096a0c0b0f790e4aa1dee4074a88eaf185be9fc66fcd9` |
| `results/reference_10k/A_t81_style.json` | 2,091 | `e5aa79715a0529ad9469c6465b3310aa6716e16a13c7f77e2749ed90744d620c` |
| `results/reference_10k/B_top3_style.json` | 2,093 | `96cd1e00cf2cd296bd6a25b07bb36d214df931b904f2ab14de7437aa0581b8ac` |
| `results/reference_10k/C_t41_style.json` | 2,090 | `15fd1a27ff034310704511e72ac4be2440c314a93e041ec725dabcb05c2613b1` |
| `results/reference_10k/D_t1_style.json` | 2,089 | `3b950813a19a20188a34e202299ee025a2db247f3b849fd854428db4f2a7bb73` |
| `results/reference_10k/E_all.json` | 2,088 | `dc3e118543921fc4ae0e226e09a10952b14249ff60ccb4c916a6d30d70da9eb5` |
| `results/reference_10k/E_random.json` | 2,091 | `7936fb2b4284a1986aed44508dd1ebe8b0b6c3111fbaf23a7920e4731acdf74c` |
| `results/reference_10k/control_base.json` | 2,092 | `35c8967620d65ecd9ec50d1708db2e2354287242a27ec8c4689fd00f054a8691` |
| `results/router_gate/no3.json` | 198,465 | `a3107ec65a33ffc1f536a1933e7784ddca56506e4c1679d1f10b6a3b43558bea` |
| `results/summary.csv` | 3,303 | `199e42eef73ee9733a94545954520abae43679a228b5aaabe7ba3c05dc647913` |
| `results/summary.md` | 2,887 | `86c10946d81f96af65c0104212ab2443f68723b24adee86d6f094e4a6e76b4ed` |

Source eval.json in `runs/` (not committed):

| Source | sha256 |
|---|---|
| `runs/sdxl_cityscapes/no1/eval.json` | `644136b6ac4a38e5b318fac94714d4f4b26d83ae5f6a5bff0e65187f1abf75dd` |
| `runs/sdxl_cityscapes/no2/eval.json` | `1d5fd8c2bd8dfe939146810582048bb7322d6cddd52bf9c7de67113b95743ec9` |
| `runs/sdxl_cityscapes/no3/eval.json` | `c9cca871fd0bb214f1af95ea6f7a3e4b060954a9de1b5d156102b6be8b743be1` |
| `runs/sdxl_cityscapes/no4/eval.json` | `aad320d793166edcf7410dd4ce1f0b8a7fbd4d0de241165a8e0fb61924f8edca` |
| `runs/sdxl_cityscapes/no5/eval.json` | `abfaf15b11b654ff7e45aaeba9bf44f138bdfbc6589a0d0be8c57d355d4de295` |
| `runs/sdxl_cityscapes/no6/eval.json` | `a824ca3c44b171a4269bdfd212975b0224d29b78f953114e52e44a0ec22f9567` |
| `runs/sdxl_cityscapes/no7/eval.json` | `0e7b4d2782c297bc60c9a0bef76d4b02a6c7ca467281daaaebb25bc3c9101ad2` |
| `runs/sdxl_cityscapes/no8/eval.json` | `0dc32e5a3539be07456d285568db0250dd8b79d205d74c834956aaf504ff4596` |
| `runs/sdxl_cityscapes/no9/eval.json` | `b6b78fb88b52f478a09096a0c0b0f790e4aa1dee4074a88eaf185be9fc66fcd9` |
| `runs/sdxl_c1/c1/eval.json` | `cee1ff7df07481f520a4c6a58bebc184fe2705eef235ba4aee8234e9326e485c` |
| `runs/sdxl_cityscapes/c2n2/eval.json` | `c0d216f90ade13eb952191e51b1258e394e974a5845c147a5cc6f8906b01796d` |
| `runs/sdxl_cityscapes/c2/eval.json` | `eda4ec422180d7a7a232c4f29bd89ae559fad47305c4535f4f69685c4775ef22` |
