# Results manifest

sha256 of every file in `results/` and `figures/` (this file excluded). Regenerate with
`python scripts/make_results.py` (and `scripts/router_gate_vs_t.py` for `router_gate/`).

Provenance: `evals/<arm>.json` are verbatim copies of `runs/<run>/<arm>/eval.json`;
`reference_10k/control_base.json` is a verbatim copy of
`/workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes_fewshot10/control_base/eval.json`;
`diagnosis_jaccard.json` is extracted from `runs/sdxl_cityscapes/diagnosis/diagnosis.json`
(source sha256 inside the file). Checkpoints, measurements and generated images stay in the
gitignored `runs/` and are not part of this repository.

| File | Bytes | sha256 |
|---|---:|---|
| `figures/fewshot10/jaccard_matrix.png` | 294,570 | `0c88d4308f2df613adb9484fa8923df48ba6ffce6f67f2c090c44fb44ebd2507` |
| `figures/fewshot10/router_gate_vs_t.png` | 142,163 | `ef8bfdeabd8bb9c6fd43d0de3ddfdaa4c2f0220d472022d8da29c5e1d3018cf1` |
| `figures/fewshot10/tradeoff_scatter.png` | 110,090 | `9e41f0803c2d1f723c5004099f42911e6f41aafd59b89d937857c16444ef50ff` |
| `results/fewshot10/diagnosis_jaccard.json` | 14,052 | `a74d46f5f36ced2f55c672a4d3c1de2662695014119a6def7e75ef58bee8251b` |
| `results/fewshot10/evals/c1.json` | 4,126 | `7c10914b48850108c22ffed4c72d836d245d5004fd06489354a2edb3dfd346d2` |
| `results/fewshot10/evals/c2.json` | 9,455 | `6e13b7e93c0373f2db7d0fd06259477026c22e6733fb3afbb3f56eb39991d1b3` |
| `results/fewshot10/evals/c2n2.json` | 4,213 | `a3c9584f941bbf6c6e049381ab5cc4c55e54a79b14474c601f99e08d489f9ba2` |
| `results/fewshot10/evals/no1.json` | 2,795 | `6bafc5082d77eee0a28693e6cbe397d70dd7b51f4b1c354f4fb5a5d934013d1c` |
| `results/fewshot10/evals/no2.json` | 4,172 | `42e81bc211508336527a1746b69bd0ebf9f9af826f7cd7a31a3be00777deaed7` |
| `results/fewshot10/evals/no3.json` | 9,427 | `22f34b5bf5a3bac8c1f7b5f7d2bf28fb66e3b432805cdf3d4b59634612ff395a` |
| `results/fewshot10/evals/no4.json` | 16,386 | `4df071396bfbe699ba39418976bb05f442c24f1a40e5786010e902bc1d9b68d9` |
| `results/fewshot10/evals/no5.json` | 17,812 | `05d95a7ecf310ae80e05240065a8b9681a3a08a79ae29280b4517d404b5a2c32` |
| `results/fewshot10/evals/no6.json` | 18,727 | `ce77d1dde4630cc3d440e63d66175a91dd491f11da787db0bfd613f0ad61711a` |
| `results/fewshot10/evals/no7.json` | 17,744 | `6449df70e5633d41497747b5efc9a126fb0aebbaf88631a09d44a843bc81d358` |
| `results/fewshot10/evals/no8.json` | 19,211 | `0fe32f37af12afd9e11a228fa32edf407ea8981a91f1278167eff46c95995d7f` |
| `results/fewshot10/evals/no9.json` | 19,146 | `b0e2092c3416197672cfece8d89fc021f6ea244232cae84363037522dd4af0d8` |
| `results/fewshot10/reference_10k/control_base.json` | 2,134 | `0253c89b2109490fca4d7357690e2cdf159d38ce33b4b33aae2288fe251835d5` |
| `results/fewshot10/router_gate/no3.json` | 198,566 | `68d5354ff491ca81553398b65aebd2ba34b5744ecbe089b61fdb8fab10aeff67` |
| `results/fewshot10/summary.csv` | 2,306 | `bf6b1436ac5d0f974fa7b424567b28a0f589b536b6c653a9dff2e4ba61b4dd34` |
| `results/fewshot10/summary.md` | 1,983 | `c6a1d02ccbc3e09c451ee604db3430e8f2eceb11c88af2cae2d1a76c439d8ee6` |

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
