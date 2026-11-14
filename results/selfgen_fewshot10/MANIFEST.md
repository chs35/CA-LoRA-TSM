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
| `figures/selfgen_fewshot10/jaccard_matrix.png` | 289,767 | `e1e9039471a9247954650188b647b357f71835fe1e6b84e1be31a0bb484719c4` |
| `figures/selfgen_fewshot10/no6_samples.png` | 1,387,280 | `6039c7c2b2c0332935f024c761511cc9cffde0f0ec5becae03261b12c02e17ce` |
| `figures/selfgen_fewshot10/router_gate_vs_t.png` | 150,212 | `4c906ce11408f45d459bd5868c402f9d9a82999685c10ce05a50c24d65d0196b` |
| `figures/selfgen_fewshot10/tradeoff_scatter.png` | 112,464 | `ca56d3ddc3e1dc8130bd722283ffb7e545022d720a3bca08c6cc8f35042553cb` |
| `results/selfgen_fewshot10/comparison_vs_dataset_x0.md` | 1,851 | `31cf3482324343697bb9c2bdd6d4f68b8d6b4e61a54ce5e93242dfa99f671ca6` |
| `results/selfgen_fewshot10/diagnosis_jaccard.json` | 13,984 | `3ac9f6f65c75d4f0a8364f37cc7ebd9c1532da8d3acf888f40dcac9dd5ace73d` |
| `results/selfgen_fewshot10/evals/c1.json` | 4,172 | `5482a0c61c41413cd13fc636fcae7fa65aa233a728bdd88f2d9a70e4ea9183e3` |
| `results/selfgen_fewshot10/evals/c2.json` | 8,159 | `032ec146c95b41ec69c670e2fd584cb188d7145cf39cec10529af7f8e76114b6` |
| `results/selfgen_fewshot10/evals/c2n2.json` | 4,259 | `8a5254e40c38a3c8dc9977fbfceb98e2d8ae2454a3246f75ddb8b936b03e8240` |
| `results/selfgen_fewshot10/evals/no1.json` | 2,818 | `4f5eb4bb82712999466156db7a8b17928690770c767bb3ac7ea314022a7cc899` |
| `results/selfgen_fewshot10/evals/no2.json` | 4,224 | `0797f01409eb57368e103bd1f4e96b26b5815ca7c0fadea71acd05e94b37be2b` |
| `results/selfgen_fewshot10/evals/no3.json` | 8,123 | `d24f6bdbe5c89946a137958642e2f643e3ce014d16c399c4b17df77046483415` |
| `results/selfgen_fewshot10/evals/no4.json` | 14,507 | `2bded64277888fa0139234fc9e34606a9398094b1cfb84e322d93efeef771b2a` |
| `results/selfgen_fewshot10/evals/no5.json` | 15,509 | `e2ba5de83408b11c069c5c428320e593c99800b3df80e77b51db1e126edaf6a6` |
| `results/selfgen_fewshot10/evals/no6.json` | 16,447 | `860ea5777272e1bbc17fd205a9bcbeb84fa7ff3c9bfd7ee5d75b7d58e657d26c` |
| `results/selfgen_fewshot10/evals/no7.json` | 15,440 | `a33c718a86f87444d6f8384af184f20433b9df3aab57476ca77b61e50255963c` |
| `results/selfgen_fewshot10/evals/no8.json` | 16,937 | `2d6e4a42d0784de86280a34f3a58e28a828ed979b2c35803a61217565c11155f` |
| `results/selfgen_fewshot10/evals/no9.json` | 16,864 | `8e49995eed3c4d367cc89d0569cb9e8284916ee8de49de729890f282001da499` |
| `results/selfgen_fewshot10/no6_diagnosis.md` | 2,175 | `5a67b6e7ac7a00d18db62cf69fa7ab3c9d8bd418c6b895cfa7295947982971b6` |
| `results/selfgen_fewshot10/reference_10k/control_base.json` | 2,134 | `0253c89b2109490fca4d7357690e2cdf159d38ce33b4b33aae2288fe251835d5` |
| `results/selfgen_fewshot10/router_gate/no3.json` | 197,429 | `1069590a1b6c025a448c72bdbde7e09f7796fdf812c802c815ee915b8b644c6c` |
| `results/selfgen_fewshot10/router_gate/no6.json` | 372,680 | `9f20de81f8292c87ce3b3146d20811dd65d2dbc7f27c79cc2bfc0bdbf25322d1` |
| `results/selfgen_fewshot10/router_gate/no8.json` | 531,382 | `7a1a86fd7e5f732a01a51c08e02077468deee6d16d1f94c54fd92058b486f4a8` |
| `results/selfgen_fewshot10/summary.csv` | 2,306 | `398daafb01b686ad35237ad97cc12b91b61c141f9481457d90d15831b9dbab0d` |
| `results/selfgen_fewshot10/summary.md` | 1,983 | `5b4be197e8470484cb89f4e3e2a864064a9019024b87d995b5a74ec340561f97` |

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
