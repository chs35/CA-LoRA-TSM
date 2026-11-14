# CA-LoRA-TSM

**English** | [한국어](README.ko.md)

CA-LoRA with head selection per timestep interval, mixed by a TimeStep-Master-style router (SDXL, Cityscapes).

[CA-LoRA](https://arxiv.org/abs/2503.22172) (CVPR 2026) attaches LoRA only to the attention heads with the highest concept-awareness (CA), and measures CA at a single timestep, t = 81. The selected heads change with t, and so does the trained adapter. This repository selects the heads separately for each noise interval, trains one LoRA expert per interval, and mixes the experts with the router of [TimeStep Master](https://arxiv.org/abs/2503.07416) (TSM).

With 10 Cityscapes training images, per-interval selection trades domain fit (CMMD) for weather-prompt controllability (CLIP-Score). The router mixture keeps CMMD at the level of CA-LoRA and recovers 2.29 of the 11.67 weather CLIP-Score points CA-LoRA loses, but it does not beat the two unmixed per-interval configurations. Whether the mixture makes the result less sensitive to the choice of t was not measured.

The full report is in Korean: **[REPORT.md](REPORT.md)**.

## Run

```bash
pip install -e ".[diffusion,dev]"
python -m pytest tests/ -q                     # toy model on CPU, no downloads

CFG=configs/sdxl_cityscapes_selfgen_fewshot10.yaml
SEL=runs/sdxl_cityscapes_selfgen/selections
python -m catsm.cli select   --config configs/sdxl_cityscapes_selfgen.yaml --scheme n4          # heads per interval
python -m catsm.cli foster   --config $CFG --expert n4_i0 --selection $SEL/n4/n4_i0.json    # one expert per interval
python -m catsm.cli assemble --config $CFG --sets n1 n4 --core n4                           # train the router
python scripts/generate_images.py --config $CFG --run-name no3 \
    --bundle runs/sdxl_cityscapes_selfgen_fewshot10/bundles/n1+n4-core-n4/bundle.safetensors
python scripts/evaluate.py        --config $CFG --run-name no3                           # CMMD, CLIP-Score
```

The CA measurements are made once in Improving-CA-LoRA and imported with `scripts/ops/import_measurements.sh`. `scripts/ops/` holds the scripts that ran the experiments on the GPU server (a single A100). Their defaults are those of earlier experiments whose results are not in this repository; the reported run was started by `scripts/ops/run_selfgen_chain.sh`, which sets the config and run names. Cityscapes is not included and must be obtained from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).

## Layout

```
configs/      experiment settings (sdxl_cityscapes_selfgen_fewshot10.yaml is the reported run)
scripts/      CA measurement, diagnosis, generation, evaluation, result tables and figures
src/catsm/    intervals, per-interval selection, multi-expert sliced LoRA, router, training
src/calora/   CA-LoRA reimplementation, copied from Improving-CA-LoRA (CA-LoRA has no official code)
tests/        unit tests on a toy model
results/      evaluation outputs behind the report (selfgen_fewshot10/) and its figures (figures/)
```

## Related repositories

[Improving-CA-LoRA](https://github.com/chs35/Improving-CA-LoRA) (choosing the CA timestep with a quantitative criterion) → **CA-LoRA-TSM** (not relying on one timestep) → [Hierarchical-CFG](https://github.com/chs35/Hierarchical-CFG) (restoring controllability at sampling time).

## License

[MIT](LICENSE). SDXL weights and Cityscapes images are not included and keep their own licenses.
