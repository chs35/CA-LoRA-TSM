# CA-LoRA-TSM

Implementation of `docs/draft.md`: CA-LoRA whose concept-aware head selection and
sliced LoRA weights are made **per timestep interval**, then combined by a router
(TimeStep-Master-style Fostering → Assembling).

Core source (per-interval selection, the multi-expert sliced LoRA module,
interval-enforced fostering, router training, the bundle format) plus the four
scripts the GPU run needs: CA measurement, timestep diagnosis, bundle-aware
generation and CMMD/CLIP-Score evaluation. mIoU is not computed (as in
Improving CA-LoRA); the experiment matrix itself is driven by hand, see
`docs/GPU_RUNBOOK.md`.

## Layout

| Path | What |
|---|---|
| `src/calora/` | Verbatim copy of Improving-CA-LoRA (`VENDOR.txt` records the commit). CA measurement, top-k head selection, sliced type A/B LoRA, training loop helpers. Never edited. |
| `src/catsm/intervals.py` | `Interval`, `IntervalScheme`, halving boundaries (n3/n4/n5 as in the draft), log-SNR record. |
| `src/catsm/select.py` | Per-interval selection: `representative` (one t), `aggregate` (all measured t in the interval), `fixed` (control C1). |
| `src/catsm/models/multi_sliced_lora.py` | `SlicedLoRABranch` + `MultiSlicedLoRALinear`: several experts with different head slices on one matrix, gated per sample. |
| `src/catsm/models/router.py` | `Router` = FC(mean z_t) + E(t) with one raw (no activation) gate **per projection matrix and context set** (TSM-style); `RoutingTable` picks the active expert per set. |
| `src/catsm/train/foster.py` | `train_expert`: the reference loop with t drawn from `[t_lo, t_hi]` and checked every step. |
| `src/catsm/train/assemble.py` | Load expert files into containers, `TSMAssembly`, `train_router` (experts frozen). |
| `src/catsm/bundle.py` | `bundle.safetensors` (`format=tsm-sliced-v1`) writer/loader with strict validation. |
| `src/catsm/cli.py` | `select` / `foster` / `assemble` commands. |
| `scripts/measure_ca.py` | CA measurement over `measure.timesteps` (calora's script, this project's config). |
| `scripts/diagnose_ca.py` | Figures for the interval t decision (no score): per-t style/viewpoint CA histograms, layer×kind heatmaps, top-2% Jaccard matrix, decision template. |
| `scripts/generate_images.py` | Sampling with `--bundle` (gate refreshed every denoising step via a UNet pre-hook), `--lora`, or the base model. |
| `scripts/evaluate.py` | CMMD + per-condition CLIP-Score for a generated arm. mIoU is not computed (no label generator). |

## Run

```bash
source scripts/env.sh                      # venv-calora + PYTHONPATH=src
python -m pytest tests/ -q                 # toy model, CPU, ~10 s

# toy end-to-end (no downloads)
python scripts/measure_ca.py --config configs/tiny_cpu.yaml
python -m catsm.cli select   --config configs/tiny_cpu.yaml --scheme n1
python -m catsm.cli select   --config configs/tiny_cpu.yaml --scheme n2
python -m catsm.cli foster   --config configs/tiny_cpu.yaml --expert n1_i0   # also n2_i0, n2_i1
python -m catsm.cli assemble --config configs/tiny_cpu.yaml --sets n1 n2 --core n2
python -m catsm.cli assemble --config configs/tiny_cpu.yaml --sets n1 n2 --core n2 --disable n2_i1   # control C2
python scripts/diagnose_ca.py       --config configs/tiny_cpu.yaml
python scripts/generate_images.py   --config configs/tiny_cpu.yaml --bundle runs/tiny_cpu/bundles/n1+n2-core-n2/bundle.safetensors --run-name no3
python scripts/evaluate.py          --config configs/tiny_cpu.yaml --run-name no3 --embedder pixel
```

Outputs land under `runs/<name>/`: `measurements/`, `selections/<scheme>/`,
`experts/<expert>/lora-final.safetensors` (+ `train_summary.json` with the
timestep range actually visited), `bundles/<sets>-core-<core>/bundle.safetensors`.

On the GPU pod use `configs/sdxl_cityscapes.yaml` (4K fostering steps per expert)
with the same commands; expert names are `n4_i0` … `n4_i3` (lowest t first).

Step-by-step GPU instructions (environment, test gate, measurement reuse, t decision,
fostering/assembling order, reporting rules, evaluation gaps): `docs/GPU_RUNBOOK.md`.

## GPU check

`tests/test_gpu.py` is skipped without CUDA. On the pod, before the real run:

```bash
python -m pytest tests/test_gpu.py -q                # toy model on CUDA: CLI pipeline with
                                                     # model.device=cuda, CPU/CUDA agreement,
                                                     # bf16/fp16, bundle round trip
CATSM_SDXL_SMOKE=1 python -m pytest tests/test_gpu.py -q -k sdxl   # SDXL config, 256 px, dummy
                                                     # data, one step per stage (weights in HF_HOME)
```

`--set` values are YAML: `train.mixed_precision=no` parses to `False`; the
default is already `"no"`, so leave it out (or quote it: `train.mixed_precision="'no'"`).
