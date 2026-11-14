"""Are the A100's tensor cores in use, and what does each stage cost on them?

    python scripts/ops/check_tensor_cores.py --config configs/sdxl_cityscapes_fewshot10.yaml

Needs the GPU to itself: the rates it records feed the schedule. Writes
runs/<name>_gate/tensor_core_gate.json and exits 0 only if every check passes.

1. Flags, dtypes and the share of GEMM/conv/attention kernel time on tensor-core
   kernels, the check Hierarchical-CFG runs in src/hcfg/gpu_check.py.
2. Fostering: s/step and peak VRAM over real steps, in a fresh process.
3. Router: the largest set (No.9), batch 8. Gradient checkpointing stays off if the
   peak stays below ROUTER_VRAM_LIMIT_MIB, and is switched on otherwise.
4. Generation with a router bundle: s/image.

Steps 2-4 only need experts and a bundle of the right shape, so they read the
full-data run's, under a separate <name>_gate run directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from calora.data import build_dataset  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.train.lora import lora_parameters  # noqa: E402
from catsm.cli import restrict_to_training_subset  # noqa: E402
from catsm.config import load_config  # noqa: E402
from catsm.intervals import Interval  # noqa: E402
from catsm.select import load_selection  # noqa: E402
from catsm.train.foster import train_expert  # noqa: E402

TENSOR_CORE_SHARE_MIN = 0.8
SECONDS_PER_STEP_MAX = 0.6
ROUTER_VRAM_LIMIT_MIB = 70_000
FOSTER_STEPS = 50
ROUTER_STEPS = 20
GATE_IMAGES = 2  # in-domain; plus 4 weather conditions x 1, 6 images in all
TC_MARKERS = ("16816", "1688", "tensorop", "xmma", "hmma", "bf16", "tf32", "flash", "fmha",
              "cutlass_80", "sm80", "sm90")
MATH_MARKERS = ("gemm", "conv", "fprop", "dgrad", "wgrad", "flash", "fmha", "attention", "xmma",
                "cutlass", "implicit")
FULL_RUN = REPO / "runs" / "sdxl_cityscapes"
SIBLING_SUBSET = Path("/workspace/Improving-CA-LoRA/runs/sdxl_cityscapes_fewshot10/training_subset.json")
PY = sys.executable


def classify_kernels(events) -> dict:
    """Split CUDA time of GEMM/conv/attention kernels into tensor-core vs other."""
    tensor_core = other = 0.0
    top_other: list[tuple[str, float]] = []
    for event in events:
        time_us = getattr(event, "self_device_time_total", None)
        if time_us is None:
            time_us = getattr(event, "self_cuda_time_total", 0.0)
        if not time_us:
            continue
        name = event.key.lower()
        if not any(marker in name for marker in MATH_MARKERS):
            continue
        if "sgemm" not in name and any(marker in name for marker in TC_MARKERS):
            tensor_core += time_us
        else:
            other += time_us
            top_other.append((event.key, time_us))
    total = tensor_core + other
    top_other.sort(key=lambda item: -item[1])
    return {
        "tensor_core_share": tensor_core / total if total else 0.0,
        "math_time_ms": total / 1e3,
        "top_non_tensor_core": [(key[:120], round(us / 1e3, 2)) for key, us in top_other[:5]],
    }


class VramPeak:
    """Polls nvidia-smi, so it also sees the memory of child processes."""

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.wait(2.0):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out.isdigit():
                self.peak = max(self.peak, int(out))

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_logged(command: list[str], log: Path) -> dict:
    """Run ``command``, logging to ``log``; report rc, wall time, peak VRAM and OOM."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    with VramPeak() as vram, log.open("w") as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO, env=env).returncode
    text = log.read_text(errors="replace")
    return {
        "returncode": rc,
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": vram.peak,
        "oom": bool(re.search(r"out of memory|OutOfMemoryError", text)),
        "log": str(log),
    }


def seconds_per_step(log: Path) -> float | None:
    """The last ``(.., X.XXs/step`` of a catsm training log."""
    found = re.findall(r"([0-9.]+)s/step", log.read_text(errors="replace"))
    return float(found[-1]) if found else None


def generation_seconds_per_image(log: Path) -> float:
    """Wall time between the first and last ``generate_images |   k/n`` lines, per image.

    Model loading dominates a run this short, so the total wall time overstates the rate,
    and a progress bar's own rate does not say how many images its batch held.
    """
    stamps = [
        (int(h) * 3600 + int(m) * 60 + int(s), int(done))
        for h, m, s, done in re.findall(r"^(\d+):(\d+):(\d+) \| INFO\s+\| generate_images \|\s+(\d+)/\d+\s*$",
                                        log.read_text(errors="replace").replace("\r", "\n"), re.M)
    ]
    if len(stamps) < 2:
        raise RuntimeError(f"fewer than two progress lines in {log}: cannot time generation")
    (t0, n0), (t1, n1) = stamps[0], stamps[-1]
    return ((t1 - t0) % 86400) / (n1 - n0)


def check_in_process(cfg, gate_dir: Path) -> dict:
    """Flags, dtypes, the 10 images and the tensor-core share, on real fostering steps."""
    from torch.profiler import ProfilerActivity, profile, schedule

    adapter = load_adapter(cfg.model)
    dataset = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=cfg.train.seed)
    subset = restrict_to_training_subset(cfg, dataset)
    sibling = json.loads(SIBLING_SUBSET.read_text()) if SIBLING_SUBSET.exists() else None
    selection = load_selection(FULL_RUN / "selections" / "n1" / "n1_i0.json")
    train = {**cfg.train.__dict__, "max_steps": 8, "log_every": 4, "checkpoint_every": 0}
    with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=2, warmup=2, active=3)) as prof:
        train_expert(
            adapter, dataset, selection, type(cfg.train)(**train),
            interval=Interval(0, 124), expert="n4_i0", output_dir=gate_dir / "profile",
            callbacks=[lambda _state: prof.step()],
        )
    return {
        "kernels": classify_kernels(prof.key_averages()),
        "flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "train_mixed_precision": cfg.train.mixed_precision,
            "assemble_mixed_precision": cfg.assemble.mixed_precision,
            "gradient_checkpointing": cfg.model.enable_gradient_checkpointing,
        },
        "dtypes": {
            "unet": str(next(adapter.unet.parameters()).dtype),
            "vae": str(next(adapter._pipe.vae.parameters()).dtype),
            "lora": sorted({str(p.dtype) for p in lora_parameters(adapter.unet)}),
        },
        "subset_keys": subset["keys"],
        "subset_matches_sibling": sibling is not None and sibling["keys"] == subset["keys"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sdxl_cityscapes_fewshot10.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, {"name": "sdxl_cityscapes_fewshot10_gate"})
    gate_dir = cfg.run_dir
    gate_dir.mkdir(parents=True, exist_ok=True)
    base = ["--config", args.config, "--name", cfg.name]

    report: dict = {"device": torch.cuda.get_device_name(0), "in_process": check_in_process(cfg, gate_dir)}
    gc.collect()
    torch.cuda.empty_cache()

    log = gate_dir / "foster.out"
    report["foster"] = run_logged(
        [PY, "-m", "catsm.cli", "foster", *base, "--expert", "n4_i0",
         "--selection", str(FULL_RUN / "selections" / "n1" / "n1_i0.json"),
         "--set", f"train.max_steps={FOSTER_STEPS}", "train.log_every=10", "train.checkpoint_every=0"],
        log,
    )
    report["foster"]["seconds_per_step"] = seconds_per_step(log)

    router = [PY, "-m", "catsm.cli", "assemble", *base, "--experts-dir", str(FULL_RUN / "experts"),
              "--sets", "n1", "n3", "n4", "n5", "--core", "n5",
              "--set", f"assemble.max_steps={ROUTER_STEPS}", "assemble.log_every=5"]
    log = gate_dir / "router_nogc.out"
    report["router_nogc"] = run_logged(router, log)
    report["router_nogc"]["seconds_per_step"] = seconds_per_step(log)
    nogc = report["router_nogc"]
    use_gc = nogc["oom"] or nogc["returncode"] != 0 or nogc["peak_vram_mib"] > ROUTER_VRAM_LIMIT_MIB
    if use_gc:
        log = gate_dir / "router_gc.out"
        report["router_gc"] = run_logged([*router, "model.enable_gradient_checkpointing=true"], log)
        report["router_gc"]["seconds_per_step"] = seconds_per_step(log)
    report["router_gradient_checkpointing"] = use_gc
    chosen = report["router_gc"] if use_gc else nogc

    generation = gate_dir / "generate.out"
    report["generate"] = run_logged(
        [PY, "scripts/generate_images.py", *base,
         "--bundle", str(FULL_RUN / "bundles" / "n1+n4-core-n4" / "bundle.safetensors"),
         "--run-name", "gate_generate",
         "--set", f"eval.num_images={GATE_IMAGES}", "eval.images_per_condition=1"],
        generation,
    )
    seconds_per_image = generation_seconds_per_image(generation)
    report["generate"]["seconds_per_image"] = seconds_per_image

    report["estimates"] = {
        "foster": (report["foster"]["seconds_per_step"] or 0.3) * cfg.train.max_steps + 120,
        "router": (chosen["seconds_per_step"] or 2.0) * cfg.assemble.max_steps + 120,
        "generate": seconds_per_image * 900 + 120,
    }

    inside = report["in_process"]
    checks = {
        "tensor_core_share": inside["kernels"]["tensor_core_share"] >= TENSOR_CORE_SHARE_MIN,
        "seconds_per_step": (report["foster"]["seconds_per_step"] or 1e9) <= SECONDS_PER_STEP_MAX,
        "dtypes_fp32": inside["dtypes"]["unet"] == "torch.float32"
        and inside["dtypes"]["vae"] == "torch.float32"
        and inside["dtypes"]["lora"] == ["torch.float32"],
        "matmul_tf32": bool(inside["flags"]["matmul_allow_tf32"]),
        "subset_matches_sibling": inside["subset_matches_sibling"],
        "foster_ok": report["foster"]["returncode"] == 0,
        "router_ok": chosen["returncode"] == 0,
        "generate_ok": report["generate"]["returncode"] == 0,
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    (gate_dir / "tensor_core_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
