"""Sample images from a TSM bundle (or a single expert file), for CMMD and CLIP-Score.

    python scripts/generate_images.py --config configs/sdxl_cityscapes.yaml \
        --bundle runs/sdxl_cityscapes/bundles/n1+n4-core-n4/bundle.safetensors --run-name no3
    python scripts/generate_images.py --config ... --lora runs/sdxl_cityscapes/experts/n1_i0/lora-final.safetensors --run-name n1_alone
    python scripts/generate_images.py --config ...                       # base model control

Copied from Optimizing-CA-LoRA/scripts/generate_images.py with one substantive
addition: ``--bundle``. A bundle's experts only act when the gate is set for the
current timestep, and the diffusers pipeline calls the UNet directly, so a bundle
loaded without the hook below would silently sample the *base* model. The hook
is a forward pre-hook on the UNet: at every denoising step it reads the batch
and the timestep the sampler passes in and refreshes the routing gate
(``assembly.set_gate_for``) before the forward runs. Classifier-free guidance
doubles the batch; the hook takes the batch as it arrives, so the gate has the
right length either way.

Two prompt families, matching the paper's two measurements:

* ``in_domain`` — ``eval.num_images`` samples of the plain training prompt (CMMD set).
* ``weather``   — ``eval.images_per_condition`` samples per ``eval.weather_conditions``
  (CLIP-Score set).

mIoU is not computed anywhere in this project (no label generator; same
decision as Improving CA-LoRA), so the generated images have no paired ground truth.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.data import ClassNameSampler  # noqa: E402
from calora.models.adapter import DenoiserAdapter  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.train.lora import load_lora  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402
from calora.utils.io_images import save_image_verified  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402
from calora.utils.seeding import derive_generator  # noqa: E402
from catsm import cli as catsm_cli  # noqa: E402
from catsm.bundle import load_bundle  # noqa: E402
from catsm.config import CatsmConfig  # noqa: E402
from catsm.train.assemble import TSMAssembly  # noqa: E402

logger = get_logger("generate_images")

MODES = ("in_domain", "weather", "both")


# --------------------------------------------------------------------------------------
# Routing during sampling
# --------------------------------------------------------------------------------------


def _timesteps_for(sample: torch.Tensor, timestep) -> torch.Tensor:
    """The sampler's ``timestep`` (python number, 0-d tensor or ``(B,)``) as ``(B,)`` long."""
    t = torch.as_tensor(timestep, device=sample.device).reshape(-1).long()
    if t.numel() == 1 and sample.shape[0] != 1:
        t = t.expand(sample.shape[0])
    if t.shape[0] != sample.shape[0]:
        raise ValueError(f"timestep batch {t.shape[0]} does not match sample batch {sample.shape[0]}")
    return t


class GateHook:
    """Keeps the bundle's gate in step with whatever the sampler feeds the UNet."""

    def __init__(self, assembly: TSMAssembly):
        self.assembly = assembly
        self.calls = 0
        self._handle = assembly.unet.register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, module, args, kwargs):
        sample = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        self.assembly.set_gate_for(_timesteps_for(sample, timestep), sample)
        self.calls += 1
        return None

    def remove(self) -> None:
        self._handle.remove()
        self.assembly.detach()


def attach_bundle(adapter: DenoiserAdapter, path: str | Path) -> tuple[TSMAssembly, GateHook]:
    assembly = load_bundle(adapter, path)
    if assembly.router is not None:
        assembly.router.eval()
    return assembly, GateHook(assembly)


# --------------------------------------------------------------------------------------
# Prompts and output layout (unchanged from the reference script)
# --------------------------------------------------------------------------------------


def build_prompts(cfg: CatsmConfig, mode: str) -> list[dict]:
    """One record per image to generate: prompt, condition and per-image seed key."""
    sampler = ClassNameSampler(cfg.data, seed=cfg.eval.seed) if cfg.eval.include_class_names else None
    records: list[dict] = []

    def add(condition: str, count: int) -> None:
        for _ in range(count):
            index = len(records)
            text = cfg.eval.prompt
            if condition != "in_domain":
                text = f"{text} in {condition}"
            classes = sampler(index) if sampler is not None else []
            if classes:
                text = f"{text} with {', '.join(classes)}"
            records.append(
                {"index": index, "prompt": text, "condition": condition,
                 "class_names": classes, "seed": cfg.eval.seed}
            )

    if mode in ("in_domain", "both"):
        add("in_domain", cfg.eval.num_images)
    if mode in ("weather", "both"):
        for condition in cfg.eval.weather_conditions:
            add(condition, cfg.eval.images_per_condition)
    return records


def _stamp_output(record: dict, path: Path, run_dir: Path, batch_seed_index: int) -> None:
    record["path"] = str(path.relative_to(run_dir))
    record["batch_seed_index"] = batch_seed_index


def _image_path(output_dir: Path, record: dict) -> Path:
    folder = "in_domain" if record["condition"] == "in_domain" else record["condition"]
    return output_dir / folder.replace(" ", "_") / f"{record['index']:05d}.png"


class _ImageWriter:
    """Encodes and saves finished batches on one background thread; the first
    failure is re-raised in the main thread so a dead write cannot pass as done."""

    def __init__(self, *, depth: int = 4, saver: Callable[..., Path] = save_image_verified) -> None:
        self._saver = saver
        self._queue: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="catsm-image-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if self._error is not None:
                continue
            try:
                image, path = job
                self._saver(image, path)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
                self._error = exc

    def submit(self, image, path: Path) -> None:
        self.raise_if_failed()
        self._queue.put((image, path))

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = cli.base_parser(__doc__ or "")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle", default=None, help="bundle.safetensors from `catsm.cli assemble`")
    source.add_argument("--lora", default=None, help="A single expert / CA-LoRA file (lora-final.safetensors)")
    parser.add_argument("--run-name", default=None, help="Arm subdirectory under the run dir")
    parser.add_argument("--mode", choices=MODES, default="both")
    parser.add_argument("--skip-existing", action="store_true", help="Leave already-rendered images alone")
    args = parser.parse_args(argv)
    cfg = catsm_cli.resolve(args)

    if args.run_name:
        arm = args.run_name
    elif args.bundle:
        arm = Path(args.bundle).parent.name
    elif args.lora:
        arm = Path(args.lora).parent.name
    else:
        arm = "base"
    output_dir = cfg.run_dir / arm / "generated"

    adapter = load_adapter(cfg.model)
    hook: GateHook | None = None
    routing_info: dict | None = None
    if args.bundle:
        assembly, hook = attach_bundle(adapter, args.bundle)
        routing_info = {
            "sets": assembly.sets, "core": assembly.core, "disabled": assembly.disabled,
            "router": assembly.router.describe() if assembly.router else None,
            "experts": [e.to_dict() for e in assembly.experts],
        }
        logger.info("Bundle %s: sets %s, core %s, router=%s, disabled=%s",
                    args.bundle, assembly.sets, assembly.core, assembly.router is not None, assembly.disabled)
    elif args.lora:
        load_lora(adapter.unet, args.lora, cfg.train.lora, expect_granularity=cfg.select.granularity)
        logger.info("Single adapter %s loaded (always active)", args.lora)
    else:
        logger.warning("No --bundle/--lora given; sampling the base model as the control arm")

    records = build_prompts(cfg, args.mode)
    logger.info("Generating %d images for arm '%s' (%d steps, guidance %.1f, %s)",
                len(records), arm, cfg.eval.num_inference_steps, cfg.eval.guidance_scale,
                adapter.scheduler_name)

    writer = _ImageWriter(depth=2 * cfg.eval.generation_batch_size)
    try:
        for start in range(0, len(records), cfg.eval.generation_batch_size):
            chunk = records[start:start + cfg.eval.generation_batch_size]
            paths = [_image_path(output_dir, record) for record in chunk]
            if args.skip_existing and all(path.exists() for path in paths):
                for record, path in zip(chunk, paths):
                    _stamp_output(record, path, cfg.run_dir, chunk[0]["index"])
                continue
            generator = derive_generator(cfg.eval.seed, "gen", chunk[0]["index"])
            images = adapter.generate(
                [record["prompt"] for record in chunk],
                num_inference_steps=cfg.eval.num_inference_steps,
                guidance_scale=cfg.eval.guidance_scale,
                generator=generator,
            )
            if hook is not None and hook.calls == 0:
                raise RuntimeError("The UNet ran without the gate hook firing; the bundle was not applied")
            for record, path, image in zip(chunk, paths, images):
                writer.submit(image, path)
                _stamp_output(record, path, cfg.run_dir, chunk[0]["index"])
            logger.info("  %d/%d", min(start + len(chunk), len(records)), len(records))
    finally:
        writer.close()
        if hook is not None:
            hook.remove()
    writer.raise_if_failed()

    manifest = {
        "run": cfg.name,
        "arm": arm,
        "mode": args.mode,
        "bundle": args.bundle,
        "lora": args.lora,
        "routing": routing_info,
        "gate_hook_calls": hook.calls if hook is not None else 0,
        "protocol": {
            "prompt": cfg.eval.prompt,
            "include_class_names": cfg.eval.include_class_names,
            "guidance_scale": cfg.eval.guidance_scale,
            "num_inference_steps": cfg.eval.num_inference_steps,
            "scheduler": adapter.scheduler_name,
            "resolution": cfg.model.resolution,
            "seed": cfg.eval.seed,
            "generation_batch_size": cfg.eval.generation_batch_size,
            "weather_conditions": cfg.eval.weather_conditions,
            "model": adapter.describe(),
        },
        "images": records,
    }
    dump_json(manifest, output_dir / "manifest.json")
    logger.info("Wrote %s (%d images)", output_dir / "manifest.json", len(records))


if __name__ == "__main__":
    main()
