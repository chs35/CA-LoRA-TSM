"""LoRA fine-tuning restricted to a selected set of attention projections.

The only thing that distinguishes CA-LoRA from plain LoRA here is which module
names go into ``target_modules`` — everything else is a standard latent
diffusion fine-tune. That is deliberate: the ablation arms differ
*only* in the :class:`~calora.ca.select.Selection` handed to :func:`train_lora`,
so any difference in the results is attributable to the projection choice.

CA-drift tracking rides along via ``stability_every``: the same measurement
code from :mod:`calora.ca.measure` is re-run against the live, LoRA-adapted
weights, so no separate training run is needed to answer whether the selected
set stays valid.

The two non-obvious pieces of the loop are :class:`_InputCache` (the VAE and
text encoders run once, not once per step) and
:func:`resolve_mixed_precision` (autocast around the forward, fp32 everywhere
that holds state). Neither changes the optimisation; both exist because a 10k
step arm is run once per ablation and there are several ablations.
"""

from __future__ import annotations

import contextlib
import math
import queue
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from calora.ca.select import Selection
from calora.config import LoraConfig_, TrainConfig
from calora.data.transforms import hflip
from calora.models.adapter import Conditioning, DenoiserAdapter
from calora.models.projections import ProjectionTable, head_unit_name
from calora.models.sliced_lora import HeadLoraPlan, attach_sliced_lora, has_sliced_lora
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator, seed_everything

logger = get_logger(__name__)

StepCallback = Callable[["TrainingState"], None]

#: ``train.mixed_precision`` -> autocast dtype. ``"no"`` means "run in the dtype
#: the model was loaded in", which is what every CPU test does.
AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "no": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class TrainingState:
    """Passed to callbacks; also the object serialised into the run summary."""

    step: int
    loss: float
    learning_rate: float
    elapsed: float
    adapter: DenoiserAdapter | None = None
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "elapsed": self.elapsed,
        }


# --------------------------------------------------------------------------------------
# LoRA wiring
# --------------------------------------------------------------------------------------


def plan_sliced_lora(unet: nn.Module, selection: Selection) -> list[HeadLoraPlan]:
    """Turn a head-granularity selection into one LoRA plan per attention matrix.

    Every structural fact — ``H``, ``d_h``, and which axis a head owns — is read
    off the *live model* through :class:`~calora.models.projections.ProjectionTable`,
    never off the selection file, so a stale JSON cannot talk the trainer into
    slicing the wrong axis or a head that does not exist. The selected heads of
    one matrix are merged into a single adapter (paper: one LoRA per module,
    covering the union of its selected heads).
    """
    if selection.granularity != "head":
        raise ValueError(
            f"plan_sliced_lora needs a head-granularity selection, got {selection.granularity!r}"
        )
    table = ProjectionTable.from_model(unet, granularity="head")
    unknown = [name for name in selection.names if name not in table]
    if unknown:
        raise KeyError(
            f"{len(unknown)} selected unit(s) are absent from the model, e.g. {unknown[:3]}. "
            "The selection was made against a different model or granularity."
        )

    grouped: dict[str, list] = {}
    for name in selection.names:
        spec = table[name]
        grouped.setdefault(spec.module_name, []).append(spec)

    plans: list[HeadLoraPlan] = []
    for module_name, specs in grouped.items():
        first = specs[0]
        head_dim = first.shape[0] if first.slice_dim == "row" else first.shape[1]
        plans.append(
            HeadLoraPlan(
                module_name=module_name,
                kind=first.kind,
                head_indices=tuple(sorted(s.head_index for s in specs)),
                num_heads=first.num_heads,
                head_dim=head_dim,
                slice_dim=first.slice_dim,
            )
        )
    plans.sort(key=lambda plan: plan.module_name)
    return plans


def resolve_target_modules(unet: nn.Module, selection: Selection) -> list[str]:
    """The attention matrices a selection adapts, whatever its granularity.

    A module-granularity selection *is* the ``peft`` target list. A
    head-granularity one names the matrices that will receive a
    :class:`~calora.models.sliced_lora.SlicedLoRALinear` — one per matrix,
    covering the union of its selected heads. Called before training starts so
    that a selection the model cannot express fails immediately, rather than
    after the (GPU-minutes) drift baseline has been measured.
    """
    if selection.granularity != "head":
        return selection.as_target_modules()

    plans = plan_sliced_lora(unet, selection)
    by_type: dict[str, int] = {}
    for plan in plans:
        by_type[plan.lora_type] = by_type.get(plan.lora_type, 0) + 1
    logger.info(
        "Head-granularity selection: %d head unit(s) over %d matrices "
        "(type A/out-column: %d, type B/qkv-row: %d; %d fully selected)",
        len(selection),
        len(plans),
        by_type.get("A", 0),
        by_type.get("B", 0),
        sum(1 for plan in plans if plan.is_full),
    )
    return sorted(plan.module_name for plan in plans)


def attach_lora(
    unet: nn.Module,
    selection: Selection,
    cfg: LoraConfig_,
    *,
    adapter_name: str = "default",
) -> nn.Module:
    """Inject LoRA into exactly the units named by ``selection``.

    Module granularity goes through ``peft`` unchanged (it is also the
    granularity ablation's baseline, so it must stay bit-identical). Head
    granularity goes through :func:`attach_sliced_lora`, because ``peft`` cannot
    express an adapter over part of a matrix.

    Raises if the number of adapted modules differs from the number requested —
    a silent mismatch there would invalidate every downstream comparison.
    """
    if selection.granularity == "head":
        return _attach_head_lora(unet, selection, cfg)
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "peft is required for LoRA training. Install with: pip install 'calora[diffusion]'"
        ) from exc

    targets = resolve_target_modules(unet, selection)
    if not targets:
        raise ValueError("Selection is empty; nothing to adapt")

    peft_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        init_lora_weights=cfg.init_lora_weights,
        bias="none",
        target_modules=targets,
    )
    inject_adapter_in_model(peft_config, unet, adapter_name=adapter_name)

    adapted = sorted(
        name.rsplit(".lora_A", 1)[0]
        for name, _ in unet.named_modules()
        if name.endswith(f".lora_A.{adapter_name}")
    )
    if len(adapted) != len(targets):
        missing = sorted(set(targets) - set(adapted))
        raise RuntimeError(
            f"peft adapted {len(adapted)} modules but the selection named {len(targets)}. "
            f"Unmatched: {missing[:5]}"
        )
    logger.info(
        "LoRA attached to %d/%d projections (rank=%d, alpha=%d)",
        len(adapted),
        len(targets),
        cfg.rank,
        cfg.alpha,
    )
    return unet


def _attach_head_lora(unet: nn.Module, selection: Selection, cfg: LoraConfig_) -> nn.Module:
    """Attach one :class:`SlicedLoRALinear` per matrix of a head selection."""
    plans = plan_sliced_lora(unet, selection)
    if not plans:
        raise ValueError("Selection is empty; nothing to adapt")
    modules = attach_sliced_lora(
        unet,
        plans,
        rank=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        init_lora_weights=cfg.init_lora_weights,
    )
    covered = sum(len(plan.head_indices) for plan in plans)
    if covered != len(selection):
        raise RuntimeError(
            f"Sliced LoRA covers {covered} head unit(s) but the selection named "
            f"{len(selection)}; the two must agree exactly."
        )
    trainable = sum(p.numel() for m in modules for p in (m.lora_A, m.lora_B))
    logger.info(
        "Sliced LoRA attached to %d matrices covering %d/%d head(s) "
        "(rank=%d, alpha=%d, %s LoRA params)",
        len(modules),
        covered,
        sum(plan.num_heads for plan in plans),
        cfg.rank,
        cfg.alpha,
        f"{trainable:,}",
    )
    return unet


def lora_parameters(unet: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in unet.named_parameters() if "lora_" in n and p.requires_grad]


def lora_state_dict(unet: nn.Module) -> dict[str, torch.Tensor]:
    """Every ``lora_*`` tensor, parameters *and* buffers.

    The buffers matter only at head granularity, where
    ``<module>.lora_head_indices`` records which heads the adapter covers. It is
    saved with the weights on purpose: the checkpoint has to be able to say what
    it adapts without consulting a selection JSON that may since have been
    regenerated. The module path has no such buffers, so its state dict is
    unchanged.
    """
    state = {n: p.detach().cpu() for n, p in unet.named_parameters() if "lora_" in n}
    state.update({n: b.detach().cpu() for n, b in unet.named_buffers() if "lora_" in n})
    return state


def save_lora(
    unet: nn.Module,
    path: str | Path,
    metadata: dict[str, str] | None = None,
    *,
    cfg: LoraConfig_ | None = None,
) -> None:
    """Write the LoRA tensors, with the hyperparameters needed to reload them.

    ``alpha`` is recorded because it is *not* recoverable from the tensors — it
    is a scalar multiplier on every delta — so a reload that guessed it would
    produce a model that runs, samples, and is quietly wrong by a constant
    factor. ``rank`` is recorded too even though the shapes imply it, so that a
    mismatch can be reported rather than inferred.

    ``granularity`` is written from what is actually attached to ``unet``, not
    from what the caller believes: mixing a head checkpoint into a module run (or
    the reverse) is the accident this whole refactor exists to prevent, and the
    file has to be able to refuse it on its own. A caller-supplied
    ``granularity`` that disagrees is an error rather than an override.
    ``axis`` (``style``/``viewpoint``) travels the same way, because the run
    matrix is two axes per arm and the checkpoints are otherwise identical in
    shape.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: str(v) for k, v in (metadata or {}).items() if v not in (None, "")}
    granularity = "head" if has_sliced_lora(unet) else "module"
    declared = payload.get("granularity")
    if declared is not None and declared != granularity:
        raise ValueError(
            f"save_lora was told granularity={declared!r} but {granularity!r} is attached to "
            "the model; the model is the ground truth."
        )
    payload["granularity"] = granularity
    if cfg is not None:
        payload.update({"lora_rank": str(cfg.rank), "lora_alpha": str(cfg.alpha)})
    save_file(lora_state_dict(unet), str(path), metadata=payload)
    logger.info("Saved LoRA weights to %s (%s granularity)", path, granularity)


#: Key suffix carrying a sliced LoRA's covered heads (head granularity only).
HEAD_INDEX_SUFFIX = ".lora_head_indices"


def _checkpoint_granularity(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> str:
    """What unit the checkpoint adapts, cross-checking metadata against the keys.

    The keys are the ground truth (a head checkpoint *has* head-index tensors);
    the metadata is what a reader can check cheaply. A file whose two answers
    disagree is refused outright — it was written by something that did not
    understand one of them, and loading it either way risks a silently wrong
    model.

    A checkpoint with neither marker is a module checkpoint written before head
    granularity existed (arm A on disk), which stays loadable.
    """
    has_head_keys = any(key.endswith(HEAD_INDEX_SUFFIX) for key in state)
    declared = metadata.get("granularity")
    if declared is not None and declared not in ("head", "module"):
        raise ValueError(f"Checkpoint declares unknown granularity {declared!r}")
    if has_head_keys and declared != "head":
        raise ValueError(
            f"Checkpoint carries '{HEAD_INDEX_SUFFIX}' tensors but its metadata says "
            f"granularity={declared!r}. Refusing to guess which is right."
        )
    if declared == "head" and not has_head_keys:
        raise ValueError(
            f"Checkpoint declares granularity='head' but has no '{HEAD_INDEX_SUFFIX}' tensors, "
            "so the heads it covers are unknowable."
        )
    return "head" if has_head_keys else "module"


def _head_selection_from_state(
    state: Mapping[str, torch.Tensor], *, path: Path, axis: str
) -> Selection:
    """Rebuild the head-granularity selection a checkpoint encodes."""
    names: list[str] = []
    for key in sorted(state):
        if not key.endswith(HEAD_INDEX_SUFFIX):
            continue
        module = key[: -len(HEAD_INDEX_SUFFIX)]
        heads = [int(h) for h in state[key].flatten().tolist()]
        if sorted(set(heads)) != heads:
            raise ValueError(f"{path}: {key} is not a sorted set of head indices: {heads}")
        names.extend(head_unit_name(module, head) for head in heads)
    return Selection.from_dict(
        {
            "names": names,
            "granularity": "head",
            "strategy": f"loaded:{path.name}",
            "meta": {"axis": axis} if axis else {},
        }
    )


def load_lora(
    unet: nn.Module,
    path: str | Path,
    cfg: LoraConfig_,
    *,
    adapter_name: str = "default",
    expect_granularity: str | None = None,
    expect_axis: str | None = None,
) -> Selection:
    """Re-attach a saved LoRA to ``unet`` and return the selection it covers.

    Which units to adapt is read back out of the checkpoint's own keys rather
    than from the selection JSON: the weights are the ground truth for what was
    trained, and a mismatched ``--selection`` would otherwise produce a silently
    half-loaded model that still generates plausible images. At head granularity
    that includes *which heads* — the ``lora_head_indices`` buffers.

    The rank comes from the ``lora_A`` shapes. ``alpha`` comes from the
    checkpoint metadata that :func:`save_lora` wrote, because it is a plain
    multiplier on every delta and nothing in the tensors reveals it; ``cfg`` is
    only the fallback for a checkpoint written before that metadata existed, and
    using it is warned about since a wrong alpha silently rescales the arm.

    ``expect_granularity`` / ``expect_axis`` are refusals, not preferences: a
    caller that knows which arm it wants gets an error instead of a plausible
    wrong model. Both are optional because loading is also how one *inspects* a
    checkpoint.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path))
    if not state:
        raise ValueError(f"{path} contains no LoRA tensors")
    names = sorted({key.split(".lora_")[0] for key in state if ".lora_" in key})
    if not names:
        raise ValueError(f"{path} has no '*.lora_[AB]*' keys; is it a LoRA checkpoint?")

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    granularity = _checkpoint_granularity(state, metadata)
    if expect_granularity is not None and granularity != expect_granularity:
        raise ValueError(
            f"{path} is a {granularity}-granularity adapter but {expect_granularity!r} was "
            "expected; head and module arms adapt incommensurable units."
        )
    axis = metadata.get("axis", "")
    if expect_axis is not None and axis != expect_axis:
        raise ValueError(
            f"{path} records axis {axis or '<none>'!r} but {expect_axis!r} was expected; "
            "the style and viewpoint adapters of an arm are otherwise indistinguishable."
        )

    ranks = {v.shape[0] for k, v in state.items() if ".lora_A" in k}
    if len(ranks) != 1:
        raise ValueError(f"{path} mixes LoRA ranks {sorted(ranks)}; cannot rebuild one adapter")
    rank = ranks.pop()
    if rank != cfg.rank:
        logger.warning(
            "Checkpoint rank %d differs from train.lora.rank %d; using the checkpoint's",
            rank, cfg.rank,
        )

    if "lora_alpha" in metadata:
        alpha = int(metadata["lora_alpha"])
    else:
        alpha = cfg.alpha
        logger.warning(
            "%s records no lora_alpha; falling back to train.lora.alpha=%d. If that is not "
            "what this arm was trained with, every LoRA delta is rescaled by "
            "%d/alpha_true.", path, alpha, alpha,
        )

    if granularity == "head":
        selection = _head_selection_from_state(state, path=path, axis=axis)
    else:
        selection = Selection.from_dict({"names": names, "strategy": f"loaded:{path.name}"})
    attach_lora(unet, selection, LoraConfig_(
        rank=rank, alpha=alpha, dropout=cfg.dropout, init_lora_weights=cfg.init_lora_weights
    ), adapter_name=adapter_name)

    missing, unexpected = unet.load_state_dict(state, strict=False)
    stale = [k for k in missing if "lora_" in k]
    if stale or unexpected:
        raise RuntimeError(
            f"LoRA load mismatch for {path}: {len(stale)} missing, {len(unexpected)} unexpected "
            f"(e.g. {(stale + list(unexpected))[:3]})"
        )
    logger.info(
        "Loaded LoRA from %s: %d %s unit(s) over %d matrices, rank %d, alpha %d, axis %s",
        path, len(selection), granularity, len(names), rank, alpha, axis or "<unrecorded>",
    )
    return selection


# --------------------------------------------------------------------------------------
# Optimiser / schedule
# --------------------------------------------------------------------------------------


def build_lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    warmup = max(cfg.lr_warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return step / max(warmup, 1)
        if cfg.lr_scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(cfg.max_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.lr_scheduler == "linear":
            return 1.0 - progress
        if cfg.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown lr_scheduler {cfg.lr_scheduler!r}")

    return lr_lambda


# --------------------------------------------------------------------------------------
# Mixed precision
# --------------------------------------------------------------------------------------


def resolve_mixed_precision(
    mode: str, device: torch.device
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None]:
    """Validate ``train.mixed_precision`` and build the autocast dtype / loss scaler.

    Only the forward pass and the loss run in reduced precision; the LoRA
    parameters, their gradients and the Adam moments stay fp32, which is what
    keeps a rank-64 update from being quantised away.

    fp16 needs a :class:`torch.amp.GradScaler` — its dynamic range is too narrow
    for raw diffusion-loss gradients — and the scaler is CUDA-only, so fp16 on
    CPU is rejected outright rather than silently producing zeros. bf16 has the
    exponent range of fp32 and needs no scaler, which is why it is the sane
    choice on both an A100 and a laptop.
    """
    if mode not in AUTOCAST_DTYPES:
        raise ValueError(
            f"Unknown train.mixed_precision {mode!r}; expected one of {sorted(AUTOCAST_DTYPES)}"
        )
    dtype = AUTOCAST_DTYPES[mode]
    if mode == "fp16" and device.type != "cuda":
        raise ValueError(
            f"train.mixed_precision='fp16' requires a CUDA device (got {device.type!r}): "
            "torch.amp.GradScaler is CUDA-only. Use 'bf16' instead."
        )
    scaler = torch.amp.GradScaler(device.type) if mode == "fp16" else None
    return dtype, scaler


def _autocast(device_type: str, dtype: torch.dtype | None):
    """``torch.autocast``, or a no-op context when mixed precision is off.

    Deliberately *not* ``autocast(enabled=False)``: entering the real context
    manager perturbs nothing numerically but the null path is what guarantees
    the ``mixed_precision="no"`` runs stay bit-identical to before this existed.
    """
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


# --------------------------------------------------------------------------------------
# Input caching
# --------------------------------------------------------------------------------------


class _InputCache:
    """Keeps the two per-step encodings that do not depend on the training step.

    Without this, every step re-does work whose answer never changes: decoding a
    PNG, running the VAE over it, and running the text encoders over one of the
    ~18 distinct template captions Cityscapes produces. At 10k steps that is
    10k VAE forwards over 3k images.

    * **Captions** are cached unconditionally, keyed by the caption string.
      Conditioning tensors are kept on CPU and moved per step.
    * **Latents** are pre-computed for the whole dataset only when
      ``__getitem__`` is deterministic. They are the reason ``crop="random"``
      and ``cache_latents`` are mutually exclusive: with a random crop the image
      behind an index is different on every read, so a cache would silently pin
      training to whatever crop happened to be drawn first.
    * **Both orientations** are cached when the dataset allows a horizontal
      flip. ``VAE(flip(x)) != flip(VAE(x))``, so the mirrored latent has to come
      from a second encode of the mirrored *pixels*; flipping the cached tensor
      would be cheaper and wrong. The cost is one extra encode per image and
      double the cache, which is why ``data.hflip`` is opt-in.
    """

    def __init__(self, adapter: DenoiserAdapter):
        self._adapter = adapter
        self._cond: dict[str, Conditioning] = {}
        self._offloaded = False
        self.latents: list[torch.Tensor] | None = None
        #: Same images encoded mirrored, or ``None`` when ``data.hflip`` is off.
        self.latents_flipped: list[torch.Tensor] | None = None
        self.captions: list[str] | None = None

    def precompute_latents(self, dataset: Sequence[dict], *, with_hflip: bool = False) -> None:
        """Encode every image once per orientation, into CPU fp32 tensors.

        Encoded one image at a time so the cached value is exactly what the
        uncached path would have produced for a batch containing it; the VAE
        takes the distribution *mode*, so this is deterministic.
        """
        latents: list[torch.Tensor] = []
        flipped: list[torch.Tensor] = []
        captions: list[str] = []
        started = time.perf_counter()
        for index in range(len(dataset)):
            item = dataset[index]
            image = item["image"]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            latents.append(self._adapter.encode_images(image).detach().to("cpu", torch.float32))
            if with_hflip:
                flipped.append(
                    self._adapter.encode_images(hflip(image)).detach().to("cpu", torch.float32)
                )
            captions.append(item["caption"])
        self.latents = latents
        self.latents_flipped = flipped if with_hflip else None
        self.captions = captions
        total = sum(t.numel() for t in latents) * 4 * (2 if with_hflip else 1)
        logger.info(
            "Cached %d latents %s%s in %.1f MiB (%.1fs)",
            len(latents),
            tuple(latents[0].shape[1:]),
            " x2 orientations" if with_hflip else "",
            total / 2**20,
            time.perf_counter() - started,
        )

    def warm(self, captions: Sequence[str]) -> int:
        """Encode any caption not already cached; returns how many were new.

        Honours the offloaded state: an unforeseen caption after
        :meth:`park_text_encoders` still gets encoded — correctness first — by
        pulling the encoders back and pushing them away again afterwards.
        """
        missing = [c for c in dict.fromkeys(captions) if c not in self._cond]
        if not missing:
            return 0
        if self._offloaded:
            logger.warning(
                "%d caption(s) unseen when the text encoders were offloaded; reloading them "
                "onto %s for this step", len(missing), self._adapter.device,
            )
            self._adapter.restore_encoders()
        for caption in missing:
            encoded = self._adapter.encode_prompts([caption])
            self._cond[caption] = {k: v.detach().cpu() for k, v in encoded.items()}
        if self._offloaded:
            self._adapter.offload_encoders(include_vae=False)
        return len(missing)

    def park_text_encoders(self) -> None:
        """Move the text encoders to CPU, keeping the VAE where it is.

        Everything already in the cache stays usable because conditioning
        tensors are held on CPU; only a *new* caption needs the encoders, and
        :meth:`warm` handles that case.
        """
        self._adapter.offload_encoders(include_vae=False)
        self._offloaded = True

    def conditioning(self, captions: Sequence[str]) -> Conditioning:
        """Conditioning for a batch of captions, encoding only the unseen ones."""
        self.warm(captions)
        per_caption = [self._cond[c] for c in captions]
        return {
            key: torch.cat([cond[key] for cond in per_caption], dim=0)
            for key in per_caption[0]
        }


def _distinct_captions(dataset: Sequence[dict]) -> list[str] | None:
    """Every caption ``dataset`` can produce, or ``None`` if it cannot say cheaply.

    Deliberately not ``{dataset[i]["caption"] for i in ...}``: that would decode
    the whole dataset to read metadata. A dataset that offers no cheap listing
    simply does not get its text encoders offloaded.
    """
    lister = getattr(dataset, "distinct_captions", None)
    if not callable(lister):
        return None
    captions = list(lister())
    return captions or None


def _park_text_encoders(
    adapter: DenoiserAdapter, cache: "_InputCache", dataset: Sequence[dict]
) -> bool:
    """Pre-encode every caption of ``dataset``, then drop the text encoders to CPU.

    Table 12's training prompt is a fixed template, so the whole 10k-step run
    needs a handful of text encodings; the encoders themselves are ~3.3 GB of
    fp32 weights that would otherwise sit in VRAM until the run ends. The VAE
    stays put — with ``data.crop="random"`` it encodes one image per step.
    """
    captions = _distinct_captions(dataset)
    if captions is None:
        logger.info("Dataset cannot list its captions; leaving the text encoders on the GPU")
        return False
    cache.warm(captions)
    cache.park_text_encoders()
    logger.info(
        "Pre-encoded %d distinct caption(s); text encoders moved to CPU (VAE stays on %s)",
        len(captions), adapter.device,
    )
    return True


def _micro_batch_plan(num_items: int, cfg: TrainConfig) -> Iterator[list[int]]:
    """The exact ``(step, micro) -> indices`` sequence the loop will consume.

    Recomputed here rather than handed over from the loop so the prefetcher needs
    no synchronisation with it: :func:`_draw_indices` is a pure function of
    ``(seed, step, micro)``.
    """
    for step in range(1, cfg.max_steps + 1):
        for micro in range(cfg.gradient_accumulation_steps):
            yield _draw_indices(num_items, cfg, step, micro)


class _ItemPrefetcher:
    """Reads the next micro-batch's dataset items on one background thread.

    Only used when the latent cache is off (``data.crop="random"``), which is
    where the CPU side — a 2048x1024 PNG decode plus a crop per item — starves
    the GPU between steps.

    **Why this is numerically free.** ``crop="random"`` does *not* draw from the
    global torch/python RNG: ``CityscapesConceptDataset._crop_generator`` derives
    a fresh generator per read from ``(seed, "cityscapes-crop", index,
    self._crop_draws)``. But ``_crop_draws`` is a counter on the dataset that
    every read increments, so which window a read gets depends on *how many reads
    preceded it* — the order matters even though the global RNG does not. One
    worker consuming one plan of micro-batches therefore reproduces the
    sequential loop's read order exactly, and nothing else touches the training
    dataset while the loop runs (the drift callback measures a separately built
    centre-cropped dataset, prepared before training starts).

    Depth stays at 1-2 for the same reason it exists: one step of lookahead is
    all that is needed to hide the decode, and each queued micro-batch holds
    full-resolution image tensors.
    """

    def __init__(
        self, dataset: Sequence[dict], plan: Iterator[Sequence[int]], *, depth: int = 1
    ):
        self._dataset = dataset
        self._plan = plan
        self._queue: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="calora-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for indices in self._plan:
                items = [self._dataset[i] for i in indices]
                if not self._put((items, None)):
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._put((None, exc))

    def _put(self, payload: tuple) -> bool:
        """Block for room, but stay interruptible so :meth:`close` cannot hang."""
        while not self._stop.is_set():
            try:
                self._queue.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def next(self) -> list[dict]:
        """The next micro-batch, re-raising whatever the worker hit while reading."""
        items, error = self._queue.get()
        if error is not None:
            raise error
        return items

    def close(self) -> None:
        """Stop the worker and join it. Safe to call twice, and on the error path."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=30.0)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _is_deterministic(dataset: Sequence[dict]) -> bool:
    """Whether ``dataset[i]`` returns the same image every time it is read.

    ``data.hflip`` does not enter this: the flip is drawn by the training loop,
    not inside ``__getitem__``, precisely so that it stays compatible with the
    latent cache.
    """
    return getattr(dataset, "crop", "center") != "random"


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------


def train_lora(
    adapter: DenoiserAdapter,
    dataset: Sequence[dict],
    selection: Selection,
    cfg: TrainConfig,
    *,
    output_dir: str | Path,
    callbacks: Sequence[StepCallback] = (),
    extra_metadata: Mapping[str, str] | None = None,
) -> TrainingState:
    """Fine-tune ``selection``'s units with LoRA on the standard diffusion loss.

    Note that training uses only ``L_Diff`` — ``L_Concept`` is a *measurement*
    device for choosing which units to adapt, not a training objective.

    At ``selection.granularity == "head"`` the adapters are the paper's sliced
    type A/B (see :mod:`calora.models.sliced_lora`); at ``module`` they are
    ordinary ``peft`` LoRAs. Nothing else about the loop differs.

    ``cfg.cache_latents`` and ``cfg.mixed_precision`` only affect how fast this
    runs, never what it converges to; see :class:`_InputCache` and
    :func:`resolve_mixed_precision` for the conditions under which that holds.

    ``extra_metadata`` is written into every checkpoint. The axis of the arm goes
    in there: a style and a viewpoint adapter of the same arm are identical in
    shape, so nothing but the metadata can tell them apart after the fact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    # Validated before anything is mutated: a bad precision setting should not
    # leave the caller's UNet with LoRA modules injected into it.
    autocast_dtype, scaler = resolve_mixed_precision(cfg.mixed_precision, adapter.device)

    unet = adapter.unet
    unet.requires_grad_(False)
    attach_lora(unet, selection, cfg.lora)
    params = lora_parameters(unet)
    if not params:
        raise RuntimeError("No trainable LoRA parameters after injection")
    trainable = sum(p.numel() for p in params)
    logger.info("Trainable LoRA parameters: %s", f"{trainable:,}")

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(cfg))

    device = adapter.device
    if autocast_dtype is not None:
        logger.info(
            "Mixed precision: %s autocast on %s (LoRA weights and Adam state stay fp32)",
            cfg.mixed_precision, device.type,
        )

    # The dataset only *declares* that flipping is allowed (data.hflip); the
    # draw lives here so that the cached and uncached paths make the identical
    # decision for a given (step, micro).
    use_hflip = bool(getattr(dataset, "hflip", False))
    if use_hflip:
        logger.info("Random horizontal flip enabled (paper augmentation: random crop + hflip)")

    cache = _InputCache(adapter)
    if cfg.cache_latents:
        if _is_deterministic(dataset):
            cache.precompute_latents(dataset, with_hflip=use_hflip)
        else:
            logger.warning(
                "train.cache_latents=true ignored: the dataset uses a random crop, so "
                "dataset[i] is not a fixed image. Falling back to per-step VAE encoding."
            )

    # Only worth doing where there is VRAM to reclaim; on CPU it would be a
    # no-op move plus a pile of eager encodes.
    if device.type == "cuda":
        _park_text_encoders(adapter, cache, dataset)

    reads_dataset = cache.latents is None or cache.captions is None
    prefetcher = (
        _ItemPrefetcher(dataset, _micro_batch_plan(len(dataset), cfg))
        if reads_dataset and cfg.prefetch
        else None
    )

    # Provenance carried by every checkpoint, intermediate ones included: an
    # arm is identified by (granularity, axis, strategy) and none of it is
    # recoverable from the tensors.
    metadata = {"strategy": selection.strategy}
    axis = str(selection.meta.get("axis", "") or "")
    if axis:
        metadata["axis"] = axis
    metadata.update({k: str(v) for k, v in (extra_metadata or {}).items()})

    unet.train()
    state = TrainingState(step=0, loss=float("nan"), learning_rate=cfg.learning_rate,
                          elapsed=0.0, adapter=adapter)
    started = time.perf_counter()

    try:
        _run_steps(
            adapter,
            dataset,
            cfg,
            state,
            cache=cache,
            prefetcher=prefetcher,
            optimizer=optimizer,
            scheduler=scheduler,
            params=params,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            use_hflip=use_hflip,
            callbacks=callbacks,
            output_dir=output_dir,
            started=started,
            metadata=metadata,
        )
    finally:
        # No dangling reader thread, whether the loop finished or raised.
        if prefetcher is not None:
            prefetcher.close()

    save_lora(
        unet,
        output_dir / "lora-final.safetensors",
        metadata={**metadata, "steps": str(cfg.max_steps)},
        cfg=cfg.lora,
    )
    unet.eval()
    return state


def _run_steps(
    adapter: DenoiserAdapter,
    dataset: Sequence[dict],
    cfg: TrainConfig,
    state: TrainingState,
    *,
    cache: _InputCache,
    prefetcher: _ItemPrefetcher | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    params: list[nn.Parameter],
    scaler: torch.amp.GradScaler | None,
    autocast_dtype: torch.dtype | None,
    use_hflip: bool,
    callbacks: Sequence[StepCallback],
    output_dir: Path,
    started: float,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """The optimisation loop proper, split out only so that :func:`train_lora` can
    wrap it in the ``finally`` that shuts the prefetch thread down."""
    unet = adapter.unet
    device, dtype = adapter.device, adapter.dtype

    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro in range(cfg.gradient_accumulation_steps):
            indices = _draw_indices(len(dataset), cfg, step, micro)
            flips = _draw_flips(len(indices), cfg, step, micro) if use_hflip else [False] * len(
                indices
            )
            if cache.latents is not None and cache.captions is not None:
                pools = (cache.latents, cache.latents_flipped or cache.latents)
                latents = torch.cat(
                    [pools[int(flip)][i] for i, flip in zip(indices, flips)], dim=0
                )
                captions = [cache.captions[i] for i in indices]
            else:
                batch = (
                    prefetcher.next() if prefetcher is not None
                    else [dataset[i] for i in indices]
                )
                images = [
                    hflip(item["image"]) if flip else item["image"]
                    for item, flip in zip(batch, flips)
                ]
                latents = adapter.encode_images(torch.stack(images))
                captions = [item["caption"] for item in batch]
            latents = latents.to(device=device, dtype=dtype)
            cond = {
                k: v.to(device=device, dtype=dtype)
                for k, v in cache.conditioning(captions).items()
            }

            noise = torch.randn(
                latents.shape,
                generator=derive_generator(cfg.seed, "train-noise", step, micro),
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)
            timesteps = torch.randint(
                0,
                adapter.num_train_timesteps,
                (latents.shape[0],),
                generator=derive_generator(cfg.seed, "train-t", step, micro),
            ).to(device)

            noisy = adapter.add_noise(latents, noise, timesteps)
            target = adapter.get_target(latents, noise, timesteps)

            with _autocast(device.type, autocast_dtype):
                prediction = adapter.predict(noisy, timesteps, cond)
                loss = (prediction.float() - target.float()).pow(2).mean()

            scaled = loss / cfg.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += float(loss.detach()) / cfg.gradient_accumulation_steps

        # Clipping has to see true gradients, so undo the fp16 loss scale first.
        if scaler is not None:
            scaler.unscale_(optimizer)
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)  # skipped internally if the unscaled grads are inf/NaN
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        state.step = step
        state.loss = total_loss
        state.learning_rate = float(scheduler.get_last_lr()[0])
        state.elapsed = time.perf_counter() - started

        if cfg.log_every and step % cfg.log_every == 0:
            state.history.append(state.to_dict())
            per_step = state.elapsed / step
            logger.info(
                "step %d/%d  loss=%.5f  lr=%.2e  (%.1fs, %.2fs/step, ETA %s)",
                step, cfg.max_steps, state.loss, state.learning_rate, state.elapsed,
                per_step, _format_duration(per_step * (cfg.max_steps - step)),
            )
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0 and step < cfg.max_steps:
            save_lora(
                unet,
                output_dir / f"lora-step{step:06d}.safetensors",
                metadata={**(metadata or {}), "steps": str(step)},
                cfg=cfg.lora,
            )

        for callback in callbacks:
            callback(state)


def _format_duration(seconds: float) -> str:
    """``H:MM:SS``. Only ever a log-line decoration — the ETA is recoverable from
    ``step`` and ``elapsed``, so it is deliberately absent from the run summary."""
    total = int(max(seconds, 0.0) + 0.5)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _draw_indices(num_items: int, cfg: TrainConfig, step: int, micro: int) -> list[int]:
    """Which dataset items this micro-batch uses. Indices, not items: with the
    latent cache the point is to never touch ``dataset[i]`` at all."""
    generator = derive_generator(cfg.seed, "train-batch", step, micro)
    return torch.randint(0, num_items, (cfg.batch_size,), generator=generator).tolist()


def _draw_flips(count: int, cfg: TrainConfig, step: int, micro: int) -> list[bool]:
    """Which members of this micro-batch are mirrored (``data.hflip``).

    Keyed on ``(seed, step, micro)`` through :func:`derive_generator` rather than
    read off the global RNG, so the decision is the same whether the latents came
    from the cache or from a fresh encode — that equality is what keeps
    ``train.cache_latents`` an optimisation instead of an experimental variable.
    """
    generator = derive_generator(cfg.seed, "train-hflip", step, micro)
    return [bool(v) for v in torch.rand(count, generator=generator).lt(0.5).tolist()]
