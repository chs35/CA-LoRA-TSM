"""Configuration schema and YAML loading.

Configs are plain nested dataclasses. ``load_config`` reads a YAML file, applies
an optional chain of ``_base_`` parents, and recursively instantiates the
dataclasses so that typos in key names fail loudly instead of being ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Which denoiser to load.

    ``kind`` selects the adapter implementation:
      * ``toy``       - offline synthetic UNet (CPU tests, no downloads)
      * ``sd``        - diffusers StableDiffusionPipeline (SD 1.5 / 2.1)
      * ``sdxl``      - diffusers StableDiffusionXLPipeline
    """

    kind: str = "toy"
    name: str = ""
    revision: str | None = None
    variant: str | None = None
    dtype: str = "float32"
    #: The SDXL VAE is numerically unstable in fp16, so it gets its own dtype.
    vae_dtype: str = "float32"
    device: str = "auto"
    resolution: int = 512
    enable_gradient_checkpointing: bool = False
    enable_xformers: bool = False
    # Toy-model-only knobs; ignored by the diffusers adapters.
    toy: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Where images and captions come from."""

    kind: str = "dummy"  # "dummy" | "cityscapes"
    root: str = ""
    split: str = "train"
    caption_mode: str = "template"  # "template" | "file"
    caption_template: str = "a photo of a city street scene in {city}"
    caption_file: str = ""
    augmentation_file: str = "configs/prompt_augmentations.json"
    concept_axes: list[str] = field(default_factory=lambda: ["style", "viewpoint"])
    max_items: int | None = None
    #: "center" | "random". Applies to training only — CA measurement forces
    #: "center" regardless, because its samples must be pairable across
    #: timesteps and concept axes. See :func:`calora.data.build_dataset`.
    crop: str = "center"
    #: Random horizontal flip, the second half of the paper's data augmentation
    #: (random crop + random hflip). Training only, and for the same reason as
    #: ``crop``: a flipped x0 is a different image, so measurement forces
    #: ``hflip=False`` at all three call sites.
    hflip: bool = False
    # dummy-only
    num_dummy_items: int = 8


@dataclass
class MeasureConfig:
    """Concept-Awareness measurement."""

    timesteps: list[int] = field(default_factory=lambda: [1, 81, 201, 481])
    num_samples: int = 256
    # NOTE: CA is defined as an expectation of a *per-sample* ratio of gradient
    # norms. Any batch_size > 1 silently changes the estimand to
    # ||sum_i g_i|| / ||sum_i d_i||, which is not the same quantity. Keep at 1
    # unless you are deliberately running that ablation.
    batch_size: int = 1
    #: How many c_Aug prompts to draw per sample and per axis. The grad-enabled
    #: forward and the L_Diff backward are shared within a sample, so each extra
    #: augmentation buys a concept-gradient sample at roughly the cost of one
    #: no-grad forward plus one backward — far cheaper than another sample.
    augs_per_sample: int = 1
    #: Accumulate even- and odd-indexed samples into separate accumulators so the
    #: timestep ranking can be checked for sampling stability from a single run.
    #: The split is by *sample*: the augmentations of one sample are correlated,
    #: so splitting per contribution would break the independence assumption.
    split_half: bool = True
    granularity: str = "matrix"  # "matrix" | "elementwise"
    accumulate_direction: bool = True
    compute_paired_cosine: bool = True
    save_direction_vectors: bool = False
    accum_device: str = "auto"  # "auto" | "cpu" | "cuda"
    eps: float = 1e-12
    seed: int = 0
    # Reuse the same (x0, noise, caption, c_aug) draws for every timestep so that
    # cross-t comparisons are paired. Strongly recommended.
    paired_across_timesteps: bool = True
    log_every: int = 16
    #: Autocast dtype for the *CA drift* measurement only (the training-loop
    #: callback in :mod:`calora.analysis.stability`), never for scripts/measure_ca.py
    #: — the selection every arm was made from must stay in one numeric regime.
    #: "bf16" roughly halves the snapshot cost; it changes the measured scores
    #: slightly, so it is only sound once the pre-registered equivalence gate of
    #: scripts/check_drift_autocast_gate.py (spearman >= 0.99, top-2% jaccard
    #: >= 0.8) has passed, and then only if applied to every arm.
    drift_autocast: str = "no"  # "no" | "bf16"

    def __post_init__(self) -> None:
        if self.augs_per_sample < 1:
            raise ValueError(f"measure.augs_per_sample must be >= 1, got {self.augs_per_sample}")
        if self.drift_autocast not in ("no", "bf16"):
            raise ValueError(
                f"Unknown measure.drift_autocast {self.drift_autocast!r}; expected 'no' or 'bf16'"
            )


@dataclass
class SelectConfig:
    """Top-k unit selection and multi-timestep aggregation."""

    #: What one selectable unit is. ``head`` (default) is the paper's unit: one
    #: theoretical attention head, i.e. a row-slice of to_q/to_k/to_v or a
    #: column-slice of to_out.0. ``module`` is the coarser fused-``nn.Linear``
    #: unit this project used first; it is kept so the module-unit results
    #: already on disk stay reproducible as a granularity ablation.
    #:
    #: NOTE: unrelated to ``measure.granularity``, which is how the CA *ratio* is
    #: read (norm-of-matrix vs elementwise) at whatever unit is in force. This
    #: one decides what a unit is; that one decides how it is scored.
    granularity: str = "head"  # "head" | "module"
    fraction: float = 0.02
    #: "auto" resolves to a count basis at head granularity and the parameter
    #: basis at module granularity — see
    #: :data:`calora.ca.select.DEFAULT_BUDGET_BASIS`. Resolved in place at load
    #: time so the run's saved config.yaml records what actually ran.
    # "auto" | "unit_count" | "attn_params" | "unet_params" | "projection_count"
    budget_basis: str = "auto"
    #: Which axis's CA ranks the projections. A concrete axis name, or "mean"/"min"/"max"
    #: to combine axes. The paper's target concept is the domain/style axis.
    axis: str = "style"
    normalization: str = "sum"  # "zscore" | "rank" | "sum" | "max" | "none"
    aggregation: str = "normalized_mean"  # "normalized_mean" | "rank_mean" | "single"
    single_timestep: int | None = None
    top_timesteps: int = 5
    # t-score S(t) = sum_theta CA_norm(theta,t) * (1 - sim(theta,t))
    similarity_source: str = "cos_of_means"  # "cos_of_means" | "mean_of_cos"
    tscore_ca_weighting: str = "mean"  # how to combine the two axes' CA into weights
    tscore_ca_normalization: str = "sum"
    #: Quantiles (over the pooled similarities of every timestep) below which a
    #: projection counts as "directionally conflicting". The whole list is swept
    #: on every run: if the timestep ranking flips between quantiles then
    #: conflict_count is too fragile to be a headline metric, and that is only
    #: visible if the sweep is always reported.
    conflict_sim_quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    #: Which member of the sweep supplies the scalar ``conflict_count`` used for
    #: ranking and logging. Must appear in ``conflict_sim_quantiles``.
    conflict_primary_quantile: float = 0.3
    #: n in "top-n·k". The weighted-similarity metric is read over the top
    #: ``fraction * n`` of the budget rather than the top-k that would actually be
    #: trained, so it does not depend on exactly where the selection boundary
    #: falls; CA weighting keeps the extra projections from dominating.
    tscore_selection_multiplier: int = 5

    def __post_init__(self) -> None:
        # Imported here rather than at module scope: calora.ca.select pulls in
        # torch, and config loading must stay cheap for CLI arg parsing.
        from calora.ca.select import BUDGET_BASES, DEFAULT_BUDGET_BASIS

        if self.granularity not in DEFAULT_BUDGET_BASIS:
            raise ValueError(
                f"select.granularity must be one of {sorted(DEFAULT_BUDGET_BASIS)}, "
                f"got {self.granularity!r}"
            )
        if self.budget_basis == "auto":
            self.budget_basis = DEFAULT_BUDGET_BASIS[self.granularity]
        elif self.budget_basis not in BUDGET_BASES:
            raise ValueError(
                f"select.budget_basis must be 'auto' or one of {list(BUDGET_BASES)}, "
                f"got {self.budget_basis!r}"
            )
        if not self.conflict_sim_quantiles:
            raise ValueError("select.conflict_sim_quantiles must not be empty")
        outside = [q for q in self.conflict_sim_quantiles if not 0.0 <= q <= 1.0]
        if outside:
            raise ValueError(f"select.conflict_sim_quantiles must lie in [0, 1]; got {outside}")
        if self.conflict_primary_quantile not in self.conflict_sim_quantiles:
            raise ValueError(
                f"select.conflict_primary_quantile={self.conflict_primary_quantile} must be one "
                f"of select.conflict_sim_quantiles={self.conflict_sim_quantiles}"
            )
        if self.tscore_selection_multiplier < 1:
            raise ValueError(
                "select.tscore_selection_multiplier must be >= 1, got "
                f"{self.tscore_selection_multiplier}"
            )


@dataclass
class LoraConfig_:
    """LoRA hyperparameters. The trailing underscore avoids a name collision with
    ``peft.LoraConfig``, which :mod:`calora.train.lora` imports alongside this."""

    rank: int = 64
    alpha: int = 64
    dropout: float = 0.0
    init_lora_weights: str = "gaussian"


@dataclass
class TrainConfig:
    lora: LoraConfig_ = field(default_factory=LoraConfig_)
    max_steps: int = 10_000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"  # "no" | "fp16" | "bf16"; "fp16" requires CUDA
    #: Encode every image's VAE latent once at startup instead of once per step.
    #: Silently impossible when data.crop="random" (the pixels change per read),
    #: in which case training warns and falls back to per-step encoding.
    cache_latents: bool = True
    #: Read the next micro-batch's images on a background thread while the GPU
    #: works on the current one. Only does anything when the latent cache is off
    #: (data.crop="random"), where each step pays a 2048x1024 PNG decode. Purely
    #: an overlap: the reads stay single-threaded and in the same order, so the
    #: numbers are unchanged (tests/test_train_and_stability.py).
    prefetch: bool = True
    seed: int = 0
    checkpoint_every: int = 2_000
    log_every: int = 50
    # CA drift measurement piggy-backed on the training loop.
    stability_every: int = 0  # 0 disables
    stability_num_samples: int = 64


@dataclass
class EvalConfig:
    """Sampling and scoring protocol for the generated-image evaluation.

    Everything the paper states is quoted from Table 12 (SDXL fine-tuning
    hyperparameters) and the weather appendix; everything it does not state is
    marked as our decision so the report can cite it as such rather than
    implying the paper fixed it.
    """

    #: In-domain images per arm, and therefore the size of the CMMD comparison
    #: set. The paper's few-shot setting generates 500.
    num_images: int = 500
    guidance_scale: float = 5.0  # Table 12
    num_inference_steps: int = 25  # Table 12
    #: The fixed training prompt of Table 12. Also what ``c_Aug`` is a rewrite of
    #: (see configs/prompt_augmentations.json), so it must stay in step with
    #: ``data.caption_template``.
    prompt: str = "photorealistic first-person urban street view"
    #: Append ", with [Class names]" read off a real gtFine label map (Table 12).
    include_class_names: bool = True
    #: The paper evaluates CLIP-Score under "four diverse weather conditions".
    weather_conditions: list[str] = field(
        default_factory=lambda: ["foggy", "night-time", "rainy", "snowy"]
    )
    images_per_condition: int = 100  # paper: 100 generated images per condition
    #: CMMD reference set. NOT STATED IN THE PAPER — our decision: Cityscapes
    #: val, which is exactly 500 images and disjoint from the training split.
    reference_split: str = "val"
    reference_num_images: int = 500
    #: NOT STATED IN THE PAPER — our decision. CMMD is defined on CLIP ViT-L/14
    #: in Jayasumana et al., and CLIP-Score shares the embedder so the two
    #: numbers stay in one feature space.
    clip_model: str = "openai/clip-vit-large-patch14"
    generation_batch_size: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_images < 1:
            raise ValueError(f"eval.num_images must be >= 1, got {self.num_images}")
        if not self.weather_conditions:
            raise ValueError(
                "eval.weather_conditions must not be empty; the paper's CLIP-Score "
                "protocol is defined over weather-conditioned prompts"
            )
        duplicates = sorted({c for c in self.weather_conditions
                             if self.weather_conditions.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"eval.weather_conditions contains duplicates {duplicates}; each condition "
                "is a separate CLIP-Score group and would otherwise be double-counted"
            )
        if self.images_per_condition < 1:
            raise ValueError(
                f"eval.images_per_condition must be >= 1, got {self.images_per_condition}"
            )
        if self.reference_num_images < 1:
            raise ValueError(
                f"eval.reference_num_images must be >= 1, got {self.reference_num_images}"
            )
        if self.num_inference_steps < 1:
            raise ValueError(
                f"eval.num_inference_steps must be >= 1, got {self.num_inference_steps}"
            )
        if self.generation_batch_size < 1:
            raise ValueError(
                f"eval.generation_batch_size must be >= 1, got {self.generation_batch_size}"
            )


@dataclass
class Config:
    name: str = "unnamed"
    output_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) is not None and type(None) in get_args(annotation)


def _strip_optional(annotation: Any) -> Any:
    if _is_optional(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def build_dataclass(cls: type, data: dict[str, Any], path: str = "") -> Any:
    """Recursively instantiate ``cls`` from ``data``, rejecting unknown keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"Unknown config key(s) at {where}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        annotation = _strip_optional(hints[key])
        child_path = f"{path}.{key}" if path else key
        if is_dataclass(annotation) and isinstance(value, dict):
            kwargs[key] = build_dataclass(annotation, value, child_path)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a ``_base_`` parent chain relative to the file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(load_config_dict(base_path), raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    data = load_config_dict(path)
    if overrides:
        data = _deep_merge(data, overrides)
    return build_dataclass(Config, data)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False, allow_unicode=True)


def parse_cli_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ``["measure.num_samples=32", "model.device=cpu"]`` into a nested dict."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out
