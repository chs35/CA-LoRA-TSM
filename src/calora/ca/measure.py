r"""The Concept-Awareness measurement loop.

For one sample and one timestep the procedure is:

1. ``x_t = sqrt(a_t) x_0 + sqrt(1-a_t) eps``
2. one **grad-enabled** forward ``eps_theta(x_t, c, t)``
3. ``L_Diff  = ||eps_theta(x_t,c,t) - target||^2`` -> backward -> ``||g_d||``
4. for each concept axis and each of its ``augs_per_sample`` augmented prompts:
   a **no-grad** forward ``eps_theta(x_t, c_Aug, t)``
   (this is the paper's ``sg[.]``), then
   ``L_Concept = ||eps_theta(x_t,c,t) - sg[eps_theta(x_t,c_Aug,t)]||^2``
   -> backward -> ``g_c``

Only step 2 is a grad-enabled forward, and its graph is reused via
``retain_graph`` for every backward, so the cost per sample is
1 forward+backward plus (n_axes * augs_per_sample) cheap no-grad forwards and
backwards. That asymmetry is why raising ``augs_per_sample`` is a much cheaper
way to buy concept-gradient samples than raising ``num_samples``.

Two properties are deliberate:

* **All concept axes share one sample.** ``L_Diff`` and every ``L_Concept`` are
  computed from the *same* ``x_t`` and the *same* prediction, so the style and
  viewpoint statistics are paired. That is what makes their per-projection
  cosine similarity meaningful.
* **All timesteps share the same draws.** With ``paired_across_timesteps`` the
  noise seed depends on the sample index but not on ``t``, so differences
  between timesteps are not confounded by resampling noise.

Reduction note: the paper writes ``||.||^2`` (a sum), we use ``mean``. CA is a
ratio of two gradient norms that both carry the same constant factor, so the
statistic is unchanged; ``mean`` just keeps fp16/bf16 losses in range.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from calora.ca.stats import ConceptAwarenessAccumulator, MeasurementResult
from calora.models.adapter import Conditioning, DenoiserAdapter
from calora.models.projections import ProjectionTable
from calora.utils.logging import get_logger
from calora.utils.seeding import derive_generator

logger = get_logger(__name__)


@dataclass
class MeasurementSample:
    """One prepared example: latents plus base and augmented conditionings.

    Held on CPU so that a few hundred of them cost a few hundred MB, and so the
    VAE / text encoders can be dropped before the measurement loop starts.
    """

    index: int
    latents: torch.Tensor
    cond: Conditioning
    cond_aug: dict[str, list[Conditioning]]
    """axis -> one conditioning per augmentation slot. Every axis carries the
    same number of slots, because slot ``j`` of one axis is paired with slot
    ``j`` of every other when the cross-axis cosine is taken."""

    caption: str = ""
    augmented: dict[str, list[str]] = field(default_factory=dict)

    @property
    def augs_per_sample(self) -> int:
        return len(next(iter(self.cond_aug.values()))) if self.cond_aug else 0


def _to_device(cond: Conditioning, device: torch.device, dtype: torch.dtype) -> Conditioning:
    return {k: v.to(device=device, dtype=dtype) for k, v in cond.items()}


def _cat_cond(conds: Sequence[Conditioning]) -> Conditioning:
    keys = conds[0].keys()
    return {k: torch.cat([c[k] for c in conds], dim=0) for k in keys}


def _squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error in fp32 regardless of the model dtype."""
    return (pred.float() - target.float()).pow(2).mean()


def _read_grads(table: ProjectionTable) -> dict[str, torch.Tensor]:
    return {name: p.grad for name, p in table.parameters.items() if p.grad is not None}


# --------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------


def prepare_samples(
    adapter: DenoiserAdapter,
    items: Iterable[dict],
    *,
    axes: Sequence[str],
    num_samples: int | None = None,
) -> list[MeasurementSample]:
    """Encode images and prompts once, for reuse across every timestep.

    Args:
        items: dicts with ``image`` ``[3,H,W]`` in ``[-1,1]``, ``caption``, and
            ``augmented``: ``{axis: [prompt, ...]}``. A bare string is accepted
            and read as a single-slot list. See :mod:`calora.data`.
        axes: concept axes to prepare; every item must provide all of them, with
            the same number of augmentation slots for each.
    """
    samples: list[MeasurementSample] = []
    for index, item in enumerate(items):
        if num_samples is not None and index >= num_samples:
            break
        missing = set(axes) - set(item["augmented"])
        if missing:
            raise KeyError(f"Item {index} lacks augmented prompts for axes {sorted(missing)}")

        prompts = {axis: _as_prompt_list(item["augmented"][axis]) for axis in axes}
        slot_counts = {axis: len(values) for axis, values in prompts.items()}
        if len(set(slot_counts.values())) != 1 or min(slot_counts.values()) == 0:
            raise ValueError(
                f"Item {index} has mismatched or empty augmentation slots {slot_counts}; every "
                "axis must supply the same non-zero number of c_Aug prompts."
            )

        image = item["image"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        latents = adapter.encode_images(image).detach().cpu()
        cond = {k: v.detach().cpu() for k, v in adapter.encode_prompts([item["caption"]]).items()}
        cond_aug = {
            axis: [
                {k: v.detach().cpu() for k, v in adapter.encode_prompts([prompt]).items()}
                for prompt in prompts[axis]
            ]
            for axis in axes
        }
        samples.append(
            MeasurementSample(
                index=index,
                latents=latents,
                cond=cond,
                cond_aug=cond_aug,
                caption=item["caption"],
                augmented=prompts,
            )
        )

    if not samples:
        raise ValueError("No samples were prepared; is the dataset empty?")
    counts = {sample.augs_per_sample for sample in samples}
    if len(counts) != 1:
        raise ValueError(
            f"Samples carry differing numbers of augmentation slots {sorted(counts)}; CA would "
            "then be averaged over an item-dependent number of contributions."
        )
    if num_samples is not None and len(samples) < num_samples:
        logger.warning(
            "Requested %d samples but the dataset only yielded %d", num_samples, len(samples)
        )
    return samples


def _as_prompt_list(value: str | Sequence[str]) -> list[str]:
    """Normalise the ``augmented`` contract, which used to be a bare string."""
    return [value] if isinstance(value, str) else list(value)


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def measure_concept_awareness(
    adapter: DenoiserAdapter,
    table: ProjectionTable,
    samples: Sequence[MeasurementSample],
    *,
    timestep: int,
    axes: Sequence[str],
    batch_size: int = 1,
    granularity: str = "matrix",
    accumulate_direction: bool = True,
    compute_paired_cosine: bool = True,
    accum_device: str | torch.device = "cpu",
    eps: float = 1e-12,
    seed: int = 0,
    paired_across_timesteps: bool = True,
    split_half: bool = False,
    log_every: int = 0,
) -> MeasurementResult:
    """Measure CA (and gradient directions) at a single timestep.

    ``batch_size > 1`` changes the estimand from ``E[||g_c||/||g_d||]`` to
    ``E[||sum_i g_c,i|| / ||sum_i g_d,i||]``. It is faster on large GPUs but is
    *not* the paper's quantity; a warning is emitted.

    ``split_half`` routes even- and odd-indexed samples into separate
    accumulators and attaches both to :attr:`MeasurementResult.halves`, so the
    timestep ranking can be checked for sampling stability without measuring
    twice. The split is by *sample index*: the augmentation slots of one sample
    share an ``x_t`` and an ``L_Diff`` gradient, so splitting per contribution
    would put correlated terms on both sides and overstate the agreement.
    """
    if not 0 <= timestep < adapter.num_train_timesteps:
        raise ValueError(
            f"timestep {timestep} outside [0, {adapter.num_train_timesteps}) for this scheduler"
        )
    if batch_size > 1:
        logger.warning(
            "batch_size=%d: CA becomes a ratio of batch-summed gradient norms, "
            "not the per-sample expectation defined in the paper.",
            batch_size,
        )
    if not samples:
        raise ValueError("No samples to measure")

    augs_per_sample = samples[0].augs_per_sample
    for sample in samples:
        for axis in axes:
            if len(sample.cond_aug.get(axis, ())) != augs_per_sample:
                raise ValueError(
                    f"Sample {sample.index} has {len(sample.cond_aug.get(axis, ()))} augmentation "
                    f"slot(s) for axis {axis!r}, expected {augs_per_sample}"
                )

    device, dtype = adapter.device, adapter.dtype

    def new_accumulator() -> ConceptAwarenessAccumulator:
        return ConceptAwarenessAccumulator(
            table,
            axes,
            accumulate_direction=accumulate_direction,
            compute_paired_cosine=compute_paired_cosine,
            granularity=granularity,
            device=accum_device,
            eps=eps,
            augs_per_sample=augs_per_sample,
        )

    accumulators = [new_accumulator() for _ in range(2 if split_half else 1)]
    budget = accumulators[0].estimate_memory_bytes()
    # Only the persistent sums are duplicated by split_half; one sample scope is
    # alive at a time regardless.
    persistent = budget["persistent_accumulators"] * len(accumulators)
    logger.info(
        "t=%d: %d projections (%.2fM params), %d aug(s)/sample, %d accumulator(s); "
        "~%.1f GiB on %s (%.1f persistent + %.1f transient). "
        "Set measure.accum_device=cpu if that is tight.",
        timestep,
        len(table),
        table.total_params / 1e6,
        augs_per_sample,
        len(accumulators),
        (persistent + budget["transient_per_sample"]) / 2**30,
        accumulators[0].device,
        persistent / 2**30,
        budget["transient_per_sample"] / 2**30,
    )

    was_training = adapter.unet.training
    adapter.unet.eval()
    grad_state = {name: p.requires_grad for name, p in adapter.unet.named_parameters()}
    adapter.unet.requires_grad_(False)
    table.requires_grad_(True)

    started = time.perf_counter()
    try:
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            # Routed on the first sample of the chunk: with the mandated
            # batch_size=1 that is the sample, and with batching the halves stay
            # disjoint in whole chunks, which is the property that matters.
            target = accumulators[chunk[0].index % len(accumulators)]
            _measure_chunk(
                adapter,
                table,
                target,
                chunk,
                timestep=timestep,
                axes=axes,
                seed=seed,
                paired_across_timesteps=paired_across_timesteps,
                device=device,
                dtype=dtype,
            )
            if log_every and (start // max(batch_size, 1)) % log_every == 0:
                logger.info(
                    "t=%d  %d/%d samples  (%.2fs)",
                    timestep,
                    min(start + batch_size, len(samples)),
                    len(samples),
                    time.perf_counter() - started,
                )
    finally:
        table.zero_grad()
        for name, param in adapter.unet.named_parameters():
            param.requires_grad_(grad_state[name])
        adapter.unet.train(was_training)

    if split_half:
        empty = [i for i, acc in enumerate(accumulators) if acc.num_items == 0]
        if empty:
            logger.warning(
                "split_half: half(s) %s received no samples; their statistics are all zero.", empty
            )
        # finalize() only reads the sums, so the halves survive the merge below.
        halves = (accumulators[0].finalize(timestep), accumulators[1].finalize(timestep))
        result = accumulators[0].merge_(accumulators[1]).finalize(timestep)
        result.halves = halves
    else:
        result = accumulators[0].finalize(timestep)

    logger.info(
        "t=%d done: %d contributions from %d samples in %.1fs",
        timestep,
        result.num_samples,
        result.num_items,
        time.perf_counter() - started,
    )
    return result


def _measure_chunk(
    adapter: DenoiserAdapter,
    table: ProjectionTable,
    accumulator: ConceptAwarenessAccumulator,
    chunk: Sequence[MeasurementSample],
    *,
    timestep: int,
    axes: Sequence[str],
    seed: int,
    paired_across_timesteps: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    latents = torch.cat([s.latents for s in chunk], dim=0).to(device=device, dtype=dtype)
    cond = _to_device(_cat_cond([s.cond for s in chunk]), device, dtype)
    cond_aug = {
        axis: [
            _to_device(_cat_cond([s.cond_aug[axis][slot] for s in chunk]), device, dtype)
            for slot in range(len(chunk[0].cond_aug[axis]))
        ]
        for axis in axes
    }

    noise = torch.cat(
        [
            torch.randn(
                s.latents.shape,
                generator=derive_generator(
                    seed,
                    "noise",
                    s.index,
                    *(() if paired_across_timesteps else (timestep,)),
                ),
                dtype=torch.float32,
            )
            for s in chunk
        ],
        dim=0,
    ).to(device=device, dtype=dtype)

    timesteps = torch.full((latents.shape[0],), timestep, dtype=torch.long, device=device)
    noisy = adapter.add_noise(latents, noise, timesteps)
    target = adapter.get_target(latents, noise, timesteps)

    table.zero_grad()
    prediction = adapter.predict(noisy, timesteps, cond)

    with accumulator.sample() as scope:
        # Denominator: the standard diffusion loss.
        _squared_error(prediction, target).backward(retain_graph=True)
        scope.set_denominator(_read_grads(table))
        table.zero_grad()

        # Numerators: one per (concept axis, augmentation slot), all reusing the
        # same graph. Only the very last backward may release it.
        total = sum(len(cond_aug[axis]) for axis in axes)
        done = 0
        for axis in axes:
            for cond_slot in cond_aug[axis]:
                with torch.no_grad():
                    augmented_prediction = adapter.predict(noisy, timesteps, cond_slot)
                done += 1
                _squared_error(prediction, augmented_prediction).backward(retain_graph=done < total)
                scope.add_axis(axis, _read_grads(table))
                table.zero_grad()
