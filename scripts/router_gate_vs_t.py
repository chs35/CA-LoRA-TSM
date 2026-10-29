"""Router gate as a function of t on held-out images (README "Router gate vs t").

    python scripts/router_gate_vs_t.py \
        --bundle runs/sdxl_cityscapes/bundles/n1+n4-core-n4/bundle.safetensors --arm no3

The router emits ``g[m, k](z_t, t) = F(z_t)[m, k] + E(t)[m, k]`` with
``F(z) = fc(mean_hw z) `` (bias included) and ``E(t) = t_embed[t]``. This script
encodes ``--num-images`` Cityscapes *val* frames (never seen in fostering or
assembling, which use the train split) with the SDXL VAE, noises each with one fixed
ε at every t in ``[0, T)`` exactly as ``train_router`` does
(``x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε``), and forwards the router. It records
E(t) and F(z_t) separately and the gate ``E(t) + mean_z F(z_t)``.

Raw gates are not comparable across modules, because the experts' ΔW differ in size,
so every module's gate is also weighted by the Frobenius norm of the active expert's
ΔW = (α/r)·B·A at that t: ``g·‖ΔW‖``, averaged over the gated modules, one curve per
expert set (the core set enters with weight 1). Only the VAE and the bundle are
loaded, not the UNet, so this runs on CPU (≈30 s per image to encode on 2 cores).

Writes ``results/router_gate/<arm>.json`` and ``figures/router_gate_vs_t.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from calora.data.cityscapes import scan_cityscapes  # noqa: E402
from calora.data.transforms import load_image  # noqa: E402
from catsm.models.router import Router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def load_bundle(path: Path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata()
    return state, {k: (json.loads(v) if k not in ("core", "format", "axis", "steps") else v) for k, v in meta.items()}


def build_router(state: dict, meta: dict) -> Router:
    r = meta["router"]
    router = Router(r["context_sets"], r["modules"], latent_channels=r["latent_channels"],
                    num_train_timesteps=r["num_train_timesteps"], init_gate=r["init_gate"])
    router.load_state_dict({k.removeprefix("router."): v for k, v in state.items() if k.startswith("router.")})
    return router.eval()


def delta_norms(state: dict, meta: dict, modules: list[str]) -> dict[str, torch.Tensor]:
    """``‖(α/r) B A‖_F`` per (expert, module) in ``modules`` order; 0 where the expert has no heads."""
    out = {}
    for e in meta["experts"]:
        scale = float(e["alpha"]) / float(e["rank"])
        norms = []
        for m in modules:
            a = state.get(f"{m}.experts.{e['name']}.lora_A")
            b = state.get(f"{m}.experts.{e['name']}.lora_B")
            norms.append(0.0 if a is None else float((b.double() @ a.double()).norm()) * scale)
        out[e["name"]] = torch.tensor(norms, dtype=torch.float64)
    return out


def active_expert(meta: dict, set_name: str, t: int) -> str:
    for e in meta["experts"]:
        if e["set"] == set_name and e["t_lo"] <= t <= e["t_hi"]:
            return e["name"]
    raise KeyError(f"no expert of {set_name} covers t={t}")


@torch.no_grad()
def encode(paths: list[str], resolution: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from diffusers import AutoencoderKL, DDPMScheduler

    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
    scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    latents = []
    for i, p in enumerate(paths):
        img = load_image(p, resolution, crop="center").unsqueeze(0)
        # Same as calora's encode_images: mode of the posterior × scaling factor.
        latents.append(vae.encode(img).latent_dist.mode() * float(vae.config.scaling_factor))
        print(f"  encoded {i + 1}/{len(paths)} {Path(p).name}", flush=True)
    info = {"vae_scaling_factor": float(vae.config.scaling_factor), "scheduler": type(scheduler).__name__}
    return torch.cat(latents), scheduler.alphas_cumprod.float(), info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--data-root", default="/workspace/data/cityscapes")
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-png", default=str(ROOT / "figures/router_gate_vs_t.png"))
    args = parser.parse_args(argv)

    state, meta = load_bundle(Path(args.bundle))
    if meta.get("router") is None:
        raise SystemExit(f"{args.bundle} is hard-routed: no router to analyse")
    router = build_router(state, meta)
    modules = list(router.modules_gated)
    ctx = list(router.context_sets)
    core = meta["core"]
    T = router.num_train_timesteps

    entries = scan_cityscapes(args.data_root, "val")
    stride = max(1, len(entries) // args.num_images)
    picked = [entries[i] for i in range(0, len(entries), stride)][: args.num_images]
    print(f"{args.arm}: {len(modules)} gated modules, context {ctx}, core {core}; "
          f"{len(picked)} val images (every {stride}th of {len(entries)})", flush=True)
    x0, alphas, enc_info = encode([str(e.image_path) for e in picked], args.resolution)

    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(x0.shape, generator=g, dtype=torch.float32)
    ts = torch.arange(T)
    E = router.t_embed.weight.detach().double().view(T, len(modules), len(ctx))      # (T, M, K)
    F = torch.empty(T, len(picked), len(modules), len(ctx), dtype=torch.float64)      # (T, N, M, K)
    with torch.no_grad():
        for t in range(T):
            a = alphas[t]
            z_t = a.sqrt() * x0 + (1 - a).sqrt() * noise                               # (N, C, h, w)
            tt = torch.full((len(picked),), t, dtype=torch.long)
            F[t] = router(z_t, tt).double() - E[t].unsqueeze(0)                        # (F + E) - E
    gate = E + F.mean(dim=1)                                                           # (T, M, K)

    norms = delta_norms(state, meta, modules)
    sets = ctx + [core]
    weighted = {s: [] for s in sets}
    raw = {s: [] for s in sets}
    for t in ts.tolist():
        for k, s in enumerate(sets):
            n = norms[active_expert(meta, s, t)]
            gk = gate[t, :, k] if s in ctx else torch.ones(len(modules), dtype=torch.float64)
            weighted[s].append(float((gk * n).mean()))
            raw[s].append(float(gk.mean()))

    intervals = {s: sorted({(e["t_lo"], e["t_hi"]) for e in meta["experts"] if e["set"] == s}) for s in sets}

    def by_interval(series: list[float], s: str) -> dict:
        return {f"[{lo},{hi}]": sum(series[lo:hi + 1]) / (hi - lo + 1) for lo, hi in intervals[core]}

    report = {
        "arm": args.arm, "bundle": args.bundle, "context_sets": ctx, "core": core,
        "num_gated_modules": len(modules), "num_images": len(picked), "split": "val",
        "images": [e.key for e in picked], "noise_seed": args.seed, "resolution": args.resolution, **enc_info,
        "definition": {
            "E(t)": "router.t_embed[t] (per module, per context set)",
            "F(z_t)": "router.fc(mean_hw z_t), bias included; z_t from one fixed noise per image",
            "gate": "E(t) + mean over images of F(z_t)",
            "weighted": "mean over gated modules of gate * ||(alpha/r) B A||_F of the expert active at t; core gate = 1",
        },
        "per_set": {},
    }
    for k, s in enumerate(ctx):
        f_img = F[:, :, :, k].mean(dim=2)                                              # (T, N) module mean
        report["per_set"][s] = {
            "E_mean_over_modules": E[:, :, k].mean(dim=1).tolist(),
            "F_mean_over_modules_and_images": f_img.mean(dim=1).tolist(),
            "F_std_over_images": f_img.std(dim=1).tolist(),
            "gate_mean_over_modules": raw[s],
            "gate_module_p10": gate[:, :, k].quantile(0.1, dim=1).tolist(),
            "gate_module_p90": gate[:, :, k].quantile(0.9, dim=1).tolist(),
            "weighted_mean": weighted[s],
            "gate_by_core_interval": by_interval(raw[s], s),
            "weighted_by_core_interval": by_interval(weighted[s], s),
            "E_abs_max": float(E[:, :, k].abs().max()),
            "gate_t0_vs_t999": [raw[s][0], raw[s][-1]],
        }
    report["per_set"][core] = {"weighted_mean": weighted[core],
                               "weighted_by_core_interval": by_interval(weighted[core], core)}

    out_json = Path(args.out_json or ROOT / f"results/router_gate/{args.arm}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_json}")
    for s in ctx:
        print(f"  {s}: gate by core interval {({k: round(v, 4) for k, v in report['per_set'][s]['gate_by_core_interval'].items()})}")
        print(f"  {s}: g*|dW| by core interval {({k: round(v, 4) for k, v in report['per_set'][s]['weighted_by_core_interval'].items()})}")
    print(f"  core {core}: |dW| by interval {({k: round(v, 4) for k, v in report['per_set'][core]['weighted_by_core_interval'].items()})}")
    plot(report, intervals[core], Path(args.out_png))


def plot(report: dict, core_ivs: list[tuple[int, int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blues = plt.get_cmap("Blues")
    t = list(range(len(report["per_set"][report["core"]]["weighted_mean"])))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    for k, s in enumerate(report["context_sets"]):
        d = report["per_set"][s]
        a.fill_between(t, d["gate_module_p10"], d["gate_module_p90"], color=blues(0.25), lw=0,
                       label=f"{s}: gate, 10–90% over modules")
        a.plot(t, d["gate_mean_over_modules"], color=blues(0.9), lw=2, label=f"{s}: E(t) + mean_z F(z_t)")
        a.plot(t, d["F_mean_over_modules_and_images"], color=blues(0.6), lw=1.2, ls="--", label=f"{s}: mean_z F(z_t)")
        a.plot(t, d["E_mean_over_modules"], color="#555555", lw=1.2, ls=":", label=f"{s}: E(t)")
    a.axhline(1.0, color="#999999", lw=0.8)
    a.set_title(f"{report['arm']}: raw context gate (module mean)", loc="left", fontsize=11)
    a.set_xlabel("t"); a.set_ylabel("gate")
    a.legend(frameon=False, fontsize=8)
    shades = [0.9, 0.55, 0.35]
    for k, s in enumerate(report["context_sets"] + [report["core"]]):
        role = "core (gate 1)" if s == report["core"] else "context"
        b.plot(t, report["per_set"][s]["weighted_mean"], color=blues(shades[k % len(shades)]) if s != report["core"] else "#333333",
               lw=2 if s != report["core"] else 1.2, ls="-" if s != report["core"] else "--", label=f"{s} {role}")
    b.set_title("gate × ‖ΔW‖_F of the active expert (module mean)", loc="left", fontsize=11)
    b.set_xlabel("t"); b.set_ylabel("g · ‖ΔW‖")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        for lo, _ in core_ivs[1:]:
            ax.axvline(lo - 0.5, color="#bbbbbb", lw=0.7, ls="-")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Router gate vs t on {report['num_images']} held-out val images (vertical lines: core-set interval boundaries)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
