"""Inspect an assembled bundle without loading the UNet: metadata, tensor sanity, router gates.

    python scripts/router_stats.py --bundle runs/sdxl_cityscapes/bundles/n1+n4-core-n4/bundle.safetensors
    python scripts/router_stats.py --bundle ... --png            # also writes gate_heatmap.png next to the bundle

Writes ``bundle_stats.json`` next to the bundle. For a routed bundle the gate that
does not depend on the latent, ``g0(t) = fc.bias + E(t)``, is tabulated per
(module, context set) over every t and summarised per interval of the core scheme:
mean / min / max, fraction of negative gates, and the same grouped by UNet block
(down/mid/up) and projection kind (to_q/to_k/to_v/to_out). The latent term
``fc.weight @ mean(z_t)`` is zero at init and is reported through the weight norm only;
the exact per-sample gate needs latents (``TSMAssembly.gate_for``).

Stage 4 acceptance check (docs/GPU_RUNBOOK.md): ``router.bias_moved`` must be true,
i.e. ``fc.bias`` left its initial value ``init_gate``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def _block_of(module: str) -> str:
    m = re.match(r"(down_blocks\.\d+|mid_block|up_blocks\.\d+)", module)
    return m.group(1) if m else module.split(".")[0]


def _kind_of(module: str) -> str:
    for kind in ("to_q", "to_k", "to_v", "to_out"):
        if f".{kind}" in module:
            return kind
    return module.rsplit(".", 1)[-1]


def _summary(x: torch.Tensor) -> dict:
    x = x.float().flatten()
    return {
        "mean": float(x.mean()), "std": float(x.std(unbiased=False)) if x.numel() > 1 else 0.0,
        "min": float(x.min()), "max": float(x.max()), "frac_negative": float((x < 0).float().mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", default=None, help="Default: <bundle dir>/bundle_stats.json")
    parser.add_argument("--png", action="store_true", help="Write gate_heatmap.png (module x t) for routed bundles")
    args = parser.parse_args(argv)

    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(args.bundle)
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}

    sets = json.loads(meta["sets"])
    intervals = {k: [tuple(iv) for iv in v] for k, v in json.loads(meta["intervals"]).items()}
    experts = json.loads(meta["experts"])
    router_meta = json.loads(meta.get("router", "null"))
    disabled = json.loads(meta.get("disabled", "[]"))
    T = int(meta["num_train_timesteps"])

    per_expert: dict[str, int] = {}
    non_finite = []
    for key, tensor in state.items():
        if not torch.isfinite(tensor.float()).all():
            non_finite.append(key)
        if ".experts." in key:
            per_expert.setdefault(key.split(".experts.")[1].split(".")[0], 0)
            per_expert[key.split(".experts.")[1].split(".")[0]] += 1
    report: dict = {
        "bundle": str(path), "format": meta.get("format"), "steps": meta.get("steps"),
        "sets": sets, "core": meta["core"], "disabled": disabled, "intervals": intervals,
        "experts": {e["name"]: {"set": e["set"], "t": [e["t_lo"], e["t_hi"]], "sha": e["selection_sha"][:12],
                                "tensors": per_expert.get(e["name"], 0)} for e in experts},
        "num_tensors": len(state), "non_finite_tensors": non_finite,
        "file_mb": round(path.stat().st_size / 1e6, 1),
    }
    print(f"{path}: format={meta.get('format')} steps={meta.get('steps')} sets={sets} core={meta['core']} "
          f"disabled={disabled} tensors={len(state)} ({report['file_mb']} MB) non_finite={len(non_finite)}")
    for name, info in report["experts"].items():
        print(f"  {name:6s} {info['set']:3s} t=[{info['t'][0]},{info['t'][1]}] sha {info['sha']} tensors {info['tensors']}")

    if router_meta is None:
        report["router"] = None
        print("  hard routing (no router): every active expert rides with weight 1")
    else:
        modules = router_meta["modules"]
        ctx = router_meta["context_sets"]
        M, K = len(modules), len(ctx)
        init = float(router_meta.get("init_gate", 1.0))
        fc_w = state["router.fc.weight"].float()          # (M*K, C)
        fc_b = state["router.fc.bias"].float()            # (M*K,)
        emb = state["router.t_embed.weight"].float()      # (T, M*K)
        bias = fc_b.view(M, K)
        g0 = (fc_b[None, :] + emb).view(T, M, K)          # latent-free gate per t
        core_ivs = intervals[meta["core"]]
        rows = [m for m in modules]
        blocks = sorted({_block_of(m) for m in modules})
        kinds = sorted({_kind_of(m) for m in modules})
        r: dict = {
            "context_sets": ctx, "num_modules": M, "init_gate": init,
            "bias_moved": bool((bias - init).abs().max() > 0),
            "bias_abs_shift_max": float((bias - init).abs().max()),
            "bias_abs_shift_mean": float((bias - init).abs().mean()),
            "fc_weight_frobenius": float(fc_w.norm()),
            "fc_weight_row_norm_max": float(fc_w.norm(dim=1).max()),
            "t_embed_abs_max": float(emb.abs().max()),
            "t_embed_rows_touched": int((emb.abs().sum(dim=1) > 0).sum()),
            "gate_g0_all": _summary(g0),
            "per_set": {},
        }
        for k, set_name in enumerate(ctx):
            gk = g0[:, :, k]                              # (T, M)
            entry = {"bias": _summary(bias[:, k]), "by_core_interval": {}, "by_block": {}, "by_kind": {},
                     "modules_mean_over_t": {m: float(gk[:, i].mean()) for i, m in enumerate(rows)}}
            for lo, hi in core_ivs:
                entry["by_core_interval"][f"[{lo},{hi}]"] = _summary(gk[lo:hi + 1])
            for b in blocks:
                idx = [i for i, m in enumerate(rows) if _block_of(m) == b]
                entry["by_block"][b] = {f"[{lo},{hi}]": float(gk[lo:hi + 1][:, idx].mean()) for lo, hi in core_ivs}
            for kd in kinds:
                idx = [i for i, m in enumerate(rows) if _kind_of(m) == kd]
                entry["by_kind"][kd] = {f"[{lo},{hi}]": float(gk[lo:hi + 1][:, idx].mean()) for lo, hi in core_ivs}
            r["per_set"][set_name] = entry
        report["router"] = r
        print(f"  router: {M} gated modules x {K} context set(s) {ctx}; init_gate {init}; "
              f"bias moved={r['bias_moved']} (max |shift| {r['bias_abs_shift_max']:.4f}, mean {r['bias_abs_shift_mean']:.4f}); "
              f"|fc.W|_F {r['fc_weight_frobenius']:.4f}; t_embed |max| {r['t_embed_abs_max']:.4f} "
              f"({r['t_embed_rows_touched']}/{T} t rows touched)")
        for set_name, entry in r["per_set"].items():
            print(f"  g0 for context set {set_name} (bias + E(t), z-term excluded):")
            for iv, s in entry["by_core_interval"].items():
                print(f"    core {iv:>10s}: mean {s['mean']:+.4f}  min {s['min']:+.4f}  max {s['max']:+.4f}  neg {s['frac_negative']:.2%}")
            print("    by block:", {b: {iv: round(v, 3) for iv, v in d.items()} for b, d in entry["by_block"].items()})
            print("    by kind: ", {k: {iv: round(v, 3) for iv, v in d.items()} for k, d in entry["by_kind"].items()})
        if args.png:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, K, figsize=(6 * K, max(4, M * 0.08)), squeeze=False)
            for k, set_name in enumerate(ctx):
                ax = axes[0, k]
                gk = g0[:, :, k].T.numpy()               # (M, T)
                vmax = max(abs(gk.min() - init), abs(gk.max() - init), 1e-6)
                im = ax.imshow(gk, aspect="auto", cmap="RdBu_r", vmin=init - vmax, vmax=init + vmax,
                               interpolation="nearest")
                for lo, hi in core_ivs[:-1]:
                    ax.axvline(hi + 0.5, color="k", lw=0.5)
                ax.set_title(f"g0 = bias + E(t) for context set {set_name} (steps {meta.get('steps')})")
                ax.set_xlabel("t"); ax.set_ylabel("gated module (router order)")
                fig.colorbar(im, ax=ax, fraction=0.02)
            fig.tight_layout()
            png = path.parent / "gate_heatmap.png"
            fig.savefig(png, dpi=120)
            print(f"  wrote {png}")

    out = Path(args.out) if args.out else path.parent / "bundle_stats.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  wrote {out}")
    if non_finite:
        raise SystemExit(f"{len(non_finite)} non-finite tensor(s): {non_finite[:3]}")


if __name__ == "__main__":
    main()
