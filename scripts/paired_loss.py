"""Paired L_Diff of the base UNet vs. a fostered expert on the same (image, t, noise) triples.

    python scripts/paired_loss.py --config configs/sdxl_cityscapes.yaml \
        --expert-file runs/sdxl_cityscapes/experts/n4_i0/lora-final.safetensors --num 96

The training loss is a single batch-1 value per step and is dominated by which image
and which t were drawn, so it cannot show whether the LoRA learned anything. This
script holds the draws fixed, evaluates base and expert on exactly the same inputs,
and reports the mean paired difference with its standard error. Interval defaults to
the expert's own [t_lo, t_hi]. Eval only (no grad); ~13 GB VRAM next to a training job.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from safetensors import safe_open
from calora import cli
from catsm import cli as catsm_cli
from calora.data import build_dataset
from calora.models.loader import load_adapter
from calora.train.lora import load_lora
from calora.utils.logging import get_logger

logger = get_logger("paired_loss")

def main():
    p = cli.base_parser(__doc__ or "")
    p.add_argument("--expert-file", required=True)
    p.add_argument("--num", type=int, default=96)
    p.add_argument("--t-lo", type=int, default=None); p.add_argument("--t-hi", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = catsm_cli.resolve(a)
    with safe_open(a.expert_file, "pt") as f:
        meta = f.metadata() or {}
    t_lo = a.t_lo if a.t_lo is not None else int(meta.get("t_lo", 0))
    t_hi = a.t_hi if a.t_hi is not None else int(meta.get("t_hi", 999))
    adapter = load_adapter(cfg.model)
    dev, dt = adapter.device, adapter.dtype
    ds = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randint(0, len(ds), (a.num,), generator=g).tolist()
    ts = torch.randint(t_lo, t_hi + 1, (a.num,), generator=g)
    logger.info("expert %s [%d,%d]; %d samples from %d items", meta.get("expert"), t_lo, t_hi, a.num, len(ds))
    # encode once
    lat, cond, noise = [], [], []
    with torch.no_grad():
        for i in idx:
            item = ds[i]
            lat.append(adapter.encode_images(item["image"][None]).to(dev, dt))
            cond.append({k: v.to(dev, dt) for k, v in adapter.encode_prompts([item["caption"]]).items()})
            noise.append(torch.randn(lat[-1].shape, generator=g, dtype=torch.float32).to(dev, dt))
    offload = getattr(adapter, "offload_encoders", None)
    if callable(offload):
        offload()
    unet = adapter.unet; unet.eval()

    def evaluate():
        out = []
        with torch.no_grad():
            for l, c, n, t in zip(lat, cond, noise, ts):
                tt = t.view(1).to(dev)
                pred = adapter.predict(adapter.add_noise(l, n, tt), tt, c)
                out.append((pred.float() - adapter.get_target(l, n, tt).float()).pow(2).mean().item())
        return torch.tensor(out)
    t0 = time.perf_counter(); base = evaluate(); logger.info("base done in %.0fs", time.perf_counter() - t0)
    load_lora(unet, a.expert_file, cfg.train.lora, expect_granularity="head")
    unet.eval()
    t0 = time.perf_counter(); ex = evaluate(); logger.info("expert done in %.0fs", time.perf_counter() - t0)
    d = ex - base
    res = {
        "expert_file": a.expert_file, "expert": meta.get("expert"), "interval": [t_lo, t_hi], "num": a.num,
        "base_mean": base.mean().item(), "expert_mean": ex.mean().item(),
        "paired_delta_mean": d.mean().item(), "paired_delta_sem": (d.std() / a.num ** 0.5).item(),
        "relative_change": (d.mean() / base.mean()).item(),
        "frac_improved": (d < 0).float().mean().item(),
        "by_t": sorted([(int(t), b, e) for t, b, e in zip(ts, base.tolist(), ex.tolist())]),
    }
    logger.info("base %.4f  expert %.4f  delta %.4f ± %.4f (%.1f%%), improved on %.0f%% of samples",
                res["base_mean"], res["expert_mean"], res["paired_delta_mean"], res["paired_delta_sem"],
                100 * res["relative_change"], 100 * res["frac_improved"])
    out = Path(a.out or Path(a.expert_file).parent / "paired_loss.json")
    out.write_text(json.dumps(res, indent=2)); logger.info("wrote %s", out)

if __name__ == "__main__":
    main()
