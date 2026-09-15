"""Re-use a fostered expert for another expert with the same interval and selection.

    python scripts/share_expert.py --config configs/sdxl_cityscapes.yaml --from n4_i2 --to n3_i1 n5_i3

With halving boundaries several experts of different schemes coincide (same [t_lo, t_hi],
same representative t, hence the same selection sha): n4_i1=n5_i2, n3_i1=n4_i2=n5_i3,
n3_i2=n4_i3=n5_i4. Training them separately would only add GPU non-determinism (user
decision 2026-09-15: train once, copy the weights, re-stamp the metadata). The copy keeps
every tensor bit-identical and rewrites only ``expert`` / ``interval_scheme`` in the
safetensors header (plus ``shared_from``), so ``load_expert_into`` accepts it under the
new name. Refuses when the target's selection sha or interval differs from the source.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from catsm import cli as catsm_cli
from catsm.intervals import build_scheme, parse_expert_name
from calora.models.loader import load_adapter  # noqa: F401  (not used; T comes from the file)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    catsm_cli.add_base_args(ap)
    ap.add_argument("--from", dest="src", required=True); ap.add_argument("--to", nargs="+", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing target expert dir")
    a = ap.parse_args()
    cfg = catsm_cli.resolve(a)
    experts = catsm_cli.experts_dir(cfg)
    src_dir = experts / a.src; src_file = src_dir / "lora-final.safetensors"
    state = load_file(str(src_file))
    with safe_open(str(src_file), "pt") as f:
        meta = dict(f.metadata() or {})
    T = int(meta["num_train_timesteps"])
    src_iv = (int(meta["t_lo"]), int(meta["t_hi"]))
    for dst in a.to:
        set_name, idx = parse_expert_name(dst)
        iv = build_scheme(set_name, T, cfg.intervals).intervals[idx]
        sel_path = catsm_cli.selections_dir(cfg, set_name) / f"{dst}.json"
        sel = json.loads(sel_path.read_text())
        sel_sha = sel["meta"]["selection_sha"]
        if (iv.lo, iv.hi) != src_iv:
            sys.exit(f"{dst}: interval [{iv.lo},{iv.hi}] != source [{src_iv[0]},{src_iv[1]}]")
        if sel_sha != meta["selection_sha"]:
            sys.exit(f"{dst}: selection sha {sel_sha[:12]} != source {meta['selection_sha'][:12]}")
        dst_dir = experts / dst
        if dst_dir.exists() and not a.force:
            sys.exit(f"{dst_dir} exists; pass --force to overwrite")
        dst_dir.mkdir(parents=True, exist_ok=True)
        new_meta = dict(meta, expert=dst, interval_scheme=set_name,
                        shared_from=a.src, shared_from_sha256=sha256(src_file))
        save_file(state, str(dst_dir / "lora-final.safetensors"), metadata=new_meta)
        summary = json.loads((src_dir / "train_summary.json").read_text())
        summary["expert"] = dst; summary["metadata"].update(expert=dst, interval_scheme=set_name)
        summary["shared_from"] = a.src
        (dst_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        shutil.copy(sel_path, dst_dir / "selection.json")
        (dst_dir / "SHARED_FROM.json").write_text(json.dumps({
            "shared_from": a.src, "source_file": str(src_file), "source_sha256": new_meta["shared_from_sha256"],
            "reason": "same interval and selection sha; weights copied bit-identical, metadata re-stamped (user decision 2026-09-15)",
        }, indent=2))
        # verify the copy reads back identically
        back = load_file(str(dst_dir / "lora-final.safetensors"))
        assert all((back[k] == state[k]).all() for k in state), "copy mismatch"
        print(f"{a.src} -> {dst}: {len(state)} tensors bit-identical, interval [{iv.lo},{iv.hi}], sha {sel_sha[:12]}")

if __name__ == "__main__":
    main()
