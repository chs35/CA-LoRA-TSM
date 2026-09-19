#!/usr/bin/env bash
# Stage 5 (evaluation) queue: generate 900 images per configuration, then CMMD + CLIP-Score.
#   scripts/stage5_eval_queue.sh no1 no2 no3 c1               # main claim (docs/GPU_RUNBOOK.md §5)
#   SMOKE=1 scripts/stage5_eval_queue.sh no1                  # 2 in-domain + 4 weather images, arm <name>_smoke
#   SEED=1 scripts/stage5_eval_queue.sh no3 no1               # generation-seed replicate, arm <name>_s1
#   scripts/stage5_eval_queue.sh base                         # base SDXL control (no adapter), 900 images
# Names as in scripts/stage4_assemble_queue.sh; the bundle must exist. Generation resumes with
# --skip-existing. After generation the manifest must list 900 images (6 for SMOKE) and, for a
# bundle arm, gate_hook_calls > 0; evaluate.py then writes <run>/<arm>/eval.json.
# The C1 run reuses the CMMD reference export of sdxl_cityscapes through a symlink.
# Launch: nohup scripts/stage5_eval_queue.sh no1 no2 no3 c1 > runs/sdxl_cityscapes/logs/eval_queue.out 2>&1 &
# PID via pgrep -f stage5_eval_queue.sh. Do not run alongside fostering/assembling.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
CFG=configs/sdxl_cityscapes.yaml
now() { TZ=Asia/Seoul date '+%F %T KST'; }
SMOKE=${SMOKE:-0}; SEED=${SEED:-0}

bundle_of() {  # name -> "run|bundle path" (base -> "sdxl_cityscapes|")
  case "$1" in
    base) echo "sdxl_cityscapes|" ;;
    no1) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1-core-n1/bundle.safetensors" ;;
    no2) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n4-core-n4/bundle.safetensors" ;;
    c1)  echo "sdxl_c1|runs/sdxl_c1/bundles/n4-core-n4/bundle.safetensors" ;;
    no3) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n4-core-n4/bundle.safetensors" ;;
    no4) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n3+n4-core-n4/bundle.safetensors" ;;
    no5) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n4+n5-core-n4/bundle.safetensors" ;;
    no6) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n3+n4+n5-core-n4/bundle.safetensors" ;;
    no7) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n4+n5-core-n5/bundle.safetensors" ;;
    no8) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n3+n4+n5-core-n4/bundle.safetensors" ;;
    no9) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n3+n4+n5-core-n5/bundle.safetensors" ;;
    c2|c2n3) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n1+n4-core-n4-disable-n4_i3/bundle.safetensors" ;;
    c2n2) echo "sdxl_cityscapes|runs/sdxl_cityscapes/bundles/n4-core-n4-disable-n4_i3/bundle.safetensors" ;;
    *)   return 1 ;;
  esac
}

# One CMMD reference export (Cityscapes val 500, centre crop) serves both run dirs.
mkdir -p runs/sdxl_c1
[ -e runs/sdxl_c1/reference ] || ln -s ../sdxl_cityscapes/reference runs/sdxl_c1/reference

for name in "$@"; do
  line=$(bundle_of "$name") || { echo "[$(now)] unknown configuration '$name'"; exit 2; }
  IFS='|' read -r run bundle <<< "$line"
  arm=$name; sets=(); expect=900
  [ "$SEED" != "0" ] && { arm="${arm}_s$SEED"; sets+=("eval.seed=$SEED"); }
  [ "$SMOKE" = "1" ] && { arm="${arm}_smoke"; sets+=("eval.num_images=2" "eval.images_per_condition=1"); expect=6; }
  src=(); [ -n "$bundle" ] && { [ -f "$bundle" ] || { echo "[$(now)] $name: $bundle missing (run stage 4 first)"; exit 1; }; src=(--bundle "$bundle"); }
  setargs=(); [ ${#sets[@]} -gt 0 ] && setargs=(--set "${sets[@]}")
  mkdir -p runs/$run/logs
  manifest=runs/$run/$arm/generated/manifest.json

  echo "[$(now)] start $name -> $run/$arm generate ${src[*]:-(base model)} ${setargs[*]}"
  python scripts/generate_images.py --config $CFG --name $run "${src[@]}" --run-name "$arm" --skip-existing "${setargs[@]}" \
    > runs/$run/logs/gen_$arm.out 2>&1
  rc=$?; echo "[$(now)] end $name generate rc=$rc"
  [ $rc -ne 0 ] && { tail -5 runs/$run/logs/gen_$arm.out; echo "QUEUE ABORTED at $name (generate)"; exit $rc; }
  python - "$manifest" "$expect" "${bundle:-}" <<'PY' || { echo "QUEUE ABORTED at $name (manifest check)"; exit 1; }
import json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); expect = int(sys.argv[2]); bundle = sys.argv[3]
imgs = [r for r in m["images"] if "path" in r]
run_dir = Path(sys.argv[1]).parents[2]
missing = [r["path"] for r in imgs if not (run_dir / r["path"]).exists()]
print(f"manifest: {len(imgs)} images, gate_hook_calls={m['gate_hook_calls']}, missing files={len(missing)}")
assert len(imgs) == expect, (len(imgs), expect)
assert not missing, missing[:3]
if bundle:
    assert m["gate_hook_calls"] > 0, "bundle arm without gate hook calls"
PY

  echo "[$(now)] start $name evaluate"
  python scripts/evaluate.py --config $CFG --name $run --run-name "$arm" "${setargs[@]}" > runs/$run/logs/eval_$arm.out 2>&1
  rc=$?; echo "[$(now)] end $name evaluate rc=$rc"
  [ $rc -ne 0 ] && { tail -5 runs/$run/logs/eval_$arm.out; echo "QUEUE ABORTED at $name (evaluate)"; exit $rc; }
  grep -E "CMMD|CLIP-Score\[" runs/$run/logs/eval_$arm.out | sed 's/^/    /'
done
echo "[$(now)] QUEUE DONE"
