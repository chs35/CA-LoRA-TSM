#!/usr/bin/env bash
# Stage 4 (assembling) queue: builds bundles one after another by configuration name.
#   scripts/stage4_assemble_queue.sh no1 no2 c1 no3            # main claim (docs/GPU_RUNBOOK.md §4)
#   scripts/stage4_assemble_queue.sh no4 no5 no6 no7 no8 no9   # every other m>1 scenario (user decision 2026-09-16)
# Names (design §5.1): no1 = n1 (CA-LoRA 4K baseline, hard), no2 = n4 hard, c1 = C1 (run sdxl_c1, n4 hard),
#   no3 = n1+n4, no4 = n1+n3+n4, no5 = n1+n4+n5, no6 = n3+n4+n5, no8 = n1+n3+n4+n5 (core n4),
#   no7 = n1+n4+n5 core n5, no9 = n1+n3+n4+n5 core n5,
#   c2 = no3 with n4_i3 disabled (router retrained), c2n2 = no2 with n4_i3 disabled (hard, no training) — C2 controls (2026-09-19).
# Skips a configuration whose bundle.safetensors exists. Runs scripts/router_stats.py on every bundle
# (bundle_stats.json + gate_heatmap.png) and aborts if the router bias did not move.
# Launch: nohup scripts/stage4_assemble_queue.sh no1 no2 c1 no3 > runs/sdxl_cityscapes/logs/assemble_queue.out 2>&1 &
# PID via pgrep -f stage4_assemble_queue.sh (never $! on an && chain). Never run two GPU jobs at once.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
CFG=configs/sdxl_cityscapes.yaml
now() { TZ=Asia/Seoul date '+%F %T KST'; }

spec() {  # name -> "run|bundle-dir|assemble args"
  case "$1" in
    no1) echo "sdxl_cityscapes|n1-core-n1|--sets n1 --core n1 --no-router" ;;
    no2) echo "sdxl_cityscapes|n4-core-n4|--sets n4 --core n4 --no-router" ;;
    c1)  echo "sdxl_c1|n4-core-n4|--sets n4 --core n4 --no-router" ;;
    no3) echo "sdxl_cityscapes|n1+n4-core-n4|--sets n1 n4 --core n4" ;;
    no4) echo "sdxl_cityscapes|n1+n3+n4-core-n4|--sets n1 n3 n4 --core n4" ;;
    no5) echo "sdxl_cityscapes|n1+n4+n5-core-n4|--sets n1 n4 n5 --core n4" ;;
    no6) echo "sdxl_cityscapes|n3+n4+n5-core-n4|--sets n3 n4 n5 --core n4" ;;
    no7) echo "sdxl_cityscapes|n1+n4+n5-core-n5|--sets n1 n4 n5 --core n5" ;;
    no8) echo "sdxl_cityscapes|n1+n3+n4+n5-core-n4|--sets n1 n3 n4 n5 --core n4" ;;
    no9) echo "sdxl_cityscapes|n1+n3+n4+n5-core-n5|--sets n1 n3 n4 n5 --core n5" ;;
    c2|c2n3) echo "sdxl_cityscapes|n1+n4-core-n4-disable-n4_i3|--sets n1 n4 --core n4 --disable n4_i3" ;;   # C2 on No.3 (alias c2n3)
    c2n2) echo "sdxl_cityscapes|n4-core-n4-disable-n4_i3|--sets n4 --core n4 --no-router --disable n4_i3" ;;   # C2 on No.2: [500,999] = base model only
    *)   return 1 ;;
  esac
}

for name in "$@"; do
  line=$(spec "$name") || { echo "[$(now)] unknown configuration '$name'"; exit 2; }
  IFS='|' read -r run dir args <<< "$line"
  bundle=runs/$run/bundles/$dir/bundle.safetensors
  mkdir -p runs/$run/logs
  if [ -f "$bundle" ]; then
    echo "[$(now)] skip $name ($bundle exists)"
  else
    echo "[$(now)] start $name: assemble --name $run $args"
    # shellcheck disable=SC2086
    python -m catsm.cli assemble --config $CFG --name $run $args > runs/$run/logs/assemble_$name.out 2>&1
    rc=$?
    echo "[$(now)] end $name rc=$rc"
    [ $rc -ne 0 ] && { tail -5 runs/$run/logs/assemble_$name.out; echo "QUEUE ABORTED at $name"; exit $rc; }
    [ -f "$bundle" ] || { echo "QUEUE ABORTED: $bundle missing after rc=0"; exit 1; }
  fi
  python scripts/router_stats.py --bundle "$bundle" --png > runs/$run/bundles/$dir/bundle_stats.out 2>&1
  rc=$?
  cat runs/$run/bundles/$dir/bundle_stats.out
  [ $rc -ne 0 ] && { echo "QUEUE ABORTED: bundle check failed for $name"; exit $rc; }
  if grep -q '"bias_moved": false' runs/$run/bundles/$dir/bundle_stats.json; then
    echo "QUEUE ABORTED: router bias did not move for $name (gate gradient not flowing?)"; exit 1
  fi
done
echo "[$(now)] QUEUE DONE"
