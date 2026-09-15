#!/usr/bin/env bash
# Run fostering jobs one after another. Each argument is RUN_NAME:EXPERT[:SELECTION_JSON].
#   scripts/foster_queue.sh sdxl_cityscapes:n4_i0:runs/sdxl_cityscapes/selections/n1/n1_i0.json sdxl_c1:n4_i0
# Skips an expert whose lora-final.safetensors already exists. Logs to runs/<run>/logs/foster_<expert>.out.
# Launch with nohup and record the PID via pgrep (see docs/GPU_RUNBOOK.md pod checklist).
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
CFG=configs/sdxl_cityscapes.yaml
for job in "$@"; do
  IFS=: read -r run expert selection <<< "$job"
  out=runs/$run/experts/$expert/lora-final.safetensors
  if [ -f "$out" ]; then echo "[$(TZ=Asia/Seoul date '+%F %T') KST] skip $run/$expert (exists)"; continue; fi
  mkdir -p runs/$run/logs
  args=(--config $CFG --name $run --expert $expert)
  [ -n "${selection:-}" ] && args+=(--selection "$selection")
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] start $run/$expert ${selection:+(selection $selection)}"
  python -m catsm.cli foster "${args[@]}" > runs/$run/logs/foster_$expert.out 2>&1
  rc=$?
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] end $run/$expert rc=$rc"
  [ $rc -ne 0 ] && { echo "QUEUE ABORTED at $run/$expert"; exit $rc; }
done
echo "[$(TZ=Asia/Seoul date '+%F %T') KST] QUEUE DONE"
