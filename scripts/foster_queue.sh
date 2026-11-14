#!/usr/bin/env bash
# Run fostering jobs one after another. Each argument is RUN_NAME:EXPERT[:SELECTION_JSON].
#   scripts/foster_queue.sh sdxl_cityscapes:n4_i0:runs/sdxl_cityscapes/selections/n1/n1_i0.json sdxl_c1:n4_i0   # earlier run
#   CFG=configs/sdxl_cityscapes_selfgen_fewshot10.yaml scripts/foster_queue.sh \
#       sdxl_cityscapes_selfgen_fewshot10:n4_i0:runs/sdxl_cityscapes_selfgen_fewshot10/selections/n1/n1_i0.json   # reported run
# CFG defaults to configs/sdxl_cityscapes.yaml, the config of an earlier experiment (full-data
# training) whose results are not in this repository. The reported run called this script one job
# at a time from scripts/ops/run_fewshot10.sh, with CFG exported by scripts/ops/run_selfgen_chain.sh.
# Skips an expert whose lora-final.safetensors already exists. Logs to runs/<run>/logs/foster_<expert>.out.
# Launch with nohup and record the PID via pgrep.
# Since 2026-11-07 `foster` refuses a selection not measured on self-generated images; to
# rerun a historical (real-image) campaign set FOSTER_EXTRA=--allow-legacy-selection.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
CFG=${CFG:-configs/sdxl_cityscapes.yaml}
for job in "$@"; do
  IFS=: read -r run expert selection <<< "$job"
  out=runs/$run/experts/$expert/lora-final.safetensors
  if [ -f "$out" ]; then echo "[$(TZ=Asia/Seoul date '+%F %T') KST] skip $run/$expert (exists)"; continue; fi
  mkdir -p runs/$run/logs
  args=(--config $CFG --name $run --expert $expert ${FOSTER_EXTRA:-})
  [ -n "${selection:-}" ] && args+=(--selection "$selection")
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] start $run/$expert ${selection:+(selection $selection)}"
  python -m catsm.cli foster "${args[@]}" > runs/$run/logs/foster_$expert.out 2>&1
  rc=$?
  echo "[$(TZ=Asia/Seoul date '+%F %T') KST] end $run/$expert rc=$rc"
  [ $rc -ne 0 ] && { echo "QUEUE ABORTED at $run/$expert"; exit $rc; }
done
echo "[$(TZ=Asia/Seoul date '+%F %T') KST] QUEUE DONE"
