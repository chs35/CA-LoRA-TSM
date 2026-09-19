#!/usr/bin/env bash
# C2 controls after the main stage-5 queue (user request 2026-09-19 08:4x KST): wait for the main queue to
# finish, build C2 on No.2 (hard, no training) and C2 on No.3 (router retrained), generate + evaluate both,
# then rebuild the results table. Re-launch to resume; every part skips finished outputs.
#   nohup scripts/stage5_c2_followup.sh > runs/sdxl_cityscapes/logs/c2_followup.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
now() { TZ=Asia/Seoul date '+%F %T KST'; }
# 2026-09-19 10:10 KST: the main queue's bash died with a syntax error right after No.9's evaluation because
# the queue script was edited while it was running (bash reads scripts incrementally; never edit a running
# script). No.9's outputs are complete, so the wait is replaced by an output check. The old pgrep wait also
# matched its own launcher's command line (heredoc text) and would have waited forever.
[ -f runs/sdxl_cityscapes/no9/eval.json ] || { echo "[$(now)] no9/eval.json missing; main queue not finished"; exit 1; }
echo "[$(now)] main queue outputs present; C2 assemble"
scripts/stage4_assemble_queue.sh c2n2 c2 || { echo "[$(now)] C2 ASSEMBLE FAILED"; exit 1; }
echo "[$(now)] C2 generate + evaluate"
scripts/stage5_eval_queue.sh c2n2 c2 || { echo "[$(now)] C2 EVAL FAILED"; exit 1; }
source scripts/env.sh
U=/workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes
python scripts/collect_results.py --extra $U/control_base/eval.json $U/A_t81_style/eval.json $U/C_t41_style/eval.json $U/B_top3_style/eval.json $U/D_t1_style/eval.json $U/E_all/eval.json $U/E_random/eval.json \
  --md runs/results_table.md --json runs/results.json
echo "[$(now)] C2 FOLLOWUP DONE"
