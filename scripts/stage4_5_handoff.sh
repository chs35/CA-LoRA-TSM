#!/usr/bin/env bash
# Unattended stage 4 -> 5 chain (user decisions 2026-09-16: n1_i0 stays imported; routers for every m>1
# scenario No.3-9; one generation seed; base control reused from the upstream control_base eval.json;
# paired-loss re-check on the final experts; C2 and the robustness sweep are decided later).
#   nohup scripts/stage4_5_handoff.sh > runs/sdxl_cityscapes/logs/stage4_5_handoff.out 2>&1 &
# Order: bundles (hard No.2, C1; routers No.3..No.9, ~25 min each) -> paired loss (~45 min)
#        -> generation + evaluation of No.1, No.2, No.3, C1 first, then No.4..No.9 (~3.5 h each).
# Every part skips finished outputs, so re-launching after a crash resumes.
set -u
cd "$(dirname "$0")/.."
now() { TZ=Asia/Seoul date '+%F %T KST'; }
echo "[$(now)] STAGE 4 start"
scripts/stage4_assemble_queue.sh no1 no2 c1 no3 no4 no5 no6 no7 no8 no9 || { echo "[$(now)] STAGE 4 FAILED"; exit 1; }
echo "[$(now)] STAGE 4 DONE"
echo "[$(now)] STAGE 3b paired loss start"
scripts/stage3b_paired_loss.sh || echo "[$(now)] paired loss had failures (non-fatal, continuing)"
echo "[$(now)] STAGE 5 start"
scripts/stage5_eval_queue.sh no1 no2 no3 c1 no4 no5 no6 no7 no8 no9 || { echo "[$(now)] STAGE 5 FAILED"; exit 1; }
echo "[$(now)] STAGE 5 DONE"
source scripts/env.sh
U=/workspace/Optimizing-CA-LoRA/runs/sdxl_cityscapes   # upstream Improving-CA-LoRA arms as reference rows (user request 2026-09-18)
python scripts/collect_results.py --extra $U/control_base/eval.json $U/A_t81_style/eval.json $U/C_t41_style/eval.json $U/B_top3_style/eval.json $U/D_t1_style/eval.json $U/E_all/eval.json $U/E_random/eval.json \
  --md runs/results_table.md --json runs/results.json
echo "[$(now)] HANDOFF DONE"
