#!/usr/bin/env bash
# Unattended hand-off after the stage-1 CA measurement (docs/GPU_RUNBOOK.md, user instruction 2026-09-14):
#   wait for the measurement python PID -> verify 20 t load -> stage 2 diagnosis (CPU, figures only, no decisions)
#   -> stage 3 fostering queues (n4_i0, n3_i0, n5_i1 from the t=81 selection; C1 n4_i0..i3 in run sdxl_c1).
# Usage: nohup scripts/stage1_to_3_handoff.sh <measure_pid> > runs/sdxl_cityscapes/logs/handoff.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
PID=$1; RUN=runs/sdxl_cityscapes; SEL=$RUN/selections/n1/n1_i0.json
now() { TZ=Asia/Seoul date '+%F %T KST'; }
echo "[$(now)] waiting for measurement PID $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[$(now)] measurement PID $PID exited"
if ! grep -q "Wrote $RUN/measure_summary.json" $RUN/logs/measure.out; then
  echo "[$(now)] HANDOFF ABORTED: measure.out has no 'Wrote measure_summary' line"; tail -5 $RUN/logs/measure.out; exit 1
fi
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
python - <<'PY' || { echo "[$(now)] HANDOFF ABORTED: measurement verification failed"; exit 1; }
from catsm.select import load_measurements
r = load_measurements('runs/sdxl_cityscapes/measurements')
units = {len(v.names) for v in r.values()}
print("verified timesteps:", sorted(r), "units:", units)
assert len(r) == 20 and units == {10400}, (len(r), units)
PY
echo "[$(now)] STAGE 1 DONE (20 t verified)"

echo "[$(now)] STAGE 2 diagnosis start (CPU only)"
python scripts/diagnose_ca.py --config configs/sdxl_cityscapes.yaml > $RUN/logs/diagnose.out 2>&1
echo "[$(now)] STAGE 2 diagnosis rc=$? -> $RUN/diagnosis/ ($(ls $RUN/diagnosis 2>/dev/null | wc -l) files)"

echo "[$(now)] STAGE 3 queue A start"
nohup scripts/foster_queue.sh sdxl_cityscapes:n4_i0:$SEL sdxl_cityscapes:n3_i0:$SEL sdxl_cityscapes:n5_i1:$SEL sdxl_c1:n4_i3 \
  > $RUN/logs/foster_queueA.out 2>&1 &
sleep 900   # let n4_i0 reach steady state, then read its VRAM peak
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "[$(now)] queue A VRAM after 15 min: ${used} MiB"
QB="scripts/foster_queue.sh sdxl_c1:n4_i0 sdxl_c1:n4_i1 sdxl_c1:n4_i2"
if false; then  # parallel fostering measured 9% slower than sequential (2026-09-15); always sequential
  echo "[$(now)] STAGE 3 queue B start in parallel (2x${used} MiB < 80 GB); OMP 8 each"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup $QB > $RUN/logs/foster_queueB.out 2>&1 &
else
  echo "[$(now)] VRAM too high for two jobs; queue B will start after queue A"
  while pgrep -f "foster_queue.sh sdxl_cityscapes:n4_i0" >/dev/null; do sleep 120; done
  echo "[$(now)] STAGE 3 queue B start (sequential)"
  nohup $QB > $RUN/logs/foster_queueB.out 2>&1 &
fi
echo "[$(now)] HANDOFF DONE (queues running; see foster_queueA.out / foster_queueB.out)"
