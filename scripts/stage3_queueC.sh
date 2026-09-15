#!/usr/bin/env bash
# Remaining unique experts after the 2026-09-15 t decision (n3 [81,375,750], n4 [81,201,375,750],
# n5 [41,81,201,375,750]): n5_i0 [0,61]@41, n4_i1 [125,249]@201, n4_i2 [250,499]@375, n4_i3 [500,999]@750.
# Waits for queue B (C1) to finish and for the n5 selection to exist, fosters the four, then shares
# weights to the coinciding experts (n5_i2<-n4_i1; n3_i1,n5_i3<-n4_i2; n3_i2,n5_i4<-n4_i3).
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
now() { TZ=Asia/Seoul date '+%F %T KST'; }
while pgrep -f "foster_queue.sh sdxl_c1:n4_i0 sdxl_c1:n4_i1" >/dev/null; do sleep 60; done
until [ -f runs/sdxl_cityscapes/selections/n5/n5_i0.json ]; do sleep 60; done
echo "[$(now)] queue B finished; starting queue C"
scripts/foster_queue.sh sdxl_cityscapes:n5_i0 sdxl_cityscapes:n4_i1 sdxl_cityscapes:n4_i2 sdxl_cityscapes:n4_i3
rc=$?; echo "[$(now)] queue C rc=$rc"; [ $rc -ne 0 ] && exit $rc
CFG=configs/sdxl_cityscapes.yaml
python scripts/share_expert.py --config $CFG --from n4_i1 --to n5_i2 && \
python scripts/share_expert.py --config $CFG --from n4_i2 --to n3_i1 n5_i3 && \
python scripts/share_expert.py --config $CFG --from n4_i3 --to n3_i2 n5_i4
echo "[$(now)] SHARE rc=$?  QUEUE C DONE"
