#!/usr/bin/env bash
# Paired base-vs-expert L_Diff on the final experts (user decision 2026-09-16: cheap, low expectations).
# Same 96 (image, t, noise) triples per interval (scripts/paired_loss.py, seed 12345), three comparisons:
#   (a) each unique trained expert on its own interval          -> experts/<e>/paired_loss.json
#   (b) n1_i0 (whole-range expert) restricted to that interval  -> experts/n1_i0/lora-final.paired_loss_t<lo>-<hi>.json
#   (c) C1 experts (selection fixed at t=81) on their interval  -> runs/sdxl_c1/experts/<e>/paired_loss.json
# (a) vs (b) = does interval training help inside the interval; (a) vs (c) = does interval *selection* help.
# Skips existing outputs. ~2.5 min per run (SDXL load + 2 x 96 forwards), 18 runs ≈ 45 min. GPU must be free.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
CFG=configs/sdxl_cityscapes.yaml
now() { TZ=Asia/Seoul date '+%F %T KST'; }
run_one() {  # run expert t_lo t_hi out
  local run=$1 expert=$2 lo=$3 hi=$4 out=$5 file=runs/$1/experts/$2/lora-final.safetensors
  if [ -f "$out" ]; then echo "[$(now)] skip $out (exists)"; return 0; fi
  echo "[$(now)] paired_loss $run/$expert on [$lo,$hi] -> $out"
  python scripts/paired_loss.py --config $CFG --name $run --expert-file "$file" --t-lo "$lo" --t-hi "$hi" --out "$out" \
    > runs/$run/logs/paired_loss_${expert}_t${lo}-${hi}.out 2>&1 || { echo "[$(now)] FAILED (see runs/$run/logs/paired_loss_${expert}_t${lo}-${hi}.out)"; return 1; }
  grep -E "delta" runs/$run/logs/paired_loss_${expert}_t${lo}-${hi}.out | tail -1 | sed 's/^/    /'
}
E=runs/sdxl_cityscapes/experts
# (a) unique trained experts on their own interval
run_one sdxl_cityscapes n5_i0   0  61 $E/n5_i0/paired_loss.json
run_one sdxl_cityscapes n5_i1  62 124 $E/n5_i1/paired_loss.json
run_one sdxl_cityscapes n4_i0   0 124 $E/n4_i0/paired_loss.json
run_one sdxl_cityscapes n4_i1 125 249 $E/n4_i1/paired_loss.json
run_one sdxl_cityscapes n3_i0   0 249 $E/n3_i0/paired_loss.json
run_one sdxl_cityscapes n4_i2 250 499 $E/n4_i2/paired_loss.json
run_one sdxl_cityscapes n4_i3 500 999 $E/n4_i3/paired_loss.json
run_one sdxl_cityscapes n1_i0   0 999 $E/n1_i0/paired_loss.json
# (b) n1_i0 restricted to each interval
for iv in "0 61" "62 124" "0 124" "125 249" "0 249" "250 499" "500 999"; do
  set -- $iv; run_one sdxl_cityscapes n1_i0 $1 $2 $E/n1_i0/lora-final.paired_loss_t$1-$2.json
done
# (c) C1 experts (t=81 selection everywhere) on their interval
run_one sdxl_c1 n4_i0   0 124 runs/sdxl_c1/experts/n4_i0/paired_loss.json
run_one sdxl_c1 n4_i1 125 249 runs/sdxl_c1/experts/n4_i1/paired_loss.json
run_one sdxl_c1 n4_i2 250 499 runs/sdxl_c1/experts/n4_i2/paired_loss.json
run_one sdxl_c1 n4_i3 500 999 runs/sdxl_c1/experts/n4_i3/paired_loss.json
echo "[$(now)] PAIRED LOSS DONE"
python - <<'PY'
import json, glob
rows = []
for p in sorted(glob.glob("runs/sdxl_c*/experts/*/paired_loss*.json")):
    d = json.load(open(p)); rows.append((p, d["interval"], d["base_mean"], d["expert_mean"], d["paired_delta_mean"], d["paired_delta_sem"], d["relative_change"], d["frac_improved"]))
print(f"{'file':70s} {'interval':>10s} {'base':>8s} {'expert':>8s} {'delta':>9s} {'sem':>7s} {'rel%':>6s} {'impr':>5s}")
for p, iv, b, e, dm, se, rel, fi in rows:
    print(f"{p:70s} {str(iv):>10s} {b:8.4f} {e:8.4f} {dm:+9.5f} {se:7.5f} {100*rel:6.2f} {100*fi:5.0f}")
PY
