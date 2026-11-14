#!/bin/bash
# Unattended hand-off (2026-11-07): when the self-generated CA measurement in
# Improving-CA-LoRA finishes on this GPU server, run CA-LoRA-TSM's whole pipeline on it.
#
#   1. wait for runs/ca_selfgen/campaign.log to say CAMPAIGN END (stop on HALT) and the GPU to idle
#   2. import the measurements into runs/sdxl_cityscapes_selfgen and runs/sdxl_c1_selfgen
#   3. selections: n1/n3/n4/n5 with the existing interval representatives (run the
#      pipeline on the measured set; no re-decision), C1 = n4 at fixed t=81
#   4. the 10-image campaign (scripts/ops/run_fewshot10.sh) under new run names, reusing the
#      tensor-core gate of 2026-11-02 and the real-val CMMD reference export
# Also drops the t=81 style selection where Hierarchical-CFG expects it.
#
#   nohup scripts/ops/run_selfgen_chain.sh > runs/selfgen_chain.out 2>&1 &
# Resumable: every step is idempotent (import verifies, select rewrites the same files,
# the campaign skips verified stages).
set -u -o pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
SRC_REPO=${SRC_REPO:-/workspace/Improving-CA-LoRA}
SRC_RUN=$SRC_REPO/runs/ca_selfgen
HCFG_SEL=${HCFG_SEL:-/workspace/Hierarchical-CFG/configs/selections/selfgen_style_t81.json}
LOG=runs/selfgen_chain.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
halt() { say "HALT $*"; exit 1; }

say "CHAIN START (waiting for $SRC_RUN/campaign.log)"
while :; do
  if grep -q "HALT" "$SRC_RUN/campaign.log" 2>/dev/null; then
    halt "the CA measurement halted: $(grep HALT "$SRC_RUN/campaign.log" | tail -1)"
  fi
  grep -q "CAMPAIGN END" "$SRC_RUN/campaign.log" 2>/dev/null && break
  sleep 120
done
while pgrep -f "run_ca_selfgen.sh|measure_ca.py|select_projections.py" > /dev/null; do sleep 30; done
say "CA measurement finished: $(grep 'CAMPAIGN END' "$SRC_RUN/campaign.log" | tail -1)"

# Hierarchical-CFG reads its CA-LoRA selection from its own configs/ (copy, verified).
if [ -s "$SRC_RUN/selection_A_t81_style.json" ] && [ -d "$(dirname "$HCFG_SEL")" ]; then
  cp "$SRC_RUN/selection_A_t81_style.json" "$HCFG_SEL.tmp" && mv "$HCFG_SEL.tmp" "$HCFG_SEL"
  cmp -s "$SRC_RUN/selection_A_t81_style.json" "$HCFG_SEL" || halt "HCFG selection copy does not verify"
  say "HCFG selection -> $HCFG_SEL ($(sha256sum < "$HCFG_SEL" | cut -c1-12))"
fi

for run in sdxl_cityscapes_selfgen sdxl_c1_selfgen; do
  bash scripts/ops/import_measurements.sh "$SRC_RUN/measurements" "$run" >> "$LOG" 2>&1 \
    || halt "import into runs/$run failed"
done
say "imported measurements"

SEL_CFG=configs/sdxl_cityscapes_selfgen.yaml
for scheme in n1 n3 n4 n5; do
  python -m catsm.cli select --config $SEL_CFG --scheme $scheme --set model.device=cpu \
    >> runs/sdxl_cityscapes_selfgen/select_$scheme.log 2>&1 || halt "select $scheme failed"
done
python -m catsm.cli select --config $SEL_CFG --name sdxl_c1_selfgen --scheme n4 --mode fixed \
  --set model.device=cpu >> runs/sdxl_c1_selfgen/select_n4.log 2>&1 || halt "select C1 failed"
python - <<'PY' >> "$LOG" 2>&1 || halt "selection check failed"
import glob, json
for path in sorted(glob.glob("runs/sdxl_c*_selfgen/selections/n*/n*_i*.json")):
    meta = json.load(open(path))["meta"]
    assert meta["data_source"] == "selfgen", path
    print(path, meta["mode"], meta["timesteps"], len(json.load(open(path))["names"]))
PY
say "selections written (data_source=selfgen)"

export CFG=configs/sdxl_cityscapes_selfgen_fewshot10.yaml
export RUN=sdxl_cityscapes_selfgen_fewshot10 RUN_C1=sdxl_c1_selfgen_fewshot10
export SEL_FROM=sdxl_cityscapes_selfgen SEL_FROM_C1=sdxl_c1_selfgen
export GATE=runs/sdxl_cityscapes_fewshot10_gate/tensor_core_gate.json
mkdir -p runs/$RUN/logs
say "TSM campaign start: runs/$RUN/campaign.log"
scripts/ops/run_fewshot10.sh > runs/$RUN/logs/runner.out 2>&1
rc=$?
say "TSM campaign end rc=$rc ($(tail -1 runs/$RUN/campaign.log))"
exit $rc
