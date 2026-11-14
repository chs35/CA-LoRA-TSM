#!/bin/bash
# Import the self-generated CA measurements made once in Improving-CA-LoRA.
#
#   bash scripts/ops/import_measurements.sh \
#       /workspace/Improving-CA-LoRA/runs/ca_selfgen/measurements sdxl_cityscapes_selfgen
#
# Since 2026-11-07 this project does not measure CA itself: the paper measures it
# on images the base model generated (CA-LoRA Sec. 3.2), and that measurement is
# shared with Improving-CA-LoRA and Hierarchical-CFG. This copies every
# t*.safetensors (full and _half*) plus SHA256SUMS into runs/<run>/measurements/,
# verifies the copy byte for byte against the source's SHA256SUMS (written by the
# source's scripts/ops/verify_measurements.py), checks that every file loads at
# head granularity with data_source=selfgen and one shared provenance, and writes
# runs/<run>/IMPORTED.json. Refuses to overwrite a different file already there.
set -u -o pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
SRC=${1:?usage: import_measurements.sh <source measurements dir> <run name>}
RUN=${2:?usage: import_measurements.sh <source measurements dir> <run name>}
DST=runs/$RUN/measurements
[ -s "$SRC/SHA256SUMS" ] || { echo "no SHA256SUMS in $SRC (run verify_measurements.py there)"; exit 1; }
(cd "$SRC" && sha256sum -c --quiet SHA256SUMS) || { echo "source $SRC fails its own SHA256SUMS"; exit 1; }
mkdir -p "$DST"
while read -r sum name; do
  if [ -e "$DST/$name" ] && [ "$(sha256sum < "$DST/$name" | cut -d' ' -f1)" != "$sum" ]; then
    echo "$DST/$name exists and differs from the source; refusing to overwrite"; exit 1
  fi
  for attempt in 1 2 3; do
    cp -p "$SRC/$name" "$DST/$name.tmp" && mv "$DST/$name.tmp" "$DST/$name"
    [ "$(sha256sum < "$DST/$name" | cut -d' ' -f1)" = "$sum" ] && break
    [ "$attempt" = 3 ] && { echo "copy of $name does not verify"; exit 1; }
    sleep 10
  done
done < "$SRC/SHA256SUMS"
cp -p "$SRC/SHA256SUMS" "$DST/SHA256SUMS"
python - "$SRC" "$DST" "runs/$RUN/IMPORTED.json" <<'PY'
import datetime, json, sys
from pathlib import Path
from calora.ca.stats import common_provenance, load_measurement
from catsm.select import load_measurements
src, dst, out = map(Path, sys.argv[1:])
results = load_measurements(dst, granularity="head")          # requires data_source=selfgen
halves = [load_measurement(p, expect_granularity="head") for p in sorted(dst.glob("t*_half*.safetensors"))]
provenance = common_provenance([*results.values(), *halves])
json.dump({
    "source": str(src.resolve()),
    "imported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "timesteps": sorted(results),
    "num_files": len(results) + len(halves),
    "num_units": len(next(iter(results.values())).names),
    "provenance": provenance,
    "sha256sums": (dst / "SHA256SUMS").read_text().splitlines(),
}, open(out, "w"), indent=2)
print(f"imported {len(results)} timesteps {sorted(results)} ({len(halves)} halves) -> {dst}; "
      f"data_source={provenance['data_source']} precision={provenance.get('precision')}")
PY
