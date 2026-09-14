# Usage: source scripts/env.sh
#
# Activates the shared venv and puts this repository's src/ first on PYTHONPATH so
# the vendored calora shadows the editable install that points at
# /workspace/Optimizing-CA-LoRA. No package installation is needed.
_catsm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /workspace/venv-calora/bin/activate
export PYTHONPATH="${_catsm_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
# torch grabs every core by default; on many-core pods that makes small CPU ops
# hundreds of times slower (see Optimizing-CA-LoRA/docs/RUNBOOK_A100.md).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
unset _catsm_root
