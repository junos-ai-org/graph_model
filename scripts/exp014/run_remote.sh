#!/usr/bin/env bash
# Exp 014 — GTLM paradigm on tables (graph prefix + generic SPD/RRWP/Magnetic
# biases) vs exp013's plain-lora baseline (reused, NOT retrained).
#
# Usage (on pod): bash scripts/exp014/run_remote.sh
# Smoke: MAX_EXAMPLES=40 EVAL_MAX=20 PUSH_HUB=0 bash scripts/exp014/run_remote.sh
# Pre-reqs on the pod env: GH_TOKEN, HF_TOKEN, WANDB_API_KEY, CODE_REF
# (branch/commit of junos-ai-org/graph_model carrying src/experiments/exp014).
set -euo pipefail
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-exp_014}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-reasoning-llm}"

HF_REPO="${HF_REPO:-akumch/graph-reasoning-llm-checkpoints}"
PUSH_HUB="${PUSH_HUB:-1}"
EPOCHS="${EPOCHS:-3}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"   # >0 = smoke train cap
EVAL_MAX="${EVAL_MAX:--1}"           # >0 = smoke eval cap
OUT=/workspace/exp014
CKPT="${OUT}/runs/gtlm-graph/final"

LOG=/workspace/run_exp014.log
exec > >(tee -a "$LOG") 2>&1
echo "=== [exp014] $(date -u +%FT%TZ) start (epochs=${EPOCHS} max_examples=${MAX_EXAMPLES} eval_max=${EVAL_MAX}) ==="

export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 1. graph_model repo at CODE_REF (pinned clone, never silent main) ----
: "${CODE_REF:?CODE_REF required (junos graph_model branch/commit with exp014)}"
REPO_URL="https://${GH_TOKEN}@github.com/junos-ai-org/graph_model.git"
if [ ! -d /workspace/graph_model/.git ]; then
  rm -rf /workspace/graph_model
  git clone -b "${CODE_REF}" --depth 1 "${REPO_URL}" /workspace/graph_model
else
  git -C /workspace/graph_model remote set-url origin "${REPO_URL}"
  git -C /workspace/graph_model fetch --depth 1 origin "${CODE_REF}"
  git -C /workspace/graph_model checkout -f FETCH_HEAD
fi
cd /workspace/graph_model
echo "[exp014] graph_model at $(git rev-parse --short HEAD) (ref=${CODE_REF})"

# ---- 2. Deps (same core pins as exp013 + networkx for the graph build) ----
pip install -q transformers==4.50.3 peft==0.18.1 datasets==4.8.4 accelerate==1.13.0 \
    'huggingface-hub[cli]==0.27.1' 'hf_transfer>=0.1.0' 'wandb==0.18.7' networkx

# ---- 3. Hydrate base model ----
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B'); \
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')"

push_dir() {  # push_dir <local_dir> <path_in_repo> <msg>
  python - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import HfApi
local, dest, msg = sys.argv[1:4]
api = HfApi()
api.create_repo("akumch/graph-reasoning-llm-checkpoints", repo_type="model",
                private=True, exist_ok=True)
api.upload_folder(folder_path=local, path_in_repo=dest,
                  repo_id="akumch/graph-reasoning-llm-checkpoints",
                  repo_type="model", commit_message=msg)
print(f"pushed {dest}")
PY
}

rm -f /workspace/DONE_exp014

# ---- 4. Prep (graph build + SPD/RRWP/Magnetic features, cached) ----
echo "=== [$(date -u +%FT%TZ)] prep ==="
PREP_ARGS=()
[ "${MAX_EXAMPLES}" -gt 0 ] && PREP_ARGS+=(--max-examples "${MAX_EXAMPLES}" --force)
python -m src.experiments.exp014.run prep --out "${OUT}" "${PREP_ARGS[@]}"

# ---- 5. Train ----
echo "=== [$(date -u +%FT%TZ)] train gtlm-graph ==="
python -m src.experiments.exp014.run train --out "${OUT}" --epochs "${EPOCHS}"

# ---- 6. Checkpoint to Hub FIRST (survival path against pod death) ----
[ -d "${CKPT}" ] || { echo "FATAL: expected checkpoint ${CKPT} missing"; exit 1; }
if [ "${PUSH_HUB}" = 1 ]; then
  push_dir "${OUT}/runs/gtlm-graph" "exp_014/gtlm-graph" \
    "exp014 gtlm-graph adapter+bias (all checkpoints)"
fi

# ---- 7. Eval on TableBench DP test (records flushed per example) ----
echo "=== [$(date -u +%FT%TZ)] eval gtlm-graph ==="
python -m src.experiments.exp014.run eval --out "${OUT}" \
    --checkpoint "${CKPT}" --eval-max "${EVAL_MAX}"

# ---- 8. Push eval artifacts ----
if [ "${PUSH_HUB}" = 1 ]; then
  push_dir "${OUT}/eval/gtlm-graph" "exp_014/gtlm-graph/eval" \
    "exp014 gtlm-graph DP-test eval (records + results)"
fi

echo "=== [exp014] $(date -u +%FT%TZ) DONE ==="
touch /workspace/DONE_exp014
