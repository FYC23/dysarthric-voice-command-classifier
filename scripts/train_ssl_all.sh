#!/usr/bin/env bash
# Train every pretrained speech backbone and seed (step 2): the TORGO control
# stage, the controls-only run and the dysarthric LOSO folds. One job at a
# time, for a single GPU.
#
# Safe to re-run after a crash or disconnect: a seed whose
# runs/<backbone>/seed<k>/eval/run.json exists is skipped; an unfinished one
# starts again from scratch. Stops at the first failure. Smoke runs
# (runs/smoke/) are never looked at.
#
# Results: runs/<backbone>/seed<k>/ and runs/<backbone>-controls/seed<k>/.
# Logs:    runs/<backbone>/seed<k>/finetune.log
#
# Usage (run inside tmux or nohup; hubert-large takes hours):
#   bash scripts/train_ssl_all.sh
#   BACKBONES="distilhubert" SEEDS="0" bash scripts/train_ssl_all.sh
#   HF_ENDPOINT=https://hf-mirror.com DEVICE=cuda:0 bash scripts/train_ssl_all.sh
#   DRY_RUN=1 bash scripts/train_ssl_all.sh   # print the plan, run nothing
#
# Order: every backbone for seed 0, smallest first, then seed 1, ... so a
# problem shows on the fast models first and a full curve exists after seed 0.

set -euo pipefail

BACKBONES="${BACKBONES:-distilhubert hubert-base hubert-large}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-}"            # empty: cuda, then mps, then cpu
NUM_WORKERS="${NUM_WORKERS:-}"  # empty: the script's default
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

common_args=()
[[ -n "${DEVICE}" ]] && common_args+=(--device "${DEVICE}")
[[ -n "${NUM_WORKERS}" ]] && common_args+=(--num-workers "${NUM_WORKERS}")

# Run one step, appending its output to a log. In a dry run, only print it.
run_step() {
    local log="$1"; shift
    echo "    \$ $*"
    [[ "${DRY_RUN}" == "1" ]] && return 0
    mkdir -p "$(dirname "${log}")"
    echo "=== $(date '+%F %T') $*" >> "${log}"
    "$@" 2>&1 | tee -a "${log}"
}

start=$(date +%s)
for seed in ${SEEDS}; do
    for backbone in ${BACKBONES}; do
        dir="runs/${backbone}/seed${seed}"
        if [[ -f "${dir}/eval/run.json" ]]; then
            echo "[skip] ${backbone} seed ${seed}: ${dir}/eval/run.json exists"
            continue
        fi
        echo "[$(date '+%F %T')] ${backbone}, seed ${seed} -> ${dir}"
        run_step "${dir}/finetune.log" "${PYTHON}" -u scripts/finetune_ssl.py \
            --backbone "${backbone}" --seed "${seed}" ${common_args[@]+"${common_args[@]}"}
    done
done
echo "[$(date '+%F %T')] All done in $(( ($(date +%s) - start) / 60 )) min."
