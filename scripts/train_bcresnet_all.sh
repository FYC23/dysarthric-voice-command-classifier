#!/usr/bin/env bash
# Train every BC-ResNet width and seed end to end: stage 1 (Speech Commands
# pretraining), then stages 2-3 (TORGO control speakers, dysarthric LOSO folds
# and the deploy model). One job at a time, for a single GPU.
#
# Safe to re-run after a crash or disconnect:
#   - a finished pretraining run (pretrain_metrics.json) is skipped;
#   - an unfinished one continues from pretrain_last.pt (--resume);
#   - a finished fine-tune (eval/run.json) is skipped.
# Stops at the first failure.
#
# Results: runs/bcresnet-<tau>/seed<k>/ (see README, "BC-ResNet (step 3)").
# Logs:    runs/bcresnet-<tau>/seed<k>/{pretrain,finetune}.log
#
# Usage (from anywhere; run inside tmux or nohup, it takes hours):
#   bash scripts/train_bcresnet_all.sh
#   TAUS="1 8" SEEDS="0" bash scripts/train_bcresnet_all.sh
#   DEVICE=cuda:0 PYTHON=.venv/bin/python bash scripts/train_bcresnet_all.sh
#   DRY_RUN=1 bash scripts/train_bcresnet_all.sh   # print the plan, run nothing
#
# Order: every width for seed 0, then every width for seed 1, ... so a complete
# accuracy-vs-size curve exists after the first seed.

set -euo pipefail

TAUS="${TAUS:-1 2 3 8}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-}"            # empty: cuda, then mps, then cpu
NUM_WORKERS="${NUM_WORKERS:-}"  # empty: each script's default
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
    for tau in ${TAUS}; do
        dir="runs/bcresnet-${tau}/seed${seed}"
        echo "[$(date '+%F %T')] BC-ResNet-${tau}, seed ${seed} -> ${dir}"

        if [[ -f "${dir}/pretrain_metrics.json" ]]; then
            echo "  [skip] stage 1: ${dir}/pretrain_metrics.json exists"
        else
            echo "  stage 1: Speech Commands pretraining"
            run_step "${dir}/pretrain.log" "${PYTHON}" -u scripts/pretrain_bcresnet.py \
                --tau "${tau}" --seed "${seed}" --resume ${common_args[@]+"${common_args[@]}"}
        fi

        if [[ -f "${dir}/eval/run.json" ]]; then
            echo "  [skip] stages 2-3: ${dir}/eval/run.json exists"
        else
            echo "  stages 2-3: TORGO control stage, LOSO folds, deploy model"
            run_step "${dir}/finetune.log" "${PYTHON}" -u scripts/finetune_bcresnet.py \
                --tau "${tau}" --seed "${seed}" ${common_args[@]+"${common_args[@]}"}
        fi
    done
done
echo "[$(date '+%F %T')] All done in $(( ($(date +%s) - start) / 60 )) min."
