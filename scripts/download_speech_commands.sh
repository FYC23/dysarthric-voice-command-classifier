#!/usr/bin/env bash
# Download Google Speech Commands v0.02 (the archives behind
# https://huggingface.co/datasets/google/speech_commands) into data/raw.
#
# Result:
#   data/raw/speech_commands_v2/{train,validation,test}/<word>/<speaker>_nohash_<n>.wav
#
# The HF dataset repo only holds a loading script; these are the tarballs it
# pulls, so the train/validation/test split matches the HF version exactly.
# Archives are streamed straight into tar, so no .tar.gz is left on disk.
#
# Usage: bash scripts/download_speech_commands.sh [dest_dir]

set -euo pipefail

VERSION="v0.02"
BASE_URL="https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/${VERSION}"
SPLITS=(train validation test)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO_ROOT}/data/raw/speech_commands_v2}"

mkdir -p "${DEST}"

for split in "${SPLITS[@]}"; do
    target="${DEST}/${split}"
    if [[ -d "${target}" ]]; then
        echo "[skip] ${split}: ${target} already exists"
        continue
    fi

    # Extract into a temp dir first so an interrupted download never leaves
    # a half-filled split that the skip check above would treat as complete.
    tmp="${DEST}/.${split}.partial"
    rm -rf "${tmp}"
    mkdir -p "${tmp}"

    url="${BASE_URL}/${VERSION}_${split}.tar.gz"
    echo "[download] ${split}: ${url}"
    curl -fL --retry 3 "${url}" | tar -xz -C "${tmp}"

    mv "${tmp}" "${target}"
    n_wavs=$(find "${target}" -name '*.wav' | wc -l | tr -d ' ')
    echo "[done] ${split}: ${n_wavs} wav files"
done

echo "Speech Commands ${VERSION} ready at ${DEST}"
