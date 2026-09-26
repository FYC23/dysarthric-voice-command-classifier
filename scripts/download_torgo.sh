#!/usr/bin/env bash
# Download the TORGO database from its source (University of Toronto) into
# data/raw.
#
# Result:
#   data/raw/TORGO/{F,FC,M,MC}/<speaker>/Session<n>/{prompts,wav_arrayMic,...}
#
# Each archive holds speaker folders with no group folder around them, so it
# is extracted into its own group directory. If <group>.tar.bz2 is already in
# the destination (e.g. downloaded by hand), it is extracted from disk and left
# in place; otherwise it is streamed straight into tar and never saved.
#
# Usage: bash scripts/download_torgo.sh [dest_dir]
# Env:   TORGO_BASE_URL overrides the source (mirror, or a local server in tests)

set -euo pipefail

BASE_URL="${TORGO_BASE_URL:-http://www.cs.toronto.edu/~complingweb/data/TORGO}"
GROUPS_=(F FC M MC)  # female, female control, male, male control

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO_ROOT}/data/raw/TORGO}"

mkdir -p "${DEST}"

for group in "${GROUPS_[@]}"; do
    target="${DEST}/${group}"
    if [[ -d "${target}" ]]; then
        echo "[skip] ${group}: ${target} already exists"
        continue
    fi

    # Extract into a temp dir first so an interrupted run never leaves a
    # half-filled group that the skip check above would treat as complete.
    tmp="${DEST}/.${group}.partial"
    rm -rf "${tmp}"
    mkdir -p "${tmp}"

    archive="${DEST}/${group}.tar.bz2"
    if [[ -f "${archive}" ]]; then
        echo "[extract] ${group}: ${archive}"
        tar -xjf "${archive}" -C "${tmp}"
    else
        url="${BASE_URL}/${group}.tar.bz2"
        echo "[download] ${group}: ${url}"
        curl -fL --retry 3 "${url}" | tar -xj -C "${tmp}"
    fi

    mv "${tmp}" "${target}"
    n_speakers=$(find "${target}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    n_wavs=$(find "${target}" -name '*.wav' | wc -l | tr -d ' ')
    echo "[done] ${group}: ${n_speakers} speakers, ${n_wavs} wav files"
done

echo "TORGO ready at ${DEST}"
