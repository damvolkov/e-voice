#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-data/ops/target/release/e-voice}"
DATA="${DATA:-data/stt/ops/datasets}"
OUT="${OUT:-data/stt/ops/results}"
STREAMS="${STREAMS:-4}"
LIVE="${LIVE:-20}"
SER="${SER:-evoice/stt/ops/bench/parakeet.toml evoice/stt/ops/bench/ser-large.toml}"
mkdir -p "${OUT}"

run() {
    local config="$1" set="$2" tag="$3"; shift 3
    local name; name="$(basename "${config}" .toml)"
    printf '\033[36m%s · %s · %s\033[0m\n' "${name}" "${set}" "${tag}"
    "${BIN}" --config "${config}" bench "${DATA}/${set}/manifest.jsonl" --out "${OUT}/${name}.${set}.${tag}.jsonl" "$@"
}

configs=("$@")
[[ ${#configs[@]} -eq 0 ]] && configs=(evoice/stt/ops/bench/*.toml)

for config in "${configs[@]}"; do
    for set in fleurs-es fleurs-en; do
        run "${config}" "${set}" file --streams "${STREAMS}"
    done
    run "${config}" fleurs-es live1 --mode live --limit "${LIVE}" --streams 1
    run "${config}" fleurs-es live4 --mode live --limit "${LIVE}" --streams 4
done

for ser in ${SER}; do
    for set in mesd crema; do
        run "${ser}" "${set}" file --streams "${STREAMS}"
    done
done
