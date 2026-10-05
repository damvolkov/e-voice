#!/usr/bin/env bash
# TTS benchmark matrix: every config × (fleurs-es, fleurs-en texts) at 1 stream, with audio and speaker
# similarity to REFERENCE (the clip VOICE was learned from; skipped when absent), then fleurs-es at 4 and 8 concurrent streams. Last, the round trip: the STT
# bench transcribes every run's audio (<run>.roundtrip.jsonl), so `make report SERVICE=tts` shows WER.
set -euo pipefail

BIN="${BIN:-data/ops/target/release/e-voice-tts}"
DATA="${DATA:-data/stt/ops/datasets}"
OUT="${OUT:-data/tts/ops/results}"
VOICE="${VOICE:-damien}"
REFERENCE="${REFERENCE:-data/tts/ops/samples/reference.mp3}"
STT="${STT:-data/ops/target/release/e-voice}"
LIMIT="${LIMIT:-40}"
mkdir -p "${OUT}"

run() {
    local config="$1" set="$2" streams="$3"; shift 3
    local name; name="$(basename "${config}" .toml)"
    printf '\033[36m%s · %s · %s streams\033[0m\n' "${name}" "${set}" "${streams}"
    EVOICE_TTS__SYNTH__WORKERS="${streams}" "${BIN}" --config "${config}" bench "${DATA}/${set}/manifest.jsonl" \
        --out "${OUT}/${name}.${set}.s${streams}.jsonl" --streams "${streams}" --limit "${LIMIT}" --voice "${VOICE}" "$@"
}

configs=("$@")
[[ ${#configs[@]} -eq 0 ]] && configs=(evoice/tts/ops/bench/*.toml)

for config in "${configs[@]}"; do
    for set in fleurs-es fleurs-en; do
        reference=()
        [[ -f "${REFERENCE}" ]] && reference=(--reference "${REFERENCE}")
        run "${config}" "${set}" 1 --wavs "${reference[@]}"
    done
    for streams in 4 8; do
        run "${config}" fleurs-es "${streams}"
    done
done

for manifest in "${OUT}"/*.wavs/manifest.jsonl; do
    [[ -f "${manifest}" ]] || continue
    run="${manifest%.wavs/manifest.jsonl}"
    printf '\033[36mround trip · %s\033[0m\n' "$(basename "${run}")"
    "${STT}" bench "${manifest}" --out "${run}.roundtrip.jsonl" --streams 4
done
