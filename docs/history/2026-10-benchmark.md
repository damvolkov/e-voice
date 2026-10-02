# 2026-10 · CPU benchmark and default selection

Run with `make datasets bench report` on one desktop CPU, onnxruntime 1.28.2 through sherpa-onnx 1.13.8,
4 intra-op threads, spinning off. Sets: FLEURS es/en (200 utterances each, WER), MESD es (129, emotion),
CREMA-D en (400, emotion). `file` is the upload path (lossless intake), `live1`/`live4` pace 12 utterances
in real time with 1 or 4 concurrent streams. Raw lines: `data/stt/ops/results/*.jsonl`.

## ASR

| run | WER es | WER en | CPU s / audio s | throughput | final p50 / p95 (live) |
|---|---|---|---|---|---|
| Parakeet TDT v3 int8 (file) | **4.5%** | **8.1%** | **0.25** | 25× | — |
| Nemotron 3.5 1120 ms | 8.4% | 11.4% | 0.41 | 21× | 739 / 1024 ms |
| Nemotron 3.5 560 ms | 8.8% | 11.7% | 0.60 | 15× | 741 / 1017 ms |
| Nemotron 3.5 160 ms | 8.8% | 12.8% | 1.38 | 6× | 1027 / 1226 ms |

- Parakeet wins on files by every measure; it has no streaming partials, so live keeps Nemotron.
- Smaller Nemotron chunks buy earlier partials (649 ms vs 1037 ms p50 at 160 ms) at 3.4× the CPU and a
  *later* final: the decoder falls behind. 1120 ms is the default.
- Nemotron's Spanish needs the `es-ES` prompt; `es` alone selects es-US and costs accuracy.

## Front end

| change | effect |
|---|---|
| AGC off (peak normalizer to −6 dBFS, ≤ 40 dB) | en WER 11.4% → 22.8%; es unchanged. FLEURS en is recorded quietly. AGC stays on. |
| TEN VAD instead of Silero | en WER 11.7% → 14.8%, es equal. Silero stays. |
| 1 thread instead of 4 | live final p95 1.0 s → 2.0 s, file throughput 15× → 8.6×, for 0.60 → 0.38 CPU s per audio s. |
| intra-op spinning on | CPU per audio second 0.44 → 0.93 on files, 0.37 → 2.7 live, for no latency gain. Disabled everywhere. |

## SER (emotion2vec+ base)

- CREMA-D en: accuracy 65.8%, macro-F1 0.655, angry recall 87.7%.
- MESD es: accuracy 24.8% — acted Spanish single words; weak, documented, not hidden.
- Cost: ~800 MB RSS; live finals p50 / p95 623 / 711 ms without it, 739 / 1024 ms with it.
  On by default; `ser.backend = "off"` removes the cost, `emotion=off` only hides the field per request.

## Wake word (kws, open vocabulary)

`make wake WORD=…` synthesizes the phrase with three Piper voices × five carriers × three speeds and plays
decoys plus real FLEURS speech as negatives over threshold × boost.

| phrase | best recall | false accepts |
|---|---|---|
| `eager` | 76% | ~17 / h |
| `hey eager` | 80% | 0 in 24 min |

A single short word is acoustically too close to everyday speech; `hey eager` is the default keyword. The
GigaSpeech 3.3M model detects the male synthetic voice reliably and the female ones rarely — real-voice
recall must be checked with `ecli` before relying on it. The wake word is off in the default pipeline.

## Defaults chosen

live Nemotron 1120 ms · files Parakeet · Silero · AGC on · SER emotion2vec on · ww off (kws `hey eager`).
