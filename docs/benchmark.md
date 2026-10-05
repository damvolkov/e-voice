# Benchmark

The benchmark runs manifests through the configured pipeline. It uses the same `Runner` the gateway
serves, so the numbers are what clients get.

```bash
make setup ARGS=--all    # every model, including tool and test models
make export              # emotion2vec+ large (local export)
make datasets            # FLEURS es/en, MESD es, CREMA-D en — pinned by commit
make bench               # evoice/stt/ops/bench/*.toml × datasets, file and live modes
make annex OUT=docs/history/<date>-<topic>   # report.md, summary.csv, charts.md
```

`e-voice bench <manifest.jsonl> --out results.jsonl [--mode file|live] [--streams N] [--limit N]` runs one
configuration. It writes one JSON line per item and a closing summary:

- per item: hypothesis, references, per-segment spans and latency;
- summary: CPU per audio second, throughput, peak RSS, and the full settings.

| Mode | Mirrors | Measures |
|---|---|---|
| `file` | `/v1/audio/transcriptions` and the other uploads | WER/CER, RTF, CPU, throughput, memory |
| `live` | `/v1/stream` and the other sockets, audio paced in real time | time from end of speech to the final, and to the first partial |

**Scoring** (in `eval/`, Python):

- WER and CER, exact and accent-folded;
- emotion accuracy, macro-F1 and angry recall against acted labels.

Each `evoice/stt/ops/bench/*.toml` overrides only what it measures; a test checks that every one of them loads
and validates.

## Results

| Round | Covers |
|---|---|
| [2026-10 · backends](history/2026-10-backends/index.md) | Canary, Cohere, Whisper, Kroko, LID, GTCRN, emotion2vec+ large — with charts and every run |
| [2026-10 · CPU benchmark](history/2026-10-benchmark.md) | Nemotron chunks, Parakeet, AGC, VAD, threads, spinning, SER base, wake word calibration |
| [2026-10 · TTS](history/2026-10-tts.md) | Pocket TTS on our loop vs sherpa, alternatives, latency, scaling, intelligibility, similarity |

## Headline

| Use | Pick | WER es / en | Cost |
|---|---|---|---|
| live, streaming partials | Nemotron 1120 ms | 8.4 / 11.4 % | 0.41 CPU s/s, final ~0.74 s |
| files, default | Parakeet | 4.5 / 8.1 % | 0.25 CPU s/s, 25× real time |
| files, English quality | Whisper turbo (`model=whisper`) | 4.9 / 5.6 % | 1.68 CPU s/s, 7× |
| emotion | emotion2vec+ base | CREMA 66 % · angry recall 88 % | 0.27 CPU s/s |

## TTS

`make bench SERVICE=tts` runs every `evoice/tts/ops/bench/*.toml` over the FLEURS texts (the STT manifests'
`text` field) through the service runner: at 1 stream with the audio kept and speaker similarity scored,
then at 4 and 8 concurrent streams. The **round trip** closes it: `e-voice bench` (STT, Parakeet)
transcribes each run's own audio, so intelligibility is the WER of what an ASR hears against the text
that was spoken. `make report SERVICE=tts` joins everything.

`e-voice-tts bench <manifest.jsonl> --out results.jsonl [--streams N] [--limit N] [--voice ID] [--wavs]
[--reference clip]` runs one configuration; per item it records time to first audio (voice priming
included), RTF, chunks and similarity; the summary adds throughput, CPU and RSS.

**Similarity** is the cosine of WeSpeaker ResNet34 (VoxCeleb) embeddings between each output and the
reference clip the voice was learned from (`pull speaker-resnet34`).

### Headline (2026-10-02, i9-14900K, voice learned from a 20 s sample)

| Config | WER es / en (round trip) | Similarity es / en | First audio | RTF, 1 stream |
|---|---|---|---|---|
| **base int8** (default) | 9.9 / 13.6 % | 0.69 / 0.72 | 158 ms | 0.33 |
| base fp32 | 9.8 / 12.1 % | 0.71 / 0.74 | 259 ms | 0.46 |
| large int8 (Spanish 24 layers) | **7.2** % / — | 0.70 / — | 558 ms | 0.96 |
| *real human speech (FLEURS)* | *4.5 / 8.1 %* | | | |

- Sampling (`temperature = 0.3`) makes runs differ: two runs of the same base model gave 9.9 and 12.9 %
  Spanish WER on these 40 items, so read differences under ~3 points as noise. int8 costs nothing
  measurable against fp32 and is 1.4× faster.
- `large` is clearly more intelligible in Spanish, at real time on one stream only.
- Aggregate throughput tops out at ≈ 6× real time from 4 streams on (memory bandwidth: the backbone
  copies its full KV cache every frame) — see [the TTS round](history/2026-10-tts.md) for the scaling
  table and the next step.
