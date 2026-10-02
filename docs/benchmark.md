# Benchmark

The benchmark runs manifests through the configured pipeline. It uses the same `Runner` the gateway
serves, so the numbers are what clients get.

```bash
make setup ARGS=--all    # every model, including tool and test models
make export              # emotion2vec+ large (local export)
make datasets            # FLEURS es/en, MESD es, CREMA-D en — pinned by commit
make bench               # stt/ops/bench/*.toml × datasets, file and live modes
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

Each `stt/ops/bench/*.toml` overrides only what it measures; a test checks that every one of them loads
and validates.

## Results

| Round | Covers |
|---|---|
| [2026-10 · backends](history/2026-10-backends/index.md) | Canary, Cohere, Whisper, Kroko, LID, GTCRN, emotion2vec+ large — with charts and every run |
| [2026-10 · CPU benchmark](history/2026-10-benchmark.md) | Nemotron chunks, Parakeet, AGC, VAD, threads, spinning, SER base, wake word calibration |

## Headline

| Use | Pick | WER es / en | Cost |
|---|---|---|---|
| live, streaming partials | Nemotron 1120 ms | 8.4 / 11.4 % | 0.41 CPU s/s, final ~0.74 s |
| files, default | Parakeet | 4.5 / 8.1 % | 0.25 CPU s/s, 25× real time |
| files, English quality | Whisper turbo (`model=whisper`) | 4.9 / 5.6 % | 1.68 CPU s/s, 7× |
| emotion | emotion2vec+ base | CREMA 66 % · angry recall 88 % | 0.27 CPU s/s |
