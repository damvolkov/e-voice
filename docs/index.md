# e-voice

CPU-first speech service in Rust. One pipeline — denoise → wake word → VAD → language ID → ASR ∥ SER →
join — behind native, OpenAI, Deepgram and ElevenLabs compatible APIs. Spanish and English, an emotion
on every segment, no GPU.

```mermaid
flowchart LR
    audio([audio]) --> denoise[denoise] --> gain[AGC] --> ww[wake word] --> vad[VAD]
    vad -- segment --> lid[language ID] --> asr[ASR]
    vad -- segment --> ser[SER]
    vad -. live audio .-> asr
    asr --> join{{join}}
    ser --> join
    join --> final([final: text · lang · emotion])
```

Every node is a port with swappable backends chosen in [configuration](guide/configuration.md); the
defaults are what the [benchmark](benchmark.md) measured best on CPU.

| Node | Default | Alternatives |
|---|---|---|
| denoise | off | GTCRN |
| ww | off | sherpa KWS (`hey eager`), openWakeWord |
| vad | Silero | TEN |
| lid | off | Whisper tiny |
| asr (live) | Nemotron 3.5 streaming | Kroko, Parakeet, Canary, Cohere, Whisper |
| asr (files) | Parakeet TDT v3 | Canary, Cohere, Whisper, the live backend |
| ser | emotion2vec+ base | emotion2vec+ large, off |

## Quickstart

```bash
make install             # OS packages, pinned Rust, uv, prek, llvm-cov, git hooks
make setup               # sherpa-onnx libraries + the models the config declares
make serve               # gateway on :5500 — reference at http://127.0.0.1:5500/docs
make stt ARGS="--flat"   # another terminal: talk; Ctrl+C to finish
```

Or in a container: `make up` (see [Docker](guide/docker.md)). Clients point at `127.0.0.1:5500` either
way.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb"), language="es").text)
```
