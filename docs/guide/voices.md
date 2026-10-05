# Voices

A voice is learned from audio of one speaker and then used by id everywhere (`voice=damien`). Learning
is the backend's `train`; with Pocket TTS it is zero-shot cloning, so it takes seconds, not a GPU.

## Learn one

```bash
e-voice-tts voice add damien me.mp3 more.wav            # the ops tool: any container ffmpeg would call common
e-voice-tts voice add damien me.mp3 --text "Lo que digo en el clip."   # with its transcript
e-voice-tts voice list
e-voice-tts voice rm damien
e-voice-tts say "Hola, soy Demian." --voice damien --out hola.wav
```

The same through the API: `POST /v1/voices` (see [Text to speech](../api/tts.md#voices-v1voices)).
Set `tts.voice` to make one the default; the server primes it at startup so the first request pays
no encoding.

What makes a good sample:

- 10–30 s of clean speech of one speaker, no music, little reverberation; more than 30 s is ignored;
- several clips are joined in order;
- trailing silence is trimmed (upstream `end_on_pause`).

## The transcript

`--text` (API: the `text` form field) stores what the clips say. Pocket ignores it. Qwen3 clones in
context with it — reference codes plus their text in the prompt — when the clip is at most 10 s; longer
clips use the x-vector alone, which measured better on a 20 s sample anyway. NeuTTS cannot clone without
it. Write it as spoken; numbers in words help.

## How the backends use it

The voice stores the prepared prompt itself (24 kHz mono, ≤ 30 s, plus the transcript) in
`data/tts/voices/<id>.safetensors`. Each backend derives its own features when a stream opens, cached
per voice and model: Pocket primes its transformer with Mimi latents, Qwen3 computes the x-vector (and
the reference codes), NeuTTS the codec codes and the phonemized transcript. One voice therefore serves
every backend, size and fine-tuned checkpoint.

## Fine-tuning a model

Pocket TTS publishes its training code (`kyutai-labs/pocket-tts`, `training/`): `finetune.yaml` continues
from a released checkpoint, but it targets languages and domains — 100 h of transcribed audio at the
least. A single voice is the prompt's job, above. A fine-tuned checkpoint enters the service like the
released ones: export it to ONNX (`eval/export/pocket_tts.py`, on the GPU workstation), pin the bundle in
`evoice/tts/models.toml`, select it by model id. Python stays on the model side; the service is Rust.
