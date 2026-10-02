# Wake word

The pipeline runs ungated by default (`ww.backend = "off"`). With a wake word, speech is ignored until
the keyword fires. The gate then stays open per `stt.pipeline.gate.close`:

| `close` | Gate stays open |
|---|---|
| `utterance` | for one utterance |
| `window` | until `idle` passes without speech |
| `session` | for the rest of the stream |

## Backends

`kws`
:   sherpa-onnx open-vocabulary keyword spotting (Zipformer, GigaSpeech, 3.3M). Any English phrase,
    tokenized at startup; no training. Default phrase: `hey eager`.

`oww`
:   openWakeWord with its pre-trained `hey_jarvis` classifier.

## Calibration

```bash
make wake WORD="hey eager"
```

There is nothing to train; calibration measures threshold × boost instead of guessing them:

- **Positives:** the phrase synthesized by three Piper voices × five carrier sentences × three speeds.
- **Negatives:** decoys ("The eagle landed…") plus real FLEURS speech.

It prints every pair's recall and false accepts per hour, then the `[stt.pipeline.ww]` block to paste.

!!! warning "Measured limits"
    - A single short word ("eager") reaches 76 % recall only at about 17 false accepts per hour.
      Two-word phrases are far safer: "hey eager" got 80 % recall with no false accepts in 24 minutes.
    - The GigaSpeech spotter detects the male synthetic voice reliably and the female ones rarely.
      Check recall on real voices with `ecli` before relying on it.
