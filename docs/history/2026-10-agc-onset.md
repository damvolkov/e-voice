# 2026-10 · AGC lifted pre-speech noise

**Symptom.** Live transcription lost the first words when speech began within about a second of the
stream opening. A Pocket TTS clip whose speech starts at 0.25 s was the reproduction: the VAD fired at
1.41 s and the final began "Esta prueba…" instead of "Hola Damien, esta prueba…".

**Cause.** `AudioGain` started with no tracked peak, so it applied its full 40 dB to the first
samples.

- The clip's 0.3 s of vocoder residue (−50 dBFS peak) left the AGC at 0.10 RMS. That is as loud as the
  speech that followed (0.14–0.20 RMS).
- Silero lost the contrast between silence and speech and missed the first phrase.
- With the AGC off, the onset came at 0.36 s.
- The same mechanism applied after any long pause, because the tracked peak decays to the noise and the
  gain rises back to its maximum.

**Fix.** The AGC now has a gain ceiling that depends on the noise.

- It tracks a causal noise floor: the quietest 10 ms block peak over 1.5 s, with digital silence
  ignored.
- The gain never lifts that floor above `stt.pipeline.gain.noise` (default −40 dBFS).
- Until a floor is known, nothing is boosted.

An absolute noise gate would not work. FLEURS-en speech peaks have a median of −41 dBFS and a p5 of
−49 dBFS, which is as low as that noise. Only the noise floor of each input separates them.

**Measured** (Nemotron 1120 ms, FLEURS; live runs paced in real time, 40 utterances):

| | before | ceiling −45 | **ceiling −40** | ceiling −35 | AGC off |
|---|---|---|---|---|---|
| WER en, files (200) | 11.4 % | 11.8 % | **11.4 %** | 11.4 % | 22.8 % |
| WER es, files (200) | 8.4 % | 8.3 % | **8.2 %** | 8.4 % | 8.6 % |
| WER en, live | 9.1 % | 9.8 % | **9.2 %** | 9.0 % | 21.0 % |
| WER es, live | 6.7 % | 7.0 % | **7.0 %** | 6.9 % | 6.4 % |
| first word lost en, live | 7/40 | 8/40 | **6/40** | 6/40 | 22/40 |
| first word lost es, live | 6/40 | 7/40 | **5/40** | 5/40 | 8/40 |
| Pocket clip onset | 1.41 s | 0.33 s | **0.33 s** | 0.33 s | 0.36 s |

The live WER differences are within the noise of 40 utterances. Synthetic residue (white, brown, or
murmured speech at −50 dBFS) does not trigger the old failure, so the regression rests on unit tests
of the mechanism plus the recorded clip, which is kept local because it is a personal voice.
