# Native stream

```
GET /v1/stream?lang=es&rate=48000&encoding=s16le&view=struct&emotion=field
```

| Parameter | Default | Values |
|---|---|---|
| `lang` | `stt.lang` | `es`, `en` |
| `rate` | 16000 | sample rate of the binary frames (resampled to 16 kHz) |
| `encoding` | `s16le` | `s16le`, `f32le`, mono |
| `view` | `struct` | `struct`: one JSON event per frame · `flat`: one text frame per final |
| `emotion` | `server.emotion` | `field`, `tag`, `off` |

**Client → server.** Binary frames carry PCM audio of any size. The text frame `{"type":"end"}`
drains pending segments. Closing the socket cancels the run.

**Server → client** (`struct`):

```json
{"type":"speech","segment":0,"state":"started","at":8288}
{"type":"partial","segment":0,"text":"No preguntes qué"}
{"type":"speech","segment":0,"state":"stopped","at":91840}
{"type":"final","segment":0,"span":{"start":8288,"end":91840},"lang":"es","text":"…","emotion":{…},"error":null}
{"type":"closed"}
```

- `at` and `span` are in samples at 16 kHz.
- `wake` events (`{"type":"wake","keyword":"hey eager","score":0.42}`) appear when a wake word is
  configured.
- Every segment gets exactly one `final`, in order. A late or failed SER yields `unknown`; a failed ASR
  sets `error`.
- On shutdown (SIGTERM) open segments are finalized before the socket closes.

**Close codes.** 1000 normal, 1003 unsupported parameters, 1011 pipeline failure.
