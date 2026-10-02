# ElevenLabs

`POST /v1/speech-to-text`, multipart:

| Field | Values |
|---|---|
| `file` | any supported container |
| `model_id` | any (accepted for compatibility) |
| `language_code` | `es`/`en` or `spa`/`eng` |
| `emotion` | `field`, `tag`, `off` (extension) |

Response:

```json
{"language_code": "es", "language_probability": 1.0, "text": "…", "words": [{"text": "…", "start": 1.0, "end": 3.2}], "emotion": {…}}
```

- `words` holds one entry per recognized segment; word-level timing is not produced.
- Errors use `{"detail": {"status", "message"}}`.
