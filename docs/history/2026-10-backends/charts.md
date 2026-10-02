### WER · FLEURS es · files

```mermaid
xychart-beta horizontal
    title "WER · FLEURS es · files"
    x-axis ["parakeet", "whisper", "cohere", "canary", "canary-lid", "ten", "nemotron-1120", "spin-nemotron-1120", "noagc-nemotron-1120", "kroko", "noser", "threads1", "noagc", "nemotron-160", "nemotron-560", "gtcrn"]
    y-axis "WER %" 0 --> 10.26
    bar [4.48, 4.9, 6.01, 7.43, 7.48, 8.38, 8.44, 8.48, 8.59, 8.65, 8.69, 8.69, 8.71, 8.82, 8.84, 8.92]
```

### WER · FLEURS en · files

```mermaid
xychart-beta horizontal
    title "WER · FLEURS en · files"
    x-axis ["whisper", "cohere", "parakeet", "canary-lid", "canary", "nemotron-1120", "nemotron-560", "noser", "threads1", "kroko", "nemotron-160", "gtcrn", "ten", "noagc", "noagc-nemotron-1120", "spin-nemotron-1120"]
    y-axis "WER %" 0 --> 26.24
    bar [5.64, 6.58, 8.14, 8.49, 8.49, 11.4, 11.7, 11.7, 11.7, 12.65, 12.84, 12.95, 14.79, 22.49, 22.82, 22.82]
```

### CPU seconds per audio second · FLEURS es · files

```mermaid
xychart-beta horizontal
    title "CPU seconds per audio second · FLEURS es · files"
    x-axis ["kroko", "parakeet", "noser", "threads1", "canary", "nemotron-1120", "noagc-nemotron-1120", "canary-lid", "gtcrn", "noagc", "nemotron-560", "ten", "spin-nemotron-1120", "cohere", "nemotron-160", "whisper"]
    y-axis "CPU s / s" 0 --> 1.93
    bar [0.17, 0.25, 0.33, 0.38, 0.41, 0.41, 0.41, 0.42, 0.45, 0.59, 0.6, 0.6, 0.94, 0.95, 1.38, 1.68]
```

### Throughput · FLEURS es · files

```mermaid
xychart-beta horizontal
    title "Throughput · FLEURS es · files"
    x-axis ["nemotron-160", "whisper", "threads1", "cohere", "ten", "spin-nemotron-1120", "nemotron-560", "noagc", "gtcrn", "canary", "noagc-nemotron-1120", "canary-lid", "nemotron-1120", "noser", "parakeet", "kroko"]
    y-axis "times real time" 0 --> 31.54
    bar [6.16, 6.86, 8.6, 11.68, 14.4, 14.64, 15.1, 15.45, 19.32, 19.98, 20.17, 20.3, 20.73, 22.29, 24.88, 27.43]
```

### Final after end of speech · p50 · live, 1 stream

```mermaid
xychart-beta horizontal
    title "Final after end of speech · p50 · live, 1 stream"
    x-axis ["noser", "kroko", "gtcrn", "noagc-nemotron-1120", "nemotron-1120", "nemotron-560", "noagc", "parakeet", "ten", "spin-nemotron-1120", "canary", "canary-lid", "nemotron-160", "threads1", "cohere", "whisper"]
    y-axis "ms" 0 --> 2459.95
    bar [623.17, 692.52, 704.54, 726.36, 738.75, 741.43, 746.98, 760.1, 792.37, 820.69, 882.26, 966.39, 1026.73, 1238.82, 1299.21, 2139.09]
```

### Final after end of speech · p95 · live, 4 streams

```mermaid
xychart-beta horizontal
    title "Final after end of speech · p95 · live, 4 streams"
    x-axis ["noser", "nemotron-1120", "noagc-nemotron-1120", "kroko", "nemotron-560", "noagc", "ten", "gtcrn", "parakeet", "spin-nemotron-1120", "nemotron-160", "canary", "canary-lid", "threads1", "cohere", "whisper"]
    y-axis "ms" 0 --> 4646.45
    bar [708.55, 947.19, 970.0, 999.5, 1011.88, 1020.42, 1039.17, 1065.6, 1072.57, 1367.38, 1382.22, 1435.94, 1553.14, 2084.77, 2520.42, 4040.39]
```

### Peak memory · FLEURS es · files

```mermaid
xychart-beta horizontal
    title "Peak memory · FLEURS es · files"
    x-axis ["noser", "kroko", "ten", "canary", "noagc-nemotron-1120", "noagc", "threads1", "nemotron-560", "spin-nemotron-1120", "nemotron-160", "nemotron-1120", "gtcrn", "canary-lid", "parakeet", "whisper", "cohere"]
    y-axis "MB" 0 --> 6730.44
    bar [1258.93, 1615.05, 2023.41, 2026.6, 2058.25, 2059.01, 2081.0, 2084.41, 2092.7, 2097.86, 2098.41, 2268.61, 2554.2, 2813.45, 4509.95, 5852.56]
```

### Emotion accuracy · CREMA-D en

```mermaid
xychart-beta horizontal
    title "Emotion accuracy · CREMA-D en"
    x-axis ["parakeet", "ser-large"]
    y-axis "accuracy %" 0 --> 84.81
    bar [65.75, 73.75]
```

### Emotion accuracy · MESD es

```mermaid
xychart-beta horizontal
    title "Emotion accuracy · MESD es"
    x-axis ["ser-large", "parakeet"]
    y-axis "accuracy %" 0 --> 28.53
    bar [20.16, 24.81]
```
