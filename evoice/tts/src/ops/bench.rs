use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use e_voice_core::probe::Probe;
use e_voice_core::schema::lang::Lang;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::core::encode::AudioFormat;
use crate::ops::similarity::Similarity;
use crate::schema::event::Event;
use crate::schema::voice::VoiceState;
use crate::workflow::runner::{Request, Runner};

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("results: {0}")]
    Io(#[from] std::io::Error),
}

/// One manifest line; the STT manifests (`make datasets`) qualify through their `text` field.
#[derive(Debug, Clone, Deserialize)]
struct BenchItem {
    id: String,
    lang: Lang,
    text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchRecord {
    kind: &'static str,
    id: String,
    lang: Lang,
    text: String,
    audio_s: f64,
    wall_s: f64,
    /// Request to first audio, voice priming included.
    ttfa_ms: f64,
    rtf: f64,
    chunks: usize,
    /// Speaker similarity to `--reference` (cosine of speaker embeddings), when given.
    similarity: Option<f64>,
    failure: Option<String>,
}

/// One line of the STT manifest written next to the audio, for the intelligibility round trip.
#[derive(Debug, Serialize)]
struct RoundTrip<'a> {
    id: &'a str,
    audio: String,
    lang: Lang,
    text: &'a str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchSummary {
    kind: &'static str,
    pub items: usize,
    pub streams: usize,
    pub failures: usize,
    pub audio_s: f64,
    pub wall_s: f64,
    pub throughput_x: f64,
    pub cpu_s: f64,
    pub cpu_per_audio_s: f64,
    pub peak_rss_mb: f64,
    pub ttfa_p50_ms: f64,
    pub ttfa_p95_ms: f64,
    pub rtf_p50: f64,
    pub rtf_p95: f64,
    pub similarity_mean: Option<f64>,
    settings: serde_json::Value,
}

/// Synthesizes a manifest through the service runner, `streams` items at a time.
#[derive(Debug)]
pub struct Bench {
    pub runner: Runner,
    pub voice: Option<VoiceState>,
    pub streams: usize,
    pub wavs: bool,
    pub similarity: Option<Similarity>,
    pub settings: serde_json::Value,
}

impl Bench {
    // ##### PRIVATE #####

    fn common_quantile(values: &mut [f64], q: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let index = ((values.len().saturating_sub(1)) as f64 * q).round() as usize;
        values.get(index).copied().unwrap_or_default()
    }

    /// Writes `<id>.wav` and its STT manifest line.
    fn run_wav(
        dir: &Path,
        record: &BenchRecord,
        pcm: &[f32],
        rate: u32,
        manifest: &mut impl Write,
    ) -> std::io::Result<()> {
        let mut bytes = AudioFormat::Wav.header(rate);
        bytes.extend(AudioFormat::Wav.samples(pcm));
        let data = u32::try_from(pcm.len().saturating_mul(2)).unwrap_or(u32::MAX);
        bytes.splice(4..8, data.saturating_add(36).to_le_bytes());
        bytes.splice(40..44, data.to_le_bytes());
        let audio = format!("{}.wav", record.id.replace(['/', '\\'], "_"));
        std::fs::write(dir.join(&audio), bytes)?;
        let line = RoundTrip {
            id: &record.id,
            audio,
            lang: record.lang,
            text: &record.text,
        };
        writeln!(manifest, "{}", serde_json::to_string(&line).unwrap_or_default())
    }

    async fn run_item(&self, item: BenchItem) -> (BenchRecord, Vec<f32>) {
        let text = item.text.clone().unwrap_or_default();
        let rate = self.runner.caps().rate;
        let start = Instant::now();
        let mut record = BenchRecord {
            kind: "item",
            id: item.id.clone(),
            lang: item.lang,
            text: text.clone(),
            audio_s: 0.0,
            wall_s: 0.0,
            ttfa_ms: 0.0,
            rtf: 0.0,
            chunks: 0,
            similarity: None,
            failure: None,
        };
        let session = match self.runner.open(item.lang, self.voice.clone()).await {
            Ok(session) => session,
            Err(error) => {
                record.failure = Some(error.to_string());
                return (record, Vec::new());
            }
        };
        let (requests, inbox) = mpsc::channel(2);
        let (outbox, mut events) = mpsc::channel(256);
        requests.send(Request::Text(text)).await.ok();
        requests.send(Request::Close).await.ok();
        let runner = self.runner.clone();
        let task = tokio::spawn(async move { runner.run(session, inbox, outbox, CancellationToken::new()).await });
        let (mut first, mut pcm) = (None::<Duration>, Vec::new());
        while let Some(event) = events.recv().await {
            match event {
                Event::Audio { audio, .. } => {
                    first.get_or_insert_with(|| start.elapsed());
                    record.chunks = record.chunks.saturating_add(1);
                    pcm.extend_from_slice(&audio);
                }
                Event::End { error: Some(error), .. } => record.failure = Some(error.to_string()),
                _ => {}
            }
        }
        if let Ok(Err(error)) = task.await {
            record.failure = Some(error.to_string());
        }
        record.wall_s = start.elapsed().as_secs_f64();
        record.audio_s = crate::schema::audio::Audio::from(pcm.clone())
            .duration(rate)
            .as_secs_f64();
        record.ttfa_ms = first.unwrap_or_default().as_secs_f64() * 1000.0;
        record.rtf = record.wall_s / record.audio_s.max(f64::EPSILON);
        (record, pcm)
    }

    // ##### PUBLIC #####

    /// # Errors
    /// Unreadable or malformed manifest, or an unwritable results file.
    pub async fn run(&self, manifest: &Path, out: &Path, limit: Option<usize>) -> Result<BenchSummary, BenchError> {
        let file = std::fs::File::open(manifest).map_err(|error| BenchError::Manifest(error.to_string()))?;
        let items = std::io::BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str::<BenchItem>(&line).map_err(|error| BenchError::Manifest(error.to_string()))
            })
            .take(limit.unwrap_or(usize::MAX))
            .collect::<Result<Vec<_>, _>>()?;
        let (before, start) = (Probe::read(), Instant::now());
        let mut results: Vec<(BenchRecord, Vec<f32>)> =
            futures_util::stream::iter(items.into_iter().filter(|item| item.text.is_some()))
                .map(|item| self.run_item(item))
                .buffered(self.streams.max(1))
                .collect()
                .await;
        let (after, wall_s) = (Probe::read(), start.elapsed().as_secs_f64());
        let rate = self.runner.caps().rate;
        for (record, pcm) in &mut results {
            record.similarity = self
                .similarity
                .as_ref()
                .and_then(|similarity| similarity.score(pcm, rate));
        }
        if self.wavs {
            let dir: PathBuf = out.with_extension("wavs");
            std::fs::create_dir_all(&dir)?;
            let mut manifest = std::io::BufWriter::new(std::fs::File::create(dir.join("manifest.jsonl"))?);
            for (record, pcm) in results.iter().filter(|(record, _)| record.failure.is_none()) {
                Self::run_wav(&dir, record, pcm, rate, &mut manifest)?;
            }
            manifest.flush()?;
        }
        let records: Vec<BenchRecord> = results.into_iter().map(|(record, _)| record).collect();
        let mut out_file = std::io::BufWriter::new(std::fs::File::create(out)?);
        for record in &records {
            writeln!(out_file, "{}", serde_json::to_string(record).unwrap_or_default())?;
        }
        let ok: Vec<&BenchRecord> = records.iter().filter(|record| record.failure.is_none()).collect();
        let audio_s: f64 = ok.iter().map(|record| record.audio_s).sum();
        let cpu_s = after.cpu.saturating_sub(before.cpu).as_secs_f64();
        let mut ttfa: Vec<f64> = ok.iter().map(|record| record.ttfa_ms).collect();
        let mut rtf: Vec<f64> = ok.iter().map(|record| record.rtf).collect();
        let summary = BenchSummary {
            kind: "summary",
            items: records.len(),
            streams: self.streams,
            failures: records.len().saturating_sub(ok.len()),
            audio_s,
            wall_s,
            throughput_x: audio_s / wall_s.max(f64::EPSILON),
            cpu_s,
            cpu_per_audio_s: cpu_s / audio_s.max(f64::EPSILON),
            peak_rss_mb: after.peak_rss_mb,
            ttfa_p50_ms: Self::common_quantile(&mut ttfa, 0.5),
            ttfa_p95_ms: Self::common_quantile(&mut ttfa, 0.95),
            rtf_p50: Self::common_quantile(&mut rtf, 0.5),
            rtf_p95: Self::common_quantile(&mut rtf, 0.95),
            similarity_mean: {
                let scores: Vec<f64> = ok.iter().filter_map(|record| record.similarity).collect();
                #[allow(clippy::cast_precision_loss)]
                let mean = (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64);
                mean
            },
            settings: self.settings.clone(),
        };
        writeln!(out_file, "{}", serde_json::to_string(&summary).unwrap_or_default())?;
        out_file.flush()?;
        Ok(summary)
    }
}
