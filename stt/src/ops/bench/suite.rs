use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::core::audio::AudioFile;
use crate::ops::bench::manifest::{Manifest, ManifestItem};
use crate::ops::bench::probe::Probe;
use crate::ops::bench::record::{Record, RecordSegment, RecordSummary};
use crate::schema::audio::Audio;
use crate::schema::event::{Event, FinalEvent};
use crate::schema::segment::SegmentId;
use crate::schema::transcript::Transcript;
use crate::workflow::runner::{Runner, RunnerError, RunnerIntake};

const PACE: Duration = Duration::from_millis(100);

/// A final with its arrival and first-partial times since the stream started (live only).
type Timed = (FinalEvent, Option<Duration>, Option<Duration>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SuiteMode {
    /// Whole files through the lossless file intake, as `/v1/audio/transcriptions` does.
    File,
    /// Audio paced in real time through the live intake, as `/v1/stream` does.
    Live,
}

impl SuiteMode {
    const fn name(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Live => "live",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SuiteError {
    #[error("cannot write results: {0}")]
    Io(#[from] std::io::Error),
    #[error("cannot encode results: {0}")]
    Json(#[from] serde_json::Error),
}

/// Runs a manifest through one configuration with `streams` items in flight, writing one JSON line
/// per item and a closing summary.
#[derive(Debug, Clone)]
pub struct Suite {
    pub runner: Runner,
    pub mode: SuiteMode,
    pub streams: usize,
    pub settings: serde_json::Value,
}

impl Suite {
    // ##### PRIVATE #####

    fn common_segment(done: &FinalEvent, arrived: Option<Duration>, partial: Option<Duration>) -> RecordSegment {
        let (start, end) = (Transcript::seconds(done.span.start), Transcript::seconds(done.span.end));
        let lag = |at: Duration, from: f64| at.as_secs_f64().mul_add(1000.0, -from * 1000.0);
        RecordSegment {
            start,
            end,
            text: done.text.clone(),
            emotion: done.emotion.clone(),
            error: done.error.clone(),
            final_ms: arrived.map(|at| lag(at, end)),
            partial_ms: partial.map(|at| lag(at, start)),
        }
    }

    async fn run_live(&self, item: &ManifestItem, samples: Vec<f32>) -> Result<Vec<Timed>, RunnerError> {
        let (audio_tx, audio_rx) = mpsc::channel(64);
        let (events_tx, mut events_rx) = mpsc::channel(256);
        let runner = self.runner.clone();
        let lang = item.lang;
        let run = tokio::spawn(async move {
            runner
                .run(lang, RunnerIntake::Live, audio_rx, events_tx, CancellationToken::new())
                .await
        });
        let start = Instant::now();
        let chunk = usize::try_from(Audio::length(PACE)).unwrap_or(usize::MAX);
        let feed = tokio::spawn(async move {
            for (tick, piece) in (0u32..).zip(samples.chunks(chunk)) {
                let due = start.checked_add(PACE.saturating_mul(tick)).unwrap_or(start);
                tokio::time::sleep_until(due).await;
                if audio_tx.send(Audio::from(piece.to_vec())).await.is_err() {
                    return;
                }
            }
        });
        let (mut partials, mut segments) = (HashMap::<SegmentId, Duration>::new(), Vec::new());
        while let Some(event) = events_rx.recv().await {
            let at = start.elapsed();
            match event {
                Event::Partial(partial) => {
                    partials.entry(partial.segment).or_insert(at);
                }
                Event::Final(done) => {
                    let partial = partials.get(&done.segment).copied();
                    segments.push((done, Some(at), partial));
                }
                Event::Wake(_) | Event::Speech(_) | Event::Closed => {}
            }
        }
        feed.abort();
        run.await.map_err(|_| RunnerError::Incomplete)??;
        Ok(segments)
    }

    async fn run_item(&self, item: ManifestItem) -> Record {
        let path = item.audio.clone();
        let decoded = tokio::task::spawn_blocking(move || {
            let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_owned);
            AudioFile::decode(bytes, extension.as_deref()).map_err(|error| error.to_string())
        })
        .await
        .map_err(|error| error.to_string())
        .and_then(|decoded| decoded);
        let audio_s = decoded
            .as_ref()
            .map_or(0.0, |samples| Transcript::seconds(samples.len() as u64));
        let started = Instant::now();
        let outcome = match (decoded, self.mode) {
            (Err(error), _) => Err(error),
            (Ok(samples), SuiteMode::File) => self
                .runner
                .transcribe(item.lang, samples)
                .await
                .map(|transcript| transcript.segments.into_iter().map(|done| (done, None, None)).collect())
                .map_err(|error| error.to_string()),
            (Ok(samples), SuiteMode::Live) => self.run_live(&item, samples).await.map_err(|error| error.to_string()),
        };
        let wall_s = started.elapsed().as_secs_f64();
        let timed: Vec<Timed> = outcome.as_ref().map_or_else(|_| Vec::new(), Clone::clone);
        let segments = timed
            .iter()
            .map(|(done, arrived, partial)| Self::common_segment(done, *arrived, *partial))
            .collect();
        let finals = timed.into_iter().map(|(done, ..)| done).collect();
        let transcript = Transcript {
            lang: item.lang,
            samples: 0,
            segments: finals,
        };
        Record {
            id: item.id,
            lang: item.lang,
            mode: self.mode.name(),
            audio_s,
            wall_s,
            rtf: wall_s / audio_s.max(f64::EPSILON),
            text: transcript.text(false),
            emotion: transcript.emotion(),
            ref_text: item.text,
            ref_emotion: item.emotion,
            segments,
            failure: outcome.err(),
        }
    }

    // ##########################################################

    // ##### PUBLIC #####

    /// # Errors
    /// The results file cannot be written.
    pub async fn run(&self, manifest: Manifest, out: &Path) -> Result<RecordSummary, SuiteError> {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = std::io::BufWriter::new(std::fs::File::create(out)?);
        let (before, started) = (Probe::read(), Instant::now());
        let items = manifest.items.len();
        let mut records = futures_util::stream::iter(manifest.items)
            .map(|item| self.run_item(item))
            .buffer_unordered(self.streams.max(1));
        let mut audio_s = 0.0;
        while let Some(record) = records.next().await {
            audio_s += record.audio_s;
            if let Some(failure) = &record.failure {
                tracing::warn!(id = %record.id, %failure, "bench.failed");
            }
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
        }
        let (after, wall_s) = (Probe::read(), started.elapsed().as_secs_f64());
        let cpu_s = after.cpu.saturating_sub(before.cpu).as_secs_f64();
        let summary = RecordSummary {
            kind: "summary",
            mode: self.mode.name(),
            items,
            streams: self.streams.max(1),
            audio_s,
            wall_s,
            cpu_s,
            cpu_per_audio_s: cpu_s / audio_s.max(f64::EPSILON),
            throughput_x: audio_s / wall_s.max(f64::EPSILON),
            peak_rss_mb: after.peak_rss_mb,
            settings: self.settings.clone(),
        };
        serde_json::to_writer(&mut writer, &summary)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(summary)
    }
}
