use std::sync::Arc;

use e_voice_stt::config::vad::VadConfig;
use e_voice_stt::schema::audio::{Audio, RATE};
use e_voice_stt::schema::segment::SegmentSpan;
use e_voice_stt::workflow::vad::base::{Vad, VadEvent};

use crate::fixture;

fn run(vad: &Arc<dyn Vad>, audio: &[f32], chunk: usize) -> Vec<VadEvent> {
    let mut session = vad.open().unwrap();
    let mut events: Vec<VadEvent> = audio.chunks(chunk).flat_map(|chunk| session.push(chunk)).collect();
    events.extend(session.flush());
    events
}

fn spans(events: &[VadEvent]) -> Vec<SegmentSpan> {
    events
        .iter()
        .filter_map(|event| match event {
            VadEvent::End { span, .. } => Some(*span),
            VadEvent::Start { .. } => None,
        })
        .collect()
}

pub fn dialogue_yields_ordered_exact_segments(vad: &Arc<dyn Vad>) {
    let audio = fixture::dialogue();
    let events = run(vad, &audio, 1600);
    let spans = spans(&events);
    assert!(spans.len() >= 2, "expected one segment per utterance, got {spans:?}");
    assert!(spans.windows(2).all(|pair| pair[0].end <= pair[1].start), "{spans:?}");
    assert!(
        spans
            .iter()
            .all(|span| !span.is_empty() && span.end <= audio.len() as u64)
    );
    assert!(
        spans[0].start >= u64::from(RATE) / 2,
        "leading second of silence was taken as speech: {spans:?}"
    );
    let pad = Audio::length(VadConfig::default().pad) as usize;
    for event in &events {
        if let VadEvent::End { span, audio: samples } = event {
            let (from, to) = (
                (span.start as usize).saturating_sub(pad),
                (span.end as usize + pad).min(audio.len()),
            );
            assert_eq!(
                &samples[..],
                &audio[from..to],
                "segment audio must be the span plus {pad} samples each side"
            );
        }
    }
    let mut open = false;
    for event in &events {
        match event {
            VadEvent::Start { .. } => open = true,
            VadEvent::End { .. } => {
                assert!(open, "End without a preceding Start: {events:?}");
                open = false;
            }
        }
    }
}

pub fn silence_yields_nothing(vad: &Arc<dyn Vad>) {
    assert!(run(vad, &fixture::silence(5.0), 1600).is_empty());
}

pub fn segments_do_not_depend_on_chunking(vad: &Arc<dyn Vad>) {
    let audio = fixture::dialogue();
    let reference = spans(&run(vad, &audio, 4800));
    for chunk in [7, 160, 512, 16_000] {
        assert_eq!(spans(&run(vad, &audio, chunk)), reference, "chunk {chunk}");
    }
}

pub fn sessions_are_isolated(vad: &Arc<dyn Vad>) {
    let audio = fixture::dialogue();
    let reference = spans(&run(vad, &audio, 1600));
    let (mut left, mut right) = (vad.open().unwrap(), vad.open().unwrap());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for chunk in audio.chunks(1600) {
        a.extend(left.push(chunk));
        b.extend(right.push(chunk));
    }
    a.extend(left.flush());
    b.extend(right.flush());
    assert_eq!(spans(&a), reference);
    assert_eq!(spans(&b), reference);
}
