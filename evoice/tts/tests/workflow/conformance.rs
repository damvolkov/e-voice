use std::time::{Duration, Instant};

use e_voice_core::schema::lang::Lang;
use e_voice_tts::schema::voice::VoiceState;
use e_voice_tts::workflow::synth::base::{CHUNK_LIMIT_MS, FIRST_LIMIT_MS, Synth};

/// A sentence long enough (~5 s of speech) for the first chunk to be clearly early.
pub const LONG: &str = "La síntesis en streaming entrega audio mientras la frase aún se genera, \
                        así el oyente escucha la primera palabra antes de que exista la última.";

/// The streaming contract every backend must pass to be registered: a short first chunk, bounded
/// chunks after it, more than one per long sentence, and the first one before a quarter of the total
/// synthesis time.
pub fn conform(synth: &dyn Synth, lang: Lang, voice: Option<&VoiceState>) -> Result<(), String> {
    let rate = synth.caps().rate;
    let (opening, limit) = (
        Duration::from_millis(FIRST_LIMIT_MS),
        Duration::from_millis(CHUNK_LIMIT_MS),
    );
    let mut session = synth.open(lang, voice).map_err(|error| error.to_string())?;
    let start = Instant::now();
    let mut first = None;
    let mut chunks = 0_usize;
    for chunk in session.speak(LONG) {
        let chunk = chunk.map_err(|error| error.to_string())?;
        first.get_or_insert_with(|| start.elapsed());
        chunks += 1;
        let span = chunk.duration(rate);
        let bound = if chunks == 1 { opening } else { limit };
        (span <= bound)
            .then_some(())
            .ok_or(format!("chunk {chunks} of {span:?} exceeds {bound:?}"))?;
    }
    let total = start.elapsed();
    let first = first.ok_or("no audio")?;
    (chunks > 1)
        .then_some(())
        .ok_or("sentence came out as a single chunk")?;
    (first * 4 <= total)
        .then_some(())
        .ok_or(format!("first chunk at {first:?} of {total:?}: not streaming"))
}
