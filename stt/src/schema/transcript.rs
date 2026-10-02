use std::collections::BTreeMap;

use crate::schema::audio::RATE;
use crate::schema::emotion::{Emotion, EmotionLabel};
use crate::schema::event::FinalEvent;
use crate::schema::lang::Lang;

/// A whole file's result: every final in order, plus the totals clients ask for.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub lang: Lang,
    pub samples: u64,
    pub segments: Vec<FinalEvent>,
}

impl Transcript {
    #[must_use]
    pub fn seconds(samples: u64) -> f64 {
        f64::from(u32::try_from(samples).unwrap_or(u32::MAX)) / f64::from(RATE)
    }

    #[must_use]
    pub fn duration(&self) -> f64 {
        Self::seconds(self.samples)
    }

    /// Segment texts joined by a space; with `tags`, each is prefixed by its emotion unless unknown.
    #[must_use]
    pub fn text(&self, tags: bool) -> String {
        self.segments
            .iter()
            .filter(|segment| !segment.text.is_empty())
            .map(|segment| match (tags, segment.emotion.label) {
                (false, _) | (true, EmotionLabel::Unknown) => segment.text.clone(),
                (true, label) => format!("[{}] {}", Self::tag(label), segment.text),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[must_use]
    pub fn tag(label: EmotionLabel) -> String {
        serde_json::to_value(label)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default()
    }

    /// Duration-weighted mean of the scored segments' distributions; `unknown` when none was scored.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn emotion(&self) -> Emotion {
        let weighted: Vec<(f32, &Emotion)> = self
            .segments
            .iter()
            .filter(|segment| !segment.emotion.scores.is_empty())
            .map(|segment| (Self::seconds(segment.span.len()) as f32, &segment.emotion))
            .collect();
        let total: f32 = weighted.iter().map(|(weight, _)| weight).sum();
        let mut scores: BTreeMap<EmotionLabel, f32> = BTreeMap::new();
        for (weight, emotion) in &weighted {
            for (label, score) in &emotion.scores {
                *scores.entry(*label).or_default() += score * weight / total.max(f32::EPSILON);
            }
        }
        let label = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(EmotionLabel::Unknown, |(label, _)| *label);
        let model = weighted.first().and_then(|(_, emotion)| emotion.model.clone());
        Emotion { label, scores, model }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use std::collections::BTreeMap;

    use crate::schema::emotion::{Emotion, EmotionLabel};
    use crate::schema::event::FinalEvent;
    use crate::schema::lang::Lang;
    use crate::schema::segment::{SegmentId, SegmentSpan};
    use crate::schema::transcript::Transcript;

    fn segment(id: u64, seconds: (u64, u64), text: &str, scores: &[(EmotionLabel, f32)]) -> FinalEvent {
        let scores: BTreeMap<EmotionLabel, f32> = scores.iter().copied().collect();
        let label = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(EmotionLabel::Unknown, |(label, _)| *label);
        FinalEvent {
            segment: SegmentId(id),
            span: SegmentSpan {
                start: seconds.0 * 16_000,
                end: seconds.1 * 16_000,
            },
            lang: Lang::Es,
            text: text.to_owned(),
            emotion: Emotion {
                label,
                scores,
                model: Some("m".into()),
            },
            error: None,
        }
    }

    fn transcript() -> Transcript {
        Transcript {
            lang: Lang::Es,
            samples: 10 * 16_000,
            segments: vec![
                segment(
                    0,
                    (0, 1),
                    "hola",
                    &[(EmotionLabel::Angry, 0.9), (EmotionLabel::Neutral, 0.1)],
                ),
                segment(1, (2, 5), "", &[]),
                segment(
                    2,
                    (5, 9),
                    "adiós",
                    &[(EmotionLabel::Angry, 0.2), (EmotionLabel::Neutral, 0.8)],
                ),
            ],
        }
    }

    #[test]
    fn test_text_joins_non_empty_segments_with_optional_tags() {
        assert_eq!(transcript().text(false), "hola adiós");
        assert_eq!(transcript().text(true), "[angry] hola [neutral] adiós");
    }

    #[test]
    fn test_emotion_is_duration_weighted() {
        let emotion = transcript().emotion();
        assert_eq!(emotion.label, EmotionLabel::Neutral);
        assert!((emotion.scores[&EmotionLabel::Angry] - (0.9 * 1.0 + 0.2 * 4.0) / 5.0).abs() < 1e-6);
        assert!((transcript().duration() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_emotion_without_scores_is_unknown() {
        let silent = Transcript {
            lang: Lang::En,
            samples: 0,
            segments: Vec::new(),
        };
        assert_eq!(silent.emotion(), Emotion::default());
    }
}
