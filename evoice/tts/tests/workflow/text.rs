use e_voice_tts::workflow::text::Chunker;
use proptest::prelude::*;

fn stream(chunker: &mut Chunker, text: &str) -> Vec<String> {
    let mut out: Vec<String> = text.chars().flat_map(|c| chunker.push(&c.to_string())).collect();
    out.extend(chunker.flush());
    out
}

#[test]
fn test_push_releases_a_sentence_once_the_next_one_starts() {
    let mut chunker = Chunker::new(0, 200);
    assert!(chunker.push("Hola mundo. ").is_empty());
    assert_eq!(chunker.push("Esto"), ["Hola mundo."]);
    assert_eq!(chunker.flush(), ["Esto"]);
}

#[test]
fn test_push_merges_sentences_shorter_than_min() {
    let mut chunker = Chunker::new(20, 200);
    let out = stream(&mut chunker, "Sí. Vale. Entonces vamos ahora mismo. Fin");
    assert_eq!(out, ["Sí. Vale. Entonces vamos ahora mismo.", "Fin"]);
}

#[test]
fn test_push_keeps_decimals_inside_a_sentence() {
    let mut chunker = Chunker::new(0, 200);
    let out = stream(&mut chunker, "Cuesta 3.5 euros. Vale.");
    assert_eq!(out, ["Cuesta 3.5 euros.", "Vale."]);
}

#[test]
fn test_push_cuts_unpunctuated_text_at_clause_marks_then_spaces() {
    let mut chunker = Chunker::new(0, 20);
    let out = stream(&mut chunker, "uno dos tres, cuatro cinco seis siete ocho nueve diez");
    assert_eq!(out, ["uno dos tres,", "cuatro cinco seis", "siete ocho nueve", "diez"]);
}

#[test]
fn test_clear_forgets_buffered_text() {
    let mut chunker = Chunker::new(0, 200);
    chunker.push("Esto no se dirá");
    chunker.clear();
    assert!(chunker.flush().is_empty());
}

proptest! {
    #[test]
    fn prop_chunks_preserve_words_and_respect_max(
        words in prop::collection::vec("[a-zñáé]{1,9}[.,?]?", 1..60),
        cuts in prop::collection::vec(1_usize..12, 1..40),
        max in 12_usize..80,
    ) {
        let text = words.join(" ");
        let mut chunker = Chunker::new(10, max);
        let mut out = Vec::new();
        let mut rest = text.as_str();
        for cut in cuts.iter().cycle() {
            let at = rest.char_indices().nth(*cut).map_or(rest.len(), |(at, _)| at);
            let (delta, tail) = rest.split_at(at);
            out.extend(chunker.push(delta));
            rest = tail;
            if rest.is_empty() { break; }
        }
        out.extend(chunker.flush());
        prop_assert!(out.iter().all(|piece| piece.chars().count() <= max), "{out:?}");
        let joined = out.join(" ");
        prop_assert_eq!(joined.split_whitespace().collect::<Vec<_>>(), text.split_whitespace().collect::<Vec<_>>());
    }
}
