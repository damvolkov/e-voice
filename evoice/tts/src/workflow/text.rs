use unicode_segmentation::UnicodeSegmentation;

/// Streaming text → sentences, sized for synthesis.
///
/// Text arrives in arbitrary deltas (e.g. LLM tokens). A sentence leaves once a later delta proves it
/// complete (UAX #29 bounds) and it holds at least `min` chars; shorter ones merge with the next.
/// Anything beyond `max` chars, finished or not, is cut at the last clause mark or space before it.
#[derive(Debug, Clone)]
pub struct Chunker {
    min: usize,
    max: usize,
    buffer: String,
}

impl Chunker {
    // ##### PRIVATE #####

    /// Byte index where `text` should be cut so the head keeps at most `max` chars.
    fn common_cut(text: &str, max: usize) -> Option<usize> {
        let end = text.char_indices().nth(max).map(|(at, _)| at)?;
        let head = text.get(..end)?;
        let soft = head
            .rfind([',', ';', ':'])
            .map(|at| at.saturating_add(1))
            .or_else(|| head.rfind(char::is_whitespace))
            .filter(|&at| at > 0);
        Some(soft.unwrap_or(end))
    }

    /// Splits a finished piece into trimmed, non-empty parts of at most `max` chars.
    fn common_split(&self, mut text: &str, out: &mut Vec<String>) {
        while let Some((head, tail)) = Self::common_cut(text, self.max).and_then(|at| text.split_at_checked(at)) {
            out.extend(Some(head.trim()).filter(|part| !part.is_empty()).map(str::to_owned));
            text = tail;
        }
        out.extend(Some(text.trim()).filter(|part| !part.is_empty()).map(str::to_owned));
    }

    // ##### PUBLIC #####

    #[must_use]
    pub fn new(min: usize, max: usize) -> Self {
        Self {
            min,
            max: max.max(1),
            buffer: String::new(),
        }
    }

    /// Feeds a delta; returns the sentences it completed, in order.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut out = Vec::new();
        let mut start = 0;
        let bounds: Vec<usize> = self
            .buffer
            .split_sentence_bound_indices()
            .map(|(at, _)| at)
            .skip(1)
            .collect();
        for bound in bounds {
            if let Some(piece) = self
                .buffer
                .get(start..bound)
                .filter(|piece| piece.trim().chars().count() >= self.min)
            {
                self.common_split(piece, &mut out);
                start = bound;
            }
        }
        while let Some(at) = self
            .buffer
            .get(start..)
            .and_then(|tail| Self::common_cut(tail, self.max))
        {
            let cut = start.saturating_add(at);
            out.extend(
                self.buffer
                    .get(start..cut)
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned),
            );
            start = cut;
        }
        self.buffer.drain(..start);
        out
    }

    /// Releases whatever is buffered, complete or not.
    pub fn flush(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buffer);
        self.common_split(&rest, &mut out);
        out
    }

    /// Drops buffered text without speaking it.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}
