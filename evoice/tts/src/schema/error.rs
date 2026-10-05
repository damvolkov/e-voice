/// Why audio clips could not become a voice.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TrainError {
    #[error("backend cannot learn voices")]
    Unsupported,
    #[error("no usable audio in the clips")]
    Empty,
    #[error("backend failure: {0}")]
    Backend(String),
}
