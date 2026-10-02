use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

pub const RATE: u32 = 16_000;

/// Mono f32 samples at [`RATE`], shared between nodes without copying.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Audio(Arc<[f32]>);

impl Audio {
    /// Number of samples spanning `duration` at [`RATE`].
    #[must_use]
    pub fn length(duration: Duration) -> u64 {
        u64::try_from(duration.as_micros().saturating_mul(u128::from(RATE)) / 1_000_000).unwrap_or(u64::MAX)
    }
}

impl From<Vec<f32>> for Audio {
    fn from(samples: Vec<f32>) -> Self {
        Self(samples.into())
    }
}

impl Deref for Audio {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.0
    }
}
