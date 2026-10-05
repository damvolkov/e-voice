use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

/// Mono f32 samples at the producing backend's rate, shared without copying.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Audio(Arc<[f32]>);

impl Audio {
    /// Time spanned by these samples at `rate`.
    #[must_use]
    pub fn duration(&self, rate: u32) -> Duration {
        Duration::from_secs_f64(f64::from(u32::try_from(self.0.len()).unwrap_or(u32::MAX)) / f64::from(rate.max(1)))
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
