use serde::{Deserialize, Serialize};

/// Wire format of synthesized audio. `wav` is 16-bit PCM behind a streaming header (sizes unknown,
/// set to the maximum, as every streaming WAV writer does).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    #[default]
    Pcm,
    F32,
    Wav,
}

impl AudioFormat {
    /// Little-endian samples, clamped to [-1, 1].
    #[must_use]
    pub fn samples(self, samples: &[f32]) -> Vec<u8> {
        let clamped = samples.iter().map(|sample| sample.clamp(-1.0, 1.0));
        match self {
            Self::F32 => clamped.flat_map(f32::to_le_bytes).collect(),
            Self::Pcm | Self::Wav => clamped
                .flat_map(|sample| {
                    #[allow(clippy::cast_possible_truncation)]
                    let value = (sample * 32_767.0).round() as i16;
                    value.to_le_bytes()
                })
                .collect(),
        }
    }

    /// Bytes sent before the first samples: a WAV header for `wav`, nothing otherwise.
    #[must_use]
    pub fn header(self, rate: u32) -> Vec<u8> {
        match self {
            Self::Pcm | Self::F32 => Vec::new(),
            Self::Wav => [
                b"RIFF".as_slice(),
                &u32::MAX.to_le_bytes(),
                b"WAVEfmt ",
                &16_u32.to_le_bytes(),
                &1_u16.to_le_bytes(),
                &1_u16.to_le_bytes(),
                &rate.to_le_bytes(),
                &rate.saturating_mul(2).to_le_bytes(),
                &2_u16.to_le_bytes(),
                &16_u16.to_le_bytes(),
                b"data",
                &u32::MAX.to_le_bytes(),
            ]
            .concat(),
        }
    }

    #[must_use]
    pub const fn mime(self) -> &'static str {
        match self {
            Self::Pcm | Self::F32 => "application/octet-stream",
            Self::Wav => "audio/wav",
        }
    }
}
