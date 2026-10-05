#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod asr;
mod denoise;
mod fake;
mod fixture;
mod lid;
mod pipeline;
mod runner;
mod ser;
mod session;
mod vad;
mod ww;
