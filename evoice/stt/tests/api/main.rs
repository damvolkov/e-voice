#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod app;
mod deepgram;
mod docs;
mod elevenlabs;
#[allow(dead_code)]
#[path = "../workflow/fake.rs"]
mod fake;
mod health;
mod lifespan;
mod realtime;
mod stream;
mod transcriptions;
