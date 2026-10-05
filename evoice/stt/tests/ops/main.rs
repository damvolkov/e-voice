#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

#[allow(dead_code)]
#[path = "../api/app.rs"]
mod app;
#[allow(dead_code)]
#[path = "../workflow/fake.rs"]
mod fake;
#[allow(dead_code)]
#[path = "../workflow/fixture.rs"]
mod fixture;
mod suite;
mod wake;
