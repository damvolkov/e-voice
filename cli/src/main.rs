#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod mic;
mod speaker;
mod stt;
mod tts;
mod wav;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::mic::Mic;
use crate::stt::{SttArgs, SttError};
use crate::tts::{TtsArgs, TtsError};

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Stt(#[from] SttError),
    #[error(transparent)]
    Tts(#[from] TtsError),
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Stream speech to the gateway and print transcripts as they arrive.
    Stt(SttArgs),
    /// Stream text to the TTS gateway and play the voice as it is generated.
    Tts(TtsArgs),
    /// List audio inputs; `*` marks the system default.
    Devices,
}

/// Terminal tester for the e-voice gateway.
#[derive(Debug, Parser)]
#[command(name = "ecli", version)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[tokio::main]
async fn main() -> ExitCode {
    let outcome = match Cli::parse().command {
        CliCommand::Stt(args) => args.run().await.map_err(CliError::from),
        CliCommand::Tts(args) => args.run().await.map_err(CliError::from),
        CliCommand::Devices => Mic::devices()
            .map_err(|error| CliError::from(SttError::from(error)))
            .map(|devices| {
                for (name, chosen) in devices {
                    println!("{} {name}", if chosen { "*" } else { " " });
                }
            }),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ecli: {error}");
            ExitCode::FAILURE
        }
    }
}
