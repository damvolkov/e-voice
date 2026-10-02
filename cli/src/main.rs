#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod mic;
mod stt;
mod wav;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::mic::Mic;
use crate::stt::{SttArgs, SttError};

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Stream speech to the gateway and print transcripts as they arrive.
    Stt(SttArgs),
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
        CliCommand::Stt(args) => args.run().await,
        CliCommand::Devices => Mic::devices().map_err(SttError::from).map(|devices| {
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
