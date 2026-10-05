use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

use crate::config::log::{LogConfig, LogFormat};

#[derive(Debug, thiserror::Error)]
pub enum LoggerError {
    #[error("invalid log level directive: {0}")]
    Level(#[from] tracing_subscriber::filter::ParseError),
    #[error("a global subscriber is already installed")]
    Installed,
}

/// Process-wide structured logging, written off the caller's thread.
#[derive(Debug)]
pub struct Logger {
    _guard: WorkerGuard,
}

impl Logger {
    /// Installs the global subscriber; keep the returned value alive until shutdown so buffered lines flush.
    ///
    /// # Errors
    /// Invalid level directive, or a subscriber already installed.
    pub fn init(config: &LogConfig) -> Result<Self, LoggerError> {
        let filter = EnvFilter::try_new(&config.level)?;
        let (writer, guard) = tracing_appender::non_blocking(std::io::stderr());
        let builder = tracing_subscriber::fmt().with_env_filter(filter).with_writer(writer);
        let installed = match config.format {
            LogFormat::Json => builder.json().flatten_event(true).try_init(),
            LogFormat::Text => builder.try_init(),
        };
        installed.map_err(|_| LoggerError::Installed)?;
        Ok(Self { _guard: guard })
    }
}

#[cfg(test)]
mod tests {
    use crate::config::log::{LogConfig, LogFormat};
    use crate::logger::{Logger, LoggerError};

    #[test]
    fn test_init_rejects_bad_level_then_installs_once() {
        let bad = LogConfig {
            level: "e_voice=loud".to_owned(),
            format: LogFormat::Json,
        };
        assert!(matches!(Logger::init(&bad), Err(LoggerError::Level(_))));
        let good = LogConfig {
            level: "warn".to_owned(),
            format: LogFormat::Json,
        };
        let first = Logger::init(&good);
        assert!(first.is_ok());
        assert!(matches!(
            Logger::init(&LogConfig::default()),
            Err(LoggerError::Installed)
        ));
    }
}
