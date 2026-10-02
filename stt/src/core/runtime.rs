use std::sync::OnceLock;

const SHERPA: &str = "e-voice-sherpa.conf";
const SPINLESS: &str =
    "SessionConfig.session.intra_op.allow_spinning=0\nSessionConfig.session.inter_op.allow_spinning=0\n";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("cannot bind ort to the shared onnxruntime: {0}")]
pub struct RuntimeError(String);

/// Native inference stack loaded by the process: one onnxruntime, shared by sherpa and ort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runtime {
    pub sherpa: &'static str,
    pub git: &'static str,
    pub onnxruntime: &'static str,
}

impl Runtime {
    /// Binds ort to the onnxruntime sherpa already loaded (once per process) and reports versions.
    ///
    /// # Errors
    /// The shared library cannot be opened by ort.
    pub fn probe() -> Result<Self, RuntimeError> {
        static BOUND: OnceLock<Result<(), RuntimeError>> = OnceLock::new();
        BOUND
            .get_or_init(|| {
                let library = format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                );
                let builder = ort::init_from(library).map_err(|error| RuntimeError(error.to_string()))?;
                builder.with_name("e-voice").commit();
                Ok(())
            })
            .clone()?;
        Ok(Self {
            sherpa: sherpa_onnx::version(),
            git: sherpa_onnx::git_sha1(),
            onnxruntime: sherpa_onnx::onnxruntime_version(),
        })
    }
}

impl Runtime {
    /// sherpa-onnx provider string: CPU, with onnxruntime threads that sleep instead of spinning
    /// between calls, so idle and paced streams cost no CPU. Falls back to plain `cpu` if the
    /// options file cannot be written.
    #[must_use]
    pub fn provider() -> String {
        static PROVIDER: OnceLock<String> = OnceLock::new();
        PROVIDER
            .get_or_init(|| {
                let path = std::env::temp_dir().join(SHERPA);
                match std::fs::write(&path, SPINLESS) {
                    Ok(()) => format!("cpu:{}", path.display()),
                    Err(error) => {
                        tracing::warn!(%error, path = %path.display(), "runtime.spinning");
                        "cpu".to_owned()
                    }
                }
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::runtime::Runtime;

    #[test]
    fn test_probe_loads_pinned_libraries_once() {
        let runtime = Runtime::probe().unwrap();
        assert_eq!(runtime.sherpa, "1.13.8");
        assert!(runtime.onnxruntime.starts_with("1.28"));
        assert_eq!(Runtime::probe().unwrap(), runtime);
    }

    #[test]
    fn test_provider_points_at_spinless_options() {
        let provider = Runtime::provider();
        let path = provider.strip_prefix("cpu:").unwrap();
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("intra_op.allow_spinning=0")
        );
    }
}
