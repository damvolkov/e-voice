//! Lets binaries find the vendored sherpa-onnx and onnxruntime libraries: next to a dev build
//! (`data/ops/sherpa/lib`) and in an install (`bin/../lib`). Windows has no rpath, so the DLLs
//! are copied beside the binaries instead.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Releases are git tags; CI passes the tag as E_VOICE_VERSION (outside the EVOICE_ settings prefix,
    // since cargo run exports it), local builds use the crate version.
    println!("cargo:rerun-if-env-changed=E_VOICE_VERSION");
    let version = std::env::var("E_VOICE_VERSION")
        .ok()
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap_or_default());
    println!("cargo:rustc-env=E_VOICE_VERSION={}", version.trim_start_matches('v'));
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../../sherpa/lib:$ORIGIN/../lib"),
        "macos" => {
            for path in ["@loader_path/../../sherpa/lib", "@loader_path/../lib"] {
                println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{path}");
            }
        }
        "windows" => {
            let libs = std::env::var_os("SHERPA_ONNX_LIB_DIR").map(PathBuf::from);
            let profile = std::env::var_os("OUT_DIR")
                .map(PathBuf::from)
                .and_then(|out| out.ancestors().nth(3).map(Path::to_path_buf));
            if let (Some(libs), Some(profile)) = (libs, profile) {
                let dlls = std::fs::read_dir(libs)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|entry| entry.path());
                for dll in dlls.filter(|path| path.extension().is_some_and(|extension| extension == "dll")) {
                    if let Some(name) = dll.file_name() {
                        std::fs::copy(&dll, profile.join(name)).ok();
                    }
                }
            }
        }
        _ => {}
    }
}
