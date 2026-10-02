//! Lets binaries find the vendored sherpa-onnx and onnxruntime libraries: next to a dev build
//! (`data/stt/ops/sherpa/lib`) and in an install (`bin/../lib`). Windows has no rpath, so the DLLs
//! are copied beside the binaries instead.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../../../stt/ops/sherpa/lib:$ORIGIN/../lib"),
        "macos" => {
            for path in ["@loader_path/../../../stt/ops/sherpa/lib", "@loader_path/../lib"] {
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
