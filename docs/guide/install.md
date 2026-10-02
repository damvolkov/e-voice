# Install

```bash
make install
```

`install` runs three targets, each usable alone:

| Target | Does |
|---|---|
| `system` | OS build packages: C/C++ compiler, cmake, pkg-config, curl, bzip2, and ALSA headers on Linux |
| `toolchain` | rustup if missing, the toolchain pinned in `rust-toolchain.toml` (1.94.0 with clippy, rustfmt, llvm-tools), uv if missing |
| `tools` | prek and cargo-llvm-cov, then the git hooks |

The platform is detected; force it with `OS`:

=== "Ubuntu / Debian"

    ```bash
    make install OS=ubuntu      # apt; sudo only when not root
    ```

=== "macOS"

    ```bash
    make install OS=mac         # Homebrew (bash, cmake, pkg-config) + Xcode command line tools
    ```

=== "Windows"

    ```bash
    make install OS=windows     # from Git Bash: winget MSVC build tools + CMake
    ```

Then:

```bash
make setup                  # sherpa-onnx shared libraries + the configured models
make setup ARGS=--all       # every model in the manifest (tests, benchmark, wake calibration)
```

`make sherpa` vendors the pinned sherpa-onnx 1.13.8 shared libraries (with onnxruntime 1.28.2) for
linux-x64, macOS universal2 or windows-x64 into `data/stt/ops/sherpa`; binaries find them through an
rpath (`$ORIGIN` / `@loader_path`) or, on Windows, a copy beside the executable.

!!! note "Platforms"
    CI builds and tests Linux x86-64. macOS and Windows are wired (packages, libraries, linker paths,
    audio host) but not exercised by CI.

## Everything lives in `data/`

The repository ignores `data/` except its `.gitkeep`:

```
data/
├── ops/target         cargo build output
├── ops/site           built documentation
└── stt/
    ├── models         pipeline models (the Docker volume)
    └── ops/           sherpa libraries, tool and test models, exports, datasets, results
```
