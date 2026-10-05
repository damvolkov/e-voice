#!/usr/bin/env bash
# Vendors the sherpa-onnx shared libraries (with their onnxruntime) for this platform into
# data/ops/sherpa/lib, shared by every service. The version follows Cargo.lock; every archive is pinned by sha256.
set -euo pipefail

declare -A PINS=(
    [linux-x64-shared-lib]=3892d184be41027e18165e67f549cd4e4cdd8dcd73ac5579e97afd55e14e30b6
    [osx-universal2-shared-lib]=c1cca5b4a1867543e27d42fbea513983e73f629d0aa53b43b51555a71d2108d9
    [win-x64-shared-MD-Release-lib]=3c43d1efba780d7ae7f5d64cae08803dcecf417109a5f593684e536f3cbd5cf1
)
declare -A FLAVORS=(
    [Linux-x86_64]=linux-x64-shared-lib
    [Darwin-arm64]=osx-universal2-shared-lib
    [Darwin-x86_64]=osx-universal2-shared-lib
    [Windows-x86_64]=win-x64-shared-MD-Release-lib
)

KERNEL="$(uname -s)"
[[ "${KERNEL}" == MINGW* || "${KERNEL}" == MSYS* || "${KERNEL}" == CYGWIN* ]] && KERNEL=Windows
PLATFORM="${KERNEL}-$(uname -m)"
FLAVOR="${FLAVORS[${PLATFORM}]:-}"
[[ -n "${FLAVOR}" ]] || { echo "sherpa: no prebuilt shared libraries for ${PLATFORM}" >&2; exit 1; }

VERSION="$(grep -A1 'name = "sherpa-onnx-sys"' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')"
NAME="sherpa-onnx-v${VERSION}-${FLAVOR}"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VERSION}/${NAME}.tar.bz2"
DEST="data/ops/sherpa"
STAMP="${DEST}/version"

[[ -f "${STAMP}" && "$(cat "${STAMP}")" == "${VERSION} ${FLAVOR}" ]] && exit 0

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
curl -fsSL --retry 3 -o "${TMP}/archive.tar.bz2" "${URL}"
DIGEST="$( (command -v sha256sum >/dev/null && sha256sum "${TMP}/archive.tar.bz2") || shasum -a 256 "${TMP}/archive.tar.bz2")"
[[ "${DIGEST%% *}" == "${PINS[${FLAVOR}]}" ]] || { echo "sherpa: sha256 mismatch for ${NAME}" >&2; exit 1; }
tar -xjf "${TMP}/archive.tar.bz2" -C "${TMP}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
mv "${TMP}/${NAME}/lib" "${DEST}/lib"
echo "${VERSION} ${FLAVOR}" > "${STAMP}"
printf '\033[32msherpa %s (%s) ready in %s\033[0m\n' "${VERSION}" "${FLAVOR}" "${DEST}"
