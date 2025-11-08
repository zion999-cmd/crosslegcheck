#!/usr/bin/env zsh
# build_with_cubeide_toolchain.sh
# Attempt to locate STM32CubeIDE's bundled arm-none-eabi toolchain (macOS)
# and run make in the cubeide directory.

set -euo pipefail

# Allow user override
if [[ -n "${ARM_TOOLCHAIN_DIR:-}" ]]; then
  TOOLCHAIN_BIN="$ARM_TOOLCHAIN_DIR"
else
  # Try to find CubeIDE installation path typical on macOS
  if [[ -d "/Applications/STM32CubeIDE.app" ]]; then
    CUBEIDE_APP="/Applications/STM32CubeIDE.app"
  else
    # try locate via mdfind/mdfind may not be available; allow fallback
    CUBEIDE_APP=""
    if command -v mdfind >/dev/null 2>&1; then
      CUBEIDE_APP=$(mdfind "kMDItemFSName == 'STM32CubeIDE.app'" | head -n1 || true)
    fi
  fi

  if [[ -n "$CUBEIDE_APP" && -d "$CUBEIDE_APP" ]]; then
    # Look for an embedded tools/bin directory under the app bundle
    # common plugin path prefix (versions may vary) - search for arm-none-eabi-gcc
    TOOLCHAIN_BIN="$(find "$CUBEIDE_APP" -type f -name arm-none-eabi-gcc -maxdepth 6 -print -quit 2>/dev/null || true)"
    if [[ -n "$TOOLCHAIN_BIN" ]]; then
      TOOLCHAIN_BIN="$(dirname "$TOOLCHAIN_BIN")"
    else
      TOOLCHAIN_BIN=""
    fi
  else
    TOOLCHAIN_BIN=""
  fi
fi

# If still empty, try system PATH
if [[ -z "$TOOLCHAIN_BIN" ]]; then
  if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    TOOLCHAIN_BIN="$(dirname "$(command -v arm-none-eabi-gcc)")"
  fi
fi

if [[ -z "$TOOLCHAIN_BIN" ]]; then
  echo "Could not locate arm-none-eabi toolchain."
  echo "Please install STM32CubeIDE or GNU Arm Embedded Toolchain and re-run."
  echo "If you have a toolchain, you can set ARM_TOOLCHAIN_DIR to the 'bin' directory containing arm-none-eabi-gcc."
  echo "Example: ARM_TOOLCHAIN_DIR=/path/to/gcc-arm-none-eabi/bin ./build_with_cubeide_toolchain.sh"
  exit 2
fi

echo "Using toolchain bin directory: $TOOLCHAIN_BIN"
export PATH="$TOOLCHAIN_BIN:$PATH"

# Compute parallel jobs
if [[ -n "${BUILD_JOBS:-}" ]]; then
  JOBS=$BUILD_JOBS
else
  if command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
  else
    JOBS=4
  fi
fi

pushd "$(dirname "$0")/cubeide" >/dev/null

echo "Running: make clean && make -j$JOBS all"
make clean || true
make -j$JOBS all
RET=$?
if [[ $RET -ne 0 ]]; then
  echo "Build failed with exit code $RET"
  popd >/dev/null
  exit $RET
fi

echo "Build succeeded. Artifacts are in cubeide/Debug (or project-specific output dirs)."
popd >/dev/null

exit 0
