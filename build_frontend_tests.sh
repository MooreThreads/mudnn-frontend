#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh.example"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

BUILD_DIR="${BUILD_DIR:-${MUDNN_FRONTEND_BUILD_DIR:-${SCRIPT_DIR}/build}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--build-dir BUILD_DIR]" >&2
      exit 2
      ;;
  esac
done

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DMUDNN_FRONTEND_BUILD_FRAMEWORK_TESTS=ON \
  -DMUDNN_BACKEND_ROOT="${MUDNN_BACKEND_ROOT:-}" \
  -DMUDNN_BACKEND_INCLUDE="${MUDNN_BACKEND_INCLUDE:-}" \
  -DMUDNN_BACKEND_LIB="${MUDNN_BACKEND_LIB:-}"
cmake --build "${BUILD_DIR}" \
  --target mudnn_frontend_framework_test mudnn_frontend_cpp_tests mudnn_frontend_upstream_cpp_tests \
  --parallel
