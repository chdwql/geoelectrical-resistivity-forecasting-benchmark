#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this command inside Ubuntu WSL2." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/geo-bench"
export UV_PROJECT_ENVIRONMENT="${cache_root}/venvs/core"
cd "${project_root}"

if [[ "${1:-}" == "test" ]]; then
  shift
  exec uv run --frozen pytest "$@"
fi

exec uv run --frozen geo-bench "$@"
