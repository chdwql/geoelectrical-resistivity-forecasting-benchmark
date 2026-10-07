#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this command inside Ubuntu WSL2." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/geo-bench"
python_executable="${cache_root}/venvs/core/bin/python"
if [[ ! -x "${python_executable}" ]]; then
  echo "Core environment is missing; run ./scripts/wsl/bootstrap.sh" >&2
  exit 3
fi
cd "${project_root}"
exec "${python_executable}" "$@"
