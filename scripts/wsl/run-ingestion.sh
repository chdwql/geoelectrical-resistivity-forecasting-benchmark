#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run ingestion inside Ubuntu WSL2." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/geo-bench"
ingestion_environment="${cache_root}/venvs/ingestion"
python_executable="${ingestion_environment}/bin/python"
cli_executable="${ingestion_environment}/bin/geo-bench"

if [[ ! -x "${python_executable}" ]]; then
  echo "Ingestion environment is missing; run ./scripts/wsl/bootstrap.sh --ingestion" >&2
  exit 3
fi

cd "${project_root}"
exec "${cli_executable}" data "$@"
