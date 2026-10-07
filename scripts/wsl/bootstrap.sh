#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "This project must be bootstrapped inside WSL2." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/geo-bench"
export UV_PROJECT_ENVIRONMENT="${cache_root}/venvs/core"

cd "${project_root}"
uv sync --frozen --dev

if [[ "${1:-}" == "--ingestion" ]]; then
  ingestion_environment="${cache_root}/venvs/ingestion"
  uv venv --python 3.12 --clear "${ingestion_environment}"
  uv pip install --python "${ingestion_environment}/bin/python" --editable "${project_root}"
  uv pip install --python "${ingestion_environment}/bin/python" addereq
  "${ingestion_environment}/bin/python" -c "import addereq.fetching, oracledb"
  echo "Published addereq package installed. Linux Oracle Instant Client may still be required."
fi

uv run --frozen geo-bench doctor
