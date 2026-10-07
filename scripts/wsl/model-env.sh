#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this command inside Ubuntu WSL2." >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  echo "Usage: model-env.sh <environment-name> <command> [args...]" >&2
  exit 2
fi

environment_name="$1"
shift
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_root="${project_root}/environments/${environment_name}"
if [[ ! -f "${environment_root}/pyproject.toml" ]]; then
  echo "Unknown model environment: ${environment_name}" >&2
  exit 2
fi
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/geo-bench"
export UV_PROJECT_ENVIRONMENT="${cache_root}/venvs/model-${environment_name}"
cd "${project_root}"
exec uv run --project "${environment_root}" --frozen "$@"
