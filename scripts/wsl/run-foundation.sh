#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this command inside Ubuntu WSL2." >&2
  exit 2
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: run-foundation.sh <model-id> <request-directory> <output-directory>" >&2
  exit 2
fi

model_id="$1"
request_directory="$2"
output_directory="$3"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "${model_id}" in
  timesfm_3)
    environment_name="timesfm_3"
    runner="environments/timesfm_3/run_requests.py"
    checkpoint="weights/timesfm-3.0-pytorch"
    ;;
  chronos_2)
    environment_name="chronos_2"
    runner="environments/chronos_2/run_requests.py"
    checkpoint="weights/chronos-2"
    ;;
  moirai_2_small)
    environment_name="moirai_2_small"
    runner="environments/moirai_2_small/run_requests.py"
    checkpoint="weights/moirai-2.0-R-small"
    ;;
  toto_2_313m)
    environment_name="toto_2_313m"
    runner="environments/toto_2_313m/run_requests.py"
    checkpoint="weights/toto-2.0-313m"
    ;;
  *)
    echo "Unsupported foundation model: ${model_id}" >&2
    exit 2
    ;;
esac

cd "${project_root}"
split="$(
  "${project_root}/scripts/wsl/python.sh" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["identity"]["split"])' \
    "${request_directory}/manifest.json"
)"
"${project_root}/scripts/wsl/run.sh" verify-freeze --split "${split}"
exec "${project_root}/scripts/wsl/model-env.sh" "${environment_name}" \
  python "${runner}" \
  --requests "${request_directory}" \
  --checkpoint "${checkpoint}" \
  --output "${output_directory}"
