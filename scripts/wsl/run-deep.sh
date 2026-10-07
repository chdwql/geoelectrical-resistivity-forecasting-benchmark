#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this command inside Ubuntu WSL2." >&2
  exit 2
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: run-deep.sh <model-id> <request-directory> <output-directory>" >&2
  exit 2
fi

model_id="$1"
request_directory="$2"
output_directory="$3"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

mapfile -t settings < <(
  "${project_root}/scripts/wsl/python.sh" -c '
import json, sys, yaml
model_id, manifest_path, config_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
identity = manifest["identity"]
if identity["model_id"] != model_id:
    raise SystemExit("Request model does not match requested deep model")
config = yaml.safe_load(open(config_path, encoding="utf-8"))[model_id]
print(identity["split"])
print(config["selected"]["epochs"])
print(config["selected"]["learning_rate"])
print(config["training_batch_size"])
' "${model_id}" "${request_directory}/manifest.json" "configs/target_domain_models.yaml"
)

split="${settings[0]}"
epochs="${settings[1]}"
learning_rate="${settings[2]}"
batch_size="${settings[3]}"
"${project_root}/scripts/wsl/run.sh" verify-freeze --split "${split}"
exec "${project_root}/scripts/wsl/model-env.sh" target_deep \
  python environments/target_deep/run_requests.py \
  --requests "${request_directory}" \
  --output "${output_directory}" \
  --epochs "${epochs}" \
  --learning-rate "${learning_rate}" \
  --batch-size "${batch_size}"
