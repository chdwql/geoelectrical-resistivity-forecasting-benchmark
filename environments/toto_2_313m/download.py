from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Datadog/Toto-2.0-313m"
REVISION = "a7bab288f5e95f8606f8306f86659357e1c001ef"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Toto-2.0-313m snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=destination)
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION, "path": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
