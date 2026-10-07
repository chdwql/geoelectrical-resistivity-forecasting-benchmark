from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Salesforce/moirai-2.0-R-small"
REVISION = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Moirai-2.0-R-small snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
    )
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION, "path": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
