from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Chronos-2 snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
    )
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION, "path": path}, indent=2))


if __name__ == "__main__":
    main()
