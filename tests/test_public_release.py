from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.public_release import FORBIDDEN_PREFIXES, build_archive, verify_archive


def test_public_release_has_no_forbidden_members(tmp_path: Path) -> None:
    archive = tmp_path / "public.zip"
    manifest = build_archive(archive)
    result = verify_archive(archive, run_cleanroom=False)
    assert manifest["contains_restricted_observations"] is False
    assert manifest["contains_model_weights"] is False
    assert result["hashes_verified"] == len(manifest["files"])
    assert not any(name.startswith(FORBIDDEN_PREFIXES) for name in manifest["files"])
    assert "analysis/reproduce.py" in manifest["files"]
    assert "CITATION.cff" in manifest["files"]
    assert "analysis/assemble_manuscript.py" not in manifest["files"]
    assert "docs/RESEARCH_PLAN.md" not in manifest["files"]
    assert "tests/test_manuscript_text.py" not in manifest["files"]
