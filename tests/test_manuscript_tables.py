from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.manuscript_tables import read_task_coverage, write_table
import analysis.manuscript_tables as tables


def test_unsupported_status_survives_export(tmp_path: Path) -> None:
    source = tmp_path / "coverage.csv"
    source.write_text("status,success_count,error_count\nN/A,,\nplanned,12,0\n", encoding="utf-8")
    frame = read_task_coverage(source)
    assert frame.loc[0, "status"] == "N/A"
    assert pd.isna(frame.loc[0, "success_count"])
    assert frame.loc[1, "success_count"] == 12
    paths = write_table(frame, "export", tmp_path, write_markdown=False)
    restored = pd.read_csv(paths[0], keep_default_na=False)
    assert restored.status.tolist() == ["N/A", "planned"]


def test_public_station_scores_preserve_float_roundtrip(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "remaining"
    folder.mkdir()
    content = "model_id,split,horizon,station,mase\narima,test_a,1,11003,0.9768772827477317\n"
    (folder / "station_all_horizons.csv").write_text(content, encoding="utf-8")
    monkeypatch.setattr(tables, "G6", tmp_path)
    output = write_table(tables.per_station_all_horizons(), "scores", tmp_path, write_markdown=False)[0]
    assert output.read_text(encoding="utf-8") == content


