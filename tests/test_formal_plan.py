from __future__ import annotations

from geoelectrical_resistivity_forecasting_benchmark.formal_plan import model_supports


def test_model_supports_tracks_and_horizon_constraints() -> None:
    model = {
        "tracks": ["U", "M", "X"],
        "track_constraints": {"U": [1, 90], "M": [1], "X": [1]},
    }
    assert model_supports(model, "U", 90)
    assert not model_supports(model, "M", 90)
    assert not model_supports(model, "O", 1)
