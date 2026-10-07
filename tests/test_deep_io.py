from __future__ import annotations

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.deep_io import _causal_mean_fill


def test_causal_mean_fill_never_uses_later_values() -> None:
    values = pd.Series([np.nan, 2.0, np.nan, 8.0, np.nan])
    filled, observed = _causal_mean_fill(values)
    assert filled.tolist() == [0.0, 2.0, 2.0, 8.0, 5.0]
    assert observed.tolist() == [False, True, False, True, False]

    changed = values.copy()
    changed.iloc[3] = 800.0
    changed_filled, _ = _causal_mean_fill(changed)
    assert changed_filled[:3].tolist() == filled[:3].tolist()
