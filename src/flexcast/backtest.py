"""Rolling-origin evaluation: refit every `refit_every` days on all history before the block."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import MapRidge


def rolling(make_model, X: pd.DataFrame, Y: pd.DataFrame, start: str, end: str, refit_every: int = 7):
    """Forecast every day in [start, end]. For a block starting on day D0 the model sees
    rows with day <= D0 - 1 only (the D0 - 1 prices are known at D0's gate closure).
    Feature maps of MapRidge models are fitted once on data before `start` and frozen."""
    days = X.loc[start:end].index
    model = make_model()
    if isinstance(model, MapRidge):
        model.featmap.fit(X.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)])
    preds = []
    for b0 in range(0, len(days), refit_every):
        block = days[b0:b0 + refit_every]
        hist = X.index < block[0]
        model.fit(X[hist], Y[hist])
        preds.append(model.predict(X.loc[block]))
    return pd.DataFrame(np.vstack(preds), index=days, columns=Y.columns)
