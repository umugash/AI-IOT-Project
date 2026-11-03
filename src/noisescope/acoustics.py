"""Small acoustic helpers (energy averaging of decibel values)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def leq(values) -> float:
    """Energy-equivalent level of a set of dB values: 10*log10(mean(10^(L/10)))."""
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(np.power(10.0, arr / 10.0))))


def leq_agg(series: pd.Series) -> float:
    """pandas-friendly aggregator for groupby(...).agg."""
    return leq(series)
