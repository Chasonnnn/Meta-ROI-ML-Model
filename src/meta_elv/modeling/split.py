from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_idx: pd.Index
    calib_idx: pd.Index
    test_idx: pd.Index
    # For reporting/repro
    train_end_time: str
    calib_end_time: str


def time_split(
    df: pd.DataFrame,
    time_col: str,
    train_frac: float,
    calib_frac: float,
    test_frac: float,
) -> TimeSplit:
    if abs((train_frac + calib_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/calib/test fractions must sum to 1.0")
    if len(df) < 10:
        raise ValueError("Not enough rows to split (need >= 10).")

    d = df.sort_values(time_col).reset_index()
    n = len(d)
    n_train = max(1, int(n * train_frac))
    n_calib = max(1, int(n * calib_frac))
    if n_train + n_calib >= n:
        n_train = max(1, n - 2)
        n_calib = 1
    n_test = n - (n_train + n_calib)
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train = d.iloc[:n_train]
    calib = d.iloc[n_train : n_train + n_calib]
    test = d.iloc[n_train + n_calib :]

    train_end_time = str(train[time_col].max())
    calib_end_time = str(calib[time_col].max())

    return TimeSplit(
        train_idx=pd.Index(train["index"]),
        calib_idx=pd.Index(calib["index"]),
        test_idx=pd.Index(test["index"]),
        train_end_time=train_end_time,
        calib_end_time=calib_end_time,
    )

