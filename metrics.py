"""Metrics and summarization utilities for experiments."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_bias(tau_hat: float, tau_true: float) -> float:
    return float(tau_hat - tau_true)


def compute_abs_bias(tau_hat: float, tau_true: float) -> float:
    return float(abs(tau_hat - tau_true))


def compute_squared_error(tau_hat: float, tau_true: float) -> float:
    err = tau_hat - tau_true
    return float(err * err)


def ci_covered(ci_low: float, ci_high: float, tau_true: float) -> int:
    return int(ci_low <= tau_true <= ci_high)


def ci_length(ci_low: float, ci_high: float) -> float:
    return float(ci_high - ci_low)


def summarize_ate_metrics(
    df: pd.DataFrame,
    group_cols: List[str],
    method_col: str = "method",
    tau_col: str = "tau_hat",
    true_col: str = "tau_true",
) -> pd.DataFrame:
    """Aggregate bias/RMSE/CI metrics from per-trial table."""
    d = df.copy()
    d["error"] = d[tau_col] - d[true_col]
    d["abs_bias"] = np.abs(d["error"])
    d["sq_error"] = d["error"] ** 2
    d["covered"] = ((d["ci_low"] <= d[true_col]) & (d["ci_high"] >= d[true_col])).astype(int)
    d["ci_len"] = d["ci_high"] - d["ci_low"]

    agg = (
        d.groupby(group_cols + [method_col], as_index=False)
        .agg(
            bias=("error", "mean"),
            abs_bias=("abs_bias", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            coverage=("covered", "mean"),
            ci_length=("ci_len", "mean"),
            n_rep=("error", "size"),
        )
        .reset_index(drop=True)
    )
    return agg


def average_marginal_tvd(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    """Average TVD across shared one-way marginals."""
    if len(real_df) == 0 or len(syn_df) == 0:
        return np.nan

    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    if not common_cols:
        return np.nan

    tvds = []
    for col in common_cols:
        p = real_df[col].value_counts(normalize=True, dropna=False)
        q = syn_df[col].value_counts(normalize=True, dropna=False)
        idx = p.index.union(q.index)
        pv = p.reindex(idx, fill_value=0.0).to_numpy()
        qv = q.reindex(idx, fill_value=0.0).to_numpy()
        tvd = 0.5 * float(np.sum(np.abs(pv - qv)))
        tvds.append(tvd)

    return float(np.mean(tvds))


def decompose_rmse(error_values: Iterable[float]) -> dict:
    errs = np.asarray(list(error_values), dtype=float)
    if len(errs) == 0:
        return {"bias_sq": np.nan, "variance": np.nan, "rmse": np.nan}

    bias = float(np.mean(errs))
    var = float(np.var(errs, ddof=1)) if len(errs) > 1 else 0.0
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    return {"bias_sq": bias ** 2, "variance": var, "rmse": rmse}
