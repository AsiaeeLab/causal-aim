"""Twins loader with fallback synthetic generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .common import CausalDataset, generate_confounded_outcomes, make_covariates
    from ..utils import make_rng
except ImportError:  # pragma: no cover
    from data.common import CausalDataset, generate_confounded_outcomes, make_covariates
    from utils import make_rng


def _try_load_twins_csv(path: Path) -> Optional[CausalDataset]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Flexible parsing for common twins-style columns.
    cols = {c.lower(): c for c in df.columns}
    t_col = cols.get("t") or cols.get("treatment")
    y_col = cols.get("y") or cols.get("outcome")
    y0_col = cols.get("y0") or cols.get("mort_0")
    y1_col = cols.get("y1") or cols.get("mort_1")

    if t_col is None:
        return None

    if y_col is None and y0_col is not None and y1_col is not None:
        t = df[t_col].to_numpy(dtype=int)
        y0 = df[y0_col].to_numpy(dtype=float)
        y1 = df[y1_col].to_numpy(dtype=float)
        y = y1 * t + y0 * (1 - t)
    elif y_col is not None:
        t = df[t_col].to_numpy(dtype=int)
        y = df[y_col].to_numpy(dtype=float)
        y0 = df[y0_col].to_numpy(dtype=float) if y0_col else None
        y1 = df[y1_col].to_numpy(dtype=float) if y1_col else None
    else:
        return None

    drop_cols = [c for c in [t_col, y_col, y0_col, y1_col] if c]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    tau_true = float(np.mean(y1 - y0)) if y0 is not None and y1 is not None else float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
    return CausalDataset(
        name="twins",
        X=X,
        T=t,
        Y=y,
        tau_true=tau_true,
        y0=y0,
        y1=y1,
        metadata={"source": "twins_csv", "path": str(path)},
    )


def load_twins(
    seed: int = 42,
    twins_csv_path: Optional[str] = None,
    use_real_if_available: bool = True,
) -> CausalDataset:
    """Load twins benchmark or synthesize a binary-outcome surrogate."""
    if use_real_if_available and twins_csv_path is not None:
        ds = _try_load_twins_csv(Path(twins_csv_path))
        if ds is not None:
            return ds

    rng = make_rng(seed, 1002)
    X = make_covariates(n=11400, d=30, rng=rng, continuous_fraction=0.7, prefix="x")
    dgp = generate_confounded_outcomes(
        X,
        rng=rng,
        tau_base=-0.08,
        tau_scale=0.06,
        confounding_strength=0.9,
        nonlinear=True,
        binary_outcome=True,
        overlap=0.18,
    )

    return CausalDataset(
        name="twins",
        X=X,
        T=dgp["T"],
        Y=dgp["Y"],
        tau_true=float(dgp["tau_true"]),
        y0=dgp["y0"],
        y1=dgp["y1"],
        metadata={"source": "synthetic_fallback"},
    )
