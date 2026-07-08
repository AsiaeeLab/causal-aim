"""LaLonde/JOBS loader with fallback synthetic generator."""

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


_LALONDE_TREAT_COLS = ["treat", "treatment", "trt"]
_LALONDE_OUTCOME_COLS = ["re78", "outcome", "y"]


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def _coerce_lalonde_dataframe(df: pd.DataFrame) -> Optional[CausalDataset]:
    t_col = _find_col(df, _LALONDE_TREAT_COLS)
    y_col = _find_col(df, _LALONDE_OUTCOME_COLS)
    if t_col is None or y_col is None:
        return None

    T = df[t_col].to_numpy(dtype=int)
    Y = df[y_col].to_numpy(dtype=float)

    # True causal effect is not directly observed for the observational blend;
    # we use experimental difference when available, else treated-control mean.
    if "sample" in {c.lower() for c in df.columns}:
        sample_col = [c for c in df.columns if c.lower() == "sample"][0]
        exp_mask = df[sample_col].astype(str).str.contains("exp|nsw", case=False, regex=True)
        if exp_mask.any():
            tau_true = float(np.mean(Y[(T == 1) & exp_mask]) - np.mean(Y[(T == 0) & exp_mask]))
        else:
            tau_true = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))
    else:
        tau_true = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))

    drop_cols = [t_col, y_col]
    if "sample" in {c.lower() for c in df.columns}:
        drop_cols.append([c for c in df.columns if c.lower() == "sample"][0])
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    return CausalDataset(
        name="lalonde",
        X=X,
        T=T,
        Y=Y,
        tau_true=tau_true,
        metadata={
            "source": "lalonde_dataframe",
            "n": str(len(df)),
            "n_treated": str(int(np.sum(T == 1))),
            "n_control": str(int(np.sum(T == 0))),
            "tau_true_definition": "experimental_difference_in_means",
        },
    )


def _try_load_from_csv(path: Path) -> Optional[CausalDataset]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return _coerce_lalonde_dataframe(df)


def _try_load_from_causaldata() -> Optional[CausalDataset]:
    try:
        import causaldata  # type: ignore
    except Exception:
        return None

    df = None
    # Try a few common APIs.
    try:
        if hasattr(causaldata, "lalonde") and hasattr(causaldata.lalonde, "load_pandas"):
            df = causaldata.lalonde.load_pandas().data
    except Exception:
        df = None

    if df is None:
        try:
            from causaldata import datasets  # type: ignore

            if hasattr(datasets, "load_lalonde"):
                df = datasets.load_lalonde()
        except Exception:
            df = None

    if df is None:
        return None

    return _coerce_lalonde_dataframe(df)


def load_lalonde(
    seed: int = 42,
    lalonde_csv_path: Optional[str] = None,
    use_real_if_available: bool = True,
) -> CausalDataset:
    """Load LaLonde data (if available) or create a small-n stress-test surrogate."""
    if use_real_if_available:
        if lalonde_csv_path is not None:
            ds = _try_load_from_csv(Path(lalonde_csv_path))
            if ds is not None:
                ds.metadata.update({"source": "lalonde_csv", "path": str(lalonde_csv_path)})
                return ds

        ds = _try_load_from_causaldata()
        if ds is not None:
            ds.metadata.update({"source": "causaldata_package"})
            return ds

    # Fallback: 445 treated-experimental + 2490 controls stress test.
    rng = make_rng(seed, 1004)
    n = 445 + 2490
    X = make_covariates(n=n, d=12, rng=rng, continuous_fraction=0.75, prefix="x")

    dgp = generate_confounded_outcomes(
        X,
        rng=rng,
        tau_base=0.9,
        tau_scale=0.35,
        confounding_strength=1.5,
        nonlinear=True,
        binary_outcome=False,
        overlap=0.08,
        noise_scale=1.5,
    )

    return CausalDataset(
        name="lalonde",
        X=X,
        T=dgp["T"],
        Y=dgp["Y"],
        tau_true=float(dgp["tau_true"]),
        y0=dgp["y0"],
        y1=dgp["y1"],
        metadata={"source": "synthetic_fallback"},
    )
