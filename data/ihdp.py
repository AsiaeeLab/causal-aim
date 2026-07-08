"""IHDP loader with fallback synthetic generator."""

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


def _extract_rep(arr: np.ndarray, rep: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] > arr.shape[1]:
            idx = rep % arr.shape[1]
            return arr[:, idx]
        idx = rep % arr.shape[0]
        return arr[idx, :]
    if arr.ndim == 3:
        idx = rep % arr.shape[2]
        return arr[:, :, idx]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _try_load_real_ihdp(path: Path, replication: int) -> Optional[CausalDataset]:
    if not path.exists():
        return None

    try:
        npz = np.load(path)
    except Exception:
        return None

    required = {"x", "t", "yf"}
    if not required.issubset(set(npz.files)):
        return None

    X_arr = _extract_rep(npz["x"], replication)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n, d = X_arr.shape

    t = _extract_rep(npz["t"], replication).astype(int).reshape(-1)
    yf = _extract_rep(npz["yf"], replication).astype(float).reshape(-1)

    y0 = None
    y1 = None
    tau_true = np.nan

    if "mu0" in npz.files and "mu1" in npz.files:
        mu0 = _extract_rep(npz["mu0"], replication).astype(float).reshape(-1)
        mu1 = _extract_rep(npz["mu1"], replication).astype(float).reshape(-1)
        tau_true = float(np.mean(mu1 - mu0))
        y0, y1 = mu0, mu1
    elif "ycf" in npz.files:
        ycf = _extract_rep(npz["ycf"], replication).astype(float).reshape(-1)
        y0 = np.where(t == 0, yf, ycf)
        y1 = np.where(t == 1, yf, ycf)
        tau_true = float(np.mean(y1 - y0))

    X = pd.DataFrame(X_arr, columns=[f"x{j+1}" for j in range(d)])
    return CausalDataset(
        name="ihdp",
        X=X,
        T=t,
        Y=yf,
        tau_true=float(tau_true if np.isfinite(tau_true) else np.mean(yf[t == 1]) - np.mean(yf[t == 0])),
        y0=y0,
        y1=y1,
        metadata={"source": "ihdp_npz", "path": str(path), "replication": str(replication)},
    )


def load_ihdp(
    seed: int = 42,
    replication: int = 0,
    ihdp_npz_path: Optional[str] = None,
    use_real_if_available: bool = True,
) -> CausalDataset:
    """Load IHDP or synthesize a close surrogate if unavailable."""
    if use_real_if_available and ihdp_npz_path is not None:
        ds = _try_load_real_ihdp(Path(ihdp_npz_path), replication)
        if ds is not None:
            return ds

    rng = make_rng(seed, 1001, replication)
    X = make_covariates(n=747, d=25, rng=rng, continuous_fraction=0.8, prefix="x")

    dgp = generate_confounded_outcomes(
        X,
        rng=rng,
        tau_base=4.0,
        tau_scale=0.6,
        confounding_strength=1.1,
        nonlinear=True,
        binary_outcome=False,
        overlap=0.12,
        noise_scale=1.0,
    )

    return CausalDataset(
        name="ihdp",
        X=X,
        T=dgp["T"],
        Y=dgp["Y"],
        tau_true=float(dgp["tau_true"]),
        y0=dgp["y0"],
        y1=dgp["y1"],
        metadata={"source": "synthetic_fallback", "replication": str(replication)},
    )
