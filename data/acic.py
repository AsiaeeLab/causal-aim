"""ACIC 2016 loader with fallback synthetic generators for selected DGPs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .common import CausalDataset, generate_confounded_outcomes, make_covariates
    from ..utils import make_rng
except ImportError:  # pragma: no cover
    from data.common import CausalDataset, generate_confounded_outcomes, make_covariates
    from utils import make_rng


_DGP_CONFIG: Dict[int, Dict[str, float]] = {
    1: {
        "confounding_strength": 0.7,
        "tau_base": 1.0,
        "tau_scale": 0.25,
        "nonlinear": 0.0,
        "overlap": 0.25,
        "noise_scale": 1.0,
    },
    7: {
        "confounding_strength": 1.2,
        "tau_base": 1.3,
        "tau_scale": 0.5,
        "nonlinear": 1.0,
        "overlap": 0.12,
        "noise_scale": 1.2,
    },
    20: {
        "confounding_strength": 1.5,
        "tau_base": 1.5,
        "tau_scale": 0.8,
        "nonlinear": 1.0,
        "overlap": 0.05,
        "noise_scale": 1.4,
    },
}


def _find_potential_outcome_columns(df: pd.DataFrame) -> Optional[tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    y0_candidates = ["y0", "y_0", "y0_mean", "mu0"]
    y1_candidates = ["y1", "y_1", "y1_mean", "mu1"]
    for y0_name in y0_candidates:
        for y1_name in y1_candidates:
            if y0_name in cols and y1_name in cols:
                return cols[y0_name], cols[y1_name]
    return None


def _try_load_simple_acic(acic_dir: Path, dgp: int) -> Optional[CausalDataset]:
    if not acic_dir.exists():
        return None

    x_candidates = [acic_dir / "x.csv", acic_dir / "X.csv"]
    X_path = next((p for p in x_candidates if p.exists()), None)
    if X_path is None:
        return None

    try:
        X = pd.read_csv(X_path)
    except Exception:
        return None

    z_path = acic_dir / f"z_dgp{dgp}.csv"
    y_path = acic_dir / f"y_dgp{dgp}.csv"
    if not z_path.exists() or not y_path.exists():
        # Try generic file names.
        z_path = acic_dir / "z.csv"
        y_path = acic_dir / "y.csv"
        if not z_path.exists() or not y_path.exists():
            return None

    try:
        T = pd.read_csv(z_path).iloc[:, 0].to_numpy(dtype=int)
        Y = pd.read_csv(y_path).iloc[:, 0].to_numpy(dtype=float)
    except Exception:
        return None

    y0 = None
    y1 = None
    tau_true = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))
    po_path = acic_dir / f"potential_outcomes_dgp{dgp}.csv"
    if po_path.exists():
        try:
            po = pd.read_csv(po_path)
            po_cols = _find_potential_outcome_columns(po)
            if po_cols is not None and len(po) == len(Y):
                y0_col, y1_col = po_cols
                y0 = po[y0_col].to_numpy(dtype=float)
                y1 = po[y1_col].to_numpy(dtype=float)
                tau_true = float(np.mean(y1 - y0))
        except Exception:
            y0 = None
            y1 = None

    metadata = {"source": "acic_simple_csv", "path": str(acic_dir), "dgp": str(dgp)}
    if y0 is not None and y1 is not None:
        metadata["potential_outcomes_path"] = str(po_path)

    return CausalDataset(
        name=f"acic_dgp{dgp}",
        X=X.select_dtypes(include=[np.number]).copy(),
        T=T,
        Y=Y,
        tau_true=tau_true,
        y0=y0,
        y1=y1,
        metadata=metadata,
    )


def load_acic(
    seed: int = 42,
    dgp: int = 7,
    n: int = 4802,
    acic_dir: Optional[str] = None,
    overlap: Optional[float] = None,
    use_real_if_available: bool = True,
) -> CausalDataset:
    """Load ACIC data if present; otherwise synthesize DGP-style surrogate."""
    if use_real_if_available and acic_dir is not None:
        ds = _try_load_simple_acic(Path(acic_dir), dgp=dgp)
        if ds is not None:
            return ds

    cfg = dict(_DGP_CONFIG.get(int(dgp), _DGP_CONFIG[7]))
    if overlap is not None:
        cfg["overlap"] = float(overlap)

    rng = make_rng(seed, 1003, int(dgp), int(1000 * cfg["overlap"]))
    X = make_covariates(n=n, d=58, rng=rng, continuous_fraction=0.65, prefix="x")

    dgp_out = generate_confounded_outcomes(
        X,
        rng=rng,
        tau_base=float(cfg["tau_base"]),
        tau_scale=float(cfg["tau_scale"]),
        confounding_strength=float(cfg["confounding_strength"]),
        nonlinear=bool(cfg["nonlinear"]),
        binary_outcome=False,
        overlap=float(cfg["overlap"]),
        noise_scale=float(cfg["noise_scale"]),
    )

    return CausalDataset(
        name=f"acic_dgp{dgp}",
        X=X,
        T=dgp_out["T"],
        Y=dgp_out["Y"],
        tau_true=float(dgp_out["tau_true"]),
        y0=dgp_out["y0"],
        y1=dgp_out["y1"],
        metadata={
            "source": "synthetic_fallback",
            "dgp": str(dgp),
            "overlap": f"{cfg['overlap']:.3f}",
        },
    )
