"""ACS semi-synthetic causal task loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .common import CausalDataset, generate_confounded_outcomes, make_covariates
    from ..utils import make_rng
except ImportError:  # pragma: no cover
    from data.common import CausalDataset, generate_confounded_outcomes, make_covariates
    from utils import make_rng


_ACS_FEATURES = [
    "AGEP", "SCHL", "MAR", "RELP", "DIS", "CIT", "MIG", "ANC", "NATIVITY", "DEAR",
    "DEYE", "DREM", "SEX", "RAC1P", "PUMA", "ST", "POBP", "HISP", "ESR", "COW",
]

_ACS_STRUCTURAL_MISSING = {"SCHL", "MIG", "DREM", "ESR", "COW"}
_STATE_PUMS_CODE = {"CA": "06"}


def _prepare_acs_frame(df: pd.DataFrame, n: int, seed: int, source: str):
    missing = [c for c in _ACS_FEATURES if c not in df.columns]
    if missing:
        return None

    out = df[_ACS_FEATURES].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in _ACS_STRUCTURAL_MISSING:
        if col in out.columns:
            out[col] = out[col].fillna(-1)
    out = out.dropna().reset_index(drop=True)
    complete_case_n = int(len(out))
    if complete_case_n == 0:
        return None

    rng = make_rng(seed, 1105, n)
    if len(out) > n:
        idx = rng.choice(len(out), size=n, replace=False)
        out = out.iloc[idx].reset_index(drop=True)
    return out, complete_case_n, source


def _try_load_acs_cache(n: int, seed: int, state: str, year: int):
    state_code = _STATE_PUMS_CODE.get(state.upper())
    if state_code is None:
        return None

    data_dir = Path(__file__).resolve().parent
    candidates = [
        data_dir / str(year) / "1-Year" / f"psam_p{state_code}.csv",
        data_dir / "raw" / "folktables" / str(year) / "1-Year" / f"psam_p{state_code}.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None

    try:
        header = pd.read_csv(csv_path, nrows=0).columns
        if any(c not in header for c in _ACS_FEATURES):
            return None
        df = pd.read_csv(csv_path, usecols=_ACS_FEATURES)
    except Exception:
        return None
    return _prepare_acs_frame(df, n=n, seed=seed, source="folktables_acs_cache")


def _standardized_dgp_covariates(X: pd.DataFrame) -> pd.DataFrame:
    arr = X.to_numpy(dtype=float)
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return pd.DataFrame((arr - mu) / sd, columns=X.columns, index=X.index)


def _try_load_acs_folktables(
    n: int,
    seed: int,
    state: str,
    year: int,
):
    try:
        from folktables import ACSDataSource  # type: ignore
    except Exception:
        return None

    try:
        data_source = ACSDataSource(
            survey_year=str(year),
            horizon="1-Year",
            survey="person",
            root_dir=str(Path(__file__).resolve().parent),
        )
        acs_data = data_source.get_data(states=[state], download=False)
    except Exception:
        return None

    return _prepare_acs_frame(acs_data, n=n, seed=seed, source="folktables_acs")


def load_acs_causal(
    seed: int = 42,
    n: int = 5000,
    state: str = "CA",
    year: int = 2018,
    use_real_if_available: bool = True,
) -> CausalDataset:
    """Load ACS covariates and generate semi-synthetic treatment/outcomes."""
    X = None
    source = "synthetic_fallback"
    complete_case_n = 0

    if use_real_if_available:
        loaded = _try_load_acs_cache(n=n, seed=seed, state=state, year=year)
        if loaded is None:
            loaded = _try_load_acs_folktables(n=n, seed=seed, state=state, year=year)
        if loaded is not None:
            X, complete_case_n, source = loaded

    if X is None:
        rng = make_rng(seed, 1005, n)
        X = make_covariates(n=n, d=20, rng=rng, continuous_fraction=0.75, prefix="x")
        complete_case_n = int(len(X))

    rng = make_rng(seed, 2005, n)
    X_dgp = _standardized_dgp_covariates(X)
    dgp = generate_confounded_outcomes(
        X_dgp,
        rng=rng,
        tau_base=0.8,
        tau_scale=0.5,
        confounding_strength=1.0,
        nonlinear=True,
        binary_outcome=False,
        overlap=0.1,
        noise_scale=1.1,
    )

    return CausalDataset(
        name=f"acs_ca_{year}",
        X=X,
        T=dgp["T"],
        Y=dgp["Y"],
        tau_true=float(dgp["tau_true"]),
        y0=dgp["y0"],
        y1=dgp["y1"],
        metadata={
            "source": source,
            "state": state,
            "year": str(year),
            "n": str(n),
            "complete_case_n": str(complete_case_n),
            "features": ",".join(_ACS_FEATURES),
        },
    )
