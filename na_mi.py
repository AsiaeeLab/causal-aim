"""Noise-aware multiple imputation for DP synthetic causal inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import norm, t as student_t

try:
    from .estimators import EstimateResult, aipw_estimate
    from .maxent import SyntheticSample
except ImportError:  # pragma: no cover
    from estimators import EstimateResult, aipw_estimate
    from maxent import SyntheticSample


@dataclass
class NAMIResult:
    tau_hat: float
    var_total: float
    ci_low: float
    ci_high: float
    within_var: float
    between_var: float
    df: float
    estimates: np.ndarray
    variances: np.ndarray
    synthetic_samples: Optional[List[SyntheticSample]] = None


def rubins_rules(
    estimates: np.ndarray,
    variances: np.ndarray,
    alpha: float = 0.05,
    complete_df: Optional[float] = None,
) -> Dict[str, float]:
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)

    m = len(estimates)
    if m == 0:
        raise ValueError("Need at least one imputation")

    q_bar = float(np.mean(estimates))
    W = float(np.mean(variances))
    B = float(np.var(estimates, ddof=1)) if m > 1 else 0.0

    T_var = float(W + (1.0 + 1.0 / m) * B)
    T_var = max(T_var, 1e-12)

    if B <= 1e-12:
        nu = 1e6
    else:
        if W <= 1e-12:
            nu = float(m - 1)
        else:
            r = ((1.0 + 1.0 / m) * B) / W
            nu_old = (m - 1.0) * (1.0 + 1.0 / r) ** 2
            nu = nu_old
            if complete_df is not None and complete_df > 2:
                nu_obs = ((complete_df + 1.0) / (complete_df + 3.0)) * complete_df * (1.0 - W / T_var)
                if nu_obs > 0:
                    nu = 1.0 / (1.0 / nu_old + 1.0 / nu_obs)

    se = float(np.sqrt(T_var))
    if np.isfinite(nu) and nu < 1e5:
        crit = float(student_t.ppf(1.0 - alpha / 2.0, df=max(nu, 1.0)))
    else:
        crit = float(norm.ppf(1.0 - alpha / 2.0))

    return {
        "tau_hat": q_bar,
        "W": W,
        "B": B,
        "var_total": T_var,
        "df": float(nu),
        "ci_low": float(q_bar - crit * se),
        "ci_high": float(q_bar + crit * se),
    }


def noise_aware_multiple_imputation(
    measured_moments: np.ndarray,
    noise_covariance: np.ndarray,
    make_synthetic_fn: Callable[[np.ndarray, np.random.Generator], SyntheticSample],
    M: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    keep_samples: bool = False,
    moment_projection_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    estimation_subsample_n: int = 20000,
) -> NAMIResult:
    """
    NA+MI: sample latent moments around DP measurements, synthesize M datasets,
    then combine ATE estimates via Rubin's rules.
    """
    if M <= 0:
        raise ValueError("M must be positive")

    measured_moments = np.asarray(measured_moments, dtype=float)
    cov = np.asarray(noise_covariance, dtype=float)

    if cov.ndim == 2:
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    elif cov.ndim == 1:
        std = np.sqrt(np.clip(cov, 0.0, None))
    else:
        raise ValueError("noise_covariance must be 1D or 2D")

    estimates = np.zeros(M, dtype=float)
    variances = np.zeros(M, dtype=float)
    samples: List[SyntheticSample] = []

    for m in range(M):
        moment_draw = measured_moments + rng.normal(0.0, std)
        if moment_projection_fn is not None:
            moment_draw = moment_projection_fn(moment_draw)

        syn = make_synthetic_fn(moment_draw, rng)

        n_syn = len(syn.Y)
        if n_syn > estimation_subsample_n:
            idx = rng.choice(n_syn, size=estimation_subsample_n, replace=False)
            est: EstimateResult = aipw_estimate(
                X=syn.phi[idx],
                T=syn.T[idx],
                Y=syn.Y[idx],
                alpha=alpha,
            )
        else:
            est = aipw_estimate(X=syn.phi, T=syn.T, Y=syn.Y, alpha=alpha)
        estimates[m] = est.tau_hat
        variances[m] = est.var_hat

        if keep_samples:
            samples.append(syn)

    pooled = rubins_rules(estimates=estimates, variances=variances, alpha=alpha)

    return NAMIResult(
        tau_hat=float(pooled["tau_hat"]),
        var_total=float(pooled["var_total"]),
        ci_low=float(pooled["ci_low"]),
        ci_high=float(pooled["ci_high"]),
        within_var=float(pooled["W"]),
        between_var=float(pooled["B"]),
        df=float(pooled["df"]),
        estimates=estimates,
        variances=variances,
        synthetic_samples=samples if keep_samples else None,
    )
