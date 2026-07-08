"""Baseline and proposed methods for causal DP synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .causal_aim import select_causal_aim_groups
    from .estimators import aipw_estimate, dp_output_perturbed_dr
    from .maxent import MaxEntModel, SyntheticSample, calibrate_maxent_from_moments
    from .mechanisms import GaussianMeasurement, measure_gaussian_vector
    from .metrics import average_marginal_tvd
    from .na_mi import noise_aware_multiple_imputation
    from .utils import FeatureGroup, project_simplex_with_sum
    from .workloads import (
        compute_causal_moments,
        compute_generic_feature_marginals,
        causal_moment_sensitivity,
        one_hot_phi_l2_bound,
        project_causal_moment_vector,
        compute_treatment_rate,
        measured_feature_mask,
    )
except ImportError:  # pragma: no cover
    from causal_aim import select_causal_aim_groups
    from estimators import aipw_estimate, dp_output_perturbed_dr
    from maxent import MaxEntModel, SyntheticSample, calibrate_maxent_from_moments
    from mechanisms import GaussianMeasurement, measure_gaussian_vector
    from metrics import average_marginal_tvd
    from na_mi import noise_aware_multiple_imputation
    from utils import FeatureGroup, project_simplex_with_sum
    from workloads import (
        compute_causal_moments,
        compute_generic_feature_marginals,
        causal_moment_sensitivity,
        one_hot_phi_l2_bound,
        project_causal_moment_vector,
        compute_treatment_rate,
        measured_feature_mask,
    )


@dataclass
class MethodResult:
    method: str
    tau_hat: float
    var_hat: float
    ci_low: float
    ci_high: float
    tvd: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def _aipw_with_subsample(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    max_n: int = 20000,
):
    n = len(Y)
    if n <= max_n:
        return aipw_estimate(X=X, T=T, Y=Y, alpha=alpha)
    idx = rng.choice(n, size=max_n, replace=False)
    return aipw_estimate(X=X[idx], T=T[idx], Y=Y[idx], alpha=alpha)


def _sample_discrete_product(
    groups: Sequence[FeatureGroup],
    n: int,
    p_treat: float,
    probs_marginal: Dict[str, np.ndarray],
    probs_cond_t0: Optional[Dict[str, np.ndarray]],
    probs_cond_t1: Optional[Dict[str, np.ndarray]],
    y_model: str,
    y_params: Dict[str, np.ndarray | float],
    y_noise: float,
    y_clip: float,
    rng: np.random.Generator,
) -> SyntheticSample:
    p = groups[-1].stop if groups else 0
    phi = np.zeros((n, p), dtype=float)
    T = rng.binomial(1, np.clip(p_treat, 1e-3, 1 - 1e-3), size=n).astype(int)

    X_cols: Dict[str, np.ndarray] = {}

    # Intercept.
    if groups and groups[0].name == "intercept":
        phi[:, groups[0].start : groups[0].stop] = 1.0

    for g in groups:
        if g.name == "intercept":
            continue

        idx = np.arange(len(g.levels))
        chosen = np.empty(n, dtype=int)

        if probs_cond_t0 is None or probs_cond_t1 is None:
            probs = probs_marginal[g.name]
            chosen = rng.choice(idx, size=n, p=probs)
        else:
            mask0 = T == 0
            mask1 = ~mask0
            if mask0.any():
                chosen[mask0] = rng.choice(idx, size=int(np.sum(mask0)), p=probs_cond_t0[g.name])
            if mask1.any():
                chosen[mask1] = rng.choice(idx, size=int(np.sum(mask1)), p=probs_cond_t1[g.name])

        for j in range(len(g.levels)):
            phi[:, g.start + j] = (chosen == j).astype(float)
        X_cols[g.name] = np.asarray(g.levels, dtype=int)[chosen]

    # Outcome model.
    if y_model == "t_only":
        mu0 = float(y_params.get("mu0", 0.0))
        mu1 = float(y_params.get("mu1", 0.0))
        mu = np.where(T == 1, mu1, mu0).astype(float)
    elif y_model == "t_plus_selected":
        mu0 = float(y_params.get("mu0", 0.0))
        mu1 = float(y_params.get("mu1", 0.0))
        mu = np.where(T == 1, mu1, mu0).astype(float)
        eff0 = y_params.get("eff0", {})
        eff1 = y_params.get("eff1", {})
        if isinstance(eff0, dict) and isinstance(eff1, dict):
            denom = max(len(eff0), 1)
            for g in groups:
                if g.name in eff0 and g.name in eff1:
                    vals = X_cols[g.name]
                    lvl_to_idx = {lvl: i for i, lvl in enumerate(g.levels)}
                    idx_vals = np.array([lvl_to_idx[int(v)] for v in vals], dtype=int)
                    adj0 = np.asarray(eff0[g.name], dtype=float)[idx_vals]
                    adj1 = np.asarray(eff1[g.name], dtype=float)[idx_vals]
                    mu += np.where(T == 1, adj1, adj0) / denom
    else:
        raise ValueError(f"Unknown y_model: {y_model}")

    Y = mu + rng.normal(0.0, max(y_noise, 0.1), size=n)
    Y = np.clip(Y, -y_clip, y_clip)

    return SyntheticSample(
        X_discrete=pd.DataFrame(X_cols),
        phi=phi,
        T=T,
        Y=Y,
    )


def run_non_private_dr(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
) -> MethodResult:
    est = _aipw_with_subsample(X=phi, T=T, Y=Y, alpha=alpha, rng=np.random.default_rng(0))
    return MethodResult(
        method="non_private_dr",
        tau_hat=est.tau_hat,
        var_hat=est.var_hat,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
    )


def run_mst_naive(
    phi: np.ndarray,
    groups: Sequence[FeatureGroup],
    X_discrete_real: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> MethodResult:
    n = len(Y)
    # DP-measure generic marginals + (T, TY) scalars (all linear moments).
    # This is a simple "MST-style" product baseline, not the official Private-PGM implementation.
    p_t = float(np.mean(T.astype(float)))
    q0_y = float(np.mean((1.0 - T.astype(float)) * Y.astype(float)))
    q1_y = float(np.mean(T.astype(float) * Y.astype(float)))
    marginals = compute_generic_feature_marginals(phi=phi, groups=groups)

    # Vector sensitivity: each record contributes one-hot X levels (sqrt(#covariates))
    # plus (T, (1-T)Y, TY) (bounded by sqrt(1+y_clip^2)).
    sens = float(2.0 * np.sqrt(float(len(groups)) + float(y_clip) ** 2) / max(n, 1))

    scalars = [p_t, q0_y, q1_y]
    keys = ["p_t", "q0_y", "q1_y"]
    for name, probs in marginals.items():
        for j, v in enumerate(probs):
            keys.append(f"{name}_{j}")
            scalars.append(float(v))

    q = np.asarray(scalars, dtype=float)
    measured = measure_gaussian_vector(q, epsilon=epsilon, delta=delta, sensitivity=sens, rng=rng)
    noisy = measured.noisy
    idx = 0

    p_t_noisy = float(np.clip(noisy[idx], 1e-3, 1 - 1e-3))
    idx += 1
    q0_y_noisy = float(np.clip(noisy[idx], -y_clip, y_clip))
    idx += 1
    q1_y_noisy = float(np.clip(noisy[idx], -y_clip, y_clip))
    idx += 1

    p0_noisy = float(np.clip(1.0 - p_t_noisy, 1e-3, 1.0))
    mu0_noisy = float(np.clip(q0_y_noisy / p0_noisy, -y_clip, y_clip))
    mu1_noisy = float(np.clip(q1_y_noisy / p_t_noisy, -y_clip, y_clip))

    noisy_marginals: Dict[str, np.ndarray] = {}
    for name, probs in marginals.items():
        k = len(probs)
        p = noisy[idx : idx + k]
        p = np.maximum(p, 1e-6)
        p = p / np.sum(p)
        noisy_marginals[name] = p
        idx += k

    syn = _sample_discrete_product(
        groups=groups,
        n=n_syn,
        p_treat=p_t_noisy,
        probs_marginal=noisy_marginals,
        probs_cond_t0=None,
        probs_cond_t1=None,
        y_model="t_only",
        y_params={"mu0": mu0_noisy, "mu1": mu1_noisy},
        y_noise=float(np.std(Y)),
        y_clip=y_clip,
        rng=rng,
    )

    est = _aipw_with_subsample(X=syn.phi, T=syn.T, Y=syn.Y, alpha=alpha, rng=rng)
    tvd = average_marginal_tvd(X_discrete_real, syn.X_discrete)

    return MethodResult(
        method="mst_naive",
        tau_hat=est.tau_hat,
        var_hat=est.var_hat,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        tvd=tvd,
    )


def run_aim_naive(
    phi: np.ndarray,
    groups: Sequence[FeatureGroup],
    X_discrete_real: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
    k_generic: int = 5,
) -> MethodResult:
    n = len(Y)
    p = phi.shape[1]
    candidates = [g for g in groups if g.name != "intercept"]
    k_eff = min(int(k_generic), len(candidates))

    # Budget split: one DP measurement for all X-marginals, then k_eff DP measurements
    # for (T,Y)-conditional moments on the selected groups.
    n_parts = 1 + max(k_eff, 1)
    eps_part = float(epsilon) / float(n_parts)
    delta_part = float(delta) / float(n_parts)

    # (1) DP measurement of X-marginals.
    marginals_true = compute_generic_feature_marginals(phi=phi, groups=groups)
    q = []
    group_order = []
    for g in candidates:
        probs = marginals_true[g.name]
        group_order.append((g.name, len(probs)))
        q.extend([float(x) for x in probs])

    # L2 sensitivity for concatenated one-hot marginals across all covariate groups.
    # Each row contributes one active level per group.
    marg_sens = float(2.0 * np.sqrt(max(len(candidates), 1)) / max(n, 1))
    m_marg = measure_gaussian_vector(
        query=np.asarray(q, dtype=float),
        epsilon=max(eps_part, 1e-8),
        delta=min(max(delta_part, 1e-12), 0.999999),
        sensitivity=marg_sens,
        rng=rng,
    )

    marginals: Dict[str, np.ndarray] = {}
    entropies: Dict[str, float] = {}
    idx = 0
    for name, k in group_order:
        v = np.asarray(m_marg.noisy[idx : idx + k], dtype=float)
        v = np.maximum(v, 1e-8)
        v = v / float(np.sum(v))
        marginals[name] = v
        entropies[name] = -float(np.sum(v * np.log(v)))
        idx += k

    selected = [name for name, _ in sorted(entropies.items(), key=lambda x: -x[1])[:k_eff]]

    # (2) DP measurements for selected groups: causal-moment subvector for the group.
    true_moments = compute_causal_moments(phi=phi, T=T, Y=Y)
    group_map = {g.name: g for g in groups}

    def _group_moment_indices(g: FeatureGroup) -> np.ndarray:
        idx2 = np.arange(g.start, g.stop)
        return np.concatenate([idx2, idx2 + p, idx2 + 2 * p, idx2 + 3 * p], axis=0)

    moment_sens = causal_moment_sensitivity(n=n, y_clip=y_clip, phi_bound=1.0)
    measured_groups: Dict[str, GaussianMeasurement] = {}
    for name in selected:
        g = group_map[name]
        idx_full = _group_moment_indices(g)
        measured_groups[name] = measure_gaussian_vector(
            query=true_moments[idx_full],
            epsilon=max(eps_part, 1e-8),
            delta=min(max(delta_part, 1e-12), 0.999999),
            sensitivity=moment_sens,
            rng=rng,
        )

    if selected:
        g0 = group_map[selected[0]]
        m0 = measured_groups[selected[0]].noisy
        k0 = g0.stop - g0.start
        q0c = m0[0:k0]
        q1c = m0[k0 : 2 * k0]
        q0y = m0[2 * k0 : 3 * k0]
        q1y = m0[3 * k0 : 4 * k0]
        p0 = float(np.clip(np.sum(q0c), 1e-3, 1.0))
        p1 = float(np.clip(np.sum(q1c), 1e-3, 1.0))
        s = p0 + p1
        p0 /= s
        p1 /= s
        p_t_noisy = float(np.clip(p1, 1e-3, 1 - 1e-3))
        mu0_noisy = float(np.clip(float(np.sum(q0y)) / max(p0, 1e-6), -y_clip, y_clip))
        mu1_noisy = float(np.clip(float(np.sum(q1y)) / max(p1, 1e-6), -y_clip, y_clip))
    else:
        p_t_noisy = 0.5
        mu0_noisy = 0.0
        mu1_noisy = 0.0

    # Build treatment-conditional probs only for selected groups; fallback to DP marginals otherwise.
    cond0: Dict[str, np.ndarray] = {}
    cond1: Dict[str, np.ndarray] = {}
    eff0: Dict[str, np.ndarray] = {}
    eff1: Dict[str, np.ndarray] = {}

    for g in candidates:
        if g.name not in selected:
            cond0[g.name] = marginals[g.name]
            cond1[g.name] = marginals[g.name]
            continue

        mg = measured_groups[g.name].noisy
        k = g.stop - g.start
        c0 = np.maximum(mg[0:k], 1e-8)
        c1 = np.maximum(mg[k : 2 * k], 1e-8)
        cond0[g.name] = c0 / float(np.sum(c0))
        cond1[g.name] = c1 / float(np.sum(c1))

        y0 = mg[2 * k : 3 * k]
        y1 = mg[3 * k : 4 * k]
        mean0_lvl = np.clip(y0 / np.maximum(c0, 1e-6), -y_clip, y_clip)
        mean1_lvl = np.clip(y1 / np.maximum(c1, 1e-6), -y_clip, y_clip)
        eff0[g.name] = 0.3 * (mean0_lvl - mu0_noisy)
        eff1[g.name] = 0.3 * (mean1_lvl - mu1_noisy)

    syn = _sample_discrete_product(
        groups=groups,
        n=n_syn,
        p_treat=p_t_noisy,
        probs_marginal=marginals,
        probs_cond_t0=cond0,
        probs_cond_t1=cond1,
        y_model="t_plus_selected",
        y_params={"mu0": mu0_noisy, "mu1": mu1_noisy, "eff0": eff0, "eff1": eff1},
        y_noise=float(np.std(Y)),
        y_clip=y_clip,
        rng=rng,
    )

    est = _aipw_with_subsample(X=syn.phi, T=syn.T, Y=syn.Y, alpha=alpha, rng=rng)
    tvd = average_marginal_tvd(X_discrete_real, syn.X_discrete)

    return MethodResult(
        method="aim_naive",
        tau_hat=est.tau_hat,
        var_hat=est.var_hat,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        tvd=tvd,
        metadata={"selected_groups": ",".join(selected)},
    )


def _fit_causal_model_from_noisy_moments(
    noisy_moments: np.ndarray,
    groups: Sequence[FeatureGroup],
    feature_names: Sequence[str],
    y_clip: float,
    base_moments: Optional[np.ndarray] = None,
    measured_feature_mask_arr: Optional[np.ndarray] = None,
    measured_moment_mask_arr: Optional[np.ndarray] = None,
    ridge_alpha: float = 0.0,
    y_noise_scale: float = 1.0,
    refinement_steps: int = 6,
    refinement_lr: float = 0.6,
) -> Tuple[MaxEntModel, Dict[str, float]]:
    model, info = calibrate_maxent_from_moments(
        noisy_moments=noisy_moments,
        groups=groups,
        feature_names=feature_names,
        y_clip=y_clip,
        base_moments=base_moments,
        measured_feature_mask=measured_feature_mask_arr,
        measured_moment_mask=measured_moment_mask_arr,
        y_noise_scale=y_noise_scale,
        ridge_alpha=ridge_alpha,
        verify_after_fit=True,
        verify_threshold=0.1,
        refinement_steps=int(refinement_steps),
        refinement_lr=float(refinement_lr),
        verify_use_sampling=False,
        return_info=True,
    )
    return model, info


def _snr_moment_mask(measured_moments: np.ndarray, noise_covariance: np.ndarray, threshold: float) -> np.ndarray:
    threshold = float(threshold)
    if threshold <= 0:
        return np.ones_like(measured_moments, dtype=bool)

    cov = np.asarray(noise_covariance, dtype=float)
    if cov.ndim == 2:
        var = np.diag(cov)
    else:
        var = cov
    std = np.sqrt(np.maximum(var, 0.0))
    denom = np.where(std > 0.0, std, np.inf)
    snr = np.abs(np.asarray(measured_moments, dtype=float)) / denom
    mask = snr >= threshold

    # Always keep intercept moments.
    p = int(len(mask) // 4)
    if 4 * p == len(mask) and p > 0:
        mask[0] = True
        mask[p] = True
        mask[2 * p] = True
        mask[3 * p] = True
    return mask


def run_causal_workload_naive(
    phi: np.ndarray,
    feature_names: Sequence[str],
    groups: Sequence[FeatureGroup],
    X_discrete_real: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
    calibration_snr_threshold: Optional[float] = None,
    calibration_ridge_alpha: float = 0.0,
    calibration_refinement_steps: Optional[int] = None,
) -> MethodResult:
    n = len(Y)
    true_moments = compute_causal_moments(phi=phi, T=T, Y=Y)
    phi_bound = one_hot_phi_l2_bound(groups)
    sens = causal_moment_sensitivity(n=n, y_clip=y_clip, phi_bound=phi_bound)

    measured = measure_gaussian_vector(
        query=true_moments,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sens,
        rng=rng,
    )

    moment_mask = None
    if calibration_snr_threshold is not None:
        moment_mask = _snr_moment_mask(measured_moments=measured.noisy, noise_covariance=measured.covariance, threshold=calibration_snr_threshold)

    model, info = _fit_causal_model_from_noisy_moments(
        noisy_moments=measured.noisy,
        groups=groups,
        feature_names=feature_names,
        y_clip=y_clip,
        base_moments=None,
        measured_feature_mask_arr=None,
        measured_moment_mask_arr=moment_mask,
        ridge_alpha=float(calibration_ridge_alpha),
        y_noise_scale=float(np.std(Y)),
        refinement_steps=int(calibration_refinement_steps) if calibration_refinement_steps is not None else 6,
    )
    syn = model.sample(n=n_syn, rng=rng)

    est = _aipw_with_subsample(X=syn.phi, T=syn.T, Y=syn.Y, alpha=alpha, rng=rng)
    tvd = average_marginal_tvd(X_discrete_real, syn.X_discrete)

    return MethodResult(
        method="causal_naive",
        tau_hat=est.tau_hat,
        var_hat=est.var_hat,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
        tvd=tvd,
        metadata={
            "maxent_max_discrepancy": f"{info.get('max_discrepancy', np.nan):.6f}",
            "maxent_mean_abs_discrepancy": f"{info.get('mean_abs_discrepancy', np.nan):.6f}",
        },
    )


def run_causal_workload_na_mi(
    phi: np.ndarray,
    feature_names: Sequence[str],
    groups: Sequence[FeatureGroup],
    X_discrete_real: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    y_clip: float,
    M: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    calibration_snr_threshold: Optional[float] = None,
    calibration_ridge_alpha: float = 0.0,
    calibration_refinement_steps: Optional[int] = None,
) -> MethodResult:
    n = len(Y)
    true_moments = compute_causal_moments(phi=phi, T=T, Y=Y)
    phi_bound = one_hot_phi_l2_bound(groups)
    sens = causal_moment_sensitivity(n=n, y_clip=y_clip, phi_bound=phi_bound)

    measured = measure_gaussian_vector(
        query=true_moments,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sens,
        rng=rng,
    )

    discrepancies: List[float] = []
    moment_mask = None
    if calibration_snr_threshold is not None:
        moment_mask = _snr_moment_mask(measured_moments=measured.noisy, noise_covariance=measured.covariance, threshold=calibration_snr_threshold)
    refinement_steps = int(calibration_refinement_steps) if calibration_refinement_steps is not None else 6

    def _moment_projection(draw: np.ndarray) -> np.ndarray:
        return project_causal_moment_vector(draw, groups=groups, y_clip=y_clip)

    def _make_syn(moment_draw: np.ndarray, local_rng: np.random.Generator) -> SyntheticSample:
        model, info = _fit_causal_model_from_noisy_moments(
            noisy_moments=moment_draw,
            groups=groups,
            feature_names=feature_names,
            y_clip=y_clip,
            base_moments=None,
            measured_feature_mask_arr=None,
            measured_moment_mask_arr=moment_mask,
            ridge_alpha=float(calibration_ridge_alpha),
            y_noise_scale=float(np.std(Y)),
            refinement_steps=refinement_steps,
        )
        discrepancies.append(float(info.get("max_discrepancy", np.nan)))
        return model.sample(n=n_syn, rng=local_rng)

    mi = noise_aware_multiple_imputation(
        measured_moments=measured.noisy,
        noise_covariance=measured.covariance,
        make_synthetic_fn=_make_syn,
        M=M,
        rng=rng,
        alpha=alpha,
        keep_samples=False,
        moment_projection_fn=_moment_projection,
    )

    # TVD from one representative synthetic draw.
    syn0 = _make_syn(measured.noisy, rng)
    tvd = average_marginal_tvd(X_discrete_real, syn0.X_discrete)

    max_disc = float(np.nanmax(discrepancies)) if discrepancies else np.nan
    return MethodResult(
        method="causal_na_mi",
        tau_hat=mi.tau_hat,
        var_hat=mi.var_total,
        ci_low=mi.ci_low,
        ci_high=mi.ci_high,
        tvd=tvd,
        metadata={"maxent_max_discrepancy": f"{max_disc:.6f}"},
    )


def run_causal_aim_na_mi(
    phi: np.ndarray,
    feature_names: Sequence[str],
    groups: Sequence[FeatureGroup],
    X_discrete_real: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    y_clip: float,
    M: int,
    K: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    calibration_ridge_alpha: float = 0.0,
    calibration_refinement_steps: Optional[int] = None,
) -> MethodResult:
    n = len(Y)
    p = phi.shape[1]

    def _group_moment_indices(g: FeatureGroup) -> np.ndarray:
        idx = np.arange(g.start, g.stop)
        return np.concatenate([idx, idx + p, idx + 2 * p, idx + 3 * p], axis=0)

    def _base_moments_from_arm_means(
        p0: float,
        p1: float,
        mu0: float,
        mu1: float,
    ) -> np.ndarray:
        q0_count = np.zeros(p, dtype=float)
        q1_count = np.zeros(p, dtype=float)
        q0_y = np.zeros(p, dtype=float)
        q1_y = np.zeros(p, dtype=float)

        # Intercept.
        q0_count[0] = p0
        q1_count[0] = p1
        q0_y[0] = p0 * mu0
        q1_y[0] = p1 * mu1

        for g in groups:
            if g.name == "intercept":
                continue
            k = max(g.stop - g.start, 1)
            sl = slice(g.start, g.stop)
            q0_count[sl] = p0 / k
            q1_count[sl] = p1 / k
            q0_y[sl] = q0_count[sl] * mu0
            q1_y[sl] = q1_count[sl] * mu1

        return np.concatenate([q0_count, q1_count, q0_y, q1_y], axis=0)

    sel = select_causal_aim_groups(
        phi=phi,
        T=T,
        Y=Y,
        groups=groups,
        epsilon=epsilon,
        delta=delta,
        K=K,
        rng=rng,
        y_clip=y_clip,
    )

    selected_groups = ["intercept"] + list(sel.selected_groups)
    feat_mask = measured_feature_mask(groups=groups, selected_groups=selected_groups)
    group_map = {g.name: g for g in groups}

    # Base moments: public prior, but make intercept consistent with the first DP-measured group
    # (since sums over any group's one-hot levels equal the intercept moments).
    if sel.selected_groups:
        g0 = group_map[sel.selected_groups[0]]
        m0 = sel.measured[g0.name].noisy
        k0 = g0.stop - g0.start
        q0c = m0[0:k0]
        q1c = m0[k0 : 2 * k0]
        q0y = m0[2 * k0 : 3 * k0]
        q1y = m0[3 * k0 : 4 * k0]

        p0 = float(np.sum(q0c))
        p1 = float(np.sum(q1c))
        p0 = float(np.clip(p0, 1e-3, 1.0))
        p1 = float(np.clip(p1, 1e-3, 1.0))
        s = p0 + p1
        p0 /= s
        p1 /= s

        mu0 = float(np.sum(q0y) / max(p0, 1e-6))
        mu1 = float(np.sum(q1y) / max(p1, 1e-6))
        mu0 = float(np.clip(mu0, -y_clip, y_clip))
        mu1 = float(np.clip(mu1, -y_clip, y_clip))
    else:
        p0, p1, mu0, mu1 = 0.5, 0.5, 0.0, 0.0

    base_moments = _base_moments_from_arm_means(p0=p0, p1=p1, mu0=mu0, mu1=mu1)

    # Full moment vector + diagonal noise covariance.
    measured_moments = base_moments.copy()
    noise_var = np.zeros(4 * p, dtype=float)
    for name in sel.selected_groups:
        g = group_map[name]
        idx_full = _group_moment_indices(g)
        m = sel.measured[name]
        measured_moments[idx_full] = m.noisy
        noise_var[idx_full] = np.diag(m.covariance)

    discrepancies: List[float] = []

    def _moment_projection(draw: np.ndarray) -> np.ndarray:
        return project_causal_moment_vector(draw, groups=groups, y_clip=y_clip)

    def _make_syn(moment_draw: np.ndarray, local_rng: np.random.Generator) -> SyntheticSample:
        model, info = _fit_causal_model_from_noisy_moments(
            noisy_moments=moment_draw,
            groups=groups,
            feature_names=feature_names,
            y_clip=y_clip,
            base_moments=base_moments,
            measured_feature_mask_arr=feat_mask,
            measured_moment_mask_arr=None,
            ridge_alpha=float(calibration_ridge_alpha),
            y_noise_scale=float(np.std(Y)),
            refinement_steps=int(calibration_refinement_steps) if calibration_refinement_steps is not None else 6,
        )
        discrepancies.append(float(info.get("max_discrepancy", np.nan)))
        return model.sample(n=n_syn, rng=local_rng)

    mi = noise_aware_multiple_imputation(
        measured_moments=measured_moments,
        noise_covariance=noise_var,
        make_synthetic_fn=_make_syn,
        M=M,
        rng=rng,
        alpha=alpha,
        keep_samples=False,
        moment_projection_fn=_moment_projection,
    )

    syn0 = _make_syn(measured_moments, rng)
    tvd = average_marginal_tvd(X_discrete_real, syn0.X_discrete)

    max_disc = float(np.nanmax(discrepancies)) if discrepancies else np.nan
    return MethodResult(
        method="causal_aim_na_mi",
        tau_hat=mi.tau_hat,
        var_hat=mi.var_total,
        ci_low=mi.ci_low,
        ci_high=mi.ci_high,
        tvd=tvd,
        metadata={
            "selected_groups": ",".join(sel.selected_groups),
            "K": str(K),
            "eps_score_total": f"{sel.eps_score_total:.6g}",
            "eps_measure_total": f"{sel.eps_measure_total:.6g}",
            "delta_measure_total": f"{sel.delta_measure_total:.6g}",
            "maxent_max_discrepancy": f"{max_disc:.6f}",
        },
    )


def run_private_dr_baseline(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> MethodResult:
    est = dp_output_perturbed_dr(
        X=phi,
        T=T,
        Y=Y,
        epsilon=epsilon,
        delta=delta,
        y_clip=y_clip,
        rng=rng,
        alpha=alpha,
    )
    return MethodResult(
        method="private_dr_output_perturb",
        tau_hat=est.tau_hat,
        var_hat=est.var_hat,
        ci_low=est.ci_low,
        ci_high=est.ci_high,
    )


def run_dp_covbal_proxy(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> MethodResult:
    """
    Simplified DP covariate-balancing proxy baseline.

    Uses private perturbation of weighted arm means after balancing weights.
    """
    n = len(Y)
    T = T.astype(int)
    Y = np.clip(Y.astype(float), -y_clip, y_clip)

    # Crude balancing: weights inverse to empirical propensity.
    p_t = np.clip(np.mean(T), 0.05, 0.95)
    w = np.where(T == 1, 1.0 / p_t, 1.0 / (1.0 - p_t))
    tau = np.mean(w * (2 * T - 1) * Y)

    sens = 2.0 * y_clip / max(n, 1)
    sigma = sens * np.sqrt(2.0 * np.log(1.25 / delta)) / max(epsilon, 1e-8)
    tau_noisy = float(tau + rng.normal(0.0, sigma))

    var_hat = float(np.var(w * (2 * T - 1) * Y, ddof=1) / n + sigma ** 2)
    se = float(np.sqrt(max(var_hat, 1e-12)))
    z = float(1.959963984540054)

    return MethodResult(
        method="dp_covbal_proxy",
        tau_hat=tau_noisy,
        var_hat=var_hat,
        ci_low=float(tau_noisy - z * se),
        ci_high=float(tau_noisy + z * se),
    )
