#!/usr/bin/env python3
"""Promoted real-data runner for the rebuttal-era experiments.

This entry point ports the six rebuttal experiments into the main code tree
while reusing the real-data loaders, replication-level process pool, manifest,
and original-outcome-unit conversion from ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge

try:
    from . import run_experiments as rx
    from .baselines import (
        MethodResult,
        _fit_causal_model_from_noisy_moments,
        run_causal_aim_na_mi,
        run_causal_workload_na_mi,
        run_mst_naive,
    )
    from .data.common import CausalDataset
    from .estimators import EstimateResult, aipw_estimate
    from .mechanisms import measure_gaussian_vector
    from .na_mi import rubins_rules
    from .utils import ensure_dir, make_rng
    from .workloads import (
        WorkloadFeatures,
        causal_moment_sensitivity,
        compute_causal_moments,
        compute_generic_feature_marginals,
        one_hot_phi_l2_bound,
        prepare_workload_features,
        project_causal_moment_vector,
        split_causal_moment_vector,
    )
except ImportError:  # pragma: no cover
    import run_experiments as rx
    from baselines import (
        MethodResult,
        _fit_causal_model_from_noisy_moments,
        run_causal_aim_na_mi,
        run_causal_workload_na_mi,
        run_mst_naive,
    )
    from data.common import CausalDataset
    from estimators import EstimateResult, aipw_estimate
    from mechanisms import measure_gaussian_vector
    from na_mi import rubins_rules
    from utils import ensure_dir, make_rng
    from workloads import (
        WorkloadFeatures,
        causal_moment_sensitivity,
        compute_causal_moments,
        compute_generic_feature_marginals,
        one_hot_phi_l2_bound,
        prepare_workload_features,
        project_causal_moment_vector,
        split_causal_moment_vector,
    )


DEFAULT_EPS = [0.5, 1.0]
AIM_EPS_GRID = [0.5, 1.0, 2.0]
AIM_K_GRID = [1, 2, 3, 5, 10]
ACS_D_GRID = [20, 40, 60]
RIDGE_GRID = [0.001, 0.01, 0.1, 1.0]
REBUTTAL_METHOD_SEEDS = {"PrivATE": 21, "OhnishiAwan": 22}

TASK_ORDER = [
    "multi_estimand",
    "hybrid_workload",
    "direct_dp",
    "aim_operating_point",
    "scalability",
    "ridge_sensitivity",
]
TASK_ALIASES = {
    "1": "multi_estimand",
    "multi": "multi_estimand",
    "multi_estimand": "multi_estimand",
    "2": "hybrid_workload",
    "hybrid": "hybrid_workload",
    "hybrid_workload": "hybrid_workload",
    "3": "direct_dp",
    "direct": "direct_dp",
    "direct_dp": "direct_dp",
    "direct_dp_baselines": "direct_dp",
    "4": "aim_operating_point",
    "aim": "aim_operating_point",
    "aim_operating_point": "aim_operating_point",
    "5": "scalability",
    "acs": "scalability",
    "scalability": "scalability",
    "6": "ridge_sensitivity",
    "ridge": "ridge_sensitivity",
    "ridge_sensitivity": "ridge_sensitivity",
}


@dataclass
class SubgroupSpec:
    name: str
    group_name: str
    level: int
    real_mask: np.ndarray


def _normal_ci(tau_hat: float, var_hat: float, alpha: float = 0.05) -> EstimateResult:
    var_hat = float(max(var_hat, 1e-12))
    se = float(math.sqrt(var_hat))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return EstimateResult(
        tau_hat=float(tau_hat),
        var_hat=var_hat,
        se_hat=se,
        ci_low=float(tau_hat - z * se),
        ci_high=float(tau_hat + z * se),
    )


def _is_binary_vector(x: np.ndarray) -> bool:
    vals = pd.Series(x).dropna().unique()
    return len(vals) == 2


def _selected_multi_datasets(args) -> List[str]:
    raw = str(getattr(args, "multi_datasets", "") or "").strip()
    if not raw:
        return ["ihdp", "acic_dgp7", "lalonde"]
    allowed = {"ihdp", "acic_dgp7", "lalonde"}
    out: List[str] = []
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        if name not in allowed:
            raise ValueError(f"Unknown --multi-datasets entry: {name}. Allowed: {sorted(allowed)}")
        if name not in out:
            out.append(name)
    return out


def _selected_comparison_datasets(args) -> List[str]:
    raw = str(getattr(args, "comparison_datasets", "") or "").strip()
    if not raw:
        return ["ihdp", "acic_dgp7"]
    allowed = {"ihdp", "acic_dgp7"}
    out: List[str] = []
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        if name not in allowed:
            raise ValueError(f"Unknown --comparison-datasets entry: {name}. Allowed: {sorted(allowed)}")
        if name not in out:
            out.append(name)
    return out


def _select_subgroup(dataset: str, ds: CausalDataset, wf: WorkloadFeatures) -> SubgroupSpec:
    raw_cols = list(ds.X.columns)

    if dataset == "ihdp" and "bw" in ds.X.columns and _is_binary_vector(ds.X["bw"].to_numpy()):
        vals = sorted(pd.Series(ds.X["bw"]).dropna().unique().tolist())
        level = int(vals[-1])
        mask = ds.X["bw"].to_numpy() == level
        if "bw" in wf.X_discrete.columns:
            dvals = wf.X_discrete.loc[mask, "bw"]
            syn_level = int(dvals.mode().iloc[0])
            return SubgroupSpec(name="subgroup_bw", group_name="bw", level=syn_level, real_mask=mask)

    binary_cols = [c for c in raw_cols if _is_binary_vector(ds.X[c].to_numpy()) and c in wf.X_discrete.columns]
    if dataset == "acic_dgp7" and binary_cols:
        chosen = binary_cols[0]
    elif binary_cols:
        chosen = min(
            binary_cols,
            key=lambda c: abs(float(np.mean(ds.X[c].to_numpy() == pd.Series(ds.X[c]).mode().iloc[0])) - 0.5),
        )
    else:
        best: Optional[Tuple[str, int, float]] = None
        for col in wf.X_discrete.columns:
            counts = wf.X_discrete[col].value_counts(normalize=True)
            for level, frac in counts.items():
                score = abs(float(frac) - 0.5)
                if best is None or score < best[2]:
                    best = (str(col), int(level), score)
        if best is None:
            raise ValueError("Could not find a subgroup column")
        col, level, _ = best
        mask = wf.X_discrete[col].to_numpy() == level
        return SubgroupSpec(name=f"subgroup_{col}={level}", group_name=col, level=level, real_mask=mask)

    mode_val = pd.Series(ds.X[chosen]).mode().iloc[0]
    mask = ds.X[chosen].to_numpy() == mode_val
    syn_level = int(wf.X_discrete.loc[mask, chosen].mode().iloc[0])
    return SubgroupSpec(name=f"subgroup_{chosen}", group_name=chosen, level=syn_level, real_mask=mask)


def _diff_in_means(T: np.ndarray, Y: np.ndarray) -> float:
    T = np.asarray(T, dtype=int)
    Y = np.asarray(Y, dtype=float)
    if np.any(T == 1) and np.any(T == 0):
        return float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))
    return float("nan")


def _true_estimands(ds: CausalDataset, subgroup: SubgroupSpec) -> Tuple[Dict[str, float], Dict[str, str]]:
    if ds.y0 is not None and ds.y1 is not None:
        ite = np.asarray(ds.y1, dtype=float) - np.asarray(ds.y0, dtype=float)
        treated = np.asarray(ds.T, dtype=int) == 1
        subgroup_mask = np.asarray(subgroup.real_mask, dtype=bool)
        values = {
            "ATE": float(np.mean(ite)),
            "ATT": float(np.mean(ite[treated])) if treated.any() else np.nan,
            "Subgroup": float(np.mean(ite[subgroup_mask])) if subgroup_mask.any() else np.nan,
        }
        conventions = {key: "potential_outcome_mean" for key in values}
        return values, conventions

    subgroup_mask = np.asarray(subgroup.real_mask, dtype=bool)
    values = {
        "ATE": float(ds.tau_true),
        "ATT": _diff_in_means(ds.T, ds.Y),
        "Subgroup": _diff_in_means(np.asarray(ds.T)[subgroup_mask], np.asarray(ds.Y)[subgroup_mask])
        if subgroup_mask.any()
        else np.nan,
    }
    conventions = {
        "ATE": str(ds.metadata.get("tau_true_definition", "observed_difference_in_means")),
        "ATT": "observed_treated_minus_control_mean_no_individual_potential_outcomes",
        "Subgroup": "observed_subgroup_treated_minus_control_mean_no_individual_potential_outcomes",
    }
    return values, conventions


def _fit_outcome(X_train: np.ndarray, Y_train: np.ndarray):
    model = Ridge(alpha=1.0)
    model.fit(X_train, Y_train)
    return model


def _fallback_difference(T: np.ndarray, Y: np.ndarray) -> EstimateResult:
    T = np.asarray(T, dtype=int)
    Y = np.asarray(Y, dtype=float)
    y1 = Y[T == 1]
    y0 = Y[T == 0]
    if len(y1) == 0 or len(y0) == 0:
        return _normal_ci(float(np.mean(Y)) if len(Y) else 0.0, 1e6)
    tau = float(np.mean(y1) - np.mean(y0))
    var1 = float(np.var(y1, ddof=1)) if len(y1) > 1 else 0.0
    var0 = float(np.var(y0, ddof=1)) if len(y0) > 1 else 0.0
    var = float(var1 / max(len(y1), 1) + var0 / max(len(y0), 1))
    if var <= 1e-12:
        var = float(np.var(Y, ddof=1) / max(len(Y), 1)) if len(Y) > 1 else 1.0
    return _normal_ci(tau, var)


def _att_estimate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> EstimateResult:
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=int)
    Y = np.asarray(Y, dtype=float)
    treated = T == 1
    control = ~treated
    if np.sum(treated) < 10 or np.sum(control) < 20:
        return _fallback_difference(T, Y)

    try:
        model0 = _fit_outcome(X[control], Y[control])
        model1 = _fit_outcome(X[treated], Y[treated])
        m0_treated = model0.predict(X[treated])
        m1_treated = model1.predict(X[treated])
        tau_i = m1_treated - m0_treated
        tau = float(np.mean(tau_i))

        resid0 = Y[control] - model0.predict(X[control])
        resid1 = Y[treated] - model1.predict(X[treated])
        var_tau = float(np.var(tau_i, ddof=1) / max(len(tau_i), 1)) if len(tau_i) > 1 else 0.0
        # Rebuttal-era correction: add residual outcome-model variance, not just
        # heterogeneity in the predicted treated effects.
        var_resid = (
            (float(np.var(resid0, ddof=1)) if len(resid0) > 1 else 0.0)
            + (float(np.var(resid1, ddof=1)) if len(resid1) > 1 else 0.0)
        ) / max(len(tau_i), 1)
        return _normal_ci(tau, var_tau + var_resid, alpha=alpha)
    except Exception:
        return _fallback_difference(T, Y)


def _subgroup_ate_estimate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.05,
) -> EstimateResult:
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=int)
    Y = np.asarray(Y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if np.sum(mask) < 5:
        return _fallback_difference(T, Y)

    treated = T == 1
    control = ~treated
    if np.sum(treated) >= 20 and np.sum(control) >= 20:
        try:
            model0 = _fit_outcome(X[control], Y[control])
            model1 = _fit_outcome(X[treated], Y[treated])
            tau_i = model1.predict(X[mask]) - model0.predict(X[mask])
            tau = float(np.mean(tau_i))
            var = float(np.var(tau_i, ddof=1) / max(len(tau_i), 1)) if len(tau_i) > 1 else 1e-6
            return _normal_ci(tau, var, alpha=alpha)
        except Exception:
            pass

    if len(np.unique(T[mask])) < 2:
        return _fallback_difference(T, Y)
    return _fallback_difference(T[mask], Y[mask])


def _estimate_on_synthetic(sample, subgroup: SubgroupSpec) -> Dict[str, EstimateResult]:
    subgroup_mask = sample.X_discrete[subgroup.group_name].to_numpy(dtype=int) == int(subgroup.level)
    return {
        "ATE": aipw_estimate(X=sample.phi, T=sample.T, Y=sample.Y),
        "ATT": _att_estimate(X=sample.phi, T=sample.T, Y=sample.Y),
        "Subgroup": _subgroup_ate_estimate(X=sample.phi, T=sample.T, Y=sample.Y, mask=subgroup_mask),
    }


def _rubin_combine(estimates: Sequence[EstimateResult]) -> EstimateResult:
    pooled = rubins_rules(
        estimates=np.array([e.tau_hat for e in estimates], dtype=float),
        variances=np.array([e.var_hat for e in estimates], dtype=float),
    )
    return EstimateResult(
        tau_hat=float(pooled["tau_hat"]),
        var_hat=float(pooled["var_total"]),
        se_hat=float(math.sqrt(max(pooled["var_total"], 1e-12))),
        ci_low=float(pooled["ci_low"]),
        ci_high=float(pooled["ci_high"]),
    )


def _run_causal_na_mi_same_release(
    ds: CausalDataset,
    wf: WorkloadFeatures,
    y: np.ndarray,
    subgroup: SubgroupSpec,
    epsilon: float,
    delta: float,
    n_syn: int,
    args,
    rng: np.random.Generator,
) -> Dict[str, EstimateResult]:
    true_moments = compute_causal_moments(phi=wf.phi, T=ds.T, Y=y)
    phi_bound = one_hot_phi_l2_bound(wf.groups)
    sens = causal_moment_sensitivity(n=len(y), y_clip=args.y_clip, phi_bound=phi_bound)
    measured = measure_gaussian_vector(
        query=true_moments,
        epsilon=epsilon,
        delta=delta,
        sensitivity=sens,
        rng=rng,
    )

    cov = np.asarray(measured.covariance, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    per_estimand: Dict[str, List[EstimateResult]] = {"ATE": [], "ATT": [], "Subgroup": []}
    refinement = int(args.calibration_refinement_steps) if args.calibration_refinement_steps is not None else 6

    for _ in range(args.mi_draws):
        draw = measured.noisy + rng.normal(0.0, std)
        draw = project_causal_moment_vector(draw, groups=wf.groups, y_clip=args.y_clip)
        model, _ = _fit_causal_model_from_noisy_moments(
            noisy_moments=draw,
            groups=wf.groups,
            feature_names=wf.feature_names,
            y_clip=args.y_clip,
            ridge_alpha=args.calibration_ridge_alpha,
            y_noise_scale=float(np.std(y)),
            refinement_steps=refinement,
        )
        syn = model.sample(n=n_syn, rng=rng)
        ests = _estimate_on_synthetic(syn, subgroup=subgroup)
        for key, value in ests.items():
            per_estimand[key].append(value)

    return {key: _rubin_combine(vals) for key, vals in per_estimand.items()}


def _generic_marginal_vector(wf: WorkloadFeatures) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    marginals = compute_generic_feature_marginals(phi=wf.phi, groups=wf.groups)
    order: List[Tuple[str, int]] = []
    vals: List[float] = []
    for group in wf.groups:
        if group.name == "intercept":
            continue
        p = np.asarray(marginals[group.name], dtype=float)
        order.append((group.name, len(p)))
        vals.extend([float(x) for x in p])
    return np.asarray(vals, dtype=float), order


def _project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=float), 1e-8)
    return v / float(np.sum(v))


def _hybridize_moments(
    causal_draw: np.ndarray,
    generic_draw: np.ndarray,
    generic_order: Sequence[Tuple[str, int]],
    wf: WorkloadFeatures,
    y_clip: float,
) -> np.ndarray:
    p = wf.phi.shape[1]
    out = project_causal_moment_vector(causal_draw, groups=wf.groups, y_clip=y_clip)
    blocks = split_causal_moment_vector(out, p=p)
    group_map = {g.name: g for g in wf.groups}

    idx = 0
    for name, k in generic_order:
        g = group_map[name]
        sl = slice(g.start, g.stop)
        generic = _project_simplex(generic_draw[idx : idx + k])
        idx += k

        current_total = blocks.q0_count[sl] + blocks.q1_count[sl]
        current_total = _project_simplex(current_total)
        blended_total = _project_simplex(0.5 * current_total + 0.5 * generic)

        p0 = float(np.sum(blocks.q0_count[sl]))
        p1 = float(np.sum(blocks.q1_count[sl]))
        denom = np.maximum(blocks.q0_count[sl] + blocks.q1_count[sl], 1e-8)
        share1 = np.clip(blocks.q1_count[sl] / denom, 1e-4, 1.0 - 1e-4)
        q1 = blended_total * share1
        q0 = blended_total * (1.0 - share1)
        q0 = q0 * (p0 / max(float(np.sum(q0)), 1e-8))
        q1 = q1 * (p1 / max(float(np.sum(q1)), 1e-8))

        mean0 = blocks.q0_y[sl] / np.maximum(blocks.q0_count[sl], 1e-8)
        mean1 = blocks.q1_y[sl] / np.maximum(blocks.q1_count[sl], 1e-8)
        blocks.q0_count[sl] = q0
        blocks.q1_count[sl] = q1
        blocks.q0_y[sl] = q0 * np.clip(mean0, -y_clip, y_clip)
        blocks.q1_y[sl] = q1 * np.clip(mean1, -y_clip, y_clip)

    return np.concatenate([blocks.q0_count, blocks.q1_count, blocks.q0_y, blocks.q1_y], axis=0)


def _run_hybrid_na_mi(
    ds: CausalDataset,
    wf: WorkloadFeatures,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    n_syn: int,
    args,
    rng: np.random.Generator,
) -> MethodResult:
    n = len(y)
    causal_true = compute_causal_moments(phi=wf.phi, T=ds.T, Y=y)
    phi_bound = one_hot_phi_l2_bound(wf.groups)
    causal_sens = causal_moment_sensitivity(n=n, y_clip=args.y_clip, phi_bound=phi_bound)
    causal_measured = measure_gaussian_vector(
        query=causal_true,
        epsilon=max(epsilon / 2.0, 1e-8),
        delta=min(max(delta / 2.0, 1e-12), 0.999999),
        sensitivity=causal_sens,
        rng=rng,
    )

    generic_true, generic_order = _generic_marginal_vector(wf)
    generic_sens = 2.0 * math.sqrt(max(len(generic_order), 1)) / max(n, 1)
    generic_measured = measure_gaussian_vector(
        query=generic_true,
        epsilon=max(epsilon / 2.0, 1e-8),
        delta=min(max(delta / 2.0, 1e-12), 0.999999),
        sensitivity=generic_sens,
        rng=rng,
    )

    causal_std = np.sqrt(np.clip(np.diag(causal_measured.covariance), 0.0, None))
    generic_std = np.sqrt(np.clip(np.diag(generic_measured.covariance), 0.0, None))
    estimates: List[EstimateResult] = []
    refinement = int(args.calibration_refinement_steps) if args.calibration_refinement_steps is not None else 6

    for _ in range(args.mi_draws):
        causal_draw = causal_measured.noisy + rng.normal(0.0, causal_std)
        generic_draw = generic_measured.noisy + rng.normal(0.0, generic_std)
        moments = _hybridize_moments(
            causal_draw=causal_draw,
            generic_draw=generic_draw,
            generic_order=generic_order,
            wf=wf,
            y_clip=args.y_clip,
        )
        model, _ = _fit_causal_model_from_noisy_moments(
            noisy_moments=moments,
            groups=wf.groups,
            feature_names=wf.feature_names,
            y_clip=args.y_clip,
            ridge_alpha=args.calibration_ridge_alpha,
            y_noise_scale=float(np.std(y)),
            refinement_steps=refinement,
        )
        syn = model.sample(n=n_syn, rng=rng)
        estimates.append(aipw_estimate(X=syn.phi, T=syn.T, Y=syn.Y))

    pooled = _rubin_combine(estimates)
    return MethodResult(
        method="hybrid_na_mi",
        tau_hat=pooled.tau_hat,
        var_hat=pooled.var_hat,
        ci_low=pooled.ci_low,
        ci_high=pooled.ci_high,
        metadata={"note": "50/50 causal and generic one-way marginal budget split"},
    )


def _private_ate_output_perturb(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    y_clip: float,
    rng: np.random.Generator,
    prop_clip: float = 0.05,
) -> MethodResult:
    Yc = np.clip(np.asarray(Y, dtype=float), -y_clip, y_clip)
    base = aipw_estimate(X=phi, T=T, Y=Yc, prop_clip=prop_clip)
    n = max(len(Yc), 1)
    sensitivity = 2.0 * y_clip / (n * max(prop_clip, 1e-6))
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-8)
    tau = float(base.tau_hat + rng.normal(0.0, sigma))
    var = float(base.var_hat + sigma**2)
    se = math.sqrt(max(var, 1e-12))
    z = float(norm.ppf(0.975))
    return MethodResult(
        method="PrivATE",
        tau_hat=tau,
        var_hat=var,
        ci_low=float(tau - z * se),
        ci_high=float(tau + z * se),
        metadata={"proxy_family": "private_style_output_perturb", "prop_clip": f"{prop_clip:.6g}"},
    )


def _dp_covbal_output_perturb(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    y_clip: float,
    rng: np.random.Generator,
    prop_clip: float = 0.05,
) -> MethodResult:
    X = np.asarray(phi, dtype=float)
    if X.shape[1] > 1:
        X = X[:, 1:]
    T = np.asarray(T, dtype=int)
    Yc = np.clip(np.asarray(Y, dtype=float), -y_clip, y_clip)
    n = max(len(Yc), 1)
    try:
        model = LogisticRegression(max_iter=800, solver="lbfgs", C=0.5)
        model.fit(X, T)
        e = np.clip(model.predict_proba(X)[:, 1], prop_clip, 1.0 - prop_clip)
    except Exception:
        e = np.full(n, np.clip(float(np.mean(T)), prop_clip, 1.0 - prop_clip))

    score = T * Yc / e - (1 - T) * Yc / (1.0 - e)
    tau = float(np.mean(score))
    sensitivity = 2.0 * y_clip / (n * max(prop_clip, 1e-6))
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-8)
    tau_noisy = float(tau + rng.normal(0.0, sigma))
    var = float(np.var(score, ddof=1) / n + sigma**2)
    z = float(norm.ppf(0.975))
    se = math.sqrt(max(var, 1e-12))
    return MethodResult(
        method="OhnishiAwan",
        tau_hat=tau_noisy,
        var_hat=var,
        ci_low=float(tau_noisy - z * se),
        ci_high=float(tau_noisy + z * se),
        metadata={"proxy_family": "ohnishi_awan_style_covbal_output_perturb", "prop_clip": f"{prop_clip:.6g}"},
    )


def _seed_columns(
    args,
    dataset: str,
    rep: int,
    stream: int,
    method_seed: Optional[int] = None,
    dataset_extra: int = 0,
) -> Dict[str, object]:
    return {
        "master_seed": int(args.master_seed),
        "dataset_seed": int(rx._dataset_seed(args.master_seed, dataset, rep, extra=dataset_extra)),
        "seed_stream": int(stream),
        "method_seed": int(method_seed) if method_seed is not None else np.nan,
    }


def _finish_tau_row(row: Dict[str, object], outcome_scale: float) -> Dict[str, object]:
    out = dict(row)
    if "tau_true" not in out and "true_value" in out:
        out["tau_true"] = out["true_value"]
    if "error" not in out and "tau_hat" in out and "tau_true" in out:
        out["error"] = float(out["tau_hat"]) - float(out["tau_true"])
    if "sq_error" not in out and "error" in out:
        out["sq_error"] = float(out["error"]) ** 2
    if "covered" not in out and {"ci_low", "ci_high", "tau_true"}.issubset(out):
        out["covered"] = int(float(out["ci_low"]) <= float(out["tau_true"]) <= float(out["ci_high"]))
    if "ci_length" not in out and {"ci_low", "ci_high"}.issubset(out):
        out["ci_length"] = float(out["ci_high"]) - float(out["ci_low"])

    out = rx.unscale_row(out, outcome_scale)
    if "ci_length" in out and pd.notna(out["ci_length"]):
        out["ci_length"] = float(out["ci_length"]) * float(outcome_scale)
    if "true_value" in row and pd.notna(row["true_value"]):
        out["true_value"] = float(row["true_value"]) * float(outcome_scale)
    elif "tau_true" in out:
        out["true_value"] = out["tau_true"]
    return out


def _method_row(
    result: MethodResult,
    dataset: str,
    eps: float,
    delta: float,
    rep: int,
    true_value: float,
    n: int,
    n_syn: int,
    outcome_scale: float,
    seed_cols: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset,
        "eps": float(eps),
        "epsilon": float(eps),
        "delta": float(delta),
        "replication": int(rep),
        "method": result.method,
        "tau_hat": float(result.tau_hat),
        "var_hat": float(result.var_hat),
        "ci_low": float(result.ci_low),
        "ci_high": float(result.ci_high),
        "true_value": float(true_value),
        "tau_true": float(true_value),
        "n": int(n),
        "n_syn": int(n_syn),
        "tvd": float(result.tvd) if result.tvd is not None else np.nan,
    }
    row.update(seed_cols)
    row.update(result.metadata)
    if extra:
        row.update(extra)
    return _finish_tau_row(row, outcome_scale)


def _write_task_csv(df: pd.DataFrame, path: Path, leading_columns: Sequence[str]) -> pd.DataFrame:
    for col in leading_columns:
        if col not in df.columns:
            df[col] = np.nan
    ordered = list(leading_columns) + [c for c in df.columns if c not in leading_columns]
    out = df.loc[:, ordered]
    out.to_csv(path, index=False)
    return out


def _run_multi_estimand_task(task: Tuple[str, float, int, object]) -> List[Dict[str, object]]:
    dataset, eps, rep, args = task
    rx._worker_thread_init()

    ds, wf, y = rx._load_dataset(dataset, args=args, rep=rep, bins=5)
    subgroup = _select_subgroup(dataset, ds, wf)
    true_vals, conventions = _true_estimands(ds, subgroup)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    method_seed = rx._method_seed("causal_na_mi")
    seed_cols = _seed_columns(args, dataset, rep, stream=20101, method_seed=method_seed)
    rng = make_rng(args.master_seed, 20101, seed_cols["dataset_seed"], int(round(100 * eps)), method_seed)
    combined = _run_causal_na_mi_same_release(
        ds=ds,
        wf=wf,
        y=y,
        subgroup=subgroup,
        epsilon=eps,
        delta=delta,
        n_syn=n_syn,
        args=args,
        rng=rng,
    )

    rows = []
    for estimand, est in combined.items():
        row = {
            "dataset": dataset,
            "eps": float(eps),
            "epsilon": float(eps),
            "delta": float(delta),
            "replication": int(rep),
            "method": "causal_na_mi_same_release",
            "estimand": estimand,
            "tau_hat": est.tau_hat,
            "var_hat": est.var_hat,
            "ci_low": est.ci_low,
            "ci_high": est.ci_high,
            "true_value": true_vals[estimand],
            "tau_true": true_vals[estimand],
            "subgroup": subgroup.name,
            "subgroup_group": subgroup.group_name,
            "subgroup_level": subgroup.level,
            "true_value_convention": conventions[estimand],
            "n": n,
            "n_syn": n_syn,
            **seed_cols,
        }
        rows.append(_finish_tau_row(row, rx._outcome_scale(ds)))
    return rows


def run_multi_estimand(args, results_dir: Path) -> pd.DataFrame:
    tasks = [
        (dataset, eps, rep, args)
        for dataset in _selected_multi_datasets(args)
        for eps in DEFAULT_EPS
        for rep in range(int(args.n_rep))
    ]
    rows: List[Dict[str, object]] = []
    for rep_rows in rx._run_replication_tasks("Promoted multi-estimand", _run_multi_estimand_task, tasks, args):
        rows.extend(rep_rows)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "multi_estimand.csv",
        [
            "dataset",
            "eps",
            "epsilon",
            "delta",
            "replication",
            "method",
            "estimand",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "subgroup",
            "subgroup_group",
            "subgroup_level",
            "true_value_convention",
            "n",
            "n_syn",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _run_hybrid_task(task: Tuple[str, float, int, object]) -> List[Dict[str, object]]:
    dataset, eps, rep, args = task
    rx._worker_thread_init()

    ds, wf, y = rx._load_dataset(dataset, args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    ds_seed = rx._dataset_seed(args.master_seed, dataset, rep)
    outcome_scale = rx._outcome_scale(ds)

    rows: List[Dict[str, object]] = []
    specs = [
        ("hybrid_na_mi", 1, None),
        ("causal_na_mi", 2, rx._method_seed("causal_na_mi")),
        ("mst_naive", 3, rx._method_seed("mst_naive")),
    ]
    for method, stream_method, method_seed in specs:
        rng = make_rng(args.master_seed, 20303, ds_seed, int(round(100 * eps)), stream_method)
        if method == "hybrid_na_mi":
            result = _run_hybrid_na_mi(ds, wf, y, eps, delta, n_syn, args, rng)
            seed_method = stream_method
        elif method == "causal_na_mi":
            result = run_causal_workload_na_mi(
                phi=wf.phi,
                feature_names=wf.feature_names,
                groups=wf.groups,
                X_discrete_real=wf.X_discrete,
                T=ds.T,
                Y=y,
                epsilon=eps,
                delta=delta,
                n_syn=n_syn,
                y_clip=args.y_clip,
                M=args.mi_draws,
                rng=rng,
                calibration_snr_threshold=args.calibration_snr_threshold,
                calibration_ridge_alpha=args.calibration_ridge_alpha,
                calibration_refinement_steps=args.calibration_refinement_steps,
            )
            seed_method = method_seed
        else:
            result = run_mst_naive(
                phi=wf.phi,
                groups=wf.groups,
                X_discrete_real=wf.X_discrete,
                T=ds.T,
                Y=y,
                epsilon=eps,
                delta=delta,
                n_syn=n_syn,
                y_clip=args.y_clip,
                rng=rng,
            )
            result.method = "mst_naive"
            seed_method = method_seed

        rows.append(
            _method_row(
                result=result,
                dataset=dataset,
                eps=eps,
                delta=delta,
                rep=rep,
                true_value=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                seed_cols=_seed_columns(args, dataset, rep, stream=20303, method_seed=seed_method),
            )
        )
    return rows


def run_hybrid_workload(args, results_dir: Path) -> pd.DataFrame:
    tasks = [
        (dataset, eps, rep, args)
        for dataset in _selected_comparison_datasets(args)
        for eps in DEFAULT_EPS
        for rep in range(int(args.n_rep))
    ]
    rows: List[Dict[str, object]] = []
    for rep_rows in rx._run_replication_tasks("Promoted hybrid workload", _run_hybrid_task, tasks, args):
        rows.extend(rep_rows)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "hybrid_workload.csv",
        [
            "dataset",
            "eps",
            "epsilon",
            "delta",
            "replication",
            "method",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "n",
            "n_syn",
            "tvd",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _run_direct_dp_task(task: Tuple[str, float, int, object]) -> List[Dict[str, object]]:
    dataset, eps, rep, args = task
    rx._worker_thread_init()

    ds, wf, y = rx._load_dataset(dataset, args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    ds_seed = rx._dataset_seed(args.master_seed, dataset, rep)
    outcome_scale = rx._outcome_scale(ds)
    eps_key = int(round(100 * eps))

    rows: List[Dict[str, object]] = []
    proxy_specs = [
        ("PrivATE", _private_ate_output_perturb),
        ("OhnishiAwan", _dp_covbal_output_perturb),
    ]
    for method, fn in proxy_specs:
        method_seed = REBUTTAL_METHOD_SEEDS[method]
        rng = make_rng(args.master_seed, 20202, ds_seed, eps_key, method_seed)
        result = fn(phi=wf.phi, T=ds.T, Y=y, epsilon=eps, delta=delta, y_clip=args.y_clip, rng=rng)
        rows.append(
            _method_row(
                result=result,
                dataset=dataset,
                eps=eps,
                delta=delta,
                rep=rep,
                true_value=ds.tau_true,
                n=n,
                n_syn=0,
                outcome_scale=outcome_scale,
                seed_cols=_seed_columns(args, dataset, rep, stream=20202, method_seed=method_seed),
                extra={"n_syn_requested": n_syn},
            )
        )

    method_seed = rx._method_seed("causal_na_mi")
    rng = make_rng(args.master_seed, 20202, ds_seed, eps_key, method_seed)
    causal = run_causal_workload_na_mi(
        phi=wf.phi,
        feature_names=wf.feature_names,
        groups=wf.groups,
        X_discrete_real=wf.X_discrete,
        T=ds.T,
        Y=y,
        epsilon=eps,
        delta=delta,
        n_syn=n_syn,
        y_clip=args.y_clip,
        M=args.mi_draws,
        rng=rng,
        calibration_snr_threshold=args.calibration_snr_threshold,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_refinement_steps=args.calibration_refinement_steps,
    )
    causal.method = "causal_na_mi"
    rows.append(
        _method_row(
            result=causal,
            dataset=dataset,
            eps=eps,
            delta=delta,
            rep=rep,
            true_value=ds.tau_true,
            n=n,
            n_syn=n_syn,
            outcome_scale=outcome_scale,
            seed_cols=_seed_columns(args, dataset, rep, stream=20202, method_seed=method_seed),
        )
    )
    return rows


def run_direct_dp(args, results_dir: Path) -> pd.DataFrame:
    tasks = [
        (dataset, eps, rep, args)
        for dataset in _selected_comparison_datasets(args)
        for eps in DEFAULT_EPS
        for rep in range(int(args.n_rep))
    ]
    rows: List[Dict[str, object]] = []
    for rep_rows in rx._run_replication_tasks("Promoted direct DP", _run_direct_dp_task, tasks, args):
        rows.extend(rep_rows)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "direct_dp_baselines.csv",
        [
            "dataset",
            "eps",
            "epsilon",
            "delta",
            "replication",
            "method",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "n",
            "n_syn",
            "n_syn_requested",
            "tvd",
            "proxy_family",
            "prop_clip",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _run_aim_task(task: Tuple[float, int, int, object]) -> Dict[str, object]:
    eps, K, rep, args = task
    rx._worker_thread_init()

    dataset = "ihdp"
    ds, wf, y = rx._load_dataset(dataset, args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    method_seed = rx._method_seed("causal_aim_na_mi")
    seed_cols = _seed_columns(args, dataset, rep, stream=20505, method_seed=method_seed)
    rng = make_rng(args.master_seed, 20505, rep, int(round(100 * eps)), K)
    out = run_causal_aim_na_mi(
        phi=wf.phi,
        feature_names=wf.feature_names,
        groups=wf.groups,
        X_discrete_real=wf.X_discrete,
        T=ds.T,
        Y=y,
        epsilon=eps,
        delta=delta,
        n_syn=n_syn,
        y_clip=args.y_clip,
        M=args.mi_draws,
        K=K,
        rng=rng,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_refinement_steps=args.calibration_refinement_steps,
    )
    out.method = "causal_aim_na_mi"
    return _method_row(
        result=out,
        dataset=dataset,
        eps=eps,
        delta=delta,
        rep=rep,
        true_value=ds.tau_true,
        n=n,
        n_syn=n_syn,
        outcome_scale=rx._outcome_scale(ds),
        seed_cols=seed_cols,
        extra={"K": int(K)},
    )


def run_aim_operating_point(args, results_dir: Path) -> pd.DataFrame:
    tasks = [(eps, K, rep, args) for eps in AIM_EPS_GRID for K in AIM_K_GRID for rep in range(int(args.n_rep))]
    rows = rx._run_replication_tasks("Promoted Causal-AIM K sweep", _run_aim_task, tasks, args)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "aim_operating_point.csv",
        [
            "dataset",
            "eps",
            "epsilon",
            "delta",
            "K",
            "replication",
            "method",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "n",
            "n_syn",
            "tvd",
            "selected_groups",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _standardized_columns(X: pd.DataFrame) -> pd.DataFrame:
    arr = X.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = (arr - mu) / sd
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(out, columns=list(X.columns), index=X.index)


def _extend_acs_features(X: pd.DataFrame, target_d: int) -> Tuple[pd.DataFrame, str]:
    if X.shape[1] >= target_d:
        return X.iloc[:, :target_d].copy(), "truncated_to_target_d"

    base = X.reset_index(drop=True).copy()
    z = _standardized_columns(base)
    extras: Dict[str, np.ndarray] = {}
    cols = list(z.columns)

    for col in cols:
        extras[f"{col}_sq"] = z[col].to_numpy(dtype=float) ** 2
        if base.shape[1] + len(extras) >= target_d:
            break

    i = 0
    while base.shape[1] + len(extras) < target_d and cols:
        c1 = cols[i % len(cols)]
        c2 = cols[(i + 1) % len(cols)]
        extras[f"{c1}_x_{c2}"] = z[c1].to_numpy(dtype=float) * z[c2].to_numpy(dtype=float)
        i += 1

    i = 0
    while base.shape[1] + len(extras) < target_d and cols:
        c = cols[i % len(cols)]
        vals = z[c].to_numpy(dtype=float)
        extras[f"{c}_hi_{i}"] = (vals > np.nanmedian(vals)).astype(int)
        i += 1

    ext = pd.concat([base, pd.DataFrame(extras)], axis=1)
    return ext.iloc[:, :target_d].copy(), "deterministic_real_acs_feature_expansion"


def _load_acs_dimension(args, rep: int, d: int, n: int = 5000, bins: int = 5):
    ds, _, y = rx._load_acs_dataset(args=args, rep=rep, n=n, bins=bins)
    Xd, note = _extend_acs_features(ds.X, target_d=d)
    if Xd.shape[1] != ds.X.shape[1]:
        ds = CausalDataset(
            name=ds.name,
            X=Xd,
            T=ds.T,
            Y=ds.Y,
            tau_true=ds.tau_true,
            y0=ds.y0,
            y1=ds.y1,
            metadata={**ds.metadata, "dimension_note": note, "target_d": str(d)},
        )
    else:
        ds.metadata.update({"dimension_note": note, "target_d": str(d)})
    wf = prepare_workload_features(ds.X, bins=bins)
    return ds, wf, y


def _run_scalability_task(task: Tuple[int, int, object]) -> Dict[str, object]:
    d, rep, args = task
    rx._worker_thread_init()

    eps = 1.0
    n_target = int(args.scalability_n)
    ds, wf, y = _load_acs_dimension(args=args, rep=rep, d=d, n=n_target, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    method_seed = rx._method_seed("causal_na_mi")
    seed_cols = _seed_columns(args, "acs", rep, stream=20404, method_seed=method_seed, dataset_extra=n_target)
    rng = make_rng(args.master_seed, 20404, d, rep)

    start = time.perf_counter()
    out = run_causal_workload_na_mi(
        phi=wf.phi,
        feature_names=wf.feature_names,
        groups=wf.groups,
        X_discrete_real=wf.X_discrete,
        T=ds.T,
        Y=y,
        epsilon=eps,
        delta=delta,
        n_syn=n_syn,
        y_clip=args.y_clip,
        M=args.mi_draws,
        rng=rng,
        calibration_snr_threshold=args.calibration_snr_threshold,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_refinement_steps=args.calibration_refinement_steps,
    )
    elapsed = time.perf_counter() - start
    out.method = "causal_na_mi"
    return _method_row(
        result=out,
        dataset="acs",
        eps=eps,
        delta=delta,
        rep=rep,
        true_value=ds.tau_true,
        n=n,
        n_syn=n_syn,
        outcome_scale=rx._outcome_scale(ds),
        seed_cols=seed_cols,
        extra={
            "d": int(d),
            "n_target": n_target,
            "wallclock_sec": float(elapsed),
            "dimension_note": ds.metadata.get("dimension_note", ""),
        },
    )


def run_scalability(args, results_dir: Path) -> pd.DataFrame:
    tasks = [(d, rep, args) for d in ACS_D_GRID for rep in range(int(args.n_rep_scalability))]
    rows = rx._run_replication_tasks("Promoted ACS scalability", _run_scalability_task, tasks, args)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "scalability.csv",
        [
            "dataset",
            "d",
            "n",
            "n_target",
            "eps",
            "epsilon",
            "delta",
            "replication",
            "method",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "n_syn",
            "wallclock_sec",
            "dimension_note",
            "tvd",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _run_ridge_task(task: Tuple[float, int, object]) -> Dict[str, object]:
    lam, rep, args = task
    rx._worker_thread_init()

    dataset = "ihdp"
    eps = 1.0
    ds, wf, y = rx._load_dataset(dataset, args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    delta = 1.0 / max(n * n, 1)
    method_seed = int(round(lam * 1000))
    seed_cols = _seed_columns(args, dataset, rep, stream=20606, method_seed=method_seed)
    rng = make_rng(args.master_seed, 20606, rep, method_seed)
    out = run_causal_workload_na_mi(
        phi=wf.phi,
        feature_names=wf.feature_names,
        groups=wf.groups,
        X_discrete_real=wf.X_discrete,
        T=ds.T,
        Y=y,
        epsilon=eps,
        delta=delta,
        n_syn=n_syn,
        y_clip=args.y_clip,
        M=args.mi_draws,
        rng=rng,
        calibration_snr_threshold=args.calibration_snr_threshold,
        calibration_ridge_alpha=lam,
        calibration_refinement_steps=args.calibration_refinement_steps,
    )
    out.method = "causal_na_mi"
    return _method_row(
        result=out,
        dataset=dataset,
        eps=eps,
        delta=delta,
        rep=rep,
        true_value=ds.tau_true,
        n=n,
        n_syn=n_syn,
        outcome_scale=rx._outcome_scale(ds),
        seed_cols=seed_cols,
        extra={"lambda": float(lam)},
    )


def run_ridge_sensitivity(args, results_dir: Path) -> pd.DataFrame:
    tasks = [(lam, rep, args) for lam in RIDGE_GRID for rep in range(int(args.n_rep))]
    rows = rx._run_replication_tasks("Promoted ridge sensitivity", _run_ridge_task, tasks, args)
    df = pd.DataFrame(rows)
    return _write_task_csv(
        df,
        results_dir / "ridge_sensitivity.csv",
        [
            "dataset",
            "lambda",
            "eps",
            "epsilon",
            "delta",
            "replication",
            "method",
            "tau_hat",
            "var_hat",
            "ci_low",
            "ci_high",
            "true_value",
            "tau_true",
            "error",
            "sq_error",
            "covered",
            "ci_length",
            "n",
            "n_syn",
            "tvd",
            "master_seed",
            "dataset_seed",
            "seed_stream",
            "method_seed",
        ],
    )


def _parse_task_list(raw: str) -> List[str]:
    raw = (raw or "all").strip()
    if raw in {"all", ""}:
        return list(TASK_ORDER)
    out: List[str] = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        key = TASK_ALIASES.get(part)
        if key is None:
            raise ValueError(f"Unknown task {part!r}; allowed aliases include all, {sorted(TASK_ALIASES)}")
        if key not in out:
            out.append(key)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run promoted real-data rebuttal experiments.")
    parser.add_argument("--results-dir", type=str, default="results_promoted")
    parser.add_argument("--tasks", type=str, default="all", help="Comma list among 1..6 or task aliases; default all.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--y-clip", type=float, default=5.0)
    parser.add_argument("--n-syn-ratio", type=float, default=10.0)
    parser.add_argument("--mi-draws", type=int, default=20)
    parser.add_argument("--n-rep", type=int, default=200, help="Replications for tasks 1-4 and 6.")
    parser.add_argument(
        "--n-rep-scalability",
        type=int,
        default=None,
        help="Replications for task 5. Default is 100 for paper-grade runs, or --n-rep when --n-rep is overridden.",
    )
    parser.add_argument("--scalability-n", type=int, default=5000)
    parser.add_argument("--multi-datasets", type=str, default="ihdp,acic_dgp7,lalonde")
    parser.add_argument("--comparison-datasets", type=str, default="ihdp,acic_dgp7")
    parser.add_argument("--datasets", type=str, default="", help="Compatibility dataset subset for run_manifest.json.")

    parser.add_argument("--calibration-snr-threshold", type=float, default=None)
    parser.add_argument("--calibration-ridge-alpha", type=float, default=0.0)
    parser.add_argument("--calibration-refinement-steps", type=int, default=None)

    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--ihdp-npz-path", type=str, default=None)
    parser.add_argument("--twins-csv-path", type=str, default=None)
    parser.add_argument("--acic-dir", type=str, default=None)
    parser.add_argument("--lalonde-csv-path", type=str, default=None)
    parser.add_argument("--acs-state", type=str, default="CA")
    parser.add_argument("--acs-year", type=int, default=2018)
    args = parser.parse_args()

    if args.n_jobs == -1:
        import os

        args.n_jobs = max(1, os.cpu_count() or 1)
    elif args.n_jobs < 1:
        args.n_jobs = 1
    args.n_jobs = rx._configure_threading(args.n_jobs)

    if args.n_rep_scalability is None:
        args.n_rep_scalability = 100 if int(args.n_rep) == 200 else int(args.n_rep)
    return args


def main() -> None:
    args = parse_args()
    tasks = _parse_task_list(args.tasks)
    results_dir = ensure_dir(Path(args.results_dir))

    manifest_run_set = set(tasks)
    if "scalability" in tasks:
        manifest_run_set.add("exp4")
    rx.write_run_manifest(args, results_dir, manifest_run_set)

    print("Results directory:", results_dir)
    print(f"replication_workers (capped): {args.n_jobs} (MAX_CORES={rx.MAX_CORES}, per_worker_threads=1)")
    print("Promoted tasks:", ",".join(tasks))
    print(f"n_rep={args.n_rep}; n_rep_scalability={args.n_rep_scalability}; mi_draws={args.mi_draws}")

    runners = {
        "multi_estimand": run_multi_estimand,
        "hybrid_workload": run_hybrid_workload,
        "direct_dp": run_direct_dp,
        "aim_operating_point": run_aim_operating_point,
        "scalability": run_scalability,
        "ridge_sensitivity": run_ridge_sensitivity,
    }
    for task in tasks:
        print(f"[Promoted] {task}", flush=True)
        runners[task](args, results_dir)

    print("All promoted tasks completed.")


if __name__ == "__main__":
    main()
