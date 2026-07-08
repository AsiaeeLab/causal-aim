"""Causal effect estimators used in experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold


@dataclass
class EstimateResult:
    tau_hat: float
    var_hat: float
    se_hat: float
    ci_low: float
    ci_high: float


def _is_binary(y: np.ndarray) -> bool:
    vals = np.unique(y)
    return len(vals) <= 2 and np.all(np.isin(vals, [0, 1]))


def _fit_propensity(X_train: np.ndarray, T_train: np.ndarray):
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X_train, T_train)
    return model


def _fit_outcome(X_train: np.ndarray, y_train: np.ndarray, is_binary: bool):
    if is_binary:
        model = LogisticRegression(max_iter=400, solver="lbfgs")
    else:
        model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def _predict_outcome(model, X_test: np.ndarray, is_binary: bool) -> np.ndarray:
    if is_binary:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        return np.asarray(pred, dtype=float)
    pred = model.predict(X_test)
    return np.asarray(pred, dtype=float)


def aipw_estimate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
    seed: int = 42,
    prop_clip: float = 0.02,
    alpha: float = 0.05,
) -> EstimateResult:
    """Cross-fitted AIPW/DR estimate with influence-function variance."""
    n = len(Y)
    if n < 10:
        raise ValueError("Need at least 10 rows for AIPW estimation")

    T = T.astype(int)
    Y = Y.astype(float)
    X = np.asarray(X, dtype=float)

    e_hat = np.zeros(n, dtype=float)
    m0_hat = np.zeros(n, dtype=float)
    m1_hat = np.zeros(n, dtype=float)

    folds = min(max(2, n_splits), n)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_binary = _is_binary(Y)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, Y_train = T[train_idx], Y[train_idx]

        # Propensity model.
        try:
            prop_model = _fit_propensity(X_train, T_train)
            e_fold = prop_model.predict_proba(X_test)[:, 1]
        except Exception:
            e_fold = np.full(len(test_idx), np.mean(T_train))
        e_hat[test_idx] = e_fold

        # Outcome models by treatment arm.
        for arm, target in [(0, m0_hat), (1, m1_hat)]:
            arm_mask = T_train == arm
            if np.sum(arm_mask) < 20:
                target[test_idx] = np.mean(Y_train[T_train == arm]) if np.any(T_train == arm) else np.mean(Y_train)
                continue

            try:
                out_model = _fit_outcome(X_train[arm_mask], Y_train[arm_mask], is_binary=y_binary)
                target[test_idx] = _predict_outcome(out_model, X_test, is_binary=y_binary)
            except Exception:
                # Stable linear fallback.
                fallback = Ridge(alpha=1.0)
                fallback.fit(X_train[arm_mask], Y_train[arm_mask])
                target[test_idx] = fallback.predict(X_test)

    e_hat = np.clip(e_hat, prop_clip, 1.0 - prop_clip)

    psi = m1_hat - m0_hat + T * (Y - m1_hat) / e_hat - (1 - T) * (Y - m0_hat) / (1.0 - e_hat)
    tau_hat = float(np.mean(psi))
    var_hat = float(np.var(psi, ddof=1) / n)
    var_hat = max(var_hat, 1e-12)
    se_hat = float(np.sqrt(var_hat))

    z = norm.ppf(1.0 - alpha / 2.0)
    ci_low = float(tau_hat - z * se_hat)
    ci_high = float(tau_hat + z * se_hat)

    return EstimateResult(
        tau_hat=tau_hat,
        var_hat=var_hat,
        se_hat=se_hat,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def regression_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> EstimateResult:
    """Simple regression-adjusted ATE via separate arm models."""
    X = np.asarray(X, dtype=float)
    T = T.astype(int)
    Y = Y.astype(float)

    y_binary = _is_binary(Y)

    model0 = _fit_outcome(X[T == 0], Y[T == 0], is_binary=y_binary)
    model1 = _fit_outcome(X[T == 1], Y[T == 1], is_binary=y_binary)

    m0 = _predict_outcome(model0, X, is_binary=y_binary)
    m1 = _predict_outcome(model1, X, is_binary=y_binary)

    tau_i = m1 - m0
    tau_hat = float(np.mean(tau_i))
    var_hat = float(np.var(tau_i, ddof=1) / len(Y))
    se_hat = float(np.sqrt(max(var_hat, 1e-12)))
    z = norm.ppf(1.0 - alpha / 2.0)

    return EstimateResult(
        tau_hat=tau_hat,
        var_hat=var_hat,
        se_hat=se_hat,
        ci_low=float(tau_hat - z * se_hat),
        ci_high=float(tau_hat + z * se_hat),
    )


def ipw_estimate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> EstimateResult:
    """IPW with logistic propensity model."""
    X = np.asarray(X, dtype=float)
    T = T.astype(int)
    Y = Y.astype(float)

    model = _fit_propensity(X, T)
    e_hat = np.clip(model.predict_proba(X)[:, 1], 0.02, 0.98)

    tau_i = T * Y / e_hat - (1 - T) * Y / (1.0 - e_hat)
    tau_hat = float(np.mean(tau_i))
    var_hat = float(np.var(tau_i, ddof=1) / len(Y))
    se_hat = float(np.sqrt(max(var_hat, 1e-12)))
    z = norm.ppf(1.0 - alpha / 2.0)

    return EstimateResult(
        tau_hat=tau_hat,
        var_hat=var_hat,
        se_hat=se_hat,
        ci_low=float(tau_hat - z * se_hat),
        ci_high=float(tau_hat + z * se_hat),
    )


def dp_output_perturbed_dr(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    delta: float,
    y_clip: float,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> EstimateResult:
    """Simplified direct DP baseline: DR estimate + Gaussian output perturbation."""
    base = aipw_estimate(X=X, T=T, Y=Y, alpha=alpha)

    n = len(Y)
    # Conservative sensitivity proxy for clipped outcomes.
    sensitivity = 2.0 * y_clip / max(n, 1)
    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / max(epsilon, 1e-8)

    noisy_tau = base.tau_hat + rng.normal(0.0, sigma)
    total_var = base.var_hat + sigma ** 2
    se = float(np.sqrt(max(total_var, 1e-12)))
    z = norm.ppf(1.0 - alpha / 2.0)

    return EstimateResult(
        tau_hat=float(noisy_tau),
        var_hat=float(total_var),
        se_hat=se,
        ci_low=float(noisy_tau - z * se),
        ci_high=float(noisy_tau + z * se),
    )
