"""DP mechanisms and simple privacy-budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GaussianMeasurement:
    noisy: np.ndarray
    sigma: float
    covariance: np.ndarray


def gaussian_noise_scale(sensitivity: float, epsilon: float, delta: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


def measure_gaussian_vector(
    query: np.ndarray,
    epsilon: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> GaussianMeasurement:
    sigma = gaussian_noise_scale(sensitivity=sensitivity, epsilon=epsilon, delta=delta)
    noise = rng.normal(0.0, sigma, size=len(query))
    noisy = query + noise
    cov = np.eye(len(query), dtype=float) * (sigma ** 2)
    return GaussianMeasurement(noisy=noisy, sigma=sigma, covariance=cov)


def measure_gaussian_subset(
    query: np.ndarray,
    measured_mask: np.ndarray,
    epsilon: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> GaussianMeasurement:
    if len(query) != len(measured_mask):
        raise ValueError("query and measured_mask must have same length")

    sigma = gaussian_noise_scale(sensitivity=sensitivity, epsilon=epsilon, delta=delta)
    noise = np.zeros_like(query)
    noise[measured_mask] = rng.normal(0.0, sigma, size=int(np.sum(measured_mask)))
    noisy = query + noise

    cov_diag = np.zeros(len(query), dtype=float)
    cov_diag[measured_mask] = sigma ** 2
    cov = np.diag(cov_diag)
    return GaussianMeasurement(noisy=noisy, sigma=sigma, covariance=cov)


def split_epsilon(epsilon: float, n_parts: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if n_parts <= 0:
        raise ValueError("n_parts must be > 0")
    if weights is None:
        return np.full(n_parts, epsilon / n_parts, dtype=float)

    w = np.asarray(weights, dtype=float)
    if len(w) != n_parts:
        raise ValueError("weights length mismatch")
    w = np.maximum(w, 1e-12)
    w = w / np.sum(w)
    return epsilon * w


def zcdp_compose_gaussian(rho_values: np.ndarray) -> float:
    """Total rho under zCDP composition."""
    rho_values = np.asarray(rho_values, dtype=float)
    return float(np.sum(np.maximum(rho_values, 0.0)))


def zcdp_to_approxdp(rho: float, delta: float) -> float:
    """Convert zCDP rho to (epsilon, delta)-DP upper bound."""
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return float(rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta)))
