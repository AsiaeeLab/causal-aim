"""Common data structures and synthetic fallback generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..utils import clip_prob, sigmoid
except ImportError:  # pragma: no cover
    from utils import clip_prob, sigmoid


@dataclass
class CausalDataset:
    name: str
    X: pd.DataFrame
    T: np.ndarray
    Y: np.ndarray
    tau_true: float
    y0: Optional[np.ndarray] = None
    y1: Optional[np.ndarray] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.Y))

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


def _correlated_normal(n: int, d: int, rng: np.random.Generator, rho: float = 0.35) -> np.ndarray:
    idx = np.arange(d)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    return rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)


def make_covariates(
    n: int,
    d: int,
    rng: np.random.Generator,
    continuous_fraction: float = 0.7,
    prefix: str = "x",
) -> pd.DataFrame:
    """Generate mixed covariates with correlated latent structure."""
    z = _correlated_normal(n=n, d=d, rng=rng)
    cols = {}
    n_cont = int(round(d * continuous_fraction))

    for j in range(d):
        col = f"{prefix}{j+1}"
        if j < n_cont:
            cols[col] = z[:, j] + 0.1 * rng.normal(size=n)
        else:
            # Ternary-ish categorical variable derived from latent score.
            q = np.quantile(z[:, j], [1 / 3, 2 / 3])
            cat = np.digitize(z[:, j], bins=q).astype(int)
            cols[col] = cat
    return pd.DataFrame(cols)


def generate_confounded_outcomes(
    X: pd.DataFrame,
    rng: np.random.Generator,
    tau_base: float = 1.0,
    tau_scale: float = 0.5,
    confounding_strength: float = 1.0,
    nonlinear: bool = True,
    binary_outcome: bool = False,
    overlap: float = 0.15,
    noise_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Build treatment assignment with confounding and potential outcomes.

    overlap controls clipping of propensity to [overlap, 1-overlap].
    """
    Xn = X.to_numpy(dtype=float)
    d = Xn.shape[1]

    beta = rng.normal(0.0, 1.0, size=d)
    beta /= np.linalg.norm(beta) + 1e-8

    gamma = rng.normal(0.0, 1.0, size=d)
    gamma /= np.linalg.norm(gamma) + 1e-8

    score = confounding_strength * (Xn @ beta)
    if nonlinear and d >= 3:
        score += 0.8 * np.sin(Xn[:, 0]) - 0.4 * Xn[:, 1] * Xn[:, 2]

    prop = sigmoid(score)
    prop = overlap + (1.0 - 2.0 * overlap) * prop
    prop = clip_prob(prop, lo=max(1e-3, overlap), hi=1.0 - max(1e-3, overlap))
    T = rng.binomial(1, prop).astype(int)

    mu0 = (Xn @ gamma)
    if nonlinear and d >= 4:
        mu0 += 0.6 * np.cos(Xn[:, 2]) + 0.3 * (Xn[:, 0] ** 2 - Xn[:, 1])

    tau_x = tau_base + tau_scale * np.tanh(Xn[:, 0] if d > 0 else 0.0)
    if nonlinear and d >= 5:
        tau_x += 0.25 * np.sin(Xn[:, 3] - Xn[:, 4])

    if binary_outcome:
        p0 = sigmoid(mu0 / 2.0)
        p1 = sigmoid((mu0 + tau_x) / 2.0)
        y0 = rng.binomial(1, clip_prob(p0, 1e-4, 1 - 1e-4)).astype(float)
        y1 = rng.binomial(1, clip_prob(p1, 1e-4, 1 - 1e-4)).astype(float)
    else:
        y0 = mu0 + rng.normal(0.0, noise_scale, size=len(X))
        y1 = mu0 + tau_x + rng.normal(0.0, noise_scale, size=len(X))

    Y = y1 * T + y0 * (1 - T)
    tau_true = float(np.mean(y1 - y0))

    return {
        "T": T,
        "Y": Y.astype(float),
        "y0": y0.astype(float),
        "y1": y1.astype(float),
        "tau_true": np.array(tau_true, dtype=float),
        "propensity": prop.astype(float),
    }
