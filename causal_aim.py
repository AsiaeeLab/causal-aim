"""Adaptive causal workload selection (Causal-AIM).

This module implements a DP adaptive feature-group selection procedure that
returns (1) the selected groups and (2) the DP measurements taken during the
selection. Downstream code can reuse those measurements to synthesize data,
avoiding any second measurement pass (and avoiding leakage via "fill with true
moments" shortcuts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from .utils import FeatureGroup
    from .workloads import compute_causal_moments
    from .mechanisms import GaussianMeasurement, measure_gaussian_vector
except ImportError:  # pragma: no cover
    from utils import FeatureGroup
    from workloads import compute_causal_moments
    from mechanisms import GaussianMeasurement, measure_gaussian_vector


@dataclass
class CausalAIMResult:
    selected_groups: List[str]
    utilities: Dict[str, float]
    measured: Dict[str, GaussianMeasurement]
    eps_score_total: float
    eps_measure_total: float
    delta_measure_total: float


def _group_moment_indices(g: FeatureGroup, p: int) -> np.ndarray:
    idx = np.arange(g.start, g.stop)
    return np.concatenate([idx, idx + p, idx + 2 * p, idx + 3 * p], axis=0)


def _initialize_public_prior_moments(groups: Sequence[FeatureGroup], p: int, y_clip: float) -> np.ndarray:
    """
    Initialize a public (data-independent) prior moment vector.

    This avoids any non-DP dependence on the private dataset before the first
    measurement. The prior is intentionally simple:
    - p(T=1)=0.5
    - each group's levels are uniform within each arm
    - E[Y|T=t] = 0
    """
    p0 = 0.5
    p1 = 0.5
    mu0 = 0.0
    mu1 = 0.0

    q0_count = np.zeros(p, dtype=float)
    q1_count = np.zeros(p, dtype=float)
    q0_y = np.zeros(p, dtype=float)
    q1_y = np.zeros(p, dtype=float)

    # Intercept.
    q0_count[0] = p0
    q1_count[0] = p1
    q0_y[0] = p0 * np.clip(mu0, -y_clip, y_clip)
    q1_y[0] = p1 * np.clip(mu1, -y_clip, y_clip)

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


def _group_utility_v2(
    true_moments: np.ndarray,
    current_moments: np.ndarray,
    group: FeatureGroup,
    p: int,
    overlap_eta: float,
) -> float:
    """
    Utility proxy for Algorithm 1 in the paper:
    Lipschitz constant from Theorem 1 times unmeasured moment norm.
    """
    idx = _group_moment_indices(group, p)
    unmeasured_moment_norm = float(np.linalg.norm(true_moments[idx] - current_moments[idx], ord=2))
    lipschitz_const = 2.0 / max(overlap_eta, 1e-3)
    return lipschitz_const * unmeasured_moment_norm


def _causal_group_moment_l2_sensitivity(n: int, y_clip: float) -> float:
    """
    Replacement-adjacency L2 sensitivity for one group's causal moment subvector.

    The group subvector contains (for that group only):
      [E[(1-T)phi_g], E[T phi_g], E[(1-T)Y phi_g], E[T Y phi_g]]
    and each record contributes to exactly one level in the group.
    """
    n = max(int(n), 1)
    return float(2.0 * np.sqrt(1.0 + float(y_clip) ** 2) / n)


def _exponential_mechanism_choice(
    utilities: Dict[str, float],
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> str:
    if not utilities:
        raise ValueError("utilities must be non-empty")
    eps = max(float(epsilon), 0.0)
    sens = max(float(sensitivity), 1e-12)
    names = list(utilities.keys())
    scores = np.array([utilities[n] for n in names], dtype=float)

    # Exponential mechanism: p(name) ∝ exp(eps * u / (2*Δu)).
    scale = eps / (2.0 * sens)
    logits = scale * (scores - float(np.max(scores)))
    w = np.exp(logits)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0:
        return names[int(rng.integers(0, len(names)))]
    p = w / w_sum
    return str(rng.choice(names, p=p))


def select_causal_aim_groups(
    phi: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    groups: Sequence[FeatureGroup],
    epsilon: float,
    delta: float,
    K: int,
    rng: np.random.Generator,
    overlap_eta: float = 0.1,
    y_clip: float = 5.0,
) -> CausalAIMResult:
    """
    Select K groups via the exponential mechanism (DP argmax sampling) over
    Lipschitz-based utilities. After each selection, privately measure the
    selected group's moments and update a current-moment proxy.

    Budget split (simple sequential composition):
    - Total scoring budget: eps_score_total = epsilon/(K+1) (pure DP)
    - Total measurement budget: eps_measure_total = epsilon - eps_score_total
      (approx DP with total delta_measure_total = delta).
    """
    candidates = [g for g in groups if g.name != "intercept"]
    selected: List[str] = []
    utilities: Dict[str, float] = {}
    measured: Dict[str, GaussianMeasurement] = {}

    if K <= 0 or not candidates:
        return CausalAIMResult(
            selected_groups=selected,
            utilities=utilities,
            measured=measured,
            eps_score_total=0.0,
            eps_measure_total=0.0,
            delta_measure_total=0.0,
        )

    p = phi.shape[1]
    n = max(len(T), 1)

    true_moments = compute_causal_moments(phi=phi, T=T, Y=Y)
    current_moments = _initialize_public_prior_moments(groups=groups, p=p, y_clip=y_clip)

    n_rounds = min(K, len(candidates))
    eps_score_total = float(epsilon) / (float(K) + 1.0)
    eps_score_round = eps_score_total / float(max(n_rounds, 1))

    eps_measure_total = float(max(float(epsilon) - eps_score_total, 0.0))
    eps_measure_round = eps_measure_total / float(max(n_rounds, 1))

    delta_measure_total = float(delta)
    delta_measure_round = delta_measure_total / float(max(n_rounds, 1))

    lipschitz_const = 2.0 / max(float(overlap_eta), 1e-3)
    moment_sens = _causal_group_moment_l2_sensitivity(n=n, y_clip=y_clip)
    utility_sens = float(lipschitz_const * moment_sens)

    available = {g.name: g for g in candidates}

    for _ in range(n_rounds):
        round_utils: Dict[str, float] = {}

        for name, g in available.items():
            base_u = _group_utility_v2(
                true_moments=true_moments,
                current_moments=current_moments,
                group=g,
                p=p,
                overlap_eta=overlap_eta,
            )
            utilities[name] = base_u
            round_utils[name] = base_u

        best_name = _exponential_mechanism_choice(
            utilities=round_utils,
            epsilon=eps_score_round,
            sensitivity=utility_sens,
            rng=rng,
        )
        g = available.pop(best_name)
        selected.append(best_name)

        # Privately measure the newly selected group's moments and update proxy.
        idx = _group_moment_indices(g, p)
        m = measure_gaussian_vector(
            query=true_moments[idx],
            epsilon=max(eps_measure_round, 1e-8),
            delta=min(max(delta_measure_round, 1e-12), 0.999999),
            sensitivity=moment_sens,
            rng=rng,
        )
        measured[g.name] = m
        current_moments[idx] = m.noisy

    return CausalAIMResult(
        selected_groups=selected,
        utilities=utilities,
        measured=measured,
        eps_score_total=float(eps_score_total),
        eps_measure_total=float(eps_measure_total),
        delta_measure_total=float(delta_measure_total),
    )
