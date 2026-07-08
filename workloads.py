"""Workload construction for causal and generic DP measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .utils import DiscretizationSpec, FeatureGroup, discretize_dataframe, one_hot_from_discrete
except ImportError:  # pragma: no cover
    from utils import DiscretizationSpec, FeatureGroup, discretize_dataframe, one_hot_from_discrete


@dataclass
class WorkloadFeatures:
    X_discrete: pd.DataFrame
    phi: np.ndarray
    feature_names: List[str]
    groups: List[FeatureGroup]
    discretization_spec: DiscretizationSpec


@dataclass
class CausalMomentBlocks:
    q0_count: np.ndarray
    q1_count: np.ndarray
    q0_y: np.ndarray
    q1_y: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.q0_count, self.q1_count, self.q0_y, self.q1_y], axis=0)


def prepare_workload_features(
    X: pd.DataFrame,
    bins: int = 5,
    spec: Optional[DiscretizationSpec] = None,
) -> WorkloadFeatures:
    Xd, fitted_spec = discretize_dataframe(X, bins=bins, spec=spec)
    phi, feature_names, groups = one_hot_from_discrete(Xd, add_intercept=True)
    return WorkloadFeatures(
        X_discrete=Xd,
        phi=phi,
        feature_names=feature_names,
        groups=groups,
        discretization_spec=fitted_spec,
    )


def compute_causal_moment_blocks(phi: np.ndarray, T: np.ndarray, Y: np.ndarray) -> CausalMomentBlocks:
    T = T.astype(float)
    Y = Y.astype(float)
    n = float(len(T))

    t0 = 1.0 - T
    t1 = T

    q0_count = (t0[:, None] * phi).sum(axis=0) / n
    q1_count = (t1[:, None] * phi).sum(axis=0) / n
    q0_y = (t0[:, None] * Y[:, None] * phi).sum(axis=0) / n
    q1_y = (t1[:, None] * Y[:, None] * phi).sum(axis=0) / n

    return CausalMomentBlocks(q0_count=q0_count, q1_count=q1_count, q0_y=q0_y, q1_y=q1_y)


def compute_causal_moments(phi: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return compute_causal_moment_blocks(phi=phi, T=T, Y=Y).as_vector()


def split_causal_moment_vector(v: np.ndarray, p: int) -> CausalMomentBlocks:
    if len(v) != 4 * p:
        raise ValueError(f"Expected vector length 4p={4*p}, got {len(v)}")
    return CausalMomentBlocks(
        q0_count=v[0:p],
        q1_count=v[p : 2 * p],
        q0_y=v[2 * p : 3 * p],
        q1_y=v[3 * p : 4 * p],
    )


def project_causal_moment_vector(
    moments: np.ndarray,
    groups: Sequence[FeatureGroup],
    y_clip: float = 5.0,
) -> np.ndarray:
    """
    Project a causal-moment vector to a basic feasible set.

    - Treatment-arm count blocks are clipped to [0, 1] and renormalized per group.
    - Outcome moments are converted to conditional means, clipped to [-y_clip, y_clip],
      then mapped back to weighted moments.
    """
    p = groups[-1].stop if groups else 0
    blocks = split_causal_moment_vector(np.asarray(moments, dtype=float), p=p)

    # Arm masses from intercept counts.
    p0 = float(np.clip(blocks.q0_count[0], 1e-4, 1.0 - 1e-4))
    p1 = float(np.clip(blocks.q1_count[0], 1e-4, 1.0 - 1e-4))
    s = p0 + p1
    p0 /= s
    p1 /= s

    q0_count = np.zeros_like(blocks.q0_count)
    q1_count = np.zeros_like(blocks.q1_count)
    q0_y = np.zeros_like(blocks.q0_y)
    q1_y = np.zeros_like(blocks.q1_y)

    # Intercept moments.
    q0_count[0] = p0
    q1_count[0] = p1
    mu0 = float(np.clip(blocks.q0_y[0] / max(p0, 1e-8), -y_clip, y_clip))
    mu1 = float(np.clip(blocks.q1_y[0] / max(p1, 1e-8), -y_clip, y_clip))
    q0_y[0] = p0 * mu0
    q1_y[0] = p1 * mu1

    for g in groups:
        if g.name == "intercept":
            continue
        sl = slice(g.start, g.stop)

        c0 = np.clip(blocks.q0_count[sl], 0.0, 1.0)
        c1 = np.clip(blocks.q1_count[sl], 0.0, 1.0)

        if np.sum(c0) <= 1e-10:
            c0 = np.full(len(c0), p0 / len(c0))
        else:
            c0 = c0 * (p0 / np.sum(c0))

        if np.sum(c1) <= 1e-10:
            c1 = np.full(len(c1), p1 / len(c1))
        else:
            c1 = c1 * (p1 / np.sum(c1))

        q0_count[sl] = c0
        q1_count[sl] = c1

        mean0 = blocks.q0_y[sl] / np.maximum(c0, 1e-8)
        mean1 = blocks.q1_y[sl] / np.maximum(c1, 1e-8)
        mean0 = np.clip(mean0, -y_clip, y_clip)
        mean1 = np.clip(mean1, -y_clip, y_clip)

        q0_y[sl] = c0 * mean0
        q1_y[sl] = c1 * mean1

    return np.concatenate([q0_count, q1_count, q0_y, q1_y], axis=0)


def one_hot_phi_l2_bound(groups: Sequence[FeatureGroup]) -> float:
    """
    L2 norm bound for rows of a one-hot design matrix built from `groups`.

    With the feature construction in `one_hot_from_discrete`, each row has
    exactly one active level per group (including the intercept group), so
    ||phi_i||_2 = sqrt(#groups).
    """
    return float(np.sqrt(max(len(groups), 1)))


def causal_moment_sensitivity(n: int, y_clip: float = 5.0, phi_bound: float = 1.0) -> float:
    """
    Replacement-adjacency L2 sensitivity for `compute_causal_moments`.

    The causal moment vector concatenates 4 blocks:
      q0_count, q1_count, q0_y, q1_y
    where each is an empirical average over n records. For a single record,
    the per-record contribution vector has L2 norm bounded by:
      ||phi_i||_2 * sqrt(1 + y_i^2) / n
    and with |y_i| <= y_clip and ||phi_i||_2 <= phi_bound, we get:
      Δ2 <= 2 * phi_bound * sqrt(1 + y_clip^2) / n.
    """
    n = max(int(n), 1)
    y_clip = float(abs(y_clip))
    phi_bound = float(abs(phi_bound))
    return float(2.0 * phi_bound * np.sqrt(1.0 + y_clip**2) / n)


def group_name_to_slice(groups: Sequence[FeatureGroup]) -> Dict[str, slice]:
    return {g.name: slice(g.start, g.stop) for g in groups}


def measured_feature_mask(groups: Sequence[FeatureGroup], selected_groups: Iterable[str]) -> np.ndarray:
    selected = set(selected_groups)
    p = groups[-1].stop if groups else 0
    mask = np.zeros(p, dtype=bool)
    for g in groups:
        if g.name in selected:
            mask[g.start : g.stop] = True
    return mask


def expand_feature_mask_to_moment_mask(feature_mask: np.ndarray) -> np.ndarray:
    return np.concatenate([feature_mask, feature_mask, feature_mask, feature_mask], axis=0)


def merge_moments_with_base(noisy: np.ndarray, base: np.ndarray, measured_mask: np.ndarray) -> np.ndarray:
    """Use noisy values for measured coordinates, base moments otherwise."""
    if len(noisy) != len(base) or len(base) != len(measured_mask):
        raise ValueError("Mask/base/noisy vectors must have same length")
    out = base.copy()
    out[measured_mask] = noisy[measured_mask]
    return out


def compute_generic_feature_marginals(phi: np.ndarray, groups: Sequence[FeatureGroup]) -> Dict[str, np.ndarray]:
    """Return marginal level probabilities for each feature group (ignores treatment/outcome)."""
    n = float(phi.shape[0])
    out: Dict[str, np.ndarray] = {}
    for g in groups:
        if g.name == "intercept":
            continue
        out[g.name] = np.sum(phi[:, g.start : g.stop], axis=0) / n
    return out


def compute_treatment_rate(T: np.ndarray) -> float:
    return float(np.mean(T.astype(float)))
