"""Shared utilities for the Paper 2 experimental pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureGroup:
    """One-hot slice for a single original covariate."""

    name: str
    start: int
    stop: int
    levels: List[int]


@dataclass
class DiscretizationSpec:
    """Stores binning metadata to reuse discretization rules."""

    numeric_edges: Dict[str, np.ndarray]
    categorical_levels: Dict[str, List[str]]


def make_rng(master_seed: int, *keys: int) -> np.random.Generator:
    """Deterministic generator based on SeedSequence."""
    entropy = [int(master_seed)] + [int(k) for k in keys]
    ss = np.random.SeedSequence(entropy)
    return np.random.default_rng(ss)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip_prob(p: np.ndarray | float, lo: float = 1e-3, hi: float = 1.0 - 1e-3):
    return np.clip(p, lo, hi)


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    return (x - mu) / (sd + eps)


def project_simplex_with_sum(v: np.ndarray, target_sum: float, min_value: float = 1e-8) -> np.ndarray:
    """Project vector onto positive simplex with fixed sum."""
    if len(v) == 0:
        return v
    if target_sum <= 0:
        return np.full_like(v, 1.0 / len(v))

    u = np.maximum(v, min_value)
    s = float(np.sum(u))
    if s <= 0:
        return np.full_like(v, target_sum / len(v))
    return u * (target_sum / s)


def discretize_dataframe(
    df: pd.DataFrame,
    bins: int = 5,
    spec: Optional[DiscretizationSpec] = None,
) -> Tuple[pd.DataFrame, DiscretizationSpec]:
    """
    Quantile-bin numeric columns and integer-encode categoricals.
    Returns a fully discrete dataframe with integer-coded columns.
    """
    out = pd.DataFrame(index=df.index)

    if spec is None:
        numeric_edges: Dict[str, np.ndarray] = {}
        categorical_levels: Dict[str, List[str]] = {}
    else:
        numeric_edges = {k: np.asarray(v) for k, v in spec.numeric_edges.items()}
        categorical_levels = {k: list(v) for k, v in spec.categorical_levels.items()}

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if spec is None or col not in numeric_edges:
                q = np.linspace(0.0, 1.0, bins + 1)
                edges = np.quantile(s.to_numpy(dtype=float), q)
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([float(np.min(s)) - 1e-6, float(np.max(s)) + 1e-6])
                numeric_edges[col] = edges
            edges = numeric_edges[col]
            # pd.cut requires monotonically increasing bin edges.
            edges = np.asarray(edges, dtype=float)
            if np.any(np.diff(edges) <= 0):
                edges = np.unique(edges)
                if len(edges) < 2:
                    mn = float(np.min(s))
                    mx = float(np.max(s))
                    edges = np.array([mn - 1e-6, mx + 1e-6])
                numeric_edges[col] = edges

            binned = pd.cut(
                s.astype(float),
                bins=edges,
                labels=False,
                include_lowest=True,
                duplicates="drop",
            )
            # Any NaNs from edge cases go to nearest valid bin.
            binned = binned.astype(float).fillna(0.0).astype(int)
            out[col] = binned
        else:
            s_str = s.astype(str)
            if spec is None or col not in categorical_levels:
                levels = sorted(s_str.unique().tolist())
                categorical_levels[col] = levels
            levels = categorical_levels[col]
            mapper = {lvl: idx for idx, lvl in enumerate(levels)}
            out[col] = s_str.map(mapper).fillna(0).astype(int)

    new_spec = DiscretizationSpec(numeric_edges=numeric_edges, categorical_levels=categorical_levels)
    return out, new_spec


def one_hot_from_discrete(
    discrete_df: pd.DataFrame,
    add_intercept: bool = True,
) -> Tuple[np.ndarray, List[str], List[FeatureGroup]]:
    """Create one-hot design matrix plus group metadata."""
    mats: List[np.ndarray] = []
    names: List[str] = []
    groups: List[FeatureGroup] = []

    cursor = 0
    if add_intercept:
        mats.append(np.ones((len(discrete_df), 1), dtype=float))
        names.append("intercept")
        groups.append(FeatureGroup(name="intercept", start=0, stop=1, levels=[1]))
        cursor = 1

    for col in discrete_df.columns:
        vals = discrete_df[col].to_numpy(dtype=int)
        levels = np.unique(vals)
        levels = np.sort(levels)
        oh = np.zeros((len(vals), len(levels)), dtype=float)
        for j, lvl in enumerate(levels):
            oh[:, j] = (vals == int(lvl)).astype(float)
            names.append(f"{col}={int(lvl)}")

        mats.append(oh)
        groups.append(
            FeatureGroup(
                name=str(col),
                start=cursor,
                stop=cursor + len(levels),
                levels=[int(x) for x in levels],
            )
        )
        cursor += len(levels)

    X = np.hstack(mats) if mats else np.empty((len(discrete_df), 0), dtype=float)
    return X, names, groups


def group_slice_map(groups: Sequence[FeatureGroup]) -> Dict[str, slice]:
    return {g.name: slice(g.start, g.stop) for g in groups}


def infer_group_from_feature_idx(groups: Sequence[FeatureGroup], idx: int) -> str:
    for g in groups:
        if g.start <= idx < g.stop:
            return g.name
    return "unknown"


def clip_outcome(y: np.ndarray, lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    return np.clip(y, lo, hi)


def stable_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if k >= len(values):
        return np.argsort(-values)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]


def coerce_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)
