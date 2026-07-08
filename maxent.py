"""Maximum-entropy style calibration for causal workload moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .utils import FeatureGroup, clip_prob, project_simplex_with_sum
    from .workloads import compute_causal_moments, expand_feature_mask_to_moment_mask, split_causal_moment_vector
except ImportError:  # pragma: no cover
    from utils import FeatureGroup, clip_prob, project_simplex_with_sum
    from workloads import compute_causal_moments, expand_feature_mask_to_moment_mask, split_causal_moment_vector

_MAXENT_WARNINGS_EMITTED = 0
_MAXENT_WARNINGS_CAP = 25


@dataclass
class SyntheticSample:
    X_discrete: pd.DataFrame
    phi: np.ndarray
    T: np.ndarray
    Y: np.ndarray


@dataclass
class MaxEntModel:
    """
    Product-style exponential-family approximation calibrated to noisy moments.
    """

    groups: List[FeatureGroup]
    feature_names: List[str]
    p_treat: float
    probs_t0: Dict[str, np.ndarray]
    probs_t1: Dict[str, np.ndarray]
    y_intercept_t0: float
    y_intercept_t1: float
    y_effects_t0: Dict[str, np.ndarray]
    y_effects_t1: Dict[str, np.ndarray]
    y_noise_scale: float = 1.0
    y_clip: float = 5.0

    def sample(self, n: int, rng: np.random.Generator) -> SyntheticSample:
        if n <= 0:
            raise ValueError("n must be positive")

        p = self.groups[-1].stop if self.groups else 0
        phi = np.zeros((n, p), dtype=float)
        discrete_cols: Dict[str, np.ndarray] = {}

        intercept_group = self.groups[0] if self.groups and self.groups[0].name == "intercept" else None
        if intercept_group is not None:
            phi[:, intercept_group.start : intercept_group.stop] = 1.0

        T = rng.binomial(1, float(clip_prob(self.p_treat)), size=n).astype(int)

        y_mu = np.where(T == 1, self.y_intercept_t1, self.y_intercept_t0).astype(float)
        denom = max(1, len(self.groups) - (1 if intercept_group is not None else 0))

        for g in self.groups:
            if g.name == "intercept":
                continue

            probs0 = self.probs_t0[g.name]
            probs1 = self.probs_t1[g.name]
            idx_choices = np.arange(len(g.levels))

            chosen = np.empty(n, dtype=int)
            t0_mask = T == 0
            t1_mask = ~t0_mask
            if t0_mask.any():
                chosen[t0_mask] = rng.choice(idx_choices, size=int(np.sum(t0_mask)), p=probs0)
            if t1_mask.any():
                chosen[t1_mask] = rng.choice(idx_choices, size=int(np.sum(t1_mask)), p=probs1)

            levels = np.asarray(g.levels, dtype=int)
            discrete_cols[g.name] = levels[chosen]

            for j in range(len(g.levels)):
                phi[:, g.start + j] = (chosen == j).astype(float)

            eff0 = self.y_effects_t0[g.name][chosen]
            eff1 = self.y_effects_t1[g.name][chosen]
            y_mu += np.where(T == 1, eff1, eff0) / denom

        Y = y_mu + rng.normal(0.0, self.y_noise_scale, size=n)
        Y = np.clip(Y, -self.y_clip, self.y_clip)

        X_discrete = pd.DataFrame(discrete_cols)
        return SyntheticSample(X_discrete=X_discrete, phi=phi, T=T, Y=Y)


def _infer_treatment_probs_from_intercept(q0_int: float, q1_int: float) -> Tuple[float, float]:
    p0 = max(float(q0_int), 1e-4)
    p1 = max(float(q1_int), 1e-4)
    s = p0 + p1
    p0 /= s
    p1 /= s
    p0 = float(np.clip(p0, 1e-3, 1.0 - 1e-3))
    p1 = 1.0 - p0
    return p0, p1


def expected_causal_moments(model: MaxEntModel) -> np.ndarray:
    """Compute model-implied causal moments analytically (no Monte Carlo)."""
    groups = model.groups
    p = groups[-1].stop if groups else 0
    q0_count = np.zeros(p, dtype=float)
    q1_count = np.zeros(p, dtype=float)
    q0_y = np.zeros(p, dtype=float)
    q1_y = np.zeros(p, dtype=float)

    p1 = float(np.clip(model.p_treat, 1e-6, 1.0 - 1e-6))
    p0 = 1.0 - p1

    denom = max(1, len(groups) - 1)

    # Intercept moments.
    q0_count[0] = p0
    q1_count[0] = p1
    q0_y[0] = p0 * model.y_intercept_t0
    q1_y[0] = p1 * model.y_intercept_t1

    for g in groups:
        if g.name == "intercept":
            continue

        sl = slice(g.start, g.stop)
        probs0 = np.asarray(model.probs_t0[g.name], dtype=float)
        probs1 = np.asarray(model.probs_t1[g.name], dtype=float)
        probs0 = probs0 / max(np.sum(probs0), 1e-12)
        probs1 = probs1 / max(np.sum(probs1), 1e-12)

        q0_count[sl] = p0 * probs0
        q1_count[sl] = p1 * probs1

        mean0 = model.y_intercept_t0 + np.asarray(model.y_effects_t0[g.name], dtype=float) / denom
        mean1 = model.y_intercept_t1 + np.asarray(model.y_effects_t1[g.name], dtype=float) / denom
        mean0 = np.clip(mean0, -model.y_clip, model.y_clip)
        mean1 = np.clip(mean1, -model.y_clip, model.y_clip)

        q0_y[sl] = q0_count[sl] * mean0
        q1_y[sl] = q1_count[sl] * mean1

    return np.concatenate([q0_count, q1_count, q0_y, q1_y], axis=0)


def verify_moments(
    model: MaxEntModel,
    measured_moments: np.ndarray,
    n_check: int = 50000,
    use_sampling: bool = False,
) -> Dict[str, float]:
    """
    Verify that model-implied moments match measured moments.

    If use_sampling=False (default), this uses analytic expectations and is cheap.
    """
    target = np.asarray(measured_moments, dtype=float)

    if use_sampling:
        rng = np.random.default_rng(0)
        syn = model.sample(n=int(max(n_check, 1000)), rng=rng)
        syn_moments = compute_causal_moments(phi=syn.phi, T=syn.T, Y=syn.Y)
    else:
        syn_moments = expected_causal_moments(model)

    abs_err = np.abs(syn_moments - target)
    return {
        "max_discrepancy": float(np.max(abs_err)),
        "mean_abs_discrepancy": float(np.mean(abs_err)),
    }


def _group_moment_indices(g: FeatureGroup, p: int) -> np.ndarray:
    idx = np.arange(g.start, g.stop)
    return np.concatenate([idx, idx + p, idx + 2 * p, idx + 3 * p], axis=0)


def _iterative_correction(
    model: MaxEntModel,
    target_moments: np.ndarray,
    steps: int = 8,
    lr: float = 0.6,
    ridge_alpha: float = 0.0,
) -> MaxEntModel:
    """Small mirror-descent style correction loop to reduce moment mismatch."""
    if steps <= 0:
        return model

    target = np.asarray(target_moments, dtype=float)
    p = model.groups[-1].stop if model.groups else 0
    t_blocks = split_causal_moment_vector(target, p=p)

    p0_tgt, p1_tgt = _infer_treatment_probs_from_intercept(t_blocks.q0_count[0], t_blocks.q1_count[0])
    model.p_treat = p1_tgt

    denom = max(1, len(model.groups) - 1)

    mu0_tgt = float(np.clip(t_blocks.q0_y[0] / max(p0_tgt, 1e-8), -model.y_clip, model.y_clip))
    mu1_tgt = float(np.clip(t_blocks.q1_y[0] / max(p1_tgt, 1e-8), -model.y_clip, model.y_clip))

    ridge_alpha = float(max(ridge_alpha, 0.0))
    decay = 1.0 / (1.0 + lr * ridge_alpha) if ridge_alpha > 0 else 1.0

    for _ in range(int(steps)):
        model.y_intercept_t0 = (1.0 - lr) * model.y_intercept_t0 + lr * mu0_tgt
        model.y_intercept_t1 = (1.0 - lr) * model.y_intercept_t1 + lr * mu1_tgt

        for g in model.groups:
            if g.name == "intercept":
                continue

            sl = slice(g.start, g.stop)

            # Update treatment-conditional level probabilities toward target.
            tgt0_joint = np.clip(t_blocks.q0_count[sl], 1e-10, None)
            tgt1_joint = np.clip(t_blocks.q1_count[sl], 1e-10, None)

            tgt0 = tgt0_joint / max(np.sum(tgt0_joint), 1e-10)
            tgt1 = tgt1_joint / max(np.sum(tgt1_joint), 1e-10)

            cur0 = np.asarray(model.probs_t0[g.name], dtype=float)
            cur1 = np.asarray(model.probs_t1[g.name], dtype=float)
            new0 = (1.0 - lr) * cur0 + lr * tgt0
            new1 = (1.0 - lr) * cur1 + lr * tgt1
            model.probs_t0[g.name] = new0 / max(np.sum(new0), 1e-10)
            model.probs_t1[g.name] = new1 / max(np.sum(new1), 1e-10)

            # Update outcome effects toward target conditional means.
            tgt_mean0 = t_blocks.q0_y[sl] / np.maximum(t_blocks.q0_count[sl], 1e-8)
            tgt_mean1 = t_blocks.q1_y[sl] / np.maximum(t_blocks.q1_count[sl], 1e-8)
            tgt_mean0 = np.clip(tgt_mean0, -model.y_clip, model.y_clip)
            tgt_mean1 = np.clip(tgt_mean1, -model.y_clip, model.y_clip)

            cur_eff0 = np.asarray(model.y_effects_t0[g.name], dtype=float)
            cur_eff1 = np.asarray(model.y_effects_t1[g.name], dtype=float)

            pred_mean0 = model.y_intercept_t0 + cur_eff0 / denom
            pred_mean1 = model.y_intercept_t1 + cur_eff1 / denom

            cur_eff0 = cur_eff0 + lr * (tgt_mean0 - pred_mean0) * denom
            cur_eff1 = cur_eff1 + lr * (tgt_mean1 - pred_mean1) * denom

            # Ridge-like shrinkage to avoid overfitting noisy constraints.
            if decay != 1.0:
                cur_eff0 = cur_eff0 * decay
                cur_eff1 = cur_eff1 * decay

            # Keep effects centered for stable intercept interpretation.
            p0_vec = model.probs_t0[g.name]
            p1_vec = model.probs_t1[g.name]
            cur_eff0 = cur_eff0 - float(np.sum(cur_eff0 * p0_vec))
            cur_eff1 = cur_eff1 - float(np.sum(cur_eff1 * p1_vec))

            model.y_effects_t0[g.name] = cur_eff0
            model.y_effects_t1[g.name] = cur_eff1

    return model


def _base_moments_from_arm_means(
    groups: Sequence[FeatureGroup],
    p0: float,
    p1: float,
    mu0: float,
    mu1: float,
) -> np.ndarray:
    """
    Public/simple prior moments: independent X with uniform mass within each group,
    and constant outcome means within each treatment arm.
    """
    p = groups[-1].stop if groups else 0
    q0_count = np.zeros(p, dtype=float)
    q1_count = np.zeros(p, dtype=float)
    q0_y = np.zeros(p, dtype=float)
    q1_y = np.zeros(p, dtype=float)

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


def _compute_snr_mask(
    moments: np.ndarray,
    noise_std: np.ndarray,
    threshold: float,
) -> np.ndarray:
    threshold = float(threshold)
    if threshold <= 0:
        return np.ones_like(moments, dtype=bool)

    denom = np.asarray(noise_std, dtype=float)
    denom = np.where(denom > 0.0, denom, np.inf)
    snr = np.abs(np.asarray(moments, dtype=float)) / denom
    return snr >= threshold


def calibrate_maxent_from_moments(
    noisy_moments: np.ndarray,
    groups: Sequence[FeatureGroup],
    feature_names: Sequence[str],
    y_clip: float = 5.0,
    base_moments: Optional[np.ndarray] = None,
    measured_feature_mask: Optional[np.ndarray] = None,
    measured_moment_mask: Optional[np.ndarray] = None,
    snr_threshold: Optional[float] = None,
    moment_noise_std: Optional[np.ndarray] = None,
    shrink_effects: float = 0.6,
    ridge_alpha: float = 0.0,
    y_noise_scale: float = 1.0,
    verify_after_fit: bool = True,
    verify_threshold: float = 0.1,
    refinement_steps: int = 8,
    refinement_lr: float = 0.6,
    verify_use_sampling: bool = False,
    verify_n_check: int = 50000,
    return_info: bool = False,
):
    """
    Fit a practical information-projection approximation from causal moments.

    Unmeasured moments can be imputed from base_moments via measured_feature_mask.
    """
    p = groups[-1].stop if groups else 0
    if len(noisy_moments) != 4 * p:
        raise ValueError(f"Expected noisy_moments length 4p={4*p}, got {len(noisy_moments)}")

    q = noisy_moments.copy()

    moment_mask = np.ones(4 * p, dtype=bool)
    if measured_feature_mask is not None:
        if len(measured_feature_mask) != p:
            raise ValueError("measured_feature_mask must have length p")
        moment_mask &= expand_feature_mask_to_moment_mask(measured_feature_mask)
    if measured_moment_mask is not None:
        mm = np.asarray(measured_moment_mask, dtype=bool)
        if len(mm) != 4 * p:
            raise ValueError("measured_moment_mask must have length 4p")
        moment_mask &= mm
    if snr_threshold is not None and moment_noise_std is not None:
        snr_mask = _compute_snr_mask(moments=noisy_moments, noise_std=moment_noise_std, threshold=float(snr_threshold))
        moment_mask &= snr_mask

    # Always treat intercept moments as measured.
    if p > 0:
        moment_mask[0] = True
        moment_mask[p] = True
        moment_mask[2 * p] = True
        moment_mask[3 * p] = True

    if not bool(np.all(moment_mask)):
        # If the caller didn't supply a base, fall back to a simple public prior based
        # on the (noisy) intercept moments.
        if base_moments is None:
            blocks0 = split_causal_moment_vector(noisy_moments, p=p)
            p0b, p1b = _infer_treatment_probs_from_intercept(blocks0.q0_count[0], blocks0.q1_count[0])
            mu0b = float(np.clip(blocks0.q0_y[0] / max(p0b, 1e-8), -y_clip, y_clip))
            mu1b = float(np.clip(blocks0.q1_y[0] / max(p1b, 1e-8), -y_clip, y_clip))
            base_moments = _base_moments_from_arm_means(groups=groups, p0=p0b, p1=p1b, mu0=mu0b, mu1=mu1b)

        q[~moment_mask] = base_moments[~moment_mask]

    blocks = split_causal_moment_vector(q, p=p)

    intercept_idx = 0
    p0, p1 = _infer_treatment_probs_from_intercept(
        q0_int=blocks.q0_count[intercept_idx],
        q1_int=blocks.q1_count[intercept_idx],
    )

    probs_t0: Dict[str, np.ndarray] = {}
    probs_t1: Dict[str, np.ndarray] = {}

    y_int0 = float(np.clip(blocks.q0_y[intercept_idx] / max(p0, 1e-6), -y_clip, y_clip))
    y_int1 = float(np.clip(blocks.q1_y[intercept_idx] / max(p1, 1e-6), -y_clip, y_clip))

    effects_t0: Dict[str, np.ndarray] = {}
    effects_t1: Dict[str, np.ndarray] = {}

    for g in groups:
        if g.name == "intercept":
            continue

        sl = slice(g.start, g.stop)

        counts0_joint = project_simplex_with_sum(blocks.q0_count[sl], target_sum=p0, min_value=1e-8)
        counts1_joint = project_simplex_with_sum(blocks.q1_count[sl], target_sum=p1, min_value=1e-8)

        probs0 = counts0_joint / max(np.sum(counts0_joint), 1e-8)
        probs1 = counts1_joint / max(np.sum(counts1_joint), 1e-8)
        probs_t0[g.name] = probs0
        probs_t1[g.name] = probs1

        mean0 = blocks.q0_y[sl] / np.maximum(blocks.q0_count[sl], 1e-6)
        mean1 = blocks.q1_y[sl] / np.maximum(blocks.q1_count[sl], 1e-6)
        mean0 = np.clip(mean0, -y_clip, y_clip)
        mean1 = np.clip(mean1, -y_clip, y_clip)

        eff0 = (mean0 - y_int0) * float(shrink_effects)
        eff1 = (mean1 - y_int1) * float(shrink_effects)

        eff0 -= np.sum(eff0 * probs0)
        eff1 -= np.sum(eff1 * probs1)

        effects_t0[g.name] = eff0
        effects_t1[g.name] = eff1

    model = MaxEntModel(
        groups=list(groups),
        feature_names=list(feature_names),
        p_treat=float(p1),
        probs_t0=probs_t0,
        probs_t1=probs_t1,
        y_intercept_t0=y_int0,
        y_intercept_t1=y_int1,
        y_effects_t0=effects_t0,
        y_effects_t1=effects_t1,
        y_noise_scale=float(max(0.1, y_noise_scale)),
        y_clip=float(y_clip),
    )

    # If mismatch is large, apply a few mirror-descent-style correction steps.
    model = _iterative_correction(
        model=model,
        target_moments=q,
        steps=refinement_steps,
        lr=refinement_lr,
        ridge_alpha=ridge_alpha,
    )

    info = {"max_discrepancy": np.nan, "mean_abs_discrepancy": np.nan}
    if verify_after_fit:
        info = verify_moments(
            model=model,
            measured_moments=q,
            n_check=verify_n_check,
            use_sampling=verify_use_sampling,
        )
        if info["max_discrepancy"] > verify_threshold:
            global _MAXENT_WARNINGS_EMITTED
            if _MAXENT_WARNINGS_EMITTED < _MAXENT_WARNINGS_CAP:
                print(
                    f"[WARN][maxent] moment mismatch max={info['max_discrepancy']:.4f} "
                    f"mean={info['mean_abs_discrepancy']:.4f} threshold={verify_threshold:.4f}"
                )
                _MAXENT_WARNINGS_EMITTED += 1

    if return_info:
        return model, info
    return model
