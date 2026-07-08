"""Main entry point for Paper 2 experiments.

This script generates raw trial-level CSVs plus aggregated summaries for:
- Experiment 1: causal vs generic workloads (ATE RMSE and bias)
- Experiment 2: coverage calibration (derived from Exp 1 raw trials)
- Experiment 3: adaptive vs fixed workloads
- Experiment 4: ACS semi-synthetic study
- Ablations 1-4
- Marginal-fidelity vs causal-utility tradeoff table
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from threadpoolctl import threadpool_limits

import numpy as np
import pandas as pd

try:
    from .baselines import (
        MethodResult,
        run_aim_naive,
        run_causal_aim_na_mi,
        run_causal_workload_na_mi,
        run_causal_workload_naive,
        run_mst_naive,
        run_non_private_dr,
    )
    from .data import load_acs_causal, load_acic, load_ihdp, load_lalonde, load_twins
    from .metrics import decompose_rmse, summarize_ate_metrics
    from .utils import clip_outcome, ensure_dir, make_rng
    from .workloads import prepare_workload_features
except ImportError:  # pragma: no cover
    from baselines import (
        MethodResult,
        run_aim_naive,
        run_causal_aim_na_mi,
        run_causal_workload_na_mi,
        run_causal_workload_naive,
        run_mst_naive,
        run_non_private_dr,
    )
    from data import load_acs_causal, load_acic, load_ihdp, load_lalonde, load_twins
    from metrics import decompose_rmse, summarize_ate_metrics
    from utils import clip_outcome, ensure_dir, make_rng
    from workloads import prepare_workload_features


MAIN_DATASETS = ["ihdp", "twins", "acic_dgp7", "lalonde"]
EPS_GRID = [0.5, 1.0, 2.0, 5.0]
METHODS_MAIN = [
    "non_private_dr",
    "mst_naive",
    "aim_naive",
    "causal_naive",
    "causal_na_mi",
    "causal_aim_na_mi",
]
METHODS_COVERAGE = ["mst_naive", "aim_naive", "causal_naive", "causal_na_mi"]

MAX_CORES = 20  # user-approved cap for the 2026-07-03 real-data refresh
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

# Public benchmark outcome metadata used to put Y on the paper's public
# bounded-B scale before DP measurement. These constants are computed from the
# acquired benchmark files once and treated as public metadata of the benchmark.
PUBLIC_Y_STANDARDIZATION = {
    "ihdp": {"mean": 12.12, "std": 15.43},
    "twins": {"mean": 0.0, "std": 1.0},
    "acic_dgp7": {"mean": -0.3167, "std": 5.706},
    "lalonde": {"mean": 5300.763698561798, "std": 6624.036389547076},
    "acs": {"mean": 0.0, "std": 1.0},
}

TAU_SCALE_COLUMNS = ("tau_hat", "ci_low", "ci_high", "tau_true", "error")


def _set_threading_limits(n_threads: int) -> int:
    n = int(n_threads)
    if n <= 0:
        n = 1
    n = min(n, MAX_CORES)
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(n)
    threadpool_limits(limits=n)
    return n


def _configure_threading(n_jobs: int) -> int:
    """
    Best-effort cap on CPU parallelism.

    Replications are the unit of parallelism. Each process gets one BLAS/OpenMP
    thread, so args.n_jobs workers stays within the global MAX_CORES cap.
    """
    n = int(n_jobs)
    if n <= 0:
        n = 1
    workers = min(n, MAX_CORES)
    _set_threading_limits(1)
    return workers


def _worker_thread_init() -> None:
    _set_threading_limits(1)


def _replication_indices(total_reps: int, args) -> List[int]:
    start = int(getattr(args, "rep_start", 0))
    if start < 0:
        raise ValueError("--rep-start must be nonnegative")

    count = getattr(args, "rep_count", None)
    if count is None:
        stop = int(total_reps)
    else:
        count = int(count)
        if count < 0:
            raise ValueError("--rep-count must be nonnegative")
        stop = min(int(total_reps), start + count)

    if start >= int(total_reps):
        return []
    return list(range(start, stop))


def _selected_main_datasets(args) -> List[str]:
    raw = str(getattr(args, "datasets", "") or "").strip()
    if not raw:
        return list(MAIN_DATASETS)

    out: List[str] = []
    allowed = set(MAIN_DATASETS)
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        if name not in allowed:
            raise ValueError(f"Unknown dataset in --datasets: {name}. Allowed: {sorted(allowed)}")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("--datasets was provided but no valid datasets were parsed")
    return out


def _run_replication_tasks(
    label: str,
    worker_fn: Callable[[Tuple], object],
    tasks: Sequence[Tuple],
    args,
) -> List[object]:
    tasks = list(tasks)
    if not tasks:
        return []

    workers = min(int(getattr(args, "n_jobs", 1)), len(tasks), MAX_CORES)
    if workers <= 1:
        return [worker_fn(task) for task in tasks]

    print(f"[{label}] replication_workers={workers} per_worker_threads=1")
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_thread_init) as pool:
        return list(pool.map(worker_fn, tasks, chunksize=1))


def _method_seed(method: str) -> int:
    mapping = {
        "mst_naive": 1,
        "aim_naive": 2,
        "causal_naive": 3,
        "causal_na_mi": 4,
        "causal_aim_na_mi": 5,
        "fixed_causal": 6,
        "causal_aim": 7,
        "non_private_dr": 8,
    }
    return int(mapping.get(method, 99))


def _dataset_seed(base_seed: int, dataset_name: str, rep: int, extra: int = 0) -> int:
    mapping = {
        "ihdp": 11,
        "twins": 12,
        "acic_dgp7": 13,
        "lalonde": 14,
        "acs": 15,
    }
    return int(base_seed + 1000 * mapping.get(dataset_name, 99) + 37 * rep + extra)


def _dataset_scaling_key(dataset_name: str) -> str:
    if dataset_name.startswith("acic_dgp"):
        return "acic_dgp7"
    if dataset_name.startswith("acs"):
        return "acs"
    return dataset_name


def _standardization_constants(dataset_name: str, source: str) -> Tuple[float, float]:
    if source == "synthetic_fallback":
        return 0.0, 1.0
    key = _dataset_scaling_key(dataset_name)
    cfg = PUBLIC_Y_STANDARDIZATION.get(key, {"mean": 0.0, "std": 1.0})
    return float(cfg["mean"]), float(cfg["std"])


def _standardize_dataset_outcomes(ds, dataset_name: str):
    source = str(ds.metadata.get("source", ""))
    mean, scale = _standardization_constants(dataset_name, source)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid outcome standardization scale for {dataset_name}: {scale}")

    ds.Y = (np.asarray(ds.Y, dtype=float) - mean) / scale
    if ds.y0 is not None:
        ds.y0 = (np.asarray(ds.y0, dtype=float) - mean) / scale
    if ds.y1 is not None:
        ds.y1 = (np.asarray(ds.y1, dtype=float) - mean) / scale
    ds.tau_true = float(ds.tau_true) / scale
    ds.metadata.update(
        {
            "outcome_standardization_mean": f"{mean:.12g}",
            "outcome_standardization_std": f"{scale:.12g}",
        }
    )
    return ds


def _outcome_scale(ds) -> float:
    return float(ds.metadata.get("outcome_standardization_std", "1.0"))


def unscale_row(row: Dict[str, object], outcome_scale: float) -> Dict[str, object]:
    """Convert tau-scale quantities from standardized units back to original units."""
    out = dict(row)
    s = float(outcome_scale)
    for col in TAU_SCALE_COLUMNS:
        if col in out and pd.notna(out[col]):
            out[col] = float(out[col]) * s
    if "var_hat" in out and pd.notna(out["var_hat"]):
        out["var_hat"] = float(out["var_hat"]) * s * s
    if "sq_error" in out and pd.notna(out["sq_error"]):
        out["sq_error"] = float(out["sq_error"]) * s * s
    return out


def _load_dataset(name: str, args, rep: int, bins: int = 5, acic_overlap: Optional[float] = None):
    seed = _dataset_seed(args.master_seed, name, rep)

    if name == "ihdp":
        ds = load_ihdp(
            seed=seed,
            replication=rep,
            ihdp_npz_path=args.ihdp_npz_path,
            use_real_if_available=args.use_real_data,
        )
    elif name == "twins":
        ds = load_twins(
            seed=seed,
            twins_csv_path=args.twins_csv_path,
            use_real_if_available=args.use_real_data,
        )
    elif name == "acic_dgp7":
        ds = load_acic(
            seed=seed,
            dgp=7,
            n=4802,
            acic_dir=args.acic_dir,
            overlap=acic_overlap,
            use_real_if_available=args.use_real_data,
        )
    elif name == "lalonde":
        ds = load_lalonde(
            seed=seed,
            lalonde_csv_path=args.lalonde_csv_path,
            use_real_if_available=args.use_real_data,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    ds = _standardize_dataset_outcomes(ds, name)
    wf = prepare_workload_features(ds.X, bins=bins)
    y = clip_outcome(ds.Y, lo=-args.y_clip, hi=args.y_clip)
    ds.metadata["post_standardization_clip_fraction"] = f"{float(np.mean(np.abs(ds.Y) > args.y_clip)):.12g}"

    return ds, wf, y


def _load_acs_dataset(args, rep: int, n: int, bins: int = 5):
    seed = _dataset_seed(args.master_seed, "acs", rep, extra=n)
    ds = load_acs_causal(
        seed=seed,
        n=n,
        state=args.acs_state,
        year=args.acs_year,
        use_real_if_available=args.use_real_data,
    )
    ds = _standardize_dataset_outcomes(ds, "acs")
    wf = prepare_workload_features(ds.X, bins=bins)
    y = clip_outcome(ds.Y, lo=-args.y_clip, hi=args.y_clip)
    ds.metadata["post_standardization_clip_fraction"] = f"{float(np.mean(np.abs(ds.Y) > args.y_clip)):.12g}"
    return ds, wf, y


def _result_to_row(
    result: MethodResult,
    dataset: str,
    epsilon: float,
    delta: float,
    rep: int,
    tau_true: float,
    n: int,
    n_syn: int,
    outcome_scale: float = 1.0,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row = {
        "dataset": dataset,
        "epsilon": float(epsilon),
        "delta": float(delta),
        "replication": int(rep),
        "method": result.method,
        "tau_hat": float(result.tau_hat),
        "var_hat": float(result.var_hat),
        "ci_low": float(result.ci_low),
        "ci_high": float(result.ci_high),
        "tau_true": float(tau_true),
        "n": int(n),
        "n_syn": int(n_syn),
        "tvd": float(result.tvd) if result.tvd is not None else np.nan,
    }
    for k, v in result.metadata.items():
        row[k] = v
    if extra:
        row.update(extra)
    return unscale_row(row, outcome_scale)


def _run_experiment_1_replication_task(task: Tuple[str, int, object]) -> List[Dict[str, object]]:
    dataset_name, rep, args = task
    _worker_thread_init()

    rows: List[Dict[str, object]] = []
    ds, wf, y = _load_dataset(dataset_name, args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    outcome_scale = _outcome_scale(ds)

    non_private = run_non_private_dr(phi=wf.phi, T=ds.T, Y=ds.Y)

    for eps in EPS_GRID:
        delta = 1.0 / max(n * n, 1)
        eps_key = int(round(100 * eps))

        rows.append(
            _result_to_row(
                non_private,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

        rng = make_rng(args.master_seed, 9001, _dataset_seed(args.master_seed, dataset_name, rep), eps_key, _method_seed("mst_naive"))
        mst = run_mst_naive(
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
        rows.append(
            _result_to_row(
                mst,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

        rng = make_rng(args.master_seed, 9001, _dataset_seed(args.master_seed, dataset_name, rep), eps_key, _method_seed("aim_naive"))
        aim = run_aim_naive(
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
            k_generic=args.aim_k,
        )
        rows.append(
            _result_to_row(
                aim,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

        rng = make_rng(args.master_seed, 9001, _dataset_seed(args.master_seed, dataset_name, rep), eps_key, _method_seed("causal_naive"))
        causal_naive = run_causal_workload_naive(
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
            rng=rng,
            calibration_snr_threshold=args.calibration_snr_threshold,
            calibration_ridge_alpha=args.calibration_ridge_alpha,
            calibration_refinement_steps=args.calibration_refinement_steps,
        )
        rows.append(
            _result_to_row(
                causal_naive,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

        rng = make_rng(args.master_seed, 9001, _dataset_seed(args.master_seed, dataset_name, rep), eps_key, _method_seed("causal_na_mi"))
        causal_mi = run_causal_workload_na_mi(
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
        rows.append(
            _result_to_row(
                causal_mi,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

        rng = make_rng(args.master_seed, 9001, _dataset_seed(args.master_seed, dataset_name, rep), eps_key, _method_seed("causal_aim_na_mi"))
        causal_aim = run_causal_aim_na_mi(
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
            K=args.causal_aim_k,
            rng=rng,
            calibration_ridge_alpha=args.calibration_ridge_alpha,
            calibration_refinement_steps=args.calibration_refinement_steps,
        )
        rows.append(
            _result_to_row(
                causal_aim,
                dataset=dataset_name,
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
            )
        )

    return rows


def run_experiment_1(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    reps = args.replications_main

    for dataset_name in _selected_main_datasets(args):
        print(f"[Exp1] Dataset={dataset_name}")
        tasks = [(dataset_name, rep, args) for rep in _replication_indices(reps, args)]
        for rep_rows in _run_replication_tasks(f"Exp1 {dataset_name}", _run_experiment_1_replication_task, tasks, args):
            rows.extend(rep_rows)

    raw = pd.DataFrame(rows)
    raw_path = results_dir / "exp1_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize_ate_metrics(raw, group_cols=["dataset", "epsilon"])
    summary = summary[summary["method"].isin(METHODS_MAIN)]
    summary.to_csv(results_dir / "exp1_summary.csv", index=False)

    run_experiment_2(results_dir=results_dir, exp1_raw=raw)

    return raw


def run_experiment_2(results_dir: Path, exp1_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Explicit coverage experiment from Exp1 trial-level outputs.
    """
    if exp1_raw is None:
        raw_path = results_dir / "exp1_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot run Exp2: missing {raw_path}")
        exp1_raw = pd.read_csv(raw_path)

    d = exp1_raw.copy()
    d = d[d["method"].isin(METHODS_COVERAGE)].copy()
    d["covered"] = ((d["ci_low"] <= d["tau_true"]) & (d["ci_high"] >= d["tau_true"])).astype(int)
    d["ci_length"] = d["ci_high"] - d["ci_low"]

    exp2 = (
        d.groupby(["dataset", "epsilon", "method"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            ci_length_mean=("ci_length", "mean"),
            ci_length_std=("ci_length", "std"),
            n_reps=("covered", "size"),
        )
        .sort_values(["dataset", "epsilon", "method"])
        .reset_index(drop=True)
    )

    exp2.to_csv(results_dir / "exp2_coverage.csv", index=False)
    exp2_plot = exp2.rename(columns={"ci_length_mean": "ci_length"})
    exp2_plot.to_csv(results_dir / "exp2_summary.csv", index=False)
    return exp2


def run_fidelity_from_exp1(results_dir: Path, exp1_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if exp1_raw is None:
        raw_path = results_dir / "exp1_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot compute fidelity: missing {raw_path}")
        exp1_raw = pd.read_csv(raw_path)

    fid = (
        exp1_raw.dropna(subset=["tvd"])
        .assign(error=lambda d: d["tau_hat"] - d["tau_true"], sq_error=lambda d: (d["tau_hat"] - d["tau_true"]) ** 2)
        .groupby(["dataset", "epsilon", "method"], as_index=False)
        .agg(tvd=("tvd", "mean"), rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))))
    )
    fid.to_csv(results_dir / "fidelity_tradeoff.csv", index=False)
    return fid


def _run_experiment_3_replication_task(task: Tuple[int, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rep, args = task
    _worker_thread_init()

    rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []

    K_grid = [1, 3, 5, 10]
    eps_grid = [0.5, 1.0, 2.0]

    ds, wf, y = _load_dataset("acic_dgp7", args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    outcome_scale = _outcome_scale(ds)

    for eps in eps_grid:
        delta = 1.0 / max(n * n, 1)
        eps_key = int(round(100 * eps))

        # Fixed workload baseline (all features), repeated across K for plotting.
        rng_fixed = make_rng(
            args.master_seed,
            9300,
            _dataset_seed(args.master_seed, "acic_dgp7", rep),
            eps_key,
            _method_seed("fixed_causal"),
        )
        fixed = run_causal_workload_na_mi(
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
            rng=rng_fixed,
            calibration_snr_threshold=args.calibration_snr_threshold,
            calibration_ridge_alpha=args.calibration_ridge_alpha,
            calibration_refinement_steps=args.calibration_refinement_steps,
        )

        for K in K_grid:
            rows.append(
                _result_to_row(
                    MethodResult(
                        method="fixed_causal",
                        tau_hat=fixed.tau_hat,
                        var_hat=fixed.var_hat,
                        ci_low=fixed.ci_low,
                        ci_high=fixed.ci_high,
                        tvd=fixed.tvd,
                    ),
                    dataset="acic_dgp7",
                    epsilon=eps,
                    delta=delta,
                    rep=rep,
                    tau_true=ds.tau_true,
                    n=n,
                    n_syn=n_syn,
                    outcome_scale=outcome_scale,
                    extra={"K": K},
                )
            )

        for K in K_grid:
            rng_adaptive = make_rng(
                args.master_seed,
                9300,
                _dataset_seed(args.master_seed, "acic_dgp7", rep),
                eps_key,
                _method_seed("causal_aim"),
                K,
            )
            adaptive = run_causal_aim_na_mi(
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
                rng=rng_adaptive,
                calibration_ridge_alpha=args.calibration_ridge_alpha,
                calibration_refinement_steps=args.calibration_refinement_steps,
            )

            row = _result_to_row(
                MethodResult(
                    method="causal_aim",
                    tau_hat=adaptive.tau_hat,
                    var_hat=adaptive.var_hat,
                    ci_low=adaptive.ci_low,
                    ci_high=adaptive.ci_high,
                    tvd=adaptive.tvd,
                ),
                dataset="acic_dgp7",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"K": K},
            )
            rows.append(row)

            selected_str = adaptive.metadata.get("selected_groups", "")
            if selected_str:
                for feat in [x for x in selected_str.split(",") if x]:
                    selection_rows.append(
                        {
                            "epsilon": eps,
                            "K": K,
                            "replication": rep,
                            "feature": feat,
                        }
                    )

    return rows, selection_rows


def run_experiment_3(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []

    tasks = [(rep, args) for rep in _replication_indices(args.replications_adaptive, args)]
    for rep_rows, rep_selection_rows in _run_replication_tasks("Exp3 ACIC adaptive", _run_experiment_3_replication_task, tasks, args):
        rows.extend(rep_rows)
        selection_rows.extend(rep_selection_rows)

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "exp3_raw.csv", index=False)
    summary = summarize_ate_metrics(raw, group_cols=["epsilon", "K"])
    summary.to_csv(results_dir / "exp3_summary.csv", index=False)

    sel_df = pd.DataFrame(selection_rows)
    if len(sel_df) > 0:
        freq = (
            sel_df.groupby("feature", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
    else:
        freq = pd.DataFrame(columns=["feature", "count"])
    freq.to_csv(results_dir / "exp3_selected_features.csv", index=False)

    return raw


def _run_experiment_4_replication_task(task: Tuple[int, int, object]) -> List[Dict[str, object]]:
    n_target, rep, args = task
    _worker_thread_init()

    rows: List[Dict[str, object]] = []
    ds, wf, y = _load_acs_dataset(args=args, rep=rep, n=n_target, bins=5)
    n = len(y)
    n_syn = int(args.n_syn_ratio * n)
    outcome_scale = _outcome_scale(ds)

    non_private = run_non_private_dr(phi=wf.phi, T=ds.T, Y=ds.Y)

    for eps in EPS_GRID:
        delta = 1.0 / max(n * n, 1)
        eps_key = int(round(100 * eps))
        ds_key = _dataset_seed(args.master_seed, "acs", rep, extra=n_target)

        rows.append(
            _result_to_row(
                non_private,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

        rng = make_rng(args.master_seed, 9400, ds_key, eps_key, _method_seed("mst_naive"))
        mst = run_mst_naive(
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
        rows.append(
            _result_to_row(
                mst,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

        rng = make_rng(args.master_seed, 9400, ds_key, eps_key, _method_seed("aim_naive"))
        aim = run_aim_naive(
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
            k_generic=args.aim_k,
        )
        rows.append(
            _result_to_row(
                aim,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

        rng = make_rng(args.master_seed, 9400, ds_key, eps_key, _method_seed("causal_naive"))
        causal_naive = run_causal_workload_naive(
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
            rng=rng,
            calibration_snr_threshold=args.calibration_snr_threshold,
            calibration_ridge_alpha=args.calibration_ridge_alpha,
            calibration_refinement_steps=args.calibration_refinement_steps,
        )
        rows.append(
            _result_to_row(
                causal_naive,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

        rng = make_rng(args.master_seed, 9400, ds_key, eps_key, _method_seed("causal_na_mi"))
        causal_mi = run_causal_workload_na_mi(
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
        rows.append(
            _result_to_row(
                causal_mi,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

        rng = make_rng(
            args.master_seed,
            9400,
            ds_key,
            eps_key,
            _method_seed("causal_aim_na_mi"),
            args.causal_aim_k,
        )
        aim_full = run_causal_aim_na_mi(
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
            K=args.causal_aim_k,
            rng=rng,
            calibration_ridge_alpha=args.calibration_ridge_alpha,
            calibration_refinement_steps=args.calibration_refinement_steps,
        )
        rows.append(
            _result_to_row(
                aim_full,
                dataset="acs",
                epsilon=eps,
                delta=delta,
                rep=rep,
                tau_true=ds.tau_true,
                n=n,
                n_syn=n_syn,
                outcome_scale=outcome_scale,
                extra={"n_target": n_target},
            )
        )

    return rows


def run_experiment_4(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_grid = [1000, 5000, 20000]

    for n_target in n_grid:
        print(f"[Exp4] ACS n={n_target}")
        tasks = [(n_target, rep, args) for rep in _replication_indices(args.replications_acs, args)]
        for rep_rows in _run_replication_tasks(f"Exp4 ACS n={n_target}", _run_experiment_4_replication_task, tasks, args):
            rows.extend(rep_rows)

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "exp4_raw.csv", index=False)

    summary = summarize_ate_metrics(raw, group_cols=["n_target", "epsilon"])
    summary.to_csv(results_dir / "exp4_summary.csv", index=False)
    return raw


def _run_ablation_workload_dimension_task(task: Tuple[int, int, object]) -> Dict[str, object]:
    bins, rep, args = task
    _worker_thread_init()
    eps = 1.0

    ds, wf, y = _load_dataset("ihdp", args=args, rep=rep, bins=bins)
    n = len(y)
    delta = 1.0 / max(n * n, 1)
    n_syn = int(args.n_syn_ratio * n)
    rng = make_rng(args.master_seed, 9500, bins, rep)

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

    return unscale_row(
        {
            "bins": bins,
            "p": wf.phi.shape[1],
            "replication": rep,
            "tau_hat": out.tau_hat,
            "tau_true": ds.tau_true,
            "error": out.tau_hat - ds.tau_true,
        },
        _outcome_scale(ds),
    )


def run_ablation_workload_dimension(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    bins_grid = [3, 5, 10, 20]

    for bins in bins_grid:
        tasks = [(bins, rep, args) for rep in _replication_indices(args.replications_ablation, args)]
        rows.extend(_run_replication_tasks(f"Ablation workload_dim bins={bins}", _run_ablation_workload_dimension_task, tasks, args))

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "ablation_workload_dim_raw.csv", index=False)

    summary_rows = []
    for bins, g in raw.groupby("bins"):
        dec = decompose_rmse(g["error"].to_numpy())
        summary_rows.append(
            {
                "bins": bins,
                "p": int(g["p"].iloc[0]),
                "rmse": dec["rmse"],
                "bias_component": dec["bias_sq"],
                "variance_component": dec["variance"],
                "n_rep": len(g),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("bins")
    summary.to_csv(results_dir / "ablation_workload_dim.csv", index=False)
    return raw


def _run_ablation_mi_draws_task(task: Tuple[int, int, object]) -> Dict[str, object]:
    M, rep, args = task
    _worker_thread_init()
    eps = 1.0

    ds, wf, y = _load_dataset("ihdp", args=args, rep=rep, bins=5)
    n = len(y)
    delta = 1.0 / max(n * n, 1)
    n_syn = int(args.n_syn_ratio * n)
    rng = make_rng(args.master_seed, 9600, M, rep)

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
        M=M,
        rng=rng,
        calibration_snr_threshold=args.calibration_snr_threshold,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        calibration_refinement_steps=args.calibration_refinement_steps,
    )

    return unscale_row(
        {
            "M": M,
            "replication": rep,
            "tau_hat": out.tau_hat,
            "tau_true": ds.tau_true,
            "ci_low": out.ci_low,
            "ci_high": out.ci_high,
            "covered": int(out.ci_low <= ds.tau_true <= out.ci_high),
        },
        _outcome_scale(ds),
    )


def run_ablation_mi_draws(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    M_grid = [5, 10, 20, 50, 100]

    for M in M_grid:
        tasks = [(M, rep, args) for rep in _replication_indices(args.replications_ablation_mi, args)]
        rows.extend(_run_replication_tasks(f"Ablation mi_draws M={M}", _run_ablation_mi_draws_task, tasks, args))

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "ablation_mi_draws_raw.csv", index=False)

    summary = raw.groupby("M", as_index=False).agg(coverage=("covered", "mean"), n_rep=("covered", "size"))
    summary.to_csv(results_dir / "ablation_mi_draws.csv", index=False)
    return raw


def _run_ablation_nsyn_task(task: Tuple[int, int, object]) -> Dict[str, object]:
    ratio, rep, args = task
    _worker_thread_init()
    eps = 1.0

    ds, wf, y = _load_dataset("ihdp", args=args, rep=rep, bins=5)
    n = len(y)
    n_syn = int(max(20, ratio * n))
    delta = 1.0 / max(n * n, 1)
    rng = make_rng(args.master_seed, 9700, ratio, rep)

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

    return unscale_row(
        {
            "ratio": ratio,
            "replication": rep,
            "tau_hat": out.tau_hat,
            "tau_true": ds.tau_true,
            "sq_error": (out.tau_hat - ds.tau_true) ** 2,
        },
        _outcome_scale(ds),
    )


def run_ablation_nsyn(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ratios = [1, 2, 5, 10, 50]

    for ratio in ratios:
        tasks = [(ratio, rep, args) for rep in _replication_indices(args.replications_ablation, args)]
        rows.extend(_run_replication_tasks(f"Ablation nsyn ratio={ratio}", _run_ablation_nsyn_task, tasks, args))

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "ablation_nsyn_raw.csv", index=False)

    summary = raw.groupby("ratio", as_index=False).agg(rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))), n_rep=("sq_error", "size"))
    summary.to_csv(results_dir / "ablation_nsyn.csv", index=False)
    return raw


def _run_ablation_overlap_task(task: Tuple[float, int, object]) -> List[Dict[str, object]]:
    eta, rep, args = task
    _worker_thread_init()

    rows: List[Dict[str, object]] = []
    eps = 1.0
    eps_key = int(round(100 * eps))

    ds, wf, y = _load_dataset("acic_dgp7", args=args, rep=rep, bins=5, acic_overlap=eta)
    n = len(y)
    delta = 1.0 / max(n * n, 1)
    n_syn = int(args.n_syn_ratio * n)
    eta_key = int(round(1000 * eta))
    ds_key = _dataset_seed(args.master_seed, "acic_dgp7", rep, extra=eta_key)

    rng = make_rng(args.master_seed, 9800, ds_key, eps_key, _method_seed("mst_naive"))
    mst = run_mst_naive(
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
    rng = make_rng(args.master_seed, 9800, ds_key, eps_key, _method_seed("aim_naive"))
    aim = run_aim_naive(
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
        k_generic=args.aim_k,
    )
    rng = make_rng(args.master_seed, 9800, ds_key, eps_key, _method_seed("causal_na_mi"))
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

    for out in [mst, aim, causal]:
        rows.append(
            unscale_row(
                {
                    "eta": eta,
                    "replication": rep,
                    "method": out.method,
                    "tau_hat": out.tau_hat,
                    "tau_true": ds.tau_true,
                    "sq_error": (out.tau_hat - ds.tau_true) ** 2,
                    "covered": int(out.ci_low <= ds.tau_true <= out.ci_high),
                },
                _outcome_scale(ds),
            )
        )
    return rows


def run_ablation_overlap(args, results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    etas = [0.01, 0.05, 0.1, 0.2, 0.3]

    for eta in etas:
        tasks = [(eta, rep, args) for rep in _replication_indices(args.replications_ablation, args)]
        for rep_rows in _run_replication_tasks(f"Ablation overlap eta={eta}", _run_ablation_overlap_task, tasks, args):
            rows.extend(rep_rows)

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "ablation_overlap_raw.csv", index=False)

    summary = (
        raw.groupby(["eta", "method"], as_index=False)
        .agg(
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            coverage=("covered", "mean"),
            n_rep=("covered", "size"),
        )
        .sort_values(["eta", "method"])
    )
    summary.to_csv(results_dir / "ablation_overlap.csv", index=False)
    return raw


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_fingerprints() -> Dict[str, str]:
    code_root = Path(__file__).resolve().parent
    fingerprints: Dict[str, str] = {}
    for path in sorted(code_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        fingerprints[path.relative_to(code_root).as_posix()] = _sha256_file(path)
    return fingerprints


def _dataset_manifest_metadata(args, include_acs: bool) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for name in _selected_main_datasets(args):
        try:
            seed = _dataset_seed(args.master_seed, name, 0)
            if name == "ihdp":
                ds = load_ihdp(
                    seed=seed,
                    replication=0,
                    ihdp_npz_path=args.ihdp_npz_path,
                    use_real_if_available=args.use_real_data,
                )
            elif name == "twins":
                ds = load_twins(
                    seed=seed,
                    twins_csv_path=args.twins_csv_path,
                    use_real_if_available=args.use_real_data,
                )
            elif name == "acic_dgp7":
                ds = load_acic(
                    seed=seed,
                    dgp=7,
                    n=4802,
                    acic_dir=args.acic_dir,
                    use_real_if_available=args.use_real_data,
                )
            elif name == "lalonde":
                ds = load_lalonde(
                    seed=seed,
                    lalonde_csv_path=args.lalonde_csv_path,
                    use_real_if_available=args.use_real_data,
                )
            else:  # pragma: no cover
                continue
            metadata[name] = dict(ds.metadata)
        except Exception as exc:
            metadata[name] = {"source": "load_error", "error": repr(exc)}

    if include_acs:
        try:
            ds = load_acs_causal(
                seed=_dataset_seed(args.master_seed, "acs", 0, extra=1000),
                n=1000,
                state=args.acs_state,
                year=args.acs_year,
                use_real_if_available=args.use_real_data,
            )
            metadata["acs"] = dict(ds.metadata)
        except Exception as exc:
            metadata["acs"] = {"source": "load_error", "error": repr(exc)}
    return metadata


def write_run_manifest(args, results_dir: Path, run_set: set[str]) -> None:
    manifest = {
        "args": vars(args),
        "code_fingerprints_sha256": _code_fingerprints(),
        "dataset_metadata": _dataset_manifest_metadata(args, include_acs=("exp4" in run_set)),
    }
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def maybe_run_plots(args):
    if not args.make_figures:
        return

    try:
        if __package__:
            from .plot_figures import plot_all_figures
        else:  # pragma: no cover
            from plot_figures import plot_all_figures
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not import plot_figures: {exc}")
        return

    plot_all_figures(results_dir=Path(args.results_dir), figures_dir=Path(args.figures_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 DP synthetic causal experiments")

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="../figures")

    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--y-clip", type=float, default=5.0)
    parser.add_argument("--n-syn-ratio", type=float, default=10.0)

    parser.add_argument("--mi-draws", type=int, default=20)
    parser.add_argument("--causal-aim-k", type=int, default=5)
    parser.add_argument("--aim-k", type=int, default=5)

    # Causal max-entropy calibration regularization (optional; defaults reproduce current behavior).
    parser.add_argument(
        "--calibration-snr-threshold",
        type=float,
        default=None,
        help="If set, drop/impute low-SNR causal moments during maxent calibration (causal methods only).",
    )
    parser.add_argument(
        "--calibration-ridge-alpha",
        type=float,
        default=0.0,
        help="Ridge-like shrinkage strength in maxent correction loop (causal methods only).",
    )
    parser.add_argument(
        "--calibration-refinement-steps",
        type=int,
        default=None,
        help="Override number of maxent correction steps (causal methods only).",
    )

    parser.add_argument("--quick", action="store_true", help="Run a fast smoke configuration")
    parser.add_argument("--all", action="store_true", help="Run all experiments, ablations, and fidelity")
    parser.add_argument("--experiments", type=str, default="", help="Comma list among {1,2,3,4}")
    parser.add_argument("--datasets", type=str, default="", help=f"Comma list subset for Exp1 among {','.join(MAIN_DATASETS)}")
    parser.add_argument("--ablations", type=str, default="", help="Comma list among {dim,mi,nsyn,overlap}")
    parser.add_argument("--fidelity", action="store_true", help="Compute fidelity-vs-causal table from Exp1")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=f"Replication workers (capped at {MAX_CORES}; each worker uses one BLAS/OpenMP thread; -1 means 'all', still capped)",
    )

    parser.add_argument("--replications-main", type=int, default=None)
    parser.add_argument("--replications-adaptive", type=int, default=None)
    parser.add_argument("--replications-acs", type=int, default=None)
    parser.add_argument("--replications-ablation", type=int, default=None)
    parser.add_argument("--replications-ablation-mi", type=int, default=None)
    parser.add_argument("--rep-start", type=int, default=0, help="First replication index to run for shard-style execution")
    parser.add_argument("--rep-count", type=int, default=None, help="Number of replication indices to run from --rep-start")

    parser.add_argument(
        "--run",
        type=str,
        default="",
        help="Comma-separated subset: exp1,exp3,exp4,ablation1,ablation2,ablation3,ablation4",
    )
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Alias for --make-figures")

    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--ihdp-npz-path", type=str, default=None)
    parser.add_argument("--twins-csv-path", type=str, default=None)
    parser.add_argument("--acic-dir", type=str, default=None)
    parser.add_argument("--lalonde-csv-path", type=str, default=None)
    parser.add_argument("--acs-state", type=str, default="CA")
    parser.add_argument("--acs-year", type=int, default=2018)

    args = parser.parse_args()

    if args.plot:
        args.make_figures = True

    if args.quick:
        args.replications_main = args.replications_main or 6
        args.replications_adaptive = args.replications_adaptive or 6
        args.replications_acs = args.replications_acs or 4
        args.replications_ablation = args.replications_ablation or 6
        args.replications_ablation_mi = args.replications_ablation_mi or 30
    else:
        args.replications_main = args.replications_main or 500
        args.replications_adaptive = args.replications_adaptive or 500
        args.replications_acs = args.replications_acs or 200
        args.replications_ablation = args.replications_ablation or 500
        args.replications_ablation_mi = args.replications_ablation_mi or 200

    if args.all:
        args.experiments = "1,2,3,4"
        args.ablations = "dim,mi,nsyn,overlap"
        args.fidelity = True

    return args


def main():
    args = parse_args()
    if args.n_jobs == -1:
        args.n_jobs = max(1, os.cpu_count() or 1)
    elif args.n_jobs < 1:
        args.n_jobs = 1
    args.n_jobs = _configure_threading(args.n_jobs)

    run_set = set()
    if args.run:
        run_set.update({x.strip() for x in args.run.split(",") if x.strip()})

    if args.experiments:
        for e in [x.strip() for x in args.experiments.split(",") if x.strip()]:
            run_set.add(f"exp{e}")

    if args.ablations:
        alias = {
            "dim": "ablation1",
            "mi": "ablation2",
            "nsyn": "ablation3",
            "overlap": "ablation4",
        }
        for a in [x.strip() for x in args.ablations.split(",") if x.strip()]:
            if a in alias:
                run_set.add(alias[a])

    if not run_set and not args.fidelity:
        run_set = {"exp1", "exp2", "exp3", "exp4", "ablation1", "ablation2", "ablation3", "ablation4"}
        args.fidelity = True

    results_dir = ensure_dir(Path(args.results_dir))
    write_run_manifest(args, results_dir, run_set)
    print("Results directory:", results_dir)
    print(f"replication_workers (capped): {args.n_jobs} (MAX_CORES={MAX_CORES}, per_worker_threads=1)")

    exp1_raw = None
    if "exp1" in run_set:
        exp1_raw = run_experiment_1(args, results_dir)

    if "exp2" in run_set:
        run_experiment_2(results_dir=results_dir, exp1_raw=exp1_raw)

    if "exp3" in run_set:
        run_experiment_3(args, results_dir)

    if "exp4" in run_set:
        run_experiment_4(args, results_dir)

    if "ablation1" in run_set:
        run_ablation_workload_dimension(args, results_dir)

    if "ablation2" in run_set:
        run_ablation_mi_draws(args, results_dir)

    if "ablation3" in run_set:
        run_ablation_nsyn(args, results_dir)

    if "ablation4" in run_set:
        run_ablation_overlap(args, results_dir)

    if args.fidelity:
        run_fidelity_from_exp1(results_dir=results_dir, exp1_raw=exp1_raw)

    maybe_run_plots(args)
    print("All requested runs completed.")


if __name__ == "__main__":
    main()
