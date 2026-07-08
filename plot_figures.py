"""Figure generation for Paper 2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "non_private_dr": "Non-private DR",
    "mst_naive": "MST + naive DR",
    "aim_naive": "AIM + naive DR",
    "causal_naive": "Causal workload + naive DR",
    "causal_na_mi": "Causal workload + NA+MI",
    "causal_aim_na_mi": "Causal-AIM + NA+MI",
    "fixed_causal": "Fixed causal workload",
    "causal_aim": "Causal-AIM",
    "private_dr_output_perturb": "PrivATE (proxy)",
    "dp_covbal_proxy": "DP covbal (proxy)",
}

DATASET_LABELS = {
    "ihdp": "IHDP",
    "twins": "Twins",
    "acic_dgp7": "ACIC DGP 7",
    "lalonde": "LaLonde",
    "acs": "ACS semi-synthetic",
}


sns.set_theme(style="whitegrid", context="notebook")
# Figures are shrunk 2-4x when placed at \columnwidth/\textwidth, so fonts and
# line widths are sized for the post-shrink render, not the raw canvas.
plt.rcParams.update({
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.1,
    "grid.linewidth": 0.9,
    "legend.framealpha": 0.95,
})


def _boost(axes, title=21, label=20, tick=16):
    """Enlarge per-axes fonts for multi-panel figures that print small."""
    for ax in np.atleast_1d(axes).ravel():
        ax.title.set_size(title)
        ax.xaxis.label.set_size(label)
        ax.yaxis.label.set_size(label)
        ax.tick_params(axis="both", labelsize=tick)


def _label_method(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["method_label"] = d["method"].map(METHOD_LABELS).fillna(d["method"])
    return d


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing results file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_fig1_rmse(exp1_summary: pd.DataFrame, out_path: Path):
    if exp1_summary.empty:
        return

    datasets = ["ihdp", "twins", "acic_dgp7", "lalonde"]
    methods = ["mst_naive", "aim_naive", "causal_naive", "causal_na_mi", "causal_aim_na_mi"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, datasets):
        dd = exp1_summary[exp1_summary["dataset"] == ds].copy()
        dd = _label_method(dd)

        oracle = dd[dd["method"] == "non_private_dr"]
        if len(oracle) > 0:
            y_oracle = float(oracle["rmse"].mean())
            ax.axhline(y_oracle, linestyle="--", color="black", linewidth=2.2, label="Non-private DR")

        plot_df = dd[dd["method"].isin(methods)]
        sns.lineplot(
            data=plot_df,
            x="epsilon",
            y="rmse",
            hue="method_label",
            marker="o",
            linewidth=3.0,
            markersize=9,
            ax=ax,
        )

        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("ATE RMSE")

    _boost(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_bias(exp1_summary: pd.DataFrame, out_path: Path):
    if exp1_summary.empty:
        return

    datasets = ["ihdp", "twins", "acic_dgp7", "lalonde"]
    methods = ["mst_naive", "aim_naive", "causal_naive", "causal_na_mi", "causal_aim_na_mi", "non_private_dr"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, datasets):
        dd = exp1_summary[(exp1_summary["dataset"] == ds) & (exp1_summary["method"].isin(methods))].copy()
        dd = _label_method(dd)

        sns.lineplot(
            data=dd,
            x="epsilon",
            y="abs_bias",
            hue="method_label",
            marker="o",
            linewidth=3.0,
            markersize=9,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("Absolute ATE bias")

    _boost(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_coverage(exp2_summary: pd.DataFrame, out_path: Path):
    if exp2_summary.empty:
        return

    datasets = ["ihdp", "twins", "acic_dgp7", "lalonde"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, datasets):
        dd = exp2_summary[exp2_summary["dataset"] == ds].copy()
        dd = _label_method(dd)

        ax.axhspan(0.90, 1.00, color="gray", alpha=0.12)
        ax.axhline(0.95, linestyle="--", color="black", linewidth=2.2)

        sns.lineplot(
            data=dd,
            x="epsilon",
            y="coverage",
            hue="method_label",
            marker="o",
            linewidth=3.0,
            markersize=9,
            ax=ax,
        )

        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_ylim(0.0, 1.05)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("95% CI coverage")

    _boost(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_ci_length(exp2_summary: pd.DataFrame, out_path: Path):
    if exp2_summary.empty:
        return

    datasets = ["ihdp", "twins", "acic_dgp7", "lalonde"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, datasets):
        dd = exp2_summary[exp2_summary["dataset"] == ds].copy()
        dd = _label_method(dd)

        sns.lineplot(
            data=dd,
            x="epsilon",
            y="ci_length",
            hue="method_label",
            marker="o",
            linewidth=3.0,
            markersize=9,
            ax=ax,
        )

        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_yscale("log")
        ax.set_ylabel("CI length (log scale)")

    _boost(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_adaptive(exp3_summary: pd.DataFrame, exp3_selected: pd.DataFrame, out_path: Path):
    if exp3_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    left = _label_method(exp3_summary)
    left["label"] = left.apply(lambda r: f"{METHOD_LABELS.get(r['method'], r['method'])}, eps={r['epsilon']}", axis=1)
    sns.lineplot(data=left, x="K", y="rmse", hue="label", marker="o",
                 linewidth=3.0, markersize=9, ax=axes[0])
    axes[0].set_title("Adaptive vs fixed workload")
    axes[0].set_xlabel("Adaptive rounds K")
    axes[0].set_ylabel("ATE RMSE")

    if exp3_selected.empty:
        axes[1].text(0.5, 0.5, "No selection data", ha="center", va="center")
        axes[1].axis("off")
    else:
        top = exp3_selected.sort_values("count", ascending=False).head(10)
        sns.barplot(data=top, x="count", y="feature", color="#4c72b0", ax=axes[1])
        axes[1].set_title("Top selected features")
        axes[1].set_xlabel("Selection frequency")
        axes[1].set_ylabel("Feature")

    _boost(axes, title=19, label=18, tick=14)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_grid(exp4_summary: pd.DataFrame, metric: str, out_path: Path, title_prefix: str):
    if exp4_summary.empty:
        return

    methods = ["non_private_dr", "mst_naive", "aim_naive", "causal_naive", "causal_na_mi", "causal_aim_na_mi"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    if metric == "coverage":
        vmin, vmax = 0.5, 1.0
        cmap = "YlGnBu"
    else:
        vmin, vmax = None, None
        cmap = "viridis"

    for ax, method in zip(axes, methods):
        dd = exp4_summary[exp4_summary["method"] == method].copy()
        if dd.empty:
            ax.axis("off")
            continue

        piv = dd.pivot_table(index="n_target", columns="epsilon", values=metric, aggfunc="mean")
        sns.heatmap(
            piv,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 15},
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            ax=ax,
        )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("$\\epsilon$")
        ax.set_ylabel("n")
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=12)

    _boost(axes, title=17, label=16, tick=14)
    fig.suptitle(title_prefix, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig8_workload_dim(df: pd.DataFrame, out_path: Path, df_snr: pd.DataFrame | None = None):
    """Workload-dimension ablation; overlays the SNR-thresholded arm when provided."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["p"], df["rmse"], marker="o", linewidth=2.8, markersize=8,
            label="RMSE, no thresholding")
    if df_snr is not None and not df_snr.empty:
        ax.plot(df_snr["p"], df_snr["rmse"], marker="s", linewidth=2.8, markersize=8,
                label=r"RMSE, SNR thresholding ($\tau_{\mathrm{SNR}}=3$)")
    ax.plot(df["p"], np.sqrt(df["bias_component"]), marker="o", linestyle="--", alpha=0.65,
            linewidth=2.4, markersize=7, label="|bias|, no thresholding")
    if df_snr is not None and not df_snr.empty:
        ax.plot(df_snr["p"], np.sqrt(df_snr["bias_component"]), marker="s", linestyle="--",
                alpha=0.65, linewidth=2.4, markersize=7,
                label=r"|bias|, $\tau_{\mathrm{SNR}}=3$")
    ax.set_xlabel("Workload dimension p", fontsize=15)
    ax.set_ylabel("Error (original outcome units)", fontsize=15)
    ax.set_title("Workload size ablation (IHDP, $\\epsilon=1$)", fontsize=16)
    ax.tick_params(labelsize=12.5)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig9_mi_draws(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.lineplot(data=df, x="M", y="coverage", marker="o", linewidth=2.8, markersize=9, ax=ax)
    ax.axhline(0.95, linestyle="--", color="black", linewidth=2.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Number of MI draws M")
    ax.set_ylabel("95% CI coverage")
    ax.set_title("Coverage vs number of MI imputations")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig10_nsyn(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.lineplot(data=df, x="ratio", y="rmse", marker="o", linewidth=2.8, markersize=9, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Synthetic sample ratio $n_{syn}/n$")
    ax.set_ylabel("ATE RMSE")
    ax.set_title("Effect of synthetic sample size")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig11_overlap(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    d = _label_method(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    sns.lineplot(data=d, x="eta", y="rmse", hue="method_label", marker="o", linewidth=3.0, markersize=9, ax=axes[0])
    axes[0].set_title("RMSE vs overlap")
    axes[0].set_xlabel("Positivity constant $\\eta$")
    axes[0].set_ylabel("ATE RMSE")

    sns.lineplot(data=d, x="eta", y="coverage", hue="method_label", marker="o", linewidth=3.0, markersize=9, ax=axes[1])
    axes[1].axhline(0.95, linestyle="--", color="black", linewidth=2.0)
    axes[1].set_title("Coverage vs overlap")
    axes[1].set_xlabel("Positivity constant $\\eta$")
    axes[1].set_ylabel("95% CI coverage")
    axes[1].set_ylim(0.0, 1.05)

    _boost(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend([], [], frameon=False)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=17)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig12_fidelity(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return

    d = _label_method(df)
    # Outcome scales differ across datasets (LaLonde is in dollars), so plot RMSE
    # in standardized outcome units (the public constants used for DP measurement).
    _Y_SCALE = {"ihdp": 15.43, "twins": 1.0, "acic_dgp7": 5.706, "lalonde": 6624.0, "acs": 1.0}
    d["rmse_std"] = d.apply(lambda r: r["rmse"] / _Y_SCALE.get(r["dataset"], 1.0), axis=1)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.scatterplot(data=d, x="tvd", y="rmse_std", hue="method_label", style="epsilon", s=160, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Average marginal TVD (lower is better)", fontsize=15)
    ax.set_ylabel("ATE RMSE, standardized units (lower is better)", fontsize=15)
    ax.set_title("Fidelity vs causal utility tradeoff", fontsize=16)
    ax.tick_params(labelsize=12.5)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["Method" if l == "method_label" else (r"$\epsilon$" if l == "epsilon" else l) for l in labels]
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_figures(results_dir: Path, figures_dir: Path, snr_results_dir: Path | None = None):
    figures_dir.mkdir(parents=True, exist_ok=True)

    exp1_summary = _read_csv(results_dir / "exp1_summary.csv")
    exp2_summary = _read_csv(results_dir / "exp2_summary.csv")
    exp3_summary = _read_csv(results_dir / "exp3_summary.csv")
    exp3_selected = _read_csv(results_dir / "exp3_selected_features.csv")
    exp4_summary = _read_csv(results_dir / "exp4_summary.csv")

    ab1 = _read_csv(results_dir / "ablation_workload_dim.csv")
    ab2 = _read_csv(results_dir / "ablation_mi_draws.csv")
    ab3 = _read_csv(results_dir / "ablation_nsyn.csv")
    ab4 = _read_csv(results_dir / "ablation_overlap.csv")
    fid = _read_csv(results_dir / "fidelity_tradeoff.csv")

    plot_fig1_rmse(exp1_summary, figures_dir / "fig1_ate_rmse_comparison.pdf")
    plot_fig2_bias(exp1_summary, figures_dir / "fig2_ate_bias_comparison.pdf")
    plot_fig3_coverage(exp2_summary, figures_dir / "fig3_coverage_comparison.pdf")
    plot_fig4_ci_length(exp2_summary, figures_dir / "fig4_ci_length_comparison.pdf")
    plot_fig5_adaptive(exp3_summary, exp3_selected, figures_dir / "fig5_adaptive_vs_fixed.pdf")

    _plot_heatmap_grid(exp4_summary, metric="rmse", out_path=figures_dir / "fig6_acs_rmse_heatmap.pdf", title_prefix="ACS RMSE by n and epsilon")
    _plot_heatmap_grid(exp4_summary, metric="coverage", out_path=figures_dir / "fig7_acs_coverage.pdf", title_prefix="ACS coverage by n and epsilon")

    ab1_snr = _read_csv(snr_results_dir / "ablation_workload_dim.csv") if snr_results_dir else pd.DataFrame()
    plot_fig8_workload_dim(ab1, figures_dir / "fig8_workload_dimension.pdf", df_snr=ab1_snr)
    plot_fig9_mi_draws(ab2, figures_dir / "fig9_mi_draws.pdf")
    plot_fig10_nsyn(ab3, figures_dir / "fig10_nsyn_effect.pdf")
    plot_fig11_overlap(ab4, figures_dir / "fig11_overlap_effect.pdf")
    plot_fig12_fidelity(fid, figures_dir / "fig12_fidelity_vs_causal.pdf")

    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    plot_all_figures(results_dir=Path("results"), figures_dir=Path("../figures"))
