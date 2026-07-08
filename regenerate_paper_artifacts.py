#!/usr/bin/env python3
"""
Regenerate LaTeX-ready paper artifacts from the finalized CSVs.

This script is intended to be the single source of truth for any paper table
that depends on experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_ORDER = ["ihdp", "twins", "acic_dgp7", "lalonde"]
DATASET_LABELS = {"ihdp": "IHDP", "twins": "Twins", "acic_dgp7": "ACIC", "lalonde": "LaLonde"}

METHOD_ORDER = [
    "non_private_dr",
    "mst_naive",
    "aim_naive",
    "causal_naive",
    "causal_na_mi",
    "causal_aim_na_mi",
]
METHOD_LABELS = {
    "non_private_dr": "Non-private DR",
    "mst_naive": "MST + naive",
    "aim_naive": "AIM + naive",
    "causal_naive": "Causal + naive",
    "causal_na_mi": "Causal + NA+MI",
    "causal_aim_na_mi": "Causal-AIM + NA+MI",
}


def _fmt_cell(rmse: float, cov: float, precision: int) -> str:
    return f"{rmse:.{precision}f}/{cov:.{precision}f}"


def write_table_main_results(exp1_summary: pd.DataFrame, epsilon: float, out_path: Path, precision: int = 3) -> None:
    df = exp1_summary.copy()
    df = df[df["epsilon"] == float(epsilon)].copy()
    if df.empty:
        raise ValueError(f"No rows found for epsilon={epsilon}")

    needed = {"dataset", "epsilon", "method", "rmse", "coverage"}
    missing = needed.difference(set(df.columns))
    if missing:
        raise ValueError(f"exp1_summary is missing columns: {sorted(missing)}")

    df = df.set_index(["method", "dataset"])

    lines: list[str] = []
    lines.append("% AUTO-GENERATED. DO NOT EDIT BY HAND.")
    lines.append(f"% Source: exp1_summary.csv (epsilon={epsilon})")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in DATASET_ORDER) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for m in METHOD_ORDER:
        label = METHOD_LABELS.get(m, m)
        cells = []
        for d in DATASET_ORDER:
            try:
                row = df.loc[(m, d)]
            except KeyError as exc:
                raise KeyError(f"Missing (method={m}, dataset={d}) in exp1_summary") from exc
            cells.append(_fmt_cell(float(row["rmse"]), float(row["coverage"]), precision=precision))
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Paper 2 LaTeX artifacts from CSV results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing exp*_*.csv")
    parser.add_argument("--out-dir", type=str, default="../auto", help="Output directory for LaTeX artifacts")
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--epsilon-table1", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    exp1_path = results_dir / "exp1_summary.csv"
    exp1 = pd.read_csv(exp1_path)

    write_table_main_results(
        exp1_summary=exp1,
        epsilon=float(args.epsilon_table1),
        out_path=tables_dir / "table_main_results.tex",
        precision=int(args.precision),
    )
    print("Wrote:", tables_dir / "table_main_results.tex")


if __name__ == "__main__":
    main()

