"""Provenance verification harness for the real benchmark datasets.

Calls each loader exactly the way run_experiments.py does (via its
_load_dataset / _load_acs_dataset helpers) with use_real_if_available=True
and the acquired file paths, then prints metadata['source'], n, d, tau_true
per dataset and asserts that no loader fell back to 'synthetic_fallback'.

Run:
    PYTHONPATH=<code root> python3 data/raw/verify_provenance.py
(any cwd works; the script chdirs to the code root so the folktables
loader's cwd-relative 'data' root resolves to code/data/2018/...)
"""

from __future__ import annotations

import argparse
import os
import sys

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(CODE_ROOT, "data", "raw")

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
os.chdir(CODE_ROOT)  # folktables ACSDataSource uses cwd-relative root_dir="data"

import run_experiments as rx  # noqa: E402


def build_args() -> argparse.Namespace:
    """Namespace mirroring run_experiments.parse_args() defaults + real-data flags."""
    return argparse.Namespace(
        master_seed=42,
        y_clip=5.0,
        use_real_data=True,
        ihdp_npz_path=os.path.join(RAW_DIR, "ihdp_npci_1-100.merged.npz"),
        twins_csv_path=os.path.join(RAW_DIR, "twins.csv"),
        acic_dir=os.path.join(RAW_DIR, "acic"),
        lalonde_csv_path=os.path.join(RAW_DIR, "lalonde.csv"),
        acs_state="CA",
        acs_year=2018,
    )


def main() -> int:
    args = build_args()
    failures = []
    print(f"{'dataset':<12} {'source':<22} {'n':>7} {'d':>4} {'tau_true':>12}")

    for name in ["ihdp", "twins", "acic_dgp7", "lalonde"]:
        ds, wf, y = rx._load_dataset(name, args=args, rep=0, bins=5)
        src = ds.metadata.get("source", "?")
        print(f"{name:<12} {src:<22} {ds.n:>7} {ds.d:>4} {ds.tau_true:>12.4f}")
        if src == "synthetic_fallback":
            failures.append(name)

    ds, wf, y = rx._load_acs_dataset(args, rep=0, n=5000, bins=5)
    src = ds.metadata.get("source", "?")
    print(f"{'acs':<12} {src:<22} {ds.n:>7} {ds.d:>4} {ds.tau_true:>12.4f}")
    if src == "synthetic_fallback":
        failures.append("acs")

    assert not failures, f"synthetic_fallback detected for: {failures}"
    print("OK: no synthetic_fallback in any dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
