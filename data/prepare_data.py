"""Reconstruct the processed benchmark files from their public raw sources.

Run each step from the repository root:

    python data/prepare_data.py ihdp     # -> data/raw/ihdp_npci_1-100.merged.npz
    python data/prepare_data.py twins    # -> data/raw/twins.csv
    python data/prepare_data.py lalonde  # -> data/raw/lalonde.csv
    python data/prepare_data.py all

ACIC 2016 needs R (see DATA.md) and ACS is fetched by folktables at run time, so
neither is handled here. After preparing the files, run data/verify_provenance.py.
"""
from __future__ import annotations

import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def prepare_ihdp() -> Path:
    """Merge the 672-unit train and 75-unit test releases into the full 747-unit sample."""
    base = "https://www.fredjo.com/files/"
    tr = np.load(_download(base + "ihdp_npci_1-100.train.npz", RAW / "ihdp_npci_1-100.train.npz"))
    te = np.load(_download(base + "ihdp_npci_1-100.test.npz", RAW / "ihdp_npci_1-100.test.npz"))
    keys = ["x", "t", "yf", "ycf", "mu0", "mu1"]
    merged = {k: np.concatenate([tr[k], te[k]], axis=0) for k in keys}
    out = RAW / "ihdp_npci_1-100.merged.npz"
    np.savez(out, **merged)
    print(f"wrote {out}  x={merged['x'].shape}")
    return out


def prepare_twins() -> Path:
    """One-year mortality from both twins, with a fixed-seed confounded treatment."""
    url = "https://raw.githubusercontent.com/jsyoon0823/GANITE/master/data/Twin_data.csv.gz"
    raw = pd.read_csv(_download(url, RAW / "Twin_data.csv.gz"))
    raw.columns = [c.strip().strip("'’") for c in raw.columns]

    covariates = raw.iloc[:, :30]
    y0 = (raw["outcome(t=0)"].to_numpy() < 9999).astype(int)  # lighter twin survives to 1yr?
    y1 = (raw["outcome(t=1)"].to_numpy() < 9999).astype(int)  # heavier twin

    # The distributed file has no treatment column; sample a covariate-dependent one
    # (GANITE's recipe) with a fixed seed so the file is exactly reproducible.
    rng = np.random.RandomState(0)
    x = covariates.to_numpy(dtype=float)
    coef = rng.uniform(-0.01, 0.01, size=x.shape[1])
    logits = x @ coef + rng.normal(0.0, 0.01, size=x.shape[0])
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.minimum(p / (2.0 * p.mean()), 1.0)
    t = rng.binomial(1, p)

    out_df = covariates.copy()
    out_df["t"] = t
    out_df["y0"] = y0
    out_df["y1"] = y1
    out = RAW / "twins.csv"
    out_df.to_csv(out, index=False)
    print(f"wrote {out}  n={len(out_df)}  P(t=1)={t.mean():.4f}  ATE={float((y1 - y0).mean()):.4f}")
    return out


_LALONDE_COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75", "re78"]


def prepare_lalonde() -> Path:
    """Dehejia-Wahba experimental sample (185 treated + 260 control)."""
    base = "https://users.nber.org/~rdehejia/data/"
    treated = np.loadtxt(_download(base + "nswre74_treated.txt", RAW / "nswre74_treated.txt"))
    control = np.loadtxt(_download(base + "nswre74_control.txt", RAW / "nswre74_control.txt"))
    df_t = pd.DataFrame(treated, columns=_LALONDE_COLS)
    df_t["sample"] = "nsw_treated"
    df_c = pd.DataFrame(control, columns=_LALONDE_COLS)
    df_c["sample"] = "nsw_control"
    out_df = pd.concat([df_t, df_c], ignore_index=True)
    out = RAW / "lalonde.csv"
    out_df.to_csv(out, index=False)
    print(f"wrote {out}  n={len(out_df)}  treated={int(out_df['treat'].sum())}")
    return out


STEPS = {"ihdp": prepare_ihdp, "twins": prepare_twins, "lalonde": prepare_lalonde}


def main(argv: list[str]) -> None:
    target = argv[1] if len(argv) > 1 else "all"
    steps = STEPS.values() if target == "all" else [STEPS[target]]
    for step in steps:
        step()


if __name__ == "__main__":
    main(sys.argv)
