# Causal-AIM: Workload-Preserving DP Synthetic Data for Causal Inference

Code for **"Workload-Preserving Differentially Private Synthetic Data for Causal Inference via Maximum-Entropy Calibration"** (UAI 2026).

The pipeline measures a *causal workload* (orthogonal-score moments) under differential privacy, reconstructs a maximum-entropy synthetic distribution, and performs noise-aware multiple-imputation (NA+MI) inference. Causal-AIM selects the workload adaptively.

## Quick start

The `notebooks/` directory reproduces the paper end to end, from data acquisition to each
table and figure. Run them in order:

1. `notebooks/00_setup_and_data.ipynb` — install, download and assemble the benchmarks, verify.
2. `notebooks/01_main_table_and_figures.ipynb` — Table 1 and the main-text figures.
3. `notebooks/02_ablations.ipynb` — the ablation and fidelity figures.
4. `notebooks/03_appendix_experiments.ipynb` — the supplementary studies and appendix tables.

The sections below cover the same steps from the command line.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+. The ACIC 2016 benchmark additionally requires R with the `aciccomp2016` package (see DATA.md).

## Data

No datasets ship with this repository. `DATA.md` documents the public source and layout for each benchmark (IHDP, Twins, ACIC 2016, LaLonde/NSW, ACS via folktables). `data/prepare_data.py` downloads and assembles IHDP, Twins, and LaLonde automatically (ACIC needs R; ACS is fetched by folktables):

```bash
python data/prepare_data.py all
python data/verify_provenance.py   # confirms each dataset loads as real, not a surrogate
```

## Reproducing the paper

Main experiments (Table 1, Figures 2-4; ~2 h on 20 cores):

```bash
python run_experiments.py --all \
  --replications-adaptive 100 --replications-acs 100 --replications-ablation 200 \
  --use-real-data \
  --ihdp-npz-path data/raw/ihdp_npci_1-100.merged.npz \
  --twins-csv-path data/raw/twins.csv \
  --acic-dir data/raw/acic \
  --lalonde-csv-path data/raw/lalonde.csv \
  --results-dir results_main --n-jobs 20
```

Supplementary studies (multi-estimand reuse, direct-DP proxies, hybrid workload, K-sweep, scalability, ridge):

```bash
python run_promoted_experiments.py --results-dir results_promoted --tasks all \
  --n-rep 200 --n-rep-scalability 100 --n-jobs 20 --use-real-data \
  --ihdp-npz-path data/raw/ihdp_npci_1-100.merged.npz \
  --twins-csv-path data/raw/twins.csv --acic-dir data/raw/acic \
  --lalonde-csv-path data/raw/lalonde.csv
```

Tables and figures:

```bash
python generate_paper_artifacts.py --results-dir results_main --out-dir auto
python -c "from pathlib import Path; import plot_figures as p; p.plot_all_figures(Path('results_main'), Path('figures'))"
```

Every run writes `run_manifest.json` (full configuration, per-file code hashes, per-dataset data source) into its results directory.

## Paper-to-code map

| Paper component | Code |
|---|---|
| Causal workload construction (Section 4.1) | `workloads.py` |
| DP measurement, Gaussian mechanism (Section 4.2) | `mechanisms.py` |
| Max-entropy calibration / synthesis (Algorithm 2) | `maxent.py` |
| Causal-AIM adaptive selection (Algorithm 1) | `causal_aim.py` |
| NA+MI inference (Algorithm 3) | `na_mi.py` |
| DR/IPW estimators, propensity clipping | `estimators.py` |
| Generic baselines (MST-style, AIM-style, direct-DP proxies) | `baselines.py` |
| Benchmark loaders + outcome standardization | `data/`, `run_experiments.py` |
| All figures (incl. two-arm workload-dimension plot) | `plot_figures.py` |

## Citation

```bibtex
@inproceedings{asiaee2026workload,
  title     = {Workload-Preserving Differentially Private Synthetic Data for Causal Inference via Maximum-Entropy Calibration},
  author    = {Asiaee, Amir and Aryan, Kaveh},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026},
  publisher = {PMLR}
}
```

## License

MIT (see LICENSE).
