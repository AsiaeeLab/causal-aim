# Reproduction notebooks

Run these in order from inside `notebooks/`. Each one says what it produces and roughly
how long it takes.

| Notebook | Produces |
|---|---|
| `00_setup_and_data.ipynb` | Installs dependencies, downloads and assembles all five benchmarks, verifies each loads as real data. Run once. |
| `01_main_table_and_figures.ipynb` | Table 1 and the main-text figures (RMSE, coverage, CI length, bias, adaptive, ACS). ~2 h on 20 cores. |
| `02_ablations.ipynb` | The five ablation and fidelity figures, including the two-arm workload-dimension figure. |
| `03_appendix_experiments.ipynb` | The six supplementary studies and the appendix tables (multi-estimand, direct-DP, hybrid, K-sweep, scalability, ridge). |

Every run writes a `run_manifest.json` next to its results recording the exact
configuration, per-file code hashes, and data sources, so a result can always be traced
back to what produced it.

To smoke-test the pipeline before committing to the full run, lower the replication
flags (`--replications-main 5`, `--n-rep 5`) and set `--n-jobs` to your core count.
