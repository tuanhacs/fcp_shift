# FCP under Distribution Shift — Experiment Suite

This repository contains reproducible implementations of the three main experiment families for the paper **Conformal Prediction under Distribution Shift: False Coverage Proportion Guarantees**:

1. Goals 1–4 under covariate shift.
2. Goals 1–4 under stratified score-transport shift.
3. The asymptotic study in the calibration size `n`, test size `m`, and joint size `n=m`.

The code is a standalone Python package. It does not depend on notebook state. Every run stores its resolved configuration, environment metadata, raw results, summaries, figures, and a completion marker.

The seven ablation workflows—Corollary convergence, delta allocation, underlying models, inference time, weight families, DKW/CoJER baselines, and efficiency versus FCP bound—are documented in [docs/ABLATIONS.md](docs/ABLATIONS.md).

## Project layout

```text
configs/main/       Full experiment YAML files
configs/smoke/      Small end-to-end configurations
scripts/            Local/server shell entry points
scripts/slurm/      SLURM array jobs
src/fcp_shift/      Reusable experiment package
tests/              Unit tests for CP, bounds, algorithms, and shifts
outputs/            Generated results (ignored by Git)
```

## Installation

Python 3.10 or newer is required. On a Linux server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

OpenML datasets are downloaded on first use and then served from the local scikit-learn/OpenML cache. Compute nodes without internet access should be given a pre-populated cache.

## Quick validation

The smoke configurations use local synthetic regression data and require no OpenML download:

```bash
bash scripts/run_smoke_tests.sh
```

Or run them separately:

```bash
python -m fcp_shift.cli run --config configs/smoke/covariate_shift_smoke.yaml
python -m fcp_shift.cli run --config configs/smoke/transport_shift_smoke.yaml
python -m fcp_shift.cli run --config configs/smoke/asymptotic_smoke.yaml
```

## Main experiments

### Covariate shift

```bash
bash scripts/run_covariate_shift.sh
```

Under oracle covariate shift, Goals 1–4 use the identity upper bound `G(eta) <= eta`. The implementation does not re-estimate `G` in this experiment.

The exponential tilt is fitted on an independent auxiliary subset (`shift.auxiliary_fraction`, default `0.2`) rather than on calibration/test scores. The resulting feature-only weight is clipped and normalized on the source pool; test pairs are resampled in proportion to this weight. Rerun older output directories with `--force` after this change, since saved curves use the previous shift construction.

### Stratified score-transport shift

```bash
bash scripts/run_transport_shift.sh
```

The transport runner fits an auxiliary Ridge projection of score on features, freezes quantile cutpoints from the auxiliary features, and defines strata as `g(X)` on the source pool. It draws a target covariate stratum using the exponential-tilt stratum masses and, with probability `rho`, replaces the selected point's score with a score drawn from the next stratum. At `rho=0`, this is covariate shift with a *stratum-coarsened* exponential weight, not the pointwise weight of the separate covariate runner. The resulting score-transport weight is `((1-rho)*q[h] + rho*q[pi_inverse(h)])/p[h]`; Algorithm 1 estimates `G`. The experiment evaluates score-level FCP and does not synthesize new target labels `Y` for transported scores. Older outputs require `--force` to be recomputed.

### Asymptotic study

```bash
bash scripts/run_asymptotic.sh
```

The asymptotic experiment uses simulated heteroscedastic regression data and exponential weights. It creates one publication-ready 1x3 figure, `asymptotic_1x3.pdf`, plus a 400-DPI PNG copy. The panels respectively show increasing `m` with fixed `n`, increasing `n` with fixed `m`, and jointly increasing `n=m`.

Every panel contains the mean calculated quantities for Goals 1–4, together with horizontal reference lines for the configured `alpha` and `beta`. Values are **not clipped at one**, the panels share the same vertical scale, and the vertical axis is a nonnegative real-valued calculated quantity. Use the vector PDF when including the figure in a paper.

Run all main experiments sequentially with:

```bash
bash scripts/run_all_main.sh
```

## Ablation studies

Validate all ablation pipelines on local synthetic data:

```bash
bash scripts/run_ablation_smoke_tests.sh
```

Run the full YAML configurations:

```bash
bash scripts/run_ablation_studies.sh
```

Submit the six workflows as a SLURM array:

```bash
sbatch scripts/slurm/ablation_array.sbatch
```

See [the ablation guide](docs/ABLATIONS.md) for estimands, figure layouts, algorithm-name mapping, and output tables. In particular, the DKW/CoJER workflow compares the baselines against empirical FCP computed from ordinary unweighted conformal p-values under shift.

## Grouped weight figures

After the covariate and transport jobs finish, combine all available weights into shared figures:

```bash
bash scripts/make_main_figures.sh
```

For each covariate-shift dataset, the command creates a separate single-axis plot for each Goal. Every available weight—exponential, quadratic, and Mahalanobis—is drawn on the same axes with consistent colors. The plots do not use multi-panel subfigures.

For transport shift, the same three grouped figures are created separately for every `(dataset, rho)` pair. The aggregation reads `curves.npz` files from all completed seed directories, so it works after weights have been run as independent SLURM jobs.

Grouped figures are written below:

```text
outputs/main_figures/covariate_shift/<dataset>/
outputs/main_figures/transport_shift/<dataset>/rho_<value>/
```

### Combined Covariate/Transport paper figure

After running the same weight on multiple datasets in both shift settings, create one combined figure with:

- row 1: Covariate Shift;
- row 2: Transport Shift at one selected `rho`;
- first `D` columns: Goals 1–2 together, one plot per dataset;
- final `D` columns: Goals 3–4 together, one plot per dataset.

For two datasets this produces the requested `2 x 4` layout:

```bash
bash scripts/make_combined_main_figure.sh exponential 0.50 year adult
```

The equivalent Python command is:

```bash
python -m fcp_shift.cli main-figure \
  --covariate-config configs/main/covariate_shift.yaml \
  --transport-config configs/main/transport_shift.yaml \
  --weight exponential \
  --rho 0.50 \
  --datasets year adult
```

If `--datasets` is omitted, the command automatically includes every configured dataset for which both result directories exist. The output is saved as both PDF and PNG below `outputs/main_figures/combined/`.

### Plot styling from the CLI

All commands that generate plots (`run`, `figures`, and `main-figure`) accept the same presentation options:

```text
--figsize WIDTH HEIGHT
--font-size SIZE
--title-font-size SIZE
--label-font-size SIZE
--tick-font-size SIZE
--legend-font-size SIZE
```

Sizes are given in inches for `--figsize` and points for font options. `--font-size` is the fallback for every text element; a specialized option overrides it. For example:

```bash
python -m fcp_shift.cli main-figure \
  --covariate-config configs/main/covariate_shift.yaml \
  --transport-config configs/main/transport_shift.yaml \
  --weight exponential \
  --rho 0.50 \
  --datasets adult fashion_mnist \
  --figsize 14 5 \
  --font-size 10 \
  --title-font-size 12 \
  --legend-font-size 8
```

For an asymptotic or ablation report generated through `run`, append the same options:

```bash
python -m fcp_shift.cli run \
  --config configs/main/asymptotic.yaml \
  --figsize 17 4.8 \
  --label-font-size 11 \
  --tick-font-size 10 \
  --legend-font-size 9
```

Completed main asymptotic runs redraw from `summary.csv`. To redraw an ablation without rerunning it, use the saved results with `plot`:

```bash
python -m fcp_shift.cli plot \
  --config configs/ablation/corollary.yaml \
  --figsize 16 7 \
  --title-font-size 13 \
  --label-font-size 12 \
  --tick-font-size 11 \
  --legend-font-size 10
```

Use `bash scripts/plot_ablation_studies.sh` to redraw all seven ablations. The dedicated `figures`, `main-figure`, and ablation `plot` commands only read saved results; they do not repeat model fitting or Monte Carlo experiments.

## Choosing the weight direction

Each weight can use either a score-informed Ridge direction (`ridge`, the default) or a seeded random direction (`random`) to form its scalar projection of the features. The weight formula, clipping, and mean-one normalization are unchanged. Set `direction` and `direction_seed` on a YAML weight entry:

```yaml
weights:
  - {name: exponential, strength: 0.35, clip_quantile: 0.995, direction: random, direction_seed: 2026}
```

Or override every configured weight from the CLI without editing YAML:

```bash
python -m fcp_shift.cli run --config configs/main/covariate_shift.yaml --weight-direction random --direction-seed 2026
python -m fcp_shift.cli run --config configs/main/transport_shift.yaml --weight-direction random --direction-seed 2026
python -m fcp_shift.cli main-figure --weight exponential --rho 0.5 --weight-direction random --direction-seed 2026
```

Use the same direction options with `figures` or ablation `plot` to read the corresponding saved results. Random-direction results are stored in `direction_random_seed_<seed>` subdirectories, so they do not overwrite existing Ridge runs. Ridge continues to use the original output paths. On the main shift experiments, the random direction is independent of scores; auxiliary features are used only to center/scale the projection and define STS strata. Older ablations that do not have an independent auxiliary split also support `random`, with source features used only for projection scaling.

## Selecting a dataset, weight, shift, or seed

The CLI can filter a multi-dataset YAML without editing it:

```bash
python -m fcp_shift.cli run \
  --config configs/main/covariate_shift.yaml \
  --dataset year \
  --weight exponential \
  --seed 31415
```

For transport shift:

```bash
python -m fcp_shift.cli run \
  --config configs/main/transport_shift.yaml \
  --dataset fashion_mnist \
  --weight quadratic \
  --rho 0.75
```

Use `--force` to rerun a completed task. Without it, a task containing a `DONE` marker is skipped.

## YAML configuration

The main configuration controls:

- the dataset registry and OpenML IDs;
- classification or regression;
- predictive-model hyperparameters;
- weight families and their clipping/strength parameters;
- calibration and test sizes;
- repetitions and random seeds;
- FCP confidence level, alpha grid, and beta grid;
- transport strata and shift strengths;
- asymptotic grids and fixed sample sizes;
- the output root.

The repository preconfigures the eight datasets whose IDs were present in the original notebook notes: Adult, Bank Marketing, Electricity, MiniBooNE, Fashion-MNIST, Year Prediction, Diamonds, and Allstate. Additional datasets can be added as YAML entries without code changes, for example:

```yaml
- name: my_dataset
  source: openml
  openml_id: 12345
  task: regression
  train_fraction: 0.4
```

CSV input is also supported with `source: csv`, `path`, and `target_column`.

## Output structure

A covariate-shift task is stored as:

```text
outputs/covariate_shift/<dataset>/<weight>/seed_<seed>/
├── config.resolved.yaml
├── metadata.json
├── metrics.csv
├── summary.json
├── curves.npz
├── forward_goals_1_2.pdf
├── inverse_goals_3_4.pdf
├── selected_alpha_goals_3_4.pdf
└── DONE
```

Transport outputs additionally include `rho_<value>` in the path. Asymptotic outputs contain `metrics.csv`, `summary.csv`, and the three requested figures.

## SLURM

Create the environment once on the login node, activate it, and submit:

```bash
mkdir -p logs
sbatch scripts/slurm/covariate_array.sbatch
sbatch scripts/slurm/transport_array.sbatch
sbatch scripts/slurm/asymptotic_array.sbatch
```

The covariate array splits jobs by `dataset × weight`. The transport array splits jobs by `dataset × weight × rho`. Each task writes to a unique directory, so jobs do not overwrite one another.

## Reproducibility notes

- Model seeds, experiment seeds, and repetition seeds are deterministic.
- Random seeds are derived from stable hashes of the experiment coordinates.
- Resolved YAML and package versions are saved with every task.
- The `DONE` marker is written only after metrics, arrays, summaries, and figures have been saved.
- The population weights are normalized to empirical mean one on the finite source population. Their empirical support maximum is used as the bounded-weight constant.
