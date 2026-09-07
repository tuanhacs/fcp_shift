# Ablation Study Guide

This guide describes the six ablation workflows, their estimands, output files, and server commands.

## Installation and smoke validation

```bash
python -m pip install -e '.[dev]'
bash scripts/run_ablation_smoke_tests.sh
```

The smoke configurations use synthetic regression datasets and do not download OpenML data. Full configurations are stored in `configs/ablation/`.

Run every full ablation sequentially with:

```bash
bash scripts/run_ablation_studies.sh
```

On SLURM, submit one array containing all six workflows:

```bash
mkdir -p logs
sbatch scripts/slurm/ablation_array.sbatch
```

Every workflow supports standard CLI filtering where applicable:

```bash
python -m fcp_shift.cli run \
  --config configs/ablation/baselines.yaml \
  --dataset year \
  --weight exponential \
  --seed 31415
```

Use `--force` to replace a completed run. Otherwise a directory containing `DONE` is skipped.

## Plot again without rerunning experiments

The `plot` command reads the saved CSV/NPZ files and only regenerates PDFs. It does not load OpenML data, fit models, or repeat Monte Carlo simulations. For example:

```bash
python -m fcp_shift.cli plot \
  --config configs/ablation/corollary.yaml \
  --figsize 16 7 \
  --title-font-size 13 \
  --label-font-size 12 \
  --tick-font-size 11 \
  --legend-font-size 10
```

Replot all six ablations with the same style overrides:

```bash
bash scripts/plot_ablation_studies.sh \
  --font-size 11 \
  --figsize 16 7
```

The YAML and output root must match the completed experiment. The regenerated PDFs overwrite only the existing figure files; metrics and completion markers are left unchanged.

## Algorithm naming

The old notebook names the calibration-only estimator of `G` as `ghat_alg2`. In the current paper draft this estimator is Algorithm 1, while Algorithm 2 estimates the generalized inverse and Algorithm 3 selects the inverse conformal level. Metadata for the Corollary and delta ablations records both names:

```text
g_estimator: algorithm_1_in_paper
notebook_alias: ghat_alg2
```

This prevents notebook numbering from silently changing the implemented estimand.

## 1. Covariate-shift Corollary convergence

Configuration: `configs/ablation/corollary.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/corollary.yaml
```

The workflow uses two datasets, three alpha values, and three oracle-weight families. For each calibration size it estimates `G_{w,n}(alpha)` from weighted calibration scores. The resulting `2 x 3` figure contains a black horizontal alpha line and one line per weight.

Primary output:

```text
outputs/ablations/corollary/seed_<seed>/corollary_convergence_2x3.pdf
```

The expected population behavior is convergence toward alpha from below. Individual Monte Carlo repetitions can fluctuate around their population values, so the plot reports means with variability bands.

## 2. Delta-allocation ablation

Configuration: `configs/ablation/delta.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/delta.yaml
```

The experiment fixes `alpha=0.1`, uses exponential oracle weights, and compares manually specified confidence-budget allocations against the optimized allocation. It evaluates three paths:

- `n_increases`: `n` increases and `m` is fixed;
- `m_increases`: `m` increases and `n` is fixed;
- `n_equals_m`: both increase together.

Two `2 x 3` figures are produced:

```text
delta_fixed_2x3.pdf
delta_uniform_2x3.pdf
```

The vertical quantity is the calibration-only estimate of

```text
G(alpha + Delta) + epsilon.
```

The optimized allocation is always drawn in black. Fixed-alpha allocations specify the fraction assigned to calibration error. Uniform allocations specify the common fraction assigned to each of `delta_plus` and `delta_minus`; the remainder is assigned to test error.

## 3. Underlying-model ablation

Configuration: `configs/ablation/models.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/models.yaml
```

The dataset, train/source split, target distribution, weights, and Monte Carlo indices are fixed across models. The first model is used only to define the shared target weights. Every listed model then receives exactly the same shifted samples.

The runner supports:

- histogram gradient boosting;
- random forests;
- linear/ridge regression or logistic regression;
- multilayer perceptrons.

For each weight, it writes separate single-axis plots for Goals 1–4. This isolates model robustness from a change in the target distribution.

## 4. Inference-time ablation

Configuration: `configs/ablation/timing.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/timing.yaml
```

The output is a `3 x 3` dataset-by-model figure. Each subplot contains two timing curves:

- Goals 1–2: calibration score computation, calibration structure construction, constants, and forward `G` evaluation;
- Goals 3–4: calibration score computation, calibration structure construction, constants, and generalized-inverse evaluation.

Model fitting, dataset loading, and weight fitting are excluded. Each coordinate receives one untimed warm-up followed by repeated measurements. The plot reports medians and 10–90% bands on log-log axes.

## 5. Weight-family ablation

Configuration: `configs/ablation/weights.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/weights.yaml
```

The workflow compares five weight families in both Covariate Shift and Score-Transport Shift (STS):

- exponential;
- quadratic;
- Mahalanobis;
- positive linear (new);
- sigmoid (new).

For each setting it produces one forward figure for the fixed/uniform alpha guarantees and one inverse figure for the fixed/uniform beta guarantees. Each panel contains the usual black nominal line and one mean curve with a standard-deviation band for every weight. This gives exactly four figures:

```text
outputs/ablations/weight_families/seed_<seed>/
weights_covariate_forward_goals_1_2.pdf
weights_covariate_inverse_goals_3_4.pdf
weights_transport_forward_goals_1_2.pdf
weights_transport_inverse_goals_3_4.pdf
```

Each family defines its corresponding shifted target distribution; this fact is explicitly recorded in `metadata.json`. The STS experiment uses the configured `rho` and estimates the required transport map with Algorithm 1.

## 6. DKW and CoJER baseline comparison

Configuration: `configs/ablation/baselines.yaml`

```bash
python -m fcp_shift.cli run --config configs/ablation/baselines.yaml
```

This implementation follows the intended notebook experiment exactly:

1. Generate test data under covariate shift.
2. Ignore weights when constructing ordinary conformal p-values.
3. Compute the empirical FCP from those ordinary p-values.
4. Compare that ordinary empirical FCP with DKW and CoJER.

Weighted FCP is never used for DKW/CoJER pass rates or violations. This choice is recorded in `metadata.json` as:

```json
{
  "empirical_reference": "ordinary_unweighted_fcp_under_shift",
  "weighted_fcp_used_for_baseline_checks": false
}
```

Outputs include:

```text
baselines_forward_<dataset>.pdf
baselines_inverse_<dataset>.pdf
baseline_comparison_table.csv
baseline_curves_summary.csv
```

The full configuration covers six paper datasets and five weight families. The table reports forward and inverse pass rates and mean maximum violations for every dataset-weight pair. DKW/CoJER Monte Carlo calibration sizes are controlled in YAML. Increase them for final paper runs and keep the random seed fixed.

## Output and reproducibility

Each ablation directory contains:

```text
config.resolved.yaml
metadata.json
metrics.csv
summary.json
*.pdf
DONE
```

Some workflows also write aggregated CSV or NPZ files. The resolved YAML, package versions, seed, algorithm naming, and estimand definition are stored alongside the results.
