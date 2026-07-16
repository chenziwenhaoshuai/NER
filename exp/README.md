# Experiment scripts

The folder contains the scripts used for the paper's component ablations,
sensitivity analysis, and robustness controls.

## Fast reproduction from frozen paper artifacts

First download and verify the release artifacts:

```bash
python reproduce.py
```

Then run all artifact-based experiments:

```bash
python exp/run_all.py
```

The individual entry points are:

| Script | Output |
|---|---|
| `component_ablation.py` | Metrics for the baseline, temporal calibration, prior, three neural experts, neural mixture, and complete router |
| `operating_sensitivity.py` | Alert-budget and candidate-spacing sweeps |
| `router_sensitivity.py` | Legacy deterministic-router sensitivity reference |
| `center_to_interval.py` | Sparse rescued-center to finite-interval conversion |
| `random_insertion_guardrail.py` | Matched-budget random-insertion control |
| `evidence_stress.py` | Noise, drift, and missing-evidence robustness |

These scripts recompute metrics from the released binary predictions and
candidate scores. They do not retrain a detector or copy table values.

## Counterfactual-temperature MoE router

`run_counterfactual_temperature_moe.py` is the final MoE-router experiment. It
uses the frozen candidate pool and expert scores as deployment inputs, learns a
three-expert prior from label-free counterfactual probes, calibrates an expert
temperature from the same counterfactual diagnostics, and then materializes the
final candidate-ranking predictions.

This script requires PyTorch and the counterfactual probe directories generated
by the full experiment pipeline. Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

Run the MoE router against the compact release artifact schema:

```bash
python exp/run_counterfactual_temperature_moe.py \
  --profile_dir artifacts/v35 \
  --strong_probe_dir /path/to/counterfactual_prototype_all25 \
  --weak_probe_dir /path/to/counterfactual_weak_prototype_all25 \
  --output_dir results/moe_counterfactual_temperature \
  --device cuda
```

The same command also accepts the full paper-profile directory as
`--profile_dir`; if the directory contains `neural_router_v7_neural_dominant/`,
the script resolves that nested layout automatically. The output directory
contains checkpoints, candidate scores, final predictions, per-pair metrics,
temperature curves, and the selected temperature for each dataset.

After one or more MoE variants have been materialized, generate the comparison
tables with explicit method specs:

```bash
python exp/make_moe_ablation_tables.py \
  --method "Previous neural router|reference|/path/to/v7_run" \
  --method "Counterfactual-temperature MoE|learned prior + counterfactual temperature|results/moe_counterfactual_temperature" \
  --final-dir results/moe_counterfactual_temperature \
  --output-dir results/moe_ablation_tables
```

Each method directory must contain `summary/overall_metrics.csv`. The final
directory may additionally contain `summary/pair_metrics.csv`,
`summary/seed_summary.csv`, and `summary/temperature_selection.csv`; when
present, the table script exports dataset-average, seed-stability, and
temperature-selection tables.

## Full component-ablation training

`train_component_ablation.py` is the complete training/materialization entry
point. It trains, in order:

1. the self-trained event-prototype scorer;
2. the geometry convolutional autoencoder;
3. the score-augmented convolutional autoencoder;
4. the final router and every binary prediction used by the component table.

It requires the preprocessed benchmark data and the common 25-pair backbone
exports. The export directory must contain:

```text
rescue/
`-- {dataset}_{backbone}_seed2021_predictions.npz
```

Each NPZ must include the shared baseline prediction, temporal prediction,
candidate indices, event prior, detector train/test energies, selected centers,
and evaluation labels. Training and candidate ranking do not use the labels;
the scripts load them only to compute the reported metrics.

Example:

```bash
python exp/train_component_ablation.py \
  --exp-dir /path/to/common_backbone_exports \
  --data-root /path/to/anomaly_transformer_datasets \
  --output-dir results/full_component_ablation
```

Inspect the exact commands without starting training:

```bash
python exp/train_component_ablation.py \
  --exp-dir /path/to/common_backbone_exports \
  --data-root /path/to/anomaly_transformer_datasets \
  --dry-run
```

The full run preserves each expert's candidate scores, all final prediction
arrays, summaries, and `training_config.json`. It never overwrites the common
backbone exports or the frozen release artifacts.

## Full retraining ablations

The following scripts run the expensive ablations from the common backbone
exports and preprocessed datasets:

| Script | Experiment |
|---|---|
| `retrain_seed_ablation.py` | Retrain all three neural experts and the router for multiple random seeds |
| `retrain_spacing_ablation.py` | Regenerate radius-specific candidate pools, train the neural scorers, and evaluate the spacing sweep |

Three-seed component retraining:

```bash
python exp/retrain_seed_ablation.py \
  --exp-dir /path/to/common_backbone_exports \
  --data-root /path/to/anomaly_transformer_datasets \
  --seeds 2021 2022 2023
```

Candidate-spacing ablation on the representative paper subset:

```bash
python exp/retrain_spacing_ablation.py \
  --exp-dir /path/to/common_backbone_exports \
  --data-root /path/to/anomaly_transformer_datasets \
  --output-dir results/retrain_spacing_ablation
```

Both scripts preserve raw per-pair rows, candidate scores, configs, and
aggregate summaries. Use separate `--output-dir` values when launching seeds
or pair shards in parallel.
