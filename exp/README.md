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
| `random_insertion_guardrail.py` | Matched-budget random-insertion control |
| `evidence_stress.py` | Noise, drift, and missing-evidence robustness |

These scripts recompute metrics from the released binary predictions and
candidate scores. They do not retrain a detector or copy table values.

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
and evaluation labels. Labels are read only after prediction materialization
to compute reported metrics.

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
