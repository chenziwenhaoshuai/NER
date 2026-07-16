# Reproduction artifact schema

The `v1.0` release contains `ner_v7_reproduction_artifacts.zip`.

After extraction:

```text
v7/
- predictions/
  - {dataset}_{backbone}_seed2021_predictions.npz
- candidate_scores/
  - {dataset}_{backbone}_candidate_scores.npz
- reference/
  - config.json
  - neural_router_v7_pair_metrics.csv
  - neural_router_v7_overall_metrics.csv
  - neural_router_v7_routes.csv
- manifest.csv
```

Prediction files contain the ground truth and the binary output of each
ablation variant. Candidate-score files contain the shared candidate indices,
three frozen neural expert scores, router score, budget, and route.

Ground-truth labels are loaded only by evaluation scripts. Router scores and
predictions are materialized without reading anomaly labels.

The counterfactual-temperature MoE experiment in `exp/` can reuse this compact
artifact package for the frozen deployment candidate pool. Its MoE-prior
training step additionally requires the generated counterfactual probe
directories, which are intermediate experiment artifacts rather than original
sensor data and are not bundled into `v1.0`.

