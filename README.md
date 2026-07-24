# Neural Event Rescue

Official reproducibility repository for **Neural Event Rescue (NER)**, a
plug-in structured decision layer for multivariate time-series anomaly
detection.

NER takes the pointwise outputs of an already trained detector, identifies
event candidates from temporal evidence, and ranks a small candidate set using
three complementary label-free neural experts:

- a self-trained event-prototype scorer;
- a geometry convolutional autoencoder;
- a score-augmented convolutional autoencoder.

The current experimental branch also includes a counterfactual-temperature
mixture-of-experts router that learns dataset-level expert priors from
label-free counterfactual event-over-normal probes and then calibrates the
expert mixture temperature before candidate ranking.

The released evaluation covers five datasets and five detector backbones:

| Datasets | Backbones |
|---|---|
| SMD, MSL, SMAP, PSM, SWaT | Anomaly Transformer, Transformer, Autoformer, TimesNet, KANAD |

## Main reproduced result

The exact frozen-paper reproduction gives:

| Method | PA-F1 | Event F1 | Range F1 |
|---|---:|---:|---:|
| Baseline | 80.97 | 29.48 | 16.76 |
| NER | **83.34** | **32.69** | **17.54** |

The script recomputes every metric from the 25 saved prediction arrays. It
does not copy summary numbers into the output.

## Quick reproduction

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/chenziwenhaoshuai/NER.git
cd NER
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and reproduce the main table:

```bash
pip install -r requirements.txt
python reproduce.py
```

On the first run, `reproduce.py` downloads the 29.8 MB frozen artifact package
from the GitHub Release, verifies its SHA256, recomputes all metrics, and
writes:

```text
results/main/
- pair_metrics.csv
- overall_metrics.csv
- dataset_metrics_fraction.csv
- backbone_metrics_fraction.csv
```

The run fails loudly if PA-F1, Event F1, or Range F1 differs from the released
paper result by more than the configured tolerance.

For an offline machine, download
`ner_v35_moe_reproduction_artifacts.zip` from Release `v1.1`, place it in
`artifacts/`, and run the same command.

## Reproduce all ablations and robustness experiments

The scripts used for paper ablations are in [`exp/`](exp):

| Script | Experiment |
|---|---|
| `component_ablation.py` | temporal calibration, event prior, each neural expert, fixed neural mixture, full router |
| `operating_sensitivity.py` | alert-budget and candidate-spacing sensitivity |
| `router_sensitivity.py` | legacy deterministic-router sensitivity reference |
| `center_to_interval.py` | rescued-center duration and temporal-coverage analysis |
| `random_insertion_guardrail.py` | matched-budget random insertion control |
| `evidence_stress.py` | candidate-score noise, drift, and dropout stress against a matched-budget prior-only control |
| `run_counterfactual_temperature_moe.py` | final counterfactual-temperature MoE router experiment |
| `make_moe_ablation_tables.py` | MoE comparison, dataset-average, seed-stability, and temperature-selection tables |
| `train_component_ablation.py` | full training and materialization of all neural branches used in the component ablation |
| `retrain_seed_ablation.py` | complete multi-seed neural-module ablation |
| `retrain_spacing_ablation.py` | regenerated candidate-spacing ablation with trained scorers |
| `make_paper_artifacts.py` | paper-ready LaTeX tables and quantitative figures from reproduced results |

Run all released experiments:

```bash
python exp/run_all.py
```

Or run one experiment:

```bash
python exp/component_ablation.py
python exp/operating_sensitivity.py
python exp/random_insertion_guardrail.py
python exp/evidence_stress.py
python exp/make_paper_artifacts.py
```

Raw rows, summaries, generated LaTeX tables, and quantitative figures are
written under `results/`. The scripts never overwrite the frozen artifacts.
`make_paper_artifacts.py` expects the main and ablation result CSVs produced by
`reproduce.py` and `exp/run_all.py`; it writes paper-ready assets to
`results/paper_artifacts/`.

The commands above reproduce the artifact-based paper experiments from the
frozen release artifacts. To retrain all three neural branches or run the
counterfactual-temperature MoE router from intermediate probes, install
`requirements-train.txt` and follow [`exp/README.md`](exp/README.md). That path
requires the preprocessed datasets, common backbone-score exports, and
counterfactual probe files because those large licensed or intermediate inputs
are not redistributed.

## Repository structure

```text
NER/
- ner/                 # metrics, router, and artifact loader
- src/                 # training/materialization reference implementation
- exp/                 # paper ablation and sensitivity scripts
- tests/               # reproducibility tests
- reproduce.py         # one-command main-result reproduction
- environment.yml
- requirements.txt
```

## What is frozen and what is trained

There are two reproducibility levels:

1. **Exact paper-result reproduction.** `reproduce.py` starts from exported
   candidate scores and prediction arrays. This path is deterministic, fast,
   and reproduces the reported 5x5 table exactly.
2. **Method reference implementation.** `src/` contains the scripts used to
   construct the prototype scorer, geometry AE, score-augmented AE, and final
   neural router. These scripts require the benchmark datasets and backbone
   score exports. Raw datasets are not redistributed because their original
   licenses and access conditions differ.

The exact reproduction package includes:

- ground-truth test labels used only for evaluation;
- baseline and temporally calibrated predictions;
- frozen candidate scores from all three neural experts;
- final NER predictions and selected event centers;
- a per-file SHA256 manifest.

It excludes original sensor data and multi-gigabyte intermediate energy arrays.

## Using NER with a new detector

NER expects:

- a training anomaly-score sequence;
- a test anomaly-score sequence;
- the baseline binary prediction;
- synchronized multivariate training and test values.

The training and scoring reference entry points are in:

- `src/run_neural_event_rescue_any_baseline.py`
- `src/experiment_self_trained_event_ranker.py`
- `src/experiment_convae_candidate_scorer.py`
- `src/experiment_augmented_convae_candidate_scorer.py`
- `src/materialize_neural_router_v7.py`
- `exp/run_counterfactual_temperature_moe.py`

The final router acts only on the candidate pool; it does not retrain or modify
the detector backbone.

## Evaluation protocol

The repository reports:

- point-adjusted precision, recall, and F1;
- unadjusted point F1;
- event-overlap F1;
- one-to-one strict event F1 and false events per 100K observations;
- range-overlap precision, recall, and F1;
- pointwise false-positive rate.

All metrics are implemented in [`ner/metrics.py`](ner/metrics.py) and are
computed directly from binary predictions.

## Artifact integrity

Release asset:

```text
ner_v35_moe_reproduction_artifacts.zip
SHA256: 1229e8b1af10fb97ea67141d06d46e6d58e6a3277cb80036ea7de865b640d10d
```

The extracted archive also contains `manifest.csv` with the SHA256 and byte
size of every prediction and candidate-score file.

## Citation

The paper is under review. A BibTeX entry will be added after publication.

## License

Code is released under the MIT License. Dataset licenses remain with their
original providers.

