# Neural Event Rescue

Official reproducibility repository for **Neural Event Rescue (NER)**, a
plug-in structured decision layer for multivariate time-series anomaly
detection.

NER takes the pointwise outputs of an already trained detector, identifies
event candidates from temporal evidence, and ranks a small candidate set using
three complementary label-free neural scorers:

- a self-trained event-prototype scorer;
- a geometry convolutional autoencoder;
- a score-augmented convolutional autoencoder.

The released evaluation covers five datasets and five detector backbones:

| Datasets | Backbones |
|---|---|
| SMD, MSL, SMAP, PSM, SWaT | Anomaly Transformer, Transformer, Autoformer, TimesNet, KANAD |

## Main reproduced result

The exact frozen-paper reproduction gives:

| Method | PA-F1 | Event F1 | Range F1 |
|---|---:|---:|---:|
| Baseline | 80.97 | 29.48 | 16.76 |
| NER | **83.33** | **32.69** | **17.53** |

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
├── pair_metrics.csv
├── overall_metrics.csv
├── dataset_metrics_fraction.csv
└── backbone_metrics_fraction.csv
```

The run fails loudly if PA-F1, Event F1, or Range F1 differs from the released
paper result by more than the configured tolerance.

For an offline machine, download
`ner_v7_reproduction_artifacts.zip` from Release `v1.0`, place it in
`artifacts/`, and run the same command.

## Reproduce all ablations and robustness experiments

The scripts used for paper ablations are in [`exp/`](exp):

| Script | Experiment |
|---|---|
| `component_ablation.py` | temporal calibration, event prior, each neural expert, fixed neural mixture, full router |
| `operating_sensitivity.py` | alert-budget and candidate-spacing sensitivity |
| `random_insertion_guardrail.py` | matched-budget random insertion control |
| `evidence_stress.py` | candidate-score noise, drift, and dropout stress |

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
```

Raw rows, summaries, and figures are written under `results/`. The scripts
never overwrite the frozen artifacts.

## Repository structure

```text
NER/
├── ner/                 # metrics, router, and artifact loader
├── src/                 # training/materialization reference implementation
├── exp/                 # paper ablation and sensitivity scripts
├── tests/               # reproducibility tests
├── reproduce.py         # one-command main-result reproduction
├── environment.yml
└── requirements.txt
```

## What is frozen and what is trained

There are two reproducibility levels:

1. **Exact paper-result reproduction.** `reproduce.py` starts from exported
   candidate scores and prediction arrays. This path is deterministic, fast,
   and reproduces the reported 5×5 table exactly.
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
ner_v7_reproduction_artifacts.zip
SHA256: e885f4087382257c9f6b7c7f66a8d40929e45d396039dac408d46b3a5b492f76
```

The extracted archive also contains `manifest.csv` with the SHA256 and byte
size of every prediction and candidate-score file.

## Citation

The paper is under review. A BibTeX entry will be added after publication.

## License

Code is released under the MIT License. Dataset licenses remain with their
original providers.

