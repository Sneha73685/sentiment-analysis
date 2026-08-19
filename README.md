# sentiment-analysis

A text-classification research/engineering project covering sentiment analysis and
depression-oriented text classification, using both classical (TF-IDF + Logistic Regression)
and DistilBERT-based models.

> **IMPORTANT:** This repository contains a research/engineering prototype, not a diagnostic or
> clinical system. Its predictions are not medical conclusions, and its confidence values are
> not probabilities of a person's true mental-health state. See
> [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) before interpreting anything in this repository.

## Overview

This repository began as a combined sentiment-analysis and depression-oriented text
classification project, using both classical TF-IDF + Logistic Regression models and
fine-tuned DistilBERT models. It also includes supporting components — keyword-based signal
detection, trajectory analysis across a sequence of texts, and a Monte Carlo / perturbation
sensitivity experiment — built on top of the core classifiers.

The depression-classification workflow was later found to have a data-leakage problem in its
original evaluation, and was corrected by constructing a deduplicated, stratified, seeded
train/validation/test split with a frozen held-out test set. This corrected generation is
referred to throughout the repository as **v2**, and is the **canonical research workflow** for
the depression classifiers. The original implementation and its leakage-affected evaluation are
preserved as **v1**, for auditability, not as a valid result.

The repository contains, for the canonical v2 depression workflow: a leakage-free data split,
retrained classical and BERT models, an official held-out evaluation, a calibration analysis,
an error analysis, and a five-seed training-stability analysis.

Full leakage-correction narrative: [`docs/data.md`](docs/data.md).

## Current Status

| Component | Status |
| --- | --- |
| Canonical depression split | Complete |
| Canonical v2 classical model | Complete |
| Canonical v2 BERT model | Complete |
| Held-out test evaluation | Complete |
| Calibration analysis | Complete |
| Error analysis | Complete |
| Multi-seed stability analysis | Complete |
| Documentation | Complete |
| Runtime migration from v1 to v2 (depression) | Complete — sentiment remains v1 (out of scope) |
| Automated test suite | Minimal (`tests/`, synthetic fixtures — see [`docs/development.md`](docs/development.md) §15) |
| CI | Not yet implemented |

This reflects the state of the **research artifacts and documentation**, not a claim that the
whole software project is a polished, deployable application — see
[Known Limitations](#known-limitations) and [`docs/development.md`](docs/development.md).

## Canonical Results

The following are the official held-out test results for the canonical seed-42 v2 models.

Test set: **n = 1148**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9556 | 0.9777 | 0.9309 | 0.9537 | 0.9900 |
| DistilBERT | 0.9843 | 0.9964 | 0.9716 | 0.9838 | 0.9976 |

Source: [`results/depression_v2_model_comparison.csv`](results/depression_v2_model_comparison.csv)

Full methodology and per-class breakdown: [`MODEL_CARD.md`](MODEL_CARD.md),
[`docs/evaluation.md`](docs/evaluation.md).

## Stability Evidence

The canonical result above is a single training run, seed 42. A separate five-seed experiment
(seeds 42, 123, 2024, 3407, 7777) retrained both models from scratch to check whether that
result is stable across training seeds.

| Metric | BERT mean ± SD |
| --- | --- |
| Accuracy | 0.9829 ± 0.0047 |
| F1 | 0.9825 ± 0.0048 |
| ROC-AUC | 0.9975 ± 0.0004 |
| ECE | 0.0143 ± 0.0018 |
| Brier | 0.0150 ± 0.0026 |
| Log-loss | 0.0768 ± 0.0074 |

BERT exceeded the classical model's fixed 0.9556 accuracy at all five tested seeds. This
supports the stability of BERT's advantage **on this specific benchmark and split** — it is not
evidence of universal generalization beyond this dataset. Full detail:
[`docs/evaluation.md`](docs/evaluation.md), [`MODEL_CARD.md`](MODEL_CARD.md).

## Important Finding: Calibration

BERT showed substantially better aggregate calibration than the classical model on this fixed
benchmark (lower Expected Calibration Error). However, **good aggregate calibration does not
mean every individual prediction is trustworthy** — BERT remained highly confident on some of
its incorrect predictions.

Model confidence should therefore be interpreted as **model-reported confidence, not clinical
probability**.

Full analysis: [`docs/evaluation.md`](docs/evaluation.md), [`MODEL_CARD.md`](MODEL_CARD.md).
Appropriate-use implications: [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## Safety and Responsible Use

> **IMPORTANT:** This repository contains a research/engineering prototype, not a diagnostic or
> clinical system. Its predictions are not medical conclusions, and its confidence values are
> not probabilities of a person's true mental-health state.

- The depression classifier's labels come from Reddit source context, **not clinician
  diagnoses** — label noise is possible.
- `compute_risk()`, in `src/mental_health_pipeline.py`, is a small, **unvalidated heuristic**
  combining sentiment and depression outputs — it is not a validated clinical risk score.
- Trajectory analysis (`src/mental_health_trajectory.py`) layers further heuristic logic on top
  of that heuristic.
- Monte Carlo / perturbation analysis (`src/monte_carlo_simulation.py`) is an exploratory
  sensitivity experiment, **not a certified robustness guarantee**.

Full policy: [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## V1 and V2

- **V1** — the historical/original implementation: ad hoc train/test handling and a depression
  BERT model evaluated against the full source dataset, including rows it had already been
  trained on.
- **V2** — the canonical, leakage-free depression research workflow: a deduplicated, stratified,
  seeded split with a frozen test set, retrained models, and a rigorous evaluation.

The historical v1 evaluation was found to have train/test overlap and is therefore **not a
valid estimate of held-out performance**. Its historical leakage-affected evaluation result,
**0.9933**, must not be cited as valid test performance; it is preserved for audit purposes
under [`results/legacy_full_dataset_eval/`](results/legacy_full_dataset_eval/).

Full detail: [`docs/data.md`](docs/data.md), [`docs/models.md`](docs/models.md),
[`MODEL_CARD.md`](MODEL_CARD.md).

## Repository Structure

```text
sentiment-analysis/
├── src/                      # Library code: preprocessing, training, prediction, runtime pipeline
├── experiments/               # Scripts that produce results/ artifacts (evaluation, calibration, etc.)
├── data/                      # Raw datasets (gitignored)
├── results/                   # Persisted splits, metrics, reports, plots
├── docs/                      # Detailed technical documentation and diagrams
├── depression_model_v2/       # Canonical v2 classical model (TF-IDF + Logistic Regression)
├── depression_bert_model_v2/  # Canonical v2 BERT model (gitignored — large weights)
├── MODEL_CARD.md
├── RESPONSIBLE_USE.md
├── requirements.txt
└── README.md
```

Full system-level picture, including the v1/v2 runtime distinction:
[`docs/architecture.md`](docs/architecture.md).

## Installation

Verified environment:

- Python 3.10
- Historical conda environment name: `bert310`
- Validated on Apple Silicon using PyTorch's MPS backend

**The exact historical conda environment-creation command is not currently recorded in the
repository.**

Install dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins verified top-level packages; it is **not a complete transitive
lockfile**. Several experiment scripts import `scipy`, and `scipy==1.15.3` is explicitly pinned
in `requirements.txt`.

NLTK resources are also required for the classical preprocessing pipeline
(`src/preprocess.py`):

```bash
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

Full environment detail and known gaps: [`docs/reproducibility.md`](docs/reproducibility.md).

## Quickstart

The repository currently exposes **interactive CLI entry points only** — there is no batch
inference API. **Depression inference now loads the canonical v2 models** (`depression_model_v2/`,
`depression_bert_model_v2/`); **sentiment inference remains on the original v1 artifacts**
(`sentiment_model.pkl`, `sentiment_bert_model/`) — sentiment was never part of the v2 canonical
research track and is out of scope for that migration:

```bash
python -m src.predict                    # classical (TF-IDF + LR): v1 sentiment + v2 depression
python -m src.predict_bert                # DistilBERT sentiment CLI (v1)
python -m src.predict_depression_bert     # DistilBERT depression CLI (v2)
python -m src.mental_health_pipeline      # combined: v1 sentiment + v2 depression + risk-heuristic CLI
```

Each of these prompts for text on stdin and prints a result; there is no non-interactive flag.
Running the CLI does **not** mean the underlying model or the `compute_risk()` heuristic is
clinically validated — see [Safety and Responsible Use](#safety-and-responsible-use).

**For the official, reproducible benchmark numbers, use the evaluation workflow in
[`docs/reproducibility.md`](docs/reproducibility.md)** — the canonical v2 test-set results
(Section "Canonical Results" above) were produced by that evaluation pipeline, not by running
the interactive CLI.

## Reproduce the Canonical Research Evaluation

```bash
python -m src.depression_split
python -m src.train_depression_v2
python -m src.train_depression_bert_v2
python -m experiments.evaluate_models
```

Additional analyses, run against the models and predictions produced above:

```bash
python -m experiments.calibration_analysis_depression_v2
python -m experiments.error_analysis_depression_v2
python -m experiments.multiseed_depression_v2
```

**Ordering matters:** `experiments.error_analysis_depression_v2` reads the prediction files
written by `experiments.evaluate_models` rather than performing its own inference, so
`evaluate_models` must run before `error_analysis_depression_v2`.

Full command-by-command detail, expected runtimes, and troubleshooting:
[`docs/reproducibility.md`](docs/reproducibility.md).

## Documentation

| Document | Purpose |
| --- | --- |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Canonical model summary — architecture, training, full results |
| [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) | Responsible-use boundary and safety guidance |
| [`docs/architecture.md`](docs/architecture.md) | System architecture, including the v1/v2 runtime split |
| [`docs/data.md`](docs/data.md) | Dataset, canonical split, and leakage-correction history |
| [`docs/models.md`](docs/models.md) | Model architectures and training configuration |
| [`docs/evaluation.md`](docs/evaluation.md) | Evaluation, calibration, error-analysis, and multi-seed methodology |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Operational guide to reproducing the canonical workflow |
| [`docs/development.md`](docs/development.md) | Maintainer/contributor practices |

## Research Context

The software/model should be stabilized before deciding whether a research paper is warranted.
High benchmark performance alone — strong accuracy, BERT outperforming the classical model,
calibration improvements, multi-seed stability — does not establish research novelty.

Future research work built on this infrastructure should begin with a literature review, an
explicit research question, a hypothesis, appropriate baselines, appropriate validation, and
reproducible experiments. This repository does not currently claim a novel research
contribution.

See [`docs/development.md`](docs/development.md), [`docs/evaluation.md`](docs/evaluation.md),
[`MODEL_CARD.md`](MODEL_CARD.md).

## Known Limitations

- Single primary depression dataset and domain (Reddit-derived).
- Labels reflect source context, not clinician diagnosis — no clinician-annotated ground truth.
- No external validation dataset; all evaluation is on held-out data from the same source.
- No demographic or subgroup analysis.
- Single-language (English), single-platform scope.
- BERT can be confidently wrong on individual predictions despite strong aggregate calibration.
- `compute_risk()` is an unvalidated heuristic, not a clinical risk score.
- Runtime depression inference now uses the canonical v2 models; runtime sentiment inference
  remains on the original v1 models (out of scope for the v2 track). Neither is clinically
  validated — see [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).
- Automated test suite is minimal — it does not cover model inference or the runtime pipeline.
- No CI yet.

Full detail: [`MODEL_CARD.md`](MODEL_CARD.md), [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## License and Data Attribution

There is currently no `LICENSE` file in this repository. Dataset provenance and licensing for
`data/depression_dataset.csv` are not fully established in the repository — this remains an
open item before any external redistribution or sharing of the data. Do not assume any specific
license applies. Full discussion: [`docs/data.md`](docs/data.md).

## Development

Contributors should read [`docs/development.md`](docs/development.md) before making changes.
Run the test suite with `python -m pytest tests/ -q`; see
[`docs/development.md`](docs/development.md) §15 for what it covers.
The repository has substantial validated research artifacts (the canonical v2 workflow), but
still has engineering work remaining — including broader test coverage and CI — before it should
be treated as a polished, deployable application.

## Citation / Evidence

This project does not have a published paper or citation. Quantitative claims in this README
are traceable to repository artifacts and the detailed documentation linked throughout. Primary
evidence locations:

- [`results/depression_v2_model_comparison.csv`](results/depression_v2_model_comparison.csv)
- [`results/calibration/`](results/calibration/)
- [`results/error_analysis/`](results/error_analysis/)
- [`results/multiseed/`](results/multiseed/)

See also [`MODEL_CARD.md`](MODEL_CARD.md) and [`docs/evaluation.md`](docs/evaluation.md).

## Final Navigation

| | |
| --- | --- |
| **Start here** | [`README.md`](README.md) |
| **Technical** | [`MODEL_CARD.md`](MODEL_CARD.md) · [`docs/architecture.md`](docs/architecture.md) · [`docs/models.md`](docs/models.md) · [`docs/data.md`](docs/data.md) · [`docs/evaluation.md`](docs/evaluation.md) |
| **Reproduction** | [`docs/reproducibility.md`](docs/reproducibility.md) |
| **Development** | [`docs/development.md`](docs/development.md) |
| **Safety** | [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) |
