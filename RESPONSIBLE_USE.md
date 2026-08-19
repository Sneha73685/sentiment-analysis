# Responsible Use

## 1. Purpose of This Document

This repository contains machine learning models and pipelines that classify
free text for sentiment and for a depression-related label. Text
classifiers of this kind are easy to misread as diagnostic or clinical
tools, especially when their output includes words like "depression risk"
or a confidence percentage. This document exists to state, precisely and
in one place, what this system does and does not do, so that no one working
with, extending, or citing this repository mistakes a benchmark result for
a clinical claim.

Every other document in this repository (including `README.md` and
`MODEL_CARD.md`) defers to this document on questions of appropriate use.
If a technical document elsewhere appears to suggest a use this document
prohibits, this document governs.

## 2. What This Project Is

This project is a machine learning / NLP research and engineering
prototype. It experiments with:

- **Sentiment classification** — labeling text as positive or negative
  using both a classical TF-IDF + Logistic Regression model and a
  fine-tuned DistilBERT model.
- **Depression-related text classification** — labeling text according to
  a dataset-defined target (see Section 4), again using both a classical
  and a DistilBERT model.
- **Combined text analysis** — a pipeline that runs the sentiment and
  depression classifiers together and combines their outputs.
- **Signal detection** — simple keyword/phrase matching for terms
  associated with hopelessness, worthlessness, emotional numbness, and
  fatigue.
- **Trajectory analysis** — aggregating the combined analysis across a
  sequence of texts to report a trend.
- **Robustness / Monte Carlo experiments** — repeatedly perturbing input
  text and re-running the pipeline to observe how much its output changes.

Every one of these components operates on text and on learned statistical
patterns in a specific dataset. None of them has access to, or reasons
about, any information beyond the text it is given.

## 3. What This Project Is NOT

- This is **not a diagnostic system**.
- This is **not a clinical decision-support system**.
- This is **not a validated mental-health screening instrument**.
- This is **not a clinical risk-assessment system**.
- This system **must not be used** to make medical, psychiatric,
  employment, insurance, education, legal, or emergency decisions about
  any individual.
- Model predictions **must not be interpreted as diagnoses**, and no
  output from this repository constitutes medical advice or a medical
  opinion.

## 4. Training Data and Label Limitations

The depression-related classifier is trained on `data/depression_dataset.csv`,
a Reddit-derived dataset. Its labels come from the source/context the text
was collected from (e.g. subreddit membership), **not from a clinician or
any clinical diagnostic process**. These labels indicate that a piece of
text was drawn from a particular source context — they are not
clinician-verified diagnoses of the person who wrote it.

Because the labels are context-derived rather than clinically annotated,
the dataset may contain:

- label noise,
- contextual artifacts (e.g. moderator notices, meta-discussion about the
  community itself, or posts that mention the topic of depression without
  describing the author's own state),
- topical posts that do not represent an individual's mental-health state,
- duplicated or near-duplicated content,
- domain-specific language particular to the originating platform and
  community.

As one observed indication of this: the project's error analysis
(`results/error_analysis/`) surfaced several test examples labeled
"depressed" whose text reads as motivational, informational, or otherwise
unrelated to a personal expression of depression (for example, a post
consisting only of a motivational quote). This is presented here as an
**observed indication of possible label noise in the source dataset**, not
as a claim that these specific examples, or any fixed proportion of the
dataset, are definitively mislabeled — no manual re-annotation was
performed to confirm this at scale.

The canonical (v2) methodology explicitly removed **81 duplicate
`clean_text` rows** before constructing the train/validation/test split, in
order to prevent the same text from appearing in more than one split. This
addresses exact-duplicate leakage; it does not address label noise or
near-duplicate content, which remain open limitations.

## 5. Model Confidence Is Not Clinical Probability

A model's reported probability (its "confidence") is the probability the
trained model assigns to a class, given the trained parameters, the
training data, and the specific task the model was trained on. It is a
statement about the model's internal state relative to a benchmark
dataset.

It is **not** a medically validated probability that a person has
depression, and it should never be described or reported as such.

The project's calibration analysis (`results/calibration/`) measured how
well the DistilBERT model's stated confidence matches its actual accuracy
on the frozen test set. The finding was: **BERT showed strong aggregate
calibration on the frozen test set (low Expected Calibration Error), but
individual incorrect predictions could still carry very high confidence.**
In other words, being well-calibrated "on average" does not mean every
individual prediction's confidence is trustworthy — a small number of
wrong predictions were made with confidence comparable to the model's
correct predictions.

The project's five-seed stability study (`results/multiseed/`) confirmed
that this confidence-on-errors phenomenon was observed across all five
training seeds tested, not only the canonical seed 42 — i.e. it is not an
artifact of one particular training run.

This is a statement about **model calibration behavior on a specific
benchmark**. It is not, and must not be read as, a clinical claim about
how reliably any individual's mental state can be assessed from text.

## 6. The Risk Heuristic

The combined pipeline (`src/mental_health_pipeline.py`) includes a function,
`compute_risk()`, that is a simple, hand-written heuristic combining:

- the sentiment classifier's label, and
- the depression classifier's label.

It currently maps combinations of these two labels into one of three
categories: `high`, `moderate`, or `low`.

**This is not a validated risk score.** It was written as a small
combination rule, not learned from data, and not validated against any
independent ground truth for what a meaningful "risk level" should be for
a given piece of text. No dataset of clinician-assigned or otherwise
independently verified risk levels was used to construct or check it.

The heuristic should be interpreted as an **engineering/demonstration
layer** — a way to show how multiple model outputs could be combined into
a single downstream signal — rather than as a clinically meaningful risk
assessment of any kind.

## 7. Trajectory Analysis

`src/mental_health_trajectory.py` aggregates the combined analysis
(Section 2) across a sequence of texts supplied over time, and reports
whether the resulting risk scores appear to be increasing, decreasing, or
stable.

Because trajectory analysis is built directly on top of the components
described above, **it inherits every limitation of**:

- the underlying depression classifier (Section 4, Section 5),
- the sentiment classifier,
- the keyword-based signal detector (Section 2),
- and the `compute_risk()` heuristic (Section 6).

A reported trend such as `"increasing_risk"`, `"decreasing_risk"`, or
`"stable"` is therefore the output of an **unvalidated heuristic applied
to unvalidated heuristic outputs**, layered twice. It must not be
interpreted as a validated clinical progression, and must not be used as
evidence of a real change in any person's mental-health state.

## 8. Monte Carlo / Robustness Analysis

`src/monte_carlo_simulation.py` implements an exploratory sensitivity
experiment: it perturbs an input text (currently by inserting a random
intensifier word) and repeatedly re-runs the combined analysis pipeline
(Section 2) on the perturbed variants, reporting the mean and variance of
the resulting risk scores.

This is an **exploratory sensitivity experiment**, and its results must
not be read as establishing:

- certified robustness,
- clinical reliability,
- adversarial robustness, or
- any statistical guarantee about a person's mental-health state.

It is also important to note what this experiment measures: because it
runs the perturbed text through the entire combined pipeline
(sentiment + depression models + `compute_risk()`), its output primarily
reflects the **sensitivity of the combined heuristic pipeline as a
whole**, not a complete, isolated robustness evaluation of either
underlying neural classifier on its own. A classifier could have stable,
well-behaved probability outputs under perturbation while the
downstream heuristic's discrete `high`/`moderate`/`low` mapping still
shows apparent variance, or vice versa — the current experiment does not
separate these effects.

## 9. Bias and Generalization Limitations

The following limitations are known and currently undocumented-away by
any mitigation in this repository:

- The depression-related model relies on a **single primary dataset**.
- That dataset is drawn from a **single platform (Reddit)** and a
  **Reddit-derived domain** (see Section 4).
- The project currently operates in a **single language** (English).
- There is **limited domain diversity** — text style, register, and topic
  distribution reflect the source community, not the general population.
- **No demographic stratification** of the dataset or of model performance
  has been performed — there is no analysis of how performance varies by
  age, gender, race, ethnicity, or any other demographic factor, because
  this information is not present in the dataset.
- **No external clinical validation** has been performed.
- **No external, independently sourced test dataset** has been used —
  all evaluation is on held-out data from the same source dataset.
- There is potential **sampling bias** in how the source dataset was
  originally collected (not controlled by this project).
- There is potential **platform- and community-specific linguistic bias**
  — language patterns typical of this Reddit community may not generalize
  to other ways depression-related experiences are expressed elsewhere.

**The absence of a measured demographic bias analysis does not demonstrate
the absence of bias.** No demographic statistics have been collected for
this dataset, and none are asserted or implied by this document or any
other part of this repository.

## 10. Intended Use

Appropriate uses of this repository include:

- machine learning / NLP experimentation,
- software engineering practice (e.g. building and evaluating classical
  vs. transformer-based text classification pipelines),
- model comparison (e.g. classical vs. BERT-based approaches),
- reproducible ML methodology (e.g. leakage-free evaluation design,
  calibration analysis, multi-seed stability testing),
- research hypothesis generation,
- offline experimentation on the project's own benchmark data,
- educational demonstration of NLP pipeline design and evaluation
  practice.

## 11. Prohibited / Inappropriate Uses

This system must not be used for:

- diagnosing depression in any individual,
- determining whether a person has a psychiatric condition,
- determining suicide or self-harm risk,
- replacing a clinician or licensed mental-health professional,
- automated mental-health screening presented to end users as
  authoritative,
- employment decisions,
- insurance decisions,
- education or admissions decisions,
- law-enforcement decisions,
- automated intervention decisions of any kind,
- ranking, sorting, or otherwise comparing people according to their
  perceived mental-health status.

**This system has not been developed or validated for suicide or
self-harm prediction in any form**, and no component of this repository
should be represented, deployed, or relied upon as performing that
function.

## 12. If This Project Were to Become a Clinical Research System

This repository does not currently satisfy the requirements that would be
needed to make any clinical claim, and nothing in this document should be
read as implying otherwise. At a high level, a project seeking to make
clinical claims would need, at minimum:

- clinically validated labels (produced through a defined clinical
  annotation process, not source-context labeling),
- an appropriate study design for the clinical question being asked,
- independent external validation on data not used in any part of model
  development,
- clinical and domain expertise as part of the design and review process,
- ethics / IRB (or equivalent) review where applicable,
- demographic and subgroup evaluation,
- prospective validation where appropriate to the intended use,
- rigorous calibration assessment specific to the clinical claim being
  made,
- clearly defined clinical endpoints agreed with domain experts,
- appropriate privacy and data-governance controls for any personal or
  health-related data, and
- a dedicated safety evaluation appropriate to the deployment context.

None of these are currently in place. This section describes a
hypothetical future direction, not a current capability.

## 13. Current Research Status

- **v2 is the canonical research/evaluation model generation** for the
  depression-related classifiers. The original (v1) evaluation is
  preserved only as a historical record (see Section 14) and is not
  presented as valid.
- The canonical test set (`results/splits/depression_test.csv`) is
  **frozen** — it was not used for training, validation, hyperparameter
  selection, or calibration fitting for either canonical model.
- Both the classical (TF-IDF + Logistic Regression) and DistilBERT models
  have been evaluated on this same frozen, held-out test set.
- A calibration analysis exists (`results/calibration/`).
- An error analysis exists (`results/error_analysis/`).
- A five-seed training-stability analysis exists (`results/multiseed/`).
- Despite this evaluation rigor, **the project remains an engineering and
  research prototype**. The existence of strong benchmark performance
  numbers does not establish clinical validity, and no part of this
  repository should be read as claiming otherwise.

**Runtime pipeline note:** the runtime/inference code path
(`src/mental_health_pipeline.py`, `src/predict_depression_bert.py`,
`src/predict.py`, and the components built on top of them — trajectory
analysis and the Monte Carlo experiment) now loads the **canonical v2
depression model** described in this section, via centralized path
constants in `src/model_config.py`. Sentiment inference is unaffected and
remains on its original v1 artifacts — sentiment was never part of the
v2 canonical research track.

**This does not mean the runtime is clinically validated, or that using
the CLI constitutes clinical use.** The rigorous evaluation described in
this section is a benchmark result on one frozen, Reddit-derived test
set — it says nothing about performance on arbitrary real-world text
entered into the CLI, and it does not validate `compute_risk()`,
trajectory analysis, or the Monte Carlo sensitivity experiment, all of
which remain the same unvalidated heuristics described elsewhere in this
document regardless of which depression classifier sits underneath them.

## 14. Reporting and Citation Guidance

When describing this project's results, scope the claim to exactly what
was measured.

**Good example:**
> "On the project's frozen Reddit-derived test set, the v2 DistilBERT
> classifier achieved 98.43% accuracy."

**Bad example:**
> "The model can detect depression with 98.43% accuracy."

The second statement overgeneralizes in at least three ways: it drops the
benchmark/dataset qualifier (implying general-purpose detection rather
than performance on one specific, Reddit-derived, non-clinically-labeled
test set), it uses "detect depression" as though the model observes a
clinical fact rather than predicting a dataset-defined label, and it
implies a level of real-world validity that has not been established
(Sections 4, 5, 9, and 12).

**The historical result of 0.9933 accuracy recorded under
`results/legacy_full_dataset_eval/` must not be reported as a valid
held-out performance result.** That evaluation was affected by train/test
overlap (the evaluated model had been trained on the majority of the rows
it was later "tested" on) and is preserved only as a historical record of
a bug that was found and corrected — see the README in that directory for
detail.

## 15. Limitations Summary

| Area | Current limitation |
| --- | --- |
| Labels | Reddit-derived, not clinician-verified |
| Domain | Single primary dataset/platform |
| Validation | No external clinical validation |
| Confidence | Model confidence, not clinical probability |
| Risk heuristic | Unvalidated |
| Trajectory | Unvalidated heuristic aggregation |
| Robustness | Exploratory Monte Carlo sensitivity analysis |
| Generalization | Unknown outside dataset/domain |
| Demographics | No subgroup validation performed |
| Clinical use | Not appropriate |

## 16. Further Reading

- [`README.md`](README.md) — project overview and quickstart.
- [`MODEL_CARD.md`](MODEL_CARD.md) — canonical (v2) model summary:
  architecture, training configuration, full held-out results,
  calibration, error analysis, and multi-seed stability.
- [`docs/architecture.md`](docs/architecture.md) — system architecture,
  including the runtime depression-model migration (v1 to v2) and the
  unmigrated v1 sentiment track referenced throughout this document.
- [`docs/data.md`](docs/data.md) — dataset provenance and limitations,
  the canonical split methodology, and the full v1-to-v2 leakage
  correction history.
- [`docs/models.md`](docs/models.md) — model architecture and training
  configuration detail.
- [`docs/evaluation.md`](docs/evaluation.md) — full evaluation,
  calibration, error-analysis, and multi-seed methodology.
- [`docs/reproducibility.md`](docs/reproducibility.md) — operational
  guide to reproducing the canonical split, training, and evaluation
  workflow.
- [`docs/development.md`](docs/development.md) — maintainer/contributor
  practices.

The detailed supporting evidence for the calibration, error-analysis, and
multi-seed claims made throughout this document also lives directly in
`results/calibration/CALIBRATION_REPORT.md`,
`results/error_analysis/ERROR_ANALYSIS_REPORT.md`,
`results/multiseed/MULTISEED_REPORT.md`, and
`results/legacy_full_dataset_eval/README.md`.
