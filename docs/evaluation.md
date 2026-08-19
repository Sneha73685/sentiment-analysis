# Evaluation Methodology and Results

## 1. Evaluation Scope

This document covers the evaluation of the **canonical v2 depression
models**: TF-IDF + Logistic Regression and DistilBERT, both evaluated on
**exactly the same frozen test set** (`results/splits/depression_test.csv`,
n = 1148). It explains how the reported numbers were obtained and why the
protocol that produced them is methodologically defensible — it does not
repeat the headline results themselves in full; those live in
[`MODEL_CARD.md`](../MODEL_CARD.md).

The original (v1) full-dataset evaluation is **historical and invalid as
a held-out estimate** — see [`docs/data.md`](data.md) and
[`results/legacy_full_dataset_eval/README.md`](../results/legacy_full_dataset_eval/README.md)
for why.

## 2. Evaluation Protocol

```
canonical split -> train-only fitting -> validation-based selection
  -> frozen test inference -> metrics -> calibration analysis
  -> error analysis -> multi-seed stability
```

| Stage | Train | Validation | Test |
| --- | --- | --- | --- |
| Model fitting | Used | Not used | **Forbidden** |
| Model selection (early stopping, checkpoint choice) | Available | Used | **Forbidden** |
| Temperature scaling (calibration parameter fit) | Not used | Used to fit T | Receives the already-fitted transformation only |
| Final evaluation (metrics, confusion matrix, ROC-AUC) | Not used | Not used | Used for measurement, no fitting |
| Error analysis | Not used | Not used | Descriptive analysis of frozen predictions, no optimization |
| Multi-seed stability | Used (retrained per seed) | Used (selection per seed) | Used for measurement only; **seeds are not chosen based on test results** |

The test set is used exclusively for **measurement after every modeling
decision has already been made**. No procedure documented in this
repository fits a parameter, selects a checkpoint, or chooses a seed
using test-set outcomes.

See [`docs/diagrams/evaluation-pipeline.mmd`](diagrams/evaluation-pipeline.mmd)
for a diagram of this protocol as one flow: frozen test-set inference,
metrics/confusion-matrix/ROC-AUC computation, validation-only temperature
fitting followed by application to the frozen test logits, error
analysis, and the multi-seed comparison.

## 3. Official Test Set

| Property | Value |
| --- | --- |
| n | 1148 |
| Class 0 (`not_depressed`) | 584 |
| Class 1 (`depressed`) | 564 |

*Source: `results/splits/depression_split_manifest.json`
(`split_counts.test`, `class_balance.test`).*

The identical 1148 test rows, in the identical order, are passed to both
models — verified directly by comparing the `text` and `true_label`
columns of `results/predictions_depression_v2_classical.csv` and
`results/predictions_depression_v2_bert.csv`. The test set was **never
used in v2 training** for either model (`test_path_used: null` in both
`depression_model_v2/training_config.json` and
`depression_bert_model_v2/training_config.json`) and has **zero text
overlap** with the train and validation splits
(`no_overlap_verified: true` in the split manifest).

The test set is **persisted** (`results/splits/depression_test.csv`) and
was not regenerated between the official evaluation, calibration
analysis, error analysis, or multi-seed study — all four analyses read
the same file.

## 4. Metrics

**Accuracy** — the fraction of the 1148 test examples correctly
classified. A single global number; does not distinguish error type.

**Precision** (for the `depressed` class) — of all examples the model
labeled `depressed`, the fraction that were actually `depressed`. Low
precision means the model over-labels `not_depressed` text as
`depressed`.

**Recall** (for the `depressed` class) — of all examples that were
actually `depressed`, the fraction the model correctly identified. Low
recall means the model misses actual `depressed` examples.

**F1** — the harmonic mean of precision and recall for the `depressed`
class, balancing the two into a single number when neither should be
optimized in isolation.

**ROC-AUC** — evaluates ranking quality across all possible decision
thresholds, not just the default 0.5 threshold used for the other
metrics above. A model can have a high ROC-AUC even if its default-
threshold accuracy is not the highest of two compared models.

**Confusion matrix** — rows are the actual label, columns are the
predicted label, in the fixed order `[not_depressed, depressed]`. The
off-diagonal cells are false positives (actual `not_depressed`, predicted
`depressed`) and false negatives (actual `depressed`, predicted
`not_depressed`) respectively.

## 5. Official Test Results

Canonical seed-42 results, n = 1148:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9556 | 0.9777 | 0.9309 | 0.9537 | 0.9900 |
| DistilBERT | 0.9843 | 0.9964 | 0.9716 | 0.9838 | 0.9976 |

*Source: `results/depression_v2_model_comparison.csv`, cross-checked
against `results/metrics_depression_v2_{classical,bert}.csv` — values
agree exactly.*

## 6. Confusion Matrices

Classical:
```
[[572,  12],
 [ 39, 525]]
```
12 false positives, 39 false negatives.

BERT:
```
[[582,   2],
 [ 16, 548]]
```
2 false positives, 16 false negatives.

*Source: `results/depression_v2_eval_report_{classical,bert}.json`
(`confusion_matrix` field).*

These figures describe **benchmark classification performance on this
dataset's labels**. They are not clinical sensitivity/specificity
figures and must not be described as such — see
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 7. Per-Class Results

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Classical | not_depressed | 0.9362 | 0.9795 | 0.9573 | 584 |
| Classical | depressed | 0.9777 | 0.9309 | 0.9537 | 564 |
| BERT | not_depressed | 0.9732 | 0.9966 | 0.9848 | 584 |
| BERT | depressed | 0.9964 | 0.9716 | 0.9838 | 564 |

*Source: `results/depression_v2_eval_report_{classical,bert}.json`
(`per_class` field), values used directly, not recalculated.*

## 8. Interpretation of the Model Comparison

**Supported statement:** BERT performs better than the classical baseline
on this fixed dataset, under this evaluation protocol.

This benchmark comparison does **not** establish that BERT is
universally better, clinically better, generalizes to all mental-health
text, or is production-ready. It is evidence of better benchmark
performance under this specific experimental setup — one dataset, one
domain, one frozen split.

## 9. Calibration

Accuracy describes whether the predicted class is correct; it says
nothing about whether the model's stated probability is trustworthy.
Three metrics assess this (lower is better for all three):

- **Brier score** — mean squared error between predicted probability and
  the true binary outcome.
- **Log-loss** — mean negative log-likelihood of the true label under the
  model's predicted probability.
- **Expected Calibration Error (ECE)** — the population-weighted gap
  between average predicted confidence and observed accuracy across
  confidence bins (10 fixed-width bins, `[0.5, 1.0]`, used throughout this
  project's calibration analysis).

| Metric | Classical (raw) | BERT (raw) |
| --- | ---: | ---: |
| Brier | 0.0487 | 0.0143 |
| ECE | 0.1033 | 0.0149 |
| Log-loss | 0.2001 | 0.0796 |

*Source: `results/calibration/calibration_metrics.csv`
(`classical_test`, `bert_raw_test` rows).*

These are statements about **model probability quality on this
benchmark**, not clinical probabilities. See
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) §5.

## 10. Temperature Scaling

Temperature scaling was applied to BERT only. The scalar temperature
**T = 1.8148** was learned by minimizing log-loss on
**validation predictions only** (`results/splits/depression_val.csv`
logits); it was then applied, unchanged, to test-set logits.

| Metric | Raw BERT | Calibrated BERT |
| --- | ---: | ---: |
| Log-loss | 0.0796 | 0.0605 |
| Brier | 0.0143 | 0.0136 |
| ECE | 0.0149 | 0.0108 |
| Accuracy | 0.9843 | 0.9843 (unchanged) |

*Source: `results/calibration/calibration_report.json`
(`temperature_learned_on_val`, `bert.raw_test`, `bert.calibrated_test`,
`accuracy_unchanged_by_temperature_scaling: true`).*

Temperature scaling divides logits by a positive scalar before the
softmax; this rescales the probability distribution over classes but
cannot change which class has the largest logit (dividing all logits by
the same positive number preserves their relative order). This is why
predicted classes cannot change under this procedure — verified directly
in the calibration report, not only argued from theory.

## 11. Confidence on Incorrect Predictions

| | BERT raw | BERT calibrated |
| --- | ---: | ---: |
| Mean confidence, correct predictions | 0.998 | 0.984 |
| Mean confidence, incorrect predictions | 0.935 | 0.886 |

*Source: `results/calibration/calibration_metrics.csv`
(`mean_confidence_correct`, `mean_confidence_incorrect` columns).*

**Aggregate calibration is strong (Section 9–10), but individual errors
can still be highly confident.** This is not a contradiction: ECE is
population-weighted across confidence bins, and BERT's 18 test errors sit
almost entirely within an otherwise near-perfectly-accurate top
confidence bin — a small number of confidently-wrong predictions barely
shifts a population-weighted average dominated by over a thousand
confidently-right ones. This is a **model-confidence failure mode**, not
a claim about clinical risk or real-world danger — see
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) §5.

## 12. Calibration Limitations

- ECE uses 10 fixed-width bins; several bins (particularly below 0.9
  confidence for BERT) contain very few samples, making their individual
  bin-level accuracy unstable to interpret visually.
- Only 18 BERT test errors exist at seed 42 — small-sample statistics.
- A single global temperature parameter cannot correct region-specific
  miscalibration (a residual gap remains visible in the calibrated
  reliability diagram's higher-confidence bins).
- No formal statistical test was performed comparing calibration quality
  between the two models — the comparison in this document is
  descriptive (the numbers as reported), not hypothesis-tested.
- Section 9–11 report the single canonical seed-42 calibration result;
  Section 17–19 extend this to five seeds.

## 13. Error Analysis

Error analysis characterizes existing errors; it is not used to modify or
improve either model.

| Quantity | Value |
| --- | --- |
| Classical test errors | 51 |
| BERT test errors | 18 |
| Classical wrong, BERT correct | 36 |
| BERT wrong, classical correct | 3 |

*Source: `results/error_analysis/error_summary.csv`;
disagreement total (39) cross-checked directly against
`results/error_analysis/model_disagreements.csv` (39 rows), which equals
36 + 3 exactly.*

This disagreement structure strongly favors BERT on this particular test
set. No formal significance test was performed on this specific
disagreement count in the error-analysis script, so this document reports
it descriptively rather than as a statistically tested claim.

## 14. Error Confidence

| | Mean confidence on wrong predictions |
| --- | ---: |
| Classical | 0.6501 |
| BERT | 0.935 (raw) |

*Source: `results/calibration/calibration_metrics.csv`.*

BERT's errors are, on average, made with substantially higher confidence
than classical's errors. This finding is examined further across
training seeds in Section 19.

## 15. Linguistic Error Analysis

The error analysis tested 11 heuristic, regex-based text patterns
(negation, hedging language, sentence length, topical mentions of
"depression" without self-report framing, and others) against each
model's error rate, using Fisher's exact test — 22 tests total (11
patterns × 2 models). These patterns are **proxy categories implemented
as regular expressions, not validated linguistic or psychological
annotations** — several are explicitly documented in the underlying
report as low-recall proxies (e.g. sarcasm and metaphor markers).

With 22 simultaneous tests, the Bonferroni-corrected significance
threshold is **0.05 / 22 ≈ 0.00227**. Four tests were nominally
significant at the uncorrected p < 0.05 level (`mixed_sentiment_language`
for classical, p=0.0294; `contextual_depression_mention_without_self_report`
for classical, p=0.0029, and for BERT, p=0.0332;
`long_context_heavy_input` for classical, p=0.0071) — **none survived
Bonferroni correction**.

*Source: `results/error_analysis/linguistic_pattern_analysis.csv`.*

Consequently, **no linguistic category from this analysis should
currently be presented as a confirmed, causal explanation for model
error.** The nominal findings are reported in
`results/error_analysis/ERROR_ANALYSIS_REPORT.md` as hypotheses for
better-powered future testing, not as established results.

## 16. Label-Noise Observation

Some of BERT's false negatives appear, on manual inspection of a small
number of representative examples, consistent with source-level label
noise — text that reads as informational or motivational rather than a
personal expression of depression, despite carrying a `depressed` label
(see `results/error_analysis/false_negatives_bert.csv` and the discussion
in `results/error_analysis/ERROR_ANALYSIS_REPORT.md`).

This is an **observation/hypothesis, not a demonstrated systematic
label-quality problem.** A systematic audit of label quality across the
full dataset was not performed; the observation rests on a small number
of illustrative examples read by eye, not a scored or sampled evaluation
of the dataset as a whole.

## 17. Multi-Seed Stability

Five independent training seeds — **42, 123, 2024, 3407, 7777** — each
retrained both models from scratch on the canonical split and evaluated
once on the frozen test set.

The classical model is deterministic under its current configuration
(scikit-learn's default `lbfgs` solver does not use `random_state`), so
its multi-seed mean equals the exact seed-42 value with SD = 0 for every
metric — this is expected behavior, not measurement error, and is
verified directly (all five classical rows in
`results/multiseed/multiseed_results.csv` are identical).

| Metric | BERT mean ± SD |
| --- | --- |
| Accuracy | 0.9829 ± 0.0047 |
| F1 | 0.9825 ± 0.0048 |
| ROC-AUC | 0.9975 ± 0.0004 |
| ECE | 0.0143 ± 0.0018 |
| Brier | 0.0150 ± 0.0026 |
| Log-loss | 0.0768 ± 0.0074 |

*Source: `results/multiseed/multiseed_summary.csv`.*

## 18. BERT Stability Relative to Classical

BERT exceeded the classical model's fixed 0.9556 accuracy at **all five**
evaluated seeds (per-seed BERT accuracy: 0.9861, 0.9861, 0.9869, 0.9782,
0.9774; `results/multiseed/multiseed_results.csv`). This supports the
stability of BERT's benchmark advantage across training-seed variation on
this dataset. **It is not proof of universal superiority** — it
demonstrates stability within the scope of this repeated experiment, not
generalization beyond it.

## 19. Confidence-on-Error Stability

Across the five seeds, BERT's mean confidence on its own errors ranged
from **0.825 to 0.986** (raw). The learned temperature ranged from
**1.433 to 2.069**, mean **1.708 ± 0.242** — every seed required T > 1,
consistently indicating the raw BERT probabilities benefited from
confidence compression (never the opposite direction, which would have
indicated underconfidence).

*Source: `results/multiseed/multiseed_summary.csv`
(`incorrect_confidence`, `temperature` rows).*

This describes **model probabilities on a fixed benchmark across
repeated training runs**, not clinical probabilities, and not a
population-level statement about depression risk.

## 20. Statistical and Methodological Limitations

- Single dataset, single domain (Reddit-derived).
- Single frozen canonical test set — every analysis in this document
  measures against the same 1148 rows.
- No external validation dataset.
- No demographic evaluation (no such attributes exist in the data).
- No clinician annotation of labels.
- Modest test size (n = 1148, 564 positive).
- The primary official result (Sections 5–16) is a single canonical seed
  (42); the multi-seed study (Sections 17–19) extends this to five seeds
  but still measures against the same frozen test set each time.
- No hyperparameter sweep was performed for either model.
- No external benchmark comparison exists.
- Linguistic error-analysis categories are regex proxies, not validated
  annotations (Section 15).
- The label-noise hypothesis (Section 16) has not been systematically
  validated.
- Calibration metrics (ECE in particular) depend on binning and other
  methodology choices documented in Section 9 and
  `results/calibration/CALIBRATION_REPORT.md`.

## 21. What the Evaluation Establishes

1. The v2 models were evaluated without train/test leakage.
2. BERT outperformed the classical baseline on the canonical test set.
3. The performance advantage persisted across five BERT training seeds.
4. BERT has substantially better aggregate calibration metrics than the
   classical model on this benchmark.
5. Temperature scaling improved BERT's calibration metrics without
   changing predicted classes.
6. BERT can still be highly confident on individual incorrect
   predictions, despite strong aggregate calibration.
7. Error analysis provides hypotheses about failure modes (linguistic
   patterns, possible label noise) but does not establish causal
   explanations for model behavior.

## 22. What the Evaluation Does NOT Establish

This evaluation does not establish:

- clinical validity,
- diagnostic validity,
- suicide/self-harm prediction capability,
- general-population performance,
- demographic fairness,
- generalization to domains outside this dataset,
- causal explanations for model behavior,
- real-world mental-health risk prediction, or
- production readiness.

See [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) for the full
appropriate-use boundaries.

## 23. Evaluation Artifacts

| Purpose | Script | Output |
| --- | --- | --- |
| Official evaluation | `experiments/evaluate_models.py` | `results/depression_v2_model_comparison.csv`, `results/metrics_depression_v2_*.csv`, `results/predictions_depression_v2_*.csv`, `results/depression_v2_eval_report_*.json` |
| Calibration | `experiments/calibration_analysis_depression_v2.py` | `results/calibration/` |
| Error analysis | `experiments/error_analysis_depression_v2.py` | `results/error_analysis/` |
| Multi-seed stability | `experiments/multiseed_depression_v2.py` | `results/multiseed/` |

## 24. Evaluation Reproducibility

The canonical split, trained model artifacts, per-row predictions,
evaluation reports, calibration outputs, error-analysis outputs, and
multi-seed summary are all persisted on disk (Section 23) rather than
regenerated on demand — every number in this document traces back to a
file that currently exists in this repository. See
[`docs/data.md`](data.md) for the split methodology those artifacts
depend on.

The complete command sequence to regenerate these artifacts from a clean
environment is documented in
[`docs/reproducibility.md`](reproducibility.md).

## 25. References to Repository Evidence

- [`experiments/evaluate_models.py`](../experiments/evaluate_models.py)
- [`results/depression_v2_model_comparison.csv`](../results/depression_v2_model_comparison.csv)
- [`results/depression_v2_eval_report_classical.json`](../results/depression_v2_eval_report_classical.json)
- [`results/depression_v2_eval_report_bert.json`](../results/depression_v2_eval_report_bert.json)
- [`experiments/calibration_analysis_depression_v2.py`](../experiments/calibration_analysis_depression_v2.py)
- [`results/calibration/calibration_report.json`](../results/calibration/calibration_report.json)
- [`results/calibration/CALIBRATION_REPORT.md`](../results/calibration/CALIBRATION_REPORT.md)
- [`experiments/error_analysis_depression_v2.py`](../experiments/error_analysis_depression_v2.py)
- [`results/error_analysis/error_analysis_report.json`](../results/error_analysis/error_analysis_report.json)
- [`results/error_analysis/linguistic_pattern_analysis.csv`](../results/error_analysis/linguistic_pattern_analysis.csv)
- [`results/error_analysis/ERROR_ANALYSIS_REPORT.md`](../results/error_analysis/ERROR_ANALYSIS_REPORT.md)
- [`experiments/multiseed_depression_v2.py`](../experiments/multiseed_depression_v2.py)
- [`results/multiseed/multiseed_summary.csv`](../results/multiseed/multiseed_summary.csv)
- [`results/multiseed/MULTISEED_REPORT.md`](../results/multiseed/MULTISEED_REPORT.md)
- [`results/splits/depression_split_manifest.json`](../results/splits/depression_split_manifest.json)
- [`results/legacy_full_dataset_eval/README.md`](../results/legacy_full_dataset_eval/README.md)
- [`docs/data.md`](data.md), [`docs/models.md`](models.md)
- [`docs/reproducibility.md`](reproducibility.md), [`docs/development.md`](development.md)
- [`MODEL_CARD.md`](../MODEL_CARD.md), [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md)
