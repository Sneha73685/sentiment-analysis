# Model Card: Depression Text Classifiers (v2)

This card describes the **canonical (v2) research/evaluation** depression
text classifiers. It does not describe a clinical model — see
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) before interpreting anything in
this document. Section 19 explains an important current discrepancy
between this canonical model generation and what the repository's runtime
code currently executes.

## 1. Model Summary

Two models predict a binary, dataset-defined label over free text:

| Label | Meaning |
| --- | --- |
| `not_depressed` (0) | Text does not carry the dataset's depression label |
| `depressed` (1) | Text carries the dataset's depression label |

Both are trained on the same Reddit-derived dataset
(`data/depression_dataset.csv`, see Section 4). **The labels are
dataset/source-context labels, not clinical diagnoses** — no clinician
was involved in labeling this data.

The canonical research comparison contains two models:

1. **TF-IDF + Logistic Regression** ("classical")
2. **DistilBERT fine-tuned for sequence classification** ("BERT")

## 2. Intended Use

Appropriate uses: NLP research, ML experimentation, model comparison
(classical vs. transformer-based text classification), reproducible
evaluation methodology, educational/engineering demonstration, and offline
experimentation on this project's own benchmark.

These models are **not clinically validated** and must not be described as
such. Full appropriate/prohibited-use guidance:
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## 3. Out-of-Scope Use

Not for: clinical diagnosis, clinical screening, psychiatric assessment,
suicide/self-harm risk prediction, medical decision-making, or any
automated decision-making about individuals (employment, insurance,
education, legal, or emergency contexts). Full detail:
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) §3, §11.

## 4. Dataset

Source file: `data/depression_dataset.csv`.

| Property | Value |
| --- | --- |
| Raw row count | 7731 |
| Duplicate `clean_text` rows removed | 81 |
| Final unique rows | 7650 |

*Source: `results/splits/depression_split_manifest.json`.*

The dataset is Reddit-derived; its `is_depression` label reflects the
source/context the text was collected from, not a clinical assessment.
**No clinician annotation was performed.** Known limitations — label
noise, contextual/topical posts mentioning depression without expressing
it personally, and domain-specific language — are documented in
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) §4 and illustrated with
concrete examples in Section 13 below.

**Provenance/license:** no dataset citation or license file is present in
this repository for `depression_dataset.csv`. This is an explicit,
unresolved documentation gap — it is stated here rather than filled with
an invented citation.

## 5. Canonical Data Split

| Split | Rows | Class 0 (not_depressed) | Class 1 (depressed) |
| --- | ---: | ---: | ---: |
| Train | 5354 | 2721 | 2633 |
| Validation | 1148 | 584 | 564 |
| Test | 1148 | 584 | 564 |

*Source: `results/splits/depression_split_manifest.json`.*

- **Random seed:** 42.
- **Stratification:** by label, at both the train/(val+test) and
  val/test splits.
- **Deduplication:** the 81 duplicate rows above were removed before
  splitting, so no duplicate text can appear across splits.
- **Overlap verification:** the split manifest records
  `"no_overlap_verified": true`; zero text overlap between train, val, and
  test was independently re-checked at multiple points in this project's
  history.

**Purpose of each split:** train fits model parameters; validation is
used for model selection (classical: diagnostic reporting; BERT: early
stopping and best-checkpoint selection); test is used exactly once, after
training and model selection are complete, purely to measure final
performance.

**The test set (`results/splits/depression_test.csv`) is frozen: it was
not used for fitting, hyperparameter selection, checkpoint selection, or
calibration fitting for either canonical model.**

## 6. Preprocessing

### Classical model
Text is cleaned with `src/preprocess.py`'s `clean_text` (lowercasing, URL
removal, non-letter character removal, tokenization, stopword removal,
lemmatization) before TF-IDF vectorization. This function is applied
identically at training time (`src/train_depression_v2.py`) and at
evaluation time (`experiments/evaluate_models.py`).

### BERT model
Raw text (the dataset's `clean_text` column, with no additional
normalization) is passed directly to the DistilBERT tokenizer with
truncation and `max_length=256`. This is identical at training time
(`src/train_depression_bert_v2.py`) and evaluation time
(`experiments/evaluate_models.py`).

*Source: `depression_bert_model_v2/training_config.json` (`"max_length": 256`).*

**Spell correction is not part of canonical BERT training or evaluation.**
An earlier version of the combined runtime pipeline
(`src/mental_health_pipeline.py`) applied spell correction before BERT
inference; this was identified as a training/inference preprocessing
mismatch and removed from that pipeline. It was never part of the v2
training or evaluation scripts described in this card.

## 7. Model Architecture

### 7.1 TF-IDF + Logistic Regression

| Component | Configuration |
| --- | --- |
| Vectorizer | `TfidfVectorizer(max_features=5000)` |
| Classifier | `LogisticRegression(max_iter=1000, class_weight="balanced")` |
| Solver | Not explicitly set — uses scikit-learn's default (`lbfgs`) |
| Label mapping | `{0: "not_depressed", 1: "depressed"}` |

*Source: `depression_model_v2/training_config.json`.*

### 7.2 DistilBERT

| Component | Configuration |
| --- | --- |
| Base checkpoint | `distilbert-base-uncased` |
| Head | Sequence classification head, `num_labels=2` |
| `id2label` | `{"0": "not_depressed", "1": "depressed"}` |
| `label2id` | `{"not_depressed": 0, "depressed": 1}` |
| Tokenizer | `DistilBertTokenizerFast` (from the same base checkpoint) |
| Max sequence length | 256 |

*Source: `depression_bert_model_v2/training_config.json`.*

## 8. Training Configuration

| Parameter | Classical | BERT |
| --- | --- | --- |
| Seed | 42 | 42 |
| Train rows | 5354 | 5354 |
| Validation rows | 1148 | 1148 |
| Learning rate | n/a (closed-form/iterative solver, not gradient-descent LR) | 2e-05 |
| Batch size | n/a | 4 (train and eval) |
| Gradient accumulation | n/a | 2 |
| Weight decay | n/a | 0.01 |
| Max epochs | n/a (`max_iter=1000` iterations) | 4 |
| Early stopping | n/a | patience = 1 (on validation accuracy) |
| Max sequence length | n/a (TF-IDF, `max_features=5000`) | 256 |
| Base checkpoint | n/a | `distilbert-base-uncased` |

*Source: `depression_model_v2/training_config.json`,
`depression_bert_model_v2/training_config.json`. Values are taken directly
from these files, not retyped from any narrative report.*

## 9. Training and Model Selection Protocol

```
raw dataset (data/depression_dataset.csv)
  -> deduplication (81 rows removed)
  -> canonical stratified split, seed 42 (train/val/test)
  -> training (train split only)
  -> validation (model selection / early stopping)
  -> frozen test evaluation (exactly once)
```

- **Test data was not used for training** for either model.
- **Test data was not used for model selection** — the classical model's
  validation-set metrics (Section 8) are diagnostic only (there is no
  hyperparameter search in the canonical script); BERT's early stopping
  and best-checkpoint selection used validation accuracy exclusively
  (`metric_for_best_model: "accuracy"`, evaluated on `depression_val.csv`).
- **These canonical v2 models were retrained after an evaluation-leakage
  problem was discovered in the original (v1) approach** — see Section 17.

## 10. Official Held-Out Test Results

Test set: `results/splits/depression_test.csv`, **n = 1148**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9556 | 0.9777 | 0.9309 | 0.9537 | 0.9900 |
| DistilBERT | 0.9843 | 0.9964 | 0.9716 | 0.9838 | 0.9976 |

*Source: `results/depression_v2_model_comparison.csv`
(cross-checked against `results/metrics_depression_v2_classical.csv`,
`results/metrics_depression_v2_bert.csv`, and
`results/depression_v2_eval_report_{classical,bert}.json` — all four
sources agree to the reported precision).*

**Confusion matrices** (rows = actual, columns = predicted; label order
`[not_depressed, depressed]`):

Classical:
```
[[572,  12],
 [ 39, 525]]
```

BERT:
```
[[582,   2],
 [ 16, 548]]
```

*Source: `results/depression_v2_eval_report_classical.json`,
`results/depression_v2_eval_report_bert.json` (`confusion_matrix` field).*

Reading example: the classical model's top-left cell (572) is the number
of truly `not_depressed` test examples it correctly predicted as
`not_depressed`; its top-right cell (12) is the number of truly
`not_depressed` examples it incorrectly predicted as `depressed` (false
positives).

## 11. Per-Class Performance

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Classical | not_depressed | 0.9362 | 0.9795 | 0.9573 | 584 |
| Classical | depressed | 0.9777 | 0.9309 | 0.9537 | 564 |
| BERT | not_depressed | 0.9732 | 0.9966 | 0.9848 | 584 |
| BERT | depressed | 0.9964 | 0.9716 | 0.9838 | 564 |

*Source: `results/depression_v2_eval_report_classical.json`,
`results/depression_v2_eval_report_bert.json` (`per_class` field, values
taken directly, not recalculated).*

## 12. Calibration

Calibration was assessed on the frozen test set; temperature scaling was
fit exclusively on `depression_val.csv` predictions and then applied,
unchanged, to test-set logits.

| Metric | BERT raw | BERT temperature-scaled |
| --- | ---: | ---: |
| Temperature | — | 1.8148 |
| Log-loss | 0.0796 | 0.0605 |
| Brier score | 0.0143 | 0.0136 |
| ECE | 0.0149 | 0.0108 |

*Source: `results/calibration/calibration_report.json`
(`temperature_learned_on_val`, `bert.raw_test`, `bert.calibrated_test`).*

- Temperature was fit on validation predictions only; **test predictions
  were never used to fit the temperature.**
- **Predicted classes did not change after temperature scaling**
  (`accuracy_unchanged_by_temperature_scaling: true` in the calibration
  report, verified directly rather than assumed).

**Key finding:** BERT shows strong *aggregate* calibration (low ECE), but
its mean confidence on its own incorrect test predictions was 0.935 raw
/ 0.886 calibrated — individual incorrect predictions can still be highly
confident. This reflects how ECE is computed (population-weighted across
confidence bins; BERT's 18 test errors sit inside an otherwise
near-perfect top confidence bin) — it does not mean every prediction's
confidence is individually trustworthy.

**Model confidence is a statement about the model's internal probability
estimate on this benchmark. It is not a clinically validated probability
of depression.** See [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) §5.

## 13. Error Analysis

| Quantity | Value |
| --- | --- |
| Classical test errors | 51 (12 FP, 39 FN) |
| BERT test errors | 18 (2 FP, 16 FN) |
| Classical wrong, BERT correct | 36 |
| BERT wrong, classical correct | 3 |
| Both models wrong | 15 |
| Both correct but low confidence (<0.65) | 69 |

*Source: `results/error_analysis/error_summary.csv`.*

**OBSERVED FINDING:** classical and BERT errors are largely disjoint — 36
of classical's 51 errors are cases BERT gets right, while only 3 of
BERT's 18 errors are cases classical gets right.

**OBSERVED FINDING:** several of BERT's false negatives are short posts
whose text does not obviously express depression despite a `depressed`
label (e.g. a post consisting only of a motivational quote) — consistent
with the label-noise caveat in Section 4.

**HYPOTHESIS, not confirmed:** an eleven-pattern linguistic analysis
(negation, hedging/uncertainty, sentence length, topical mentions of
"depression" without self-report framing, and others, all regex-based
heuristics — not validated NLP or psychological classifiers) found two
nominally significant associations at the uncorrected p<0.05 level
(longer posts and topical-mention posts had *lower* error rates for the
classical model). **None survived Bonferroni correction** for the 22
simultaneous tests performed (11 patterns × 2 models; corrected threshold
≈0.00227). These are reported as a hypothesis for future, better-powered
testing, not as an established finding. Regex-derived categories in this
analysis should not be read as validated linguistic or psychological
phenomena.

*Source: `results/error_analysis/ERROR_ANALYSIS_REPORT.md`,
`results/error_analysis/linguistic_pattern_analysis.csv`.*

## 14. Multi-Seed Stability

Five independent training seeds — **42, 123, 2024, 3407, 7777** — each
retrained both models from scratch on the same canonical split and
evaluated once on the frozen test set.

| Metric | Classical (mean ± SD) | BERT (mean ± SD) |
| --- | ---: | ---: |
| Accuracy | 0.9556 ± 0.0000 | 0.9829 ± 0.0047 |
| F1 | 0.9537 ± 0.0000 | 0.9825 ± 0.0048 |
| ROC-AUC | 0.9900 ± 0.0000 | 0.9975 ± 0.0004 |
| ECE | 0.1033 ± 0.0000 | 0.0143 ± 0.0018 |
| Brier | 0.0487 ± 0.0000 | 0.0150 ± 0.0026 |
| Log-loss | 0.2001 ± 0.0000 | 0.0768 ± 0.0074 |

*Source: `results/multiseed/multiseed_summary.csv`.*

The classical model's exact-zero standard deviation is expected, not an
error: scikit-learn's `LogisticRegression` default solver (`lbfgs`) is
deterministic and does not use `random_state` — the classical fit is
identical regardless of seed. **BERT outperformed the classical model at
all five evaluated seeds** (per-seed BERT accuracy: 0.9861, 0.9861,
0.9869, 0.9782, 0.9774 — every value above classical's fixed 0.9556; see
`results/multiseed/multiseed_results.csv`). This supports the stability
of BERT's advantage on this benchmark across training-seed variation; it
does not establish generalization beyond this dataset and split (Section
16).

## 15. Known Failure Modes

### Classical model
- Bag-of-words/TF-IDF representation has no mechanism to interpret
  idiom, sarcasm, or context — e.g., one classical false positive was a
  self-deprecating joke ("...if it had happened to anyone else i would
  have peed myself laughing...") that BERT correctly classified as
  `not_depressed`.
- Most classical errors occur near the decision boundary — mean
  confidence on its 51 errors was 0.650, well below its 0.862 mean
  confidence on correct predictions (Section 12).

### BERT
- Individual incorrect predictions can carry high confidence (Section
  12) — this is the model's principal documented weakness.
- Some of its errors coincide with likely label-noise examples (Section
  13).
- Its representations are fit to this specific dataset's linguistic
  patterns; **the model can exploit contextual relationships present in
  the benchmark text — it does not "understand" mental health** in any
  sense beyond the statistical patterns learned from this training
  distribution.

## 16. Generalization Limitations

- Single primary dataset (`data/depression_dataset.csv`), single platform
  (Reddit).
- No external test set — all evaluation is on held-out data from the same
  source.
- No clinical validation.
- No demographic or subgroup analysis (no such attributes exist in the
  dataset).
- No cross-platform validation.
- No prospective validation.
- Single-language scope (English).

**Benchmark performance on this frozen test set should not be interpreted
as population-level performance, or as evidence of validity outside this
specific dataset and domain.** Full discussion:
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) §9.

## 17. Version History and V1 → V2 Correction

**V1** (original approach): models were trained with ad hoc, per-script
train/test splits, and the depression BERT model was evaluated with
`experiments/evaluate_models.py` against the **entire** source dataset —
including rows the model had already been trained on. This produced a
reported accuracy of **0.9933**, which is invalid as a held-out
generalization estimate because of this train/test overlap.

**V2** (canonical, this card): the dataset was deduplicated, a
stratified, seeded, zero-overlap train/val/test split was constructed
once (Section 5), both models were retrained using only the train split,
validation was used only for model selection, and the frozen test split
was evaluated exactly once, after training was complete.

**The 0.9933 historical result is preserved for auditability
(`results/legacy_full_dataset_eval/`) but must not be cited as the
project's official performance.** See
`results/legacy_full_dataset_eval/README.md` for the full record of what
went wrong and why it was invalid.

## 18. Canonical Artifacts

| Artifact | Type | Location |
| --- | --- | --- |
| Canonical split | Lightweight (CSV + JSON manifest) | `results/splits/` |
| Classical v2 model | Small binary (~224KB, `.pkl`) | `depression_model_v2/` |
| BERT v2 model | Large binary (weights, ~256MB) | `depression_bert_model_v2/` |
| Official evaluation | Lightweight (CSV/JSON) | `results/depression_v2_*`, `results/metrics_depression_v2_*.csv` |
| Calibration results | Lightweight (CSV/JSON/MD/PNG) | `results/calibration/` |
| Error analysis | Lightweight (CSV/JSON/MD/PNG) | `results/error_analysis/` |
| Multi-seed results | Lightweight (CSV/JSON/MD/PNG); disposable per-seed weights | `results/multiseed/` (`tmp_models/` subdirectory is disposable) |

Large model weight directories are excluded from version control by
`.gitignore`; the lightweight CSV/JSON/Markdown reproducibility artifacts
listed above are not.

## 19. Runtime Status

**The canonical v2 models described in this card are the validated
research artifacts, and are now also the default runtime depression
models.** The runtime/research gap previously documented in this section
has been closed for depression inference via a runtime migration
(`src/model_config.py` centralizes the v2 paths).

Specifically:
- `src/mental_health_pipeline.py` loads `depression_bert_model_v2/` (v2).
- `src/predict_depression_bert.py` loads `depression_bert_model_v2/` (v2).
- `src/predict.py` loads `depression_model_v2/model.pkl` /
  `depression_model_v2/vectorizer.pkl` (v2 classical, via `joblib.load`).
- `src/mental_health_trajectory.py` and `src/monte_carlo_simulation.py`
  both call into `mental_health_pipeline.py`, and therefore also run on
  the v2 depression model.

**Sentiment inference is unaffected and remains on its original v1
artifacts** (`sentiment_model.pkl`, `sentiment_bert_model/`) — there is
no canonical v2 sentiment retraining track, and sentiment was explicitly
out of scope for this migration.

**Running this repository's CLI tools now exercises the v2 depression
model this card describes. This does not mean the model, or the
`compute_risk()` heuristic built on top of it, is clinically validated**
— the benchmark results in this card remain benchmark results on a
specific frozen test set, not a clinical claim. See
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## 20. Ethical / Responsible Use

This model was trained on non-clinically-labeled data, is not validated
for any clinical purpose, and must not be used for diagnosis, screening,
risk assessment, or any automated decision-making about individuals. Full
guidance, including prohibited uses and bias/generalization limitations:
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).

## 21. Reproducibility

- Split seed: 42 (`results/splits/depression_split_manifest.json`).
- Training configuration recorded per-model in
  `depression_model_v2/training_config.json` and
  `depression_bert_model_v2/training_config.json`.
- Evaluation artifacts: `results/depression_v2_model_comparison.csv`,
  `results/depression_v2_eval_report_{classical,bert}.json`.
- Five-seed stability analysis: `results/multiseed/`.

A dedicated reproducibility guide,
[`docs/reproducibility.md`](docs/reproducibility.md), documents the
complete command sequence, environment setup, and expected runtimes for
reproducing the canonical split, training, and evaluation workflow.

## 22. Model Limitations Summary

| Dimension | Limitation |
| --- | --- |
| Dataset | Reddit-derived |
| Labels | Not clinician-verified |
| Domain | Limited (single platform, single language) |
| External validation | None |
| Clinical validation | None |
| Demographic analysis | Not performed |
| Calibration | Strong aggregate BERT calibration, but confident individual errors (Section 12) |
| Risk heuristic | Unvalidated (see `RESPONSIBLE_USE.md` §6) |
| Trajectory | Unvalidated heuristic aggregation (see `RESPONSIBLE_USE.md` §7) |
| Robustness | Exploratory Monte Carlo sensitivity only (see `RESPONSIBLE_USE.md` §8) |
| Runtime | v2 depression model wired (Section 19); sentiment remains v1 (out of scope); no clinical validation |
| Generalization | Unknown outside this benchmark/dataset |

## 23. References to Repository Evidence

- Split manifest: `results/splits/depression_split_manifest.json`
- Official comparison table: `results/depression_v2_model_comparison.csv`
- Per-model evaluation reports: `results/depression_v2_eval_report_classical.json`,
  `results/depression_v2_eval_report_bert.json`
- Calibration report: `results/calibration/calibration_report.json`,
  `results/calibration/CALIBRATION_REPORT.md`
- Error analysis report: `results/error_analysis/error_analysis_report.json`,
  `results/error_analysis/ERROR_ANALYSIS_REPORT.md`
- Multi-seed report: `results/multiseed/multiseed_report.json`,
  `results/multiseed/MULTISEED_REPORT.md`
- Legacy leakage warning: `results/legacy_full_dataset_eval/README.md`
- Responsible-use guidance: `RESPONSIBLE_USE.md`
