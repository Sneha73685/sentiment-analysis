# Data and Dataset Methodology

## 1. Scope

This document describes the data lifecycle for the **depression-classification
component** of this project: the source dataset, its known limitations, the
canonical (v2) deduplication and splitting methodology, and the discovery and
correction of a train/test leakage problem in the original (v1) evaluation.

The v1 sentiment datasets (Section 2) are a separate, unrelated data track.
They were not part of, and are not affected by, the v2 depression
leakage correction described here.

For model-level architecture, training configuration, and performance
results, see [`MODEL_CARD.md`](../MODEL_CARD.md). For appropriate-use
guidance, see [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 2. Dataset Inventory

| File | Used by | Part of canonical v2 depression pipeline? |
| --- | --- | --- |
| `data/depression_dataset.csv` | Depression classifiers (v1 and v2) | **Yes** — the subject of this document |
| `data/sentiment140.csv` | Sentiment BERT model (v1 only) | No |
| `data/amazon_cells_labelled.txt` | Classical sentiment model (v1 only) | No |
| `data/imdb_labelled.txt` | Classical sentiment model (v1 only) | No |
| `data/yelp_labelled.txt` | Classical sentiment model (v1 only) | No |

The four sentiment-related files are used exclusively by the original
sentiment models and are not read by any v2 depression script
(`src/depression_split.py`, `src/train_depression_v2.py`,
`src/train_depression_bert_v2.py`, `experiments/evaluate_models.py`,
`experiments/calibration_analysis_depression_v2.py`,
`experiments/error_analysis_depression_v2.py`,
`experiments/multiseed_depression_v2.py`) — confirmed by inspecting each of
these scripts directly. The remainder of this document concerns
`data/depression_dataset.csv` only.

## 3. Depression Dataset

| Property | Value |
| --- | --- |
| File path | `data/depression_dataset.csv` |
| Columns | `clean_text`, `is_depression` |
| Raw row count | 7731 (7732 lines including header) |
| Label values | `is_depression` ∈ {0, 1} |

*Verified directly against the file (`wc -l`, header inspection) and
against `results/splits/depression_split_manifest.json`
(`raw_row_count: 7731`), which agree.*

The dataset's text content is consistent with Reddit posts (visible
moderator/community-context language, e.g. references to subreddit rules,
in a sample of rows) — this is an observation from the file's content,
not an externally documented fact. **The `is_depression` column is a
dataset label, not a clinical diagnosis.** No clinician annotation process
is referenced anywhere in this repository, and none should be assumed.

**Provenance and licensing:** exact dataset provenance and licensing
information are not currently documented in this repository. The
column name `clean_text` implies the dataset was pre-processed by its
original source/authors before being placed in this repository, but the
exact collection methodology, licensing terms, and original annotation
process are not recorded here.

## 4. Dataset Limitations

- The dataset content is consistent with a single platform/domain
  (Reddit-style community text); no cross-platform data is present.
- Labels appear to be source-context-derived rather than individually
  clinician-verified (Section 3) — this affects label reliability at the
  level of any individual row.
- The project's error analysis
  (`results/error_analysis/ERROR_ANALYSIS_REPORT.md`) contains
  representative examples consistent with possible label noise — for
  instance, a test row labeled `depressed` whose text consists only of a
  motivational quote. This is reported there as an **observed indication**
  from a small number of illustrative examples, not as a systematic,
  dataset-wide audit — no claim is made here about what proportion of the
  dataset is affected.
- No demographic metadata is present in the dataset, and no demographic
  analysis has been performed on it.
- The dataset is single-language (English).
- No external, independently sourced validation dataset has been used
  anywhere in this project's evaluation.

## 5. Original V1 Data/Evaluation Workflow

The original (v1) depression workflow, as implemented in
`src/train_depression_bert.py` and the pre-correction version of
`experiments/evaluate_models.py`, was:

```
data/depression_dataset.csv
  -> train_test_split(test_size=0.1, random_state=42)   (src/train_depression_bert.py)
  -> model trained on the resulting ~90% training portion
  -> depression_bert_model/ saved

experiments/evaluate_models.py (original version)
  -> pd.read_csv("data/depression_dataset.csv")   (the FULL, unsplit file)
  -> every one of the 7731 rows scored against depression_bert_model/
  -> accuracy = 0.9933
```

*Verified by inspecting `src/train_depression_bert.py` (`test_size=0.1`
train/validation split logic) and the original `experiments/evaluate_models.py`
as it existed in git history prior to correction
(`git show HEAD:experiments/evaluate_models.py`), which loads the entire
dataset with no split applied.*

**The critical issue:** the model being evaluated had already been
trained on approximately 90% of the exact rows it was later scored
against (matching the `test_size=0.1` split used during its training).
The evaluation set was therefore not independent of the training data.

This is a **historical leakage-affected evaluation result**. The reported
0.9933 accuracy is not a valid held-out generalization estimate — it
reflects the model's performance on data it had substantially already
seen. It should never be described as "real performance" or cited as the
project's official result. See
[`results/legacy_full_dataset_eval/README.md`](../results/legacy_full_dataset_eval/README.md)
for the preserved record.

## 6. How the Leakage Was Identified

The issue was identified by direct inspection of the evaluation code:

- `experiments/evaluate_models.py` (original version) loaded the entire
  `data/depression_dataset.csv` with a single `pd.read_csv` call and no
  train/test filtering.
- `depression_bert_model/`, the model being evaluated, had previously
  been produced by `src/train_depression_bert.py`, which fit the model on
  a `train_test_split(test_size=0.1, ...)` partition of that same file.
- Because evaluation used the full file and training used ~90% of the
  same file, the evaluation set necessarily included the majority of the
  training set.

This document does not claim an exact, independently re-measured
overlap percentage beyond what is directly derivable from the `test_size`
parameter in the training script (i.e., training used approximately 90%
of the dataset, so the fully-overlapping evaluation set includes at least
that fraction of previously-seen rows).

## 7. Canonical V2 Data Preparation

```
raw dataset (data/depression_dataset.csv)
  -> duplicate detection/removal (on clean_text)
  -> deterministic, stratified split (seed 42)
  -> train / validation / frozen test
```

| Quantity | Value |
| --- | --- |
| Raw rows | 7731 |
| Duplicate `clean_text` rows removed | 81 |
| Unique rows after deduplication | 7650 |
| Train | 5354 |
| Validation | 1148 |
| Test | 1148 |
| Split seed | 42 |

*Verified against `results/splits/depression_split_manifest.json`
(`raw_row_count: 7731`, `duplicates_removed: 81`, `deduped_row_count: 7650`,
`split_counts: {train: 5354, val: 1148, test: 1148}`, `random_seed: 42`).*

## 8. Deduplication

Deduplication is implemented in `src/depression_split.py`
(`remove_duplicates`): rows are considered duplicates if they have an
**exactly identical `clean_text` value**, and duplicates are removed with
pandas' `drop_duplicates(subset=["clean_text"], keep="first")` — the
first occurrence of each unique text is kept, later occurrences are
dropped.

This is an **exact-string-match** method; it does not detect
near-duplicate or paraphrased text. 81 rows were removed by this process
(Section 7).

Deduplication is performed **before** the train/validation/test split is
constructed. This ordering matters: if deduplication were performed after
splitting, or not at all, a duplicate of a training-set row could still
appear in the validation or test set, which would reintroduce a form of
leakage (the model would be evaluated on text it had already seen, even
if under a different row index). Deduplicating first guarantees this
cannot happen, subject to the exact-match limitation above.

## 9. Canonical Split Method

Implemented in `src/depression_split.py` (`make_split`):

- **Random seed:** 42, passed to every `train_test_split` call.
- **Stratification:** by `is_depression`, applied at both split stages
  (`stratify=df["is_depression"]` for the test split,
  `stratify=train_val["is_depression"]` for the validation split).
- **Proportions:** the deduplicated dataset is split 70% train / 15%
  validation / 15% test. This is implemented as two sequential
  `train_test_split` calls: first a 15% test split off the full
  deduplicated set, then a validation split off the remaining 85%, using
  `val_fraction_of_remainder = 0.15 / (1 - 0.15)` so the final validation
  proportion of the *original* deduplicated set is 15%.
- **Determinism:** given the same input file and the same seed, this
  procedure produces byte-identical output — re-running
  `python -m src.depression_split` against an unchanged
  `data/depression_dataset.csv` reproduces the same three files.
- **Zero-overlap verification:** `assert_no_overlap` in
  `src/depression_split.py` checks that no `clean_text` value appears in
  more than one of the three resulting sets, and raises an assertion
  error if it does. This check passed when the canonical split was
  generated (`no_overlap_verified: true` in the manifest), and has been
  independently re-verified at multiple later points in this project's
  history by directly recomputing set intersections across the three
  split files.

**Why validation exists:** to provide data for model selection —
diagnostic performance reporting for the classical model, and early
stopping / best-checkpoint selection for the BERT model — without
touching the test set.

**Why the test set is kept separate:** to provide a final measurement
that has not influenced any modeling decision (architecture,
hyperparameter, checkpoint, or calibration-parameter choice), so that the
reported test metrics are a genuine held-out estimate rather than a
number the modeling process was, directly or indirectly, optimized
against.

## 10. Data Workflow Diagram

See [`docs/diagrams/data-workflow.mmd`](diagrams/data-workflow.mmd).

## 11. Train / Validation / Test Contract

| Split | Purpose | Used for fitting? | Used for model selection? | Used for final reporting? |
| --- | --- | --- | --- | --- |
| Train | Fit model parameters | Yes | No | No |
| Validation | Model/checkpoint selection, early stopping, temperature-scaling fit | No | Yes | No |
| Test | Final held-out measurement | No | No | Yes |

## 12. Data Leakage Prevention Rules

The following rules govern the canonical v2 depression data pipeline:

1. Deduplicate before splitting.
2. Split deterministically (fixed seed, recorded in the manifest).
3. Never fit a model on test data.
4. Never tune hyperparameters using test data.
5. Fit calibration parameters (temperature scaling) on validation data
   only.
6. Use the frozen test set only for final measurement.
7. Preserve the canonical split once established.
8. Do not regenerate the canonical test set casually.

These rules exist to preserve comparability across experiments: every
canonical evaluation, calibration, error-analysis, and multi-seed result
in this repository is measured against the identical 1148-row test set,
which is what makes those results directly comparable to one another.

## 13. Canonical Split Artifacts

| Artifact | Contents |
| --- | --- |
| `results/splits/depression_train.csv` | 5354 rows, `clean_text` + `is_depression`, used for fitting |
| `results/splits/depression_val.csv` | 1148 rows, used for model selection only |
| `results/splits/depression_test.csv` | 1148 rows, frozen, used for final reporting only |
| `results/splits/depression_split_manifest.json` | Seed, row counts, class balance, dedup count, overlap-verification record |

These four files constitute the canonical v2 split. They were generated
once by `python -m src.depression_split` and have not been regenerated
since.

## 14. Frozen Test Set Policy

**The canonical test set is treated as a frozen evaluation artifact.**

It must not be used to:

- train models,
- tune hyperparameters,
- select architectures,
- choose preprocessing steps,
- select training seeds,
- fit calibration parameters.

If a future experiment requires any of the above, the correct approach is
to use a separate validation protocol, or to construct a new,
distinctly-named experimental split — **not** to silently reuse or modify
the canonical test set described in this document.

## 15. V1 → V2 Correction

**V1:**
```
full dataset -> model trained on a ~90% subset -> full dataset evaluated
  -> 0.9933 (historical, invalid as a held-out estimate)
```

**V2:**
```
raw dataset -> deduplication -> canonical split
  -> train-only fitting -> validation-only selection -> frozen test
  -> official result: 0.9843 (BERT, held-out accuracy)
```

The v2 test accuracy (0.9843) is lower than the v1 figure (0.9933). **This
is not evidence that the model "got worse."** The v1 and v2 numbers are
not measuring the same thing: v1 measured performance on data
substantially already seen during training (Section 5–6); v2 measures
performance on data never seen during training or model selection
(Sections 7–9). The lower v2 number reflects a stricter, leakage-free
evaluation protocol, not a regression in model capability. This document
does not claim anything further about the two numbers' relationship
beyond what the evaluation protocols themselves support.

## 16. Why the Legacy Results Are Preserved

`results/legacy_full_dataset_eval/` is intentionally preserved rather
than deleted. Its purpose is auditability and historical traceability:
deleting the original leakage-affected metrics would remove the evidence
that this methodological problem existed and was found and corrected,
making the project's history harder to audit rather than cleaner.

**These historical metrics must never be used as official model
performance.** The directory's own
[`README.md`](../results/legacy_full_dataset_eval/README.md) states this
directly and should be treated as authoritative for that directory.

## 17. Data Reproducibility

The canonical split can be regenerated with:

```
python -m src.depression_split
```

Given an unchanged `data/depression_dataset.csv`, this reproduces the
same row counts, class balances, and (subject to library-version
stability of `sklearn.model_selection.train_test_split`'s seeded
behavior) the same row assignments documented in this file.

The full environment/reproduction guide is
[`docs/reproducibility.md`](reproducibility.md), which documents the
complete command sequence, expected runtimes, and troubleshooting for the
canonical split, v2 training, evaluation, calibration, error analysis, and
multi-seed workflow; this section is intentionally limited to the
data-regeneration command itself.

## 18. Data Governance / Provenance Gaps

The following information is not currently established by this
repository and should not be assumed:

- Exact original source/collection methodology for
  `data/depression_dataset.csv`.
- Licensing terms for the dataset.
- The original annotation/labeling process that produced the
  `is_depression` column.
- Any ethics or consent context for the original data collection.
- Demographic representativeness of the dataset — no demographic
  attributes are present, and none have been inferred or estimated.

## 19. Relationship to the Research Question

This repository currently establishes a rigorous, leakage-free benchmark
comparing TF-IDF + Logistic Regression against fine-tuned DistilBERT on
this specific dataset, under a fixed train/validation/test protocol. This
document describes that benchmark's data methodology; it does not
establish or argue for a novel research contribution. Whether this
project's results motivate a research question worth pursuing further is
a separate decision, to be made after the software/model project itself
is complete — see the project's development roadmap for that later phase.

## 20. Evidence and References

- Split generation code: [`src/depression_split.py`](../src/depression_split.py)
- Split manifest: [`results/splits/depression_split_manifest.json`](../results/splits/depression_split_manifest.json)
- Legacy evaluation record: [`results/legacy_full_dataset_eval/README.md`](../results/legacy_full_dataset_eval/README.md)
- Model-level summary: [`MODEL_CARD.md`](../MODEL_CARD.md)
- Responsible-use guidance: [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md)
- Reproduction guide: [`docs/reproducibility.md`](reproducibility.md)
