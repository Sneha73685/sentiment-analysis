# Model Architecture and Training

## 1. Scope

This document describes the model implementations in this repository: what
each model is, how it was trained, what configuration it uses, and which
artifacts it produces. It is an implementation-level reference, not a
performance summary and not a research discussion.

- The sentiment models (classical and BERT) are a separate legacy
  component, unrelated to the depression-classification methodology
  described in [`docs/data.md`](data.md).
- The **v2 depression models** are the canonical research models for this
  project.
- The **v1 depression models** are preserved as historical/legacy
  artifacts and are not part of the canonical benchmark.

For performance results, calibration, error analysis, and multi-seed
stability, see [`MODEL_CARD.md`](../MODEL_CARD.md) — this document does
not repeat those numbers.

## 2. Model Generations

| Generation | Component | Architecture | Status |
| --- | --- | --- | --- |
| V1 | Classical sentiment | TF-IDF + Logistic Regression | Legacy, unchanged |
| V1 | BERT sentiment | Fine-tuned `distilbert-base-uncased` | Legacy, unchanged |
| V1 | Classical depression | TF-IDF + Logistic Regression | Legacy, superseded — evaluation was leakage-affected |
| V1 | BERT depression | Fine-tuned `distilbert-base-uncased` | Legacy, superseded — evaluation was leakage-affected |
| V2 | Classical depression | TF-IDF + Logistic Regression | **Canonical** |
| V2 | DistilBERT depression | Fine-tuned `distilbert-base-uncased` | **Canonical** |

**"Canonical" in this repository refers exclusively to the two v2
depression models.** No sentiment model, and no v1 depression model, is
canonical. V1 depression model performance figures must not be presented
as current performance — see [`docs/data.md`](data.md) §5–6 and §15 for
why the v1 evaluation was invalid.

## 3. Classical Depression Model (v2)

Implemented in `src/train_depression_v2.py`.

```
raw text -> clean_text preprocessing -> TF-IDF -> LogisticRegression -> binary label
```

| Component | Configuration | Source |
| --- | --- | --- |
| Vectorizer | `TfidfVectorizer(max_features=5000)` | `depression_model_v2/training_config.json` (`vectorizer_params`) |
| Classifier | `LogisticRegression(max_iter=1000, class_weight="balanced")` | `depression_model_v2/training_config.json` (`model_params`) |
| Solver | Not explicitly configured — uses scikit-learn's default (`lbfgs`) | Absent from `model_params`; confirmed by inspecting `src/train_depression_v2.py`, which does not pass a `solver` argument |
| Random seed | 42 | `depression_model_v2/training_config.json` (`random_seed`) |

**Split usage:** fit on the training split only
(`results/splits/depression_train.csv`, 5354 rows); the validation split
(`results/splits/depression_val.csv`, 1148 rows) is used only for
diagnostic performance reporting after fitting — there is no
hyperparameter search in this script, so validation does not currently
drive any selection decision, only visibility. The test split is not
loaded by this script at all.

## 4. BERT Depression Model (v2)

Implemented in `src/train_depression_bert_v2.py`.

```
raw text -> DistilBertTokenizerFast -> token IDs / attention mask
  -> distilbert-base-uncased encoder -> sequence classification head -> 2-class logits
```

| Component | Configuration | Source |
| --- | --- | --- |
| Base checkpoint | `distilbert-base-uncased` | `depression_bert_model_v2/training_config.json` (`base_model`) |
| Tokenizer | `DistilBertTokenizerFast` | `src/train_depression_bert_v2.py` |
| Max sequence length | 256 | `training_config.json` (`max_length`) |
| Number of labels | 2 | `depression_bert_model_v2/config.json` (`architectures`, `id2label` has 2 entries) |
| Batch size (train/eval) | 4 / 4 | `training_config.json` (`training_args.per_device_train_batch_size`, `per_device_eval_batch_size`) |
| Learning rate | 2e-05 | `training_config.json` (`training_args.learning_rate`) |
| Weight decay | 0.01 | `training_config.json` (`training_args.weight_decay`) |
| Gradient accumulation | 2 | `training_config.json` (`training_args.gradient_accumulation_steps`) |
| Maximum epochs | 4 | `training_config.json` (`training_args.num_train_epochs`) |
| Early stopping | patience = 1, on validation accuracy | `training_config.json` (`early_stopping_patience`), `src/train_depression_bert_v2.py` (`EarlyStoppingCallback`, `metric_for_best_model="accuracy"`) |
| Random seed | 42 | `training_config.json` (`random_seed`) |
| Validation strategy | Evaluated once per epoch | `src/train_depression_bert_v2.py` (`eval_strategy`/`evaluation_strategy` set to `"epoch"`) |
| Best-model restoration | `load_best_model_at_end=True` | `src/train_depression_bert_v2.py` |
| `id2label` | `{"0": "not_depressed", "1": "depressed"}` | `depression_bert_model_v2/config.json` and `training_config.json` |
| `label2id` | `{"not_depressed": 0, "depressed": 1}` | `depression_bert_model_v2/config.json` and `training_config.json` |

Underlying DistilBERT architecture dimensions, as recorded in
`depression_bert_model_v2/config.json`: 6 transformer layers (`n_layers`),
12 attention heads (`n_heads`), hidden dimension 768 (`dim`), feed-forward
dimension 3072 (`hidden_dim`), GELU activation, dropout 0.1, sequence
classification head dropout 0.2 (`seq_classif_dropout`), vocabulary size
30522. These are the standard `distilbert-base-uncased` architecture
values, unmodified by this project's fine-tuning process.

## 5. Input Preprocessing

**Classical v2:**
```
raw text -> src.preprocess.clean_text -> TF-IDF
```
`clean_text` (lowercasing, URL removal, non-letter character removal,
tokenization, stopword removal, lemmatization) is applied identically at
training time (`src/train_depression_v2.py`) and at evaluation time
(`experiments/evaluate_models.py`).

**BERT v2:**
```
raw text -> DistilBertTokenizerFast (truncation, max_length=256) -> model
```
No additional text normalization is applied before tokenization, at
either training or evaluation time.

**BERT v2 does not use spell-correction preprocessing.** An earlier
version of the combined runtime pipeline
(`src/mental_health_pipeline.py`) applied a spellchecker-based correction
step (`src/text_preprocessing.py`) before BERT inference. This has since
been removed from that pipeline, because the BERT depression model was
never trained on spell-corrected text — applying it only at inference
time would have created a train/serve preprocessing mismatch (the model
would see input systematically different from its training distribution).
Canonical v2 training and evaluation have never used spell correction at
any point, in either direction.

For the broader data methodology (deduplication, canonical split), see
[`docs/data.md`](data.md).

## 6. Training Pipeline Diagram

See [`docs/diagrams/training-pipeline.mmd`](diagrams/training-pipeline.mmd).

## 7. Training Configuration

**Classical (`depression_model_v2/training_config.json`):**

| Field | Value |
| --- | --- |
| `script` | `src/train_depression_v2.py` |
| `model_type` | `TfidfVectorizer + LogisticRegression` |
| `train_path` | `results/splits/depression_train.csv` |
| `val_path` | `results/splits/depression_val.csv` |
| `test_path_used` | `null` |
| `random_seed` | 42 |
| `vectorizer_params` | `{"max_features": 5000}` |
| `model_params` | `{"max_iter": 1000, "class_weight": "balanced"}` |
| `train_rows` | 5354 |
| `val_rows` | 1148 |
| `git_commit` | `d68b419e10ded167b87c99f3533c16b5209851d1` |

**BERT (`depression_bert_model_v2/training_config.json`):**

| Field | Value |
| --- | --- |
| `script` | `src/train_depression_bert_v2.py` |
| `base_model` | `distilbert-base-uncased` |
| `train_path` | `results/splits/depression_train.csv` |
| `val_path` | `results/splits/depression_val.csv` |
| `test_path_used` | `null` |
| `random_seed` | 42 |
| `max_length` | 256 |
| `training_args.per_device_train_batch_size` | 4 |
| `training_args.per_device_eval_batch_size` | 4 |
| `training_args.num_train_epochs` | 4 |
| `training_args.learning_rate` | 2e-05 |
| `training_args.weight_decay` | 0.01 |
| `training_args.gradient_accumulation_steps` | 2 |
| `early_stopping_patience` | 1 |
| `train_rows` | 5354 |
| `val_rows` | 1148 |
| `git_commit` | `d68b419e10ded167b87c99f3533c16b5209851d1` |

The two configs record different fields where their underlying training
procedures differ (e.g. the classical config has `vectorizer_params` and
`model_params`; the BERT config has `training_args` and
`early_stopping_patience`) — this difference is preserved above rather
than forced into a single shared schema, since it accurately reflects two
different training procedures.

Both configs additionally record `id2label`/`label2id` (Section 9) and,
for the classical model, a full validation classification report
(`val_accuracy`, `val_classification_report`); the BERT config records
`final_validation_metrics` (`eval_loss`, `eval_accuracy`, and timing
fields). These are not reproduced here in full — see the JSON files
directly.

## 8. Model Artifacts

**`depression_model_v2/`:**
| File | Contents |
| --- | --- |
| `model.pkl` | Fitted `LogisticRegression` object (joblib) |
| `vectorizer.pkl` | Fitted `TfidfVectorizer` object (joblib) |
| `training_config.json` | Recorded training configuration (Section 7) |

**`depression_bert_model_v2/`:**
| File | Contents |
| --- | --- |
| `config.json` | HuggingFace model configuration (architecture, `id2label`/`label2id`) |
| `model.safetensors` | Fine-tuned model weights |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json` | Tokenizer files |
| `training_args.bin` | Serialized HuggingFace `TrainingArguments` |
| `training_config.json` | Recorded training configuration (Section 7) |

Large binary weight files (`model.safetensors`, and any `.bin`/`.pt`/`.pth`
files) are excluded from version control by `.gitignore`. This project
does not recommend committing large BERT weight files to git. The
small, human-readable `training_config.json` (and, for BERT, `config.json`)
in each artifact directory carry the reproducibility metadata needed to
regenerate or audit the model, independent of whether the weight file
itself is version-controlled.

## 9. BERT Labels

Canonical mapping, used consistently in `depression_bert_model_v2/config.json`
and `depression_bert_model_v2/training_config.json`:

| ID | Label |
| --- | --- |
| 0 | `not_depressed` |
| 1 | `depressed` |

Both `id2label` and `label2id` are stored directly in the model
configuration, so that any code loading this model via
`DistilBertForSequenceClassification.from_pretrained(...)` inherits the
mapping from the artifact itself, rather than depending on an
undocumented numeric convention maintained separately in application
code.

## 10. Model Selection

**Classical:** fit on the training split; the validation split is used
for evaluation/reporting only (Section 3) — there is currently no
hyperparameter search for this script to select over.

**BERT:** the validation split is used for per-epoch evaluation, early
stopping (patience 1, Section 4), and — via
`load_best_model_at_end=True` — restoration of the checkpoint with the
best validation accuracy before the final model is saved.

**In both cases, the test split is not loaded by the training scripts at
all** (`test_path_used: null` in both `training_config.json` files,
Section 7) and remains untouched until the separate evaluation step
described in [`docs/data.md`](data.md) and [`MODEL_CARD.md`](../MODEL_CARD.md).

## 11. V1 → V2 Model Transition

The v1 and v2 depression models use the **same underlying architectures**
(TF-IDF + Logistic Regression; fine-tuned `distilbert-base-uncased`). The
transition from v1 to v2 was not primarily an architecture change — it
was a **methodological pipeline correction**:

```
V1: ad hoc per-script train/test split -> training -> later full-dataset evaluation
V2: canonical deduplicated split -> train-only fitting -> validation-based
    selection -> frozen test evaluation
```

See [`docs/data.md`](data.md) for the full leakage-discovery narrative.
The `_v2` naming convention on directories and scripts
(`depression_model_v2/`, `depression_bert_model_v2/`,
`src/train_depression_v2.py`, `src/train_depression_bert_v2.py`) exists
specifically to prevent accidental confusion between the historical (v1)
and canonical (v2) artifacts — a reader or a script cannot mistake one
for the other by directory name alone. V1 artifacts have not been
deleted; they remain on disk as `depression_model.pkl`,
`depression_vectorizer.pkl`, and `depression_bert_model/`.

## 12. Legacy Sentiment Models

The repository also contains a classical sentiment model
(`sentiment_model.pkl`, `tfidf_vectorizer.pkl`, trained by
`src/train_sentiment.py` on `data/amazon_cells_labelled.txt`,
`data/imdb_labelled.txt`, and `data/yelp_labelled.txt`) and a BERT
sentiment model (`sentiment_bert_model/`, trained by
`src/train_sentiment_bert.py` on `data/sentiment140.csv`).

**These were not rebuilt under the canonical v2 depression methodology.**
They use their own, separate ad hoc train/test splits (`test_size=0.2`
and `test_size=0.1` respectively, both with `random_state=42`, verified
directly in their training scripts) and have not been evaluated against
any leakage-free, deduplicated, frozen test protocol. They remain
entirely outside the canonical v2 depression benchmark described in
[`MODEL_CARD.md`](../MODEL_CARD.md) and must not be described as v2
models or as canonical.

## 13. Runtime Model Loading

**As of this document, depression-model loading sites use the canonical
v2 artifacts, via `src/model_config.py`:**

| File | Loads |
| --- | --- |
| `src/mental_health_pipeline.py` | `"sentiment_bert_model"` (v1, hardcoded, unchanged) and `depression_bert_model_v2/` (v2, via `src/model_config.py`) |
| `src/predict_depression_bert.py` | `depression_bert_model_v2/` (v2, via `src/model_config.py`) |
| `src/predict_bert.py` | `"sentiment_bert_model"` (v1, hardcoded, unchanged — sentiment only, no depression coupling) |
| `src/predict.py` | `depression_model_v2/model.pkl` / `depression_model_v2/vectorizer.pkl` (v2, via `src/model_config.py` + `joblib.load`); sentiment paths unchanged |

`src/mental_health_trajectory.py` and `src/monte_carlo_simulation.py` both
call `analyze_text` from `src/mental_health_pipeline.py`, and therefore
inherit the v2 depression model loading described above without any
change to those two files.

**Sentiment was explicitly out of scope for this migration and remains
on its v1 artifacts** — there is no canonical v2 sentiment track.

| | Directories |
| --- | --- |
| Research canonical models | `depression_model_v2/`, `depression_bert_model_v2/` |
| Current runtime depression paths | `depression_model_v2/model.pkl`, `depression_model_v2/vectorizer.pkl`, `depression_bert_model_v2/` |
| Current runtime sentiment paths (unchanged) | `sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `sentiment_bert_model/` |

See [`MODEL_CARD.md`](../MODEL_CARD.md) §19 ("Runtime Status") for the
corresponding statement in the model-level summary. This migration
changes which weights are loaded; it is not a claim of clinical
validation — see [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 14. Model Loading and Versioning Risk

Depression model paths are now centralized in `src/model_config.py` and
imported by `src/predict.py`, `src/predict_depression_bert.py`, and
`src/mental_health_pipeline.py`, resolving the single-source-of-truth
risk this section previously documented **for depression paths**.
Sentiment paths remain hard-coded as string literals, independently, in
`src/predict.py`, `src/predict_bert.py`, and
`src/mental_health_pipeline.py` — this residual risk is unchanged, since
sentiment was out of scope for this migration and has no v2 track to
centralize toward. Combined with the continued coexistence of v1 and v2
artifact directories (Section 2) and multiple BERT checkpoint
subdirectories (`depression_bert/`, `depression_bert_v2/`), a future edit
to one sentiment loading site still would not propagate to the others.

`src/model_config.py` is intentionally a bare constants module (no
loading logic, no third-party imports) — it is not a shared model
*loader*; each file still contains its own `joblib.load` /
`from_pretrained` calls using the imported path constants.

## 15. Reproducibility

- Training configuration is stored alongside each v2 model artifact
  (`training_config.json`, Section 7).
- The canonical split is persisted (`results/splits/`, see
  [`docs/data.md`](data.md)).
- The training seed (42) is recorded in both training configs.
- The git commit at training time is recorded in both training configs
  (`d68b419e10ded167b87c99f3533c16b5209851d1`).
- Large binary model weights are intentionally excluded from normal git
  tracking (Section 8); the configuration files that make the model
  reproducible are not.

The full environment/reproduction guide is
[`docs/reproducibility.md`](reproducibility.md), which documents the exact
commands, expected runtimes, and environment setup needed to retrain and
re-evaluate these models.

## 16. Known Engineering Limitations

- No unified model loader — each prediction/pipeline script independently
  hard-codes its model path (Section 14).
- BERT tokenizer/model loading boilerplate is duplicated near-identically
  across `src/predict_bert.py`, `src/predict_depression_bert.py`, and
  `src/mental_health_pipeline.py`.
- No type hints anywhere in the model-loading or training code.
- No automated tests exist yet for training or inference code paths.
- Training configuration is partly defined directly in each script (e.g.
  `TfidfVectorizer(max_features=5000)` is a literal in
  `src/train_depression_v2.py`, not read from an external config file) —
  the `training_config.json` files record what was used after the fact,
  they do not drive the training run.
- V1 and V2 artifact coexistence (Section 2, Section 13) creates ongoing
  maintenance risk until the runtime loading gap is resolved.

## 17. References to Repository Evidence

- [`src/train_depression_v2.py`](../src/train_depression_v2.py)
- [`src/train_depression_bert_v2.py`](../src/train_depression_bert_v2.py)
- [`depression_model_v2/training_config.json`](../depression_model_v2/training_config.json)
- [`depression_bert_model_v2/training_config.json`](../depression_bert_model_v2/training_config.json)
- [`depression_bert_model_v2/config.json`](../depression_bert_model_v2/config.json)
- [`docs/data.md`](data.md)
- [`docs/reproducibility.md`](reproducibility.md)
- [`MODEL_CARD.md`](../MODEL_CARD.md)
- [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md)
