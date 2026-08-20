# Architecture

## 1. System Overview

This repository contains two related tracks:

**Runtime / application track** — the prediction and analysis code a user
actually invokes (`src/predict.py`, `src/predict_bert.py`,
`src/predict_depression_bert.py`, `src/mental_health_pipeline.py`, and the
trajectory/Monte Carlo modules built on top of it).

**Canonical research / evaluation track** — the corrected v2
depression-model methodology and its validation artifacts (the canonical
split, `depression_model_v2/`, `depression_bert_model_v2/`, and the
`experiments/*_v2.py` evaluation/calibration/error-analysis/multi-seed
scripts).

**For depression, these two tracks are now connected at the
model-loading boundary:** the runtime track loads the same canonical v2
depression artifacts the research track validated, via centralized path
constants in `src/model_config.py`. **Sentiment remains a separate,
unmigrated v1 artifact** on the runtime side — there is no canonical v2
sentiment track to connect it to. Detail in Section 12. See
[`MODEL_CARD.md`](../MODEL_CARD.md) §19 and
[`docs/evaluation.md`](evaluation.md) for the corresponding statements in
those documents. **Connecting the two tracks is a statement about which
weights are loaded, not a claim of clinical validity** — see
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 2. High-Level Architecture

**Runtime track:**
```
data (at training time, historical)
  -> preprocessing
  -> v1 models
  -> inference (mental_health_pipeline.py)
  -> analysis (signals, risk heuristic, trajectory, Monte Carlo)
```

**Canonical research track:**
```
data/depression_dataset.csv
  -> canonical split (src/depression_split.py)
  -> v2 training (train-only fitting)
  -> official evaluation (frozen test)
  -> calibration
  -> error analysis
  -> multi-seed stability
```

The second of these is the validated research track described in
[`MODEL_CARD.md`](../MODEL_CARD.md) and [`docs/evaluation.md`](evaluation.md).
The first is what currently executes when the repository's CLI tools are
run.

## 3. Architecture Diagram

See [`docs/diagrams/architecture-overview.mmd`](diagrams/architecture-overview.mmd).

## 4. Repository Layers

| Layer | Main location | Responsibility |
| --- | --- | --- |
| Data | `data/` | Raw datasets |
| Splitting | `src/depression_split.py` | Canonical deterministic split |
| Training (canonical) | `src/train_depression_v2.py`, `src/train_depression_bert_v2.py` | Canonical model training |
| Training (legacy) | `src/train_depression.py`, `src/train_depression_bert.py`, `src/train_sentiment.py`, `src/train_sentiment_bert.py` | Original ad hoc training |
| Runtime inference | `src/predict.py`, `src/predict_bert.py`, `src/predict_depression_bert.py` | User-facing prediction (v2 depression, v1 sentiment, Section 12) |
| Combined pipeline | `src/mental_health_pipeline.py` | Combined sentiment + depression + signal + risk analysis (v2 depression, v1 sentiment, Section 12) |
| Signals | `src/mental_health_signals.py` | Hand-curated keyword/phrase text signals |
| Trajectory | `src/mental_health_trajectory.py` | Sequential analysis over multiple texts |
| Monte Carlo | `src/monte_carlo_simulation.py` | Perturbation sensitivity experiment |
| Evaluation | `experiments/evaluate_models.py` | Official held-out metrics (v2) |
| Calibration | `experiments/calibration_analysis_depression_v2.py` | Probability calibration (v2) |
| Error analysis | `experiments/error_analysis_depression_v2.py` | Frozen-prediction characterization (v2) |
| Stability | `experiments/multiseed_depression_v2.py` | Multi-seed analysis (v2) |
| Results | `results/` | Persisted evaluation artifacts |
| Documentation | `docs/` | Technical documentation |

*Verified directly against the current repository tree
(`src/`, `experiments/`, `results/`, `docs/`).*

## 5. Data Flow

```
data/depression_dataset.csv
  -> deduplication (81 rows removed)
  -> canonical split (seed 42)
  -> train (5354) / validation (1148) / test (1148)
  -> v2 training (train-only fitting)
  -> frozen evaluation (test, single use)
```

The detailed data methodology — deduplication method, split proportions,
leakage discovery and correction — is owned by
[`docs/data.md`](data.md) and is not repeated here.

## 6. Model Layer

Two canonical v2 models exist:

- **Classical:** TF-IDF + Logistic Regression (`depression_model_v2/`).
- **BERT:** fine-tuned `distilbert-base-uncased` sequence classifier
  (`depression_bert_model_v2/`).

Full architecture and training configuration are owned by
[`docs/models.md`](models.md) and are not reproduced here.

V1 depression models (`depression_model.pkl`, `depression_vectorizer.pkl`,
`depression_bert_model/`) and the sentiment models
(`sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `sentiment_bert_model/`)
remain in the repository as historical/legacy artifacts — see Section 11.

## 7. Runtime Inference Architecture

This traces the **actual current behavior** of
`src/mental_health_pipeline.py`'s `analyze_text()`, read directly from
source rather than assumed:

```
user text
  -> mental_health_signals.detect_signals(text)      [runs first]
  -> sentiment_bert_model/     (v1, tokenize -> forward pass -> softmax -> argmax)
  -> depression_bert_model_v2/ (v2, tokenize -> forward pass -> softmax -> argmax)
  -> compute_risk(sentiment_label, depression_label)  [runs after both labels exist]
  -> result dict {text, sentiment, mental_health, risk_level, signals_detected}
```

Note the actual order: signal detection runs **before** either model
inference call, not after (the code comment in
`mental_health_pipeline.py` — "RISK (must be AFTER depression is
defined)" — confirms `compute_risk` is deliberately sequenced last). Spell
correction, which was previously applied before model inference in this
pipeline, has been removed (see [`docs/models.md`](models.md) §5).

Both model instances (`sent_model`, `dep_model`) are loaded once at
**module import time**, not per call. `sentiment_model_path =
"sentiment_bert_model"` remains a v1 model directory, hardcoded directly
in `src/mental_health_pipeline.py`. `depression_model_path` now resolves
to `DEPRESSION_BERT_MODEL_PATH` from `src/model_config.py`, which points
to `depression_bert_model_v2/` — the canonical v2 model.

**Other runtime entry points and what they currently load:**

| File | Loads |
| --- | --- |
| `src/predict.py` | `depression_model_v2/model.pkl` / `depression_model_v2/vectorizer.pkl` (v2 classical, via `src/model_config.py` + `joblib.load`), plus `sentiment_model.pkl` / `tfidf_vectorizer.pkl` (v1, unchanged) |
| `src/predict_bert.py` | `"sentiment_bert_model"` (v1, unchanged — no depression coupling) |
| `src/predict_depression_bert.py` | `depression_bert_model_v2/` (v2, via `src/model_config.py`) |

All three depression-loading sites now resolve through the constants in
`src/model_config.py` rather than a hardcoded literal; sentiment paths
are untouched and remain hardcoded v1 literals in each file, since
sentiment has no canonical v2 track. Verified by direct inspection of
each file's model-path assignment, not inferred.

## 8. Inference Diagram

See [`docs/diagrams/inference-pipeline.mmd`](diagrams/inference-pipeline.mmd).
**This diagram predates the runtime migration described in Section 7 and
still depicts the depression model as v1/"not wired in"; it has not been
redrawn as part of this migration and is a known, pending documentation
follow-up.** The prose in Section 7 above is the current source of truth
for runtime model loading.

## 9. Research / Evaluation Architecture

```
canonical split -> train / validation / test
  -> v2 classical + v2 BERT (train-only fitting, validation-only selection)
  -> official test evaluation (frozen, single use)
  -> calibration (temperature fit on validation only)
  -> error analysis (descriptive, on frozen predictions)
  -> multi-seed stability (5 independent retrainings, same frozen test set)
```

Test data is not part of fitting or model selection at any stage of this
track — see [`docs/evaluation.md`](evaluation.md) §2 for the complete
stage-by-stage train/validation/test usage table. Full methodology:
[`docs/data.md`](data.md), [`docs/models.md`](models.md),
[`docs/evaluation.md`](evaluation.md).

## 10. Artifact Ownership

| Artifact | Location |
| --- | --- |
| Canonical split | `results/splits/` |
| Canonical classical model | `depression_model_v2/` |
| Canonical BERT model | `depression_bert_model_v2/` |
| Official evaluation | `results/` (root-level `depression_v2_*` and `metrics_depression_v2_*`/`predictions_depression_v2_*` files) |
| Calibration | `results/calibration/` |
| Error analysis | `results/error_analysis/` |
| Multi-seed | `results/multiseed/` |
| Legacy evidence | `results/legacy_full_dataset_eval/` |
| Disposable multi-seed weights | `results/multiseed/tmp_models/` |

`results/multiseed/tmp_models/` is **disposable** — it holds per-seed
BERT checkpoints produced during the multi-seed stability study and is
excluded from version control. It must not be confused with the
canonical `results/multiseed/` CSV/JSON/Markdown/PNG outputs, which are
the actual canonical multi-seed results and are retained.

## 11. V1 and V2 Separation

**V1:** original project generation; ad hoc, per-script train/test
splitting; the original runtime models (Section 7); a historical
evaluation that scored a model against data it had already substantially
seen, producing a leakage-affected metric
(`results/legacy_full_dataset_eval/`); retained for historical
reproducibility and transparency, not deleted.

**V2:** canonical deduplicated split; train-only fitting; validation-only
model selection; frozen test evaluation; calibration; error analysis;
multi-seed stability; the canonical research results referenced
throughout [`MODEL_CARD.md`](../MODEL_CARD.md).

The important distinction between v1 and v2 is **methodology and
validation status**, not architecture — both generations of the
depression models use the same underlying algorithms (TF-IDF+Logistic
Regression; fine-tuned DistilBERT). V1 is not "broken" or generally
useless; it is simply not evaluated under a leakage-free protocol and
must not be cited as the project's official performance. Full narrative:
[`docs/data.md`](data.md).

## 12. Runtime Model Migration (Depression)

**The validated v2 depression models now exist, have been evaluated, and
are the default runtime depression models.** The runtime/research gap
previously documented in this section for depression has been closed via
a minimal path migration: `src/model_config.py` centralizes the v2
depression paths, and the three files that load a depression model
import them from there instead of hardcoding v1 paths.

Migrated files:
- `src/mental_health_pipeline.py`
- `src/predict_depression_bert.py`
- `src/predict.py`

`src/mental_health_trajectory.py` and `src/monte_carlo_simulation.py`
both call `analyze_text` from `src/mental_health_pipeline.py` (verified
by direct import inspection), and therefore inherit its v2 depression
model selection without any change to those two files themselves.

**Sentiment inference (`src/predict_bert.py`, and the sentiment half of
`src/predict.py` / `src/mental_health_pipeline.py`) was explicitly out of
scope and remains on the original v1 artifacts** — there is no canonical
v2 sentiment retraining track for it to migrate to.

**This migration changes which model weights are loaded. It does not
validate `compute_risk()`, trajectory analysis, or the Monte Carlo
experiment, and it does not constitute clinical validation of any kind**
— see [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) and Section 13
below, both unchanged by this migration.

## 13. Safety-Critical Architectural Boundary

Three distinct things exist in this pipeline and must not be conflated:

- **Model prediction** — BERT/classical model output is a benchmark
  classification label with an associated probability, evaluated on this
  project's specific dataset (Section 9).
- **Risk heuristic** — `mental_health_signals.detect_signals()` is
  hand-curated keyword/phrase matching; `compute_risk()` is an
  unvalidated, hand-written combination rule over the sentiment and
  depression labels.
- **Clinical assessment** — none of the above constitutes one. Trajectory
  analysis builds a trend on top of the risk heuristic; Monte Carlo
  analysis measures the sensitivity of the combined heuristic pipeline to
  input perturbation, not a certified robustness or clinical property.

Model prediction ≠ risk heuristic ≠ clinical assessment. Full guidance:
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 14. Design Decisions

**Why v2 exists:** to correct the leakage problem identified in the
original evaluation and establish a canonical, frozen, held-out test
evaluation (Section 11, [`docs/data.md`](data.md)).

**Why the test split is persisted:** to prevent accidental regeneration
and evaluation drift — every canonical result in this repository measures
against the identical, unchanging 1148-row test file.

**Why v1 artifacts are retained:** historical reproducibility and
transparency; deleting them would remove the evidence that the leakage
problem existed and was corrected.

**Why large BERT weights are gitignored:** git safety and repository
size — multi-hundred-megabyte binary weight files are not suited to
normal git tracking.

**Why canonical lightweight results remain trackable:** the small
CSV/JSON/Markdown/PNG outputs under `results/` provide reproducibility
evidence (configuration, metrics, predictions) without requiring the
multi-hundred-MB model weights themselves to be stored in git.

## 15. Known Architectural Gaps

- Depression runtime now points to v2 (Section 12); sentiment runtime
  remains v1 and unmigrated (no canonical v2 sentiment track exists).
- `pipeline.sh` is not yet aligned with the canonical v2 workflow — it
  does not invoke `src/depression_split.py` or any `experiments/*_v2.py`
  script.
- A minimal automated test suite exists (`tests/`, added in Phase D) and currently passes
  (36 passed, 2 skipped) — see [`development.md`](development.md) §15 for coverage scope.
- No CI exists yet.
- `app.py` is empty.
- `experiments/plot_confusion_matrix.py` and `experiments/plot_monte_carlo.py`
  previously referenced stale result-file paths; both were fixed to read
  from current, canonical/expected paths. `plot_confusion_matrix.py` runs
  immediately against the existing canonical v2 prediction CSVs.
  `plot_monte_carlo.py` requires `results/monte_carlo_results.csv`,
  produced by `experiments/run_monte_carlo_batch.py`, which has not yet
  been run in this repository; it now fails with an explicit, actionable
  error rather than an unexplained traceback if that file is missing.
- Model-loading boilerplate (tokenizer + model + device setup) is
  duplicated across `src/predict_bert.py`, `src/predict_depression_bert.py`,
  and `src/mental_health_pipeline.py`.
- `src/text_preprocessing.py` (spell-correction utility) is currently
  unused — not imported by any other file in the repository.
- A minimal depression model-path configuration layer now exists
  (`src/model_config.py`) — `predict.py`, `predict_depression_bert.py`,
  and `mental_health_pipeline.py` import their depression paths from it.
  It is intentionally a bare constants module, not a shared model
  *loader* — each file still contains its own `joblib.load` /
  `from_pretrained` calls. Sentiment paths are still independently
  hard-coded in each file, since sentiment has no v2 track to
  centralize toward.

This section is a status record, not a task list to execute here.

## 16. Runtime Model Configuration

`src/model_config.py` is a small, dependency-free module containing only
plain path constants for the canonical v2 depression artifacts:

```
src/model_config.py (path constants only)
  -> src/predict.py                  (classical depression paths)
  -> src/predict_depression_bert.py  (BERT depression path)
  -> src/mental_health_pipeline.py   (BERT depression path)
  -> analysis consumers (trajectory, Monte Carlo) inherit via mental_health_pipeline.py
```

...while the research/evaluation track (Section 9) continues to operate
independently, reading the same underlying `depression_model_v2/` /
`depression_bert_model_v2/` artifacts directly rather than through this
module (the evaluation scripts predate `model_config.py` and were not
modified by this migration). This is deliberately minimal: it is a
constants module, not a generalized model-loading framework, an
argparse model-version selector, or an environment-variable-driven
configuration system — none of those were judged necessary for the
scope of this migration.

## 17. Related Documentation

- [`MODEL_CARD.md`](../MODEL_CARD.md)
- [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md)
- [`docs/data.md`](data.md)
- [`docs/models.md`](models.md)
- [`docs/evaluation.md`](evaluation.md)
- [`docs/reproducibility.md`](reproducibility.md)
- [`docs/development.md`](development.md)
