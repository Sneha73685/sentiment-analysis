# Reproducibility Guide

This is the operational guide for reproducing the canonical **v2** depression-classification
research workflow: environment setup, the exact commands, the order they must run in, what each
command touches, and the limits of what "reproduction" means for this project.

It intentionally does not repeat the full technical explanations already written in
[`MODEL_CARD.md`](../MODEL_CARD.md), [`docs/data.md`](data.md), [`docs/models.md`](models.md),
and [`docs/evaluation.md`](evaluation.md) — those are linked from each relevant section instead.

## 1. Scope

This guide covers reproduction of the **canonical v2 research workflow only**:

- generating the canonical data split,
- training the classical (TF-IDF + Logistic Regression) and BERT (DistilBERT) v2 models,
- running the official held-out evaluation, calibration analysis, error analysis, and
  multi-seed stability analysis.

It does **not** cover the v1 models, the legacy leakage-affected evaluation (documented for
audit purposes only — see [Section 18](#18-historical-leakage-correction)), or the runtime
inference layer's interactive CLI — which now loads the v2 depression models but is not how
the official metrics in this guide are produced (see
[Section 20](#20-current-runtime-vs-research-workflow)).

## 2. Environment

| Property | Value |
| --- | --- |
| Python version | 3.10 |
| Conda environment name (canonical) | `bert310` |
| Hardware used for canonical runs | Apple Silicon Mac |
| BERT training backend used for canonical runs | MPS |
| CPU fallback | Present in code (`torch.backends.mps.is_available()` checks throughout `src/` and `experiments/`) |
| CUDA | Not used for any canonical run in this repository |

**The exact historical `conda create` command used to build the `bert310` environment is not
recorded anywhere in this repository.** No `environment.yml`, `environment.yaml`, or complete
dependency lockfile exists. Do not assume a specific `conda create` invocation — none is
documented, and inventing one here would misrepresent what is actually known.

What **is** verified and repeatable is the package-installation step into an existing `bert310`
environment (Section 4).

## 3. Hardware and Runtime Expectations

These are the canonical run's observed conditions, not guarantees for other hardware:

| Step | Approximate time (canonical run, Apple Silicon MPS) |
| --- | --- |
| Classical training (`src/train_depression_v2.py`) | Seconds — a single `TfidfVectorizer` fit + `LogisticRegression.fit` over 5,354 rows |
| BERT training (`src/train_depression_bert_v2.py`) | ~7 minutes 50 seconds |
| Official evaluation (`experiments/evaluate_models.py`) | Not separately benchmarked; classical inference is near-instant, BERT inference is a single forward pass (no gradients) over 1,148 test rows in batches of 16, so it is much faster than training |
| Calibration analysis (`experiments/calibration_analysis_depression_v2.py`) | Not separately benchmarked; dominated by two BERT forward passes (val + test) plus a fast NumPy-based bootstrap (999 resamples) |
| Error analysis (`experiments/error_analysis_depression_v2.py`) | Seconds — operates only on already-computed prediction CSVs, no model inference |
| Multi-seed analysis (`experiments/multiseed_depression_v2.py`) | ~36 minutes 55 seconds total (5 seeds × classical + BERT training + test-set inference) |

> Wall-clock time is hardware- and backend-dependent (Section 16). Do not treat the figures
> above as guarantees on different hardware.

## 4. Installation

Verified, repeatable installation step, assuming the `bert310` conda environment already
exists:

```bash
conda activate bert310
pip install -r requirements.txt
```

**This installs the pinned top-level Python packages into an already-existing environment. It
does not create the conda environment itself** (Section 2).

`requirements.txt` currently pins:

```text
torch==2.10.0
transformers==4.41.2
datasets==2.21.0
evaluate==0.4.6
accelerate==0.34.2
pandas==2.3.3
scikit-learn==1.7.2
nltk==3.9.2
joblib==1.5.3
numpy==2.2.6
pyspellchecker==0.9.0
matplotlib==3.10.9
seaborn==0.13.2
```

These are top-level, verified-working pins — **not a complete transitive lockfile.** Exact
versions of indirect dependencies (e.g. tokenizers, huggingface-hub, urllib3) are whatever `pip`
resolves at install time and are not separately pinned.

> **Note:** `experiments/calibration_analysis_depression_v2.py`,
> `experiments/error_analysis_depression_v2.py`, and `experiments/multiseed_depression_v2.py`
> all import `scipy` (`scipy.optimize.minimize_scalar`, `scipy.stats`). `scipy==1.15.3` is
> explicitly pinned in `requirements.txt`, so `pip install -r requirements.txt` installs the
> exact version these scripts were verified against.

## 5. Verify Repository State Before Running

Before running any step, confirm what already exists so you don't unintentionally overwrite
canonical artifacts:

```bash
ls results/splits/                 # canonical split, if already generated
ls depression_model_v2/            # canonical classical model, if already trained
ls depression_bert_model_v2/       # canonical BERT model, if already trained
```

If `results/splits/depression_test.csv` already exists and you did not intend to regenerate it,
**stop** — regenerating the split is a frozen-artifact change (Section 17) and should only be
done deliberately.

## 6. Generate the Canonical Split

```bash
python -m src.depression_split
```

- **Reads:** `data/depression_dataset.csv` (raw, gitignored — see
  [Section 19](#19-fresh-clone-limitations)).
- **Writes:** `results/splits/depression_train.csv`, `depression_val.csv`, `depression_test.csv`,
  and `depression_split_manifest.json`.
- **Does not touch any model.**
- **Does not touch the test set** (it produces it).
- Deduplicates on `clean_text` (81 rows removed), then performs a seed-42, label-stratified
  train/val/test split (70/15/15), asserting zero text overlap across splits before writing
  anything.
- Computationally cheap (a `pandas` read + two `sklearn.model_selection.train_test_split` calls).

Full methodology: [`docs/data.md`](data.md).

## 7. Train the Classical v2 Model

```bash
python -m src.train_depression_v2
```

- **Reads:** `results/splits/depression_train.csv` (fitting) and `depression_val.csv`
  (diagnostic reporting only — there is no hyperparameter search in this script, so validation
  does not change what gets fit).
- **Does not read** `depression_test.csv` at all — the path constant `TEST_PATH` does not even
  exist in this script.
- **Writes:** `depression_model_v2/model.pkl`, `depression_model_v2/vectorizer.pkl`,
  `depression_model_v2/training_config.json` (records seed, hyperparameters, and the val-set
  classification report).
- **Modifies models:** yes — this is the training step. It does not overwrite the v1
  `depression_model.pkl` / `depression_vectorizer.pkl`.
- Cheap: `TfidfVectorizer(max_features=5000)` + `LogisticRegression(max_iter=1000,
  class_weight="balanced")`.

Expected validation accuracy: **0.9486** (`depression_model_v2/training_config.json`,
`val_accuracy`).

## 8. Train the BERT v2 Model

```bash
python -m src.train_depression_bert_v2
```

- **Reads:** `results/splits/depression_train.csv` (fitting) and `depression_val.csv`
  (early stopping and best-checkpoint selection only, via
  `load_best_model_at_end` / `metric_for_best_model="accuracy"`).
- **Does not read** `depression_test.csv` — this script has no reference to the test path.
- **Writes:** checkpoints under `depression_bert_v2/`, and the final exported model + tokenizer
  + `training_config.json` under `depression_bert_model_v2/`.
- **Modifies models:** yes. Does not overwrite the v1 `depression_bert_model/`.
- Base checkpoint `distilbert-base-uncased`, `max_length=256`, batch size 4 (train/eval) with
  gradient accumulation 2, learning rate `2e-5`, weight decay `0.01`, up to 4 epochs, early
  stopping patience 1 on validation accuracy, seed 42.
- Expensive relative to the classical model: ~7m50s on Apple Silicon MPS (Section 3).

Expected validation result (`depression_bert_model_v2/training_config.json`,
`final_validation_metrics`): best validation accuracy **0.9782**, reached before early stopping
halted the configured 4-epoch run.

Full architecture/training details: [`docs/models.md`](models.md).

## 9. Official Held-Out Evaluation

```bash
python -m experiments.evaluate_models
```

- **Reads:** `depression_model_v2/`, `depression_bert_model_v2/`, and
  `results/splits/depression_test.csv` — **this is the only step that reads the test set.**
- **Does not modify either model** — inference only (`.predict_proba`, forward pass under
  `torch.no_grad()`).
- **Writes:** `results/metrics_depression_v2_{classical,bert}.csv`,
  `results/predictions_depression_v2_{classical,bert}.csv`,
  `results/confusion_matrix_depression_v2_{classical,bert}.png`,
  `results/depression_v2_eval_report_{classical,bert}.json`,
  `results/depression_v2_model_comparison.csv`.
- Computationally light — see Section 3.

**Official held-out test results** (n = 1148), from
`results/depression_v2_model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical (TF-IDF + LR) | 0.9556 | 0.9777 | 0.9309 | 0.9537 | 0.9900 |
| BERT | 0.9843 | 0.9964 | 0.9716 | 0.9838 | 0.9976 |

Full methodology and confusion matrices: [`docs/evaluation.md`](evaluation.md),
[`MODEL_CARD.md`](../MODEL_CARD.md).

## 10. Calibration Analysis

```bash
python -m experiments.calibration_analysis_depression_v2
```

- **Reads:** `depression_model_v2/`, `depression_bert_model_v2/`,
  `results/splits/depression_val.csv`, `results/splits/depression_test.csv`.
- Runs inference (not fitting) with both models on val and test. The only thing actually fit is
  a single scalar temperature for BERT, learned by minimizing NLL on **validation logits only**,
  then frozen and applied once to test logits.
- **Does not modify either model's weights.** Temperature scaling is a post-hoc rescaling of
  logits, not retraining.
- **Writes:** `results/calibration/calibration_report.json`, `calibration_metrics.csv`,
  `calibration_predictions.csv`, `CALIBRATION_REPORT.md`, and three reliability-diagram PNGs
  plus a confidence-distribution PNG.
- Independent of `experiments/evaluate_models.py` — it performs its own inference rather than
  reading that script's prediction CSVs.

**Canonical calibration results** (`results/calibration/calibration_report.json`):

| Metric | BERT raw | BERT temperature-scaled |
| --- | ---: | ---: |
| Temperature | — | 1.8148 |
| Log-loss | 0.0796 | 0.0605 |
| Brier score | 0.0143 | 0.0136 |
| ECE | 0.0149 | 0.0108 |

Predicted classes are verified unchanged by temperature scaling
(`accuracy_unchanged_by_temperature_scaling: true`) — this is asserted in the script, not just
reported.

Temperature scaling is a **post-hoc calibration procedure only**. It does not retrain the
classifier and does not make model confidence a clinical probability — see
[`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) §5. Full methodology:
[`docs/evaluation.md`](evaluation.md), [`MODEL_CARD.md`](../MODEL_CARD.md).

## 11. Error Analysis

```bash
python -m experiments.error_analysis_depression_v2
```

- **Reads:** `results/predictions_depression_v2_classical.csv` and
  `results/predictions_depression_v2_bert.csv` — **the prediction files written by
  `experiments/evaluate_models.py`, not the models or the test set directly.**
- **Ordering requirement: `experiments.evaluate_models` must be run first**, or this script will
  fail (`FileNotFoundError` on the prediction CSVs) or, worse, silently analyze stale
  predictions from a previous run.
- **Does not modify any model** and does not re-run inference.
- **Writes:** `results/error_analysis/error_summary.csv`, several per-category CSVs (false
  positives/negatives, model disagreements, etc.), `linguistic_pattern_analysis.csv`, two PNGs,
  `error_analysis_report.json`, and `ERROR_ANALYSIS_REPORT.md`.
- Cheap — pandas/regex over already-computed predictions.

**Canonical results** (`results/error_analysis/error_summary.csv`,
`error_analysis_report.json`):

| Quantity | Value |
| --- | --- |
| Classical errors | 51 / 1148 |
| BERT errors | 18 / 1148 |
| Classical wrong, BERT correct | 36 |
| BERT wrong, classical correct | 3 |

22 linguistic-pattern significance tests were run (11 patterns × 2 models); **none survived
Bonferroni correction.** The label-noise discussion in this analysis is a hypothesis illustrated
by representative examples — it is not a proven, systematic finding about the dataset. Full
methodology and caveats: [`docs/evaluation.md`](evaluation.md).

## 12. Multi-Seed Stability Analysis

```bash
python -m experiments.multiseed_depression_v2
```

- **Reads only** `results/splits/{depression_train,depression_val,depression_test}.csv` — it
  does **not** read `depression_model_v2/` or `depression_bert_model_v2/` at all; it trains five
  fresh classical + BERT models from scratch (seeds `42, 123, 2024, 3407, 7777`), using the same
  hyperparameters as Sections 7–8.
- **Does not touch or modify** the canonical seed-42 artifacts
  (`depression_model_v2/`, `depression_bert_model_v2/`,
  `results/depression_v2_model_comparison.csv`, `results/calibration/`) — it writes everything
  to its own directory.
- Test data is used only for final, post-hoc, per-seed measurement — never for seed selection,
  checkpointing, or temperature fitting (temperature is fit per-seed on that seed's validation
  logits only).
- **Writes:** `results/multiseed/multiseed_results.csv`, `multiseed_summary.csv`,
  `multiseed_report.json`, `MULTISEED_REPORT.md`, four PNGs, and disposable per-seed model
  weights under `results/multiseed/tmp_models/` (Section 14).
- **Expensive: ~36 minutes 55 seconds total** on the canonical hardware (5× the BERT training
  cost of Section 8, plus 5× classical training). Treat this step as optional / opt-in.

**Canonical five-seed results** (`results/multiseed/multiseed_summary.csv`):

| Metric | Classical (mean ± SD) | BERT (mean ± SD) |
| --- | ---: | ---: |
| Accuracy | 0.9556 ± 0.0000 | 0.9829 ± 0.0047 |
| F1 | 0.9537 ± 0.0000 | 0.9825 ± 0.0048 |
| ROC-AUC | 0.9900 ± 0.0000 | 0.9975 ± 0.0004 |
| ECE | 0.1033 ± 0.0000 | 0.0143 ± 0.0018 |
| Brier | 0.0487 ± 0.0000 | 0.0150 ± 0.0026 |
| Log-loss | 0.2001 ± 0.0000 | 0.0768 ± 0.0074 |

The classical model's exact-zero standard deviation is expected, not a bug: scikit-learn's
`LogisticRegression` default solver (`lbfgs`) is deterministic and does not use `random_state`,
so the classical fit is identical regardless of seed **in this specific configuration** — this
does not mean the classical model is deterministic under every possible solver/configuration.
Full discussion: [`docs/evaluation.md`](evaluation.md), [`MODEL_CARD.md`](../MODEL_CARD.md).

## 13. Expected Output Artifacts

| Step | Key outputs |
| --- | --- |
| Split (§6) | `results/splits/depression_{train,val,test}.csv`, `depression_split_manifest.json` |
| Classical training (§7) | `depression_model_v2/{model.pkl,vectorizer.pkl,training_config.json}` |
| BERT training (§8) | `depression_bert_v2/` (checkpoints), `depression_bert_model_v2/` (final model + tokenizer + `training_config.json`) |
| Official evaluation (§9) | `results/metrics_depression_v2_*.csv`, `results/predictions_depression_v2_*.csv`, `results/confusion_matrix_depression_v2_*.png`, `results/depression_v2_eval_report_*.json`, `results/depression_v2_model_comparison.csv` |
| Calibration (§10) | `results/calibration/*` (metrics CSV/JSON, predictions CSV, 4 PNGs, Markdown report) |
| Error analysis (§11) | `results/error_analysis/*` (category CSVs, linguistic-pattern CSV, 2 PNGs, JSON + Markdown reports) |
| Multi-seed (§12) | `results/multiseed/{multiseed_results.csv,multiseed_summary.csv,multiseed_report.json,MULTISEED_REPORT.md}`, 4 PNGs, `results/multiseed/tmp_models/` (disposable) |

## 14. Canonical vs Disposable Artifacts

**Canonical / important — do not delete:**

- `results/splits/` (frozen split + manifest)
- `depression_model_v2/`, `depression_bert_model_v2/`
- `results/depression_v2_model_comparison.csv`
- `results/calibration/`
- `results/error_analysis/`
- `results/multiseed/` analysis files (CSV/JSON/MD/PNG — everything except `tmp_models/`)
- `results/legacy_full_dataset_eval/` (preserved leakage-correction record — see
  [Section 18](#18-historical-leakage-correction))

**Disposable:**

- `results/multiseed/tmp_models/` — five sets of per-seed BERT checkpoints and final models,
  **approximately 5 GB** on disk (verified: `du -sh results/multiseed/tmp_models` → `5.0G`).
  All persisted multi-seed metrics live in the CSV/JSON/Markdown files alongside this directory,
  which remain intact if `tmp_models/` is deleted. Safe to remove once those files are
  confirmed present; this directory is already gitignored.

Do not delete anything from the canonical list above.

## 15. Reproducibility Boundary

**Reproducible (methodology and procedure):**

- The split procedure (`src/depression_split.py`) — deterministic given the same input CSV and
  seed 42.
- The training scripts' logic, hyperparameters, and data sources
  (`src/train_depression_v2.py`, `src/train_depression_bert_v2.py`).
- The evaluation, calibration, error-analysis, and multi-seed procedures — what they read, what
  they compute, and what they never touch (the test set, during training/selection).

**Not guaranteed bit-for-bit:**

- Exact floating-point model weights across different hardware/backends.
- MPS training/inference numerical behavior (Section 16).
- Wall-clock time (Section 3) — hardware-dependent.
- The full transitive dependency environment (`requirements.txt` pins top-level packages only —
  Section 4).
- Raw dataset availability on a fresh clone (`data/` is gitignored — Section 19).
- The historical conda environment creation command (not recorded — Section 2).

## 16. MPS and Numerical Reproducibility

Canonical BERT training and inference were performed on Apple Silicon using PyTorch's MPS
backend. CPU fallback exists in the code (every model-loading site in `src/` and
`experiments/` selects `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`),
but **CUDA was never used or tested for any canonical run in this repository.**

MPS floating-point kernels do not guarantee bit-identical results to CPU or CUDA execution, and
are not guaranteed to be bit-identical across PyTorch/macOS versions even on the same hardware.
Reported metrics (Sections 7–12) should be expected to reproduce **to the stated precision**
(typically 4 decimal places, matching how this project's scripts round and report values) on the
same or similar MPS hardware/software configuration — exact trailing digits are not guaranteed,
and reproduction on CPU-only or CUDA hardware may diverge further. This has not been tested by
this project.

## 17. Test-Set Integrity

| Split | Role |
| --- | --- |
| Train | Model fitting only |
| Validation | Model selection, early stopping, temperature fitting where applicable |
| Test | Final, frozen evaluation — used exactly once, after everything else is finished |

The test set (`results/splits/depression_test.csv`) must never be used for:

- fitting,
- hyperparameter selection,
- early stopping,
- temperature fitting,
- any iterative development decision.

By construction in this repository:

- `src/train_depression_v2.py` and `src/train_depression_bert_v2.py` do not reference the test
  path at all.
- `experiments/calibration_analysis_depression_v2.py` fits its temperature parameter on
  validation logits only, then applies it, frozen, to test logits — and asserts that doing so
  does not change any predicted class.
- `experiments/evaluate_models.py` performs inference and metric computation only; it does not
  fit anything.
- `experiments/multiseed_depression_v2.py` explicitly checks the test file's mtime before and
  after the run and reports whether it was left untouched.

## 18. Historical Leakage Correction

The original (v1) evaluation loaded the **entire** `data/depression_dataset.csv` (7,731 rows)
for evaluation, but the v1 depression BERT model had already been trained on a subset of that
same file — producing a reported accuracy of **0.9933**, which is invalid as a held-out
generalization estimate because of this train/test overlap.

**This 0.9933 figure must not be cited as a valid result.** It is preserved, unmodified, under
`results/legacy_full_dataset_eval/` (with its own explanatory `README.md`) purely as an
auditable record of the bug and its correction — not as a current or comparable metric.

The corrected v2 workflow:

```text
raw data (data/depression_dataset.csv)
  → deduplication (81 rows removed)
  → canonical, stratified, seed-42 train/val/test split
  → train-only model fitting
  → validation-only model selection / early stopping
  → frozen test-set evaluation, exactly once
```

Full narrative: [`docs/data.md`](data.md).

## 19. Fresh-Clone Limitations

A fresh clone of this repository does **not** provide everything needed for full reproduction:

- **`data/` is gitignored.** `data/depression_dataset.csv` (the source for
  `src/depression_split.py`) is not tracked in Git and will not be present after a fresh clone.
- **`depression_bert_model_v2/` is gitignored** (large-weight directory), along with
  `depression_bert_v2/`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, and
  `results/multiseed/tmp_models/`. `depression_model_v2/` (small `.pkl` files) is intentionally
  **not** gitignored and remains trackable.
- **No dataset download URL, Hugging Face dataset repository, or model-hosting location is
  recorded anywhere in this repository.** None is invented here.
- **The exact historical `conda create` command for `bert310` is not recorded** (Section 2).

A fully independent reproduction from a fresh clone therefore requires separately obtaining:
the raw dataset (`data/depression_dataset.csv`, in the schema expected by
`src/depression_split.py`: `clean_text`, `is_depression` columns), and — if not retraining —
the trained model artifacts under `depression_bert_model_v2/`. If retraining from the canonical
split, `distilbert-base-uncased` is fetched automatically from the Hugging Face Hub by
`transformers` at training time (standard `from_pretrained` behavior); this repository does not
vendor that base checkpoint.

## 20. Current Runtime vs Research Workflow

The canonical v2 research artifacts (`depression_model_v2/`, `depression_bert_model_v2/`) are
now what the repository's runtime inference code loads **for depression**. Sentiment inference
remains on its original v1 artifacts, unaffected by this. Verified directly from source:

| File | Model path loaded |
| --- | --- |
| `src/mental_health_pipeline.py` | `sentiment_bert_model` (v1, unchanged) and `depression_bert_model_v2` (v2, via `src/model_config.py`) |
| `src/predict.py` | `sentiment_model.pkl`, `tfidf_vectorizer.pkl` (v1, unchanged); `depression_model_v2/model.pkl`, `depression_model_v2/vectorizer.pkl` (v2, via `src/model_config.py`) |
| `src/predict_bert.py` | `sentiment_bert_model` (v1, unchanged — sentiment only) |
| `src/predict_depression_bert.py` | `depression_bert_model_v2` (v2, via `src/model_config.py`) |
| `src/mental_health_trajectory.py` | Calls `analyze_text` from `mental_health_pipeline.py` → inherits v2 depression |
| `src/monte_carlo_simulation.py` | Calls `analyze_text` from `mental_health_pipeline.py` → inherits v2 depression |

**Consequence:** running the repository's interactive CLI tools now exercises the same
canonical v2 depression model this guide reproduces (sentiment remains v1). **This does not
mean the CLI reproduces the official evaluation metrics in Sections 9–12** — those metrics
come from running inference against the frozen test set (`results/splits/depression_test.csv`)
and computing aggregate metrics, which the interactive CLI does not do; it only classifies
whatever single piece of text it is given, with no metric computation or comparison to ground
truth. **Use the evaluation workflow in Sections 9–12 for official, reproducible numbers — not
the CLI** — and note that running the CLI does not constitute clinical validation of any kind.

Full discussion: [`MODEL_CARD.md`](../MODEL_CARD.md) §19,
[`docs/architecture.md`](architecture.md), [`RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md).

## 21. Recommended Reproduction Order

1. Obtain the repository (clone).
2. Obtain the required dataset (`data/depression_dataset.csv`) and, if not retraining, the
   `depression_bert_model_v2/` artifacts — neither ships via Git (Section 19).
3. Activate the verified `bert310` conda environment (Section 2).
4. Install requirements (`pip install -r requirements.txt`, which includes `scipy==1.15.3`).
5. Generate the canonical split (§6).
6. Train the classical v2 model (§7).
7. Train the BERT v2 model (§8).
8. Run the official evaluation (§9) — **required before step 10**, since error analysis reads
   its prediction files.
9. Run calibration analysis (§10) — independent of step 8, but requires steps 6–7.
10. Run error analysis (§11) — requires step 8's output files.
11. *(Optional, expensive — ~37 minutes)* Run multi-seed stability analysis (§12) — requires
    only step 5.
12. Inspect the generated outputs (Section 13).
13. Compare against the canonical results quoted in this guide, [`MODEL_CARD.md`](../MODEL_CARD.md),
    and [`docs/evaluation.md`](evaluation.md).

## 22. Troubleshooting

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'scipy'` | `scipy==1.15.3` is pinned in `requirements.txt` (Section 4), but `pip install -r requirements.txt` was not run, or was run before this pin was added | `pip install -r requirements.txt` (or `pip install scipy==1.15.3` directly) inside the active environment |
| `ModuleNotFoundError` for `torch`, `transformers`, etc. | Wrong or inactive conda environment | Confirm `conda activate bert310` was run, then re-run `pip install -r requirements.txt` |
| `FileNotFoundError` on `results/splits/depression_*.csv` | Split not generated yet | Run `python -m src.depression_split` (§6) first |
| `FileNotFoundError` on `data/depression_dataset.csv` | Fresh clone — `data/` is gitignored (Section 19) | Obtain the dataset separately; it is not distributed via this Git repository |
| `experiments.evaluate_models` fails to load `depression_model_v2/` or `depression_bert_model_v2/` | Models not trained yet | Run §7 and §8 before §9 |
| `experiments.error_analysis_depression_v2` fails on missing `results/predictions_depression_v2_*.csv` | Run out of order | Run `experiments.evaluate_models` (§9) first — error analysis does not perform its own inference |
| `torch.backends.mps.is_available()` returns `False` | Non-Apple-Silicon hardware, or an macOS/PyTorch version without MPS support | Training/inference will silently fall back to CPU (much slower for BERT training in particular); this is expected behavior, not an error |
| Disk fills up during/after multi-seed run | `results/multiseed/tmp_models/` (~5 GB, Section 14) | Safe to delete once `multiseed_results.csv` / `multiseed_summary.csv` / `multiseed_report.json` are confirmed present |
| Results differ from the canonical numbers in this guide by a small amount | Different hardware/backend (Section 16) | Expect reproduction to the stated decimal precision on similar MPS hardware; exact bit-for-bit reproduction is not guaranteed, especially on CPU-only or CUDA hardware |
| Accidentally regenerated `results/splits/` and now local test set differs from what's committed | `src/depression_split.py` was re-run | Check `git status` / `git diff` on `results/splits/` before treating any new run as canonical — these files are frozen (Section 17) |

## 23. Reproducibility Checklist

- [ ] Python 3.10 environment available
- [ ] Correct conda environment (`bert310`) activated
- [ ] `pip install -r requirements.txt` completed (includes `scipy==1.15.3`)
- [ ] `data/depression_dataset.csv` available locally
- [ ] Canonical split generated (`python -m src.depression_split`)
- [ ] Zero train/val/test overlap verified (`results/splits/depression_split_manifest.json` → `no_overlap_verified: true`)
- [ ] Classical v2 model trained (`python -m src.train_depression_v2`)
- [ ] BERT v2 model trained (`python -m src.train_depression_bert_v2`)
- [ ] Test set was not used during training or model selection for either model
- [ ] Official evaluation completed (`python -m experiments.evaluate_models`)
- [ ] Calibration analysis completed, if desired (`python -m experiments.calibration_analysis_depression_v2`)
- [ ] Error analysis completed, if desired — after official evaluation (`python -m experiments.error_analysis_depression_v2`)
- [ ] Multi-seed stability analysis completed, if desired — optional, ~37 minutes (`python -m experiments.multiseed_depression_v2`)
- [ ] Results compared against the canonical figures in this guide, `MODEL_CARD.md`, and `docs/evaluation.md`

## 24. Reproducibility Statement

This project provides a deterministic, seeded data-splitting procedure and scripted
training/evaluation procedures, with a persisted canonical split
(`results/splits/`) and recorded model configurations
(`depression_model_v2/training_config.json`, `depression_bert_model_v2/training_config.json`).
The canonical results reported throughout this repository were produced on Apple Silicon using
PyTorch's MPS backend and may not be bit-for-bit identical across different hardware, backends,
or software versions. Reproduction of this project's methodology and results is therefore
**methodological, and numerical to the reported precision — not a guarantee of byte-for-byte
identical model weights or metrics** on other systems.
