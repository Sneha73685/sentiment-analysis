# Development Guide

## 1. Purpose and Scope

This document is a **maintainer/contributor guide**: how to safely modify, extend, test,
document, and eventually release this repository without breaking the validated v2 research
workflow.

It is **not** the primary reference for models, evaluation methodology, or reproduction steps.
For those, see:

- [`../MODEL_CARD.md`](../MODEL_CARD.md) — canonical model summary
- [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) — responsible-use boundary
- [`reproducibility.md`](reproducibility.md) — operational reproduction guide
- [`architecture.md`](architecture.md) — system architecture

This guide assumes familiarity with those documents and focuses on contributor-facing process:
what is safe to change, what must not be touched casually, and how to keep the repository in a
state a future contributor (or the project's author, months later) can trust.

## 2. Repository Layout

| Path | Responsibility |
| --- | --- |
| `src/` | Library code: preprocessing, training scripts, prediction/runtime pipeline, split generation |
| `experiments/` | Standalone scripts that produce `results/` artifacts (evaluation, calibration, error analysis, multi-seed, plotting) |
| `results/` | Persisted outputs — canonical splits, metrics, reports, plots (Section 4) |
| `data/` | Raw datasets; gitignored (Section 7, Section 13) |
| `docs/` | Detailed technical documentation, including this file and `docs/diagrams/` (Section 19) |
| `depression_model_v2/`, `depression_bert_model_v2/` | Canonical v2 model artifacts (Section 3) |
| `depression_model.pkl`, `depression_bert_model/`, `sentiment_model.pkl`, `sentiment_bert_model/`, etc. | v1 (historical) model artifacts. Sentiment (`sentiment_model.pkl`, `sentiment_bert_model/`) is still loaded by the runtime track; depression v1 artifacts are retained for history/audit but no longer loaded by runtime (Section 11) |
| `tests/` | Minimal pytest regression suite added in Phase D. See Section 15. |

This is a summary, not the full tree — see [`architecture.md`](architecture.md) for the
complete system-level picture.

## 3. V1 vs V2 Naming Convention

The project distinguishes two artifact generations by an explicit `_v2` suffix:

| V1 (historical) | V2 (canonical) |
| --- | --- |
| `depression_model.pkl`, `depression_vectorizer.pkl` | `depression_model_v2/` |
| `depression_bert_model/` | `depression_bert_model_v2/` |

- **V1** artifacts are the original, pre-leakage-correction implementation. They still exist for
  history/audit and are not the validated research result. Sentiment v1 artifacts are still
  loaded by the runtime track (Section 11); depression v1 artifacts are no longer loaded by
  runtime as of the migration described there.
- **V2** artifacts are the canonical depression research generation — trained on the leakage-free
  split, evaluated on a frozen test set. **"Canonical" refers to v2 only.**

The explicit suffix exists **to prevent accidentally loading or reporting results from the
historical (leakage-affected) workflow** as if they were current. Never remove or rename a `_v2`
suffix in a way that makes a v2 artifact indistinguishable from a v1 one.

Full story: [`data.md`](data.md) (leakage discovery and correction),
[`models.md`](models.md) (architecture/training differences),
[`../MODEL_CARD.md`](../MODEL_CARD.md) (canonical summary).

## 4. Canonical and Frozen Artifacts

The following are **frozen**. Do not regenerate, overwrite, delete, or modify them unless a task
explicitly requires a new canonical experiment and the consequences (Section 21) are understood:

```text
results/splits/depression_train.csv
results/splits/depression_val.csv
results/splits/depression_test.csv
results/splits/depression_split_manifest.json

depression_model_v2/
depression_bert_model_v2/

results/depression_v2_model_comparison.csv

results/calibration/
results/error_analysis/
results/multiseed/            (analysis files — see Section 5 for the exception)
results/legacy_full_dataset_eval/
```

`results/legacy_full_dataset_eval/` in particular **must be preserved** — it is not stale
clutter, it is the auditable record of the historical leakage bug and its correction
([`data.md`](data.md)). Deleting it would erase the evidence of why v2 exists.

## 5. Disposable Artifacts

`results/multiseed/tmp_models/` is the one explicitly disposable directory:

- Contains temporary per-seed model artifacts (5 seeds × classical + BERT) produced by
  `experiments/multiseed_depression_v2.py`.
- Approximately **5 GB** in the current repository state.
- The important research artifacts are the persisted CSV/JSON/Markdown/PNG outputs alongside
  it (`multiseed_results.csv`, `multiseed_summary.csv`, `multiseed_report.json`,
  `MULTISEED_REPORT.md`, and the plots) — those are canonical (Section 4) and must be kept.
- The temporary per-seed models may be deleted once those persisted outputs are confirmed
  present.
- Already gitignored — deleting it has no Git-tracking consequence.

Do not extend this "disposable" reasoning to any other artifact in Section 4.

## 6. Source Code Modification Rules

Before changing any source file:

- Inspect its current behavior first — read it, don't assume.
- Preserve validated methodology unless the task explicitly changes methodology.
- Avoid unnecessary rewrites of working code; a bug fix does not need a refactor attached.
- Keep training and evaluation behavior explicit — avoid hidden defaults or implicit fallbacks
  that change what data or model a script uses.
- Do not silently change model paths (e.g. quietly pointing a script at `depression_model_v2/`
  instead of `depression_model.pkl`, or vice versa) — that is an architectural decision
  (Section 11), not an incidental edit.
- Keep experimental/exploratory changes isolated from canonical artifacts — write new output
  paths rather than overwriting canonical ones.
- Preserve reproducibility metadata — scripts that write `training_config.json` or similar
  should keep doing so if modified.

**The test-set rule is non-negotiable:** the frozen test set
(`results/splits/depression_test.csv`) must not be used for model fitting, model selection,
hyperparameter tuning, early stopping, calibration fitting, or any iterative development
decision. See [`evaluation.md`](evaluation.md) and [`reproducibility.md`](reproducibility.md)
Section 17 for the full rule and how the canonical scripts enforce it.

## 7. Data Handling Rules

- Raw data lives under `data/`.
- `data/` is gitignored — it is not distributed via this Git repository (see Section 13 and
  [`reproducibility.md`](reproducibility.md) Section 19).
- Canonical split files live under `results/splits/` and are frozen (Section 4).
- Duplicate removal (`clean_text`-based) is part of canonical split creation, not a separate
  cleanup step — do not re-run deduplication logic outside `src/depression_split.py`.
- Train/validation/test separation, once established by the canonical split, must be preserved
  in any script that consumes it — do not re-merge or re-shuffle the splits.
- Do not modify the canonical test set casually; see Section 4.

This document does not restate dataset provenance or its limitations — those are documented,
including known gaps, in [`data.md`](data.md). Do not invent provenance information not already
recorded there.

## 8. Model Development Rules

Distinguish clearly between three different activities:

| Activity | What it means | Where it happens |
| --- | --- | --- |
| Developing a new model | Training a new model generation with new methodology/architecture | A new versioned script and output directory (see below) |
| Modifying a canonical model | Changing `depression_model_v2/` or `depression_bert_model_v2/` in place | **Do not do this** — retrain into a new version instead (Section 21) |
| Evaluating an existing frozen model | Running inference against an already-trained model | `experiments/evaluate_models.py` or similar — read-only with respect to the model |

If a genuinely new methodology is introduced, use an **explicit versioned artifact name** rather
than silently replacing v2 — for example `depression_model_v3/`, `depression_bert_model_v3/`.
This is a naming **convention for future work**, not an instruction to create v3 now.

Model configuration (hyperparameters, seed, data paths, git commit) should always be recorded in
a `training_config.json` alongside the model weights, following the pattern already used by
`depression_model_v2/training_config.json` and `depression_bert_model_v2/training_config.json`.

Full architecture/training detail: [`models.md`](models.md).

## 9. Evaluation Rules

Evaluation scripts should:

- Use a clearly defined dataset split — reference `results/splits/` files explicitly rather than
  re-deriving a split.
- Never fit anything on the frozen test set (Section 6).
- Record metrics to `results/` in a reproducible, versioned filename.
- Record model/version information (which model directory, which git commit) inside the output.
- Preserve reproducibility metadata (timestamps, seeds, split manifest references) the way the
  existing v2 evaluation scripts already do.
- Distinguish exploratory results from canonical results — write exploratory output to a
  clearly non-canonical path, never overwrite `results/depression_v2_model_comparison.csv` or
  similar canonical files with exploratory numbers.

The official evaluation methodology, metric definitions, and canonical numbers are documented in
[`evaluation.md`](evaluation.md) — this section does not duplicate them.

## 10. Experiment Ordering and Dependencies

Canonical order:

1. Canonical split (`src/depression_split.py`)
2. Model training (`src/train_depression_v2.py`, `src/train_depression_bert_v2.py`)
3. Official evaluation (`experiments/evaluate_models.py`)
4. Calibration analysis (`experiments/calibration_analysis_depression_v2.py`)
5. Error analysis (`experiments/error_analysis_depression_v2.py`)
6. Optional multi-seed analysis (`experiments/multiseed_depression_v2.py`)

**These experiments are not all independent.** In particular:
`experiments/error_analysis_depression_v2.py` reads
`results/predictions_depression_v2_{classical,bert}.csv`, which are written by
`experiments/evaluate_models.py` — it does not perform its own inference. **Error analysis must
run after official evaluation**, or it will fail outright or silently analyze stale predictions
from a previous run.

Calibration analysis and error analysis both **consume existing predictions/results rather than
retraining the canonical models** — calibration performs its own read-only inference against the
already-trained v2 models, and error analysis reads evaluation's output CSVs directly. Neither
modifies `depression_model_v2/` or `depression_bert_model_v2/`.

Full dependency detail and per-step I/O contracts: [`reproducibility.md`](reproducibility.md)
Sections 6–12 and 21.

## 11. Runtime vs Research Track

The repository has two tracks:

**Research track** (validated, canonical):
- `depression_model_v2/`, `depression_bert_model_v2/`
- Canonical split (`results/splits/`)
- `experiments/evaluate_models.py`, `experiments/calibration_analysis_depression_v2.py`,
  `experiments/error_analysis_depression_v2.py`, `experiments/multiseed_depression_v2.py`

**Runtime track:**
- `src/mental_health_pipeline.py`
- `src/predict.py`
- `src/predict_bert.py`
- `src/predict_depression_bert.py`
- Downstream components built on the pipeline: `src/mental_health_trajectory.py`,
  `src/monte_carlo_simulation.py`

**The runtime track's depression inference now loads the canonical v2 model artifacts**
(`depression_model_v2/`, `depression_bert_model_v2/`), via centralized path constants in
`src/model_config.py`. **Sentiment inference remains on v1 artifacts** (`sentiment_model.pkl`,
`sentiment_bert_model/`) — sentiment was explicitly out of scope for this migration and has no
canonical v2 track to migrate to. Historical v1 depression artifacts
(`depression_model.pkl`, `depression_vectorizer.pkl`, `depression_bert_model/`) remain on disk
for audit purposes (Section 4) but are no longer loaded by any runtime code path.

**This migration is a change in which model weights are loaded, not a claim of clinical
validity.** `compute_risk()`, trajectory analysis, and the Monte Carlo experiment are unchanged
and remain the same unvalidated heuristics described in Section 12 below, regardless of which
depression classifier sits underneath them.

Full detail: [`../MODEL_CARD.md`](../MODEL_CARD.md) §19, [`architecture.md`](architecture.md)
Section 12, [`reproducibility.md`](reproducibility.md) Section 20.

## 12. Responsible-Use Requirements

All contributions must respect [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md). In particular,
contributors must not introduce language, code comments, docstrings, log messages, or
documentation that turns this classifier into a diagnostic or clinical tool. Concretely, do not
reframe:

- model confidence as clinical probability,
- the heuristic `compute_risk()` output as a validated clinical risk score,
- trajectory trends (`increasing_risk` / `decreasing_risk` / `stable`) as clinical conclusions,
- Monte Carlo sensitivity results as a certified robustness guarantee.

If contributor-facing material needs more detail than a one-line reminder, link to
[`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) rather than duplicating its content here.

## 13. Git and Artifact Policy

Large model binaries should not normally be committed. `.gitignore` currently protects (among
other patterns):

```text
*.safetensors
depression_bert_v2/
depression_bert_model_v2/
results/multiseed/tmp_models/
data/
```

`depression_model_v2/` is **intentionally not ignored** — it contains only small `.pkl` files
plus a lightweight `training_config.json` and is treated as a trackable reproducibility
artifact, not a large-binary risk.

Some small v1 `.pkl` files (e.g. `depression_model.pkl`, `sentiment_model.pkl`,
`tfidf_vectorizer.pkl`, `depression_vectorizer.pkl`) are already tracked historically in Git.
**Do not remove or rewrite them merely as cleanup** — that is a deliberate decision outside the
scope of routine development work and would affect the runtime track (Section 11).

This task does not modify `.gitignore`; if a future change to ignore patterns is needed, treat
it as an explicit, reviewed decision.

## 14. Working Tree Safety

Standard safe sequence for any change:

```bash
# Before modifying anything
git status
```

```bash
# After making changes
git diff
git diff --check
git status
```

```bash
# Before staging
git diff --cached
```

**Never recommend destructive commands casually.** In particular, do not suggest or run the
following when uncommitted research artifacts may be present, unless the consequences are fully
understood and explicitly approved:

```bash
git reset --hard
git clean -fd
git checkout -- <file>
```

This documentation task does not perform any Git operation.

## 15. Testing Strategy

A minimal `tests/` pytest suite exists (added in the Phase D repository-hygiene work). It is
intentionally small: it protects specific, already-fixed behaviors from silently regressing,
not general application correctness or model quality.

Run it with:

```bash
python -m pytest tests/ -q
```

(`pytest` is a test-only dependency in `requirements.txt` — it is not required by any
`src/` or `experiments/` runtime code.)

**What it covers:**

- `tests/test_preprocess.py` — determinism and current normalization behavior of
  `src/preprocess.py`'s `clean_text` (lowercasing, URL/punctuation stripping, stopword removal,
  lemmatization, empty-string handling).
- `tests/test_signals.py` — `src/mental_health_signals.py`'s `detect_signals`, including an
  explicit regression test that `"number"` does **not** match the `"numb"` signal pattern (the
  word-boundary fix — Section 3 of this document's naming-convention discussion and the source
  fixes already completed elsewhere in this project), a positive match, case-insensitivity, and
  a negative (no-signal) example.
- `tests/test_split.py` — `src/depression_split.py`'s reusable functions
  (`remove_duplicates`, `make_split`, `class_balance`, `assert_no_overlap`) exercised against a
  small **synthetic, in-memory DataFrame** built inside the test itself: reproducibility given
  the same seed, zero train/validation/test overlap, that `assert_no_overlap` actually detects a
  deliberately forced overlap, deduplication before splitting, and stratification.
- `tests/test_metrics.py` — `experiments/evaluate_models.py`'s `compute_metrics` against a tiny,
  hand-verified synthetic example (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
- `tests/test_monte_carlo.py` — a single, explicit `pytest.skip` documenting why Monte Carlo
  helpers are not unit tested: `src/monte_carlo_simulation.py` transitively imports
  `src/mental_health_pipeline.py`, which loads two BERT models (sentiment v1, depression v2) at
  module import time: this is documented in the test file rather than worked around.
- `tests/test_model_config.py` — added as part of the Phase E runtime migration; protects the
  runtime **depression** model-path selection without loading any model weights. It asserts the
  `src/model_config.py` path constants resolve to the v2 artifacts and exist on disk, that
  `src/predict.py` imports those constants (a safe live import — `predict.py` loads models
  lazily, not at import time), and — via **static source-text inspection, not import** — that
  `src/predict_depression_bert.py` and `src/mental_health_pipeline.py` reference the centralized
  v2 constant and no longer contain the old hard-coded v1 path string. The latter two modules
  are never imported in this test suite, since importing either loads BERT.

**What it does not cover** (deliberately, per this phase's scope): BERT model loading or
inference, the full combined pipeline (`mental_health_pipeline.analyze_text`), trajectory
analysis, the Monte Carlo batch experiment, calibration/error-analysis/multi-seed scripts, or
anything requiring the real dataset or canonical result files. Runtime correctness after the
Phase E migration (that inference actually succeeds end-to-end against the v2 artifacts) was
validated separately, by manual smoke test outside this suite — not as a permanent, repeatable
automated test, per this phase's "no BERT in pytest" constraint.

**All tests use small synthetic fixtures, hand-built in-memory data, or path/string assertions.**
None of them read `data/depression_dataset.csv`, `results/splits/depression_test.csv`, or any
other canonical result file, and none of them load model weights — running the suite requires no
dataset and no trained model weights.

Remaining test-coverage gap (future work, not a blocker): any coverage of the runtime/
combined-pipeline code path's actual inference behavior (as opposed to which path it loads),
which would require either loading BERT in the suite or refactoring the import-time model
loading described above — neither was undertaken in this phase.

## 16. Known Engineering Gaps

These are documented gaps, not urgent problems — read each one's framing before acting on it.

**Blockers to a "fully finished" state (tracked, not urgent):**
- Runtime depression inference now uses v2 models; runtime sentiment inference remains v1 and
  unmigrated (Section 11) — this is expected, not an oversight, since sentiment has no v2 track.
- Automated test suite exists but is minimal (Section 15) — it does not cover the runtime/
  combined-pipeline code path or model inference.
- No CI configuration.

**Optional / future improvements:**
- No `pyproject.toml` — the project currently uses `requirements.txt` only.
- No complete dependency lockfile — `requirements.txt` pins top-level packages only
  (Section 17).
- The historical `conda create` command used to build the `bert310` environment is not recorded
  anywhere in this repository ([`reproducibility.md`](reproducibility.md) Section 2).
- `scipy==1.15.3` is explicitly pinned in `requirements.txt` (Section 17), resolving the
  earlier gap where it arrived only transitively for the three experiment scripts that import
  it (`experiments/calibration_analysis_depression_v2.py`,
  `experiments/error_analysis_depression_v2.py`, `experiments/multiseed_depression_v2.py`).
- `experiments/plot_confusion_matrix.py` and `experiments/plot_monte_carlo.py`
  were reviewed and fixed to use current canonical/expected result paths.
  `plot_confusion_matrix.py` runs immediately against the existing
  canonical v2 predictions. `plot_monte_carlo.py` still requires a Monte
  Carlo batch run (`experiments/run_monte_carlo_batch.py`) that has not
  yet been executed in this repository.
- Shared BERT model-loading code (tokenizer + model instantiation, repeated across
  `src/predict_bert.py`, `src/predict_depression_bert.py`, `src/mental_health_pipeline.py`, and
  several `experiments/` scripts) could be consolidated into a shared utility.
- Type hints are limited or absent throughout the codebase.
- Batch inference is not yet a polished, reusable interface — current batch-style usage
  (e.g. `experiments/run_monte_carlo_batch.py`) is script-specific rather than a general API.
- The Monte Carlo methodology (`src/monte_carlo_simulation.py`) currently measures the
  **sensitivity of the combined heuristic pipeline as a whole**, not a certified, isolated
  robustness guarantee for either underlying classifier — see
  [`../RESPONSIBLE_USE.md`](../RESPONSIBLE_USE.md) §8 for the full caveat.

## 17. Dependency Management

`requirements.txt` contains verified top-level package pins for the working `bert310`
environment. It is:

- **Not a complete transitive lockfile** — indirect dependency versions are resolved by `pip`
  at install time and are not separately pinned.
- `scipy`, imported by three experiment scripts (Section 16), is explicitly pinned
  (`scipy==1.15.3`) rather than relying on it arriving transitively via `scikit-learn`'s own
  dependency graph.

The historical `conda create` command used to build `bert310` is not recorded — see
[`reproducibility.md`](reproducibility.md) Section 2.

## 18. Documentation Maintenance

| Document | Owns |
| --- | --- |
| `README.md` | Project orientation |
| `MODEL_CARD.md` | Canonical model summary |
| `RESPONSIBLE_USE.md` | Responsible-use boundary |
| `docs/data.md` | Data and leakage-correction story |
| `docs/models.md` | Model architecture and training details |
| `docs/evaluation.md` | Evaluation, calibration, error-analysis, multi-seed methodology |
| `docs/reproducibility.md` | Operational reproduction workflow |
| `docs/development.md` | Maintainer/contributor practices (this document) |
| `docs/architecture.md` | System architecture |

Each fact should have exactly one authoritative home. **Do not duplicate facts across
documents** — link instead. If a fact changes (a metric, a path, a status), update its
authoritative source first, then verify any document that links to it still makes sense; do not
patch a downstream mention without updating the source of truth.

## 19. Diagram Maintenance

Existing Mermaid diagrams live under `docs/diagrams/`:

- `data-workflow.mmd`
- `training-pipeline.mmd`
- `evaluation-pipeline.mmd`
- `inference-pipeline.mmd`
- `architecture-overview.mmd`
- `repository-structure.mmd`

Rules:

- Diagrams should match the current code/documentation — a diagram that no longer reflects
  reality is worse than no diagram.
- Do not create a duplicate diagram for a concept an existing diagram already covers; extend or
  correct the existing one instead.
- Update a diagram when the architecture it represents materially changes (e.g. if the runtime
  track is repointed to v2 — Section 11 — `inference-pipeline.mmd` would need review).
- Validate Mermaid syntax whenever a diagram is modified (render it, or at minimum check it
  parses) before considering the change complete.

This task does not modify any existing diagram.

## 20. Adding a New Experiment

1. Define the research question the experiment is meant to answer.
2. Identify the correct data split for that question (usually the canonical
   `results/splits/` files).
3. Avoid test-set use during development — only touch the test set for a final, deliberate
   measurement (Section 6).
4. Create a clearly named script (e.g. following the `experiments/<name>_depression_v2.py`
   convention already in use).
5. Record configuration — seed, data paths, hyperparameters — in the script's output.
6. Save outputs under `results/`, in a clearly distinct, non-canonical path unless the
   experiment is intended to become part of the canonical record (Section 21).
7. Document the experiment — what it measures, what it does not measure, and its limitations.
8. Distinguish exploratory output from canonical output explicitly, in both the filename/path
   and any accompanying report text.
9. Preserve seeds where relevant, so the experiment itself is reproducible.
10. Validate before interpreting results — sanity-check output shapes, row counts, and any
    assertions (e.g. non-overlap, accuracy-unchanged-by-calibration) the way existing v2 scripts
    do.

**A new experiment is not automatically part of the canonical research results** just because it
runs successfully — promoting exploratory results to canonical status is a separate, deliberate
decision (Section 21, Section 4).

## 21. Creating a New Model Version

This is a future-safe procedure. **This document does not create v3** — it documents the
convention for if/when a genuinely new methodology is introduced.

1. Define why a new version is needed (new architecture, new methodology, new data — not merely
   a hyperparameter tweak, which can stay within v2's existing training script if the canonical
   result is being deliberately superseded).
2. Preserve the existing v2 artifacts — do not overwrite or delete them (Section 4).
3. Create a new, explicitly versioned training script (e.g. `src/train_depression_v3.py`), not a
   silent modification of the v2 script.
4. Use the canonical split unless the research question specifically requires a new split — if a
   new split is required, document why and treat it with the same rigor as the original
   (deduplication, stratification, zero-overlap verification, recorded manifest).
5. Do not touch the frozen test set during development of the new version (Section 6).
6. Train using the train split only.
7. Select using the validation split only.
8. Evaluate once on the test split, after training and selection are complete.
9. Record configuration in a `training_config.json`, following the existing v2 pattern
   (Section 8).
10. Create separate, explicitly versioned result artifacts (e.g.
    `results/depression_v3_model_comparison.csv`) — do not overwrite v2's canonical results.
11. Update `MODEL_CARD.md` only after the new version has been validated end-to-end, not while
    still exploratory.
12. Clearly state, in `MODEL_CARD.md` and anywhere else relevant, which version is canonical —
    if v3 becomes canonical, v2's status changes from "canonical" to "previous canonical
    generation," analogous to how this repository currently documents v1's relationship to v2.

## 22. Before a Research Paper

The software/model should be **stabilized before deciding on a paper**. A paper contribution
should not be inferred merely from:

- high accuracy,
- BERT outperforming the classical model,
- calibration improvements from temperature scaling,
- multi-seed stability.

These are engineering validation results, not automatically a research contribution. Potential
research directions built on this infrastructure (e.g. classical-vs-BERT comparison,
calibration, confidence-on-errors behavior, multi-seed stability, model disagreement, dataset/
label-noise questions) can be explored later, but **novelty must be established through a
systematic literature review and a clearly defined research question** — not asserted from this
repository's own results in isolation.

This document does not make any novelty claim. See [`evaluation.md`](evaluation.md) and
[`../MODEL_CARD.md`](../MODEL_CARD.md) for what has actually been measured.

## 23. Release and GitHub Workflow

The repository is eventually intended to be pushed to GitHub. **This document does not commit or
push anything.** The intended eventual release process:

1. Finish implementation.
2. Finish documentation.
3. Run validation (`python -m pytest tests/ -q`, Section 15, plus manual checks for anything
   outside its coverage).
4. Inspect `git diff` in full.
5. Verify `.gitignore` still excludes large artifacts as intended (Section 13).
6. Ensure large artifacts are not staged (`git status`, `git diff --cached --stat`).
7. Stage appropriate source/docs/lightweight-results files — specific files, not `git add -A`.
8. Review the staged diff (`git diff --cached`).
9. Create meaningful, scoped commits.
10. Verify each commit (`git show`, `git log`).
11. Push to GitHub.
12. Verify the remote repository state matches what was intended.

**No force push. No history rewriting unless separately approved.** Large model binaries should
remain outside ordinary Git tracking (Section 13) unless a deliberate artifact-storage strategy
(e.g. Git LFS, external model hosting) is adopted later as its own decision — none is currently
in place, and none is assumed here. This document does not reference or invent a GitHub
repository URL.

## 24. Definition of Done for Development Changes

- [ ] Change stayed within the requested file/task scope
- [ ] Current source behavior was inspected before modifying it
- [ ] Tests added/updated where appropriate (`tests/`, Section 15)
- [ ] No frozen artifacts (Section 4) modified accidentally
- [ ] Test-set integrity preserved (Section 6) — no fitting, selection, or tuning on the test set
- [ ] Relevant documentation updated, at its authoritative source (Section 18)
- [ ] `git diff` reviewed in full
- [ ] `git diff --check` clean (no whitespace errors)
- [ ] No large binaries staged accidentally (`git status`, `git diff --cached --stat`)
- [ ] Reproducibility metadata preserved (seeds, `training_config.json`, timestamps where
      applicable)
- [ ] Responsible-use implications considered (Section 12) — no clinical-tool language introduced

## 25. Maintainer Checklist

**Before change:**
- [ ] Inspect `git status`
- [ ] Inspect the relevant source file(s) directly — don't assume behavior
- [ ] Identify which canonical/frozen artifacts (Section 4) are near this change
- [ ] Identify any test-set implications (Section 6)

**During change:**
- [ ] Keep scope narrow — resist unrelated cleanup or refactors
- [ ] Preserve reproducibility (seeds, config recording, explicit paths)
- [ ] Avoid hidden data leakage (test-set use, train/val/test boundary violations)
- [ ] Record meaningful configuration for anything that trains or evaluates a model

**After change:**
- [ ] Run relevant validation (tests once they exist; manual checks otherwise)
- [ ] Inspect `git diff`
- [ ] Inspect `git status`
- [ ] Verify canonical artifacts (Section 4) are unchanged unless the task explicitly intended
      a new canonical experiment
- [ ] Verify no accidental binaries are staged
- [ ] Update documentation at its authoritative source
- [ ] Only then consider staging/committing — and only when explicitly asked to (Section 23)
