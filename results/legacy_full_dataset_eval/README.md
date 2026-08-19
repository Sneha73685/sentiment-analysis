# LEGACY — INVALID FOR FINAL EVALUATION

These files (`metrics.csv`, `predictions.csv`) were produced by the original
`experiments/evaluate_models.py`, which scored the depression BERT pipeline
against the **entire** `data/depression_dataset.csv` — including rows the
model was trained on (`depression_bert_model` was trained on ~90% of this
same file).

Reported accuracy here (0.9933) is inflated by train/test leakage and must
**not** be cited as a generalization estimate or used in any paper, plot, or
comparison.

For the leakage-free evaluation methodology and current results, see:

- `src/depression_split.py` — deterministic, deduplicated, stratified split
- `results/splits/` — persisted train/val/test split + manifest
- `results/metrics_depression_*.csv`, `results/predictions_depression_*.csv`

Kept here for historical reference only, not deleted.
