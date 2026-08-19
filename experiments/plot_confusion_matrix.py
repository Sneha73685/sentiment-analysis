"""Standalone confusion-matrix plot for the canonical v2 depression models.

Read-only: loads the prediction CSVs already written by
`experiments/evaluate_models.py` and plots a confusion matrix per model. It
does not run inference, does not touch the frozen test set, and does not
modify any canonical result file.

Requires results/predictions_depression_v2_{classical,bert}.csv to already
exist -- run `python -m experiments.evaluate_models` first if they don't.

Canonical confusion-matrix PNGs are already produced by evaluate_models.py
under results/confusion_matrix_depression_v2_{classical,bert}.png; this
script is a lightweight, standalone way to regenerate the same plot from
the persisted predictions without re-running the full evaluation, and
writes to plots/ rather than results/ so it never overwrites those
canonical outputs.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

MODELS = {
    "classical": "results/predictions_depression_v2_classical.csv",
    "bert": "results/predictions_depression_v2_bert.csv",
}

os.makedirs("plots", exist_ok=True)

for name, path in MODELS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m experiments.evaluate_models` "
            "first to generate the canonical v2 predictions."
        )

    df = pd.read_csv(path)
    cm = confusion_matrix(df["true_label"], df["prediction"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Depressed", "Depressed"],
        yticklabels=["Not Depressed", "Depressed"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix -- Depression v2 ({name})")

    output_path = f"plots/confusion_matrix_{name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote {output_path}")
