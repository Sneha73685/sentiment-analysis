"""Standalone Monte Carlo risk-distribution plot.

Read-only: loads results/monte_carlo_results.csv, produced by
`python -m experiments.run_monte_carlo_batch`. This script does not run the
Monte Carlo experiment itself and does not touch any canonical/frozen
artifact.

The underlying experiment perturbs input text and re-runs the combined
sentiment + depression + compute_risk() pipeline (src/mental_health_pipeline.py,
which currently loads v1 models). It measures the sensitivity of that
unvalidated heuristic pipeline as a whole, not a certified model-robustness
guarantee -- see RESPONSIBLE_USE.md Section 8.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = "results/monte_carlo_results.csv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"{INPUT_PATH} not found. Run `python -m experiments.run_monte_carlo_batch` "
        "first to generate it -- this is a substantial batch operation (200 sampled "
        "texts x 100 perturbations each, through the full combined pipeline) and is "
        "not run automatically by this script."
    )

df = pd.read_csv(INPUT_PATH)

os.makedirs("plots", exist_ok=True)

plt.hist(df["mean_risk"], bins=10)

plt.xlabel("Mean Risk Score")
plt.ylabel("Frequency")
plt.title("Monte Carlo Risk Distribution")

output_path = "plots/monte_carlo_distribution.png"
plt.savefig(output_path)
plt.close()
print(f"Wrote {output_path}")
