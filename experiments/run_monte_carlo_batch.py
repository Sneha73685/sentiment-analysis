import os

import pandas as pd
from src.monte_carlo_simulation import run_simulation

RANDOM_SEED = 42

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/depression_dataset.csv")

texts = df["clean_text"].sample(n=200, random_state=RANDOM_SEED)

results = []

for text in texts:

    mc = run_simulation(text, seed=RANDOM_SEED)

    results.append({
        "text": text,
        "mean_risk": mc["mean_risk_score"],
        "variance": mc["risk_variance"]
    })

pd.DataFrame(results).to_csv(
    "results/monte_carlo_results.csv",
    index=False
)