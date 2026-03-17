import pandas as pd
from src.monte_carlo_simulation import run_monte_carlo

df = pd.read_csv("data/depression_dataset.csv")

texts = df["clean_text"].sample(200)

results = []

for text in texts:

    mc = run_monte_carlo(text)

    results.append({
        "text": text,
        "mean_risk": mc["mean_risk_score"],
        "variance": mc["risk_variance"]
    })

pd.DataFrame(results).to_csv(
    "results/monte_carlo_results.csv",
    index=False
)