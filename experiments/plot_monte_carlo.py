import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/monte_carlo_results.csv")

plt.hist(df["mean_risk"], bins=10)

plt.xlabel("Mean Risk Score")
plt.ylabel("Frequency")
plt.title("Monte Carlo Risk Distribution")

plt.savefig("plots/monte_carlo_distribution.png")