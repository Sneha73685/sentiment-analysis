import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv("results/predictions.csv")

cm = confusion_matrix(df["true_label"], df["prediction"])

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Depressed","Depressed"],
    yticklabels=["Not Depressed","Depressed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")