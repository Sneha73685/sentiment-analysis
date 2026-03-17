import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.mental_health_pipeline import analyze_text

df = pd.read_csv("data/depression_dataset.csv")

texts = df["clean_text"].tolist()
labels = df["is_depression"].tolist()

preds = []

for i, text in enumerate(texts):

    result = analyze_text(text)

    pred = 1 if result["mental_health"]["depression_risk"] == "depressed" else 0
    preds.append(pred)

    if i % 100 == 0:
        print(f"Processed {i}/{len(texts)} samples")

accuracy = accuracy_score(labels, preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary"
)

cm = confusion_matrix(labels, preds)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv("results/metrics.csv", index=False)

predictions_df = pd.DataFrame({
    "text": texts,
    "true_label": labels,
    "prediction": preds
})

predictions_df.to_csv("results/predictions.csv", index=False)

print(metrics)
print(cm)