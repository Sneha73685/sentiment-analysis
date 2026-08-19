"""Official held-out test evaluation for the leakage-free depression models
(v2), trained exclusively on results/splits/depression_train.csv with
results/splits/depression_val.csv used only for validation/early stopping.

Evaluates ONLY:
  - depression_model_v2/model.pkl + depression_model_v2/vectorizer.pkl
  - depression_bert_model_v2/

Against ONLY:
  - results/splits/depression_test.csv (never seen during training or
    validation of either model -- see split_manifest for the seed/dedup
    record that established this split).

This intentionally does not touch the v1 models or the legacy leakage-
affected results in results/legacy_full_dataset_eval/.
"""

import datetime
import json
import os
import subprocess

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.preprocess import clean_text

RESULTS_DIR = "results"
TEST_SPLIT_PATH = "results/splits/depression_test.csv"
SPLIT_MANIFEST_PATH = "results/splits/depression_split_manifest.json"

CLASSICAL_MODEL_DIR = "depression_model_v2"
BERT_MODEL_DIR = "depression_bert_model_v2"

BERT_BATCH_SIZE = 16
BERT_MAX_LENGTH = 256

CLASS_NAMES = ["Not Depressed", "Depressed"]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def load_test_split():
    if not os.path.exists(TEST_SPLIT_PATH):
        raise FileNotFoundError(
            f"{TEST_SPLIT_PATH} not found. Run `python -m src.depression_split` first."
        )
    df = pd.read_csv(TEST_SPLIT_PATH)
    return df["clean_text"].tolist(), df["is_depression"].tolist()


def load_split_manifest():
    if not os.path.exists(SPLIT_MANIFEST_PATH):
        return None
    with open(SPLIT_MANIFEST_PATH) as f:
        return json.load(f)


def compute_metrics(labels, preds, probs):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    cm = confusion_matrix(labels, preds)
    per_class = classification_report(labels, preds, output_dict=True)
    roc_auc = roc_auc_score(labels, probs) if probs is not None else None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate_classical(texts):
    print(f"Evaluating classical depression model ({CLASSICAL_MODEL_DIR})...")
    model = joblib.load(os.path.join(CLASSICAL_MODEL_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(CLASSICAL_MODEL_DIR, "vectorizer.pkl"))

    # Same preprocessing as src/train_depression_v2.py: src.preprocess.clean_text
    # applied before TF-IDF, then vectorizer.transform only (never re-fit).
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)

    preds = model.predict(vec)
    probs = model.predict_proba(vec)[:, 1]

    return preds.tolist(), probs.tolist()


def evaluate_bert(texts):
    print(f"Evaluating BERT depression model ({BERT_MODEL_DIR})...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR).to(device)
    model.eval()

    preds = []
    probs = []

    for start in range(0, len(texts), BERT_BATCH_SIZE):
        batch = texts[start : start + BERT_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=BERT_MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_probs = F.softmax(outputs.logits, dim=1)[:, 1]
            batch_preds = torch.argmax(outputs.logits, dim=1)

        preds.extend(batch_preds.cpu().tolist())
        probs.extend(batch_probs.cpu().tolist())

        if start % (BERT_BATCH_SIZE * 10) == 0:
            print(f"  Processed {min(start + BERT_BATCH_SIZE, len(texts))}/{len(texts)} samples")

    return preds, probs


def save_confusion_matrix_png(cm, output_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_results(name, model_dir, texts, labels, preds, probs, split_manifest, timestamp, git_commit):
    metrics = compute_metrics(labels, preds, probs)

    flat_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }
    pd.DataFrame([flat_metrics]).to_csv(
        os.path.join(RESULTS_DIR, f"metrics_depression_v2_{name}.csv"), index=False
    )

    predictions_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": labels,
            "prediction": preds,
            "probability_depressed": probs,
        }
    )
    predictions_df.to_csv(
        os.path.join(RESULTS_DIR, f"predictions_depression_v2_{name}.csv"), index=False
    )

    save_confusion_matrix_png(
        metrics["confusion_matrix"],
        os.path.join(RESULTS_DIR, f"confusion_matrix_depression_v2_{name}.png"),
        f"Depression v2 ({name}) — Held-out Test Confusion Matrix",
    )

    report = {
        "model": name,
        "model_dir": model_dir,
        "test_set_path": TEST_SPLIT_PATH,
        "test_set_size": len(texts),
        "split_manifest": split_manifest,
        "evaluation_timestamp_utc": timestamp,
        "git_commit": git_commit,
        **metrics,
    }
    report_path = os.path.join(RESULTS_DIR, f"depression_v2_eval_report_{name}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[{name}] accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1_score']:.4f} "
          f"roc_auc={metrics['roc_auc']:.4f}")
    print(f"[{name}] confusion matrix: {metrics['confusion_matrix']}")

    return flat_metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_commit = _git_commit()
    split_manifest = load_split_manifest()

    print("Official held-out test evaluation -- depression models v2")
    print(f"Timestamp (UTC): {timestamp}")
    print(f"Git commit: {git_commit}")
    print(f"Test set: {TEST_SPLIT_PATH}")
    print()

    texts, labels = load_test_split()
    print(f"Loaded {len(texts)} held-out test examples\n")

    classical_preds, classical_probs = evaluate_classical(texts)
    classical_flat = save_model_results(
        "classical", CLASSICAL_MODEL_DIR, texts, labels, classical_preds, classical_probs,
        split_manifest, timestamp, git_commit,
    )

    bert_preds, bert_probs = evaluate_bert(texts)
    bert_flat = save_model_results(
        "bert", BERT_MODEL_DIR, texts, labels, bert_preds, bert_probs,
        split_manifest, timestamp, git_commit,
    )

    comparison_df = pd.DataFrame(
        [
            {"model": "classical", "test_samples": len(texts), "model_path": CLASSICAL_MODEL_DIR, **classical_flat},
            {"model": "bert", "test_samples": len(texts), "model_path": BERT_MODEL_DIR, **bert_flat},
        ]
    )
    comparison_path = os.path.join(RESULTS_DIR, "depression_v2_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\nWrote comparison table to {comparison_path}")
    print("Done.")


if __name__ == "__main__":
    main()
