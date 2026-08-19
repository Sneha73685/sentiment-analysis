"""Classical (TF-IDF + LogisticRegression) depression model, retrained on
the leakage-free persisted split (results/splits/).

Trains ONLY on results/splits/depression_train.csv. Reports metrics on
results/splits/depression_val.csv for visibility -- there is no
hyperparameter search yet, so validation is diagnostic only, not used to
fit anything. results/splits/depression_test.csv is never loaded here.

Does not overwrite depression_model.pkl / depression_vectorizer.pkl --
writes to depression_model_v2/ instead.
"""

import json
import os
import subprocess

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import clean_text

TRAIN_PATH = "results/splits/depression_train.csv"
VAL_PATH = "results/splits/depression_val.csv"
OUTPUT_DIR = "depression_model_v2"
RANDOM_SEED = 42

ID2LABEL = {0: "not_depressed", 1: "depressed"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def load_split(path):
    df = pd.read_csv(path)
    return df[["clean_text", "is_depression"]].rename(
        columns={"clean_text": "text", "is_depression": "label"}
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)

    train_df["text"] = train_df["text"].apply(clean_text)
    val_df["text"] = val_df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X_train, train_df["label"])

    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(val_df["label"], val_preds)
    val_report = classification_report(val_df["label"], val_preds, output_dict=True)

    print("Validation accuracy:", val_accuracy)
    print(classification_report(val_df["label"], val_preds))

    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer.pkl"))

    config = {
        "script": "src/train_depression_v2.py",
        "model_type": "TfidfVectorizer + LogisticRegression",
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path_used": None,
        "random_seed": RANDOM_SEED,
        "vectorizer_params": {"max_features": 5000},
        "model_params": {"max_iter": 1000, "class_weight": "balanced"},
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "val_accuracy": val_accuracy,
        "val_classification_report": val_report,
        "git_commit": _git_commit(),
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model + vectorizer + training_config.json to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
