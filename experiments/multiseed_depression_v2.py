"""Multi-seed stability experiment for the depression v2 models.

Determines whether the observed performance/calibration gap between
TF-IDF+LogisticRegression and DistilBERT (established in the canonical
seed-42 run) is stable across training seeds, or an artifact of one
particular seed.

Does NOT touch the canonical seed-42 artifacts:
  depression_model_v2/, depression_bert_model_v2/,
  results/splits/depression_test.csv, results/depression_v2_model_comparison.csv,
  results/calibration/, or any existing v2 evaluation output.

Uses the EXISTING canonical split files as-is (no re-splitting):
  results/splits/depression_train.csv
  results/splits/depression_val.csv
  results/splits/depression_test.csv

For each seed in SEEDS, trains a fresh classical model and a fresh BERT
model on depression_train.csv, using depression_val.csv only for
validation / early stopping / temperature fitting, then evaluates ONCE on
the frozen depression_test.csv. The test set is never used to select a
seed, checkpoint, hyperparameter, or temperature -- only for final,
post-hoc measurement.

Hyperparameters are copied verbatim from src/train_depression_v2.py and
src/train_depression_bert_v2.py; only `seed` and output paths vary.
Note: sklearn's LogisticRegression with the default `lbfgs` solver is
deterministic and ignores `random_state` entirely (it only affects
`sag`/`saga`/`liblinear`). The classical model is therefore expected to be
identical across all 5 seeds -- this is verified, not assumed, and is
itself a reported finding rather than a bug.

Temporary BERT checkpoints/final models are written under
results/multiseed/tmp_models/ (outside the canonical model directories)
and are left on disk after the run for inspection; they are not part of
the canonical model set and are safe to delete later.
"""

import inspect
import json
import os
import subprocess
import time
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate as hf_evaluate

from src.preprocess import clean_text

SEEDS = [42, 123, 2024, 3407, 7777]

TRAIN_PATH = "results/splits/depression_train.csv"
VAL_PATH = "results/splits/depression_val.csv"
TEST_PATH = "results/splits/depression_test.csv"

OUTPUT_DIR = "results/multiseed"
TMP_MODEL_DIR = "results/multiseed/tmp_models"

ECE_N_BINS = 10
BERT_MAX_LENGTH = 256
BERT_INFERENCE_BATCH_SIZE = 16

CLASSICAL_VECTORIZER_PARAMS = {"max_features": 5000}
CLASSICAL_MODEL_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}

BERT_TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
    "save_strategy": "epoch",
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "report_to": "none",
    "save_total_limit": 1,  # housekeeping only -- does not affect trained weights
}
EARLY_STOPPING_PATIENCE = 1
ID2LABEL = {0: "not_depressed", 1: "depressed"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split_texts_labels(path):
    df = pd.read_csv(path)
    return df["clean_text"].tolist(), df["is_depression"].tolist()


def load_split_df(path):
    df = pd.read_csv(path)[["clean_text", "is_depression"]]
    return df.rename(columns={"clean_text": "text", "is_depression": "labels"})


# ---------------------------------------------------------------------------
# Classical model: train + inference
# ---------------------------------------------------------------------------

def train_classical(seed, train_texts, train_labels):
    cleaned = [clean_text(t) for t in train_texts]
    vectorizer = TfidfVectorizer(**CLASSICAL_VECTORIZER_PARAMS)
    X_train = vectorizer.fit_transform(cleaned)
    model = LogisticRegression(random_state=seed, **CLASSICAL_MODEL_PARAMS)
    model.fit(X_train, train_labels)
    return model, vectorizer


def predict_classical(model, vectorizer, texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    probs = model.predict_proba(vec)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ---------------------------------------------------------------------------
# BERT model: train + inference
# ---------------------------------------------------------------------------

def train_bert(seed, train_df, val_df, checkpoint_dir):
    set_seed(seed)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    train_ds = train_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=BERT_MAX_LENGTH), batched=True
    )
    val_ds = val_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=BERT_MAX_LENGTH), batched=True
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(DEVICE)

    kwargs = dict(BERT_TRAINING_ARGS)
    kwargs["output_dir"] = checkpoint_dir
    kwargs["seed"] = seed
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    training_args = TrainingArguments(**kwargs)

    metric = hf_evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()
    return trainer, tokenizer


def get_bert_logits(model, tokenizer, texts):
    model.eval()
    all_logits = []
    for start in range(0, len(texts), BERT_INFERENCE_BATCH_SIZE):
        batch = texts[start : start + BERT_INFERENCE_BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=BERT_MAX_LENGTH
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    val_labels = np.asarray(val_labels)

    def nll_for_temperature(t):
        probs = softmax(val_logits / t)[:, 1]
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        return log_loss(val_labels, probs)

    result = minimize_scalar(nll_for_temperature, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Calibration metrics (self-contained; not imported from other experiments)
# ---------------------------------------------------------------------------

def confidence_and_correctness(probs_pos, labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    confidence = np.where(preds == 1, probs_pos, 1 - probs_pos)
    correct = preds == labels
    return confidence, correct


def expected_calibration_error(confidence, correct, n_bins=ECE_N_BINS):
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi) if i == n_bins - 1 else (confidence >= lo) & (confidence < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_row_metrics(model_name, seed, probs_pos, labels, preds):
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    probs_pos = np.asarray(probs_pos)
    probs_clipped = np.clip(probs_pos, 1e-12, 1 - 1e-12)

    accuracy = accuracy_score(labels_arr, preds_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_arr, preds_arr, average="binary")
    roc_auc = roc_auc_score(labels_arr, probs_pos)
    ll = log_loss(labels_arr, probs_clipped)
    brier = brier_score_loss(labels_arr, probs_pos)

    confidence, correct = confidence_and_correctness(probs_pos, labels_arr, preds_arr)
    ece = expected_calibration_error(confidence, correct)
    conf_correct = float(confidence[correct].mean()) if correct.sum() else None
    conf_incorrect = float(confidence[~correct].mean()) if (~correct).sum() else None

    return {
        "model": model_name,
        "seed": seed,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "log_loss": round(float(ll), 4),
        "brier": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "correct_confidence": round(conf_correct, 4) if conf_correct is not None else None,
        "incorrect_confidence": round(conf_incorrect, 4) if conf_incorrect is not None else None,
        "num_errors": int((~correct).sum()),
        "temperature": None,
        "calibrated_log_loss": None,
        "calibrated_brier": None,
        "calibrated_ece": None,
        "calibrated_incorrect_confidence": None,
    }


# ---------------------------------------------------------------------------
# Cross-seed summary statistics
# ---------------------------------------------------------------------------

def mean_std_ci95(values):
    """95% CI for the MEAN ACROSS SEEDS (t-distribution, n=len(values)).
    This is NOT a bootstrap-over-test-examples CI -- it treats each seed's
    point estimate as one observation and describes seed-to-seed spread."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        sem = std / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low, ci_high = mean - t_crit * sem, mean + t_crit * sem
    else:
        ci_low, ci_high = mean, mean
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "n": n,
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def series_for(df, model, col):
    sub = df[df["model"] == model].set_index("seed")[col]
    return sub.reindex(SEEDS)


def plot_metric_by_seed(results_df, col, title, ylabel, output_path, ylim=None, extra_series=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = list(range(len(SEEDS)))  # evenly-spaced categorical positions -- seed values are not
    # uniformly spaced (42, 123, 2024, 3407, 7777), so plotting by raw seed value crowds small seeds together
    classical_vals = series_for(results_df, "classical", col)
    bert_vals = series_for(results_df, "bert", col)
    ax.plot(x_positions, classical_vals.values, marker="o", label="classical", color="#4C72B0")
    ax.plot(x_positions, bert_vals.values, marker="o", label="bert (raw)", color="#C44E52")
    if extra_series:
        for label, series_col, color in extra_series:
            vals = series_for(results_df, "bert", series_col)
            ax.plot(x_positions, vals.values, marker="s", linestyle="--", label=label, color=color)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_MODEL_DIR, exist_ok=True)

    test_mtime_before = os.path.getmtime(TEST_PATH)

    train_texts, train_labels = load_split_texts_labels(TRAIN_PATH)
    val_texts, val_labels = load_split_texts_labels(VAL_PATH)
    test_texts, test_labels = load_split_texts_labels(TEST_PATH)
    train_df = load_split_df(TRAIN_PATH)
    val_df = load_split_df(VAL_PATH)

    print(f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}")
    print(f"Seeds: {SEEDS}")

    rows = []
    failures = []

    for seed in SEEDS:
        seed_start = time.time()
        print(f"\n=== SEED {seed} ===")

        # --- Classical ---
        try:
            model_c, vectorizer_c = train_classical(seed, train_texts, train_labels)
            preds_c, probs_c = predict_classical(model_c, vectorizer_c, test_texts)
            row_c = compute_row_metrics("classical", seed, probs_c, test_labels, preds_c)
            rows.append(row_c)
            print(f"  classical: acc={row_c['accuracy']} f1={row_c['f1']} ece={row_c['ece']}")
        except Exception as e:
            print(f"  CLASSICAL SEED {seed} FAILED: {e}")
            failures.append({"model": "classical", "seed": seed, "error": str(e), "traceback": traceback.format_exc()})

        # --- BERT ---
        try:
            checkpoint_dir = os.path.join(TMP_MODEL_DIR, f"bert_checkpoints_seed_{seed}")
            final_dir = os.path.join(TMP_MODEL_DIR, f"bert_final_seed_{seed}")

            trainer, tokenizer = train_bert(seed, train_df, val_df, checkpoint_dir)
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)

            model_b = trainer.model
            val_logits = get_bert_logits(model_b, tokenizer, val_texts)
            test_logits = get_bert_logits(model_b, tokenizer, test_texts)

            temperature = fit_temperature(val_logits, val_labels)  # fit on VAL ONLY

            test_probs_raw = softmax(test_logits)[:, 1]
            test_preds_raw = np.argmax(test_logits, axis=1)

            test_probs_cal = softmax(test_logits / temperature)[:, 1]
            test_preds_cal = np.argmax(test_logits / temperature, axis=1)

            accuracy_unchanged = bool(np.array_equal(test_preds_raw, test_preds_cal))
            if not accuracy_unchanged:
                failures.append({
                    "model": "bert_temperature_check", "seed": seed,
                    "error": "predicted classes changed after temperature scaling -- should be impossible for T>0",
                })

            row_b = compute_row_metrics("bert", seed, test_probs_raw, test_labels, test_preds_raw)

            cal_confidence, cal_correct = confidence_and_correctness(test_probs_cal, test_labels, test_preds_cal)
            cal_ece = expected_calibration_error(cal_confidence, cal_correct)
            cal_incorrect_conf = float(cal_confidence[~cal_correct].mean()) if (~cal_correct).sum() else None
            cal_probs_clipped = np.clip(test_probs_cal, 1e-12, 1 - 1e-12)

            row_b["temperature"] = round(temperature, 4)
            row_b["calibrated_log_loss"] = round(float(log_loss(test_labels, cal_probs_clipped)), 4)
            row_b["calibrated_brier"] = round(float(brier_score_loss(test_labels, test_probs_cal)), 4)
            row_b["calibrated_ece"] = round(float(cal_ece), 4)
            row_b["calibrated_incorrect_confidence"] = round(cal_incorrect_conf, 4) if cal_incorrect_conf is not None else None
            row_b["accuracy_unchanged_by_calibration"] = accuracy_unchanged

            rows.append(row_b)
            print(f"  bert: acc={row_b['accuracy']} f1={row_b['f1']} ece={row_b['ece']} "
                  f"T={temperature:.3f} cal_ece={row_b['calibrated_ece']}")
        except Exception as e:
            print(f"  BERT SEED {seed} FAILED: {e}")
            failures.append({"model": "bert", "seed": seed, "error": str(e), "traceback": traceback.format_exc()})

        print(f"  seed {seed} total time: {time.time() - seed_start:.1f}s")

    test_mtime_after = os.path.getmtime(TEST_PATH)
    test_untouched = test_mtime_before == test_mtime_after
    print(f"\nTest set mtime unchanged: {test_untouched}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "multiseed_results.csv"), index=False)

    # --- Summary ---
    metric_cols_common = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier", "ece",
        "correct_confidence", "incorrect_confidence", "num_errors",
    ]
    metric_cols_bert_only = [
        "temperature", "calibrated_log_loss", "calibrated_brier", "calibrated_ece", "calibrated_incorrect_confidence",
    ]

    summary_rows = []
    for model_name in ["classical", "bert"]:
        sub = results_df[results_df["model"] == model_name]
        for col in metric_cols_common:
            stats_dict = mean_std_ci95(sub[col].tolist())
            summary_rows.append({"model": model_name, "metric": col, **stats_dict})
        if model_name == "bert":
            for col in metric_cols_bert_only:
                stats_dict = mean_std_ci95(sub[col].tolist())
                summary_rows.append({"model": model_name, "metric": col, **stats_dict})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "multiseed_summary.csv"), index=False)

    # --- Plots ---
    plot_metric_by_seed(
        results_df, "accuracy", "Accuracy by seed", "accuracy",
        os.path.join(OUTPUT_DIR, "accuracy_by_seed.png"), ylim=(0.9, 1.0),
    )
    plot_metric_by_seed(
        results_df, "f1", "F1 by seed", "f1",
        os.path.join(OUTPUT_DIR, "f1_by_seed.png"), ylim=(0.9, 1.0),
    )
    plot_metric_by_seed(
        results_df, "ece", "ECE by seed (lower = better calibrated)", "ECE",
        os.path.join(OUTPUT_DIR, "ece_by_seed.png"), ylim=(0.0, 0.15),
        extra_series=[("bert (calibrated)", "calibrated_ece", "#DD8452")],
    )
    plot_metric_by_seed(
        results_df, "incorrect_confidence", "Mean confidence on errors, by seed", "confidence on errors",
        os.path.join(OUTPUT_DIR, "error_confidence_by_seed.png"), ylim=(0.5, 1.0),
        extra_series=[("bert (calibrated)", "calibrated_incorrect_confidence", "#DD8452")],
    )

    # --- JSON report ---
    report = {
        "git_commit": _git_commit(),
        "seeds": SEEDS,
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "test_set_untouched": test_untouched,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "ece_n_bins": ECE_N_BINS,
        "classical_hyperparameters": {**CLASSICAL_VECTORIZER_PARAMS, **CLASSICAL_MODEL_PARAMS},
        "bert_hyperparameters": {**BERT_TRAINING_ARGS, "early_stopping_patience": EARLY_STOPPING_PATIENCE, "max_length": BERT_MAX_LENGTH},
        "note_classical_determinism": (
            "LogisticRegression's default solver (lbfgs) is deterministic and does not use "
            "random_state; classical results are expected to be identical across all seeds. "
            "This is verified below (see per-seed rows), not assumed."
        ),
        "failures": failures,
        "n_seed_failures": len(failures),
        "summary": summary_rows,
    }
    with open(os.path.join(OUTPUT_DIR, "multiseed_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    if failures:
        print(f"\n{len(failures)} FAILURE(S) RECORDED -- see multiseed_report.json['failures']")
    print(f"\nWrote all outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
