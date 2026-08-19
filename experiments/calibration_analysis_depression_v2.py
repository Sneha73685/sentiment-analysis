"""Calibration and uncertainty analysis of the frozen depression v2 models.

Read-only with respect to models and data: no retraining, no weight
changes, no modification of results/splits/depression_test.csv. The only
thing "fit" here is a single scalar temperature for BERT, learned
exclusively on results/splits/depression_val.csv and then applied, frozen,
to the test set.

Models:
  A. depression_model_v2/  (TF-IDF + LogisticRegression)
  B. depression_bert_model_v2/  (DistilBERT)

Methodology:
  1. Run both models on depression_val.csv and depression_test.csv
     (forward pass / .predict_proba only -- nothing is fit on model
     inputs except the temperature, which is fit on val logits only).
  2. Learn BERT's temperature T by minimizing NLL on the VAL logits.
  3. Freeze T, apply it once to TEST logits. Verify argmax (predicted
     class) is unchanged by construction (temperature scaling of positive
     T is a monotonic rescaling of logits, so it cannot flip an argmax).
  4. Report accuracy / log-loss / Brier / ECE / reliability diagrams on
     the frozen TEST set for: classical (raw), BERT (raw), BERT
     (temperature-calibrated). Classical is diagnosed on VAL only (to
     decide whether it warrants calibration) but never itself calibrated
     here -- that recommendation is reported, not auto-applied.
  5. Bootstrap resampling (999 resamples, seed 42) of the TEST set gives
     95% CIs for ECE and Brier score.

Terminology note: "confidence" throughout means the model's own predicted
probability of its predicted class (max(p, 1-p)). This is NOT a claim
about clinical risk or true probability of depression -- see
CALIBRATION_REPORT.md.
"""

import json
import os
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.preprocess import clean_text

VAL_PATH = "results/splits/depression_val.csv"
TEST_PATH = "results/splits/depression_test.csv"
CLASSICAL_MODEL_DIR = "depression_model_v2"
BERT_MODEL_DIR = "depression_bert_model_v2"
OUTPUT_DIR = "results/calibration"

BERT_BATCH_SIZE = 16
BERT_MAX_LENGTH = 256
ECE_N_BINS = 10  # equal-width bins over confidence in [0.5, 1.0]
N_BOOTSTRAP = 999
RANDOM_SEED = 42


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data / inference
# ---------------------------------------------------------------------------

def load_split(path):
    df = pd.read_csv(path)
    return df["clean_text"].tolist(), df["is_depression"].tolist()


def run_classical(texts):
    model = joblib.load(os.path.join(CLASSICAL_MODEL_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(CLASSICAL_MODEL_DIR, "vectorizer.pkl"))
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    probs = model.predict_proba(vec)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def run_bert(texts):
    """Returns (preds, probs_of_class1, raw_logits[n,2])."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR).to(device)
    model.eval()

    all_logits = []
    for start in range(0, len(texts), BERT_BATCH_SIZE):
        batch = texts[start : start + BERT_BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=BERT_MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs_class1 = _softmax(logits)[:, 1]
    preds = np.argmax(logits, axis=1)
    return preds, probs_class1, logits


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(val_logits, val_labels):
    """Learn scalar T > 0 minimizing NLL of softmax(logits / T) on val set.
    T is fit ONLY on validation data, never on test data."""
    val_labels = np.asarray(val_labels)

    def nll_for_temperature(t):
        probs = _softmax(val_logits / t)
        probs_pos = np.clip(probs[:, 1], 1e-12, 1 - 1e-12)
        return log_loss(val_labels, probs_pos)

    result = minimize_scalar(nll_for_temperature, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def confidence_and_correctness(probs_pos, labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    confidence = np.where(preds == 1, probs_pos, 1 - probs_pos)
    correct = (preds == labels)
    return confidence, correct


def expected_calibration_error(confidence, correct, n_bins=ECE_N_BINS):
    """Standard ECE (Guo et al. 2017): equal-width bins over confidence in
    [0.5, 1.0] (binary classification confidence never falls below 0.5).
    ECE = sum_b (n_b / N) * |acc_b - avg_conf_b|."""
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    bin_records = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n_b = mask.sum()
        if n_b == 0:
            bin_records.append({"bin_low": lo, "bin_high": hi, "n": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = confidence[mask].mean()
        acc = correct[mask].mean()
        ece += (n_b / n) * abs(acc - avg_conf)
        bin_records.append(
            {"bin_low": round(float(lo), 3), "bin_high": round(float(hi), 3), "n": int(n_b),
             "avg_confidence": round(float(avg_conf), 4), "accuracy": round(float(acc), 4)}
        )
    return float(ece), bin_records


def bootstrap_ci(probs_pos, labels, preds, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs_pos = np.asarray(probs_pos)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(probs_pos[idx], labels[idx], preds[idx]))
    values = np.array(values)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def ece_metric_fn(probs_pos, labels, preds):
    confidence, correct = confidence_and_correctness(probs_pos, labels, preds)
    ece, _ = expected_calibration_error(confidence, correct)
    return ece


def brier_metric_fn(probs_pos, labels, preds):
    return brier_score_loss(labels, probs_pos)


def full_metrics(name, probs_pos, labels, preds, compute_ci=True):
    labels_arr = np.asarray(labels)
    probs_pos_clipped = np.clip(np.asarray(probs_pos), 1e-12, 1 - 1e-12)

    accuracy = accuracy_score(labels_arr, preds)
    nll = log_loss(labels_arr, probs_pos_clipped)
    brier = brier_score_loss(labels_arr, probs_pos)

    confidence, correct = confidence_and_correctness(np.asarray(probs_pos), labels_arr, preds)
    ece, bins = expected_calibration_error(confidence, correct)

    conf_correct = confidence[correct]
    conf_incorrect = confidence[~correct]

    result = {
        "name": name,
        "n": len(labels_arr),
        "accuracy": round(float(accuracy), 4),
        "log_loss": round(float(nll), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "mean_confidence_correct": round(float(conf_correct.mean()), 4) if len(conf_correct) else None,
        "mean_confidence_incorrect": round(float(conf_incorrect.mean()), 4) if len(conf_incorrect) else None,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "reliability_bins": bins,
    }

    if compute_ci:
        ece_lo, ece_hi = bootstrap_ci(probs_pos, labels, preds, ece_metric_fn)
        brier_lo, brier_hi = bootstrap_ci(probs_pos, labels, preds, brier_metric_fn)
        result["ece_ci95"] = [round(ece_lo, 4), round(ece_hi, 4)]
        result["brier_ci95"] = [round(brier_lo, 4), round(brier_hi, 4)]

    return result, confidence, correct


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability_diagram(bins, title, output_path, ece):
    lows = [b["bin_low"] for b in bins if b["n"] > 0]
    accs = [b["accuracy"] for b in bins if b["n"] > 0]
    confs = [b["avg_confidence"] for b in bins if b["n"] > 0]
    counts = [b["n"] for b in bins if b["n"] > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="perfect calibration")
    width = 0.5 / ECE_N_BINS
    ax1.bar(lows, accs, width=width, align="edge", alpha=0.7, edgecolor="black", label="observed accuracy", color="#4C72B0")
    ax1.bar(lows, confs, width=width, align="edge", alpha=0.35, edgecolor="black", label="avg confidence", color="#DD8452")
    ax1.set_ylabel("accuracy / confidence")
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.legend(loc="upper left")

    ax2.bar(lows, counts, width=width, align="edge", color="#55A868", edgecolor="black")
    ax2.set_xlabel("confidence bin")
    ax2.set_ylabel("count")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_correct_vs_incorrect(panels, output_path):
    """panels: list of (title, confidence, correct) tuples."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    bins = np.linspace(0.5, 1.0, 21)
    for ax, (title, confidence, correct) in zip(axes, panels):
        ax.hist(confidence[correct], bins=bins, alpha=0.6, label=f"correct (n={correct.sum()})", color="#4C72B0")
        ax.hist(confidence[~correct], bins=bins, alpha=0.6, label=f"incorrect (n={(~correct).sum()})", color="#C44E52")
        ax.set_title(title)
        ax.set_xlabel("confidence")
        ax.legend()
    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    val_texts, val_labels = load_split(VAL_PATH)
    test_texts, test_labels = load_split(TEST_PATH)
    print(f"Loaded val={len(val_texts)} test={len(test_texts)} rows.")

    # --- Classical: val (diagnostic only) and test (official) ---
    print("Running classical model on val + test (inference only, no fitting)...")
    val_preds_c, val_probs_c = run_classical(val_texts)
    test_preds_c, test_probs_c = run_classical(test_texts)

    # --- BERT: val (for temperature fit) and test (official) ---
    print("Running BERT model on val + test (inference only, no fitting)...")
    val_preds_b, val_probs_b, val_logits_b = run_bert(val_texts)
    test_preds_b, test_probs_b, test_logits_b = run_bert(test_texts)

    # --- Fit temperature on VAL ONLY ---
    print("Fitting temperature on val logits only...")
    temperature = fit_temperature(val_logits_b, val_labels)
    print(f"Learned temperature T = {temperature:.4f}")

    # --- Apply frozen temperature to TEST logits ---
    test_probs_b_cal = _softmax(test_logits_b / temperature)[:, 1]
    test_preds_b_cal = np.argmax(_softmax(test_logits_b / temperature), axis=1)

    accuracy_unchanged = bool(np.array_equal(test_preds_b, test_preds_b_cal))
    print(f"Accuracy unchanged by temperature scaling: {accuracy_unchanged}")
    assert accuracy_unchanged, "Temperature scaling changed predicted classes -- this should be impossible for T>0"

    # --- Metrics: classical (val diagnostic, test official) ---
    classical_val_metrics, classical_val_conf, classical_val_correct = full_metrics(
        "classical_val_diagnostic", val_probs_c, val_labels, val_preds_c, compute_ci=False
    )
    classical_test_metrics, classical_test_conf, classical_test_correct = full_metrics(
        "classical_test", test_probs_c, test_labels, test_preds_c
    )

    # --- Metrics: BERT raw (test) ---
    bert_raw_metrics, bert_raw_conf, bert_raw_correct = full_metrics(
        "bert_raw_test", test_probs_b, test_labels, test_preds_b
    )

    # --- Metrics: BERT calibrated (test) ---
    bert_cal_metrics, bert_cal_conf, bert_cal_correct = full_metrics(
        "bert_calibrated_test", test_probs_b_cal, test_labels, test_preds_b_cal
    )
    bert_cal_metrics["temperature"] = round(temperature, 4)

    # --- Save calibration_metrics.csv ---
    metrics_rows = []
    for m in [classical_val_metrics, classical_test_metrics, bert_raw_metrics, bert_cal_metrics]:
        row = {k: v for k, v in m.items() if k != "reliability_bins"}
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_metrics.csv"), index=False)

    # --- Save calibration_predictions.csv (test set only) ---
    pred_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": test_labels,
            "pred_classical": test_preds_c,
            "prob_classical": test_probs_c,
            "conf_classical": classical_test_conf,
            "pred_bert_raw": test_preds_b,
            "prob_bert_raw": test_probs_b,
            "conf_bert_raw": bert_raw_conf,
            "pred_bert_calibrated": test_preds_b_cal,
            "prob_bert_calibrated": test_probs_b_cal,
            "conf_bert_calibrated": bert_cal_conf,
        }
    )
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_predictions.csv"), index=False)

    # --- Reliability diagrams ---
    plot_reliability_diagram(
        classical_test_metrics["reliability_bins"],
        "Classical (TF-IDF+LR) -- Test Set, Raw",
        os.path.join(OUTPUT_DIR, "reliability_diagram_classical.png"),
        classical_test_metrics["ece"],
    )
    plot_reliability_diagram(
        bert_raw_metrics["reliability_bins"],
        "BERT -- Test Set, Raw (T=1.0)",
        os.path.join(OUTPUT_DIR, "reliability_diagram_bert.png"),
        bert_raw_metrics["ece"],
    )
    plot_reliability_diagram(
        bert_cal_metrics["reliability_bins"],
        f"BERT -- Test Set, Temperature-Calibrated (T={temperature:.3f})",
        os.path.join(OUTPUT_DIR, "reliability_diagram_bert_calibrated.png"),
        bert_cal_metrics["ece"],
    )

    # --- Confidence correct vs incorrect (3-panel) ---
    plot_confidence_correct_vs_incorrect(
        [
            ("Classical (raw)", classical_test_conf, classical_test_correct),
            ("BERT (raw)", bert_raw_conf, bert_raw_correct),
            ("BERT (calibrated)", bert_cal_conf, bert_cal_correct),
        ],
        os.path.join(OUTPUT_DIR, "confidence_correct_vs_incorrect.png"),
    )

    # --- Classical calibration recommendation (evidence-based, not applied) ---
    classical_recommendation = {
        "val_ece": classical_val_metrics["ece"],
        "val_brier": classical_val_metrics["brier_score"],
        "test_ece": classical_test_metrics["ece"],
        "test_brier": classical_test_metrics["brier_score"],
        "assessment": (
            "reasonably calibrated -- ECE/Brier are low and consistent between val and test"
            if classical_val_metrics["ece"] < 0.05 and classical_test_metrics["ece"] < 0.05
            else "some miscalibration observed -- consider Platt scaling / isotonic regression fit on val "
            "in a future phase (NOT applied here per instructions)"
        ),
        "calibration_applied": False,
    }

    # --- JSON report ---
    report = {
        "git_commit": _git_commit(),
        "val_set_path": VAL_PATH,
        "test_set_path": TEST_PATH,
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "ece_n_bins": ECE_N_BINS,
        "bootstrap_n": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "temperature_learned_on_val": round(temperature, 4),
        "accuracy_unchanged_by_temperature_scaling": accuracy_unchanged,
        "classical": {
            "val_diagnostic": classical_val_metrics,
            "test": classical_test_metrics,
            "calibration_recommendation": classical_recommendation,
        },
        "bert": {
            "raw_test": bert_raw_metrics,
            "calibrated_test": bert_cal_metrics,
        },
        "disclaimer": (
            "All 'confidence' values in this analysis are raw model predictive "
            "probabilities (softmax / logistic outputs), not calibrated estimates "
            "of clinical risk or true probability of depression. They describe "
            "how confident the MODEL is in its own prediction, nothing more."
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Summary ===")
    print(metrics_df[["name", "n", "accuracy", "log_loss", "brier_score", "ece"]].to_string(index=False))
    print(f"\nWrote all outputs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
