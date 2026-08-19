"""Read-only error analysis of the frozen depression v2 models' predictions
on the locked held-out test set (results/splits/depression_test.csv).

Does NOT retrain, tune, or modify either model, and does NOT touch the test
set -- it only reads the two prediction files already produced by
experiments/evaluate_models.py:

    results/predictions_depression_v2_classical.csv
    results/predictions_depression_v2_bert.csv

All outputs are written under results/error_analysis/.

Design choices (documented here so the analysis is reproducible and its
thresholds are not hidden):
  - "confidence" for a prediction = the model's own probability assigned to
    the predicted class, i.e. max(p, 1-p). Ranges [0.5, 1.0].
  - LOW_CONFIDENCE_THRESHOLD = 0.65 -- a correct prediction is "low
    confidence" if the model's confidence in it is below this.
  - STRONG_DISAGREEMENT_THRESHOLD = 0.85 -- two models "strongly disagree"
    on a sample if they predict different classes AND both have confidence
    above this threshold in their own (opposing) predictions.
  - Linguistic pattern flags are regex/keyword heuristics, not validated
    NLP classifiers. Several (sarcasm, metaphor, ambiguity, register) are
    explicitly weak proxies for the underlying phenomenon; this is called
    out per-pattern in the report rather than treated as ground truth.
  - Statistical association between a pattern and model error is tested
    with Fisher's exact test on the 2x2 (pattern present/absent) x
    (error/correct) table, per model. With ~11 patterns x 2 models = 22
    tests, a Bonferroni-corrected threshold of 0.05/22 ~= 0.00227 is also
    reported alongside the raw p-value.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CLASSICAL_PRED_PATH = "results/predictions_depression_v2_classical.csv"
BERT_PRED_PATH = "results/predictions_depression_v2_bert.csv"
OUTPUT_DIR = "results/error_analysis"
RANDOM_SEED = 42
N_EXAMPLES_PER_CATEGORY = 5
EXAMPLE_TRUNCATE_CHARS = 200

LOW_CONFIDENCE_THRESHOLD = 0.65
STRONG_DISAGREEMENT_THRESHOLD = 0.85

LABEL_NAMES = {0: "not_depressed", 1: "depressed"}


# ---------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------

def load_and_verify():
    c = pd.read_csv(CLASSICAL_PRED_PATH)
    b = pd.read_csv(BERT_PRED_PATH)

    assert len(c) == len(b) == 1148, f"expected 1148 rows each, got {len(c)}/{len(b)}"
    assert (c["text"] == b["text"]).all(), "text columns diverge between prediction files"
    assert (c["true_label"] == b["true_label"]).all(), "true_label columns diverge"

    df = pd.DataFrame(
        {
            "text": c["text"],
            "true_label": c["true_label"],
            "pred_classical": c["prediction"],
            "prob_classical": c["probability_depressed"],
            "pred_bert": b["prediction"],
            "prob_bert": b["probability_depressed"],
        }
    )
    df["conf_classical"] = np.where(
        df["pred_classical"] == 1, df["prob_classical"], 1 - df["prob_classical"]
    )
    df["conf_bert"] = np.where(df["pred_bert"] == 1, df["prob_bert"], 1 - df["prob_bert"])
    df["correct_classical"] = df["pred_classical"] == df["true_label"]
    df["correct_bert"] = df["pred_bert"] == df["true_label"]
    df["word_count"] = df["text"].str.split().str.len()

    return df


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

def build_categories(df):
    cats = {}
    cats["classical_false_positives"] = df[(df["true_label"] == 0) & (df["pred_classical"] == 1)]
    cats["classical_false_negatives"] = df[(df["true_label"] == 1) & (df["pred_classical"] == 0)]
    cats["bert_false_positives"] = df[(df["true_label"] == 0) & (df["pred_bert"] == 1)]
    cats["bert_false_negatives"] = df[(df["true_label"] == 1) & (df["pred_bert"] == 0)]
    cats["classical_wrong_bert_correct"] = df[~df["correct_classical"] & df["correct_bert"]]
    cats["bert_wrong_classical_correct"] = df[~df["correct_bert"] & df["correct_classical"]]
    cats["both_wrong"] = df[~df["correct_classical"] & ~df["correct_bert"]]
    cats["both_correct_low_confidence"] = df[
        df["correct_classical"]
        & df["correct_bert"]
        & ((df["conf_classical"] < LOW_CONFIDENCE_THRESHOLD) | (df["conf_bert"] < LOW_CONFIDENCE_THRESHOLD))
    ]
    cats["strong_disagreement"] = df[
        (df["pred_classical"] != df["pred_bert"])
        & (df["conf_classical"] > STRONG_DISAGREEMENT_THRESHOLD)
        & (df["conf_bert"] > STRONG_DISAGREEMENT_THRESHOLD)
    ]
    cats["model_disagreements"] = df[df["pred_classical"] != df["pred_bert"]]
    return cats


def confidence_stats(sub_df, col):
    if len(sub_df) == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    s = sub_df[col]
    return {
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
    }


def truncate(text, n=EXAMPLE_TRUNCATE_CHARS):
    text = str(text)
    return text if len(text) <= n else text[:n].rstrip() + "..."


def representative_examples(sub_df, n=N_EXAMPLES_PER_CATEGORY):
    if len(sub_df) == 0:
        return []
    sample = sub_df.sample(n=min(n, len(sub_df)), random_state=RANDOM_SEED)
    out = []
    for _, row in sample.iterrows():
        out.append(
            {
                "text_snippet": truncate(row["text"]),
                "true_label": LABEL_NAMES[int(row["true_label"])],
                "pred_classical": LABEL_NAMES[int(row["pred_classical"])],
                "conf_classical": round(float(row["conf_classical"]), 4),
                "pred_bert": LABEL_NAMES[int(row["pred_bert"])],
                "conf_bert": round(float(row["conf_bert"]), 4),
                "word_count": int(row["word_count"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Linguistic pattern flags (heuristic, regex-based -- see module docstring)
# ---------------------------------------------------------------------------

def compile_patterns():
    negation_re = re.compile(r"\b(not|n't|no|never|nobody|nothing|none|nowhere)\b")
    uncertainty_re = re.compile(r"\b(maybe|i think|i guess|perhaps|not sure|possibly|might|probably)\b")
    first_person_re = re.compile(r"\b(i|i'm|im|me|my|mine|myself)\b")
    third_person_re = re.compile(r"\b(he|she|they|them|his|her|their|someone|somebody|people|my friend|my brother|my sister|my mom|my dad)\b")
    temporary_affect_re = re.compile(
        r"\b(today|right now|lately|this week|tonight|these days)\b.{0,40}\b(tired|sad|exhausted|down|drained)\b"
        r"|\b(tired|sad|exhausted|down|drained)\b.{0,40}\b(today|right now|lately|this week|tonight|these days)\b"
    )
    mixed_sentiment_re_pos = re.compile(r"\b(happy|good|fine|okay|ok|great|better)\b")
    mixed_sentiment_re_neg = re.compile(r"\b(sad|bad|depressed|awful|terrible|worse)\b")
    contextual_mention_re = re.compile(r"\bdepress(ed|ion)\b")
    contextual_self_report_re = re.compile(r"\bi\s+(feel|am|ve been|have been|struggle with|deal with)\b[^.]{0,20}\bdepress")
    sarcasm_re = re.compile(r"(/s\b|yeah,? right|totally fine\b|just great\b|oh joy\b)")
    metaphor_re = re.compile(r"\b(drowning|empty inside|black hole|weight on my chest|hole in my chest|dark cloud|sinking)\b")
    short_word_count = None  # filled in dynamically from data quantiles
    long_word_count = None

    return {
        "negation": lambda t: bool(negation_re.search(t)),
        "uncertainty": lambda t: bool(uncertainty_re.search(t)),
        "third_person_dominant": lambda t: bool(third_person_re.search(t)) and not bool(first_person_re.search(t)),
        "temporary_affect_language": lambda t: bool(temporary_affect_re.search(t)),
        "mixed_sentiment_language": lambda t: bool(mixed_sentiment_re_pos.search(t)) and bool(mixed_sentiment_re_neg.search(t)),
        "contextual_depression_mention_without_self_report": lambda t: bool(contextual_mention_re.search(t))
        and not bool(contextual_self_report_re.search(t)),
        "sarcasm_markers": lambda t: bool(sarcasm_re.search(t)),
        "metaphorical_language": lambda t: bool(metaphor_re.search(t)),
    }


PATTERN_VALIDITY_NOTES = {
    "negation": "Reliable surface-form detection (word-boundary regex on standard negation cues).",
    "uncertainty": "Reasonable surface-form proxy for hedging language.",
    "third_person_dominant": "Weak proxy: flags absence of first-person pronouns + presence of third-person "
    "pronouns/kinship terms. Does not capture true grammatical person/perspective.",
    "temporary_affect_language": "Weak proxy: co-occurrence of a temporal marker and a fatigue/sadness word "
    "within ~40 characters. Does not verify semantic relationship.",
    "mixed_sentiment_language": "Weak proxy for ambiguous/mixed emotional language: co-occurrence of a positive "
    "and negative sentiment word anywhere in the text.",
    "contextual_depression_mention_without_self_report": "Moderate proxy: flags mentions of "
    "'depression'/'depressed' not immediately preceded by a first-person self-report frame. Will misclassify "
    "some genuine self-reports phrased unusually.",
    "sarcasm_markers": "Very weak proxy: a handful of explicit sarcasm cues (e.g. '/s'). Almost certainly low "
    "recall -- most sarcasm in this corpus, if any, will NOT be flagged. A null result here is not evidence "
    "against sarcasm-related errors, only evidence against these specific surface markers.",
    "metaphorical_language": "Very weak proxy: a small fixed list of metaphor phrases common in mental-health "
    "text. Low recall by construction.",
    "very_short_input": "Reliable, purely quantitative (word count <= data-derived 10th percentile).",
    "long_context_heavy_input": "Reliable, purely quantitative (word count >= data-derived 90th percentile).",
}


def run_linguistic_analysis(df):
    patterns = compile_patterns()
    text_lower = df["text"].str.lower()

    flags = {}
    for name, fn in patterns.items():
        flags[name] = text_lower.apply(fn)

    q10 = df["word_count"].quantile(0.10)
    q90 = df["word_count"].quantile(0.90)
    flags["very_short_input"] = df["word_count"] <= q10
    flags["long_context_heavy_input"] = df["word_count"] >= q90

    n_tests = len(flags) * 2
    bonferroni_alpha = 0.05 / n_tests

    rows = []
    for pattern_name, flag in flags.items():
        n_flagged = int(flag.sum())
        for model in ["classical", "bert"]:
            correct_col = f"correct_{model}"
            table = [
                [int((flag & ~df[correct_col]).sum()), int((flag & df[correct_col]).sum())],
                [int((~flag & ~df[correct_col]).sum()), int((~flag & df[correct_col]).sum())],
            ]
            odds_ratio, p_value = fisher_exact(table)
            error_rate_flagged = table[0][0] / n_flagged if n_flagged > 0 else None
            n_unflagged = len(df) - n_flagged
            error_rate_unflagged = table[1][0] / n_unflagged if n_unflagged > 0 else None

            rows.append(
                {
                    "pattern": pattern_name,
                    "model": model,
                    "n_flagged": n_flagged,
                    "proportion_flagged": round(n_flagged / len(df), 4),
                    "error_rate_flagged": round(error_rate_flagged, 4) if error_rate_flagged is not None else None,
                    "error_rate_unflagged": round(error_rate_unflagged, 4) if error_rate_unflagged is not None else None,
                    "odds_ratio": round(float(odds_ratio), 4) if np.isfinite(odds_ratio) else None,
                    "p_value": round(float(p_value), 6),
                    "significant_at_0.05": bool(p_value < 0.05),
                    "significant_after_bonferroni": bool(p_value < bonferroni_alpha),
                    "validity_note": PATTERN_VALIDITY_NOTES.get(pattern_name, ""),
                }
            )

    result_df = pd.DataFrame(rows)
    metadata = {
        "n_tests": n_tests,
        "bonferroni_alpha": round(bonferroni_alpha, 6),
        "word_count_q10": float(q10),
        "word_count_q90": float(q90),
    }
    return result_df, metadata


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confidence_distribution(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model, conf_col, correct_col in [
        (axes[0], "Classical", "conf_classical", "correct_classical"),
        (axes[1], "BERT", "conf_bert", "correct_bert"),
    ]:
        correct_conf = df.loc[df[correct_col], conf_col]
        wrong_conf = df.loc[~df[correct_col], conf_col]
        bins = np.linspace(0.5, 1.0, 21)
        ax.hist(correct_conf, bins=bins, alpha=0.6, label=f"correct (n={len(correct_conf)})", color="#4C72B0")
        ax.hist(wrong_conf, bins=bins, alpha=0.6, label=f"incorrect (n={len(wrong_conf)})", color="#C44E52")
        ax.set_title(f"{model} prediction confidence")
        ax.set_xlabel("confidence (prob. of predicted class)")
        ax.legend()
    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_disagreement(df, output_path):
    plt.figure(figsize=(7, 7))
    colors = np.where(df["true_label"] == 1, "#C44E52", "#4C72B0")
    plt.scatter(df["prob_classical"], df["prob_bert"], c=colors, alpha=0.4, s=12)
    plt.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Classical P(depressed)")
    plt.ylabel("BERT P(depressed)")
    plt.title("Classical vs BERT predicted probability (color = true label)")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", label="true: depressed", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", label="true: not depressed", markersize=8),
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_verify()
    print(f"Loaded and verified {len(df)} matched test predictions.")

    cats = build_categories(df)

    summary_rows = []
    for name, sub in cats.items():
        summary_rows.append(
            {
                "category": name,
                "n": len(sub),
                "proportion_of_test_set": round(len(sub) / len(df), 4),
                "conf_classical_mean": confidence_stats(sub, "conf_classical")["mean"],
                "conf_classical_median": confidence_stats(sub, "conf_classical")["median"],
                "conf_bert_mean": confidence_stats(sub, "conf_bert")["mean"],
                "conf_bert_median": confidence_stats(sub, "conf_bert")["median"],
            }
        )
    error_summary_df = pd.DataFrame(summary_rows)
    error_summary_df.to_csv(os.path.join(OUTPUT_DIR, "error_summary.csv"), index=False)

    file_map = {
        "classical_false_positives": "false_positives_classical.csv",
        "classical_false_negatives": "false_negatives_classical.csv",
        "bert_false_positives": "false_positives_bert.csv",
        "bert_false_negatives": "false_negatives_bert.csv",
        "classical_wrong_bert_correct": "classical_wrong_bert_correct.csv",
        "bert_wrong_classical_correct": "bert_wrong_classical_correct.csv",
        "both_wrong": "both_wrong.csv",
        "both_correct_low_confidence": "both_correct_low_confidence.csv",
        "model_disagreements": "model_disagreements.csv",
        "strong_disagreement": "strong_disagreements.csv",
    }
    for cat_name, filename in file_map.items():
        cats[cat_name].to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    linguistic_df, linguistic_meta = run_linguistic_analysis(df)
    linguistic_df.to_csv(os.path.join(OUTPUT_DIR, "linguistic_pattern_analysis.csv"), index=False)

    plot_confidence_distribution(df, os.path.join(OUTPUT_DIR, "error_confidence_distribution.png"))
    plot_model_disagreement(df, os.path.join(OUTPUT_DIR, "model_disagreement.png"))

    examples = {name: representative_examples(sub) for name, sub in cats.items()}

    report = {
        "test_set_size": len(df),
        "config": {
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "strong_disagreement_threshold": STRONG_DISAGREEMENT_THRESHOLD,
            "random_seed": RANDOM_SEED,
        },
        "overall": {
            "classical_accuracy": round(float(df["correct_classical"].mean()), 4),
            "bert_accuracy": round(float(df["correct_bert"].mean()), 4),
            "classical_errors": int((~df["correct_classical"]).sum()),
            "bert_errors": int((~df["correct_bert"]).sum()),
        },
        "category_summary": summary_rows,
        "linguistic_analysis_metadata": linguistic_meta,
        "representative_examples": examples,
    }
    with open(os.path.join(OUTPUT_DIR, "error_analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Wrote all outputs to", OUTPUT_DIR)
    print(error_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
