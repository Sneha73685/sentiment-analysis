"""Deterministic train/val/test split for the depression dataset.

Run this once (or whenever data/depression_dataset.csv changes) to produce a
fixed, stratified, deduplicated split. All future training and evaluation of
the depression models should read the split files produced here rather than
re-splitting the raw CSV, so the held-out test set stays identical across
experiments.
"""

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
SOURCE_PATH = "data/depression_dataset.csv"
OUTPUT_DIR = "results/splits"


def load_source():
    df = pd.read_csv(SOURCE_PATH)
    df = df[["clean_text", "is_depression"]].reset_index(drop=True)
    return df


def remove_duplicates(df):
    before = len(df)
    deduped = df.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    return deduped, removed


def make_split(df):
    train_val, test = train_test_split(
        df,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=df["is_depression"],
    )
    val_fraction_of_remainder = VAL_FRACTION / (1 - TEST_FRACTION)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remainder,
        random_state=RANDOM_SEED,
        stratify=train_val["is_depression"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def class_balance(df):
    counts = df["is_depression"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def assert_no_overlap(train_df, val_df, test_df):
    train_set = set(train_df["clean_text"])
    val_set = set(val_df["clean_text"])
    test_set = set(test_df["clean_text"])
    assert not (train_set & val_set), "train/val overlap detected"
    assert not (train_set & test_set), "train/test overlap detected"
    assert not (val_set & test_set), "val/test overlap detected"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_df = load_source()
    raw_count = len(raw_df)

    deduped_df, duplicates_removed = remove_duplicates(raw_df)

    train_df, val_df, test_df = make_split(deduped_df)
    assert_no_overlap(train_df, val_df, test_df)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "depression_train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "depression_val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "depression_test.csv"), index=False)

    manifest = {
        "source_file": SOURCE_PATH,
        "random_seed": RANDOM_SEED,
        "raw_row_count": raw_count,
        "duplicates_removed": duplicates_removed,
        "deduped_row_count": len(deduped_df),
        "split_fractions": {
            "train": round(1 - TEST_FRACTION - VAL_FRACTION, 4),
            "val": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_balance": {
            "train": class_balance(train_df),
            "val": class_balance(val_df),
            "test": class_balance(test_df),
        },
        "no_overlap_verified": True,
        "notes": (
            "This split governs all FUTURE training/evaluation of the depression "
            "models. depression_model.pkl and depression_bert_model/ were trained "
            "BEFORE this split existed and were not constrained to exclude these "
            "test rows. Metrics computed against those already-trained models are "
            "NOT a valid final generalization estimate -- they only verify that "
            "the evaluation pipeline itself is correct. Valid final numbers "
            "require retraining both models using depression_train.csv / "
            "depression_val.csv only, in a future phase."
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "depression_split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Raw rows:        {raw_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Deduped rows:    {len(deduped_df)}")
    print(f"Train / Val / Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Wrote split files + manifest to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
