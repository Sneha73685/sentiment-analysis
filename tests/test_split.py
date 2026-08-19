"""Regression tests for src.depression_split's reusable split functions.

Uses a small synthetic in-memory DataFrame only -- never
data/depression_dataset.csv and never results/splits/. These tests do
not call main() or load_source(), so nothing is read from or written to
disk; the canonical split files are never touched.

Covers: reproducibility (same seed -> same split), zero train/val/test
overlap (and that assert_no_overlap actually detects a forced overlap),
deduplication before splitting, and stratification.
"""

import pandas as pd
import pytest

from src.depression_split import (
    assert_no_overlap,
    class_balance,
    make_split,
    remove_duplicates,
)


def _synthetic_dataset():
    """40 unique rows (20 per class) plus one exact duplicate row, so both
    deduplication and stratified splitting have enough data to exercise
    meaningfully."""
    rows = []
    for i in range(20):
        rows.append({"clean_text": f"not depressed text number {i}", "is_depression": 0})
        rows.append({"clean_text": f"depressed text number {i}", "is_depression": 1})
    rows.append({"clean_text": "not depressed text number 0", "is_depression": 0})  # duplicate
    return pd.DataFrame(rows)


def test_remove_duplicates_drops_exact_duplicate_rows():
    df = _synthetic_dataset()
    assert len(df) == 41

    deduped, removed = remove_duplicates(df)

    assert removed == 1
    assert len(deduped) == 40
    assert deduped["clean_text"].duplicated().sum() == 0


def test_make_split_is_reproducible_given_same_seed():
    deduped, _ = remove_duplicates(_synthetic_dataset())

    train_a, val_a, test_a = make_split(deduped)
    train_b, val_b, test_b = make_split(deduped)

    assert set(train_a["clean_text"]) == set(train_b["clean_text"])
    assert set(val_a["clean_text"]) == set(val_b["clean_text"])
    assert set(test_a["clean_text"]) == set(test_b["clean_text"])


def test_make_split_produces_zero_overlap():
    deduped, _ = remove_duplicates(_synthetic_dataset())
    train, val, test = make_split(deduped)

    # Should not raise -- this is the same invariant check used by
    # src/depression_split.py's own main() before writing the canonical
    # split files.
    assert_no_overlap(train, val, test)


def test_assert_no_overlap_detects_a_forced_overlap():
    deduped, _ = remove_duplicates(_synthetic_dataset())
    train, val, test = make_split(deduped)

    # Deliberately reintroduce a training row into validation to confirm
    # the overlap check actually catches it rather than passing silently.
    contaminated_val = pd.concat([val, train.iloc[[0]]], ignore_index=True)

    with pytest.raises(AssertionError):
        assert_no_overlap(train, contaminated_val, test)


def test_make_split_keeps_both_classes_represented_in_every_split():
    deduped, _ = remove_duplicates(_synthetic_dataset())
    train, val, test = make_split(deduped)

    for split_df in (train, val, test):
        balance = class_balance(split_df)
        assert balance.get("0", 0) > 0
        assert balance.get("1", 0) > 0
