"""Regression tests for experiments.evaluate_models.compute_metrics.

Uses a tiny, hand-verifiable synthetic example only -- never the real
predictions/results under results/. compute_metrics is a pure function
(no file I/O, no model loading), so importing
experiments.evaluate_models for this test does not evaluate or load
either canonical model.

Hand-verified expected values for labels=[0,0,1,1], preds=[0,1,1,1],
probs=[0.1,0.6,0.9,0.8] (one false positive at index 1):
  accuracy  = 3/4               = 0.75
  precision = TP/(TP+FP) = 2/3  ~= 0.6667
  recall    = TP/(TP+FN) = 2/2  = 1.0
  f1        = 2PR/(P+R)         = 0.8
  roc_auc   = 1.0 (probs perfectly rank-separate the two classes)
  confusion_matrix = [[1, 1], [0, 2]]  (rows=actual, cols=predicted, label order [0, 1])
"""

import pytest

from experiments.evaluate_models import compute_metrics

LABELS = [0, 0, 1, 1]
PREDS = [0, 1, 1, 1]
PROBS = [0.1, 0.6, 0.9, 0.8]


def _metrics():
    return compute_metrics(LABELS, PREDS, PROBS)


def test_accuracy():
    assert _metrics()["accuracy"] == pytest.approx(0.75)


def test_precision():
    assert _metrics()["precision"] == pytest.approx(2 / 3)


def test_recall():
    assert _metrics()["recall"] == pytest.approx(1.0)


def test_f1_score():
    assert _metrics()["f1_score"] == pytest.approx(0.8)


def test_roc_auc():
    assert _metrics()["roc_auc"] == pytest.approx(1.0)


def test_confusion_matrix():
    assert _metrics()["confusion_matrix"] == [[1, 1], [0, 2]]
