"""Regression tests protecting the runtime depression-model version.

These tests check path CONSTANTS and their existence on disk only -- they
never load model weights or a tokenizer, so they stay fast and do not
require BERT. Their purpose is narrow: if someone accidentally repoints
runtime inference back to a v1 depression path (or to a nonexistent
path), this suite should catch it.

Sentiment paths are intentionally not covered here -- sentiment remains
on its existing v1 artifacts and is out of scope for src/model_config.py.
"""

import os

from src.model_config import (
    DEPRESSION_BERT_MODEL_PATH,
    DEPRESSION_CLASSICAL_MODEL_PATH,
    DEPRESSION_CLASSICAL_VECTORIZER_PATH,
)


def test_classical_depression_model_path_points_to_v2():
    assert DEPRESSION_CLASSICAL_MODEL_PATH == "depression_model_v2/model.pkl"


def test_classical_depression_vectorizer_path_points_to_v2():
    assert DEPRESSION_CLASSICAL_VECTORIZER_PATH == "depression_model_v2/vectorizer.pkl"


def test_bert_depression_model_path_points_to_v2():
    assert DEPRESSION_BERT_MODEL_PATH == "depression_bert_model_v2"


def test_classical_depression_model_path_exists_on_disk():
    assert os.path.exists(DEPRESSION_CLASSICAL_MODEL_PATH)


def test_classical_depression_vectorizer_path_exists_on_disk():
    assert os.path.exists(DEPRESSION_CLASSICAL_VECTORIZER_PATH)


def test_bert_depression_model_directory_exists_on_disk():
    assert os.path.isdir(DEPRESSION_BERT_MODEL_PATH)


def test_predict_module_imports_centralized_depression_paths():
    import src.predict as predict

    assert predict.DEPRESSION_CLASSICAL_MODEL_PATH == DEPRESSION_CLASSICAL_MODEL_PATH
    assert predict.DEPRESSION_CLASSICAL_VECTORIZER_PATH == DEPRESSION_CLASSICAL_VECTORIZER_PATH


def _source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_predict_depression_bert_module_uses_centralized_path():
    # src/predict_depression_bert.py loads a BERT model at import time
    # (see tests/test_monte_carlo.py for why that coupling exists
    # elsewhere in this codebase), so this module is never imported in
    # this test suite. Instead we check the source text directly for the
    # centralized import and the absence of the old hard-coded v1 path.
    source = _source("src/predict_depression_bert.py")

    assert "from src.model_config import DEPRESSION_BERT_MODEL_PATH" in source
    assert 'model_path = DEPRESSION_BERT_MODEL_PATH' in source
    assert '"depression_bert_model"' not in source


def test_mental_health_pipeline_module_uses_centralized_depression_path():
    # Same reasoning as above: src/mental_health_pipeline.py loads two
    # BERT models at import time, so it is never imported here either.
    source = _source("src/mental_health_pipeline.py")

    assert "from src.model_config import DEPRESSION_BERT_MODEL_PATH" in source
    assert "depression_model_path = DEPRESSION_BERT_MODEL_PATH" in source
    assert '"depression_bert_model"' not in source
    # Sentiment must remain on its existing v1 path -- out of scope for
    # this migration.
    assert 'sentiment_model_path = "sentiment_bert_model"' in source


def test_predict_module_sentiment_paths_remain_v1():
    # src/predict.py's sentiment loading isn't covered by
    # test_predict_module_imports_centralized_depression_paths above
    # (that test only checks the depression constants). Sentiment has no
    # v2 track and must stay on its existing hard-coded v1 literals.
    source = _source("src/predict.py")

    assert 'joblib.load("sentiment_model.pkl")' in source
    assert 'joblib.load("tfidf_vectorizer.pkl")' in source


def test_predict_models_remain_unloaded_until_first_call():
    # predict.py loads models lazily inside _load_models(), called only
    # from predict_all(). Importing the module alone must not load any
    # real model file -- this protects that laziness without ever
    # calling predict_all() or _load_models(), so no real model artifact
    # is read.
    import src.predict as predict

    assert predict.sentiment_model is None
    assert predict.sentiment_vectorizer is None
    assert predict.depression_model is None
    assert predict.depression_vectorizer is None


def test_predict_all_return_schema_unchanged():
    # Static check, not an invocation: predict_all() cannot be safely
    # called here without loading real classical model files. This
    # protects its return shape (a 2-tuple of sentiment, depression)
    # from silently changing.
    source = _source("src/predict.py")

    assert "def predict_all(text):" in source
    assert "return sentiment, depression" in source


def test_predict_depression_bert_return_schema_unchanged():
    # Static check of the return dict's keys -- predict() cannot be
    # safely called here without loading BERT.
    source = _source("src/predict_depression_bert.py")

    assert '"label": label' in source
    assert '"confidence": confidence' in source
    assert '"non_depression_prob": non_dep_prob' in source
    assert '"depression_prob": dep_prob' in source


def test_mental_health_pipeline_analyze_text_return_schema_unchanged():
    # Static check of the result dict's keys -- analyze_text() cannot be
    # safely called here without loading BERT.
    source = _source("src/mental_health_pipeline.py")

    assert '"text": text' in source
    assert '"sentiment": {' in source
    assert '"mental_health": {' in source
    assert '"depression_risk": depression' in source
    assert '"risk_level": risk_level' in source
    assert '"signals_detected": signals' in source
