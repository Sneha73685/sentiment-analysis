"""Centralized runtime model-path constants for the depression models.

Plain path constants only -- no loading logic, no third-party imports.
These are the canonical v2 depression artifacts (see MODEL_CARD.md,
docs/models.md); sentiment paths are intentionally not included here --
sentiment remains on its existing v1 artifacts, hardcoded in
src/predict.py, src/predict_bert.py, and src/mental_health_pipeline.py,
and is out of scope for this module.
"""

DEPRESSION_CLASSICAL_MODEL_PATH = "depression_model_v2/model.pkl"
DEPRESSION_CLASSICAL_VECTORIZER_PATH = "depression_model_v2/vectorizer.pkl"
DEPRESSION_BERT_MODEL_PATH = "depression_bert_model_v2"
