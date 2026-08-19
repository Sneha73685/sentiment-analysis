"""Monte Carlo unit-test coverage -- intentionally limited, and why.

src/monte_carlo_simulation.py contains two small, pure, deterministic
helper functions that would otherwise be easy to unit test directly:

  - generate_variation(sentence, rng): inserts a random intensifier word
    given a seeded random.Random instance.
  - risk_to_score(risk): a fixed string->int mapping.

However, src/monte_carlo_simulation.py has a module-level import,
`from src.mental_health_pipeline import analyze_text`, and
src/mental_health_pipeline.py loads two DistilBERT models
(sentiment_bert_model/ [v1], depression_bert_model_v2/ [v2, as of the
Phase E runtime migration]) at MODULE IMPORT TIME (module-level
`DistilBertForSequenceClassification.from_pretrained(...)` calls, not
inside a function) -- verified directly by reading both files.

This means `import src.monte_carlo_simulation`, for any reason, always
loads both BERT models first, regardless of their version. Per this
phase's constraints (no BERT loading in the unit suite unless
unavoidable, no full-pipeline inference), that import is not exercised
here. Testing
generate_variation/risk_to_score in isolation would require either
loading those models anyway or refactoring the module's import
structure -- both out of scope for this minimal test phase (see
docs/development.md's Testing section and Section 16's engineering-gaps
list).

results/monte_carlo_results.csv does not currently exist in this
repository (experiments/run_monte_carlo_batch.py has never been run) and
this test suite does not generate it.
"""

import pytest


def test_monte_carlo_helpers_not_unit_tested_due_to_model_loading_at_import():
    pytest.skip(
        "src.monte_carlo_simulation transitively imports "
        "src.mental_health_pipeline, which loads BERT models at module "
        "import time. Skipped to avoid model loading in the unit suite -- "
        "see this file's module docstring."
    )
