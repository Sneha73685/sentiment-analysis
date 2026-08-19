"""Regression tests for src.preprocess.clean_text.

These tests document the actual current behavior of clean_text (verified
directly against the implementation) -- they do not invent expected
behavior. Requires the NLTK resources referenced in README.md /
docs/reproducibility.md (`python -m nltk.downloader punkt stopwords
wordnet omw-1.4`); if those resources are missing, clean_text itself
raises a LookupError with instructions, which will surface as a clear
test failure rather than a silent skip.
"""

from src.preprocess import clean_text


def test_clean_text_is_deterministic():
    text = "I LOVE running, and Jumping!! Visit http://example.com now."
    assert clean_text(text) == clean_text(text)


def test_clean_text_lowercases_strips_urls_and_punctuation():
    result = clean_text("Visit http://example.com or www.example.org NOW!!!")
    assert result == result.lower()
    assert "http" not in result
    assert "www" not in result
    assert "!" not in result
    assert "." not in result


def test_clean_text_removes_stopwords():
    # "the" and "and" are English stopwords removed via nltk.corpus.stopwords.
    result = clean_text("the cat and the dog are running")
    tokens = result.split()
    assert "the" not in tokens
    assert "and" not in tokens


def test_clean_text_lemmatizes_plural_nouns():
    # WordNetLemmatizer.lemmatize(word) defaults to noun POS in clean_text
    # (no pos= argument is passed), so plural nouns are reduced to singular.
    assert clean_text("dogs") == "dog"


def test_clean_text_empty_string_returns_empty_string():
    assert clean_text("") == ""
