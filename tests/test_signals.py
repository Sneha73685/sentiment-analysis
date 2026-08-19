"""Regression tests for src.mental_health_signals.detect_signals.

The most important test in this module is
test_numb_does_not_match_number: it protects the word-boundary fix that
replaced naive substring matching (which incorrectly matched "numb"
inside "number") with compiled \\b-anchored regex matching. If someone
later reverts to substring matching, this test fails.

These tests check implementation correctness only (does the function
match what its own regex patterns say it should match), not whether the
hand-curated signal vocabulary is clinically meaningful -- that is out of
scope, see RESPONSIBLE_USE.md.
"""

from src.mental_health_signals import detect_signals


def test_numb_does_not_match_number():
    # Regression test for the word-boundary bug: "number" contains the
    # substring "numb" but must not trigger emotional_numbness.
    assert detect_signals("three numbers") == []


def test_numb_matches_as_a_standalone_word():
    assert "emotional_numbness" in detect_signals("I feel numb")


def test_detect_signals_is_case_insensitive():
    assert detect_signals("I FEEL NUMB") == detect_signals("i feel numb")


def test_positive_hopelessness_signal():
    assert "hopelessness" in detect_signals("there is no hope for tomorrow")


def test_negative_example_returns_no_signals():
    assert detect_signals("I love sunny days") == []


def test_multiple_signals_can_be_detected_in_one_text():
    detected = detect_signals("I am exhausted and I feel useless")
    assert "fatigue" in detected
    assert "worthlessness" in detected
