import re

SIGNAL_PATTERNS = {

    "hopelessness": [
        "no hope",
        "hopeless",
        "nothing will get better",
        "no future"
    ],

    "worthlessness": [
        "i feel useless",
        "i am worthless",
        "i hate myself"
    ],

    "emotional_numbness": [
        "i feel empty",
        "i feel nothing",
        "numb"
    ],

    "fatigue": [
        "i am tired of everything",
        "exhausted",
        "burned out"
    ]
}


_COMPILED_PATTERNS = {
    signal: [re.compile(r"\b" + re.escape(phrase) + r"\b") for phrase in phrases]
    for signal, phrases in SIGNAL_PATTERNS.items()
}


def detect_signals(text):

    text = text.lower()
    detected = []

    for signal, patterns in _COMPILED_PATTERNS.items():

        for pattern in patterns:

            if pattern.search(text):
                detected.append(signal)
                break

    return detected