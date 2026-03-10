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


def detect_signals(text):

    text = text.lower()
    detected = []

    for signal, phrases in SIGNAL_PATTERNS.items():

        for phrase in phrases:

            if phrase in text:
                detected.append(signal)
                break

    return detected