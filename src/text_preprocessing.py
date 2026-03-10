from spellchecker import SpellChecker

spell = SpellChecker()

def correct_text(text):

    words = text.split()

    corrected = []

    for word in words:

        corrected_word = spell.correction(word)

        if corrected_word:
            corrected.append(corrected_word)
        else:
            corrected.append(word)

    return " ".join(corrected)