import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_stop_words = None
_lemmatizer = None


def _init_nltk_tools():
    global _stop_words, _lemmatizer
    if _stop_words is not None and _lemmatizer is not None:
        return
    try:
        _stop_words = set(stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
    except LookupError as exc:
        raise LookupError(
            "Missing NLTK resources. Run: "
            "python -m nltk.downloader punkt stopwords wordnet omw-1.4"
        ) from exc

def clean_text(text):
    _init_nltk_tools()
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError as exc:
        raise LookupError(
            "Missing NLTK punkt tokenizer. Run: "
            "python -m nltk.downloader punkt"
        ) from exc
    tokens = [_lemmatizer.lemmatize(word) for word in tokens if word not in _stop_words]
    return " ".join(tokens)
