import joblib
from src.preprocess import clean_text

sentiment_model = None
sentiment_vectorizer = None
depression_model = None
depression_vectorizer = None


def _load_models():
    global sentiment_model, sentiment_vectorizer
    global depression_model, depression_vectorizer
    if sentiment_model is not None and depression_model is not None:
        return
    sentiment_model = joblib.load("sentiment_model.pkl")
    sentiment_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    depression_model = joblib.load("depression_model.pkl")
    depression_vectorizer = joblib.load("depression_vectorizer.pkl")


def predict_all(text):
    _load_models()
    cleaned = clean_text(text)

    sent_vec = sentiment_vectorizer.transform([cleaned])
    sent_pred = sentiment_model.predict(sent_vec)[0]
    sentiment = "Positive" if sent_pred == 1 else "Negative"

    dep_vec = depression_vectorizer.transform([cleaned])
    dep_pred = depression_model.predict(dep_vec)[0]
    depression = "Depressed" if dep_pred == 1 else "Not Depressed"

    return sentiment, depression


if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or type 'exit'): ")
        if text.lower() == "exit":
            break
        if not text.strip():
            print("Empty input detected. Try again.")
            print("-" * 40)
            continue

        sentiment, depression = predict_all(text)

        print("Sentiment:", sentiment)
        print("Mental Health Risk:", depression)
        print("-" * 40)
