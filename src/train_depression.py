import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.preprocess import clean_text


def main():
    df = pd.read_csv("data/depression_dataset.csv")

    df = df[["clean_text", "is_depression"]]
    df.columns = ["text", "label"]

    df["text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, "depression_model.pkl")
    joblib.dump(vectorizer, "depression_vectorizer.pkl")


if __name__ == "__main__":
    main()
