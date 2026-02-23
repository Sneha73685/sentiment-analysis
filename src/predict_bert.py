import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_path = "sentiment_bert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()


def predict(text):
    if not text.strip():
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

    negative_prob = probs[0].item()
    positive_prob = probs[1].item()

    if positive_prob > negative_prob:
        label = "Positive"
        confidence = positive_prob
    else:
        label = "Negative"
        confidence = negative_prob

    return {
        "label": label,
        "confidence": confidence,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob
    }


if __name__ == "__main__":
    print("BERT Sentiment Analyzer Ready\n")

    while True:
        sentence = input("Enter a sentence (or type 'exit'): ")

        if sentence.lower() == "exit":
            break

        result = predict(sentence)

        if result is None:
            print("Empty input detected. Try again.\n")
            continue

        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Positive Probability: {result['positive_prob']:.4f}")
        print(f"Negative Probability: {result['negative_prob']:.4f}")
        print("-" * 50)