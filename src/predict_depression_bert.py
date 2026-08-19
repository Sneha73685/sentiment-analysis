import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.model_config import DEPRESSION_BERT_MODEL_PATH

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_path = DEPRESSION_BERT_MODEL_PATH

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
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()
    non_dep_prob = probs[0][0].item()
    dep_prob = probs[0][1].item()

    label = "Depressed" if pred == 1 else "Not Depressed"

    return {
        "label": label,
        "confidence": confidence,
        "non_depression_prob": non_dep_prob,
        "depression_prob": dep_prob,
    }


if __name__ == "__main__":
    print("Depression Detection Model Ready")
    while True:
        text = input("\nEnter a sentence (or type 'exit'): ")
        if text.lower() == "exit":
            break

        result = predict(text)
        if result is None:
            print("Empty input detected. Try again.\n")
            continue

        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Non-Depression Probability: {result['non_depression_prob']:.4f}")
        print(f"Depression Probability: {result['depression_prob']:.4f}")
        print("-" * 50)
