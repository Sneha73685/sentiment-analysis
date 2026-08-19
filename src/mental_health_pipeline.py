import torch
import torch.nn.functional as F
from src.mental_health_signals import detect_signals
from src.model_config import DEPRESSION_BERT_MODEL_PATH
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

sentiment_model_path = "sentiment_bert_model"
depression_model_path = DEPRESSION_BERT_MODEL_PATH

sent_tokenizer = DistilBertTokenizerFast.from_pretrained(sentiment_model_path)
sent_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_path).to(device)

dep_tokenizer = DistilBertTokenizerFast.from_pretrained(depression_model_path)
dep_model = DistilBertForSequenceClassification.from_pretrained(depression_model_path).to(device)

sent_model.eval()
dep_model.eval()

def compute_risk(sentiment, depression):

    if depression == "depressed":
        return "high"

    if sentiment == "negative" and depression == "not_depressed":
        return "moderate"

    return "low"

def analyze_text(text):
    # Spell correction intentionally removed: models were trained on raw text.
    signals = detect_signals(text)
    # SENTIMENT
    inputs_sent = sent_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs_sent = sent_model(**inputs_sent)
        probs_sent = F.softmax(outputs_sent.logits, dim=1)
        sent_pred = torch.argmax(probs_sent, dim=1).item()

    sentiment = "positive" if sent_pred == 1 else "negative"
    sentiment_conf = probs_sent[0][sent_pred].item()


    # DEPRESSION
    inputs_dep = dep_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs_dep = dep_model(**inputs_dep)
        probs_dep = F.softmax(outputs_dep.logits, dim=1)
        dep_pred = torch.argmax(probs_dep, dim=1).item()

    depression = "depressed" if dep_pred == 1 else "not_depressed"
    depression_conf = probs_dep[0][dep_pred].item()


    # RISK (must be AFTER depression is defined)
    risk_level = compute_risk(sentiment, depression)


    result = {
        "text": text,
        "sentiment": {
            "label": sentiment,
            "confidence": round(sentiment_conf, 4)
        },
        "mental_health": {
            "depression_risk": depression,
            "confidence": round(depression_conf, 4)
            },
        "risk_level": risk_level,
        "signals_detected": signals
    }

    return result


if __name__ == "__main__":

    print("Mental Health NLP Pipeline Ready")

    while True:

        text = input("\nEnter text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        result = analyze_text(text)

        print("\nAnalysis Result:")
        print(result)