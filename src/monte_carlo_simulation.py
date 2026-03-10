import random
import numpy as np
from src.mental_health_pipeline import analyze_text

INTENSIFIERS = [
    "very",
    "really",
    "so",
    "extremely",
    "quite",
    "somewhat"
]

def generate_variation(sentence):

    words = sentence.split()

    if len(words) < 3:
        return sentence

    idx = random.randint(0, len(words)-1)
    words.insert(idx, random.choice(INTENSIFIERS))

    return " ".join(words)


def risk_to_score(risk):

    if risk == "low":
        return 1
    if risk == "moderate":
        return 2
    if risk == "high":
        return 3


def run_simulation(text, n=100):

    scores = []

    for _ in range(n):

        variation = generate_variation(text)

        result = analyze_text(variation)

        score = risk_to_score(result["risk_level"])

        scores.append(score)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    return {
        "original_text": text,
        "runs": n,
        "mean_risk_score": mean_score,
        "risk_variance": std_score,
        "all_scores": scores
    }


if __name__ == "__main__":

    text = input("Enter text: ")

    result = run_simulation(text, 50)

    print("\nMonte Carlo Result:")
    print(result)