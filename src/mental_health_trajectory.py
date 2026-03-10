from src.mental_health_pipeline import analyze_text

def risk_to_score(risk):

    if risk == "low":
        return 1
    if risk == "moderate":
        return 2
    if risk == "high":
        return 3

    return 0


def detect_trend(scores):

    increasing = 0
    decreasing = 0

    for i in range(1, len(scores)):
        if scores[i] > scores[i-1]:
            increasing += 1
        elif scores[i] < scores[i-1]:
            decreasing += 1

    if increasing > decreasing:
        return "increasing_risk"

    if decreasing > increasing:
        return "decreasing_risk"

    return "stable"


def analyze_trajectory(posts):

    results = []
    scores = []

    for post in posts:

        result = analyze_text(post)

        risk = result["risk_level"]
        score = risk_to_score(risk)

        scores.append(score)
        results.append(result)

    trend = detect_trend(scores)

    output = {
        "posts_analyzed": len(posts),
        "risk_scores": scores,
        "trend": trend,
        "analysis": results
    }

    return output


if __name__ == "__main__":

    print("Mental Health Trajectory Analyzer Ready")

    posts = []

    while True:

        text = input("\nEnter post (or type 'done'): ")

        if text.lower() == "done":
            break

        posts.append(text)

    if len(posts) == 0:
        print("No posts provided.")
    else:

        result = analyze_trajectory(posts)

        print("\nTrajectory Analysis:")
        print(result)