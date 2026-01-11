import random

def ensemble_predict(features):
    crime = features["crime_density"] / 10        # normalize
    lighting = 1 - (features["lighting_score"] / 10)
    crowd = 1 - (features["crowd_density"] / 10)
    route_dev = features["route_deviation"]
    hour = features["hour"]

    # Night risk multiplier
    night_multiplier = 1.5 if hour >= 22 or hour <= 5 else 1.0

    # Core risk score
    base_score = (
        crime * 0.35 +
        lighting * 0.25 +
        crowd * 0.20 +
        route_dev * 0.20
    )

    # Apply night amplification
    score = base_score * night_multiplier

    # Hard danger rule (dark + empty + night)
    if lighting > 0.7 and crowd > 0.7 and night_multiplier > 1:
        score = max(score, 0.85)

    # Clamp
    score = min(1.0, max(0.0, score))

    confidence = random.uniform(0.88, 0.96)

    return score, confidence
