def explain(features, score):
    if features["lighting_score"] < 3 and features["crowd_density"] < 2 and features["hour"] >= 22:
        return "Dark and isolated area at night"
    if features["crime_density"] > 7:
        return "High crime density area"
    if features["route_deviation"] > 0.6:
        return "Suspicious route deviation detected"
    return "Normal travel pattern"
