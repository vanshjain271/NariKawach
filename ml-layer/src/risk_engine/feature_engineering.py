def build_features(payload):
    features = {
        "crime_density": payload.crime_density,
        "crowd_density": payload.crowd_density,
        "lighting_score": payload.lighting_score,
        "speed": payload.speed,
        "route_deviation": payload.route_deviation_score,
        "hour": payload.time.hour
    }
    return features
