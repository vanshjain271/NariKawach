from src.risk_engine.feature_engineering import build_features
from src.risk_engine.ensemble_predictor import ensemble_predict
from src.risk_engine.risk_explainer import explain
from src.decision_agent.rule_engine import classify_risk

def calculate_risk(payload):
    features = build_features(payload)
    score, confidence = ensemble_predict(features)
    risk_level = classify_risk(score)
    reason = explain(features, score)

    return {
        "risk_score": round(score, 2),
        "risk_level": risk_level,
        "confidence": round(confidence, 2),
        "reason": reason
    }
