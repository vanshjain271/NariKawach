from config.constants import RiskLevel, RISK_THRESHOLDS

def classify_risk(score: float) -> str:
    if score >= RISK_THRESHOLDS["critical"]:
        return RiskLevel.CRITICAL.value
    elif score >= RISK_THRESHOLDS["high"]:
        return RiskLevel.HIGH.value
    elif score >= RISK_THRESHOLDS["medium"]:
        return RiskLevel.MEDIUM.value
    elif score >= RISK_THRESHOLDS["low"]:
        return RiskLevel.LOW.value
    return RiskLevel.SAFE.value
