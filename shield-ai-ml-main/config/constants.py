from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    SAFE = "SAFE"
    LOW = "LOW_RISK"
    MEDIUM = "MEDIUM_RISK"
    HIGH = "HIGH_RISK"
    CRITICAL = "CRITICAL_RISK"


class InterventionType(str, Enum):
    """Type of interventions"""
    SILENT_MONITORING = "SILENT_MONITORING"
    GUARDIAN_NOTIFICATION = "GUARDIAN_NOTIFICATION"
    EMERGENCY_ALERT = "EMERGENCY_ALERT"
    POLICE_NOTIFICATION = "POLICE_NOTIFICATION"
    SAFE_NAVIGATION = "SAFE_NAVIGATION"


class AnomalyType(str, Enum):
    """Type of anomalies detected"""
    ROUTE_DEVIATION = "ROUTE_DEVIATION"
    SPEED_ANOMALY = "SPEED_ANOMALY"
    TIME_ANOMALY = "TIME_ANOMALY"
    STALKING_PATTERN = "STALKING_PATTERN"
    BEHAVIORAL_CHANGE = "BEHAVIORAL_CHANGE"


class DeviceStatus(str, Enum):
    """Device status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    LOW_BATTERY = "LOW_BATTERY"
    OFFLINE = "OFFLINE"


# Feature engineering constants
FEATURE_WEIGHTS = {
    "temporal": 0.25,
    "spatial": 0.30,
    "environmental": 0.20,
    "behavioral": 0.15,
    "social": 0.10
}

# Risk score thresholds
RISK_THRESHOLDS = {
    "safe": 0.3,
    "low": 0.5,
    "medium": 0.7,
    "high": 0.85,
    "critical": 0.95
}

# Emergency response priorities
EMERGENCY_PRIORITIES = {
    "high": ["police", "guardians", "safe_zones"],
    "medium": ["guardians", "safe_zones"],
    "low": ["guardians"]
}

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 8,
        "learning_rate": 0.1
    },
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": 0.1
    }
}