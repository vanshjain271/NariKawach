"""
SHIELD AI - Constants and Enums
Global constants used across the application
"""

from enum import Enum
from typing import Dict


class RiskLevel(str, Enum):
    """Risk level categories"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Types of detected anomalies"""
    ROUTE_DEVIATION = "route_deviation"
    SPEED_ANOMALY = "speed_anomaly"
    STOP_ANOMALY = "stop_anomaly"
    TIME_ANOMALY = "time_anomaly"
    LOCATION_ANOMALY = "location_anomaly"
    STALKING_PATTERN = "stalking_pattern"
    DEVICE_FOLLOWING = "device_following"
    BEHAVIORAL_CHANGE = "behavioral_change"


class AlertPriority(str, Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class InterventionType(str, Enum):
    """Types of safety interventions"""
    NOTIFICATION = "notification"
    ALERT_GUARDIANS = "alert_guardians"
    SHARE_LOCATION = "share_location"
    SUGGEST_SAFE_ROUTE = "suggest_safe_route"
    EMERGENCY_CALL = "emergency_call"
    SILENT_ALERT = "silent_alert"
    FULL_EMERGENCY = "full_emergency"


class EmergencyProtocol(str, Enum):
    """Emergency protocol types"""
    STALKING_EMERGENCY = "stalking_emergency"
    ROUTE_EMERGENCY = "route_emergency"
    SOS_ACTIVATED = "sos_activated"
    GUARDIAN_ALERT = "guardian_alert"
    POLICE_DISPATCH = "police_dispatch"


# Risk score thresholds
RISK_THRESHOLDS: Dict[str, float] = {
    'safe': 0.2,
    'low': 0.4,
    'medium': 0.6,
    'high': 0.8,
    'critical': 1.0
}

# Feature weights for risk calculation
FEATURE_WEIGHTS: Dict[str, float] = {
    'temporal': 0.20,
    'spatial': 0.25,
    'environmental': 0.20,
    'behavioral': 0.20,
    'social': 0.15
}

# Stalking detection thresholds
STALKING_THRESHOLDS: Dict[str, float] = {
    'coincidence_score': 0.6,
    'route_following_score': 0.7,
    'proximity_score': 0.5,
    'temporal_correlation': 0.6,
    'overall_risk': 0.7
}

# Time risk multipliers (hour: multiplier)
TIME_RISK_MULTIPLIERS: Dict[int, float] = {
    0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.4, 5: 1.2,
    6: 0.9, 7: 0.8, 8: 0.7, 9: 0.7, 10: 0.7, 11: 0.7,
    12: 0.7, 13: 0.7, 14: 0.7, 15: 0.8, 16: 0.8, 17: 0.9,
    18: 1.0, 19: 1.1, 20: 1.2, 21: 1.3, 22: 1.4, 23: 1.5
}

# Location type risk scores
LOCATION_RISK_SCORES: Dict[str, float] = {
    'residential': 0.3,
    'commercial': 0.4,
    'industrial': 0.6,
    'park': 0.5,
    'alley': 0.8,
    'highway': 0.5,
    'transit': 0.4,
    'unknown': 0.5
}

# Crowd density interpretations
CROWD_DENSITY_RISK: Dict[str, float] = {
    'empty': 0.8,
    'sparse': 0.6,
    'moderate': 0.3,
    'crowded': 0.2,
    'packed': 0.4  # Too crowded can also be risky
}

# API response codes
API_CODES: Dict[str, int] = {
    'success': 200,
    'created': 201,
    'bad_request': 400,
    'unauthorized': 401,
    'forbidden': 403,
    'not_found': 404,
    'rate_limited': 429,
    'internal_error': 500
}

# Maximum values for validation
MAX_VALUES: Dict[str, float] = {
    'latitude': 90.0,
    'longitude': 180.0,
    'speed_kmh': 200.0,
    'distance_km': 1000.0,
    'risk_score': 1.0,
    'confidence': 1.0
}

# Minimum values for validation
MIN_VALUES: Dict[str, float] = {
    'latitude': -90.0,
    'longitude': -180.0,
    'speed_kmh': 0.0,
    'distance_km': 0.0,
    'risk_score': 0.0,
    'confidence': 0.0
}

# Cache TTL values (in seconds)
CACHE_TTL: Dict[str, int] = {
    'risk_calculation': 60,
    'crime_data': 3600,
    'crowd_density': 900,
    'lighting_data': 86400,
    'user_profile': 300
}

# Rate limiting defaults
RATE_LIMITS: Dict[str, int] = {
    'default_rpm': 60,
    'risk_assessment_rpm': 30,
    'emergency_rpm': 100,
    'admin_rpm': 120
}
