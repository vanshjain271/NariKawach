from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW_RISK"
    MEDIUM = "MEDIUM_RISK"
    HIGH = "HIGH_RISK"
    CRITICAL = "CRITICAL_RISK"


class InterventionType(str, Enum):
    SILENT_MONITORING = "SILENT_MONITORING"
    GUARDIAN_NOTIFICATION = "GUARDIAN_NOTIFICATION"
    EMERGENCY_ALERT = "EMERGENCY_ALERT"
    POLICE_NOTIFICATION = "POLICE_NOTIFICATION"
    SAFE_NAVIGATION = "SAFE_NAVIGATION"


class AnomalyType(str, Enum):
    ROUTE_DEVIATION = "ROUTE_DEVIATION"
    SPEED_ANOMALY = "SPEED_ANOMALY"
    TIME_ANOMALY = "TIME_ANOMALY"
    STALKING_PATTERN = "STALKING_PATTERN"
    BEHAVIORAL_CHANGE = "BEHAVIORAL_CHANGE"


# Base Models
class LocationData(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    accuracy: float = Field(10.0, ge=0, description="GPS accuracy in meters")
    speed: Optional[float] = Field(None, ge=0, description="Speed in meters per second")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    bearing: Optional[float] = Field(None, ge=0, le=360, description="Bearing in degrees")
    timestamp: datetime = Field(default_factory=datetime.now, description="Location timestamp")
    battery_level: Optional[int] = Field(100, ge=0, le=100, description="Battery percentage")
    network_status: Optional[str] = Field("good", description="Network connection status")


class EnvironmentalData(BaseModel):
    lighting_score: float = Field(0.5, ge=0, le=1, description="Lighting conditions score")
    crowd_density: float = Field(0.5, ge=0, le=1, description="Crowd density estimate")
    crime_density: float = Field(0.0, ge=0, le=1, description="Crime density in area")
    weather_score: float = Field(0.0, ge=0, le=1, description="Weather risk score")
    safe_zone_distance: float = Field(10.0, ge=0, description="Distance to safe zone in km")
    police_station_distance: float = Field(5.0, ge=0, description="Distance to police station in km")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")


class UserProfile(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=120, description="User age")
    gender: Optional[str] = Field(None, description="User gender")
    emergency_contacts: List[Dict] = Field(default_factory=list, description="Emergency contacts")
    medical_conditions: Optional[List[str]] = Field(None, description="Medical conditions")
    disabilities: Optional[List[str]] = Field(None, description="Disabilities or special needs")
    preferred_language: str = Field("en", description="Preferred language")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="User risk tolerance")
    notification_preferences: Dict = Field(default_factory=dict, description="Notification preferences")


class DeviceInfo(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    distance_meters: float = Field(..., ge=0, description="Distance from user in meters")
    signal_strength: Optional[float] = Field(None, ge=0, le=1, description="Signal strength")
    last_seen: Optional[datetime] = Field(None, description="Last seen timestamp")
    device_type: Optional[str] = Field(None, description="Type of device")


# Request Models
class RiskAssessmentRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    location_data: LocationData = Field(..., description="Current location data")
    environmental_data: EnvironmentalData = Field(..., description="Environmental data")
    user_profile: UserProfile = Field(..., description="User profile information")
    nearby_devices: List[DeviceInfo] = Field(default_factory=list, description="Nearby devices")


class BatchRiskAssessment(BaseModel):
    user_id: str = Field(..., description="User identifier")
    location_data: LocationData = Field(..., description="Current location data")
    environmental_data: EnvironmentalData = Field(..., description="Environmental data")
    user_profile: UserProfile = Field(..., description="User profile information")
    nearby_devices: List[DeviceInfo] = Field(default_factory=list, description="Nearby devices")


class BatchRiskRequest(BaseModel):
    assessments: List[BatchRiskAssessment] = Field(..., description="List of assessments")


class AnomalyDetectionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    location_data: LocationData = Field(..., description="Current location data")
    historical_locations: Optional[List[LocationData]] = Field(None, description="Historical locations")
    user_patterns: Optional[Dict] = Field(None, description="User behavior patterns")


class AnomalyHistoryRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    hours: int = Field(24, ge=1, le=168, description="Hours of history to retrieve")


class StalkingDetectionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    location_data: LocationData = Field(..., description="Current location data")
    nearby_devices: List[DeviceInfo] = Field(..., description="Nearby devices")
    historical_encounters: Optional[List[Dict]] = Field(None, description="Historical device encounters")


class InterventionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    risk_assessment: Dict = Field(..., description="Risk assessment results")
    anomalies: Dict = Field(..., description="Anomaly detection results")
    context: Dict = Field(..., description="Additional context information")
    user_preferences: Optional[Dict] = Field(None, description="User intervention preferences")


class EmergencyActivationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    emergency_data: Dict = Field(..., description="Emergency data and context")
    verification_method: Optional[str] = Field(None, description="Emergency verification method")
    manual_trigger: bool = Field(False, description="Whether manually triggered")


class EmergencyResolution(BaseModel):
    resolution_type: str = Field(..., description="Type of resolution")
    details: Optional[str] = Field(None, description="Resolution details")
    outcome: str = Field("success", description="Resolution outcome")


class RuleRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    context: Dict = Field(..., description="Context for rule recommendations")
    pattern_data: Optional[Dict] = Field(None, description="Pattern data for analysis")


# Response Models
class HealthResponse(BaseModel):
    status: str = Field(..., description="System health status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")


class StatusResponse(BaseModel):
    timestamp: datetime = Field(..., description="Response timestamp")
    system_status: str = Field(..., description="Overall system status")
    queue_status: Dict = Field(..., description="Queue status information")
    intervention_stats: Dict = Field(..., description="Intervention statistics")
    active_emergencies: int = Field(..., description="Number of active emergencies")
    resource_utilization: Dict = Field(..., description="Resource utilization percentages")


class RiskAssessmentResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    risk_assessment: Dict = Field(..., description="Risk assessment results")
    risk_prediction: Dict = Field(..., description="Risk prediction results")
    anomaly_detection: Dict = Field(..., description="Anomaly detection results")
    stalking_analysis: Dict = Field(..., description="Stalking analysis results")
    pattern_analysis: Dict = Field(..., description="Pattern analysis results")
    triggered_rules: List[Dict] = Field(..., description="Triggered rules")
    engineered_features: Dict = Field(..., description="Engineered features")


class BatchRiskResponse(BaseModel):
    batch_id: str = Field(..., description="Batch identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    total_assessments: int = Field(..., description="Total assessments")
    successful_assessments: int = Field(..., description="Successful assessments")
    failed_assessments: int = Field(..., description="Failed assessments")
    average_processing_time: float = Field(..., description="Average processing time")
    results: List[Dict] = Field(..., description="Individual results")


class AnomalyDetectionResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    route_anomalies: Dict = Field(..., description="Route anomaly results")
    pattern_analysis: Dict = Field(..., description="Pattern analysis results")
    anomaly_confidence: float = Field(..., description="Anomaly detection confidence")
    requires_attention: bool = Field(..., description="Whether attention is required")


class AnomalyHistoryResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    hours: int = Field(..., description="Hours of history retrieved")
    total_anomalies: int = Field(..., description="Total anomalies found")
    high_risk_anomalies: int = Field(..., description="High risk anomalies")
    history: List[Dict] = Field(..., description="Anomaly history")


class StalkingDetectionResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    stalking_analysis: Dict = Field(..., description="Stalking analysis results")
    immediate_action_required: bool = Field(..., description="Whether immediate action is required")


class StalkingPatternResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    analysis_period_days: int = Field(..., description="Analysis period in days")
    pattern_data: Dict = Field(..., description="Pattern data")
    trend_analysis: Dict = Field(..., description="Trend analysis results")
    recommendations: List[str] = Field(..., description="Safety recommendations")


class InterventionResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    intervention_plan: Dict = Field(..., description="Intervention plan")
    queue_position: str = Field(..., description="Queue position")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time in seconds")
    emergency_contact_notified: bool = Field(..., description="Whether emergency contacts notified")


class InterventionStatusResponse(BaseModel):
    intervention_id: str = Field(..., description="Intervention identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    status: Dict = Field(..., description="Status information")
    detailed_info: Optional[Dict] = Field(None, description="Detailed intervention information")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    queue_info: Dict = Field(..., description="Queue information")


class CancelInterventionResponse(BaseModel):
    intervention_id: str = Field(..., description="Intervention identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    cancelled: bool = Field(..., description="Whether cancelled successfully")
    reason: Optional[str] = Field(None, description="Cancellation reason")
    cancellation_time: datetime = Field(..., description="Cancellation timestamp")


class EmergencyActivationResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    emergency_activated: bool = Field(..., description="Whether emergency activated")
    emergency_id: str = Field(..., description="Emergency identifier")
    protocol_actions: List[str] = Field(..., description="Protocol actions initiated")
    services_notified: List[str] = Field(..., description="Services notified")
    emergency_level: str = Field(..., description="Emergency level")
    instructions: str = Field(..., description="Emergency instructions")


class EmergencyResolutionResponse(BaseModel):
    emergency_id: str = Field(..., description="Emergency identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    resolved: bool = Field(..., description="Whether emergency resolved")
    resolution_type: str = Field(..., description="Resolution type")
    resolution_details: Optional[str] = Field(None, description="Resolution details")
    resolution_time: datetime = Field(..., description="Resolution timestamp")


class ActiveEmergenciesResponse(BaseModel):
    timestamp: datetime = Field(..., description="Response timestamp")
    total_active: int = Field(..., description="Total active emergencies")
    emergencies: List[Dict] = Field(..., description="Active emergencies")


class RulesResponse(BaseModel):
    timestamp: datetime = Field(..., description="Response timestamp")
    total_rules: int = Field(..., description="Total number of rules")
    active_rules: int = Field(..., description="Number of active rules")
    triggered_rules_today: int = Field(..., description="Rules triggered today")
    rules: List[Dict] = Field(..., description="List of rules")


class RuleRecommendationResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    recommended_rules: List[Dict] = Field(..., description="Recommended rules")
    justification: str = Field(..., description="Justification for recommendations")
    priority_order: List[str] = Field(..., description="Priority order of recommendations")


class AnalyticsResponse(BaseModel):
    timestamp: datetime = Field(..., description="Response timestamp")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    total_users: int = Field(..., description="Total users in period")
    total_assessments: int = Field(..., description="Total risk assessments")
    total_interventions: int = Field(..., description="Total interventions")
    total_emergencies: int = Field(..., description="Total emergencies")
    risk_distribution: Dict = Field(..., description="Risk level distribution")
    anomaly_breakdown: Dict = Field(..., description="Anomaly type breakdown")
    intervention_types: Dict = Field(..., description="Intervention type distribution")
    response_times: Dict = Field(..., description="Average response times")
    user_feedback: Dict = Field(..., description="User feedback statistics")


class UserFeedbackRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    feedback_type: str = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comments: Optional[str] = Field(None, description="User comments")
    context: Optional[Dict] = Field(None, description="Feedback context")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")


class UserFeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description="Feedback identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    received: bool = Field(..., description="Whether feedback received")
    message: str = Field(..., description="Response message")
    follow_up_required: bool = Field(False, description="Whether follow-up required")


class SystemConfiguration(BaseModel):
    config_id: str = Field(..., description="Configuration identifier")
    timestamp: datetime = Field(..., description="Configuration timestamp")
    risk_thresholds: Dict = Field(..., description="Risk level thresholds")
    intervention_settings: Dict = Field(..., description="Intervention settings")
    anomaly_settings: Dict = Field(..., description="Anomaly detection settings")
    emergency_protocols: Dict = Field(..., description="Emergency protocols")
    notification_settings: Dict = Field(..., description="Notification settings")
    data_retention: Dict = Field(..., description="Data retention policies")


class ConfigurationUpdateRequest(BaseModel):
    config_id: str = Field(..., description="Configuration identifier")
    updates: Dict = Field(..., description="Configuration updates")
    apply_immediately: bool = Field(False, description="Whether to apply immediately")
    notes: Optional[str] = Field(None, description="Update notes")


class ConfigurationUpdateResponse(BaseModel):
    config_id: str = Field(..., description="Configuration identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    updated: bool = Field(..., description="Whether update successful")
    applied: bool = Field(..., description="Whether applied successfully")
    changes: Dict = Field(..., description="Changes made")
    warnings: Optional[List[str]] = Field(None, description="Any warnings")


class AuditLogRequest(BaseModel):
    start_date: datetime = Field(..., description="Start date for audit logs")
    end_date: datetime = Field(..., description="End date for audit logs")
    event_types: Optional[List[str]] = Field(None, description="Event types to filter")
    user_id: Optional[str] = Field(None, description="User ID to filter")
    limit: int = Field(100, ge=1, le=1000, description="Maximum logs to return")


class AuditLogEntry(BaseModel):
    log_id: str = Field(..., description="Log entry identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Event type")
    user_id: Optional[str] = Field(None, description="User identifier")
    action: str = Field(..., description="Action performed")
    details: Dict = Field(..., description="Event details")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")


class AuditLogResponse(BaseModel):
    timestamp: datetime = Field(..., description="Response timestamp")
    total_logs: int = Field(..., description="Total logs found")
    returned_logs: int = Field(..., description="Logs returned")
    logs: List[AuditLogEntry] = Field(..., description="Audit log entries")


class ErrorResponse(BaseModel):
    timestamp: datetime = Field(..., description="Error timestamp")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    suggestion: Optional[str] = Field(None, description="Suggested action")


class ValidationError(BaseModel):
    loc: List[Union[str, int]] = Field(..., description="Location of error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    timestamp: datetime = Field(..., description="Error timestamp")
    detail: List[ValidationError] = Field(..., description="Validation errors")


# Additional models for specific functionalities
class RouteAnalysis(BaseModel):
    route_id: str = Field(..., description="Route identifier")
    user_id: str = Field(..., description="User identifier")
    start_location: LocationData = Field(..., description="Start location")
    end_location: LocationData = Field(..., description="End location")
    waypoints: Optional[List[LocationData]] = Field(None, description="Route waypoints")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    risk_score: float = Field(..., ge=0, le=1, description="Route risk score")
    alternative_routes: Optional[List[Dict]] = Field(None, description="Alternative routes")
    recommendations: Optional[List[str]] = Field(None, description="Safety recommendations")


class SafetyCheckResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    check_passed: bool = Field(..., description="Whether safety check passed")
    issues_found: List[str] = Field(default_factory=list, description="Issues found")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    next_check_suggested: Optional[datetime] = Field(None, description="Next check time")


class LocationHistoryRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    max_points: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum points to return")


class LocationHistoryResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    total_points: int = Field(..., description="Total location points")
    points_returned: int = Field(..., description="Points returned")
    locations: List[LocationData] = Field(..., description="Location history")
    statistics: Optional[Dict] = Field(None, description="Location statistics")


class PatternAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    analysis_type: str = Field(..., description="Type of analysis")
    date_range: Dict = Field(..., description="Date range for analysis")
    parameters: Optional[Dict] = Field(None, description="Analysis parameters")


class PatternAnalysisResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    analysis_type: str = Field(..., description="Type of analysis")
    patterns_found: List[Dict] = Field(..., description="Patterns found")
    confidence_scores: Dict = Field(..., description="Confidence scores")
    insights: List[str] = Field(..., description="Insights from analysis")
    recommendations: List[str] = Field(..., description="Recommendations")


class NotificationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    notification_type: str = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: str = Field("medium", description="Notification priority")
    data: Optional[Dict] = Field(None, description="Additional data")
    silent: bool = Field(False, description="Whether to send silently")


class NotificationResponse(BaseModel):
    notification_id: str = Field(..., description="Notification identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    sent: bool = Field(..., description="Whether notification sent")
    channels: List[str] = Field(..., description="Channels used")
    status: str = Field(..., description="Delivery status")


# Composite response model for API endpoints that combine multiple functionalities
class ComprehensiveSafetyReport(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Report timestamp")
    risk_assessment: RiskAssessmentResponse = Field(..., description="Risk assessment")
    anomaly_detection: AnomalyDetectionResponse = Field(..., description="Anomaly detection")
    stalking_analysis: Optional[StalkingDetectionResponse] = Field(None, description="Stalking analysis")
    pattern_analysis: Optional[PatternAnalysisResponse] = Field(None, description="Pattern analysis")
    intervention_plan: Optional[InterventionResponse] = Field(None, description="Intervention plan")
    recommendations: List[str] = Field(..., description="Overall recommendations")
    safety_score: float = Field(..., ge=0, le=100, description="Overall safety score")
    next_steps: List[str] = Field(..., description="Next steps")


# Model for storing and retrieving user preferences
class UserPreferences(BaseModel):
    user_id: str = Field(..., description="User identifier")
    notification_settings: Dict = Field(default_factory=dict, description="Notification settings")
    privacy_settings: Dict = Field(default_factory=dict, description="Privacy settings")
    emergency_contacts: List[Dict] = Field(default_factory=list, description="Emergency contacts")
    safety_rules: List[Dict] = Field(default_factory=list, description="Safety rules")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Risk tolerance")
    preferred_interventions: List[str] = Field(default_factory=list, description="Preferred interventions")
    data_sharing_preferences: Dict = Field(default_factory=dict, description="Data sharing preferences")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class UpdateUserPreferencesRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    updates: Dict = Field(..., description="Preferences updates")
    partial: bool = Field(True, description="Whether update is partial")


class UpdateUserPreferencesResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    updated: bool = Field(..., description="Whether update successful")
    previous_values: Optional[Dict] = Field(None, description="Previous values")
    new_values: Dict = Field(..., description="New values")


# Model for system metrics and monitoring
class SystemMetrics(BaseModel):
    timestamp: datetime = Field(..., description="Metrics timestamp")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_throughput: Dict = Field(..., description="Network throughput")
    active_connections: int = Field(..., ge=0, description="Active connections")
    request_rate: float = Field(..., ge=0, description="Requests per second")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate")
    queue_lengths: Dict = Field(..., description="Queue lengths")
    response_times: Dict = Field(..., description="Response times in milliseconds")
    database_metrics: Dict = Field(..., description="Database metrics")