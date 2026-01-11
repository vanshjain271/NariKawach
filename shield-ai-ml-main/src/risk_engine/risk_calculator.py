import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from collections import defaultdict
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import RiskLevel, FEATURE_WEIGHTS, RISK_THRESHOLDS


class RiskCalculator:
    """
    Comprehensive risk calculation engine
    Combines multiple risk factors with advanced weighting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'temporal_weight': 0.25,
            'spatial_weight': 0.30,
            'environmental_weight': 0.20,
            'behavioral_weight': 0.15,
            'social_weight': 0.10,
            'decay_factor': 0.95,
            'history_window_hours': 168  # 7 days
        }
        
        self.logger = setup_logger(__name__)
        
        # Risk history for trend analysis
        self.risk_history = defaultdict(list)
        
        # Risk factor dictionaries
        self.risk_factors = {
            'temporal': self._calculate_temporal_risk,
            'spatial': self._calculate_spatial_risk,
            'environmental': self._calculate_environmental_risk,
            'behavioral': self._calculate_behavioral_risk,
            'social': self._calculate_social_risk
        }
    
    def calculate_risk(self, user_id: str, context: Dict) -> Dict:
        """
        Calculate comprehensive risk score
        """
        try:
            self.logger.debug(f"Calculating risk for user {user_id}")
            
            # Calculate individual risk components
            risk_components = {}
            
            for factor_name, factor_func in self.risk_factors.items():
                risk_components[factor_name] = factor_func(context)
            
            # Calculate weighted risk score
            weighted_risk = self._calculate_weighted_risk(risk_components)
            
            # Apply trend analysis
            trend_adjustment = self._analyze_risk_trend(user_id, weighted_risk)
            adjusted_risk = weighted_risk * trend_adjustment
            
            # Categorize risk level
            risk_category = self._categorize_risk(adjusted_risk)
            
            # Calculate confidence
            confidence = self._calculate_risk_confidence(risk_components, context)
            
            # Generate risk breakdown
            risk_breakdown = self._generate_risk_breakdown(risk_components, adjusted_risk)
            
            # Update risk history
            self._update_risk_history(user_id, adjusted_risk, risk_components)
            
            result = {
                'user_id': user_id,
                'risk_score': float(adjusted_risk),
                'risk_level': risk_category,
                'confidence': float(confidence),
                'risk_components': risk_components,
                'risk_breakdown': risk_breakdown,
                'weighted_risk': float(weighted_risk),
                'trend_adjustment': float(trend_adjustment),
                'high_risk_factors': self._identify_high_risk_factors(risk_components),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Risk calculation complete for user {user_id}. Score: {adjusted_risk}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating risk for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'risk_score': 0.0,
                'risk_level': RiskLevel.SAFE,
                'confidence': 0.0,
                'risk_components': {},
                'risk_breakdown': ['Error in risk calculation'],
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_temporal_risk(self, context: Dict) -> float:
        """Calculate risk based on time factors"""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            weekday = current_time.weekday()
            
            temporal_risk = 0.0
            
            # Time of day risk
            if 22 <= hour <= 6:  # Late night to early morning
                temporal_risk += 0.4
            elif 20 <= hour < 22:  # Evening
                temporal_risk += 0.2
            elif 6 <= hour < 8:  # Early morning
                temporal_risk += 0.1
            
            # Day of week risk
            if weekday >= 5:  # Weekend
                temporal_risk += 0.1
            
            # Holiday/special day risk
            if context.get('is_holiday', False):
                temporal_risk += 0.2
            
            # Seasonality risk (higher in winter months)
            month = current_time.month
            if month in [11, 12, 1, 2]:  # Winter months
                temporal_risk += 0.1
            
            # User's usual time pattern deviation
            if context.get('time_deviation_score', 0) > 0.5:
                temporal_risk += context['time_deviation_score'] * 0.3
            
            return min(1.0, temporal_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal risk: {e}")
            return 0.0
    
    def _calculate_spatial_risk(self, context: Dict) -> float:
        """Calculate risk based on location factors"""
        try:
            spatial_risk = 0.0
            
            # Crime density
            crime_density = context.get('crime_density', 0.0)
            spatial_risk += crime_density * 0.4
            
            # Distance from safe zones
            safe_zone_distance = context.get('safe_zone_distance', 10.0)
            if safe_zone_distance > 5.0:  # More than 5km from safe zone
                distance_risk = min(1.0, (safe_zone_distance - 5.0) / 10.0)
                spatial_risk += distance_risk * 0.3
            
            # Police proximity
            police_distance = context.get('police_station_distance', 5.0)
            if police_distance > 2.0:  # More than 2km from police
                police_risk = min(1.0, (police_distance - 2.0) / 5.0)
                spatial_risk += police_risk * 0.2
            
            # Hospital proximity
            hospital_distance = context.get('hospital_distance', 10.0)
            if hospital_distance > 5.0:  # More than 5km from hospital
                hospital_risk = min(1.0, (hospital_distance - 5.0) / 10.0)
                spatial_risk += hospital_risk * 0.1
            
            # Location type risk
            location_type = context.get('location_type', 'unknown')
            if location_type in ['alley', 'park', 'industrial_area']:
                spatial_risk += 0.2
            elif location_type in ['residential', 'commercial']:
                spatial_risk += 0.1
            
            return min(1.0, spatial_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial risk: {e}")
            return 0.0
    
    def _calculate_environmental_risk(self, context: Dict) -> float:
        """Calculate risk based on environmental factors"""
        try:
            environmental_risk = 0.0
            
            # Lighting conditions
            lighting_score = context.get('lighting_score', 0.5)
            environmental_risk += (1.0 - lighting_score) * 0.4
            
            # Crowd density
            crowd_density = context.get('crowd_density', 0.5)
            if crowd_density < 0.2:  # Very low crowd
                environmental_risk += 0.3
            elif crowd_density < 0.5:  # Low crowd
                environmental_risk += 0.15
            
            # Weather conditions
            weather_score = context.get('weather_score', 0.0)
            environmental_risk += weather_score * 0.2
            
            # Visibility
            visibility_score = context.get('visibility_score', 1.0)
            environmental_risk += (1.0 - visibility_score) * 0.1
            
            # Noise level (low noise might indicate isolation)
            noise_level = context.get('noise_level', 0.5)
            if noise_level < 0.3:
                environmental_risk += 0.1
            
            return min(1.0, environmental_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating environmental risk: {e}")
            return 0.0
    
    def _calculate_behavioral_risk(self, context: Dict) -> float:
        """Calculate risk based on behavioral factors"""
        try:
            behavioral_risk = 0.0
            
            # Route deviation
            route_deviation = context.get('route_deviation_score', 0.0)
            behavioral_risk += route_deviation * 0.3
            
            # Speed anomaly
            speed_anomaly = context.get('speed_anomaly_score', 0.0)
            behavioral_risk += speed_anomaly * 0.2
            
            # Stop anomaly
            stop_anomaly = context.get('stop_anomaly_score', 0.0)
            behavioral_risk += stop_anomaly * 0.15
            
            # Direction anomaly
            direction_anomaly = context.get('direction_anomaly_score', 0.0)
            behavioral_risk += direction_anomaly * 0.1
            
            # User confidence (from app usage)
            user_confidence = context.get('user_confidence_score', 0.5)
            behavioral_risk += (1.0 - user_confidence) * 0.15
            
            # Previous alerts
            previous_alerts = context.get('previous_alerts_count', 0)
            if previous_alerts > 0:
                alert_risk = min(1.0, previous_alerts / 10.0)
                behavioral_risk += alert_risk * 0.1
            
            return min(1.0, behavioral_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating behavioral risk: {e}")
            return 0.0
    
    def _calculate_social_risk(self, context: Dict) -> float:
        """Calculate risk based on social factors"""
        try:
            social_risk = 0.0
            
            # Guardian availability
            guardian_online = context.get('guardian_online_count', 0)
            if guardian_online == 0:
                social_risk += 0.3
            elif guardian_online == 1:
                social_risk += 0.15
            
            # Social check-ins
            social_checkins = context.get('social_checkin_density', 0.0)
            if social_checkins < 0.2:  # Low social activity in area
                social_risk += 0.2
            
            # Emergency response time
            response_time = context.get('emergency_response_time', 10.0)
            if response_time > 15.0:  # Slow response time
                response_risk = min(1.0, (response_time - 15.0) / 30.0)
                social_risk += response_risk * 0.2
            
            # Community safety score
            community_score = context.get('community_safety_score', 0.5)
            social_risk += (1.0 - community_score) * 0.2
            
            # Local crime reporting
            crime_reporting = context.get('crime_reporting_rate', 0.5)
            social_risk += (1.0 - crime_reporting) * 0.1
            
            return min(1.0, social_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating social risk: {e}")
            return 0.0
    
    def _calculate_weighted_risk(self, risk_components: Dict) -> float:
        """Calculate weighted risk score from components"""
        weighted_score = 0.0
        total_weight = 0.0
        
        weights = self.config
        
        for component_name, component_score in risk_components.items():
            weight_key = f"{component_name}_weight"
            if weight_key in weights:
                weight = weights[weight_key]
                weighted_score += component_score * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        
        return min(1.0, weighted_score)
    
    def _analyze_risk_trend(self, user_id: str, current_risk: float) -> float:
        """Analyze risk trend and apply adjustment"""
        user_history = self.risk_history[user_id]
        
        if len(user_history) < 3:
            return 1.0  # No adjustment
        
        # Get recent risks (last 6 hours)
        recent_time = datetime.now() - timedelta(hours=6)
        recent_risks = [
            risk for risk in user_history
            if risk['timestamp'] > recent_time
        ]
        
        if len(recent_risks) < 2:
            return 1.0
        
        # Calculate trend
        recent_scores = [r['risk_score'] for r in recent_risks]
        avg_recent_risk = np.mean(recent_scores)
        
        # If current risk is significantly higher than recent average
        if current_risk > avg_recent_risk * 1.5:
            trend_factor = min(1.5, current_risk / (avg_recent_risk + 0.001))
            return trend_factor
        else:
            return 1.0
    
    def _categorize_risk(self, risk_score: float) -> RiskLevel:
        """Categorize risk score into level"""
        if risk_score < RISK_THRESHOLDS['safe']:
            return RiskLevel.SAFE
        elif risk_score < RISK_THRESHOLDS['low']:
            return RiskLevel.LOW
        elif risk_score < RISK_THRESHOLDS['medium']:
            return RiskLevel.MEDIUM
        elif risk_score < RISK_THRESHOLDS['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _calculate_risk_confidence(self, risk_components: Dict, 
                                  context: Dict) -> float:
        """Calculate confidence in risk assessment"""
        confidence_factors = []
        
        # Data completeness
        required_data = ['crime_density', 'lighting_score', 'crowd_density', 
                        'route_deviation_score', 'guardian_online_count']
        available_data = sum(1 for field in required_data if field in context)
        data_completeness = available_data / len(required_data)
        confidence_factors.append(data_completeness * 0.3)
        
        # Data recency
        data_age = context.get('data_age_minutes', 60)
        if data_age < 5:
            recency_score = 1.0
        elif data_age < 30:
            recency_score = 0.8
        elif data_age < 60:
            recency_score = 0.6
        else:
            recency_score = 0.3
        confidence_factors.append(recency_score * 0.2)
        
        # Component consistency
        component_scores = list(risk_components.values())
        if component_scores:
            consistency = 1.0 - (np.std(component_scores) / 0.5)  # Normalize
            consistency = max(0.0, min(1.0, consistency))
            confidence_factors.append(consistency * 0.3)
        
        # Historical consistency
        historical_count = len(self.risk_history.get('user_id', []))
        if historical_count > 50:
            historical_score = 0.9
        elif historical_count > 20:
            historical_score = 0.7
        elif historical_count > 5:
            historical_score = 0.5
        else:
            historical_score = 0.3
        confidence_factors.append(historical_score * 0.2)
        
        return min(1.0, sum(confidence_factors))
    
    def _generate_risk_breakdown(self, risk_components: Dict, 
                                overall_risk: float) -> List[str]:
        """Generate human-readable risk breakdown"""
        breakdown = []
        
        # Add overall risk description
        if overall_risk < 0.3:
            breakdown.append("Low overall risk - normal conditions")
        elif overall_risk < 0.5:
            breakdown.append("Moderate risk - stay alert")
        elif overall_risk < 0.7:
            breakdown.append("Elevated risk - take precautions")
        elif overall_risk < 0.9:
            breakdown.append("High risk - immediate action recommended")
        else:
            breakdown.append("CRITICAL RISK - emergency intervention needed")
        
        # Add high risk factors
        high_risk_factors = self._identify_high_risk_factors(risk_components)
        for factor in high_risk_factors:
            breakdown.append(f"High {factor.replace('_', ' ')} risk detected")
        
        return breakdown
    
    def _identify_high_risk_factors(self, risk_components: Dict) -> List[str]:
        """Identify factors contributing most to risk"""
        high_risk_threshold = 0.6
        
        high_risk_factors = []
        for factor_name, factor_score in risk_components.items():
            if factor_score > high_risk_threshold:
                high_risk_factors.append(factor_name)
        
        return high_risk_factors
    
    def _update_risk_history(self, user_id: str, risk_score: float,
                            risk_components: Dict):
        """Update risk history for trend analysis"""
        history_entry = {
            'timestamp': datetime.now(),
            'risk_score': risk_score,
            'risk_components': risk_components
        }
        
        self.risk_history[user_id].append(history_entry)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=self.config['history_window_hours'])
        self.risk_history[user_id] = [
            entry for entry in self.risk_history[user_id]
            if entry['timestamp'] > cutoff_time
        ]
    
    def get_risk_history(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Get risk history for a user"""
        if user_id not in self.risk_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'risk_score': entry['risk_score'],
                'risk_level': self._categorize_risk(entry['risk_score']).value
            }
            for entry in self.risk_history[user_id]
            if entry['timestamp'] > cutoff_time
        ]
    
    def calculate_risk_trend(self, user_id: str, window_hours: int = 6) -> Dict:
        """Calculate risk trend over time"""
        history = self.get_risk_history(user_id, window_hours * 2)
        
        if len(history) < 3:
            return {'trend': 'insufficient_data', 'slope': 0.0}
        
        # Extract timestamps and scores
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in history]
        scores = [entry['risk_score'] for entry in history]
        
        # Convert timestamps to numerical values
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate linear trend
        if len(set(time_numeric)) > 1:
            slope, intercept = np.polyfit(time_numeric, scores, 1)
        else:
            slope = 0.0
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'data_points': len(history),
            'current_risk': scores[-1] if scores else 0.0,
            'average_risk': np.mean(scores) if scores else 0.0
        }
    
    def export_risk_data(self, user_id: str) -> Dict:
        """Export risk data for a user"""
        history = self.get_risk_history(user_id, 168)  # 7 days
        
        if not history:
            return {'user_id': user_id, 'history_available': False}
        
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'history_available': True,
            'total_data_points': len(history),
            'average_risk': np.mean([h['risk_score'] for h in history]),
            'max_risk': max([h['risk_score'] for h in history]),
            'risk_trend': self.calculate_risk_trend(user_id),
            'recent_history': history[-50:]  # Last 50 entries
        }