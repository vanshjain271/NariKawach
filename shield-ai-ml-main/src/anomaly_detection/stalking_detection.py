
### **src/anomaly_detection/stalking_detection.py**

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import json
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import AnomalyType


class AdvancedStalkingDetector:
    """
    Advanced stalking pattern detection system
    Detects repeated proximity patterns and suspicious device tracking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'coincidence_window_minutes': 30,
            'min_coincidence_count': 3,
            'proximity_threshold_meters': 100,
            'time_threshold_minutes': 10,
            'stalking_confidence_threshold': 0.7,
            'device_history_days': 30
        }
        
        self.logger = setup_logger(__name__)
        
        # Device tracking
        self.device_profiles = defaultdict(lambda: {
            'encounters': deque(maxlen=1000),
            'locations': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000),
            'user_encounters': defaultdict(lambda: deque(maxlen=100)),
            'risk_score': 0.0,
            'last_seen': None,
            'first_seen': None,
            'patterns': []
        })
        
        # User behavior
        self.user_profiles = defaultdict(lambda: {
            'encountered_devices': defaultdict(lambda: deque(maxlen=100)),
            'normal_routes': defaultdict(list),
            'suspicious_encounters': deque(maxlen=100),
            'stalking_alerts': deque(maxlen=50)
        })
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Pattern cache
        self.pattern_cache = {}
    
    def detect_stalking_patterns(self, user_id: str, 
                                current_location: Dict,
                                nearby_devices: List[Dict]) -> Dict:
        """
        Detect stalking patterns for a user
        """
        try:
            self.logger.debug(f"Detecting stalking patterns for user {user_id}")
            
            # Update device profiles
            self._update_device_profiles(user_id, current_location, nearby_devices)
            
            # Analyze patterns
            patterns = {
                'device_coincidence': self._analyze_device_coincidence(user_id, nearby_devices),
                'route_following': self._analyze_route_following(user_id, current_location, nearby_devices),
                'temporal_patterns': self._analyze_temporal_patterns(user_id, nearby_devices),
                'proximity_analysis': self._analyze_proximity_patterns(user_id, current_location, nearby_devices)
            }
            
            # Calculate overall stalking risk
            stalking_risk = self._calculate_stalking_risk(patterns)
            
            # Detect specific stalking patterns
            detected_patterns = self._detect_specific_patterns(patterns, stalking_risk)
            
            # Update user profile
            self._update_user_profile(user_id, patterns, stalking_risk)
            
            # Generate results
            result = {
                'stalking_risk': float(stalking_risk),
                'stalking_detected': stalking_risk > self.config['stalking_confidence_threshold'],
                'detected_patterns': detected_patterns,
                'detailed_analysis': patterns,
                'suspicious_devices': self._identify_suspicious_devices(user_id),
                'confidence': self._calculate_detection_confidence(user_id),
                'recommendations': self._generate_stalking_recommendations(stalking_risk, patterns),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Stalking detection complete for user {user_id}. Risk: {stalking_risk}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting stalking patterns for user {user_id}: {e}")
            return {
                'stalking_risk': 0.0,
                'stalking_detected': False,
                'detected_patterns': [],
                'detailed_analysis': {},
                'suspicious_devices': [],
                'confidence': 0.0,
                'recommendations': ['Continue normal monitoring'],
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_device_profiles(self, user_id: str, 
                               current_location: Dict,
                               nearby_devices: List[Dict]):
        """Update profiles for nearby devices"""
        current_time = datetime.now()
        
        for device in nearby_devices:
            device_id = device.get('device_id')
            if not device_id:
                continue
            
            device_profile = self.device_profiles[device_id]
            
            # Update device metadata
            if device_profile['first_seen'] is None:
                device_profile['first_seen'] = current_time
            
            device_profile['last_seen'] = current_time
            
            # Record encounter
            encounter = {
                'user_id': user_id,
                'timestamp': current_time,
                'location': {
                    'latitude': current_location.get('latitude'),
                    'longitude': current_location.get('longitude')
                },
                'device_data': device,
                'distance': device.get('distance_meters', 0)
            }
            
            device_profile['encounters'].append(encounter)
            
            # Update user-specific encounters
            device_profile['user_encounters'][user_id].append({
                'timestamp': current_time,
                'location': encounter['location'],
                'distance': encounter['distance']
            })
            
            # Update locations
            device_profile['locations'].append(encounter['location'])
            device_profile['timestamps'].append(current_time)
    
    def _analyze_device_coincidence(self, user_id: str, 
                                   nearby_devices: List[Dict]) -> Dict:
        """Analyze device coincidence patterns"""
        try:
            user_profile = self.user_profiles[user_id]
            current_time = datetime.now()
            
            # Track devices seen multiple times
            coincidence_scores = {}
            
            for device in nearby_devices:
                device_id = device.get('device_id')
                if not device_id:
                    continue
                
                device_profile = self.device_profiles[device_id]
                
                # Get encounters with this user
                user_encounters = list(device_profile['user_encounters'][user_id])
                
                if len(user_encounters) < 2:
                    coincidence_score = 0.0
                else:
                    # Calculate time between encounters
                    time_diffs = []
                    for i in range(1, len(user_encounters)):
                        diff = (user_encounters[i]['timestamp'] - 
                               user_encounters[i-1]['timestamp']).total_seconds() / 60  # minutes
                        time_diffs.append(diff)
                    
                    # Score based on frequency and regularity
                    if time_diffs:
                        avg_time_diff = np.mean(time_diffs)
                        std_time_diff = np.std(time_diffs)
                        
                        # More frequent and regular encounters = higher suspicion
                        frequency_score = 1.0 / (1.0 + avg_time_diff/60)  # Normalize
                        regularity_score = 1.0 / (1.0 + std_time_diff/30)  # Normalize
                        
                        coincidence_score = 0.6 * frequency_score + 0.4 * regularity_score
                    else:
                        coincidence_score = 0.0
                
                coincidence_scores[device_id] = {
                    'score': coincidence_score,
                    'encounter_count': len(user_encounters),
                    'first_encounter': user_encounters[0]['timestamp'].isoformat() if user_encounters else None,
                    'last_encounter': user_encounters[-1]['timestamp'].isoformat() if user_encounters else None
                }
            
            # Calculate overall coincidence risk
            if coincidence_scores:
                max_score = max([s['score'] for s in coincidence_scores.values()])
                avg_score = np.mean([s['score'] for s in coincidence_scores.values()])
            else:
                max_score = 0.0
                avg_score = 0.0
            
            return {
                'coincidence_scores': coincidence_scores,
                'max_coincidence_score': float(max_score),
                'avg_coincidence_score': float(avg_score),
                'high_risk_devices': [
                    device_id for device_id, scores in coincidence_scores.items()
                    if scores['score'] > 0.6
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in device coincidence analysis: {e}")
            return {
                'coincidence_scores': {},
                'max_coincidence_score': 0.0,
                'avg_coincidence_score': 0.0,
                'high_risk_devices': []
            }
    
    def _analyze_route_following(self, user_id: str,
                                current_location: Dict,
                                nearby_devices: List[Dict]) -> Dict:
        """Analyze if devices are following user's route"""
        try:
            user_profile = self.user_profiles[user_id]
            route_following_scores = {}
            
            # Get user's recent route
            user_recent_locations = self._get_user_recent_locations(user_id, count=20)
            
            if len(user_recent_locations) < 5:
                return {
                    'route_following_scores': {},
                    'max_following_score': 0.0,
                    'detailed_analysis': 'Insufficient user route data'
                }
            
            for device in nearby_devices:
                device_id = device.get('device_id')
                if not device_id:
                    continue
                
                device_profile = self.device_profiles[device_id]
                
                # Get device's recent locations
                device_locations = list(device_profile['locations'])
                
                if len(device_locations) < 5:
                    route_following_scores[device_id] = {
                        'score': 0.0,
                        'reason': 'Insufficient device location data'
                    }
                    continue
                
                # Calculate route similarity using DTW
                try:
                    # Convert to arrays for DTW
                    user_route = np.array([
                        [loc['latitude'], loc['longitude']] 
                        for loc in user_recent_locations[-10:]
                    ])
                    
                    device_route = np.array([
                        [loc['latitude'], loc['longitude']] 
                        for loc in device_locations[-10:]
                    ])
                    
                    # Calculate DTW distance
                    dtw_distance, _ = fastdtw(user_route, device_route, dist=euclidean)
                    
                    # Normalize distance to score (0-1, lower distance = higher score)
                    normalized_distance = dtw_distance / 1000  # Assuming 1km max
                    following_score = 1.0 / (1.0 + normalized_distance)
                    
                    # Adjust for temporal alignment
                    time_correlation = self._calculate_time_correlation(
                        user_id, device_id
                    )
                    
                    following_score = 0.7 * following_score + 0.3 * time_correlation
                    
                    route_following_scores[device_id] = {
                        'score': float(following_score),
                        'dtw_distance': float(dtw_distance),
                        'time_correlation': float(time_correlation),
                        'reason': 'Route similarity detected' if following_score > 0.5 else 'Normal route variation'
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error calculating route following for device {device_id}: {e}")
                    route_following_scores[device_id] = {
                        'score': 0.0,
                        'reason': f'Analysis error: {str(e)}'
                    }
            
            # Calculate overall route following risk
            if route_following_scores:
                max_score = max([s['score'] for s in route_following_scores.values()])
            else:
                max_score = 0.0
            
            return {
                'route_following_scores': route_following_scores,
                'max_following_score': float(max_score),
                'devices_following_route': [
                    device_id for device_id, scores in route_following_scores.items()
                    if scores['score'] > 0.6
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in route following analysis: {e}")
            return {
                'route_following_scores': {},
                'max_following_score': 0.0,
                'devices_following_route': []
            }
    
    def _analyze_temporal_patterns(self, user_id: str,
                                  nearby_devices: List[Dict]) -> Dict:
        """Analyze temporal patterns of device encounters"""
        try:
            temporal_patterns = {}
            current_time = datetime.now()
            
            for device in nearby_devices:
                device_id = device.get('device_id')
                if not device_id:
                    continue
                
                device_profile = self.device_profiles[device_id]
                user_encounters = list(device_profile['user_encounters'][user_id])
                
                if len(user_encounters) < 3:
                    temporal_patterns[device_id] = {
                        'score': 0.0,
                        'pattern': 'insufficient_data',
                        'encounter_times': [],
                        'time_consistency': 0.0
                    }
                    continue
                
                # Extract encounter times
                encounter_times = [e['timestamp'] for e in user_encounters]
                encounter_hours = [t.hour for t in encounter_times]
                encounter_days = [t.weekday() for t in encounter_times]
                
                # Analyze temporal consistency
                hour_consistency = 1.0 - (np.std(encounter_hours) / 12.0)  # Normalize by half day
                day_consistency = 1.0 - (np.std(encounter_days) / 3.5)    # Normalize by half week
                
                temporal_consistency = 0.6 * hour_consistency + 0.4 * day_consistency
                
                # Check for patterns (same time each day, etc.)
                patterns = []
                
                # Same hour pattern
                hour_counts = {}
                for hour in encounter_hours:
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                
                if max(hour_counts.values(), default=0) > len(encounter_hours) * 0.5:
                    patterns.append('consistent_hour')
                
                # Same day pattern
                day_counts = {}
                for day in encounter_days:
                    day_counts[day] = day_counts.get(day, 0) + 1
                
                if max(day_counts.values(), default=0) > len(encounter_days) * 0.5:
                    patterns.append('consistent_day')
                
                temporal_patterns[device_id] = {
                    'score': float(temporal_consistency),
                    'pattern': patterns[0] if patterns else 'random',
                    'encounter_count': len(user_encounters),
                    'hour_consistency': float(hour_consistency),
                    'day_consistency': float(day_consistency),
                    'most_common_hour': max(hour_counts, key=hour_counts.get) if hour_counts else None,
                    'most_common_day': max(day_counts, key=day_counts.get) if day_counts else None
                }
            
            # Calculate overall temporal pattern risk
            if temporal_patterns:
                max_score = max([s['score'] for s in temporal_patterns.values()])
                pattern_devices = [
                    device_id for device_id, patterns in temporal_patterns.items()
                    if patterns['score'] > 0.6 and patterns['pattern'] != 'random'
                ]
            else:
                max_score = 0.0
                pattern_devices = []
            
            return {
                'temporal_patterns': temporal_patterns,
                'max_temporal_score': float(max_score),
                'pattern_devices': pattern_devices,
                'overall_temporal_risk': float(max_score) if max_score > 0.7 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal pattern analysis: {e}")
            return {
                'temporal_patterns': {},
                'max_temporal_score': 0.0,
                'pattern_devices': [],
                'overall_temporal_risk': 0.0
            }
    
    def _analyze_proximity_patterns(self, user_id: str,
                                   current_location: Dict,
                                   nearby_devices: List[Dict]) -> Dict:
        """Analyze proximity patterns and duration"""
        try:
            proximity_patterns = {}
            
            for device in nearby_devices:
                device_id = device.get('device_id')
                distance = device.get('distance_meters', 0)
                
                if not device_id:
                    continue
                
                device_profile = self.device_profiles[device_id]
                user_encounters = list(device_profile['user_encounters'][user_id])
                
                # Calculate proximity duration
                proximity_durations = []
                current_proximity_start = None
                
                for i in range(len(user_encounters)):
                    encounter = user_encounters[i]
                    
                    if encounter['distance'] < self.config['proximity_threshold_meters']:
                        if current_proximity_start is None:
                            current_proximity_start = encounter['timestamp']
                    else:
                        if current_proximity_start is not None:
                            duration = (encounter['timestamp'] - current_proximity_start).total_seconds() / 60
                            proximity_durations.append(duration)
                            current_proximity_start = None
                
                # Handle ongoing proximity
                if current_proximity_start is not None:
                    duration = (datetime.now() - current_proximity_start).total_seconds() / 60
                    proximity_durations.append(duration)
                
                # Calculate proximity score
                if proximity_durations:
                    avg_duration = np.mean(proximity_durations)
                    max_duration = max(proximity_durations)
                    
                    # Score based on duration and frequency
                    duration_score = min(1.0, avg_duration / 30.0)  # Normalize by 30 minutes
                    frequency_score = min(1.0, len(proximity_durations) / 5.0)  # Normalize by 5 occurrences
                    
                    proximity_score = 0.7 * duration_score + 0.3 * frequency_score
                else:
                    avg_duration = 0.0
                    max_duration = 0.0
                    proximity_score = 0.0
                
                proximity_patterns[device_id] = {
                    'score': float(proximity_score),
                    'current_distance': float(distance),
                    'avg_proximity_duration': float(avg_duration),
                    'max_proximity_duration': float(max_duration),
                    'proximity_count': len(proximity_durations),
                    'is_currently_near': distance < self.config['proximity_threshold_meters']
                }
            
            # Calculate overall proximity risk
            if proximity_patterns:
                max_score = max([s['score'] for s in proximity_patterns.values()])
                high_proximity_devices = [
                    device_id for device_id, patterns in proximity_patterns.items()
                    if patterns['score'] > 0.6
                ]
            else:
                max_score = 0.0
                high_proximity_devices = []
            
            return {
                'proximity_patterns': proximity_patterns,
                'max_proximity_score': float(max_score),
                'high_proximity_devices': high_proximity_devices,
                'current_nearby_count': len([
                    d for d in nearby_devices 
                    if d.get('distance_meters', 0) < self.config['proximity_threshold_meters']
                ])
            }
            
        except Exception as e:
            self.logger.error(f"Error in proximity pattern analysis: {e}")
            return {
                'proximity_patterns': {},
                'max_proximity_score': 0.0,
                'high_proximity_devices': [],
                'current_nearby_count': 0
            }
    
    def _calculate_stalking_risk(self, patterns: Dict) -> float:
        """Calculate overall stalking risk from pattern analysis"""
        # Weight different pattern types
        weights = {
            'device_coincidence': 0.30,
            'route_following': 0.35,
            'temporal_patterns': 0.20,
            'proximity_analysis': 0.15
        }
        
        risk_score = 0.0
        total_weight = 0.0
        
        for pattern_type, weight in weights.items():
            if pattern_type in patterns:
                pattern_data = patterns[pattern_type]
                
                # Extract max score from pattern analysis
                if pattern_type == 'device_coincidence':
                    score = pattern_data.get('max_coincidence_score', 0.0)
                elif pattern_type == 'route_following':
                    score = pattern_data.get('max_following_score', 0.0)
                elif pattern_type == 'temporal_patterns':
                    score = pattern_data.get('max_temporal_score', 0.0)
                elif pattern_type == 'proximity_analysis':
                    score = pattern_data.get('max_proximity_score', 0.0)
                else:
                    score = 0.0
                
                risk_score += score * weight
                total_weight += weight
        
        # Normalize risk score
        if total_weight > 0:
            risk_score = risk_score / total_weight
        
        return risk_score
    
    def _detect_specific_patterns(self, patterns: Dict, 
                                 stalking_risk: float) -> List[str]:
        """Detect specific stalking patterns"""
        detected_patterns = []
        
        if stalking_risk > 0.7:
            detected_patterns.append("HIGH_RISK_STALKING_PATTERN")
        
        # Check device coincidence
        coincidence_data = patterns.get('device_coincidence', {})
        if coincidence_data.get('max_coincidence_score', 0.0) > 0.7:
            detected_patterns.append("REPEATED_DEVICE_COINCIDENCE")
        
        # Check route following
        route_data = patterns.get('route_following', {})
        if route_data.get('max_following_score', 0.0) > 0.7:
            detected_patterns.append("ROUTE_FOLLOWING")
        
        # Check temporal patterns
        temporal_data = patterns.get('temporal_patterns', {})
        if temporal_data.get('max_temporal_score', 0.0) > 0.7:
            detected_patterns.append("REGULAR_TEMPORAL_PATTERN")
        
        # Check proximity patterns
        proximity_data = patterns.get('proximity_analysis', {})
        if proximity_data.get('max_proximity_score', 0.0) > 0.7:
            detected_patterns.append("PERSISTENT_PROXIMITY")
        
        return detected_patterns
    
    def _update_user_profile(self, user_id: str, patterns: Dict, 
                            stalking_risk: float):
        """Update user profile with stalking analysis results"""
        user_profile = self.user_profiles[user_id]
        
        if stalking_risk > 0.6:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'risk_score': stalking_risk,
                'detected_patterns': self._detect_specific_patterns(patterns, stalking_risk),
                'analysis': patterns
            }
            user_profile['stalking_alerts'].append(alert)
    
    def _identify_suspicious_devices(self, user_id: str) -> List[Dict]:
        """Identify suspicious devices for a user"""
        suspicious_devices = []
        
        for device_id, device_profile in self.device_profiles.items():
            # Check if device has encounters with this user
            user_encounters = list(device_profile['user_encounters'][user_id])
            
            if len(user_encounters) >= 3:
                # Calculate device risk score
                risk_factors = []
                
                # Frequency factor
                if len(user_encounters) > 5:
                    risk_factors.append(('high_frequency', 0.3))
                
                # Recency factor
                last_encounter = user_encounters[-1]['timestamp']
                hours_since = (datetime.now() - last_encounter).total_seconds() / 3600
                if hours_since < 24:
                    risk_factors.append(('recent_encounter', 0.2))
                
                # Proximity factor
                avg_distance = np.mean([e['distance'] for e in user_encounters])
                if avg_distance < 50:
                    risk_factors.append(('close_proximity', 0.25))
                
                # Temporal pattern factor
                encounter_hours = [e['timestamp'].hour for e in user_encounters]
                if np.std(encounter_hours) < 3:  # Consistent times
                    risk_factors.append(('temporal_pattern', 0.25))
                
                # Calculate risk score
                risk_score = sum([weight for _, weight in risk_factors])
                
                if risk_score > 0.5:
                    suspicious_devices.append({
                        'device_id': device_id,
                        'risk_score': risk_score,
                        'encounter_count': len(user_encounters),
                        'first_seen': device_profile['first_seen'].isoformat() if device_profile['first_seen'] else None,
                        'last_seen': device_profile['last_seen'].isoformat() if device_profile['last_seen'] else None,
                        'risk_factors': [factor for factor, _ in risk_factors]
                    })
        
        # Sort by risk score
        suspicious_devices.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return suspicious_devices
    
    def _calculate_detection_confidence(self, user_id: str) -> float:
        """Calculate confidence in stalking detection"""
        user_profile = self.user_profiles[user_id]
        
        # Base confidence on data volume
        total_encounters = sum([
            len(device_profile['user_encounters'][user_id])
            for device_profile in self.device_profiles.values()
        ])
        
        if total_encounters < 10:
            return 0.4
        elif total_encounters < 50:
            return 0.7
        else:
            return 0.9
    
    def _generate_stalking_recommendations(self, stalking_risk: float,
                                          patterns: Dict) -> List[str]:
        """Generate recommendations based on stalking risk"""
        recommendations = []
        
        if stalking_risk > 0.8:
            recommendations.extend([
                "IMMEDIATE: Contact authorities if feeling unsafe",
                "Share live location with trusted contacts",
                "Avoid isolated areas",
                "Document all suspicious encounters",
                "Consider changing regular routes and schedules"
            ])
        elif stalking_risk > 0.6:
            recommendations.extend([
                "Increase situational awareness",
                "Vary your daily routes and times",
                "Share your concerns with trusted friends/family",
                "Keep phone charged and accessible",
                "Note descriptions of suspicious individuals"
            ])
        elif stalking_risk > 0.4:
            recommendations.extend([
                "Stay alert to surroundings",
                "Trust your instincts",
                "Keep emergency contacts readily available",
                "Use well-lit, populated routes"
            ])
        
        return recommendations
    
    def _get_user_recent_locations(self, user_id: str, count: int = 20) -> List[Dict]:
        """Get user's recent locations from all device encounters"""
        locations = []
        
        for device_profile in self.device_profiles.values():
            user_encounters = device_profile['user_encounters'].get(user_id, [])
            for encounter in user_encounters[-5:]:  # Get last 5 per device
                locations.append({
                    'timestamp': encounter['timestamp'],
                    'location': encounter['location']
                })
        
        # Sort by timestamp and get most recent
        locations.sort(key=lambda x: x['timestamp'], reverse=True)
        return locations[:count]
    
    def _calculate_time_correlation(self, user_id: str, device_id: str) -> float:
        """Calculate time correlation between user and device movements"""
        device_profile = self.device_profiles[device_id]
        user_encounters = device_profile['user_encounters'][user_id]
        
        if len(user_encounters) < 3:
            return 0.0
        
        # Calculate time intervals between encounters
        intervals = []
        for i in range(1, len(user_encounters)):
            interval = (user_encounters[i]['timestamp'] - 
                       user_encounters[i-1]['timestamp']).total_seconds() / 3600
            intervals.append(interval)
        
        # Calculate consistency (lower std = more consistent)
        if intervals:
            std_intervals = np.std(intervals)
            # Normalize: 0 std = perfect consistency (1.0), >24h std = poor consistency (0.0)
            consistency = 1.0 / (1.0 + std_intervals / 12.0)
            return consistency
        
        return 0.0
    
    def get_device_statistics(self, device_id: str) -> Dict:
        """Get statistics for a specific device"""
        if device_id not in self.device_profiles:
            return {}
        
        device_profile = self.device_profiles[device_id]
        
        return {
            'device_id': device_id,
            'total_encounters': len(device_profile['encounters']),
            'unique_users': len(device_profile['user_encounters']),
            'first_seen': device_profile['first_seen'].isoformat() if device_profile['first_seen'] else None,
            'last_seen': device_profile['last_seen'].isoformat() if device_profile['last_seen'] else None,
            'risk_score': device_profile['risk_score'],
            'encounter_patterns': device_profile['patterns']
        }
    
    def get_user_stalking_history(self, user_id: str) -> Dict:
        """Get stalking history for a user"""
        if user_id not in self.user_profiles:
            return {}
        
        user_profile = self.user_profiles[user_id]
        
        return {
            'user_id': user_id,
            'total_alerts': len(user_profile['stalking_alerts']),
            'recent_alerts': list(user_profile['stalking_alerts'])[-10:],
            'suspicious_encounters': len(user_profile['suspicious_encounters']),
            'monitored_devices': len(user_profile['encountered_devices'])
        }
    
    def reset_device_data(self, device_id: str):
        """Reset all data for a device"""
        if device_id in self.device_profiles:
            del self.device_profiles[device_id]
            self.logger.info(f"Reset data for device {device_id}")
    
    def export_detection_data(self, user_id: str) -> Dict:
        """Export stalking detection data for a user"""
        user_profile = self.user_profiles.get(user_id, {})
        
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'stalking_alerts': list(user_profile.get('stalking_alerts', [])),
            'suspicious_devices': self._identify_suspicious_devices(user_id),
            'detection_statistics': {
                'total_devices_encountered': len(user_profile.get('encountered_devices', {})),
                'total_alerts': len(user_profile.get('stalking_alerts', [])),
                'high_risk_alerts': len([
                    alert for alert in user_profile.get('stalking_alerts', [])
                    if alert.get('risk_score', 0) > 0.7
                ])
            }
        }