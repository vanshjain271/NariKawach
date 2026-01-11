import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import AnomalyType


class RouteAnomalyDetector:
    """
    Advanced route anomaly detection system
    Detects deviations from normal routes and unusual movement patterns
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'learning_period_days': 7,
            'max_route_points': 1000,
            'anomaly_threshold': 0.7,
            'cluster_eps': 0.001,
            'cluster_min_samples': 3
        }
        
        self.logger = setup_logger(__name__)
        
        # User route history
        self.user_routes = defaultdict(lambda: {
            'normal_routes': deque(maxlen=self.config['max_route_points']),
            'timestamps': deque(maxlen=self.config['max_route_points']),
            'route_clusters': None,
            'last_updated': None,
            'statistics': {}
        })
        
        # Route clustering
        self.route_clusterer = DBSCAN(
            eps=self.config['cluster_eps'],
            min_samples=self.config['cluster_min_samples']
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, user_id: str, current_location: Dict) -> Dict:
        """
        Detect route anomalies for a user
        """
        try:
            self.logger.debug(f"Detecting anomalies for user {user_id}")
            
            # Get user's route history
            user_data = self.user_routes[user_id]
            
            # Add current location to history
            self._update_user_history(user_id, current_location)
            
            # If insufficient history, return baseline
            if len(user_data['normal_routes']) < 10:
                return self._baseline_assessment(current_location)
            
            # Detect different types of anomalies
            anomalies = {
                'route_deviation': self._detect_route_deviation(user_id, current_location),
                'speed_anomaly': self._detect_speed_anomaly(user_id, current_location),
                'time_anomaly': self._detect_time_anomaly(user_id, current_location),
                'stop_anomaly': self._detect_stop_anomaly(user_id, current_location),
                'direction_anomaly': self._detect_direction_anomaly(user_id, current_location)
                }
                    # Calculate overall anomaly score
            anomaly_score = self._calculate_overall_score(anomalies)
            
            # Determine anomaly type
            anomaly_type = self._determine_anomaly_type(anomalies, anomaly_score)
            
            # Generate insights
            insights = self._generate_insights(anomalies, anomaly_score)
            
            # Update user statistics
            self._update_user_statistics(user_id, anomaly_score)
            
            result = {
                'anomaly_detected': anomaly_score > self.config['anomaly_threshold'],
                'anomaly_score': float(anomaly_score),
                'anomaly_type': anomaly_type,
                'detailed_anomalies': anomalies,
                'insights': insights,
                'confidence': self._calculate_confidence(user_data),
                'recommendations': self._generate_recommendations(anomaly_score, anomalies),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Anomaly detection complete for user {user_id}. Score: {anomaly_score}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for user {user_id}: {e}")
            return {
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'anomaly_type': AnomalyType.ROUTE_DEVIATION,
                'detailed_anomalies': {},
                'insights': ['Error in anomaly detection'],
                'confidence': 0.0,
                'recommendations': ['Continue normal monitoring'],
                'timestamp': datetime.now().isoformat()
            }

    def _update_user_history(self, user_id: str, location: Dict):
        """Update user's location history"""
        user_data = self.user_routes[user_id]
        
        # Convert location to feature vector
        location_vector = self._location_to_vector(location)
        
        # Add to history
        user_data['normal_routes'].append(location_vector)
        user_data['timestamps'].append(
            datetime.fromisoformat(location.get('timestamp', datetime.now().isoformat()))
        )
        
        # Update last updated timestamp
        user_data['last_updated'] = datetime.now()
        
        # Recalculate statistics periodically
        if len(user_data['normal_routes']) % 100 == 0:
            self._recalculate_statistics(user_id)

    def _location_to_vector(self, location: Dict) -> np.ndarray:
        """Convert location dictionary to feature vector"""
        features = [
            location.get('latitude', 0.0),
            location.get('longitude', 0.0),
            location.get('speed', 0.0),
            location.get('accuracy', 0.0),
            location.get('altitude', 0.0),
            datetime.fromisoformat(location.get('timestamp', datetime.now().isoformat())).hour,
            datetime.fromisoformat(location.get('timestamp', datetime.now().isoformat())).weekday()
        ]
        
        return np.array(features)

    def _detect_route_deviation(self, user_id: str, current_location: Dict) -> Dict:
        """Detect deviation from normal routes"""
        try:
            user_data = self.user_routes[user_id]
            
            if not user_data['normal_routes']:
                return {'score': 0.0, 'reason': 'Insufficient history'}
            
            # Convert current location to vector
            current_vector = self._location_to_vector(current_location)
            
            # Get recent routes (last 24 hours)
            recent_routes = list(user_data['normal_routes'])[-100:] if len(user_data['normal_routes']) > 100 else list(user_data['normal_routes'])
            
            if not recent_routes:
                return {'score': 0.0, 'reason': 'No recent routes'}
            
            # Calculate distance to nearest normal route point
            distances = []
            for route_point in recent_routes:
                # Calculate Euclidean distance for spatial features
                spatial_distance = euclidean(current_vector[:2], route_point[:2])
                
                # Adjust for time of day similarity
                time_diff = abs(current_vector[5] - route_point[5])
                time_penalty = 1.0 if time_diff > 4 else (time_diff / 4.0)
                
                # Combined distance
                combined_distance = spatial_distance * (1 + time_penalty * 0.5)
                distances.append(combined_distance)
            
            # Normalize distance to 0-1 score
            if distances:
                max_distance = max(distances) if max(distances) > 0 else 1.0
                deviation_score = min(1.0, np.mean(distances) / max_distance)
            else:
                deviation_score = 0.0
            
            # Check if route clusters exist
            if user_data['route_clusters'] is not None:
                # Predict cluster for current location
                current_scaled = self.scaler.transform([current_vector]) if hasattr(self.scaler, 'n_samples_seen_') else current_vector.reshape(1, -1)
                # This would require a trained clusterer
            
            reason = "Normal route" if deviation_score < 0.3 else "Minor deviation" if deviation_score < 0.6 else "Major deviation"
            
            return {
                'score': float(deviation_score),
                'reason': reason,
                'nearest_distance': float(np.min(distances)) if distances else 0.0,
                'avg_distance': float(np.mean(distances)) if distances else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in route deviation detection: {e}")
            return {'score': 0.0, 'reason': f'Error: {str(e)}'}

    def _detect_speed_anomaly(self, user_id: str, current_location: Dict) -> Dict:
        """Detect unusual speed patterns"""
        try:
            user_data = self.user_routes[user_id]
            current_speed = current_location.get('speed', 0.0)
            
            if not user_data['normal_routes']:
                return {'score': 0.0, 'reason': 'Insufficient history'}
            
            # Extract speed history
            speed_history = [route[2] for route in user_data['normal_routes']]
            
            if len(speed_history) < 10:
                return {'score': 0.0, 'reason': 'Insufficient speed data'}
            
            # Calculate statistics
            avg_speed = np.mean(speed_history)
            std_speed = np.std(speed_history)
            
            if std_speed == 0:
                std_speed = 0.1  # Avoid division by zero
            
            # Calculate z-score
            z_score = abs(current_speed - avg_speed) / std_speed
            
            # Convert to anomaly score (0-1)
            anomaly_score = 1 / (1 + np.exp(-z_score + 3))  # Sigmoid with offset
            
            # Determine reason
            if current_speed > avg_speed + 2 * std_speed:
                reason = "Unusually high speed"
            elif current_speed < avg_speed - 2 * std_speed:
                reason = "Unusually low speed or stopped"
            else:
                reason = "Normal speed"
            
            return {
                'score': float(anomaly_score),
                'reason': reason,
                'current_speed': float(current_speed),
                'average_speed': float(avg_speed),
                'speed_std': float(std_speed),
                'z_score': float(z_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error in speed anomaly detection: {e}")
            return {'score': 0.0, 'reason': f'Error: {str(e)}'}

    def _detect_time_anomaly(self, user_id: str, current_location: Dict) -> Dict:
        """Detect unusual time patterns"""
        try:
            user_data = self.user_routes[user_id]
            current_time = datetime.fromisoformat(
                current_location.get('timestamp', datetime.now().isoformat())
            )
            
            if not user_data['timestamps']:
                return {'score': 0.0, 'reason': 'Insufficient history'}
            
            # Extract hour and day from timestamps
            time_history = [ts for ts in user_data['timestamps']]
            hour_history = [ts.hour for ts in time_history]
            day_history = [ts.weekday() for ts in time_history]
            
            current_hour = current_time.hour
            current_day = current_time.weekday()
            
            # Calculate time probability
            hour_prob = self._calculate_time_probability(
                current_hour, hour_history, 24, 'hour'
            )
            day_prob = self._calculate_time_probability(
                current_day, day_history, 7, 'day'
            )
            
            # Combined time anomaly score
            time_anomaly_score = 1 - ((hour_prob + day_prob) / 2)
            
            # Check for unusual time combinations
            is_night = 22 <= current_hour <= 6
            is_weekend = current_day >= 5
            
            time_context = []
            if is_night:
                time_context.append("Night time")
            if is_weekend:
                time_context.append("Weekend")
            
            reason = "Usual time" if time_anomaly_score < 0.3 else \
                    "Unusual time" if time_anomaly_score < 0.7 else \
                    "Highly unusual time"
            
            if time_context:
                reason += f" ({', '.join(time_context)})"
            
            return {
                'score': float(time_anomaly_score),
                'reason': reason,
                'current_hour': current_hour,
                'current_day': current_day,
                'hour_probability': float(hour_prob),
                'day_probability': float(day_prob),
                'is_night': is_night,
                'is_weekend': is_weekend
            }
            
        except Exception as e:
            self.logger.error(f"Error in time anomaly detection: {e}")
            return {'score': 0.0, 'reason': f'Error: {str(e)}'}

    def _detect_stop_anomaly(self, user_id: str, current_location: Dict) -> Dict:
        """Detect unusual stopping patterns"""
        try:
            user_data = self.user_routes[user_id]
            current_speed = current_location.get('speed', 0.0)
            
            if len(user_data['normal_routes']) < 20:
                return {'score': 0.0, 'reason': 'Insufficient history'}
            
            # Check if stopped (speed < 0.5 m/s)
            is_stopped = current_speed < 0.5
            
            if not is_stopped:
                return {
                    'score': 0.0,
                    'reason': 'Moving normally',
                    'is_stopped': False,
                    'current_speed': float(current_speed)
                }
            
            # Get recent locations to calculate stop duration
            recent_timestamps = list(user_data['timestamps'])[-20:]
            recent_speeds = [route[2] for route in list(user_data['normal_routes'])[-20:]]
            
            # Find consecutive stops
            stop_duration = 0
            for i in range(len(recent_speeds)-1, -1, -1):
                if recent_speeds[i] < 0.5:
                    stop_duration += 1
                else:
                    break
            
            # Convert to minutes (assuming 1 point per minute)
            stop_duration_minutes = stop_duration
            
            # Calculate stop anomaly score
            stop_anomaly_score = min(1.0, stop_duration_minutes / 30.0)  # Max at 30 minutes
            
            reason = f"Stopped for {stop_duration_minutes} minutes"
            if stop_duration_minutes > 10:
                reason += " (unusually long stop)"
            
            return {
                'score': float(stop_anomaly_score),
                'reason': reason,
                'is_stopped': True,
                'stop_duration_minutes': stop_duration_minutes,
                'current_speed': float(current_speed)
            }
            
        except Exception as e:
            self.logger.error(f"Error in stop anomaly detection: {e}")
            return {'score': 0.0, 'reason': f'Error: {str(e)}'}

    def _detect_direction_anomaly(self, user_id: str, current_location: Dict) -> Dict:
        """Detect unusual direction changes"""
        try:
            user_data = self.user_routes[user_id]
            
            if len(user_data['normal_routes']) < 10:
                return {'score': 0.0, 'reason': 'Insufficient history'}
            
            # Get recent locations
            recent_routes = list(user_data['normal_routes'])[-10:]
            
            if len(recent_routes) < 3:
                return {'score': 0.0, 'reason': 'Insufficient recent data'}
            
            # Calculate bearing changes
            bearings = []
            for i in range(1, len(recent_routes)):
                lat1, lon1 = recent_routes[i-1][:2]
                lat2, lon2 = recent_routes[i][:2]
                
                # Calculate bearing
                bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
                bearings.append(bearing)
            
            if len(bearings) < 2:
                return {'score': 0.0, 'reason': 'Insufficient bearing data'}
            
            # Calculate bearing changes
            bearing_changes = []
            for i in range(1, len(bearings)):
                change = abs(bearings[i] - bearings[i-1])
                # Normalize to 0-180 degrees
                change = min(change, 360 - change)
                bearing_changes.append(change)
            
            avg_bearing_change = np.mean(bearing_changes) if bearing_changes else 0
            
            # Calculate anomaly score (sharp turns > 45 degrees are suspicious)
            direction_anomaly_score = min(1.0, avg_bearing_change / 90.0)
            
            reason = "Normal direction changes"
            if avg_bearing_change > 60:
                reason = "Frequent sharp turns"
            elif avg_bearing_change > 30:
                reason = "Moderate direction changes"
            
            return {
                'score': float(direction_anomaly_score),
                'reason': reason,
                'avg_bearing_change': float(avg_bearing_change),
                'recent_bearings': [float(b) for b in bearings[-3:]] if bearings else []
            }
            
        except Exception as e:
            self.logger.error(f"Error in direction anomaly detection: {e}")
            return {'score': 0.0, 'reason': f'Error: {str(e)}'}

    def _calculate_time_probability(self, current_value: int, history: List[int], 
                                max_value: int, unit: str) -> float:
        """Calculate probability of current time value based on history"""
        if not history:
            return 0.5
        
        # Create histogram
        hist, _ = np.histogram(history, bins=max_value, range=(0, max_value))
        
        # Add smoothing
        hist = hist + 1  # Laplace smoothing
        
        # Normalize to probabilities
        prob_dist = hist / hist.sum()
        
        # Return probability of current value
        return prob_dist[current_value]

    def _calculate_bearing(self, lat1: float, lon1: float, 
                        lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        import math
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon_diff_rad = math.radians(lon2 - lon1)
        
        y = math.sin(lon_diff_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff_rad)
        
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def _calculate_overall_score(self, anomalies: Dict) -> float:
        """Calculate overall anomaly score"""
        # Weight different anomaly types
        weights = {
            'route_deviation': 0.35,
            'speed_anomaly': 0.25,
            'time_anomaly': 0.20,
            'stop_anomaly': 0.10,
            'direction_anomaly': 0.10
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for anomaly_type, weight in weights.items():
            if anomaly_type in anomalies:
                anomaly_data = anomalies[anomaly_type]
                if 'score' in anomaly_data:
                    overall_score += anomaly_data['score'] * weight
                    total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        return overall_score

    def _determine_anomaly_type(self, anomalies: Dict, overall_score: float) -> str:
        """Determine the primary anomaly type"""
        if overall_score < 0.3:
            return AnomalyType.ROUTE_DEVIATION
        
        # Find the anomaly with highest score
        max_score = -1
        primary_anomaly = AnomalyType.ROUTE_DEVIATION
        
        for anomaly_type, anomaly_data in anomalies.items():
            if 'score' in anomaly_data and anomaly_data['score'] > max_score:
                max_score = anomaly_data['score']
                
                # Map to enum
                if anomaly_type == 'route_deviation':
                    primary_anomaly = AnomalyType.ROUTE_DEVIATION
                elif anomaly_type == 'speed_anomaly':
                    primary_anomaly = AnomalyType.SPEED_ANOMALY
                elif anomaly_type == 'time_anomaly':
                    primary_anomaly = AnomalyType.TIME_ANOMALY
                elif anomaly_type == 'stop_anomaly':
                    primary_anomaly = AnomalyType.BEHAVIORAL_CHANGE
                elif anomaly_type == 'direction_anomaly':
                    primary_anomaly = AnomalyType.BEHAVIORAL_CHANGE
        
        return primary_anomaly

    def _generate_insights(self, anomalies: Dict, overall_score: float) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        if overall_score > 0.7:
            insights.append("High anomaly detected - requires immediate attention")
        
        # Add specific insights based on anomaly types
        for anomaly_type, anomaly_data in anomalies.items():
            if 'score' in anomaly_data and anomaly_data['score'] > 0.6:
                if 'reason' in anomaly_data:
                    insights.append(f"{anomaly_type.replace('_', ' ').title()}: {anomaly_data['reason']}")
        
        if not insights:
            insights.append("No significant anomalies detected")
        
        return insights

    def _update_user_statistics(self, user_id: str, anomaly_score: float):
        """Update user statistics"""
        user_data = self.user_routes[user_id]
        
        if 'statistics' not in user_data:
            user_data['statistics'] = {
                'total_checks': 0,
                'anomaly_count': 0,
                'avg_anomaly_score': 0.0,
                'last_anomaly': None
            }
        
        stats = user_data['statistics']
        stats['total_checks'] += 1
        
        if anomaly_score > 0.5:
            stats['anomaly_count'] += 1
            stats['last_anomaly'] = datetime.now().isoformat()
        
        # Update average score (exponential moving average)
        alpha = 0.1
        stats['avg_anomaly_score'] = alpha * anomaly_score + (1 - alpha) * stats['avg_anomaly_score']

    def _calculate_confidence(self, user_data: Dict) -> float:
        """Calculate confidence in anomaly detection"""
        stats = user_data.get('statistics', {})
        total_checks = stats.get('total_checks', 0)
        
        if total_checks < 10:
            return 0.5  # Low confidence with little data
        elif total_checks < 50:
            return 0.7  # Moderate confidence
        else:
            return 0.9  # High confidence

    def _generate_recommendations(self, anomaly_score: float, 
                                anomalies: Dict) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if anomaly_score > 0.8:
            recommendations.extend([
                "Enable emergency monitoring",
                "Share live location with guardians",
                "Consider calling emergency services"
            ])
        elif anomaly_score > 0.6:
            recommendations.extend([
                "Increase location sharing frequency",
                "Notify trusted contacts",
                "Move to well-lit, populated area"
            ])
        elif anomaly_score > 0.4:
            recommendations.extend([
                "Stay alert to surroundings",
                "Keep phone accessible",
                "Share your route with someone"
            ])
        
        # Specific recommendations based on anomaly types
        if anomalies.get('speed_anomaly', {}).get('score', 0) > 0.7:
            recommendations.append("Check if moving at unsafe speed")
        
        if anomalies.get('stop_anomaly', {}).get('is_stopped', False):
            recommendations.append("If intentionally stopped, consider moving to safer location")
        
        return recommendations

    def _baseline_assessment(self, current_location: Dict) -> Dict:
        """Return baseline assessment when insufficient data"""
        current_time = datetime.fromisoformat(
            current_location.get('timestamp', datetime.now().isoformat())
        )
        is_night = 22 <= current_time.hour <= 6
        
        baseline_score = 0.3 if is_night else 0.1
        
        return {
            'anomaly_detected': False,
            'anomaly_score': float(baseline_score),
            'anomaly_type': AnomalyType.ROUTE_DEVIATION,
            'detailed_anomalies': {
                'route_deviation': {'score': baseline_score, 'reason': 'Baseline assessment'},
                'speed_anomaly': {'score': 0.1, 'reason': 'Insufficient data'},
                'time_anomaly': {'score': 0.1, 'reason': 'Insufficient data'},
                'stop_anomaly': {'score': 0.1, 'reason': 'Insufficient data'},
                'direction_anomaly': {'score': 0.1, 'reason': 'Insufficient data'}
            },
            'insights': ['Learning normal patterns - collect more data'],
            'confidence': 0.3,
            'recommendations': ['Continue normal activity while system learns your patterns'],
            'timestamp': datetime.now().isoformat()
        }

    def _recalculate_statistics(self, user_id: str):
        """Recalculate user statistics and clusters"""
        user_data = self.user_routes[user_id]
        
        if len(user_data['normal_routes']) < 20:
            return
        
        # Convert routes to array
        routes_array = np.array(list(user_data['normal_routes']))
        
        # Standardize features for clustering
        if len(routes_array) > 0:
            self.scaler.fit(routes_array)
            
            # Cluster routes
            routes_scaled = self.scaler.transform(routes_array)
            clusters = self.route_clusterer.fit_predict(routes_scaled[:, :2])  # Only spatial features
            
            user_data['route_clusters'] = clusters
            
            # Calculate cluster statistics
            unique_clusters = set(clusters)
            cluster_stats = {}
            
            for cluster in unique_clusters:
                if cluster != -1:  # Ignore noise
                    cluster_points = routes_array[clusters == cluster]
                    cluster_stats[cluster] = {
                        'size': len(cluster_points),
                        'center': np.mean(cluster_points[:, :2], axis=0).tolist(),
                        'std': np.std(cluster_points[:, :2], axis=0).tolist()
                    }
            
            user_data['cluster_stats'] = cluster_stats

    def get_user_statistics(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        if user_id not in self.user_routes:
            return {}
        
        user_data = self.user_routes[user_id]
        
        return {
            'total_data_points': len(user_data['normal_routes']),
            'statistics': user_data.get('statistics', {}),
            'cluster_count': len(user_data.get('cluster_stats', {})),
            'last_updated': user_data.get('last_updated'),
            'learning_progress': min(1.0, len(user_data['normal_routes']) / 100.0)
        }

    def reset_user_data(self, user_id: str):
        """Reset all data for a user"""
        if user_id in self.user_routes:
            del self.user_routes[user_id]
            self.logger.info(f"Reset data for user {user_id}")

    def export_user_patterns(self, user_id: str) -> Dict:
        """Export learned patterns for a user"""
        if user_id not in self.user_routes:
            return {}
        
        user_data = self.user_routes[user_id]
        
        return {
            'user_id': user_id,
            'data_points': len(user_data['normal_routes']),
            'statistics': user_data.get('statistics', {}),
            'cluster_stats': user_data.get('cluster_stats', {}),
            'last_updated': user_data.get('last_updated'),
            'export_timestamp': datetime.now().isoformat()
        }