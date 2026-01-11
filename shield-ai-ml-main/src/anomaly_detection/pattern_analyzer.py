import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
from loguru import logger
from ...utils.logger import setup_logger


class PatternAnalyzer:
    """
    Advanced pattern analysis for safety system
    Analyzes user behavior patterns and detects anomalies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'pattern_memory_days': 30,
            'cluster_count': 5,
            'pattern_threshold': 0.8,
            'seasonality_detection': True,
            'trend_detection': True
        }
        
        self.logger = setup_logger(__name__)
        
        # Pattern storage
        self.user_patterns = defaultdict(lambda: {
            'location_patterns': defaultdict(list),
            'time_patterns': defaultdict(list),
            'movement_patterns': defaultdict(list),
            'behavior_clusters': None,
            'pattern_signatures': {},
            'last_analysis': None
        })
        
        # Analysis models
        self.cluster_models = {}
        
        # Cache for frequent patterns
        self.pattern_cache = {}
    
    def analyze_user_patterns(self, user_id: str, 
                             user_data: Dict) -> Dict:
        """
        Analyze user patterns comprehensively
        """
        try:
            self.logger.debug(f"Analyzing patterns for user {user_id}")
            
            analyses = {
                'location_analysis': self._analyze_location_patterns(user_id, user_data),
                'time_analysis': self._analyze_time_patterns(user_id, user_data),
                'movement_analysis': self._analyze_movement_patterns(user_id, user_data),
                'behavior_clustering': self._cluster_behavior_patterns(user_id, user_data),
                'pattern_deviation': self._calculate_pattern_deviation(user_id, user_data),
                'seasonality_detection': self._detect_seasonality(user_id, user_data)
            }
            
            # Generate pattern signature
            pattern_signature = self._generate_pattern_signature(analyses)
            
            # Update user patterns
            self._update_user_patterns(user_id, analyses, pattern_signature)
            
            # Calculate pattern consistency
            pattern_consistency = self._calculate_pattern_consistency(user_id, analyses)
            
            result = {
                'pattern_analyses': analyses,
                'pattern_signature': pattern_signature,
                'pattern_consistency': pattern_consistency,
                'behavior_summary': self._generate_behavior_summary(analyses),
                'anomaly_flags': self._detect_pattern_anomalies(analyses, pattern_consistency),
                'predictions': self._generate_pattern_predictions(user_id, analyses),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Pattern analysis complete for user {user_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns for user {user_id}: {e}")
            return {
                'pattern_analyses': {},
                'pattern_signature': {},
                'pattern_consistency': 0.0,
                'behavior_summary': 'Pattern analysis failed',
                'anomaly_flags': [],
                'predictions': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_location_patterns(self, user_id: str, 
                                  user_data: Dict) -> Dict:
        """Analyze location patterns and frequent places"""
        try:
            locations = user_data.get('locations', [])
            
            if not locations:
                return {'error': 'No location data available'}
            
            # Extract location data
            lats = [loc.get('latitude', 0) for loc in locations]
            lons = [loc.get('longitude', 0) for loc in locations]
            timestamps = [loc.get('timestamp') for loc in locations]
            
            # Cluster locations to find frequent places
            location_points = np.column_stack([lats, lons])
            
            if len(location_points) >= self.config['cluster_count']:
                kmeans = KMeans(
                    n_clusters=min(self.config['cluster_count'], len(location_points)),
                    random_state=42,
                    n_init=10
                )
                clusters = kmeans.fit_predict(location_points)
                
                # Calculate cluster statistics
                cluster_stats = {}
                for cluster_id in range(kmeans.n_clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_points = location_points[cluster_mask]
                    
                    if len(cluster_points) > 0:
                        cluster_stats[cluster_id] = {
                            'center': kmeans.cluster_centers_[cluster_id].tolist(),
                            'size': len(cluster_points),
                            'radius': np.max([
                                np.linalg.norm(point - kmeans.cluster_centers_[cluster_id])
                                for point in cluster_points
                            ]) if len(cluster_points) > 1 else 0.0,
                            'visit_times': [
                                datetime.fromisoformat(timestamps[i]).hour 
                                for i, mask in enumerate(cluster_mask) if mask
                            ] if timestamps[0] else []
                        }
                
                # Calculate location entropy (regularity)
                if clusters.size > 0:
                    cluster_counts = np.bincount(clusters)
                    cluster_probs = cluster_counts / clusters.size
                    location_entropy = stats.entropy(cluster_probs)
                    location_regularity = 1.0 / (1.0 + location_entropy)
                else:
                    location_regularity = 0.0
                
                # Identify primary locations
                primary_locations = []
                for cluster_id, stats in cluster_stats.items():
                    if stats['size'] > len(locations) * 0.1:  # At least 10% of visits
                        primary_locations.append({
                            'cluster_id': cluster_id,
                            'center': stats['center'],
                            'visit_count': stats['size'],
                            'visit_frequency': stats['size'] / len(locations)
                        })
                
                # Sort by frequency
                primary_locations.sort(key=lambda x: x['visit_count'], reverse=True)
                
                return {
                    'total_locations': len(locations),
                    'location_clusters': cluster_stats,
                    'primary_locations': primary_locations[:5],  # Top 5
                    'location_regularity': float(location_regularity),
                    'location_diversity': len(cluster_stats),
                    'home_location': primary_locations[0] if primary_locations else None,
                    'work_location': primary_locations[1] if len(primary_locations) > 1 else None
                }
            
            else:
                return {
                    'total_locations': len(locations),
                    'location_clusters': {},
                    'primary_locations': [],
                    'location_regularity': 0.0,
                    'location_diversity': 0,
                    'home_location': None,
                    'work_location': None
                }
            
        except Exception as e:
            self.logger.error(f"Error in location pattern analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_time_patterns(self, user_id: str, 
                              user_data: Dict) -> Dict:
        """Analyze temporal patterns and routines"""
        try:
            locations = user_data.get('locations', [])
            
            if not locations:
                return {'error': 'No location data available'}
            
            # Extract time data
            timestamps = []
            for loc in locations:
                if 'timestamp' in loc:
                    try:
                        timestamps.append(datetime.fromisoformat(loc['timestamp']))
                    except:
                        continue
            
            if not timestamps:
                return {'error': 'No valid timestamps'}
            
            # Extract time features
            hours = [ts.hour for ts in timestamps]
            days = [ts.weekday() for ts in timestamps]
            dates = [ts.date() for ts in timestamps]
            
            # Calculate daily patterns
            hour_distribution = {}
            for hour in range(24):
                hour_distribution[hour] = hours.count(hour)
            
            # Find peak hours
            peak_hours = sorted(
                hour_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Calculate day patterns
            day_distribution = {}
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in range(7):
                day_distribution[day_names[day]] = days.count(day)
            
            # Calculate routine strength
            hour_entropy = stats.entropy(list(hour_distribution.values()))
            day_entropy = stats.entropy(list(day_distribution.values()))
            
            routine_strength = 1.0 - ((hour_entropy + day_entropy) / (np.log(24) + np.log(7)))
            
            # Detect daily routine
            daily_routine = []
            for hour, count in peak_hours:
                daily_routine.append({
                    'hour': hour,
                    'activity_level': count / len(hours),
                    'label': self._get_time_label(hour)
                })
            
            # Calculate active hours
            active_hours = [hour for hour, count in hour_distribution.items() 
                           if count > len(hours) * 0.05]  # More than 5% of activity
            
            return {
                'total_timestamps': len(timestamps),
                'hour_distribution': hour_distribution,
                'day_distribution': day_distribution,
                'peak_hours': peak_hours,
                'daily_routine': daily_routine,
                'active_hours': active_hours,
                'routine_strength': float(routine_strength),
                'is_night_owl': sum(hour_distribution.get(h, 0) for h in range(22, 24) + list(range(0, 6))) > len(hours) * 0.3,
                'is_early_bird': sum(hour_distribution.get(h, 0) for h in range(5, 10)) > len(hours) * 0.3,
                'weekday_activity': sum(days.count(d) for d in range(5)) / len(days) if len(days) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in time pattern analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_movement_patterns(self, user_id: str, 
                                  user_data: Dict) -> Dict:
        """Analyze movement patterns and transportation modes"""
        try:
            locations = user_data.get('locations', [])
            
            if len(locations) < 2:
                return {'error': 'Insufficient movement data'}
            
            # Calculate movement statistics
            speeds = []
            distances = []
            directions = []
            
            for i in range(1, len(locations)):
                loc1 = locations[i-1]
                loc2 = locations[i]
                
                # Calculate distance
                lat1, lon1 = loc1.get('latitude', 0), loc1.get('longitude', 0)
                lat2, lon2 = loc2.get('latitude', 0), loc2.get('longitude', 0)
                
                distance = self._calculate_distance(lat1, lon1, lat2, lon2)
                distances.append(distance)
                
                # Calculate speed if timestamps available
                if 'timestamp' in loc1 and 'timestamp' in loc2:
                    try:
                        time1 = datetime.fromisoformat(loc1['timestamp'])
                        time2 = datetime.fromisoformat(loc2['timestamp'])
                        time_diff = (time2 - time1).total_seconds()
                        
                        if time_diff > 0:
                            speed = distance / time_diff  # meters per second
                            speeds.append(speed)
                    except:
                        pass
                
                # Calculate direction
                if distance > 0:
                    direction = self._calculate_bearing(lat1, lon1, lat2, lon2)
                    directions.append(direction)
            
            # Calculate statistics
            movement_stats = {
                'total_distance': sum(distances),
                'avg_distance': np.mean(distances) if distances else 0.0,
                'max_distance': max(distances) if distances else 0.0,
                'avg_speed': np.mean(speeds) if speeds else 0.0,
                'max_speed': max(speeds) if speeds else 0.0,
                'speed_variance': np.var(speeds) if len(speeds) > 1 else 0.0,
                'direction_consistency': self._calculate_direction_consistency(directions),
                'movement_regularity': self._calculate_movement_regularity(distances)
            }
            
            # Detect transportation modes
            transportation_modes = self._detect_transportation_modes(speeds, distances)
            
            # Calculate movement entropy
            if distances:
                distance_bins = np.histogram(distances, bins=10)[0]
                movement_entropy = stats.entropy(distance_bins + 1)  # Add 1 for smoothing
                movement_pattern_strength = 1.0 / (1.0 + movement_entropy)
            else:
                movement_pattern_strength = 0.0
            
            return {
                'movement_statistics': movement_stats,
                'transportation_modes': transportation_modes,
                'movement_pattern_strength': float(movement_pattern_strength),
                'is_high_mobility': movement_stats['total_distance'] > 10000,  # 10km
                'is_consistent_traveler': movement_stats['direction_consistency'] > 0.7,
                'preferred_speed_range': self._get_speed_range(movement_stats['avg_speed'])
            }
            
        except Exception as e:
            self.logger.error(f"Error in movement pattern analysis: {e}")
            return {'error': str(e)}
    
    def _cluster_behavior_patterns(self, user_id: str, 
                                  user_data: Dict) -> Dict:
        """Cluster behavior patterns into categories"""
        try:
            # Prepare features for clustering
            features = self._extract_behavior_features(user_data)
            
            if len(features) < self.config['cluster_count']:
                return {'error': 'Insufficient data for clustering'}
            
            # Apply clustering
            kmeans = KMeans(
                n_clusters=min(self.config['cluster_count'], len(features)),
                random_state=42,
                n_init=10
            )
            clusters = kmeans.fit_predict(features)
            
            # Calculate clustering quality
            if len(set(clusters)) > 1:
                silhouette_avg = silhouette_score(features, clusters)
            else:
                silhouette_avg = 0.0
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(kmeans.n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 0:
                    # Calculate cluster characteristics
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    
                    cluster_analysis[cluster_id] = {
                        'size': len(cluster_features),
                        'center': cluster_center.tolist(),
                        'density': len(cluster_features) / len(features),
                        'characteristics': self._interpret_cluster(cluster_center)
                    }
            
            # Identify dominant behavior pattern
            if cluster_analysis:
                dominant_cluster = max(cluster_analysis.items(), 
                                      key=lambda x: x[1]['size'])
            else:
                dominant_cluster = (0, {'size': 0, 'characteristics': {}})
            
            return {
                'total_patterns': len(features),
                'cluster_count': len(cluster_analysis),
                'clustering_quality': float(silhouette_avg),
                'cluster_analysis': cluster_analysis,
                'dominant_pattern': {
                    'cluster_id': dominant_cluster[0],
                    'characteristics': dominant_cluster[1]['characteristics'],
                    'prevalence': dominant_cluster[1]['size'] / len(features)
                },
                'behavior_diversity': len(cluster_analysis) / self.config['cluster_count']
            }
            
        except Exception as e:
            self.logger.error(f"Error in behavior clustering: {e}")
            return {'error': str(e)}
    
    def _calculate_pattern_deviation(self, user_id: str, 
                                    user_data: Dict) -> Dict:
        """Calculate deviation from established patterns"""
        try:
            user_pattern = self.user_patterns[user_id]
            
            if not user_pattern['pattern_signatures']:
                return {'error': 'No established patterns'}
            
            # Compare current behavior with patterns
            current_features = self._extract_current_features(user_data)
            pattern_signature = user_pattern['pattern_signatures']
            
            # Calculate deviations
            deviations = {}
            
            if 'location_pattern' in pattern_signature:
                location_dev = self._calculate_location_deviation(
                    current_features, pattern_signature['location_pattern']
                )
                deviations['location'] = location_dev
            
            if 'time_pattern' in pattern_signature:
                time_dev = self._calculate_time_deviation(
                    current_features, pattern_signature['time_pattern']
                )
                deviations['time'] = time_dev
            
            if 'movement_pattern' in pattern_signature:
                movement_dev = self._calculate_movement_deviation(
                    current_features, pattern_signature['movement_pattern']
                )
                deviations['movement'] = movement_dev
            
            # Calculate overall deviation
            if deviations:
                overall_deviation = np.mean(list(deviations.values()))
            else:
                overall_deviation = 0.0
            
            return {
                'deviations': deviations,
                'overall_deviation': float(overall_deviation),
                'is_significant_deviation': overall_deviation > 0.7,
                'most_deviated_pattern': max(deviations.items(), key=lambda x: x[1])[0] if deviations else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern deviation: {e}")
            return {'error': str(e)}
    
    def _detect_seasonality(self, user_id: str, 
                           user_data: Dict) -> Dict:
        """Detect seasonal and periodic patterns"""
        try:
            locations = user_data.get('locations', [])
            
            if len(locations) < 30:  # Need at least 30 data points
                return {'error': 'Insufficient data for seasonality detection'}
            
            # Extract time series data
            dates = []
            activities = []
            
            for loc in locations:
                if 'timestamp' in loc:
                    try:
                        date = datetime.fromisoformat(loc['timestamp']).date()
                        dates.append(date)
                        activities.append(1)  # Binary activity indicator
                    except:
                        continue
            
            if len(dates) < 30:
                return {'error': 'Insufficient date data'}
            
            # Create daily activity time series
            date_range = max(dates) - min(dates)
            if date_range.days < 7:
                return {'error': 'Time range too short'}
            
            # Group by day of week
            day_of_week_activity = defaultdict(int)
            for date, activity in zip(dates, activities):
                day_of_week_activity[date.weekday()] += activity
            
            # Calculate weekly patterns
            weekly_pattern = {}
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in range(7):
                weekly_pattern[day_names[day]] = day_of_week_activity.get(day, 0)
            
            # Detect weekly seasonality
            weekly_variance = np.var(list(weekly_pattern.values()))
            weekly_seasonality = weekly_variance / (np.mean(list(weekly_pattern.values())) + 1e-10)
            
            # Group by hour of day
            hour_activity = defaultdict(int)
            for loc in locations:
                if 'timestamp' in loc:
                    try:
                        hour = datetime.fromisoformat(loc['timestamp']).hour
                        hour_activity[hour] += 1
                    except:
                        continue
            
            # Detect daily patterns
            daily_pattern = {hour: hour_activity.get(hour, 0) for hour in range(24)}
            daily_variance = np.var(list(daily_pattern.values()))
            daily_seasonality = daily_variance / (np.mean(list(daily_pattern.values())) + 1e-10)
            
            return {
                'weekly_pattern': weekly_pattern,
                'daily_pattern': daily_pattern,
                'weekly_seasonality': float(weekly_seasonality),
                'daily_seasonality': float(daily_seasonality),
                'has_weekly_pattern': weekly_seasonality > 0.5,
                'has_daily_pattern': daily_seasonality > 0.5,
                'peak_activity_day': day_names[max(weekly_pattern.items(), key=lambda x: x[1])[0]] if weekly_pattern else None,
                'peak_activity_hour': max(daily_pattern.items(), key=lambda x: x[1])[0] if daily_pattern else None
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonality: {e}")
            return {'error': str(e)}
    
    def _generate_pattern_signature(self, analyses: Dict) -> Dict:
        """Generate a signature for user patterns"""
        signature = {}
        
        # Location signature
        if 'location_analysis' in analyses and 'error' not in analyses['location_analysis']:
            loc_analysis = analyses['location_analysis']
            signature['location_pattern'] = {
                'primary_locations': loc_analysis.get('primary_locations', []),
                'location_regularity': loc_analysis.get('location_regularity', 0.0),
                'home_location': loc_analysis.get('home_location'),
                'work_location': loc_analysis.get('work_location')
            }
        
        # Time signature
        if 'time_analysis' in analyses and 'error' not in analyses['time_analysis']:
            time_analysis = analyses['time_analysis']
            signature['time_pattern'] = {
                'peak_hours': time_analysis.get('peak_hours', []),
                'routine_strength': time_analysis.get('routine_strength', 0.0),
                'is_night_owl': time_analysis.get('is_night_owl', False),
                'is_early_bird': time_analysis.get('is_early_bird', False)
            }
        
        # Movement signature
        if 'movement_analysis' in analyses and 'error' not in analyses['movement_analysis']:
            movement_analysis = analyses['movement_analysis']
            signature['movement_pattern'] = {
                'avg_speed': movement_analysis.get('movement_statistics', {}).get('avg_speed', 0.0),
                'transportation_modes': movement_analysis.get('transportation_modes', []),
                'movement_pattern_strength': movement_analysis.get('movement_pattern_strength', 0.0)
            }
        
        return signature
    
    def _update_user_patterns(self, user_id: str, analyses: Dict, 
                             pattern_signature: Dict):
        """Update user pattern storage"""
        user_pattern = self.user_patterns[user_id]
        
        # Update patterns
        for analysis_type, analysis_data in analyses.items():
            if 'error' not in analysis_data:
                if analysis_type == 'location_analysis':
                    if 'primary_locations' in analysis_data:
                        for loc in analysis_data['primary_locations']:
                            user_pattern['location_patterns'][loc['cluster_id']].append({
                                'timestamp': datetime.now().isoformat(),
                                'location': loc['center'],
                                'visit_count': loc['visit_count']
                            })
                
                elif analysis_type == 'time_analysis':
                    if 'peak_hours' in analysis_data:
                        for hour_data in analysis_data['daily_routine']:
                            user_pattern['time_patterns'][hour_data['hour']].append({
                                'timestamp': datetime.now().isoformat(),
                                'activity_level': hour_data['activity_level']
                            })
        
        # Update pattern signature
        user_pattern['pattern_signatures'] = pattern_signature
        user_pattern['last_analysis'] = datetime.now().isoformat()
    
    def _calculate_pattern_consistency(self, user_id: str, 
                                      analyses: Dict) -> float:
        """Calculate consistency of user patterns"""
        consistency_scores = []
        
        # Location consistency
        if 'location_analysis' in analyses and 'error' not in analyses['location_analysis']:
            loc_consistency = analyses['location_analysis'].get('location_regularity', 0.0)
            consistency_scores.append(loc_consistency)
        
        # Time consistency
        if 'time_analysis' in analyses and 'error' not in analyses['time_analysis']:
            time_consistency = analyses['time_analysis'].get('routine_strength', 0.0)
            consistency_scores.append(time_consistency)
        
        # Movement consistency
        if 'movement_analysis' in analyses and 'error' not in analyses['movement_analysis']:
            movement_consistency = analyses['movement_analysis'].get('movement_pattern_strength', 0.0)
            consistency_scores.append(movement_consistency)
        
        if consistency_scores:
            return float(np.mean(consistency_scores))
        else:
            return 0.0
    
    def _generate_behavior_summary(self, analyses: Dict) -> str:
        """Generate a summary of user behavior"""
        summaries = []
        
        # Location summary
        if 'location_analysis' in analyses and 'error' not in analyses['location_analysis']:
            loc_data = analyses['location_analysis']
            if loc_data.get('primary_locations'):
                summaries.append(f"Frequent visitor to {len(loc_data['primary_locations'])} main locations")
        
        # Time summary
        if 'time_analysis' in analyses and 'error' not in analyses['time_analysis']:
            time_data = analyses['time_analysis']
            if time_data.get('is_night_owl'):
                summaries.append("Night owl pattern")
            elif time_data.get('is_early_bird'):
                summaries.append("Early bird pattern")
        
        # Movement summary
        if 'movement_analysis' in analyses and 'error' not in analyses['movement_analysis']:
            movement_data = analyses['movement_analysis']
            if movement_data.get('is_high_mobility'):
                summaries.append("High mobility user")
            if movement_data.get('is_consistent_traveler'):
                summaries.append("Consistent travel routes")
        
        if summaries:
            return "; ".join(summaries)
        else:
            return "Pattern analysis in progress"
    
    def _detect_pattern_anomalies(self, analyses: Dict, 
                                 pattern_consistency: float) -> List[str]:
        """Detect anomalies in user patterns"""
        anomalies = []
        
        if pattern_consistency < 0.3:
            anomalies.append("LOW_PATTERN_CONSISTENCY")
        
        # Check for specific anomalies
        if 'location_analysis' in analyses and 'error' not in analyses['location_analysis']:
            loc_data = analyses['location_analysis']
            if loc_data.get('location_diversity', 0) > 20:
                anomalies.append("HIGH_LOCATION_DIVERSITY")
        
        if 'time_analysis' in analyses and 'error' not in analyses['time_analysis']:
            time_data = analyses['time_analysis']
            if time_data.get('routine_strength', 0) < 0.2:
                anomalies.append("WEAK_TIME_ROUTINE")
        
        return anomalies
    
    def _generate_pattern_predictions(self, user_id: str, 
                                     analyses: Dict) -> List[Dict]:
        """Generate predictions based on user patterns"""
        predictions = []
        current_time = datetime.now()
        
        # Predict next location based on time
        if 'time_analysis' in analyses and 'error' not in analyses['time_analysis']:
            time_data = analyses['time_analysis']
            current_hour = current_time.hour
            
            # Find similar time in routine
            for routine in time_data.get('daily_routine', []):
                if abs(routine['hour'] - current_hour) <= 2:
                    predictions.append({
                        'type': 'LOCATION_PREDICTION',
                        'confidence': routine['activity_level'],
                        'prediction': f"Likely at frequent location based on {routine['label']} routine"
                    })
                    break
        
        # Predict movement based on day of week
        if 'seasonality_detection' in analyses and 'error' not in analyses['seasonality_detection']:
            seasonality_data = analyses['seasonality_detection']
            current_day = current_time.weekday()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            if 'weekly_pattern' in seasonality_data:
                day_activity = seasonality_data['weekly_pattern'].get(day_names[current_day], 0)
                if day_activity > np.mean(list(seasonality_data['weekly_pattern'].values())):
                    predictions.append({
                        'type': 'ACTIVITY_PREDICTION',
                        'confidence': day_activity / max(seasonality_data['weekly_pattern'].values()),
                        'prediction': f"Higher than average activity expected for {day_names[current_day]}"
                    })
        
        return predictions
    
    # Helper methods
    def _get_time_label(self, hour: int) -> str:
        """Get label for time of day"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points (simplified)"""
        import math
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth radius in meters
        radius = 6371000
        return radius * c
    
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
    
    def _calculate_direction_consistency(self, directions: List[float]) -> float:
        """Calculate consistency of movement directions"""
        if len(directions) < 2:
            return 0.0
        
        # Convert to unit vectors
        vectors = []
        for direction in directions:
            rad = math.radians(direction)
            vectors.append([math.cos(rad), math.sin(rad)])
        
        vectors = np.array(vectors)
        mean_vector = np.mean(vectors, axis=0)
        
        # Calculate mean resultant length
        mean_length = np.linalg.norm(mean_vector)
        return mean_length
    
    def _calculate_movement_regularity(self, distances: List[float]) -> float:
        """Calculate regularity of movement distances"""
        if len(distances) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist > 0:
            cv = std_dist / mean_dist
            regularity = 1.0 / (1.0 + cv)
        else:
            regularity = 0.0
        
        return regularity
    
    def _detect_transportation_modes(self, speeds: List[float], 
                                    distances: List[float]) -> List[Dict]:
        """Detect transportation modes from speed and distance"""
        if not speeds:
            return []
        
        # Define speed ranges for different modes (m/s)
        mode_ranges = {
            'walking': (0.5, 1.5),
            'running': (1.5, 5.0),
            'cycling': (3.0, 8.0),
            'driving': (5.0, 30.0),
            'public_transport': (8.0, 20.0)
        }
        
        # Count occurrences in each mode
        mode_counts = {mode: 0 for mode in mode_ranges}
        
        for speed in speeds:
            for mode, (min_speed, max_speed) in mode_ranges.items():
                if min_speed <= speed <= max_speed:
                    mode_counts[mode] += 1
                    break
        
        # Calculate percentages
        total = sum(mode_counts.values())
        if total > 0:
            modes = []
            for mode, count in mode_counts.items():
                if count > 0:
                    percentage = count / total
                    modes.append({
                        'mode': mode,
                        'percentage': float(percentage),
                        'count': count
                    })
            
            # Sort by percentage
            modes.sort(key=lambda x: x['percentage'], reverse=True)
            return modes
        
        return []
    
    def _get_speed_range(self, speed: float) -> str:
        """Get descriptive speed range"""
        if speed < 0.5:
            return "stationary"
        elif speed < 1.5:
            return "walking"
        elif speed < 5.0:
            return "running/cycling"
        elif speed < 15.0:
            return "driving"
        else:
            return "fast_moving"
    
    def _extract_behavior_features(self, user_data: Dict) -> np.ndarray:
        """Extract features for behavior clustering"""
        features = []
        
        # Extract features from location data
        locations = user_data.get('locations', [])
        for i in range(0, len(locations), 5):  # Sample every 5th location
            if i < len(locations):
                loc = locations[i]
                features.append([
                    loc.get('latitude', 0),
                    loc.get('longitude', 0),
                    datetime.fromisoformat(loc.get('timestamp', datetime.now().isoformat())).hour,
                    datetime.fromisoformat(loc.get('timestamp', datetime.now().isoformat())).weekday()
                ])
        
        if features:
            return np.array(features)
        else:
            return np.array([[0, 0, 0, 0]])
    
    def _extract_current_features(self, user_data: Dict) -> Dict:
        """Extract current behavior features"""
        current_features = {}
        
        # Extract from recent locations
        locations = user_data.get('locations', [])
        if locations:
            current_loc = locations[-1]
            current_features['latitude'] = current_loc.get('latitude', 0)
            current_features['longitude'] = current_loc.get('longitude', 0)
            current_features['hour'] = datetime.fromisoformat(
                current_loc.get('timestamp', datetime.now().isoformat())
            ).hour
            current_features['weekday'] = datetime.fromisoformat(
                current_loc.get('timestamp', datetime.now().isoformat())
            ).weekday()
        
        return current_features
    
    def _calculate_location_deviation(self, current_features: Dict, 
                                     pattern_signature: Dict) -> float:
        """Calculate deviation from location pattern"""
        if 'home_location' not in pattern_signature:
            return 0.0
        
        # Calculate distance from home location
        home_loc = pattern_signature['home_location']
        if home_loc and 'center' in home_loc:
            home_lat, home_lon = home_loc['center']
            current_lat = current_features.get('latitude', 0)
            current_lon = current_features.get('longitude', 0)
            
            distance = self._calculate_distance(
                home_lat, home_lon, current_lat, current_lon
            )
            
            # Normalize deviation (0-1 scale)
            deviation = min(1.0, distance / 10000)  # 10km max
            
            # Adjust for time of day
            current_hour = current_features.get('hour', 12)
            if 22 <= current_hour <= 6:  # Night time
                deviation *= 1.5  # More suspicious at night
            
            return deviation
        
        return 0.0
    
    def _calculate_time_deviation(self, current_features: Dict, 
                                 pattern_signature: Dict) -> float:
        """Calculate deviation from time pattern"""
        if 'peak_hours' not in pattern_signature:
            return 0.0
        
        current_hour = current_features.get('hour', 12)
        
        # Check if current hour is in peak hours
        peak_hours = [hour for hour, _ in pattern_signature['peak_hours']]
        
        if current_hour in peak_hours:
            return 0.0  # No deviation
        else:
            # Calculate distance to nearest peak hour
            distances = [abs(current_hour - hour) for hour in peak_hours]
            min_distance = min(distances) if distances else 12
            
            # Normalize deviation (0-1 scale)
            deviation = min(1.0, min_distance / 6.0)  # 6 hours max
            
            return deviation
    
    def _calculate_movement_deviation(self, current_features: Dict, 
                                     pattern_signature: Dict) -> float:
        """Calculate deviation from movement pattern"""
        # This is a simplified version
        # In production, you would compare with historical movement patterns
        return 0.0
    
    def _interpret_cluster(self, cluster_center: np.ndarray) -> Dict:
        """Interpret cluster characteristics"""
        # Extract features from cluster center
        # Assuming features: [latitude, longitude, hour, weekday]
        
        lat, lon, hour, weekday = cluster_center
        
        interpretation = {
            'typical_location': f"Location around ({lat:.4f}, {lon:.4f})",
            'typical_time': f"{self._get_time_label(int(hour))} (hour: {int(hour)})",
            'typical_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday'][int(weekday) % 7],
            'behavior_type': self._classify_behavior(hour, weekday)
        }
        
        return interpretation
    
    def _classify_behavior(self, hour: float, weekday: float) -> str:
        """Classify behavior based on time"""
        hour_int = int(hour)
        weekday_int = int(weekday)
        
        if weekday_int < 5:  # Weekday
            if 7 <= hour_int <= 9:
                return "morning_commute"
            elif 17 <= hour_int <= 19:
                return "evening_commute"
            elif 9 <= hour_int <= 17:
                return "daytime_activity"
            else:
                return "off_hours"
        else:  # Weekend
            if 10 <= hour_int <= 18:
                return "weekend_daytime"
            else:
                return "weekend_off_hours"
    
    def get_user_pattern_summary(self, user_id: str) -> Dict:
        """Get summary of user patterns"""
        if user_id not in self.user_patterns:
            return {'error': 'User not found'}
        
        user_pattern = self.user_patterns[user_id]
        
        return {
            'user_id': user_id,
            'last_analysis': user_pattern['last_analysis'],
            'pattern_signature': user_pattern['pattern_signatures'],
            'location_pattern_count': len(user_pattern['location_patterns']),
            'time_pattern_count': len(user_pattern['time_patterns']),
            'movement_pattern_count': len(user_pattern['movement_patterns']),
            'has_behavior_clusters': user_pattern['behavior_clusters'] is not None
        }
    
    def export_pattern_data(self, user_id: str) -> Dict:
        """Export pattern data for a user"""
        if user_id not in self.user_patterns:
            return {}
        
        user_pattern = self.user_patterns[user_id]
        
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'pattern_signature': user_pattern['pattern_signatures'],
            'location_patterns': dict(user_pattern['location_patterns']),
            'time_patterns': dict(user_pattern['time_patterns']),
            'movement_patterns': dict(user_pattern['movement_patterns']),
            'last_analysis': user_pattern['last_analysis']
        }