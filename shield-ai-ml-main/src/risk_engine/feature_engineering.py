import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import holidays
from geopy.distance import geodesic
from scipy import stats
import math
from loguru import logger
from ...utils.logger import setup_logger


class SafetyFeatureEngineer:
    """
    Advanced feature engineering for safety prediction
    Creates meaningful features from raw data
    """
    
    def __init__(self, country: str = 'IN'):
        self.country = country
        self.indian_holidays = holidays.India(years=range(2020, 2026))
        self.logger = setup_logger(__name__)
        
        # Safe zones database (would be loaded from external source)
        self.safe_zones = self._initialize_safe_zones()
        
        # Crime data cache
        self.crime_data_cache = {}
        
        # Feature statistics for normalization
        self.feature_stats = {}
    
    def engineer_features(self, raw_data: Dict) -> Dict:
        """
        Transform raw data into engineered features
        """
        try:
            features = {}
            
            # Extract base data
            timestamp = pd.to_datetime(raw_data.get('timestamp', datetime.now()))
            latitude = raw_data.get('latitude', 0.0)
            longitude = raw_data.get('longitude', 0.0)
            
            # 1. Temporal Features
            features.update(self._engineer_temporal_features(timestamp))
            
            # 2. Spatial Features
            features.update(self._engineer_spatial_features(latitude, longitude, timestamp))
            
            # 3. Environmental Features
            features.update(self._engineer_environmental_features(
                latitude, longitude, timestamp, raw_data
            ))
            
            # 4. Behavioral Features
            features.update(self._engineer_behavioral_features(raw_data))
            
            # 5. Device & Network Features
            features.update(self._engineer_device_features(raw_data))
            
            # 6. Social & Contextual Features
            features.update(self._engineer_social_features(raw_data))
            
            # 7. Derived Features
            features.update(self._engineer_derived_features(features))
            
            # Normalize features
            normalized_features = self._normalize_features(features)
            
            self.logger.debug(f"Engineered {len(features)} features")
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return {}
    
    def _engineer_temporal_features(self, timestamp: datetime) -> Dict:
        """Engineer temporal features"""
        features = {}
        
        # Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * timestamp.day / 31)
        features['day_cos'] = np.cos(2 * np.pi * timestamp.day / 31)
        features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
        
        # Categorical time features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['day_of_month'] = timestamp.day
        features['month'] = timestamp.month
        features['year'] = timestamp.year
        
        # Time-based indicators
        features['is_night'] = 1 if 22 <= timestamp.hour <= 6 else 0
        features['is_dusk_dawn'] = 1 if 18 <= timestamp.hour <= 20 or 5 <= timestamp.hour <= 7 else 0
        features['is_rush_hour'] = 1 if (7 <= timestamp.hour <= 9) or (17 <= timestamp.hour <= 19) else 0
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_holiday'] = 1 if timestamp.date() in self.indian_holidays else 0
        
        # Season features
        month = timestamp.month
        if month in [12, 1, 2]:
            season = 0  # Winter
        elif month in [3, 4, 5]:
            season = 1  # Spring
        elif month in [6, 7, 8]:
            season = 2  # Summer
        else:
            season = 3  # Fall
        features['season'] = season
        
        return features
    
    def _engineer_spatial_features(self, latitude: float, longitude: float, 
                                  timestamp: datetime) -> Dict:
        """Engineer spatial features"""
        features = {}
        
        # Basic coordinates
        features['latitude'] = latitude
        features['longitude'] = longitude
        
        # Location type (simplified - would use GIS data in production)
        location_type = self._classify_location_type(latitude, longitude)
        features['location_type'] = location_type
        
        # Urban/rural classification
        features['is_urban'] = 1 if location_type in ['residential', 'commercial', 'industrial'] else 0
        
        # Proximity features
        features['crime_density'] = self._get_crime_density(latitude, longitude, 1.0)
        features['safe_zone_distance'] = self._distance_to_safe_zone(latitude, longitude)
        features['police_station_distance'] = self._distance_to_police(latitude, longitude)
        features['hospital_distance'] = self._distance_to_hospital(latitude, longitude)
        features['public_transport_distance'] = self._distance_to_transport(latitude, longitude)
        
        # Area characteristics
        features['population_density'] = self._get_population_density(latitude, longitude)
        features['land_use_mix'] = self._calculate_land_use_mix(latitude, longitude)
        
        # Historical risk
        features['historical_incidents'] = self._get_historical_incidents(
            latitude, longitude, 0.5
        )
        
        return features
    
    def _engineer_environmental_features(self, latitude: float, longitude: float,
                                        timestamp: datetime, raw_data: Dict) -> Dict:
        """Engineer environmental features"""
        features = {}
        
        # Lighting conditions
        features['lighting_score'] = self._calculate_lighting_score(
            latitude, longitude, timestamp
        )
        
        # Crowd density (from external data or estimation)
        features['crowd_density'] = raw_data.get('crowd_density', 
                                                self._estimate_crowd_density(
                                                    latitude, longitude, timestamp
                                                ))
        
        # Weather conditions
        weather_data = raw_data.get('weather', {})
        features['temperature'] = weather_data.get('temperature', 25.0)
        features['humidity'] = weather_data.get('humidity', 50.0)
        features['precipitation'] = weather_data.get('precipitation', 0.0)
        features['wind_speed'] = weather_data.get('wind_speed', 0.0)
        features['visibility'] = weather_data.get('visibility', 10.0)
        
        # Weather risk score
        features['weather_risk_score'] = self._calculate_weather_risk(weather_data)
        
        # Noise level (estimated)
        features['noise_level'] = self._estimate_noise_level(
            latitude, longitude, timestamp, features['crowd_density']
        )
        
        # Air quality
        features['air_quality_index'] = raw_data.get('air_quality', 50.0)
        
        return features
    
    def _engineer_behavioral_features(self, raw_data: Dict) -> Dict:
        """Engineer behavioral features"""
        features = {}
        
        # Movement patterns
        features['speed'] = raw_data.get('speed', 0.0)
        features['acceleration'] = raw_data.get('acceleration', 0.0)
        features['bearing'] = raw_data.get('bearing', 0.0)
        
        # Route information
        features['route_deviation_score'] = raw_data.get('route_deviation_score', 0.0)
        features['route_familiarity'] = raw_data.get('route_familiarity', 0.5)
        
        # Stop patterns
        features['stop_duration'] = raw_data.get('stop_duration', 0.0)
        features['stop_frequency'] = raw_data.get('stop_frequency', 0.0)
        
        # User behavior
        features['user_confidence_score'] = raw_data.get('user_confidence', 0.5)
        features['app_usage_frequency'] = raw_data.get('app_usage_frequency', 0.5)
        features['response_time'] = raw_data.get('response_time', 5.0)
        
        # Previous incidents
        features['previous_alerts_count'] = raw_data.get('previous_alerts', 0)
        features['recent_incidents'] = raw_data.get('recent_incidents', 0)
        
        return features
    
    def _engineer_device_features(self, raw_data: Dict) -> Dict:
        """Engineer device and network features"""
        features = {}
        
        # Device status
        features['battery_level'] = raw_data.get('battery_level', 100) / 100.0
        features['battery_health'] = raw_data.get('battery_health', 0.8)
        
        # Network information
        features['network_type'] = self._encode_network_type(
            raw_data.get('network_type', 'unknown')
        )
        features['network_strength'] = raw_data.get('network_strength', 0.8)
        features['data_connection'] = 1 if raw_data.get('data_connected', False) else 0
        
        # GPS accuracy
        features['gps_accuracy'] = 1.0 / (1.0 + raw_data.get('accuracy', 50.0))
        features['gps_satellites'] = raw_data.get('satellites', 0)
        
        # Device capabilities
        features['has_flashlight'] = 1 if raw_data.get('has_flashlight', False) else 0
        features['has_speaker'] = 1 if raw_data.get('has_speaker', False) else 0
        
        return features
    
    def _engineer_social_features(self, raw_data: Dict) -> Dict:
        """Engineer social and contextual features"""
        features = {}
        
        # Guardian network
        features['guardian_count'] = raw_data.get('guardian_count', 0)
        features['guardian_online_count'] = raw_data.get('guardian_online_count', 0)
        features['guardian_response_rate'] = raw_data.get('guardian_response_rate', 0.7)
        
        # Social activity
        features['social_checkin_density'] = raw_data.get('social_checkin_density', 0.0)
        features['nearby_users_count'] = raw_data.get('nearby_users', 0)
        
        # Community factors
        features['community_safety_score'] = raw_data.get('community_safety_score', 0.5)
        features['crime_reporting_rate'] = raw_data.get('crime_reporting_rate', 0.5)
        
        # Emergency services
        features['emergency_response_time'] = raw_data.get('emergency_response_time', 10.0)
        features['police_presence'] = raw_data.get('police_presence', 0.0)
        
        # Event detection
        features['nearby_event'] = 1 if raw_data.get('nearby_event', False) else 0
        features['event_size'] = raw_data.get('event_size', 0)
        
        return features
    
    def _engineer_derived_features(self, features: Dict) -> Dict:
        """Engineer derived features from existing features"""
        derived = {}
        
        # Risk combinations
        if 'is_night' in features and 'crowd_density' in features:
            derived['night_isolation_risk'] = features['is_night'] * (1 - features['crowd_density'])
        
        if 'route_deviation_score' in features and 'is_night' in features:
            derived['night_deviation_risk'] = features['route_deviation_score'] * features['is_night']
        
        if 'battery_level' in features and 'safe_zone_distance' in features:
            derived['battery_distance_risk'] = (1 - features['battery_level']) * \
                                              min(1.0, features['safe_zone_distance'] / 10.0)
        
        # Interaction features
        if 'crime_density' in features and 'lighting_score' in features:
            derived['crime_lighting_risk'] = features['crime_density'] * (1 - features['lighting_score'])
        
        if 'speed' in features and 'route_deviation_score' in features:
            derived['speed_deviation_risk'] = min(1.0, features['speed'] / 20.0) * \
                                             features['route_deviation_score']
        
        # Time-location interactions
        if 'hour_of_day' in features and 'location_type' in features:
            hour = features['hour_of_day']
            location_type = features['location_type']
            
            # High risk combinations
            high_risk_combinations = [
                (22 <= hour <= 6, 'park'),
                (22 <= hour <= 6, 'industrial_area'),
                (0 <= hour <= 4, 'commercial')
            ]
            
            derived['time_location_risk'] = 0.0
            for time_cond, loc_type in high_risk_combinations:
                if time_cond and location_type == loc_type:
                    derived['time_location_risk'] = 0.8
                    break
        
        return derived
    
    def _normalize_features(self, features: Dict) -> Dict:
        """Normalize features to consistent ranges"""
        normalized = {}
        
        for feature_name, value in features.items():
            # Skip categorical features
            if feature_name in ['location_type', 'network_type', 'season']:
                normalized[feature_name] = value
                continue
            
            # Apply appropriate normalization
            if isinstance(value, (int, float)):
                # Check if feature needs special handling
                if feature_name.endswith('_sin') or feature_name.endswith('_cos'):
                    # Cyclical features already normalized
                    normalized[feature_name] = float(value)
                elif feature_name in ['is_night', 'is_weekend', 'is_holiday', 
                                     'is_urban', 'data_connection', 'has_flashlight', 
                                     'has_speaker', 'nearby_event']:
                    # Binary features
                    normalized[feature_name] = int(value)
                else:
                    # Scale to 0-1 range based on known ranges
                    normalized[feature_name] = self._scale_feature(feature_name, value)
            else:
                normalized[feature_name] = value
        
        return normalized
    
    def _scale_feature(self, feature_name: str, value: float) -> float:
        """Scale feature to 0-1 range"""
        # Define scaling ranges for different features
        scaling_ranges = {
            'hour_of_day': (0, 23),
            'day_of_week': (0, 6),
            'day_of_month': (1, 31),
            'month': (1, 12),
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'crime_density': (0, 1),
            'safe_zone_distance': (0, 20),  # km
            'police_station_distance': (0, 10),  # km
            'hospital_distance': (0, 20),  # km
            'public_transport_distance': (0, 5),  # km
            'population_density': (0, 20000),  # people per sq km
            'historical_incidents': (0, 100),
            'lighting_score': (0, 1),
            'crowd_density': (0, 1),
            'temperature': (-10, 50),  # Celsius
            'humidity': (0, 100),
            'precipitation': (0, 100),  # mm
            'wind_speed': (0, 100),  # km/h
            'visibility': (0, 20),  # km
            'weather_risk_score': (0, 1),
            'noise_level': (0, 1),
            'air_quality_index': (0, 500),
            'speed': (0, 40),  # m/s
            'acceleration': (-5, 5),  # m/s²
            'bearing': (0, 360),
            'route_deviation_score': (0, 1),
            'route_familiarity': (0, 1),
            'stop_duration': (0, 3600),  # seconds
            'stop_frequency': (0, 100),  # per hour
            'user_confidence_score': (0, 1),
            'app_usage_frequency': (0, 100),  # per day
            'response_time': (0, 60),  # seconds
            'previous_alerts_count': (0, 100),
            'recent_incidents': (0, 50),
            'battery_level': (0, 1),
            'battery_health': (0, 1),
            'network_strength': (0, 1),
            'gps_accuracy': (0, 1),
            'gps_satellites': (0, 30),
            'guardian_count': (0, 20),
            'guardian_online_count': (0, 20),
            'guardian_response_rate': (0, 1),
            'social_checkin_density': (0, 1),
            'nearby_users_count': (0, 1000),
            'community_safety_score': (0, 1),
            'crime_reporting_rate': (0, 1),
            'emergency_response_time': (0, 60),  # minutes
            'police_presence': (0, 1),
            'event_size': (0, 100000),
            'night_isolation_risk': (0, 1),
            'night_deviation_risk': (0, 1),
            'battery_distance_risk': (0, 1),
            'crime_lighting_risk': (0, 1),
            'speed_deviation_risk': (0, 1),
            'time_location_risk': (0, 1)
        }
        
        if feature_name in scaling_ranges:
            min_val, max_val = scaling_ranges[feature_name]
            if max_val > min_val:
                scaled = (value - min_val) / (max_val - min_val)
                return float(max(0.0, min(1.0, scaled)))
        
        # Default scaling for unknown features
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0
    
    # Helper methods for feature engineering
    def _classify_location_type(self, latitude: float, longitude: float) -> str:
        """Classify location type (simplified)"""
        # In production, this would use GIS data or external APIs
        # For now, use a simple heuristic based on coordinates
        
        # Simulate different location types
        location_types = ['residential', 'commercial', 'industrial', 
                         'park', 'transportation', 'educational', 'healthcare']
        
        # Use coordinates to determine (pseudo-random but deterministic)
        seed = int(abs(latitude * 1000 + longitude * 1000)) % 100
        if seed < 40:
            return 'residential'
        elif seed < 60:
            return 'commercial'
        elif seed < 70:
            return 'industrial'
        elif seed < 80:
            return 'park'
        elif seed < 90:
            return 'transportation'
        elif seed < 95:
            return 'educational'
        else:
            return 'healthcare'
    
    def _get_crime_density(self, latitude: float, longitude: float, 
                          radius_km: float) -> float:
        """Get crime density in area"""
        # In production, this would query a crime database
        # For simulation, generate based on location and time
        
        # Base crime density
        seed = int(abs(latitude * 1000 + longitude * 1000)) % 100
        base_density = seed / 100.0
        
        # Adjust for urban areas
        location_type = self._classify_location_type(latitude, longitude)
        if location_type in ['commercial', 'industrial']:
            base_density *= 1.5
        elif location_type == 'residential':
            base_density *= 0.7
        
        return min(1.0, base_density)
    
    def _distance_to_safe_zone(self, latitude: float, longitude: float) -> float:
        """Distance to nearest safe zone in km"""
        if not self.safe_zones:
            return 10.0  # Default
        
        min_distance = float('inf')
        for zone in self.safe_zones:
            distance = geodesic(
                (latitude, longitude), 
                (zone['latitude'], zone['longitude'])
            ).km
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else 10.0
    
    def _distance_to_police(self, latitude: float, longitude: float) -> float:
        """Distance to nearest police station"""
        # Simulated police stations
        police_stations = [
            {'latitude': latitude + 0.01, 'longitude': longitude + 0.01},
            {'latitude': latitude - 0.02, 'longitude': longitude + 0.005},
            {'latitude': latitude + 0.015, 'longitude': longitude - 0.01}
        ]
        
        min_distance = float('inf')
        for station in police_stations:
            distance = geodesic(
                (latitude, longitude), 
                (station['latitude'], station['longitude'])
            ).km
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else 5.0
    
    def _distance_to_hospital(self, latitude: float, longitude: float) -> float:
        """Distance to nearest hospital"""
        # Simulated hospitals
        hospitals = [
            {'latitude': latitude + 0.02, 'longitude': longitude + 0.015},
            {'latitude': latitude - 0.01, 'longitude': longitude + 0.02}
        ]
        
        min_distance = float('inf')
        for hospital in hospitals:
            distance = geodesic(
                (latitude, longitude), 
                (hospital['latitude'], hospital['longitude'])
            ).km
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else 8.0
    
    def _distance_to_transport(self, latitude: float, longitude: float) -> float:
        """Distance to public transport"""
        # Simulated transport stops
        transport_stops = [
            {'latitude': latitude + 0.005, 'longitude': longitude + 0.003},
            {'latitude': latitude - 0.004, 'longitude': longitude + 0.006}
        ]
        
        min_distance = float('inf')
        for stop in transport_stops:
            distance = geodesic(
                (latitude, longitude), 
                (stop['latitude'], stop['longitude'])
            ).km
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else 1.0
    
    def _get_population_density(self, latitude: float, longitude: float) -> float:
        """Get population density"""
        # Simulated population density
        seed = int(abs(latitude * 1000 + longitude * 1000)) % 100
        base_density = 5000 + (seed * 150)  # 5000-20000 per sq km
        
        location_type = self._classify_location_type(latitude, longitude)
        if location_type == 'residential':
            base_density *= 1.5
        elif location_type == 'commercial':
            base_density *= 2.0
        
        return float(base_density)
    
    def _calculate_land_use_mix(self, latitude: float, longitude: float) -> float:
        """Calculate land use mix diversity"""
        # Simulated land use mix (0-1, higher = more mixed)
        seed = int(abs(latitude * 1000 + longitude * 1000)) % 100
        return seed / 100.0
    
    def _get_historical_incidents(self, latitude: float, longitude: float,
                                 radius_km: float) -> int:
        """Get historical incidents in area"""
        # Simulated incident count
        crime_density = self._get_crime_density(latitude, longitude, radius_km)
        return int(crime_density * 50)  # 0-50 incidents
    
    def _calculate_lighting_score(self, latitude: float, longitude: float,
                                 timestamp: datetime) -> float:
        """Calculate lighting conditions score"""
        hour = timestamp.hour
        
        # Base lighting on time of day
        if 6 <= hour <= 18:
            base_score = 0.9  # Daylight
        elif 18 < hour <= 20 or 5 <= hour < 6:
            base_score = 0.6 
        elif 18 < hour <= 20 or 5 <= hour < 6:
            base_score = 0.6  # Twilight
        else:
            base_score = 0.3  # Night
        
        # Adjust for location type
        location_type = self._classify_location_type(latitude, longitude)
        if location_type in ['commercial', 'transportation']:
            base_score = min(1.0, base_score + 0.3)  # Better lighting
        elif location_type == 'park':
            base_score = max(0.1, base_score - 0.2)  # Poorer lighting
        
        # Urban areas have better lighting
        population_density = self._get_population_density(latitude, longitude)
        if population_density > 10000:
            base_score = min(1.0, base_score + 0.2)
        
        return float(base_score)
    
    def _estimate_crowd_density(self, latitude: float, longitude: float,
                               timestamp: datetime) -> float:
        """Estimate crowd density"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Base crowd density
        if weekday < 5:  # Weekday
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_density = 0.8  # Rush hour
            elif 12 <= hour <= 14:
                base_density = 0.7  # Lunch time
            elif 9 <= hour <= 17:
                base_density = 0.6  # Business hours
            else:
                base_density = 0.3  # Off hours
        else:  # Weekend
            if 11 <= hour <= 20:
                base_density = 0.7  # Daytime weekend
            else:
                base_density = 0.4  # Evening/night weekend
        
        # Adjust for location type
        location_type = self._classify_location_type(latitude, longitude)
        if location_type in ['commercial', 'transportation']:
            base_density = min(1.0, base_density + 0.2)
        elif location_type == 'residential':
            base_density = max(0.1, base_density - 0.1)
        elif location_type == 'park':
            # Parks busy during day, empty at night
            if 8 <= hour <= 20:
                base_density = min(1.0, base_density + 0.1)
            else:
                base_density = max(0.1, base_density - 0.3)
        
        return float(base_density)
    
    def _calculate_weather_risk(self, weather_data: Dict) -> float:
        """Calculate weather-related risk"""
        risk = 0.0
        
        # Precipitation risk
        precipitation = weather_data.get('precipitation', 0.0)
        if precipitation > 20:  # Heavy rain
            risk += 0.4
        elif precipitation > 5:  # Moderate rain
            risk += 0.2
        
        # Visibility risk
        visibility = weather_data.get('visibility', 10.0)
        if visibility < 1:  # Very poor visibility
            risk += 0.3
        elif visibility < 5:  # Poor visibility
            risk += 0.15
        
        # Wind risk
        wind_speed = weather_data.get('wind_speed', 0.0)
        if wind_speed > 50:  # Strong wind
            risk += 0.2
        elif wind_speed > 30:  # Moderate wind
            risk += 0.1
        
        # Temperature extremes
        temperature = weather_data.get('temperature', 25.0)
        if temperature < 0 or temperature > 35:
            risk += 0.1
        
        return min(1.0, risk)
    
    def _estimate_noise_level(self, latitude: float, longitude: float,
                            timestamp: datetime, crowd_density: float) -> float:
        """Estimate noise level"""
        hour = timestamp.hour
        
        # Base noise level
        if 7 <= hour <= 22:
            base_noise = 0.7  # Daytime
        else:
            base_noise = 0.3  # Nighttime
        
        # Adjust for crowd density
        base_noise = base_noise * 0.7 + crowd_density * 0.3
        
        # Adjust for location type
        location_type = self._classify_location_type(latitude, longitude)
        if location_type in ['commercial', 'transportation', 'industrial']:
            base_noise = min(1.0, base_noise + 0.2)
        elif location_type == 'park':
            base_noise = max(0.1, base_noise - 0.2)
        
        return float(base_noise)
    
    def _encode_network_type(self, network_type: str) -> int:
        """Encode network type as numeric"""
        encoding = {
            'wifi': 0,
            'cellular_5g': 1,
            'cellular_4g': 2,
            'cellular_3g': 3,
            'cellular_2g': 4,
            'unknown': 5
        }
        return encoding.get(network_type.lower(), 5)
    
    def _initialize_safe_zones(self) -> List[Dict]:
        """Initialize safe zones database"""
        # In production, this would load from a database or API
        safe_zones = [
            {
                'name': 'Police Station Central',
                'latitude': 28.6139,
                'longitude': 77.2090,
                'type': 'police_station',
                'safety_score': 0.95
            },
            {
                'name': 'City Hospital',
                'latitude': 28.6280,
                'longitude': 77.2021,
                'type': 'hospital',
                'safety_score': 0.90
            },
            {
                'name': 'Main Shopping Mall',
                'latitude': 28.5276,
                'longitude': 77.2101,
                'type': 'shopping_mall',
                'safety_score': 0.85
            },
            {
                'name': 'Community Center',
                'latitude': 28.5400,
                'longitude': 77.1900,
                'type': 'community_center',
                'safety_score': 0.80
            },
            {
                'name': 'University Campus Security',
                'latitude': 28.5500,
                'longitude': 77.1800,
                'type': 'educational',
                'safety_score': 0.88
            }
        ]
        return safe_zones
    
    def get_feature_statistics(self) -> Dict:
        """Get statistics about engineered features"""
        return {
            'feature_count': len(self.feature_stats),
            'feature_categories': {
                'temporal': 15,
                'spatial': 12,
                'environmental': 10,
                'behavioral': 10,
                'device': 8,
                'social': 10,
                'derived': 6
            },
            'last_updated': datetime.now().isoformat()
        }