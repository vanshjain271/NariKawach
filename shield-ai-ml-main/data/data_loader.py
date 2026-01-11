import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import requests
import json
from datetime import datetime, timedelta
import os
from loguru import logger
from ..utils.logger import setup_logger


class DataLoader:
    """Load and manage safety-related datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.logger = setup_logger(__name__)
        
        # Dataset URLs and sources
        self.dataset_sources = {
            "crime_data": {
                "url": "https://data.gov.in/dataset/crime-statistics",
                "local_path": f"{data_dir}/raw/crime_data.csv",
                "description": "Official crime statistics from data.gov.in"
            },
            "safety_reports": {
                "url": "https://safecity.in/api/reports",
                "local_path": f"{data_dir}/raw/safety_reports.json",
                "description": "Community-reported safety incidents"
            },
            "street_lighting": {
                "url": "https://openstreetmap.org/data/street-lights",
                "local_path": f"{data_dir}/raw/street_lighting.geojson",
                "description": "Street lighting coverage data"
            },
            "population_density": {
                "url": "https://censusindia.gov.in/data",
                "local_path": f"{data_dir}/raw/population_density.csv",
                "description": "Population density by area"
            },
            "public_transport": {
                "url": "https://transitfeeds.com/data",
                "local_path": f"{data_dir}/raw/transport_stops.csv",
                "description": "Public transport stops and routes"
            }
        }
    
    def load_crime_data(self, city: Optional[str] = None) -> pd.DataFrame:
        """Load crime data with filtering options"""
        try:
            filepath = self.dataset_sources["crime_data"]["local_path"]
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                self.logger.info(f"Loaded crime data from {filepath}")
            else:
                # Load sample data if file doesn't exist
                df = self._load_sample_crime_data()
                self.logger.warning(f"Using sample crime data for {city}")
            
            # Filter by city if specified
            if city:
                df = df[df['city'].str.lower() == city.lower()]
            
            # Add additional features
            df = self._enhance_crime_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading crime data: {e}")
            return pd.DataFrame()
    
    def load_safety_reports(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[Dict]:
        """Load community safety reports"""
        try:
            filepath = self.dataset_sources["safety_reports"]["local_path"]
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    reports = json.load(f)
            else:
                reports = self._load_sample_reports()
            
            # Filter by date range
            if start_date and end_date:
                filtered = []
                for report in reports:
                    report_date = datetime.fromisoformat(report.get('timestamp', ''))
                    if start_date <= report_date <= end_date:
                        filtered.append(report)
                reports = filtered
            
            return reports
            
        except Exception as e:
            self.logger.error(f"Error loading safety reports: {e}")
            return []
    
    def load_environmental_data(self, latitude: float, longitude: float, 
                               radius_km: float = 5.0) -> Dict:
        """Load environmental data for a location"""
        try:
            # This would integrate with external APIs
            environmental_data = {
                "lighting_coverage": self._get_lighting_coverage(latitude, longitude),
                "crowd_density": self._estimate_crowd_density(latitude, longitude),
                "police_stations": self._get_nearby_police_stations(latitude, longitude, radius_km),
                "hospitals": self._get_nearby_hospitals(latitude, longitude, radius_km),
                "safe_zones": self._get_safe_zones(latitude, longitude, radius_km),
                "crime_density": self._calculate_crime_density(latitude, longitude, radius_km)
            }
            
            return environmental_data
            
        except Exception as e:
            self.logger.error(f"Error loading environmental data: {e}")
            return {}
    
    def _load_sample_crime_data(self) -> pd.DataFrame:
        """Generate sample crime data for development"""
        cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
        crime_types = ['theft', 'harassment', 'assault', 'robbery', 'stalking']
        
        data = []
        np.random.seed(42)
        
        for _ in range(1000):
            city = np.random.choice(cities)
            lat = np.random.uniform(12.0, 30.0)
            lon = np.random.uniform(70.0, 90.0)
            crime_type = np.random.choice(crime_types)
            hour = np.random.randint(0, 24)
            
            data.append({
                'city': city,
                'latitude': lat,
                'longitude': lon,
                'crime_type': crime_type,
                'hour_of_day': hour,
                'day_of_week': np.random.randint(0, 7),
                'month': np.random.randint(1, 13),
                'year': 2023,
                'severity': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2])
            })
        
        return pd.DataFrame(data)
    
    def _load_sample_reports(self) -> List[Dict]:
        """Generate sample safety reports"""
        reports = []
        
        for i in range(100):
            reports.append({
                'id': f"report_{i}",
                'latitude': np.random.uniform(12.0, 30.0),
                'longitude': np.random.uniform(70.0, 90.0),
                'incident_type': np.random.choice(['harassment', 'stalking', 'theft', 'other']),
                'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
                'severity': np.random.choice(['low', 'medium', 'high']),
                'description': f"Safety incident report #{i}",
                'verified': np.random.random() > 0.5
            })
        
        return reports
    
    def _enhance_crime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features to crime data"""
        if df.empty:
            return df
        
        # Add temporal features
        df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if 20 <= x <= 6 else 0)
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Add severity scores
        severity_map = {'low': 1, 'medium': 2, 'high': 3}
        df['severity_score'] = df['severity'].map(severity_map)
        
        # Add spatial clustering
        df['location_cluster'] = self._create_spatial_clusters(df)
        
        return df
    
    def _create_spatial_clusters(self, df: pd.DataFrame, n_clusters: int = 10) -> pd.Series:
        """Create spatial clusters for crime hotspots"""
        from sklearn.cluster import KMeans
        
        if len(df) < n_clusters:
            return pd.Series([0] * len(df))
        
        coords = df[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        
        return pd.Series(clusters, index=df.index)
    
    def _get_lighting_coverage(self, lat: float, lon: float) -> float:
        """Estimate lighting coverage (0-1 scale)"""
        # In production, this would query OSM or municipal data
        # For now, return a simulated value
        np.random.seed(int(lat * 100 + lon))
        return np.random.uniform(0.3, 0.9)
    
    def _estimate_crowd_density(self, lat: float, lon: float) -> float:
        """Estimate crowd density (0-1 scale)"""
        # This would integrate with Google Maps or similar APIs
        # For now, use time-based simulation
        hour = datetime.now().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return np.random.uniform(0.7, 0.9)  # Rush hours
        elif 22 <= hour <= 6:
            return np.random.uniform(0.1, 0.3)  # Late night
        else:
            return np.random.uniform(0.4, 0.6)  # Normal hours
    
    def _get_nearby_police_stations(self, lat: float, lon: float, radius_km: float) -> List[Dict]:
        """Get nearby police stations"""
        # In production, this would query a database or API
        stations = []
        
        for i in range(np.random.randint(1, 5)):
            stations.append({
                'name': f"Police Station {i+1}",
                'latitude': lat + np.random.uniform(-0.01, 0.01),
                'longitude': lon + np.random.uniform(-0.01, 0.01),
                'distance_km': np.random.uniform(0.5, radius_km),
                'response_time_min': np.random.uniform(5, 15)
            })
        
        return sorted(stations, key=lambda x: x['distance_km'])
    
    def _get_nearby_hospitals(self, lat: float, lon: float, radius_km: float) -> List[Dict]:
        """Get nearby hospitals"""
        hospitals = []
        
        for i in range(np.random.randint(1, 3)):
            hospitals.append({
                'name': f"Hospital {i+1}",
                'latitude': lat + np.random.uniform(-0.02, 0.02),
                'longitude': lon + np.random.uniform(-0.02, 0.02),
                'distance_km': np.random.uniform(1, radius_km),
                'emergency_services': True
            })
        
        return hospitals
    
    def _get_safe_zones(self, lat: float, lon: float, radius_km: float) -> List[Dict]:
        """Get safe zones (shopping malls, restaurants, etc.)"""
        safe_zones = []
        zone_types = ['shopping_mall', 'restaurant', 'hotel', 'police_station', 'hospital']
        
        for i in range(np.random.randint(2, 6)):
            zone_type = np.random.choice(zone_types)
            safe_zones.append({
                'name': f"{zone_type.replace('_', ' ').title()} {i+1}",
                'type': zone_type,
                'latitude': lat + np.random.uniform(-0.015, 0.015),
                'longitude': lon + np.random.uniform(-0.015, 0.015),
                'distance_km': np.random.uniform(0.2, radius_km),
                'safety_score': np.random.uniform(0.7, 1.0)
            })
        
        return sorted(safe_zones, key=lambda x: x['distance_km'])
    
    def _calculate_crime_density(self, lat: float, lon: float, radius_km: float) -> float:
        """Calculate crime density in the area"""
        # In production, this would query the crime database
        # For now, return a simulated value
        np.random.seed(int(lat * 1000 + lon * 100))
        
        # Higher crime density at night in certain areas
        hour = datetime.now().hour
        base_density = np.random.uniform(0.1, 0.5)
        
        if 22 <= hour <= 6:  # Night time
            base_density *= np.random.uniform(1.5, 3.0)
        
        return min(1.0, base_density)