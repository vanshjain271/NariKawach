import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from loguru import logger
from ..utils.logger import setup_logger


class DatasetManager:
    """Manage datasets for the safety system"""
    
    def __init__(self, db_path: str = "data/safety_data.db"):
        self.db_path = db_path
        self.logger = setup_logger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create user profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    normal_routes TEXT,
                    usual_times TEXT,
                    frequent_locations TEXT,
                    risk_preferences TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            
            # Create location history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    accuracy REAL,
                    speed REAL,
                    timestamp TIMESTAMP,
                    battery_level INTEGER,
                    network_status TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # Create incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    incident_type TEXT,
                    risk_level TEXT,
                    location TEXT,
                    timestamp TIMESTAMP,
                    response_time REAL,
                    outcome TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # Create model training data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features TEXT,
                    target INTEGER,
                    split_type TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_history_user_timestamp ON location_history(user_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_user_timestamp ON incidents(user_id, timestamp)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def save_user_location(self, user_id: str, location_data: Dict):
        """Save user location to history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO location_history 
                (user_id, latitude, longitude, accuracy, speed, timestamp, battery_level, network_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                location_data.get('latitude'),
                location_data.get('longitude'),
                location_data.get('accuracy', 0.0),
                location_data.get('speed', 0.0),
                location_data.get('timestamp', datetime.now().isoformat()),
                location_data.get('battery_level', 100),
                location_data.get('network_status', 'good')
            ))
            
            conn.commit()
            conn.close()
            
            # Keep only last 1000 locations per user
            self._cleanup_old_locations(user_id)
            
            self.logger.debug(f"Saved location for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving user location: {e}")
    
    def get_user_location_history(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Get user location history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT latitude, longitude, accuracy, speed, timestamp, battery_level, network_status
                FROM location_history
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (user_id, cutoff_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'latitude': row[0],
                    'longitude': row[1],
                    'accuracy': row[2],
                    'speed': row[3],
                    'timestamp': row[4],
                    'battery_level': row[5],
                    'network_status': row[6]
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting location history: {e}")
            return []
    
    def save_incident(self, user_id: str, incident_data: Dict):
        """Save incident record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO incidents 
                (user_id, incident_type, risk_level, location, timestamp, response_time, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                incident_data.get('incident_type'),
                incident_data.get('risk_level'),
                json.dumps(incident_data.get('location', {})),
                incident_data.get('timestamp', datetime.now().isoformat()),
                incident_data.get('response_time', 0.0),
                incident_data.get('outcome', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved incident for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving incident: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT normal_routes, usual_times, frequent_locations, risk_preferences, created_at, updated_at
                FROM user_profiles
                WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'user_id': user_id,
                    'normal_routes': json.loads(row[0]) if row[0] else [],
                    'usual_times': json.loads(row[1]) if row[1] else {},
                    'frequent_locations': json.loads(row[2]) if row[2] else [],
                    'risk_preferences': json.loads(row[3]) if row[3] else {},
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    def update_user_profile(self, user_id: str, profile_data: Dict):
        """Update user profile"""
        try:
            existing = self.get_user_profile(user_id)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if existing:
                # Update existing profile
                cursor.execute('''
                    UPDATE user_profiles 
                    SET normal_routes = ?, usual_times = ?, frequent_locations = ?, 
                        risk_preferences = ?, updated_at = ?
                    WHERE user_id = ?
                ''', (
                    json.dumps(profile_data.get('normal_routes', [])),
                    json.dumps(profile_data.get('usual_times', {})),
                    json.dumps(profile_data.get('frequent_locations', [])),
                    json.dumps(profile_data.get('risk_preferences', {})),
                    datetime.now().isoformat(),
                    user_id
                ))
            else:
                # Create new profile
                cursor.execute('''
                    INSERT INTO user_profiles 
                    (user_id, normal_routes, usual_times, frequent_locations, risk_preferences, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    json.dumps(profile_data.get('normal_routes', [])),
                    json.dumps(profile_data.get('usual_times', {})),
                    json.dumps(profile_data.get('frequent_locations', [])),
                    json.dumps(profile_data.get('risk_preferences', {})),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated profile for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
    
    def save_training_data(self, features: List, target: int, split_type: str = "train"):
        """Save training data for model retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_data (features, target, split_type, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                json.dumps(features),
                target,
                split_type,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
    
    def get_training_data(self, limit: int = 10000) -> Tuple[List, List]:
        """Get training data for model retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT features, target FROM training_data
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            X = []
            y = []
            
            for row in rows:
                X.append(json.loads(row[0]))
                y.append(row[1])
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return [], []
    
    def calculate_user_metrics(self, user_id: str) -> Dict:
        """Calculate metrics for a user"""
        try:
            # Get recent locations
            locations = self.get_user_location_history(user_id, hours=168)  # 7 days
            
            if not locations:
                return {}
            
            # Calculate various metrics
            metrics = {
                'total_locations': len(locations),
                'avg_accuracy': np.mean([loc.get('accuracy', 0) for loc in locations]),
                'avg_speed': np.mean([loc.get('speed', 0) for loc in locations]),
                'time_coverage': self._calculate_time_coverage(locations),
                'location_variety': self._calculate_location_variety(locations),
                'battery_pattern': self._analyze_battery_pattern(locations),
                'network_reliability': self._calculate_network_reliability(locations)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating user metrics: {e}")
            return {}
    
    def _cleanup_old_locations(self, user_id: str, keep_count: int = 1000):
        """Clean up old location records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get count of records
            cursor.execute('SELECT COUNT(*) FROM location_history WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            
            if count > keep_count:
                # Delete oldest records
                cursor.execute('''
                    DELETE FROM location_history 
                    WHERE id IN (
                        SELECT id FROM location_history 
                        WHERE user_id = ? 
                        ORDER BY timestamp ASC 
                        LIMIT ?
                    )
                ''', (user_id, count - keep_count))
                
                conn.commit()
                self.logger.debug(f"Cleaned up {count - keep_count} old locations for user {user_id}")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old locations: {e}")
    
    def _calculate_time_coverage(self, locations: List[Dict]) -> Dict:
        """Calculate when user is typically active"""
        hours = [datetime.fromisoformat(loc['timestamp']).hour for loc in locations if 'timestamp' in loc]
        
        if not hours:
            return {}
        
        hour_counts = {hour: hours.count(hour) for hour in set(hours)}
        total = len(hours)
        
        return {
            'peak_hours': sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'coverage_percentage': len(set(hours)) / 24,
            'most_active': max(hour_counts, key=hour_counts.get) if hour_counts else None
        }
    
    def _calculate_location_variety(self, locations: List[Dict]) -> float:
        """Calculate variety of locations visited"""
        if not locations:
            return 0.0
        
        # Group locations by rounded coordinates
        location_groups = set()
        for loc in locations:
            lat_rounded = round(loc.get('latitude', 0), 3)
            lon_rounded = round(loc.get('longitude', 0), 3)
            location_groups.add((lat_rounded, lon_rounded))
        
        return len(location_groups) / len(locations)
    
    def _analyze_battery_pattern(self, locations: List[Dict]) -> Dict:
        """Analyze battery usage patterns"""
        battery_levels = [loc.get('battery_level', 100) for loc in locations]
        
        if not battery_levels:
            return {}
        
        return {
            'avg_battery': np.mean(battery_levels),
            'min_battery': min(battery_levels),
            'max_battery': max(battery_levels),
            'low_battery_count': sum(1 for level in battery_levels if level < 20)
        }
    
    def _calculate_network_reliability(self, locations: List[Dict]) -> float:
        """Calculate network reliability"""
        if not locations:
            return 0.0
        
        good_count = sum(1 for loc in locations if loc.get('network_status') == 'good')
        return good_count / len(locations)