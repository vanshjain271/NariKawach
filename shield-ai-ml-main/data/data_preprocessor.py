import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os
from loguru import logger
from ..utils.logger import setup_logger


class DataPreprocessor:
    """Preprocess safety data for ML models"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = setup_logger(__name__)
        
        # Preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Feature columns
        self.numerical_features = [
            'latitude', 'longitude', 'hour_of_day', 'day_of_week',
            'month', 'severity_score', 'is_night', 'is_weekend'
        ]
        
        self.categorical_features = [
            'crime_type', 'city', 'location_cluster'
        ]
    
    def preprocess_crime_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Preprocess crime data for model training"""
        try:
            if df.empty:
                raise ValueError("Empty dataframe provided")
            
            self.logger.info(f"Preprocessing {len(df)} crime records")
            
            # Step 1: Handle missing values
            df_clean = self._handle_missing_values(df)
            
            # Step 2: Encode categorical features
            df_encoded = self._encode_categorical_features(df_clean)
            
            # Step 3: Scale numerical features
            df_scaled = self._scale_numerical_features(df_encoded)
            
            # Step 4: Create target variable
            # For now, create a binary target based on severity
            y = self._create_target_variable(df_clean)
            
            # Step 5: Extract features
            X = self._extract_features(df_scaled)
            
            # Step 6: Save preprocessing artifacts
            self._save_preprocessing_artifacts()
            
            self.logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, {
                'feature_names': self._get_feature_names(),
                'preprocessing_info': self._get_preprocessing_info()
            }
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise
    
    def preprocess_realtime_data(self, data: Dict) -> np.ndarray:
        """Preprocess real-time data for prediction"""
        try:
            # Convert dict to dataframe
            df = pd.DataFrame([data])
            
            # Apply the same transformations as training data
            df_clean = self._handle_missing_values(df)
            df_encoded = self._encode_categorical_features(df_clean, is_training=False)
            df_scaled = self._scale_numerical_features(df_encoded, is_training=False)
            
            X = self._extract_features(df_scaled)
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error preprocessing realtime data: {e}")
            return np.array([])
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        df_clean = df.copy()
        
        # Numerical features
        for col in self.numerical_features:
            if col in df_clean.columns:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_clean[col] = self.imputers[col].fit_transform(df_clean[[col]])
                else:
                    df_clean[col] = self.imputers[col].transform(df_clean[[col]])
        
        # Categorical features
        for col in self.categorical_features:
            if col in df_clean.columns:
                # Fill with mode for categorical
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
        
        return df_clean
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col])
                else:
                    # For new data, transform with existing encoder
                    if col in self.encoders:
                        # Handle unseen categories
                        unseen_mask = ~df_encoded[col].isin(self.encoders[col].classes_)
                        if unseen_mask.any():
                            # Replace unseen categories with a default value
                            df_encoded.loc[unseen_mask, col] = 'unknown'
                        
                        # Transform
                        df_encoded[col] = self.encoders[col].transform(df_encoded[col])
                    else:
                        # If encoder doesn't exist, use simple ordinal encoding
                        df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        for col in self.numerical_features:
            if col in df_scaled.columns:
                if is_training:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]]).flatten()
                else:
                    if col in self.scalers:
                        df_scaled[col] = self.scalers[col].transform(df_scaled[[col]]).flatten()
                    else:
                        # If scaler doesn't exist, use min-max scaling
                        df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
        
        return df_scaled
    
    def _create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable for training"""
        # Create binary classification: high risk vs not high risk
        # In production, this would be more sophisticated
        if 'severity' in df.columns:
            y = (df['severity'] == 'high').astype(int).values
        elif 'severity_score' in df.columns:
            y = (df['severity_score'] >= 2).astype(int).values
        else:
            # If no severity, create synthetic target based on time and location
            y = self._create_synthetic_target(df)
        
        return y
    
    def _create_synthetic_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic target for development"""
        # Simulate risk based on time and location
        # Higher risk at night in certain areas
        
        # Create a hash-based random but reproducible target
        y = []
        for idx, row in df.iterrows():
            seed = int(row.get('latitude', 0) * 1000 + 
                      row.get('longitude', 0) * 100 + 
                      row.get('hour_of_day', 0))
            np.random.seed(seed)
            
            # Higher probability at night
            if 20 <= row.get('hour_of_day', 12) <= 6:
                risk_prob = 0.4
            else:
                risk_prob = 0.1
            
            y.append(1 if np.random.random() < risk_prob else 0)
        
        return np.array(y)
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for model training"""
        # Combine all features
        feature_cols = self.numerical_features + self.categorical_features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError("No features available for extraction")
        
        return df[available_cols].values
    
    def _get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return self.numerical_features + self.categorical_features
    
    def _get_preprocessing_info(self) -> Dict:
        """Get information about preprocessing"""
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'imputers': list(self.imputers.keys())
        }
    
    def _save_preprocessing_artifacts(self, save_dir: str = "storage/models"):
        """Save preprocessing artifacts"""
        os.makedirs(save_dir, exist_ok=True)
        
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'config': {
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features
            }
        }
        
        joblib.dump(artifacts, f"{save_dir}/preprocessing_artifacts.pkl")
        self.logger.info(f"Saved preprocessing artifacts to {save_dir}/preprocessing_artifacts.pkl")
    
    def load_preprocessing_artifacts(self, load_dir: str = "storage/models"):
        """Load preprocessing artifacts"""
        try:
            filepath = f"{load_dir}/preprocessing_artifacts.pkl"
            if os.path.exists(filepath):
                artifacts = joblib.load(filepath)
                self.scalers = artifacts.get('scalers', {})
                self.encoders = artifacts.get('encoders', {})
                self.imputers = artifacts.get('imputers', {})
                config = artifacts.get('config', {})
                self.numerical_features = config.get('numerical_features', self.numerical_features)
                self.categorical_features = config.get('categorical_features', self.categorical_features)
                self.logger.info(f"Loaded preprocessing artifacts from {filepath}")
            else:
                self.logger.warning(f"Preprocessing artifacts not found at {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading preprocessing artifacts: {e}")