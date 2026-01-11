import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import MODEL_CONFIG


class AdvancedRiskPredictor:
    """
    Ensemble-based risk predictor with multiple ML/DL models
    Combines traditional ML with deep learning for robustness
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODEL_CONFIG
        self.logger = setup_logger(__name__)
        
        # Ensemble models
        self.models = {}
        self.ensemble = None
        self.feature_names = []
        
        # Model metadata
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'last_trained': None,
            'training_stats': {},
            'feature_importance': {}
        }
        
        # Prediction cache
        self.prediction_cache = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble models"""
        # Random Forest
        rf_config = self.config.get('random_forest', {})
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 15),
            min_samples_split=rf_config.get('min_samples_split', 5),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost
        xgb_config = self.config.get('xgboost', {})
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 150),
            max_depth=xgb_config.get('max_depth', 8),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None):
        """Train all models in ensemble"""
        try:
            self.logger.info(f"Training ensemble on {len(X_train)} samples")
            
            # Store feature names if available
            if hasattr(X_train, 'columns'):
                self.feature_names = list(X_train.columns)
            
            # Train individual models
            model_performance = {}
            
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                
                try:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Evaluate on validation set if available
                    if X_val is not None and y_val is not None:
                        y_pred = model.predict(X_val)
                        performance = self._evaluate_model(y_val, y_pred)
                        model_performance[name] = performance
                        
                        self.logger.info(f"{name} performance: {performance}")
                    
                    # Store feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        if self.feature_names:
                            importance = dict(zip(self.feature_names, model.feature_importances_))
                        else:
                            importance = {f'feature_{i}': imp 
                                         for i, imp in enumerate(model.feature_importances_)}
                        self.model_metadata['feature_importance'][name] = importance
                
                except Exception as e:
                    self.logger.error(f"Error training {name}: {e}")
                    continue
            
            # Create voting ensemble
            self._create_voting_ensemble()
            
            # Update metadata
            self.model_metadata['last_trained'] = datetime.now().isoformat()
            self.model_metadata['training_stats'] = {
                'training_samples': len(X_train),
                'feature_count': X_train.shape[1],
                'class_distribution': {
                    'class_0': int(np.sum(y_train == 0)),
                    'class_1': int(np.sum(y_train == 1))
                },
                'model_performance': model_performance
            }
            
            self.logger.info("Ensemble training complete!")
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            raise
    
    def _create_voting_ensemble(self):
        """Create voting ensemble from trained models"""
        try:
            # Create list of (name, model) tuples for voting
            estimators = [(name, model) for name, model in self.models.items()]
            
            # Create soft voting classifier
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Use probability voting
                weights=self._calculate_model_weights(),
                n_jobs=-1
            )
            
            # Note: The ensemble needs to be fit, but models are already trained
            # In practice, we might refit or use pre-trained models
            
            self.logger.info("Voting ensemble created")
            
        except Exception as e:
            self.logger.error(f"Error creating voting ensemble: {e}")
    
    def _calculate_model_weights(self) -> List[float]:
        """Calculate weights for ensemble voting"""
        # Default weights
        weights = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'neural_network': 0.15,
            'gradient_boosting': 0.15
        }
        
        # Adjust weights based on model performance if available
        if 'model_performance' in self.model_metadata['training_stats']:
            performance = self.model_metadata['training_stats']['model_performance']
            
            for name, perf in performance.items():
                if name in weights and 'f1_score' in perf:
                    # Increase weight for better performing models
                    weights[name] = weights[name] * (1 + perf['f1_score'])
            
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        return list(weights.values())
    
    def predict_risk(self, features: Dict) -> Dict:
        """
        Enhanced risk prediction with feature engineering
        """
        try:
            # Convert features to array
            feature_array = self._prepare_features(features)
            
            # Get predictions from all models
            predictions = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(feature_array)[0]
                        predictions[name] = {
                            'risk_level': int(np.argmax(pred_proba)),
                            'confidence': float(np.max(pred_proba)),
                            'probabilities': pred_proba.tolist(),
                            'class_0_prob': float(pred_proba[0]),
                            'class_1_prob': float(pred_proba[1])
                        }
                    else:
                        pred = model.predict(feature_array)[0]
                        predictions[name] = {
                            'risk_level': int(pred),
                            'confidence': 0.5,  # Default confidence
                            'probabilities': [0.5, 0.5]
                        }
                
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
                    predictions[name] = {
                        'risk_level': 0,
                        'confidence': 0.0,
                        'probabilities': [1.0, 0.0],
                        'error': str(e)
                    }
            
            # Get ensemble prediction if available
            ensemble_prediction = None
            if self.ensemble is not None and hasattr(self.ensemble, 'predict_proba'):
                try:
                    ensemble_proba = self.ensemble.predict_proba(feature_array)[0]
                    ensemble_prediction = {
                        'risk_level': int(np.argmax(ensemble_proba)),
                        'confidence': float(np.max(ensemble_proba)),
                        'probabilities': ensemble_proba.tolist()
                    }
                except Exception as e:
                    self.logger.error(f"Error with ensemble prediction: {e}")
            
            # Calculate weighted ensemble voting
            weighted_prediction = self._weighted_voting(predictions)
            
            # Generate risk explanation
            risk_explanation = self._explain_risk(features, predictions)
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(predictions)
            
            result = {
                'ensemble_prediction': weighted_prediction,
                'individual_predictions': predictions,
                'voting_ensemble_prediction': ensemble_prediction,
                'risk_explanation': risk_explanation,
                'prediction_confidence': confidence,
                'feature_analysis': self._analyze_features(features),
                'model_metadata': {
                    'models_used': list(predictions.keys()),
                    'timestamp': datetime.now().isoformat(),
                    'ensemble_version': '2.0.0'
                }
            }
            
            # Cache prediction
            prediction_key = hash(str(features))
            self.prediction_cache[prediction_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Clean old cache entries
            self._clean_prediction_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in risk prediction: {e}")
            return {
                'error': str(e),
                'ensemble_prediction': {
                    'risk_level': 0,
                    'confidence': 0.0,
                    'risk_label': 'ERROR'
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Prepare features for prediction"""
        # Convert dict to array in consistent order
        if self.feature_names:
            # Use known feature order
            feature_array = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_array.append(features[feature_name])
                else:
                    # Handle missing features
                    feature_array.append(0.0)
            return np.array([feature_array])
        else:
            # Use all features in dict order
            return np.array([list(features.values())])
    
    def _weighted_voting(self, predictions: Dict) -> Dict:
        """Weighted ensemble decision"""
        # Calculate weighted average
        risk_scores = []
        confidences = []
        weights = []
        
        for name, pred in predictions.items():
            if 'error' not in pred:
                # Get model weight
                weight = self._get_model_weight(name)
                
                risk_scores.append(pred['risk_level'] * weight)
                confidences.append(pred['confidence'] * weight)
                weights.append(weight)
        
        if weights:
            total_weight = sum(weights)
            final_risk = 1 if sum(risk_scores) / total_weight > 0.5 else 0
            avg_confidence = sum(confidences) / total_weight
        else:
            final_risk = 0
            avg_confidence = 0.0
        
        return {
            'risk_level': final_risk,
            'confidence': avg_confidence,
            'risk_label': self._map_risk_label(final_risk, avg_confidence),
            'voting_method': 'weighted_average'
        }
    
    def _get_model_weight(self, model_name: str) -> float:
        """Get weight for a specific model"""
        weight_map = {
            'random_forest': 0.25,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'neural_network': 0.15,
            'gradient_boosting': 0.15
        }
        return weight_map.get(model_name, 0.1)
    
    def _map_risk_label(self, risk_level: int, confidence: float) -> str:
        """Map numeric risk to human-readable labels"""
        if risk_level == 0:
            if confidence > 0.8:
                return "SAFE"
            elif confidence > 0.6:
                return "LOW_RISK"
            else:
                return "UNKNOWN"
        else:
            if confidence > 0.8:
                return "HIGH_RISK"
            elif confidence > 0.6:
                return "MEDIUM_RISK"
            else:
                return "ELEVATED_RISK"
    
    def _explain_risk(self, features: Dict, predictions: Dict) -> Dict:
        """Explainable AI - Risk factor breakdown"""
        explanations = {
            'primary_factors': [],
            'secondary_factors': [],
            'model_agreement': self._calculate_model_agreement(predictions),
            'feature_contributions': self._calculate_feature_contributions(features)
        }
        
        # Analyze high-impact features
        high_impact_threshold = 0.7
        
        # Temporal factors
        if features.get('is_night', 0) == 1:
            explanations['primary_factors'].append({
                'factor': 'Night Time',
                'impact': 'high',
                'description': 'Increased risk during night hours'
            })
        
        if features.get('is_weekend', 0) == 1:
            explanations['secondary_factors'].append({
                'factor': 'Weekend',
                'impact': 'medium',
                'description': 'Different risk patterns on weekends'
            })
        
        # Spatial factors
        crime_density = features.get('crime_density', 0)
        if crime_density > high_impact_threshold:
            explanations['primary_factors'].append({
                'factor': 'High Crime Area',
                'impact': 'high',
                'score': crime_density,
                'description': f'Area with {crime_density*100:.0f}% crime density'
            })
        
        safe_zone_distance = features.get('safe_zone_distance', 10)
        if safe_zone_distance > 5:
            explanations['secondary_factors'].append({
                'factor': 'Distance from Safe Zone',
                'impact': 'medium',
                'distance_km': safe_zone_distance,
                'description': f'{safe_zone_distance:.1f} km from nearest safe zone'
            })
        
        # Environmental factors
        lighting_score = features.get('lighting_score', 0.5)
        if lighting_score < 0.3:
            explanations['primary_factors'].append({
                'factor': 'Poor Lighting',
                'impact': 'high',
                'score': lighting_score,
                'description': 'Low visibility conditions'
            })
        
        crowd_density = features.get('crowd_density', 0.5)
        if crowd_density < 0.2:
            explanations['secondary_factors'].append({
                'factor': 'Low Crowd Density',
                'impact': 'medium',
                'score': crowd_density,
                'description': 'Isolated area with few people'
            })
        
        # Behavioral factors
        route_deviation = features.get('route_deviation_score', 0)
        if route_deviation > 0.6:
            explanations['primary_factors'].append({
                'factor': 'Route Deviation',
                'impact': 'high',
                'score': route_deviation,
                'description': 'Unusual route detected'
            })
        
        # Device factors
        battery_level = features.get('battery_level', 1.0)
        if battery_level < 0.2:
            explanations['secondary_factors'].append({
                'factor': 'Low Battery',
                'impact': 'medium',
                'level': f'{battery_level*100:.0f}%',
                'description': 'Device battery critically low'
            })
        
        return explanations
    
    def _calculate_model_agreement(self, predictions: Dict) -> Dict:
        """Calculate agreement between models"""
        if not predictions:
            return {'agreement_score': 0.0, 'consensus': 'no_models'}
        
        # Count risk level predictions
        risk_counts = {'0': 0, '1': 0}
        confidences = []
        
        for pred in predictions.values():
            if 'error' not in pred:
                risk_counts[str(pred['risk_level'])] += 1
                confidences.append(pred.get('confidence', 0.0))
        
        total_models = sum(risk_counts.values())
        if total_models == 0:
            return {'agreement_score': 0.0, 'consensus': 'no_valid_models'}
        
        # Calculate agreement
        max_count = max(risk_counts.values())
        agreement_score = max_count / total_models
        
        # Determine consensus
        if agreement_score > 0.8:
            consensus = 'strong'
        elif agreement_score > 0.6:
            consensus = 'moderate'
        else:
            consensus = 'weak'
        
        return {
            'agreement_score': agreement_score,
            'consensus': consensus,
            'risk_votes': risk_counts,
            'avg_confidence': np.mean(confidences) if confidences else 0.0
        }
    
    def _calculate_feature_contributions(self, features: Dict) -> List[Dict]:
        """Calculate feature contributions to risk"""
        contributions = []
        
        # This is a simplified version
        # In production, use SHAP or LIME for proper feature attribution
        
        # High-impact features
        high_impact_features = [
            ('crime_density', 'Crime Density', 0.4),
            ('is_night', 'Night Time', 0.3),
            ('lighting_score', 'Lighting Conditions', 0.3),
            ('route_deviation_score', 'Route Deviation', 0.25),
            ('crowd_density', 'Crowd Density', 0.2)
        ]
        
        for feature_name, display_name, max_impact in high_impact_features:
            if feature_name in features:
                value = features[feature_name]
                
                # Calculate contribution (simplified)
                if feature_name == 'lighting_score':
                    contribution = (1 - value) * max_impact
                elif feature_name == 'crowd_density':
                    contribution = (1 - value) * max_impact
                else:
                    contribution = value * max_impact
                
                contributions.append({
                    'feature': display_name,
                    'value': value,
                    'contribution': contribution,
                    'impact': 'high' if contribution > 0.2 else 'medium'
                })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        return contributions[:5]  # Return top 5 contributors
    
    def _analyze_features(self, features: Dict) -> Dict:
        """Analyze feature values and patterns"""
        analysis = {
            'feature_count': len(features),
            'missing_values': self._count_missing_features(features),
            'risk_indicators': self._identify_risk_indicators(features),
            'data_quality': self._assess_data_quality(features)
        }
        
        # Feature statistics
        numeric_features = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features.append((key, value))
        
        if numeric_features:
            values = [v for _, v in numeric_features]
            analysis['feature_statistics'] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return analysis
    
    def _count_missing_features(self, features: Dict) -> int:
        """Count missing or default-valued features"""
        missing_count = 0
        
        # Check for default/zero values in important features
        important_features = [
            'crime_density', 'lighting_score', 'crowd_density',
            'route_deviation_score', 'battery_level'
        ]
        
        for feature in important_features:
            if features.get(feature, 0) == 0:
                missing_count += 1
        
        return missing_count
    
    def _identify_risk_indicators(self, features: Dict) -> List[str]:
        """Identify potential risk indicators in features"""
        indicators = []
        
        # Check various risk conditions
        if features.get('is_night', 0) == 1:
            indicators.append('night_time')
        
        if features.get('crime_density', 0) > 0.7:
            indicators.append('high_crime_area')
        
        if features.get('lighting_score', 0.5) < 0.3:
            indicators.append('poor_lighting')
        
        if features.get('crowd_density', 0.5) < 0.2:
            indicators.append('low_crowd_density')
        
        if features.get('route_deviation_score', 0) > 0.6:
            indicators.append('route_deviation')
        
        if features.get('battery_level', 1.0) < 0.2:
            indicators.append('low_battery')
        
        return indicators
    
    def _assess_data_quality(self, features: Dict) -> str:
        """Assess overall data quality"""
        # Count valid features
        valid_count = 0
        total_count = len(features)
        
        for value in features.values():
            if isinstance(value, (int, float)):
                if value != 0:  # Assuming 0 might indicate missing data
                    valid_count += 1
            elif value:  # Non-empty string or other non-zero values
                valid_count += 1
        
        quality_ratio = valid_count / total_count if total_count > 0 else 0
        
        if quality_ratio > 0.9:
            return 'excellent'
        elif quality_ratio > 0.7:
            return 'good'
        elif quality_ratio > 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_prediction_confidence(self, predictions: Dict) -> float:
        """Calculate overall prediction confidence"""
        if not predictions:
            return 0.0
        
        confidences = []
        weights = []
        
        for name, pred in predictions.items():
            if 'error' not in pred:
                confidence = pred.get('confidence', 0.0)
                weight = self._get_model_weight(name)
                
                confidences.append(confidence)
                weights.append(weight)
        
        if weights and confidences:
            # Weighted average confidence
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _clean_prediction_cache(self):
        """Clean old entries from prediction cache"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        keys_to_remove = []
        for key, entry in self.prediction_cache.items():
            if entry['timestamp'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.prediction_cache[key]
        
        if keys_to_remove:
            self.logger.debug(f"Cleaned {len(keys_to_remove)} old cache entries")
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate model performance"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate specificity (true negative rate)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'error': str(e)
            }
    
    def save_models(self, directory: str = "storage/models"):
        """Save trained models and metadata"""
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            
            # Save individual models
            for name, model in self.models.items():
                filename = f"{directory}/{name}_model.pkl"
                joblib.dump(model, filename)
                self.logger.info(f"Saved {name} model to {filename}")
            
            # Save ensemble if available
            if self.ensemble is not None:
                ensemble_file = f"{directory}/voting_ensemble.pkl"
                joblib.dump(self.ensemble, ensemble_file)
                self.logger.info(f"Saved voting ensemble to {ensemble_file}")
            
            # Save metadata
            metadata_file = f"{directory}/model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            self.logger.info(f"Saved model metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, directory: str = "storage/models"):
        """Load trained models and metadata"""
        try:
            # Load individual models
            for name in self.models.keys():
                filename = f"{directory}/{name}_model.pkl"
                if os.path.exists(filename):
                    self.models[name] = joblib.load(filename)
                    self.logger.info(f"Loaded {name} model from {filename}")
            
            # Load ensemble if available
            ensemble_file = f"{directory}/voting_ensemble.pkl"
            if os.path.exists(ensemble_file):
                self.ensemble = joblib.load(ensemble_file)
                self.logger.info(f"Loaded voting ensemble from {ensemble_file}")
            
            # Load metadata
            metadata_file = f"{directory}/model_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                self.logger.info(f"Loaded model metadata from {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the trained models"""
        info = {
            'ensemble_type': 'weighted_voting',
            'model_count': len(self.models),
            'models': list(self.models.keys()),
            'feature_count': len(self.feature_names) if self.feature_names else 'unknown',
            'last_trained': self.model_metadata.get('last_trained'),
            'training_samples': self.model_metadata.get('training_stats', {}).get('training_samples', 0)
        }
        
        # Add performance metrics if available
        if 'model_performance' in self.model_metadata.get('training_stats', {}):
            performance = self.model_metadata['training_stats']['model_performance']
            info['model_performance'] = {
                name: {k: v for k, v in perf.items() if k != 'error'}
                for name, perf in performance.items()
            }
        
        return info
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """Predict risk for multiple feature sets"""
        results = []
        
        for features in features_list:
            result = self.predict_risk(features)
            results.append(result)
        
        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(results)
        
        return {
            'predictions': results,
            'batch_statistics': batch_stats,
            'total_predictions': len(results)
        }
    
    def _calculate_batch_statistics(self, predictions: List[Dict]) -> Dict:
        """Calculate statistics for batch predictions"""
        if not predictions:
            return {}
        
        risk_levels = []
        confidences = []
        
        for pred in predictions:
            ensemble_pred = pred.get('ensemble_prediction', {})
            risk_levels.append(ensemble_pred.get('risk_level', 0))
            confidences.append(ensemble_pred.get('confidence', 0.0))
        
        risk_levels = np.array(risk_levels)
        confidences = np.array(confidences)
        
        return {
            'risk_distribution': {
                'safe_count': int(np.sum(risk_levels == 0)),
                'risk_count': int(np.sum(risk_levels == 1)),
                'safe_percentage': float(np.mean(risk_levels == 0)),
                'risk_percentage': float(np.mean(risk_levels == 1))
            },
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'high_risk_cases': int(np.sum((risk_levels == 1) & (confidences > 0.7)))
        }