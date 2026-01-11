import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from loguru import logger
from ..utils.logger import setup_logger
from ..config.constants import MODEL_CONFIG


class ModelTrainer:
    """Train and manage ML models for safety system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODEL_CONFIG
        self.logger = setup_logger(__name__)
        
        # Models
        self.models = {}
        self.model_metrics = {}
        
        # Training history
        self.training_history = []
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, 
                            model_type: str = "risk_prediction") -> Dict:
        """Train ensemble model for risk prediction"""
        try:
            self.logger.info(f"Training {model_type} ensemble model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train individual models
            trained_models = {}
            
            if model_type == "risk_prediction":
                trained_models = self._train_risk_models(X_train, y_train)
            elif model_type == "anomaly_detection":
                trained_models = self._train_anomaly_models(X_train)
            elif model_type == "stalking_detection":
                trained_models = self._train_stalking_models(X_train, y_train)
            
            # Evaluate models
            metrics = self._evaluate_models(trained_models, X_test, y_test)
            
            # Select best model
            best_model = self._select_best_model(trained_models, metrics)
            
            # Save models
            self._save_models(trained_models, model_type)
            
            # Update training history
            self._update_training_history(model_type, metrics)
            
            self.logger.info(f"Training complete for {model_type}")
            
            return {
                'models': trained_models,
                'metrics': metrics,
                'best_model': best_model,
                'training_info': {
                    'dataset_size': len(X),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'feature_count': X.shape[1],
                    'positive_ratio': np.mean(y)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {e}")
            raise
    
    def _train_risk_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train risk prediction models"""
        models = {}
        
        # Random Forest
        rf_config = self.config.get("random_forest", {})
        rf_model = RandomForestClassifier(**rf_config)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # XGBoost
        xgb_config = self.config.get("xgboost", {})
        xgb_model = xgb.XGBClassifier(**xgb_config)
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # Neural Network
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        models['neural_network'] = nn_model
        
        return models
    
    def _train_anomaly_models(self, X_train: np.ndarray) -> Dict:
        """Train anomaly detection models"""
        models = {}
        
        # Isolation Forest
        iso_config = self.config.get("isolation_forest", {})
        iso_model = IsolationForest(**iso_config)
        iso_model.fit(X_train)
        models['isolation_forest'] = iso_model
        
        # One-Class SVM
        from sklearn.svm import OneClassSVM
        svm_model = OneClassSVM(
            nu=0.1,
            kernel="rbf",
            gamma="scale"
        )
        svm_model.fit(X_train)
        models['one_class_svm'] = svm_model
        
        # Local Outlier Factor
        from sklearn.neighbors import LocalOutlierFactor
        lof_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        lof_model.fit(X_train)
        models['local_outlier_factor'] = lof_model
        
        return models
    
    def _train_stalking_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train stalking detection models"""
        models = {}
        
        # Stalking detection is a specialized task
        # We'll use ensemble of classifiers
        
        # Random Forest for stalking
        rf_stalking = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        rf_stalking.fit(X_train, y_train)
        models['stalking_rf'] = rf_stalking
        
        # Gradient Boosting for stalking
        gb_stalking = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        gb_stalking.fit(X_train, y_train)
        models['stalking_gb'] = gb_stalking
        
        return models
    
    def _evaluate_models(self, models: Dict, X_test: np.ndarray, 
                        y_test: Optional[np.ndarray] = None) -> Dict:
        """Evaluate trained models"""
        metrics = {}
        
        for name, model in models.items():
            try:
                model_metrics = {}
                
                if hasattr(model, 'predict_proba'):
                    # Classification models
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    model_metrics['accuracy'] = np.mean(y_pred == y_test)
                    
                    if y_pred_proba is not None:
                        model_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                    
                    # Classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    model_metrics['precision'] = report['weighted avg']['precision']
                    model_metrics['recall'] = report['weighted avg']['recall']
                    model_metrics['f1_score'] = report['weighted avg']['f1-score']
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    model_metrics['confusion_matrix'] = cm.tolist()
                    
                elif hasattr(model, 'decision_function'):
                    # Anomaly detection models
                    scores = model.decision_function(X_test)
                    model_metrics['anomaly_scores'] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
                
                metrics[name] = model_metrics
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {name}: {e}")
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def _select_best_model(self, models: Dict, metrics: Dict) -> Dict:
        """Select the best model based on evaluation metrics"""
        best_score = -1
        best_model_name = None
        best_model = None
        
        for name, model_metrics in metrics.items():
            if 'roc_auc' in model_metrics:
                score = model_metrics['roc_auc']
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = models[name]
            elif 'f1_score' in model_metrics:
                score = model_metrics['f1_score']
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = models[name]
        
        if best_model_name:
            return {
                'name': best_model_name,
                'model': best_model,
                'score': best_score,
                'metrics': metrics[best_model_name]
            }
        else:
            # Return first model if no clear best
            first_name = list(models.keys())[0]
            return {
                'name': first_name,
                'model': models[first_name],
                'score': 0.0,
                'metrics': metrics.get(first_name, {})
            }
    
    def _save_models(self, models: Dict, model_type: str):
        """Save trained models"""
        save_dir = f"storage/models/{model_type}"
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in models.items():
            filename = f"{save_dir}/{name}_{timestamp}.pkl"
            joblib.dump(model, filename)
            self.logger.info(f"Saved model {name} to {filename}")
        
        # Save model metadata
        metadata = {
            'model_type': model_type,
            'timestamp': timestamp,
            'models_trained': list(models.keys()),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_file = f"{save_dir}/metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_training_history(self, model_type: str, metrics: Dict):
        """Update training history"""
        history_entry = {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'best_model': self._select_best_model(self.models, metrics) if self.models else None
        }
        
        self.training_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             model_name: str = "random_forest") -> Dict:
        """Perform hyperparameter tuning for a model"""
        try:
            self.logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            # Define parameter grids
            param_grids = {
                "random_forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                "xgboost": {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                "lightgbm": {
                    'n_estimators': [100, 150, 200],
                    'num_leaves': [31, 63, 127],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
            }
            
            if model_name not in param_grids:
                raise ValueError(f"Hyperparameter tuning not supported for {model_name}")
            
            # Get base model
            base_models = {
                "random_forest": RandomForestClassifier(random_state=42),
                "xgboost": xgb.XGBClassifier(random_state=42),
                "lightgbm": lgb.LGBMClassifier(random_state=42)
            }
            
            base_model = base_models.get(model_name)
            if not base_model:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best score: {grid_search.best_score_}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'best_estimator': grid_search.best_estimator_
            }
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {e}")
            raise
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                   model_name: str = "random_forest") -> Dict:
        """Train model with cross-validation"""
        try:
            models = {
                "random_forest": RandomForestClassifier(**self.config.get("random_forest", {})),
                "xgboost": xgb.XGBClassifier(**self.config.get("xgboost", {})),
                "lightgbm": lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42
                )
            }
            
            if model_name not in models:
                raise ValueError(f"Model {model_name} not supported for cross-validation")
            
            model = models[model_name]
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Train final model on all data
            model.fit(X, y)
            
            return {
                'model': model,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation training: {e}")
            raise