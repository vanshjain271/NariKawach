import joblib
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import shutil
from pathlib import Path
import hashlib
from loguru import logger
from ..utils.logger import setup_logger


class ModelManager:
    """Manage model persistence, versioning, and deployment"""
    
    def __init__(self, model_dir: str = "storage/models"):
        self.model_dir = model_dir
        self.logger = setup_logger(__name__)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        directories = [
            self.model_dir,
            f"{self.model_dir}/risk_prediction",
            f"{self.model_dir}/anomaly_detection",
            f"{self.model_dir}/stalking_detection",
            f"{self.model_dir}/archive",
            f"{self.model_dir}/metadata"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, model_type: str,
                  metadata: Optional[Dict] = None) -> str:
        """Save a model with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_{timestamp}"
            
            # Create model directory
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            os.makedirs(model_path, exist_ok=True)
            
            # Save the model
            model_file = f"{model_path}/model.pkl"
            joblib.dump(model, model_file)
            
            # Generate model hash
            model_hash = self._generate_model_hash(model_file)
            
            # Prepare metadata
            model_metadata = {
                'model_id': model_id,
                'model_name': model_name,
                'model_type': model_type,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'model_hash': model_hash,
                'model_size': os.path.getsize(model_file),
                'python_version': os.sys.version,
                'dependencies': self._get_dependencies(),
                'metadata': metadata or {}
            }
            
            # Save metadata
            metadata_file = f"{model_path}/metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Update model registry
            self._update_model_registry(model_metadata)
            
            self.logger.info(f"Saved model {model_name} to {model_path}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_id: str, model_type: str) -> tuple[Any, Dict]:
        """Load a model and its metadata"""
        try:
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_id} not found")
            
            # Load model
            model_file = f"{model_path}/model.pkl"
            model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = f"{model_path}/metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify model hash
            current_hash = self._generate_model_hash(model_file)
            if current_hash != metadata.get('model_hash'):
                self.logger.warning(f"Model hash mismatch for {model_id}")
            
            self.logger.info(f"Loaded model {model_id}")
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def load_latest_model(self, model_type: str, model_name: Optional[str] = None) -> tuple[Any, Dict]:
        """Load the latest model of a given type"""
        try:
            model_dir = f"{self.model_dir}/{model_type}"
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"No models found for type {model_type}")
            
            # Get all model directories
            model_dirs = [d for d in os.listdir(model_dir) 
                         if os.path.isdir(f"{model_dir}/{d}")]
            
            if not model_dirs:
                raise FileNotFoundError(f"No models found for type {model_type}")
            
            # Filter by model name if specified
            if model_name:
                model_dirs = [d for d in model_dirs if d.startswith(model_name)]
            
            if not model_dirs:
                raise FileNotFoundError(f"No models found with name {model_name}")
            
            # Sort by timestamp (newest first)
            model_dirs.sort(reverse=True)
            latest_model_id = model_dirs[0]
            
            return self.load_model(latest_model_id, model_type)
            
        except Exception as e:
            self.logger.error(f"Error loading latest model: {e}")
            raise
    
    def delete_model(self, model_id: str, model_type: str):
        """Delete a model"""
        try:
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            
            if os.path.exists(model_path):
                # Move to archive instead of deleting
                archive_path = f"{self.model_dir}/archive/{model_type}_{model_id}"
                shutil.move(model_path, archive_path)
                self.logger.info(f"Archived model {model_id}")
            else:
                self.logger.warning(f"Model {model_id} not found")
            
            # Update registry
            self._remove_from_registry(model_id)
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all available models"""
        try:
            registry_file = f"{self.model_dir}/metadata/registry.json"
            
            if not os.path.exists(registry_file):
                return []
            
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            if model_type:
                return [model for model in registry if model.get('model_type') == model_type]
            else:
                return registry
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        try:
            registry = self.list_models()
            
            for model in registry:
                if model.get('model_id') == model_id:
                    # Load full metadata
                    model_type = model.get('model_type')
                    model_path = f"{self.model_dir}/{model_type}/{model_id}"
                    metadata_file = f"{model_path}/metadata.json"
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            full_metadata = json.load(f)
                        return full_metadata
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def deploy_model(self, model_id: str, model_type: str, 
                    deployment_name: str = "production") -> bool:
        """Deploy a model to production"""
        try:
            # Load model and metadata
            model, metadata = self.load_model(model_id, model_type)
            
            # Create deployment directory
            deployment_dir = f"{self.model_dir}/deployments/{deployment_name}"
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Copy model files
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            deployment_files = ['model.pkl', 'metadata.json']
            
            for file in deployment_files:
                src = f"{model_path}/{file}"
                dst = f"{deployment_dir}/{file}"
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            
            # Update deployment registry
            self._update_deployment_registry(model_id, model_type, deployment_name, metadata)
            
            # Create symlink for easy access
            symlink_path = f"{self.model_dir}/current_{deployment_name}"
            if os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(deployment_dir, symlink_path)
            
            self.logger.info(f"Deployed model {model_id} as {deployment_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return False
    
    def get_deployed_model(self, deployment_name: str = "production") -> Optional[tuple[Any, Dict]]:
        """Get the currently deployed model"""
        try:
            deployment_dir = f"{self.model_dir}/deployments/{deployment_name}"
            
            if not os.path.exists(deployment_dir):
                return None
            
            # Load model
            model_file = f"{deployment_dir}/model.pkl"
            if not os.path.exists(model_file):
                return None
            
            model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = f"{deployment_dir}/metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error getting deployed model: {e}")
            return None
    
    def _generate_model_hash(self, model_file: str) -> str:
        """Generate hash for a model file"""
        try:
            with open(model_file, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            
            return file_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating model hash: {e}")
            return "unknown"
    
    def _get_dependencies(self) -> Dict:
        """Get current Python dependencies"""
        try:
            import pkg_resources
            
            dependencies = {}
            for dist in pkg_resources.working_set:
                dependencies[dist.key] = dist.version
            
            return dependencies
            
        except:
            return {}
    
    def _update_model_registry(self, model_metadata: Dict):
        """Update the model registry"""
        try:
            registry_file = f"{self.model_dir}/metadata/registry.json"
            
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
            else:
                registry = []
            
            # Remove old entry if exists
            registry = [model for model in registry 
                       if model.get('model_id') != model_metadata['model_id']]
            
            # Add new entry
            registry_entry = {
                'model_id': model_metadata['model_id'],
                'model_name': model_metadata['model_name'],
                'model_type': model_metadata['model_type'],
                'created_at': model_metadata['created_at'],
                'model_hash': model_metadata['model_hash'],
                'deployed': False
            }
            
            registry.append(registry_entry)
            
            # Sort by creation date (newest first)
            registry.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Keep only last 1000 entries
            registry = registry[:1000]
            
            # Save registry
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error updating model registry: {e}")
    
    def _remove_from_registry(self, model_id: str):
        """Remove a model from the registry"""
        try:
            registry_file = f"{self.model_dir}/metadata/registry.json"
            
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                # Remove entry
                registry = [model for model in registry 
                           if model.get('model_id') != model_id]
                
                # Save updated registry
                with open(registry_file, 'w') as f:
                    json.dump(registry, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error removing from registry: {e}")
    
    def _update_deployment_registry(self, model_id: str, model_type: str,
                                   deployment_name: str, metadata: Dict):
        """Update deployment registry"""
        try:
            deployment_file = f"{self.model_dir}/metadata/deployments.json"
            
            if os.path.exists(deployment_file):
                with open(deployment_file, 'r') as f:
                    deployments = json.load(f)
            else:
                deployments = {}
            
            # Update deployment entry
            deployments[deployment_name] = {
                'model_id': model_id,
                'model_type': model_type,
                'deployed_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            # Save deployments
            with open(deployment_file, 'w') as f:
                json.dump(deployments, f, indent=2)
            
            # Update main registry
            registry_file = f"{self.model_dir}/metadata/registry.json"
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                for model in registry:
                    if model['model_id'] == model_id:
                        model['deployed'] = True
                        model['deployment_name'] = deployment_name
                        model['deployed_at'] = datetime.now().isoformat()
                
                with open(registry_file, 'w') as f:
                    json.dump(registry, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error updating deployment registry: {e}")
    
    def export_model(self, model_id: str, model_type: str, 
                    export_format: str = "onnx") -> Optional[str]:
        """Export model to different format"""
        try:
            if export_format == "onnx":
                return self._export_to_onnx(model_id, model_type)
            elif export_format == "tensorflow":
                return self._export_to_tensorflow(model_id, model_type)
            elif export_format == "pytorch":
                return self._export_to_pytorch(model_id, model_type)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return None
    
    def _export_to_onnx(self, model_id: str, model_type: str) -> Optional[str]:
        """Export model to ONNX format"""
        try:
            # Load model
            model, metadata = self.load_model(model_id, model_type)
            
            # This would require ONNX conversion logic
            # For now, return the path to the original model
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error exporting to ONNX: {e}")
            return None
    
    def validate_model(self, model_id: str, model_type: str, 
                      validation_data: Dict) -> Dict:
        """Validate a model with new data"""
        try:
            # Load model
            model, metadata = self.load_model(model_id, model_type)
            
            # Extract data
            X_test = validation_data.get('X_test')
            y_test = validation_data.get('y_test')
            
            if X_test is None or y_test is None:
                raise ValueError("Validation data must contain X_test and y_test")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Compare with original metrics if available
            original_metrics = metadata.get('metadata', {}).get('metrics', {})
            
            if original_metrics:
                metrics['comparison'] = {
                    'accuracy_change': metrics['accuracy'] - original_metrics.get('accuracy', 0),
                    'f1_change': metrics['f1_score'] - original_metrics.get('f1_score', 0)
                }
            
            # Save validation results
            validation_results = {
                'model_id': model_id,
                'validation_date': datetime.now().isoformat(),
                'metrics': metrics,
                'validation_data_size': len(X_test)
            }
            
            # Save to model directory
            model_path = f"{self.model_dir}/{model_type}/{model_id}"
            validation_file = f"{model_path}/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            raise