import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                           average_precision_score, confusion_matrix, 
                           classification_report)
from datetime import datetime
import json
import io
import base64
from loguru import logger
from ..utils.logger import setup_logger


class ModelEvaluator:
    """Evaluate model performance and generate reports"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
    def evaluate_model_performance(self, model: Any, X_test: np.ndarray, 
                                  y_test: np.ndarray, model_name: str = "") -> Dict:
        """Comprehensive model evaluation"""
        try:
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {}
            
            # Basic classification metrics
            metrics['classification_report'] = classification_report(
                y_test, y_pred, output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
            
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            
            # ROC and AUC if probabilities available
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            
            # Calculate F-beta scores
            metrics['f2_score'] = self._calculate_f_beta_score(
                metrics['precision'], metrics['recall'], beta=2
            )
            metrics['f0_5_score'] = self._calculate_f_beta_score(
                metrics['precision'], metrics['recall'], beta=0.5
            )
            
            # Calculate Matthews Correlation Coefficient
            metrics['mcc'] = self._calculate_mcc(tp, tn, fp, fn)
            
            # Calculate Balanced Accuracy
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
            
            # Generate visualizations
            visualization_data = self._generate_visualizations(
                model, X_test, y_test, y_pred, y_pred_proba, model_name
            )
            
            metrics['visualizations'] = visualization_data
            
            # Performance interpretation
            metrics['performance_interpretation'] = self._interpret_performance(metrics)
            
            self.logger.info(f"Evaluation complete for {model_name}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            raise
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """Compare multiple models"""
        try:
            self.logger.info(f"Comparing {len(models)} models")
            
            comparison = {}
            
            for name, model in models.items():
                comparison[name] = self.evaluate_model_performance(
                    model, X_test, y_test, name
                )
            
            # Generate comparison metrics
            comparison_summary = self._generate_comparison_summary(comparison)
            
            # Generate comparison visualizations
            comparison_plots = self._generate_comparison_visualizations(comparison)
            
            return {
                'detailed_comparison': comparison,
                'summary': comparison_summary,
                'visualizations': comparison_plots,
                'best_model': self._select_best_model_from_comparison(comparison)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            raise
    
    def evaluate_model_fairness(self, model: Any, X_test: np.ndarray, 
                               y_test: np.ndarray, 
                               sensitive_features: Dict[str, np.ndarray]) -> Dict:
        """Evaluate model fairness across sensitive attributes"""
        try:
            self.logger.info("Evaluating model fairness")
            
            fairness_metrics = {}
            
            for feature_name, feature_values in sensitive_features.items():
                unique_values = np.unique(feature_values)
                
                group_metrics = {}
                for value in unique_values:
                    mask = feature_values == value
                    X_group = X_test[mask]
                    y_group = y_test[mask]
                    
                    if len(X_group) > 0:
                        group_metrics[str(value)] = self.evaluate_model_performance(
                            model, X_group, y_group, f"{feature_name}={value}"
                        )
                
                # Calculate fairness metrics
                fairness_metrics[feature_name] = {
                    'group_metrics': group_metrics,
                    'fairness_scores': self._calculate_fairness_scores(group_metrics)
                }
            
            return fairness_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model fairness: {e}")
            raise
    
    def _calculate_f_beta_score(self, precision: float, recall: float, 
                               beta: float) -> float:
        """Calculate F-beta score"""
        if precision + recall == 0:
            return 0.0
        
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculate Matthews Correlation Coefficient"""
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _generate_visualizations(self, model: Any, X_test: np.ndarray, 
                                y_test: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: Optional[np.ndarray], 
                                model_name: str) -> Dict:
        """Generate visualization data for model evaluation"""
        visualizations = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Confusion Matrix Heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # 2. ROC Curve (if probabilities available)
            if y_pred_proba is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve - {model_name}')
                ax.legend(loc="lower right")
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                visualizations['roc_curve'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            
            # 3. Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importance = model.feature_importances_
                features = [f'Feature {i}' for i in range(len(feature_importance))]
                
                # Sort features by importance
                indices = np.argsort(feature_importance)[::-1]
                
                ax.bar(range(len(feature_importance)), feature_importance[indices])
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Importance')
                ax.set_title(f'Feature Importance - {model_name}')
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                visualizations['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            
            # 4. Precision-Recall Curve
            if y_pred_proba is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                average_precision = average_precision_score(y_test, y_pred_proba)
                
                ax.plot(recall, precision, color='blue', lw=2, 
                       label=f'PR curve (AP = {average_precision:.3f})')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve - {model_name}')
                ax.legend(loc="upper right")
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                visualizations['pr_curve'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _interpret_performance(self, metrics: Dict) -> Dict:
        """Interpret model performance metrics"""
        interpretation = {}
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1_score = metrics.get('f1_score', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        # Accuracy interpretation
        if accuracy >= 0.9:
            interpretation['accuracy'] = 'Excellent'
        elif accuracy >= 0.8:
            interpretation['accuracy'] = 'Good'
        elif accuracy >= 0.7:
            interpretation['accuracy'] = 'Fair'
        else:
            interpretation['accuracy'] = 'Poor'
        
        # F1 Score interpretation
        if f1_score >= 0.8:
            interpretation['f1_score'] = 'Excellent balance of precision and recall'
        elif f1_score >= 0.6:
            interpretation['f1_score'] = 'Good balance'
        elif f1_score >= 0.4:
            interpretation['f1_score'] = 'Moderate balance'
        else:
            interpretation['f1_score'] = 'Poor balance'
        
        # ROC-AUC interpretation
        if roc_auc >= 0.9:
            interpretation['roc_auc'] = 'Excellent discrimination'
        elif roc_auc >= 0.8:
            interpretation['roc_auc'] = 'Good discrimination'
        elif roc_auc >= 0.7:
            interpretation['roc_auc'] = 'Fair discrimination'
        else:
            interpretation['roc_auc'] = 'Poor discrimination'
        
        # Precision-Recall tradeoff
        if precision > recall:
            interpretation['tradeoff'] = 'Model is more precise but misses some positives'
        elif recall > precision:
            interpretation['tradeoff'] = 'Model catches more positives but has more false positives'
        else:
            interpretation['tradeoff'] = 'Balanced precision and recall'
        
        # Recommendations
        recommendations = []
        
        if metrics.get('false_negative', 0) > metrics.get('false_positive', 0):
            recommendations.append('Reduce false negatives (missed risks) by adjusting threshold')
        
        if metrics.get('false_positive', 0) > metrics.get('false_negative', 0):
            recommendations.append('Reduce false positives (false alarms) by adjusting threshold')
        
        if roc_auc < 0.7:
            recommendations.append('Consider feature engineering or try different algorithms')
        
        interpretation['recommendations'] = recommendations
        
        return interpretation
    
    def _generate_comparison_summary(self, comparison: Dict) -> Dict:
        """Generate summary of model comparison"""
        summary = {}
        
        for model_name, metrics in comparison.items():
            summary[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'mcc': metrics.get('mcc', 0),
                'training_time': metrics.get('training_time', 0),
                'prediction_time': metrics.get('prediction_time', 0)
            }
        
        return summary
    
    def _generate_comparison_visualizations(self, comparison: Dict) -> Dict:
        """Generate comparison visualizations"""
        visualizations = {}
        
        try:
            # Comparison bar chart
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc']
            
            for idx, metric in enumerate(metrics_to_plot):
                if idx < len(axes):
                    model_names = list(comparison.keys())
                    metric_values = [comparison[name].get(metric, 0) for name in model_names]
                    
                    axes[idx].bar(model_names, metric_values, color='skyblue')
                    axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
                    axes[idx].set_ylabel(metric)
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['metrics_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error generating comparison visualizations: {e}")
        
        return visualizations
    
    def _select_best_model_from_comparison(self, comparison: Dict) -> Dict:
        """Select best model based on comparison"""
        best_score = -1
        best_model = None
        
        for model_name, metrics in comparison.items():
            # Use weighted score: 40% accuracy, 30% f1, 30% roc_auc
            accuracy = metrics.get('accuracy', 0)
            f1_score = metrics.get('f1_score', 0)
            roc_auc = metrics.get('roc_auc', 0)
            
            weighted_score = 0.4 * accuracy + 0.3 * f1_score + 0.3 * roc_auc
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model = {
                    'name': model_name,
                    'score': weighted_score,
                    'metrics': {
                        'accuracy': accuracy,
                        'f1_score': f1_score,
                        'roc_auc': roc_auc
                    }
                }
        
        return best_model
    
    def _calculate_fairness_scores(self, group_metrics: Dict) -> Dict:
        """Calculate fairness scores across groups"""
        fairness_scores = {}
        
        # Extract metrics for each group
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return fairness_scores
        
        # Calculate demographic parity
        positive_rates = []
        for group in groups:
            metrics = group_metrics[group]
            tp = metrics.get('true_positive', 0)
            fp = metrics.get('false_positive', 0)
            tn = metrics.get('true_negative', 0)
            fn = metrics.get('false_negative', 0)
            
            if (tp + fp + tn + fn) > 0:
                positive_rate = (tp + fp) / (tp + fp + tn + fn)
                positive_rates.append(positive_rate)
        
        if positive_rates:
            fairness_scores['demographic_parity_difference'] = max(positive_rates) - min(positive_rates)
            fairness_scores['demographic_parity_ratio'] = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
        
        # Calculate equal opportunity
        recall_rates = []
        for group in groups:
            recall = group_metrics[group].get('recall', 0)
            recall_rates.append(recall)
        
        if recall_rates:
            fairness_scores['equal_opportunity_difference'] = max(recall_rates) - min(recall_rates)
            fairness_scores['equal_opportunity_ratio'] = min(recall_rates) / max(recall_rates) if max(recall_rates) > 0 else 0
        
        return fairness_scores