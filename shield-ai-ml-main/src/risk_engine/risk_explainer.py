import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import shap
import lime
import lime.lime_tabular
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from loguru import logger
from ...utils.logger import setup_logger


class RiskExplainer:
    """
    Explainable AI module for risk predictions
    Provides human-interpretable explanations for model decisions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'shap_samples': 100,
            'lime_samples': 5000,
            'top_features': 10,
            'explanation_depth': 'detailed'
        }
        
        self.logger = setup_logger(__name__)
        
        # SHAP explainer
        self.shap_explainer = None
        self.shap_values_cache = {}
        
        # LIME explainer
        self.lime_explainer = None
        
        # Explanation cache
        self.explanation_cache = {}
        
        # Feature descriptions
        self.feature_descriptions = self._initialize_feature_descriptions()
    
    def _initialize_feature_descriptions(self) -> Dict:
        """Initialize human-readable feature descriptions"""
        return {
            # Temporal features
            'hour_of_day': 'Time of day (0-23)',
            'is_night': 'Night time (10 PM - 6 AM)',
            'is_weekend': 'Weekend day',
            'is_holiday': 'Public holiday',
            'season': 'Season of the year',
            
            # Spatial features
            'crime_density': 'Crime incidents per square km',
            'safe_zone_distance': 'Distance to nearest safe zone (km)',
            'police_station_distance': 'Distance to police station (km)',
            'hospital_distance': 'Distance to hospital (km)',
            'location_type': 'Type of location (residential, commercial, etc.)',
            
            # Environmental features
            'lighting_score': 'Street lighting quality (0-1)',
            'crowd_density': 'Estimated crowd density (0-1)',
            'weather_risk_score': 'Weather-related risk (0-1)',
            'noise_level': 'Environmental noise level (0-1)',
            
            # Behavioral features
            'route_deviation_score': 'Deviation from normal route (0-1)',
            'speed': 'Current movement speed (m/s)',
            'stop_duration': 'Duration of current stop (seconds)',
            'user_confidence_score': 'User confidence based on app usage (0-1)',
            
            # Device features
            'battery_level': 'Device battery level (0-1)',
            'network_strength': 'Network signal strength (0-1)',
            'gps_accuracy': 'GPS accuracy (0-1, higher is better)',
            
            # Social features
            'guardian_online_count': 'Number of online guardians',
            'social_checkin_density': 'Social media check-ins in area (0-1)',
            'community_safety_score': 'Community-reported safety score (0-1)',
            
            # Derived features
            'night_isolation_risk': 'Night time isolation risk',
            'crime_lighting_risk': 'Combined crime and lighting risk',
            'battery_distance_risk': 'Battery and distance from safety risk'
        }
    
    def initialize_explainers(self, model: Any, training_data: np.ndarray,
                             feature_names: List[str]):
        """Initialize SHAP and LIME explainers"""
        try:
            self.logger.info("Initializing explainers...")
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.TreeExplainer(model)
            self.logger.info("SHAP explainer initialized")
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=['safe', 'risk'],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            self.logger.info("LIME explainer initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing explainers: {e}")
    
    def explain_prediction(self, features: Dict, prediction: Dict,
                          model: Optional[Any] = None) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        """
        try:
            self.logger.debug("Generating prediction explanation")
            
            # Convert features to array
            feature_array = self._features_to_array(features)
            feature_names = list(features.keys())
            
            # Generate multiple types of explanations
            explanations = {
                'feature_importance': self._explain_feature_importance(
                    feature_array, feature_names, model
                ),
                'local_explanation': self._generate_local_explanation(
                    feature_array, feature_names, prediction
                ),
                'counterfactual': self._generate_counterfactual(
                    features, prediction
                ),
                'rule_based': self._generate_rule_based_explanation(features),
                'visual_explanations': self._generate_visual_explanations(
                    feature_array, feature_names
                ),
                'human_readable': self._generate_human_readable_explanation(
                    features, prediction
                )
            }
            
            # Combine explanations
            combined = self._combine_explanations(explanations, prediction)
            
            # Cache explanation
            explanation_key = hash(str(features))
            self.explanation_cache[explanation_key] = {
                'explanation': combined,
                'timestamp': datetime.now()
            }
            
            self.logger.info("Prediction explanation generated")
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {
                'error': str(e),
                'basic_explanation': self._generate_basic_explanation(features, prediction)
            }
    
    def _features_to_array(self, features: Dict) -> np.ndarray:
        """Convert features dict to numpy array"""
        return np.array([list(features.values())])
    
    def _explain_feature_importance(self, feature_array: np.ndarray,
                                   feature_names: List[str],
                                   model: Any) -> Dict:
        """Explain feature importance using SHAP"""
        try:
            if self.shap_explainer is None:
                return {'error': 'SHAP explainer not initialized'}
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_array)
            
            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get values for risk class
            
            # Get feature contributions
            contributions = []
            for i, (name, value) in enumerate(zip(feature_names, feature_array[0])):
                contribution = float(shap_values[0][i])
                contributions.append({
                    'feature': name,
                    'value': float(value),
                    'contribution': contribution,
                    'absolute_contribution': abs(contribution),
                    'description': self.feature_descriptions.get(name, 'No description')
                })
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: x['absolute_contribution'], reverse=True)
            
            # Calculate statistics
            total_positive = sum(max(0, c['contribution']) for c in contributions)
            total_negative = sum(min(0, c['contribution']) for c in contributions)
            total_absolute = sum(c['absolute_contribution'] for c in contributions)
            
            return {
                'contributions': contributions[:self.config['top_features']],
                'statistics': {
                    'total_positive_impact': total_positive,
                    'total_negative_impact': total_negative,
                    'total_absolute_impact': total_absolute,
                    'top_positive': [
                        c for c in contributions if c['contribution'] > 0
                    ][:3],
                    'top_negative': [
                        c for c in contributions if c['contribution'] < 0
                    ][:3]
                },
                'method': 'SHAP',
                'confidence': self._calculate_shap_confidence(shap_values)
            }
            
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            return {'error': f'SHAP explanation failed: {str(e)}'}
    
    def _generate_local_explanation(self, feature_array: np.ndarray,
                                   feature_names: List[str],
                                   prediction: Dict) -> Dict:
        """Generate local explanation using LIME"""
        try:
            if self.lime_explainer is None:
                return {'error': 'LIME explainer not initialized'}
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                feature_array[0],
                lambda x: self._predict_proba_wrapper(x, prediction),
                num_features=self.config['top_features'],
                top_labels=1
            )
            
            # Extract explanation
            lime_explanation = []
            for feature, weight in explanation.as_list(label=1):
                # Parse feature and weight
                feature_name = feature.split(' ') [0].replace('<=', '').replace('>', '')
                lime_explanation.append({
                    'feature': feature_name,
                    'condition': feature,
                    'weight': weight,
                    'description': self.feature_descriptions.get(feature_name, 'No description')
                })
            
            return {
                'local_explanation': lime_explanation,
                'method': 'LIME',
                'confidence': explanation.score,
                'intercept': explanation.intercept[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error in LIME explanation: {e}")
            return {'error': f'LIME explanation failed: {str(e)}'}
    
    def _predict_proba_wrapper(self, instances: np.ndarray, 
                              prediction: Dict) -> np.ndarray:
        """Wrapper for prediction probability (for LIME)"""
        # This is a simplified version
        # In production, you would use the actual model's predict_proba
        
        # Return probabilities based on the provided prediction
        risk_prob = prediction.get('ensemble_prediction', {}).get('probabilities', [0.5, 0.5])[1]
        
        # Create array of probabilities for all instances
        n_instances = len(instances)
        return np.array([[1 - risk_prob, risk_prob]] * n_instances)
    
    def _generate_counterfactual(self, features: Dict, 
                                prediction: Dict) -> Dict:
        """Generate counterfactual explanations"""
        try:
            risk_level = prediction.get('ensemble_prediction', {}).get('risk_level', 0)
            
            if risk_level == 0:
                # For safe predictions: what would make it risky?
                return self._counterfactual_for_safe(features)
            else:
                # For risk predictions: what would make it safe?
                return self._counterfactual_for_risk(features)
            
        except Exception as e:
            self.logger.error(f"Error generating counterfactual: {e}")
            return {'error': str(e)}
    
    def _counterfactual_for_safe(self, features: Dict) -> Dict:
        """Generate counterfactual for safe predictions"""
        counterfactuals = []
        
        # Check what changes would increase risk
        current_features = features.copy()
        
        # 1. Change time to night
        if current_features.get('is_night', 0) == 0:
            counterfactual = current_features.copy()
            counterfactual['is_night'] = 1
            counterfactuals.append({
                'change': 'Change time to night',
                'features_changed': ['is_night'],
                'expected_impact': 'High risk increase',
                'reason': 'Night time significantly increases risk'
            })
        
        # 2. Reduce lighting
        if current_features.get('lighting_score', 0.5) > 0.3:
            counterfactual = current_features.copy()
            counterfactual['lighting_score'] = 0.1
            counterfactuals.append({
                'change': 'Reduce lighting to very poor',
                'features_changed': ['lighting_score'],
                'expected_impact': 'Medium risk increase',
                'reason': 'Poor lighting increases vulnerability'
            })
        
        # 3. Increase crime density
        if current_features.get('crime_density', 0) < 0.7:
            counterfactual = current_features.copy()
            counterfactual['crime_density'] = 0.9
            counterfactuals.append({
                'change': 'Move to high crime area',
                'features_changed': ['crime_density'],
                'expected_impact': 'High risk increase',
                'reason': 'High crime areas are inherently riskier'
            })
        
        # 4. Reduce crowd density
        if current_features.get('crowd_density', 0.5) > 0.3:
            counterfactual = current_features.copy()
            counterfactual['crowd_density'] = 0.1
            counterfactuals.append({
                'change': 'Go to isolated area',
                'features_changed': ['crowd_density'],
                'expected_impact': 'Medium risk increase',
                'reason': 'Isolation reduces help availability'
            })
        
        return {
            'type': 'safe_to_risk',
            'counterfactuals': counterfactuals[:3],  # Top 3
            'minimum_changes': self._find_minimum_changes(counterfactuals),
            'explanation': 'These changes would increase risk level'
        }
    
    def _counterfactual_for_risk(self, features: Dict) -> Dict:
        """Generate counterfactual for risk predictions"""
        counterfactuals = []
        
        # Check what changes would decrease risk
        current_features = features.copy()
        
        # 1. Move to safe zone
        if current_features.get('safe_zone_distance', 10) > 1:
            counterfactual = current_features.copy()
            counterfactual['safe_zone_distance'] = 0.5
            counterfactuals.append({
                'change': 'Move closer to safe zone (within 500m)',
                'features_changed': ['safe_zone_distance'],
                'expected_impact': 'High risk reduction',
                'reason': 'Proximity to safe zones increases safety'
            })
        
        # 2. Improve lighting
        if current_features.get('lighting_score', 0.5) < 0.7:
            counterfactual = current_features.copy()
            counterfactual['lighting_score'] = 0.9
            counterfactuals.append({
                'change': 'Move to well-lit area',
                'features_changed': ['lighting_score'],
                'expected_impact': 'Medium risk reduction',
                'reason': 'Good lighting improves visibility and safety'
            })
        
        # 3. Increase crowd density
        if current_features.get('crowd_density', 0.5) < 0.6:
            counterfactual = current_features.copy()
            counterfactual['crowd_density'] = 0.8
            counterfactuals.append({
                'change': 'Move to crowded area',
                'features_changed': ['crowd_density'],
                'expected_impact': 'Medium risk reduction',
                'reason': 'Higher crowd density provides witnesses and help'
            })
        
        # 4. Connect with guardians
        if current_features.get('guardian_online_count', 0) == 0:
            counterfactual = current_features.copy()
            counterfactual['guardian_online_count'] = 3
            counterfactuals.append({
                'change': 'Have multiple guardians online',
                'features_changed': ['guardian_online_count'],
                'expected_impact': 'High risk reduction',
                'reason': 'More guardians increases emergency response capability'
            })
        
        return {
            'type': 'risk_to_safe',
            'counterfactuals': counterfactuals[:3],  # Top 3
            'minimum_changes': self._find_minimum_changes(counterfactuals),
            'explanation': 'These changes would decrease risk level',
            'safety_actions': self._generate_safety_actions(counterfactuals)
        }
    
    def _find_minimum_changes(self, counterfactuals: List[Dict]) -> List[Dict]:
        """Find minimum changes to flip prediction"""
        if not counterfactuals:
            return []
        
        # Sort by expected impact and number of changes
        sorted_cf = sorted(counterfactuals, 
                          key=lambda x: (len(x['features_changed']), 
                                       x.get('expected_impact', 'Low')))
        
        return sorted_cf[:2]  # Return top 2 minimal changes
    
    def _generate_safety_actions(self, counterfactuals: List[Dict]) -> List[str]:
        """Generate actionable safety recommendations"""
        actions = []
        
        for cf in counterfactuals[:3]:
            change = cf.get('change', '')
            reason = cf.get('reason', '')
            actions.append(f"{change}: {reason}")
        
        # Add general safety actions
        actions.extend([
            "Share your live location with trusted contacts",
            "Keep emergency services number ready",
            "Stay in well-lit, populated areas",
            "Trust your instincts - if something feels wrong, leave"
        ])
        
        return actions
    
    def _generate_rule_based_explanation(self, features: Dict) -> Dict:
        """Generate rule-based explanation"""
        rules = []
        
        # Define risk rules
        risk_rules = [
            {
                'condition': lambda f: f.get('is_night', 0) == 1 and f.get('crowd_density', 0.5) < 0.3,
                'explanation': 'Night time with low crowd density',
                'risk_level': 'high'
            },
            {
                'condition': lambda f: f.get('crime_density', 0) > 0.7,
                'explanation': 'High crime area',
                'risk_level': 'high'
            },
            {
                'condition': lambda f: f.get('lighting_score', 0.5) < 0.3,
                'explanation': 'Poor lighting conditions',
                'risk_level': 'medium'
            },
            {
                'condition': lambda f: f.get('route_deviation_score', 0) > 0.6,
                'explanation': 'Unusual route deviation',
                'risk_level': 'medium'
            },
            {
                'condition': lambda f: f.get('battery_level', 1.0) < 0.2 and f.get('safe_zone_distance', 10) > 5,
                'explanation': 'Low battery far from safe zone',
                'risk_level': 'medium'
            },
            {
                'condition': lambda f: f.get('guardian_online_count', 0) == 0,
                'explanation': 'No guardians online',
                'risk_level': 'low'
            }
        ]
        
        # Apply rules
        triggered_rules = []
        for rule in risk_rules:
            if rule['condition'](features):
                triggered_rules.append({
                    'rule': rule['explanation'],
                    'risk_level': rule['risk_level'],
                    'features_involved': self._get_rule_features(features, rule['condition'])
                })
        
        return {
            'rule_based': triggered_rules,
            'rule_count': len(triggered_rules),
            'highest_risk_rule': max(triggered_rules, key=lambda x: x['risk_level']) if triggered_rules else None
        }
    
    def _get_rule_features(self, features: Dict, condition) -> List[str]:
        """Get features involved in a rule"""
        # This is a simplified implementation
        # In production, you would parse the condition function
        
        # Check common feature patterns
        involved_features = []
        
        if features.get('is_night', 0) == 1 and 'is_night' in str(condition):
            involved_features.append('is_night')
        
        if features.get('crowd_density', 0.5) < 0.3 and 'crowd_density' in str(condition):
            involved_features.append('crowd_density')
        
        if features.get('crime_density', 0) > 0.7 and 'crime_density' in str(condition):
            involved_features.append('crime_density')
        
        return involved_features
    
    def _generate_visual_explanations(self, feature_array: np.ndarray,
                                     feature_names: List[str]) -> Dict:
        """Generate visual explanations (SHAP plots)"""
        try:
            if self.shap_explainer is None:
                return {}
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_array)
            
            # Create plots
            visualizations = {}
            
            # 1. Force plot
            plt.figure(figsize=(10, 4))
            shap.force_plot(
                self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) 
                else self.shap_explainer.expected_value,
                shap_values[0] if isinstance(shap_values, list) else shap_values,
                feature_array[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['force_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # 2. Waterfall plot
            plt.figure(figsize=(12, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0] if isinstance(shap_values, list) else shap_values,
                    base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list)
                    else self.shap_explainer.expected_value,
                    data=feature_array[0],
                    feature_names=feature_names
                ),
                max_display=10,
                show=False
            )
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['waterfall_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # 3. Bar plot (feature importance)
            plt.figure(figsize=(10, 6))
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0] if isinstance(shap_values, list) else shap_values,
                    feature_names=feature_names
                ),
                max_display=15,
                show=False
            )
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            visualizations['bar_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return {
                'visualizations': visualizations,
                'plot_descriptions': {
                    'force_plot': 'Shows how each feature pushes the prediction from base value',
                    'waterfall_plot': 'Detailed breakdown of feature contributions',
                    'bar_plot': 'Overall feature importance ranking'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating visual explanations: {e}")
            return {'error': f'Visual explanation failed: {str(e)}'}
    
    def _generate_human_readable_explanation(self, features: Dict,
                                            prediction: Dict) -> Dict:
        """Generate human-readable natural language explanation"""
        try:
            risk_level = prediction.get('ensemble_prediction', {}).get('risk_level', 0)
            confidence = prediction.get('ensemble_prediction', {}).get('confidence', 0.5)
            
            # Base explanation
            if risk_level == 0:
                base_explanation = "The situation appears to be safe."
            else:
                base_explanation = "Potential risk detected."
            
            # Key factors
            key_factors = []
            
            # Check important features
            if features.get('is_night', 0) == 1:
                key_factors.append("It's night time, which increases vulnerability.")
            
            if features.get('crime_density', 0) > 0.7:
                key_factors.append(f"You're in a high-crime area (crime density: {features['crime_density']*100:.0f}%).")
            
            if features.get('lighting_score', 0.5) < 0.3:
                key_factors.append("Poor lighting reduces visibility and safety.")
            
            if features.get('crowd_density', 0.5) < 0.2:
                key_factors.append("The area appears isolated with few people around.")
            
            if features.get('route_deviation_score', 0) > 0.6:
                key_factors.append("You're on an unusual route compared to your normal patterns.")
            
            if features.get('battery_level', 1.0) < 0.2:
                key_factors.append(f"Your device battery is low ({features['battery_level']*100:.0f}%).")
            
            if features.get('guardian_online_count', 0) == 0:
                key_factors.append("No guardians are currently online to assist.")
            
            # Safety factors
            safety_factors = []
            if features.get('safe_zone_distance', 10) < 1:
                safety_factors.append("You're close to a safe zone.")
            
            if features.get('crowd_density', 0.5) > 0.7:
                safety_factors.append("The area is well-populated.")
            
            if features.get('guardian_online_count', 0) > 2:
                safety_factors.append("Multiple guardians are online and available.")
            
            # Generate summary
            if risk_level == 0:
                summary = "Overall, the current situation appears safe."
                if safety_factors:
                    summary += " Contributing safety factors include: " + ", ".join(safety_factors)
            else:
                summary = "Several risk factors have been identified: " + " ".join(key_factors[:3])
                if safety_factors:
                    summary += f" However, note that {', '.join(safety_factors)}"
            
            # Confidence statement
            if confidence > 0.8:
                confidence_stmt = "High confidence in this assessment."
            elif confidence > 0.6:
                confidence_stmt = "Moderate confidence in this assessment."
            else:
                confidence_stmt = "Lower confidence in this assessment due to limited data or conflicting signals."
            
            # Recommendations
            recommendations = []
            if risk_level == 1:
                if features.get('is_night', 0) == 1:
                    recommendations.append("Consider moving to a well-lit area.")
                
                if features.get('crowd_density', 0.5) < 0.3:
                    recommendations.append("Try to move to a more populated area.")
                
                if features.get('safe_zone_distance', 10) > 2:
                    recommendations.append("Head towards the nearest safe zone.")
                
                recommendations.append("Share your live location with trusted contacts.")
                recommendations.append("Stay alert to your surroundings.")
            else:
                recommendations.append("Continue with normal precautions.")
                recommendations.append("Stay aware of your surroundings.")
            
            return {
                'summary': base_explanation + " " + summary,
                'key_factors': key_factors,
                'safety_factors': safety_factors,
                'confidence_statement': confidence_stmt,
                'recommendations': recommendations,
                'risk_level_description': self._describe_risk_level(risk_level, confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating human-readable explanation: {e}")
            return {
                'summary': 'Unable to generate detailed explanation.',
                'key_factors': [],
                'safety_factors': [],
                'confidence_statement': '',
                'recommendations': ['Exercise normal caution.']
            }
    
    def _describe_risk_level(self, risk_level: int, confidence: float) -> str:
        """Describe risk level in human terms"""
        if risk_level == 0:
            if confidence > 0.8:
                return "Very safe conditions"
            elif confidence > 0.6:
                return "Generally safe conditions"
            else:
                return "Likely safe conditions"
        else:
            if confidence > 0.8:
                return "High risk situation requiring attention"
            elif confidence > 0.6:
                return "Elevated risk situation"
            else:
                return "Possible risk situation"
    
    def _combine_explanations(self, explanations: Dict, 
                             prediction: Dict) -> Dict:
        """Combine all explanations into a comprehensive response"""
        combined = {
            'timestamp': datetime.now().isoformat(),
            'prediction_summary': {
                'risk_level': prediction.get('ensemble_prediction', {}).get('risk_level', 0),
                'confidence': prediction.get('ensemble_prediction', {}).get('confidence', 0.0),
                'risk_label': prediction.get('ensemble_prediction', {}).get('risk_label', 'UNKNOWN')
            },
            'explanations': explanations,
            'key_insights': self._extract_key_insights(explanations),
            'actionable_insights': self._extract_actionable_insights(explanations)
        }
        
        # Calculate explanation confidence
        explanation_confidence = self._calculate_explanation_confidence(explanations)
        combined['explanation_confidence'] = explanation_confidence
        
        return combined
    
    def _extract_key_insights(self, explanations: Dict) -> List[str]:
        """Extract key insights from all explanations"""
        insights = []
        
        # From feature importance
        if 'feature_importance' in explanations:
            fi = explanations['feature_importance']
            if 'contributions' in fi:
                top_features = fi['contributions'][:3]
                for feature in top_features:
                    insights.append(
                        f"{feature['feature']} has {'positive' if feature['contribution'] > 0 else 'negative'} "
                        f"impact on risk (contribution: {feature['contribution']:.3f})"
                    )
        
        # From rule-based
        if 'rule_based' in explanations:
            rb = explanations['rule_based']
            if 'rule_based' in rb and rb['rule_based']:
                top_rule = rb['rule_based'][0]
                insights.append(
                    f"Rule triggered: {top_rule['rule']} ({top_rule['risk_level']} risk)"
                )
        
        # From counterfactual
        if 'counterfactual' in explanations:
            cf = explanations['counterfactual']
            if 'counterfactuals' in cf and cf['counterfactuals']:
                top_change = cf['counterfactuals'][0]
                insights.append(
                    f"To change risk level: {top_change['change']}"
                )
        
        return insights[:5]  # Limit to 5 insights
    
    def _extract_actionable_insights(self, explanations: Dict) -> List[Dict]:
        """Extract actionable insights"""
        actions = []
        
        # From counterfactual
        if 'counterfactual' in explanations:
            cf = explanations['counterfactual']
            if 'safety_actions' in cf:
                for i, action in enumerate(cf['safety_actions'][:3]):
                    actions.append({
                        'action': action,
                        'priority': 'high' if i == 0 else 'medium',
                        'source': 'counterfactual_analysis'
                    })
        
        # From human readable
        if 'human_readable' in explanations:
            hr = explanations['human_readable']
            if 'recommendations' in hr:
                for i, recommendation in enumerate(hr['recommendations'][:3]):
                    actions.append({
                        'action': recommendation,
                        'priority': 'high' if i < 2 else 'medium',
                        'source': 'expert_system'
                    })
        
        return actions
    
    def _calculate_explanation_confidence(self, explanations: Dict) -> float:
        """Calculate confidence in the explanations"""
        confidence_scores = []
        
        # Feature importance confidence
        if 'feature_importance' in explanations:
            fi = explanations['feature_importance']
            if 'confidence' in fi:
                confidence_scores.append(fi['confidence'])
        
        # Local explanation confidence
        if 'local_explanation' in explanations:
            le = explanations['local_explanation']
            if 'confidence' in le:
                confidence_scores.append(le['confidence'])
        
        # Rule-based confidence
        if 'rule_based' in explanations:
            rb = explanations['rule_based']
            if rb.get('rule_count', 0) > 0:
                # More rules triggered = higher confidence
                rule_confidence = min(1.0, rb['rule_count'] / 5.0)
                confidence_scores.append(rule_confidence)
        
        # Average confidence
        if confidence_scores:
            return float(np.mean(confidence_scores))
        else:
            return 0.5  # Default medium confidence
    
    def _calculate_shap_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate confidence based on SHAP values"""
        if shap_values is None or shap_values.size == 0:
            return 0.0
        
        # Calculate magnitude of SHAP values
        shap_magnitude = np.abs(shap_values).sum()
        
        # Normalize to 0-1 range (empirical threshold)
        confidence = min(1.0, shap_magnitude / 10.0)
        
        return float(confidence)
    
    def _generate_basic_explanation(self, features: Dict, 
                                   prediction: Dict) -> Dict:
        """Generate basic explanation when detailed explanation fails"""
        risk_level = prediction.get('ensemble_prediction', {}).get('risk_level', 0)
        
        if risk_level == 0:
            return {
                'summary': 'The system assesses the situation as safe based on available data.',
                'recommendations': ['Continue with normal activities.', 'Stay aware of surroundings.'],
                'confidence': 'basic_assessment'
            }
        else:
            return {
                'summary': 'Potential risk factors detected. Exercise caution.',
                'recommendations': [
                    'Increase situational awareness.',
                    'Consider moving to a safer location.',
                    'Share your location with trusted contacts.'
                ],
                'confidence': 'basic_assessment'
            }
    
    def clear_cache(self):
        """Clear explanation cache"""
        self.explanation_cache.clear()
        self.shap_values_cache.clear()
        self.logger.info("Explanation cache cleared")
    
    def get_explanation_stats(self) -> Dict:
        """Get statistics about explanations"""
        return {
            'cache_size': len(self.explanation_cache),
            'shap_cache_size': len(self.shap_values_cache),
            'feature_descriptions_count': len(self.feature_descriptions),
            'explainer_initialized': self.shap_explainer is not None and self.lime_explainer is not None,
            'last_clear': datetime.now().isoformat()
        }