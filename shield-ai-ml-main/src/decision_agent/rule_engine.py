import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import RiskLevel, InterventionType, AnomalyType


class RuleType(Enum):
    """Types of rules in the rule engine"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    BEHAVIORAL = "behavioral"
    DEVICE = "device"
    SOCIAL = "social"
    COMPOSITE = "composite"


class RulePriority(Enum):
    """Priority levels for rules"""
    CRITICAL = 10
    HIGH = 7
    MEDIUM = 5
    LOW = 3
    INFO = 1


@dataclass
class Rule:
    """Rule definition"""
    id: str
    name: str
    rule_type: RuleType
    priority: RulePriority
    condition: str
    action: str
    description: str
    enabled: bool = True
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class RuleEngine:
    """
    Advanced rule engine for safety decision making
    Supports complex rule evaluation and pattern matching
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'rule_evaluation_threshold': 0.7,
            'consecutive_triggers': 3,
            'cooldown_period_minutes': 30,
            'rule_confidence_threshold': 0.6
        }
        
        self.logger = setup_logger(__name__)
        
        # Rule storage
        self.rules: Dict[str, Rule] = {}
        self.rule_groups: Dict[RuleType, List[str]] = {}
        
        # Rule evaluation cache
        self.rule_cache = {}
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default safety rules"""
        default_rules = [
            Rule(
                id="rule_001",
                name="Night Isolation Risk",
                rule_type=RuleType.TEMPORAL,
                priority=RulePriority.HIGH,
                condition="context.is_night == 1 and context.crowd_density < 0.2 and context.guardian_online_count == 0",
                action="intervention.notify_guardians",
                description="High risk when alone at night in isolated area"
            ),
            Rule(
                id="rule_002",
                name="High Crime Area Alert",
                rule_type=RuleType.SPATIAL,
                priority=RulePriority.HIGH,
                condition="context.crime_density > 0.7 and context.is_night == 1",
                action="intervention.suggest_alternative_route",
                description="High crime area during night time"
            ),
            Rule(
                id="rule_003",
                name="Route Deviation Anomaly",
                rule_type=RuleType.BEHAVIORAL,
                priority=RulePriority.MEDIUM,
                condition="anomalies.route_deviation.score > 0.7 and context.is_night == 1",
                action="intervention.verify_user_status",
                description="Unusual route deviation at night"
            ),
            Rule(
                id="rule_004",
                name="Stalking Pattern Detected",
                rule_type=RuleType.BEHAVIORAL,
                priority=RulePriority.CRITICAL,
                condition="anomalies.stalking_risk > 0.8 and anomalies.stalking_detected == True",
                action="emergency.activate_emergency_protocol",
                description="Potential stalking pattern detected"
            ),
            Rule(
                id="rule_005",
                name="Low Battery Emergency",
                rule_type=RuleType.DEVICE,
                priority=RulePriority.HIGH,
                condition="context.battery_level < 0.2 and context.safe_zone_distance > 5",
                action="intervention.alert_low_battery",
                description="Low battery far from safe zone"
            ),
            Rule(
                id="rule_006",
                name="Poor Lighting Risk",
                rule_type=RuleType.SPATIAL,
                priority=RulePriority.MEDIUM,
                condition="context.lighting_score < 0.3 and context.is_night == 1",
                action="intervention.suggest_well_lit_route",
                description="Poor lighting conditions at night"
            ),
            Rule(
                id="rule_007",
                name="Speed Anomaly Alert",
                rule_type=RuleType.BEHAVIORAL,
                priority=RulePriority.MEDIUM,
                condition="anomalies.speed_anomaly.score > 0.8 and context.speed > 20",
                action="intervention.check_user_status",
                description="Unusually high speed detected"
            ),
            Rule(
                id="rule_008",
                name="Stop Anomaly Concern",
                rule_type=RuleType.BEHAVIORAL,
                priority=RulePriority.MEDIUM,
                condition="anomalies.stop_anomaly.is_stopped == True and anomalies.stop_anomaly.stop_duration_minutes > 10",
                action="intervention.verify_stop_intentional",
                description="Prolonged stop in unusual location"
            ),
            Rule(
                id="rule_009",
                name="Guardian Network Weak",
                rule_type=RuleType.SOCIAL,
                priority=RulePriority.LOW,
                condition="context.guardian_online_count == 0 and context.previous_alerts_count > 0",
                action="intervention.expand_guardian_network",
                description="No guardians online with previous alerts"
            ),
            Rule(
                id="rule_010",
                name="Composite High Risk",
                rule_type=RuleType.COMPOSITE,
                priority=RulePriority.CRITICAL,
                condition="context.risk_score > 0.8 and anomalies.stalking_risk > 0.6 and context.is_night == 1",
                action="emergency.full_emergency_response",
                description="Multiple high-risk factors combined"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: Rule):
        """Add a new rule to the engine"""
        self.rules[rule.id] = rule
        
        # Group by type
        if rule.rule_type not in self.rule_groups:
            self.rule_groups[rule.rule_type] = []
        self.rule_groups[rule.rule_type].append(rule.id)
        
        self.logger.info(f"Added rule: {rule.name} ({rule.id})")
    
    def remove_rule(self, rule_id: str):
        """Remove a rule from the engine"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            
            # Remove from groups
            if rule.rule_type in self.rule_groups:
                self.rule_groups[rule.rule_type].remove(rule_id)
            
            del self.rules[rule_id]
            self.logger.info(f"Removed rule: {rule_id}")
    
    def evaluate_rules(self, context: Dict, 
                      anomalies: Optional[Dict] = None,
                      risk_assessment: Optional[Dict] = None) -> List[Dict]:
        """
        Evaluate all rules against current context
        """
        try:
            self.logger.debug("Evaluating rules against context")
            
            triggered_rules = []
            
            # Prepare evaluation context
            eval_context = {
                'context': context,
                'anomalies': anomalies or {},
                'risk': risk_assessment or {},
                'current_time': datetime.now(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Evaluate each rule
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown period
                if self._is_in_cooldown(rule):
                    continue
                
                # Evaluate rule condition
                try:
                    condition_met = self._evaluate_condition(
                        rule.condition, eval_context
                    )
                    
                    if condition_met:
                        # Calculate rule confidence
                        confidence = self._calculate_rule_confidence(
                            rule, eval_context
                        )
                        
                        if confidence >= self.config['rule_confidence_threshold']:
                            # Rule triggered
                            triggered_rule = {
                                'rule_id': rule_id,
                                'rule_name': rule.name,
                                'rule_type': rule.rule_type.value,
                                'priority': rule.priority.value,
                                'action': rule.action,
                                'description': rule.description,
                                'confidence': confidence,
                                'condition_met': rule.condition,
                                'trigger_time': datetime.now().isoformat(),
                                'context_snapshot': self._create_context_snapshot(eval_context)
                            }
                            
                            triggered_rules.append(triggered_rule)
                            
                            # Update rule statistics
                            rule.last_triggered = datetime.now()
                            rule.trigger_count += 1
                            
                            self.logger.info(f"Rule triggered: {rule.name} (confidence: {confidence:.2f})")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule_id}: {e}")
                    continue
            
            # Sort by priority and confidence
            triggered_rules.sort(
                key=lambda x: (x['priority'], x['confidence']), 
                reverse=True
            )
            
            # Apply consecutive trigger logic
            filtered_rules = self._filter_consecutive_triggers(triggered_rules)
            
            self.logger.info(f"Total rules triggered: {len(filtered_rules)}")
            
            return filtered_rules
            
        except Exception as e:
            self.logger.error(f"Error in rule evaluation: {e}")
            return []
    
    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a rule condition against context"""
        try:
            # Replace context references with actual values
            evaluated_condition = self._replace_context_references(condition, context)
            
            # Safe evaluation using Python's eval with restricted globals
            allowed_globals = {
                '__builtins__': {},
                'True': True,
                'False': False,
                'None': None
            }
            
            # Add math functions
            import math
            allowed_globals.update({
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool
            })
            
            result = eval(evaluated_condition, allowed_globals, {})
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _replace_context_references(self, condition: str, context: Dict) -> str:
        """Replace context references with actual values"""
        try:
            # Pattern to match context references: context.attribute or anomalies.attribute
            pattern = r'(context|anomalies|risk)\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
            
            def replace_match(match):
                obj_type = match.group(1)
                path = match.group(2)
                
                # Navigate through nested dictionaries
                value = context.get(obj_type, {})
                for key in path.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        # If path not found, return original string
                        return match.group(0)
                
                # Convert value to appropriate Python literal
                if isinstance(value, str):
                    return f"'{value}'"
                elif isinstance(value, bool):
                    return str(value)
                elif isinstance(value, (int, float)):
                    return str(value)
                elif value is None:
                    return 'None'
                else:
                    return str(value)
            
            # Replace all matches
            result = re.sub(pattern, replace_match, condition)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error replacing context references: {e}")
            return condition
    
    def _calculate_rule_confidence(self, rule: Rule, context: Dict) -> float:
        """Calculate confidence for a triggered rule"""
        base_confidence = 0.7  # Base confidence for any rule
        
        # Adjust based on rule type
        if rule.rule_type == RuleType.TEMPORAL:
            temporal_factors = self._evaluate_temporal_factors(context)
            base_confidence *= temporal_factors
        
        elif rule.rule_type == RuleType.SPATIAL:
            spatial_factors = self._evaluate_spatial_factors(context)
            base_confidence *= spatial_factors
        
        elif rule.rule_type == RuleType.BEHAVIORAL:
            behavioral_factors = self._evaluate_behavioral_factors(context)
            base_confidence *= behavioral_factors
        
        # Adjust based on rule trigger history
        if rule.trigger_count > 10:
            # Established rule with many triggers
            base_confidence *= 1.1
        elif rule.trigger_count == 0:
            # New rule, slightly lower confidence
            base_confidence *= 0.9
        
        # Adjust based on time of day relevance
        if 'is_night' in context.get('context', {}):
            if context['context']['is_night'] == 1:
                if 'night' in rule.description.lower():
                    base_confidence *= 1.2
        
        return min(1.0, base_confidence)
    
    def _evaluate_temporal_factors(self, context: Dict) -> float:
        """Evaluate temporal factors for confidence"""
        factors = []
        
        current_hour = datetime.now().hour
        
        # Time of day factor
        if 22 <= current_hour <= 6:  # Night
            factors.append(1.2)
        elif 20 <= current_hour < 22 or 5 <= current_hour < 7:  # Twilight
            factors.append(1.1)
        else:
            factors.append(1.0)
        
        # Day of week factor
        current_weekday = datetime.now().weekday()
        if current_weekday >= 5:  # Weekend
            factors.append(1.1)
        else:
            factors.append(1.0)
        
        # Average the factors
        return np.mean(factors)
    
    def _evaluate_spatial_factors(self, context: Dict) -> float:
        """Evaluate spatial factors for confidence"""
        factors = []
        
        ctx = context.get('context', {})
        
        # Crime density factor
        crime_density = ctx.get('crime_density', 0)
        if crime_density > 0.7:
            factors.append(1.3)
        elif crime_density > 0.4:
            factors.append(1.1)
        else:
            factors.append(1.0)
        
        # Distance from safe zone factor
        safe_zone_distance = ctx.get('safe_zone_distance', 10)
        if safe_zone_distance > 5:
            factors.append(1.2)
        elif safe_zone_distance > 2:
            factors.append(1.1)
        else:
            factors.append(0.9)  # Closer to safe zone reduces confidence in spatial risk
        
        # Average the factors
        return np.mean(factors)
    
    def _evaluate_behavioral_factors(self, context: Dict) -> float:
        """Evaluate behavioral factors for confidence"""
        factors = []
        
        anomalies = context.get('anomalies', {})
        
        # Multiple anomalies increase confidence
        anomaly_count = 0
        if anomalies.get('route_deviation', {}).get('score', 0) > 0.5:
            anomaly_count += 1
        if anomalies.get('speed_anomaly', {}).get('score', 0) > 0.5:
            anomaly_count += 1
        if anomalies.get('stop_anomaly', {}).get('is_stopped', False):
            anomaly_count += 1
        
        if anomaly_count >= 2:
            factors.append(1.3)
        elif anomaly_count == 1:
            factors.append(1.1)
        else:
            factors.append(1.0)
        
        # User confidence factor
        user_confidence = context.get('context', {}).get('user_confidence_score', 0.5)
        factors.append(1.2 - user_confidence)  # Lower user confidence = higher rule confidence
        
        # Average the factors
        return np.mean(factors)
    
    def _create_context_snapshot(self, context: Dict) -> Dict:
        """Create a snapshot of relevant context for rule triggering"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': context.get('risk', {}).get('risk_score', 0),
            'risk_level': context.get('risk', {}).get('risk_level', 'UNKNOWN'),
            'location': {
                'latitude': context.get('context', {}).get('latitude', 0),
                'longitude': context.get('context', {}).get('longitude', 0)
            },
            'temporal': {
                'is_night': context.get('context', {}).get('is_night', 0),
                'hour': datetime.now().hour
            },
            'environmental': {
                'lighting_score': context.get('context', {}).get('lighting_score', 0.5),
                'crowd_density': context.get('context', {}).get('crowd_density', 0.5)
            },
            'device': {
                'battery_level': context.get('context', {}).get('battery_level', 1.0)
            },
            'social': {
                'guardian_online_count': context.get('context', {}).get('guardian_online_count', 0)
            }
        }
        
        return snapshot
    
    def _is_in_cooldown(self, rule: Rule) -> bool:
        """Check if rule is in cooldown period"""
        if rule.last_triggered is None:
            return False
        
        cooldown_minutes = self.config['cooldown_period_minutes']
        cooldown_end = rule.last_triggered + timedelta(minutes=cooldown_minutes)
        
        return datetime.now() < cooldown_end
    
    def _filter_consecutive_triggers(self, triggered_rules: List[Dict]) -> List[Dict]:
        """Filter rules based on consecutive trigger logic"""
        if not triggered_rules:
            return []
        
        # Group by rule type
        rule_groups = {}
        for rule in triggered_rules:
            rule_type = rule['rule_type']
            if rule_type not in rule_groups:
                rule_groups[rule_type] = []
            rule_groups[rule_type].append(rule)
        
        filtered_rules = []
        
        for rule_type, rules in rule_groups.items():
            # Sort by confidence within each group
            rules.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Take top rules based on consecutive trigger threshold
            max_rules = min(self.config['consecutive_triggers'], len(rules))
            filtered_rules.extend(rules[:max_rules])
        
        # Re-sort by priority
        filtered_rules.sort(key=lambda x: x['priority'], reverse=True)
        
        return filtered_rules
    
    def generate_rule_recommendations(self, context: Dict) -> List[Dict]:
        """Generate recommendations for new rules based on patterns"""
        recommendations = []
        
        # Analyze context for potential new rules
        ctx = context.get('context', {})
        anomalies = context.get('anomalies', {})
        
        # Check for patterns that might need new rules
        
        # Pattern 1: Repeated anomalies at same location
        if anomalies.get('route_deviation', {}).get('score', 0) > 0.6:
            recommendations.append({
                'pattern': 'repeated_route_deviation',
                'suggested_rule': {
                    'name': 'Frequent Route Deviation',
                    'condition': 'anomalies.route_deviation.score > 0.6 and context.route_familiarity < 0.3',
                    'action': 'intervention.investigate_pattern',
                    'description': 'Repeated route deviations in unfamiliar area'
                },
                'confidence': 0.7,
                'evidence': {
                    'current_deviation': anomalies['route_deviation']['score'],
                    'location_familiarity': ctx.get('route_familiarity', 0)
                }
            })
        
        # Pattern 2: Time-based risk combinations
        if ctx.get('is_night', 0) == 1 and ctx.get('battery_level', 1.0) < 0.3:
            recommendations.append({
                'pattern': 'night_low_battery',
                'suggested_rule': {
                    'name': 'Night Time Low Battery',
                    'condition': 'context.is_night == 1 and context.battery_level < 0.3',
                    'action': 'intervention.prioritize_charging',
                    'description': 'Low battery during night time increases vulnerability'
                },
                'confidence': 0.8,
                'evidence': {
                    'is_night': ctx['is_night'],
                    'battery_level': ctx['battery_level']
                }
            })
        
        # Pattern 3: Social isolation with previous alerts
        if ctx.get('guardian_online_count', 0) == 0 and ctx.get('previous_alerts_count', 0) > 2:
            recommendations.append({
                'pattern': 'isolated_with_history',
                'suggested_rule': {
                    'name': 'Isolated User with Alert History',
                    'condition': 'context.guardian_online_count == 0 and context.previous_alerts_count > 2',
                    'action': 'intervention.enhance_monitoring',
                    'description': 'User with alert history currently has no guardians online'
                },
                'confidence': 0.75,
                'evidence': {
                    'guardian_online': ctx['guardian_online_count'],
                    'previous_alerts': ctx['previous_alerts_count']
                }
            })
        
        return recommendations
    
    def get_rule_statistics(self) -> Dict:
        """Get statistics about rules and their usage"""
        stats = {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
            'rule_types': {},
            'most_triggered': [],
            'recently_triggered': []
        }
        
        # Count by rule type
        for rule in self.rules.values():
            rule_type = rule.rule_type.value
            stats['rule_types'][rule_type] = stats['rule_types'].get(rule_type, 0) + 1
        
        # Most triggered rules
        triggered_rules = [r for r in self.rules.values() if r.trigger_count > 0]
        triggered_rules.sort(key=lambda x: x.trigger_count, reverse=True)
        
        for rule in triggered_rules[:5]:
            stats['most_triggered'].append({
                'rule_id': rule.id,
                'name': rule.name,
                'trigger_count': rule.trigger_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            })
        
        # Recently triggered (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_rules = [r for r in self.rules.values() 
                       if r.last_triggered and r.last_triggered > recent_cutoff]
        
        for rule in recent_rules[:5]:
            stats['recently_triggered'].append({
                'rule_id': rule.id,
                'name': rule.name,
                'last_triggered': rule.last_triggered.isoformat(),
                'trigger_count': rule.trigger_count
            })
        
        return stats
    
    def export_rules(self, format: str = 'json') -> str:
        """Export rules in specified format"""
        rules_data = []
        
        for rule in self.rules.values():
            rule_data = {
                'id': rule.id,
                'name': rule.name,
                'type': rule.rule_type.value,
                'priority': rule.priority.value,
                'condition': rule.condition,
                'action': rule.action,
                'description': rule.description,
                'enabled': rule.enabled,
                'created_at': rule.created_at.isoformat(),
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                'trigger_count': rule.trigger_count
            }
            rules_data.append(rule_data)
        
        if format == 'json':
            return json.dumps(rules_data, indent=2)
        else:
            # Could add other formats like YAML, CSV
            return json.dumps(rules_data, indent=2)
    
    def import_rules(self, rules_data: str, format: str = 'json'):
        """Import rules from specified format"""
        try:
            if format == 'json':
                imported_rules = json.loads(rules_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            imported_count = 0
            for rule_data in imported_rules:
                try:
                    rule = Rule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        rule_type=RuleType(rule_data['type']),
                        priority=RulePriority(rule_data['priority']),
                        condition=rule_data['condition'],
                        action=rule_data['action'],
                        description=rule_data['description'],
                        enabled=rule_data.get('enabled', True),
                        created_at=datetime.fromisoformat(rule_data.get('created_at', datetime.now().isoformat())),
                        last_triggered=datetime.fromisoformat(rule_data['last_triggered']) if rule_data.get('last_triggered') else None,
                        trigger_count=rule_data.get('trigger_count', 0)
                    )
                    
                    self.add_rule(rule)
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error importing rule {rule_data.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Imported {imported_count} rules")
            
        except Exception as e:
            self.logger.error(f"Error importing rules: {e}")
            raise