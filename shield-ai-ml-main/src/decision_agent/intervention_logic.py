import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import RiskLevel, InterventionType, EMERGENCY_PRIORITIES


class InterventionStage(Enum):
    """Stages of intervention"""
    MONITORING = "monitoring"
    ALERT = "alert"
    VERIFICATION = "verification"
    ACTION = "action"
    ESCALATION = "escalation"
    RESOLUTION = "resolution"


class CommunicationChannel(Enum):
    """Communication channels for interventions"""
    IN_APP_NOTIFICATION = "in_app"
    PUSH_NOTIFICATION = "push"
    SMS = "sms"
    CALL = "call"
    EMAIL = "email"
    EMERGENCY_CALL = "emergency_call"


@dataclass
class Intervention:
    """Intervention definition"""
    id: str
    user_id: str
    stage: InterventionStage
    intervention_type: InterventionType
    priority: int
    triggered_rules: List[str]
    context: Dict
    actions: List[Dict]
    status: str = "pending"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_received: bool = False
    response_time: Optional[float] = None
    outcome: str = "unknown"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class InterventionAgent:
    """
    Intelligent intervention agent for safety system
    Manages intervention logic and escalation protocols
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'intervention_timeout_minutes': 5,
            'max_intervention_attempts': 3,
            'escalation_threshold': 0.8,
            'verification_timeout_seconds': 60,
            'guardian_response_timeout': 120,
            'emergency_services_threshold': 0.9
        }
        
        self.logger = setup_logger(__name__)
        
        # Active interventions
        self.active_interventions: Dict[str, Intervention] = {}
        
        # Intervention history
        self.intervention_history: List[Intervention] = []
        
        # User intervention preferences
        self.user_preferences: Dict[str, Dict] = {}
        
        # Guardian information
        self.guardian_networks: Dict[str, List[Dict]] = {}
        
        # Initialize communication channels
        self.communication_channels = self._initialize_channels()
    
    def _initialize_channels(self) -> Dict[str, Dict]:
        """Initialize communication channels"""
        return {
            'in_app': {
                'name': 'In-App Notification',
                'priority': 1,
                'response_rate': 0.8,
                'enabled': True
            },
            'push': {
                'name': 'Push Notification',
                'priority': 2,
                'response_rate': 0.7,
                'enabled': True
            },
            'sms': {
                'name': 'SMS',
                'priority': 3,
                'response_rate': 0.9,
                'enabled': True
            },
            'call': {
                'name': 'Phone Call',
                'priority': 4,
                'response_rate': 0.95,
                'enabled': True
            },
            'emergency_call': {
                'name': 'Emergency Call',
                'priority': 5,
                'response_rate': 1.0,
                'enabled': True
            }
        }
    
    def decide_intervention(self, user_id: str, 
                          risk_assessment: Dict,
                          anomalies: Dict,
                          triggered_rules: List[Dict],
                          user_context: Dict) -> Dict:
        """
        Decide on appropriate intervention based on situation
        """
        try:
            self.logger.info(f"Deciding intervention for user {user_id}")
            
            # Determine intervention type and stage
            intervention_type, stage = self._determine_intervention_type(
                risk_assessment, anomalies, triggered_rules
            )
            
            # Calculate intervention priority
            priority = self._calculate_intervention_priority(
                risk_assessment, anomalies, triggered_rules
            )
            
            # Generate intervention actions
            actions = self._generate_intervention_actions(
                intervention_type, stage, priority, user_context
            )
            
            # Create intervention
            intervention_id = f"intv_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            intervention = Intervention(
                id=intervention_id,
                user_id=user_id,
                stage=stage,
                intervention_type=intervention_type,
                priority=priority,
                triggered_rules=[r['rule_id'] for r in triggered_rules],
                context={
                    'risk_assessment': risk_assessment,
                    'anomalies': anomalies,
                    'user_context': user_context,
                    'triggered_rules': triggered_rules
                },
                actions=actions,
                status="planned"
            )
            
            # Store intervention
            self.active_interventions[intervention_id] = intervention
            
            # Generate intervention plan
            intervention_plan = self._create_intervention_plan(intervention)
            
            self.logger.info(f"Intervention decided: {intervention_type.value} (priority: {priority})")
            
            return intervention_plan
            
        except Exception as e:
            self.logger.error(f"Error deciding intervention: {e}")
            return {
                'error': str(e),
                'intervention_type': InterventionType.SILENT_MONITORING,
                'priority': 1,
                'actions': [{'action': 'monitor', 'channel': 'system'}]
            }
    
    def _determine_intervention_type(self, risk_assessment: Dict,
                                   anomalies: Dict,
                                   triggered_rules: List[Dict]) -> Tuple[InterventionType, InterventionStage]:
        """Determine appropriate intervention type and stage"""
        risk_score = risk_assessment.get('risk_score', 0)
        risk_level = risk_assessment.get('risk_level', RiskLevel.SAFE)
        
        # Check for critical conditions first
        if risk_score > self.config['emergency_services_threshold']:
            return InterventionType.EMERGENCY_ALERT, InterventionStage.ESCALATION
        
        # Check for stalking patterns
        if anomalies.get('stalking_detected', False):
            stalking_risk = anomalies.get('stalking_risk', 0)
            if stalking_risk > 0.8:
                return InterventionType.POLICE_NOTIFICATION, InterventionStage.ESCALATION
        
        # Determine based on risk level
        if risk_level == RiskLevel.CRITICAL:
            return InterventionType.EMERGENCY_ALERT, InterventionStage.ESCALATION
        elif risk_level == RiskLevel.HIGH:
            return InterventionType.EMERGENCY_ALERT, InterventionStage.ACTION
        elif risk_level == RiskLevel.MEDIUM:
            return InterventionType.GUARDIAN_NOTIFICATION, InterventionStage.VERIFICATION
        elif risk_level == RiskLevel.LOW:
            return InterventionType.GUARDIAN_NOTIFICATION, InterventionStage.ALERT
        else:
            return InterventionType.SILENT_MONITORING, InterventionStage.MONITORING
    
    def _calculate_intervention_priority(self, risk_assessment: Dict,
                                       anomalies: Dict,
                                       triggered_rules: List[Dict]) -> int:
        """Calculate intervention priority (1-10)"""
        base_priority = 1
        
        # Risk score contribution
        risk_score = risk_assessment.get('risk_score', 0)
        base_priority += int(risk_score * 5)
        
        # Rule priority contribution
        if triggered_rules:
            max_rule_priority = max([r.get('priority', 0) for r in triggered_rules], default=0)
            base_priority += min(3, max_rule_priority // 3)
        
        # Anomaly contribution
        if anomalies.get('stalking_detected', False):
            base_priority += 2
        
        # Time of day contribution
        current_hour = datetime.now().hour
        if 22 <= current_hour <= 6:  # Night
            base_priority += 1
        
        return min(10, base_priority)
    
    def _generate_intervention_actions(self, intervention_type: InterventionType,
                                     stage: InterventionStage,
                                     priority: int,
                                     user_context: Dict) -> List[Dict]:
        """Generate specific actions for the intervention"""
        actions = []
        
        # Base monitoring action
        actions.append({
            'action_id': 'monitor_user',
            'action_type': 'monitoring',
            'description': 'Continue monitoring user location and status',
            'channel': 'system',
            'priority': 1,
            'timeout': 300,  # 5 minutes
            'expected_response': False
        })
        
        # Add actions based on intervention type and stage
        if intervention_type == InterventionType.SILENT_MONITORING:
            actions.append({
                'action_id': 'increase_monitoring_frequency',
                'action_type': 'monitoring',
                'description': 'Increase location tracking frequency',
                'channel': 'system',
                'priority': 2,
                'timeout': 600,
                'expected_response': False
            })
        
        elif intervention_type == InterventionType.GUARDIAN_NOTIFICATION:
            # Notify guardians
            guardian_count = user_context.get('guardian_online_count', 0)
            
            if guardian_count > 0:
                actions.append({
                    'action_id': 'notify_primary_guardian',
                    'action_type': 'notification',
                    'description': 'Send alert to primary guardian',
                    'channel': self._select_notification_channel(priority),
                    'priority': 3,
                    'timeout': self.config['guardian_response_timeout'],
                    'expected_response': True,
                    'recipients': ['primary_guardian']
                })
            
            if guardian_count > 1 and priority > 5:
                actions.append({
                    'action_id': 'notify_secondary_guardians',
                    'action_type': 'notification',
                    'description': 'Send alert to secondary guardians',
                    'channel': self._select_notification_channel(priority - 1),
                    'priority': 2,
                    'timeout': self.config['guardian_response_timeout'],
                    'expected_response': False,
                    'recipients': ['secondary_guardians']
                })
        
        elif intervention_type == InterventionType.EMERGENCY_ALERT:
            # Emergency actions
            actions.append({
                'action_id': 'activate_emergency_mode',
                'action_type': 'emergency',
                'description': 'Activate emergency mode for user',
                'channel': 'system',
                'priority': 10,
                'timeout': 30,
                'expected_response': False
            })
            
            actions.append({
                'action_id': 'notify_all_guardians',
                'action_type': 'notification',
                'description': 'Send emergency alert to all guardians',
                'channel': self._select_notification_channel(10),
                'priority': 9,
                'timeout': 60,
                'expected_response': True,
                'recipients': ['all_guardians']
            })
            
            actions.append({
                'action_id': 'share_live_location',
                'action_type': 'sharing',
                'description': 'Share live location with guardians',
                'channel': 'system',
                'priority': 8,
                'timeout': 30,
                'expected_response': False
            })
        
        elif intervention_type == InterventionType.POLICE_NOTIFICATION:
            # Police notification actions
            actions.append({
                'action_id': 'prepare_police_report',
                'action_type': 'reporting',
                'description': 'Prepare incident report for police',
                'channel': 'system',
                'priority': 10,
                'timeout': 120,
                'expected_response': False
            })
            
            actions.append({
                'action_id': 'contact_emergency_services',
                'action_type': 'emergency',
                'description': 'Contact emergency services',
                'channel': 'emergency_call',
                'priority': 10,
                'timeout': 300,
                'expected_response': True,
                'recipients': ['emergency_services']
            })
        
        elif intervention_type == InterventionType.SAFE_NAVIGATION:
            # Safe navigation actions
            actions.append({
                'action_id': 'calculate_safe_route',
                'action_type': 'navigation',
                'description': 'Calculate safest route to destination',
                'channel': 'system',
                'priority': 4,
                'timeout': 30,
                'expected_response': False
            })
            
            actions.append({
                'action_id': 'suggest_safe_route',
                'action_type': 'notification',
                'description': 'Suggest safe route to user',
                'channel': 'in_app',
                'priority': 3,
                'timeout': 60,
                'expected_response': True,
                'recipients': ['user']
            })
        
        # Sort actions by priority
        actions.sort(key=lambda x: x['priority'], reverse=True)
        
        return actions
    
    def _select_notification_channel(self, priority: int) -> str:
        """Select appropriate notification channel based on priority"""
        if priority >= 9:
            return 'call'
        elif priority >= 7:
            return 'sms'
        elif priority >= 5:
            return 'push'
        else:
            return 'in_app'
    
    def _create_intervention_plan(self, intervention: Intervention) -> Dict:
        """Create comprehensive intervention plan"""
        return {
            'intervention_id': intervention.id,
            'user_id': intervention.user_id,
            'intervention_type': intervention.intervention_type.value,
            'stage': intervention.stage.value,
            'priority': intervention.priority,
            'status': intervention.status,
            'triggered_rules': intervention.triggered_rules,
            'actions': intervention.actions,
            'estimated_duration': self._estimate_duration(intervention.actions),
            'required_resources': self._identify_resources(intervention),
            'success_criteria': self._define_success_criteria(intervention),
            'escalation_path': self._define_escalation_path(intervention),
            'created_at': intervention.created_at.isoformat()
        }
    
    def _estimate_duration(self, actions: List[Dict]) -> int:
        """Estimate total duration of intervention in seconds"""
        total_duration = 0
        
        for action in actions:
            timeout = action.get('timeout', 0)
            if action.get('expected_response', False):
                total_duration += timeout
            else:
                total_duration += min(timeout, 60)  # Non-response actions capped at 60s
        
        return total_duration
    
    def _identify_resources(self, intervention: Intervention) -> List[str]:
        """Identify required resources for intervention"""
        resources = ['location_tracking', 'communication_system']
        
        if intervention.intervention_type in [InterventionType.EMERGENCY_ALERT, 
                                            InterventionType.POLICE_NOTIFICATION]:
            resources.extend(['emergency_services_api', 'police_database'])
        
        if intervention.intervention_type == InterventionType.SAFE_NAVIGATION:
            resources.append('mapping_service')
        
        if 'guardian' in intervention.intervention_type.value.lower():
            resources.append('guardian_network')
        
        return resources
    
    def _define_success_criteria(self, intervention: Intervention) -> Dict:
        """Define success criteria for intervention"""
        criteria = {
            'primary': 'User reaches safe status',
            'secondary': [],
            'safety_indicators': []
        }
        
        if intervention.intervention_type == InterventionType.SILENT_MONITORING:
            criteria['primary'] = 'Risk level returns to safe'
            criteria['secondary'].append('No new anomalies detected for 15 minutes')
        
        elif intervention.intervention_type == InterventionType.GUARDIAN_NOTIFICATION:
            criteria['primary'] = 'Guardian acknowledges and responds'
            criteria['secondary'].append('User confirms safety')
            criteria['safety_indicators'].extend([
                'User moves to safe location',
                'Guardian establishes contact'
            ])
        
        elif intervention.intervention_type == InterventionType.EMERGENCY_ALERT:
            criteria['primary'] = 'Emergency resolved successfully'
            criteria['secondary'].extend([
                'User confirmed safe',
                'Emergency services informed if needed'
            ])
            criteria['safety_indicators'].extend([
                'User location stable in safe area',
                'No further risk indicators'
            ])
        
        return criteria
    
    def _define_escalation_path(self, intervention: Intervention) -> List[Dict]:
        """Define escalation path for intervention"""
        escalation_path = []
        
        base_stage = intervention.stage
        
        # Define escalation stages
        stages = {
            InterventionStage.MONITORING: [
                {'next_stage': 'alert', 'condition': 'risk_increases', 'timeout': 300},
                {'next_stage': 'verification', 'condition': 'anomaly_detected', 'timeout': 180}
            ],
            InterventionStage.ALERT: [
                {'next_stage': 'verification', 'condition': 'no_response', 'timeout': 120},
                {'next_stage': 'action', 'condition': 'risk_escalates', 'timeout': 60}
            ],
            InterventionStage.VERIFICATION: [
                {'next_stage': 'action', 'condition': 'verification_failed', 'timeout': 90},
                {'next_stage': 'escalation', 'condition': 'emergency_confirmed', 'timeout': 30}
            ],
            InterventionStage.ACTION: [
                {'next_stage': 'escalation', 'condition': 'action_ineffective', 'timeout': 120},
                {'next_stage': 'resolution', 'condition': 'action_successful', 'timeout': 300}
            ],
            InterventionStage.ESCALATION: [
                {'next_stage': 'resolution', 'condition': 'emergency_resolved', 'timeout': 600}
            ]
        }
        
        current_stage_name = base_stage.value
        if current_stage_name in stages:
            escalation_path = stages[current_stage_name]
        
        return escalation_path
    
    async def execute_intervention(self, intervention_id: str):
        """Execute an intervention plan"""
        try:
            if intervention_id not in self.active_interventions:
                raise ValueError(f"Intervention {intervention_id} not found")
            
            intervention = self.active_interventions[intervention_id]
            intervention.status = "executing"
            intervention.started_at = datetime.now()
            
            self.logger.info(f"Executing intervention {intervention_id}")
            
            # Execute actions in order of priority
            successful_actions = []
            failed_actions = []
            
            for action in intervention.actions:
                try:
                    action_result = await self._execute_action(action, intervention)
                    
                    if action_result['success']:
                        successful_actions.append({
                            'action_id': action['action_id'],
                            'result': action_result
                        })
                        self.logger.info(f"Action {action['action_id']} completed successfully")
                    else:
                        failed_actions.append({
                            'action_id': action['action_id'],
                            'result': action_result,
                            'error': action_result.get('error', 'Unknown error')
                        })
                        self.logger.warning(f"Action {action['action_id']} failed")
                        
                        # Check if we need to escalate
                        if self._should_escalate(intervention, failed_actions):
                            await self._escalate_intervention(intervention)
                            break
                
                except Exception as e:
                    self.logger.error(f"Error executing action {action['action_id']}: {e}")
                    failed_actions.append({
                        'action_id': action['action_id'],
                        'error': str(e)
                    })
            
            # Update intervention status
            if len(failed_actions) == 0:
                intervention.status = "completed"
                intervention.outcome = "success"
                intervention.completed_at = datetime.now()
            elif len(successful_actions) > len(failed_actions):
                intervention.status = "partial_success"
                intervention.outcome = "partial"
                intervention.completed_at = datetime.now()
            else:
                intervention.status = "failed"
                intervention.outcome = "failure"
                intervention.completed_at = datetime.now()
            
            # Calculate response time if applicable
            if intervention.response_received:
                if intervention.started_at:
                    intervention.response_time = (
                        intervention.completed_at - intervention.started_at
                    ).total_seconds()
            
            # Move to history
            self.intervention_history.append(intervention)
            del self.active_interventions[intervention_id]
            
            # Log completion
            self.logger.info(f"Intervention {intervention_id} completed with status: {intervention.status}")
            
            return {
                'intervention_id': intervention_id,
                'status': intervention.status,
                'outcome': intervention.outcome,
                'successful_actions': successful_actions,
                'failed_actions': failed_actions,
                'response_time': intervention.response_time,
                'completed_at': intervention.completed_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing intervention {intervention_id}: {e}")
            
            if intervention_id in self.active_interventions:
                intervention = self.active_interventions[intervention_id]
                intervention.status = "error"
                intervention.outcome = "execution_error"
                intervention.completed_at = datetime.now()
            
            raise
    
    async def _execute_action(self, action: Dict, intervention: Intervention) -> Dict:
        """Execute a single intervention action"""
        action_type = action.get('action_type', '')
        channel = action.get('channel', '')
        timeout = action.get('timeout', 30)
        
        try:
            # Simulate different action types
            if action_type == 'monitoring':
                result = await self._execute_monitoring_action(action, intervention, timeout)
            elif action_type == 'notification':
                result = await self._execute_notification_action(action, intervention, timeout)
            elif action_type == 'emergency':
                result = await self._execute_emergency_action(action, intervention, timeout)
            elif action_type == 'reporting':
                result = await self._execute_reporting_action(action, intervention, timeout)
            elif action_type == 'navigation':
                result = await self._execute_navigation_action(action, intervention, timeout)
            elif action_type == 'sharing':
                result = await self._execute_sharing_action(action, intervention, timeout)
            else:
                result = {
                    'success': False,
                    'error': f'Unknown action type: {action_type}'
                }
            
            # Update response received status
            if action.get('expected_response', False) and result.get('success', False):
                if result.get('response_received', False):
                    intervention.response_received = True
            
            return result
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f'Action timeout after {timeout} seconds',
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_monitoring_action(self, action: Dict, 
                                       intervention: Intervention,
                                       timeout: int) -> Dict:
        """Execute monitoring action"""
        # Simulate monitoring action
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            'success': True,
            'action_type': 'monitoring',
            'details': 'Monitoring frequency increased',
            'monitoring_level': 'enhanced',
            'duration_seconds': 2
        }
    
    async def _execute_notification_action(self, action: Dict,
                                         intervention: Intervention,
                                         timeout: int) -> Dict:
        """Execute notification action"""
        channel = action.get('channel', 'in_app')
        recipients = action.get('recipients', [])
        
        # Simulate notification sending
        await asyncio.sleep(1)
        
        # Simulate response based on channel
        response_rate = self.communication_channels.get(channel, {}).get('response_rate', 0.7)
        response_received = np.random.random() < response_rate
        
        return {
            'success': True,
            'action_type': 'notification',
            'channel': channel,
            'recipients': recipients,
            'message_sent': True,
            'response_received': response_received,
            'response_rate': response_rate,
            'simulated_response': response_received  # In production, this would be actual response
        }
    
    async def _execute_emergency_action(self, action: Dict,
                                      intervention: Intervention,
                                      timeout: int) -> Dict:
        """Execute emergency action"""
        # Simulate emergency action
        await asyncio.sleep(3)
        
        return {
            'success': True,
            'action_type': 'emergency',
            'details': 'Emergency protocol activated',
            'emergency_level': 'high',
            'services_notified': ['guardians', 'system_monitoring']
        }
    
    async def _execute_reporting_action(self, action: Dict,
                                      intervention: Intervention,
                                      timeout: int) -> Dict:
        """Execute reporting action"""
        # Simulate report generation
        await asyncio.sleep(2)
        
        report_data = {
            'incident_summary': 'Safety intervention report',
            'risk_factors': intervention.context.get('risk_assessment', {}).get('risk_factors', []),
            'actions_taken': [a['action_id'] for a in intervention.actions],
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'success': True,
            'action_type': 'reporting',
            'report_generated': True,
            'report_data': report_data
        }
    
    async def _execute_navigation_action(self, action: Dict,
                                       intervention: Intervention,
                                       timeout: int) -> Dict:
        """Execute navigation action"""
        # Simulate route calculation
        await asyncio.sleep(1)
        
        return {
            'success': True,
            'action_type': 'navigation',
            'safe_route_calculated': True,
            'alternative_routes': 3,
            'safety_score': 0.85
        }
    
    async def _execute_sharing_action(self, action: Dict,
                                    intervention: Intervention,
                                    timeout: int) -> Dict:
        """Execute sharing action"""
        # Simulate location sharing
        await asyncio.sleep(1)
        
        return {
            'success': True,
            'action_type': 'sharing',
            'sharing_active': True,
            'shared_with': ['guardians', 'emergency_contacts'],
            'update_frequency': '10s'
        }
    
    def _should_escalate(self, intervention: Intervention, 
                        failed_actions: List[Dict]) -> bool:
        """Determine if intervention should be escalated"""
        if not failed_actions:
            return False
        
        # Count critical failures
        critical_failures = sum(1 for fa in failed_actions 
                              if fa.get('action_id', '').startswith('notify_'))
        
        # Check if too many critical actions failed
        if critical_failures >= 2:
            return True
        
        # Check if emergency action failed
        emergency_failed = any('emergency' in fa.get('action_id', '') 
                             for fa in failed_actions)
        if emergency_failed:
            return True
        
        # Check timeout
        if intervention.started_at:
            elapsed = (datetime.now() - intervention.started_at).total_seconds()
            if elapsed > self.config['intervention_timeout_minutes'] * 60:
                return True
        
        return False
    
    async def _escalate_intervention(self, intervention: Intervention):
        """Escalate intervention to higher level"""
        self.logger.warning(f"Escalating intervention {intervention.id}")
        
        # Update intervention stage
        intervention.stage = InterventionStage.ESCALATION
        
        # Add escalation actions
        escalation_actions = [
            {
                'action_id': 'escalate_to_supervisor',
                'action_type': 'notification',
                'description': 'Escalate to human supervisor',
                'channel': 'call',
                'priority': 10,
                'timeout': 180,
                'expected_response': True,
                'recipients': ['supervisor']
            },
            {
                'action_id': 'activate_backup_protocol',
                'action_type': 'emergency',
                'description': 'Activate backup emergency protocol',
                'channel': 'system',
                'priority': 9,
                'timeout': 60,
                'expected_response': False
            }
        ]
        
        intervention.actions.extend(escalation_actions)
        
        # Re-sort actions by priority
        intervention.actions.sort(key=lambda x: x['priority'], reverse=True)
    
    def get_active_interventions(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get active interventions"""
        interventions = list(self.active_interventions.values())
        
        if user_id:
            interventions = [i for i in interventions if i.user_id == user_id]
        
        return [
            {
                'id': i.id,
                'user_id': i.user_id,
                'type': i.intervention_type.value,
                'stage': i.stage.value,
                'priority': i.priority,
                'status': i.status,
                'created_at': i.created_at.isoformat(),
                'started_at': i.started_at.isoformat() if i.started_at else None,
                'actions_count': len(i.actions)
            }
            for i in interventions
        ]
    
    def get_intervention_history(self, user_id: Optional[str] = None, 
                               hours: int = 24) -> List[Dict]:
        """Get intervention history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            i for i in self.intervention_history
            if i.completed_at and i.completed_at > cutoff_time
        ]
        
        if user_id:
            history = [i for i in history if i.user_id == user_id]
        
        return [
            {
                'id': i.id,
                'user_id': i.user_id,
                'type': i.intervention_type.value,
                'outcome': i.outcome,
                'priority': i.priority,
                'response_time': i.response_time,
                'created_at': i.created_at.isoformat(),
                'completed_at': i.completed_at.isoformat() if i.completed_at else None,
                'triggered_rules_count': len(i.triggered_rules)
            }
            for i in history[-50:]  # Last 50 interventions
        ]
    
    def get_intervention_statistics(self, user_id: Optional[str] = None) -> Dict:
        """Get intervention statistics"""
        if user_id:
            user_history = [i for i in self.intervention_history if i.user_id == user_id]
            user_active = [i for i in self.active_interventions.values() if i.user_id == user_id]
        else:
            user_history = self.intervention_history
            user_active = list(self.active_interventions.values())
        
        # Calculate statistics
        total_interventions = len(user_history) + len(user_active)
        
        if total_interventions == 0:
            return {
                'total_interventions': 0,
                'active_interventions': 0,
                'success_rate': 0,
                'average_response_time': 0
            }
        
        # Count by outcome
        outcomes = {}
        for intervention in user_history:
            outcome = intervention.outcome
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        # Calculate success rate
        successful = outcomes.get('success', 0) + outcomes.get('partial', 0)
        success_rate = successful / len(user_history) if len(user_history) > 0 else 0
        
        # Calculate average response time
        response_times = [i.response_time for i in user_history if i.response_time]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Count by type
        types = {}
        for intervention in user_history + user_active:
            int_type = intervention.intervention_type.value
            types[int_type] = types.get(int_type, 0) + 1
        
        return {
            'total_interventions': total_interventions,
            'active_interventions': len(user_active),
            'historical_interventions': len(user_history),
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'outcomes': outcomes,
            'types': types,
            'recent_interventions': len([i for i in user_history 
                                       if i.completed_at and 
                                       (datetime.now() - i.completed_at).total_seconds() < 3600])
        }
    
    def set_user_preferences(self, user_id: str, preferences: Dict):
        """Set user intervention preferences"""
        self.user_preferences[user_id] = preferences
        self.logger.info(f"Set preferences for user {user_id}")
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user intervention preferences"""
        return self.user_preferences.get(user_id, {
            'notification_channels': ['in_app', 'push', 'sms'],
            'escalation_contacts': ['primary_guardian'],
            'auto_escalate': True,
            'emergency_services_consent': True,
            'data_sharing_level': 'guardians_only'
        })
    
    def add_guardian_network(self, user_id: str, guardians: List[Dict]):
        """Add guardian network for a user"""
        self.guardian_networks[user_id] = guardians
        self.logger.info(f"Added {len(guardians)} guardians for user {user_id}")
    
    def get_guardian_network(self, user_id: str) -> List[Dict]:
        """Get guardian network for a user"""
        return self.guardian_networks.get(user_id, [])