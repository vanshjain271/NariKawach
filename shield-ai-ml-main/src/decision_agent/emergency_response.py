import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import RiskLevel, InterventionType, EMERGENCY_PRIORITIES


class EmergencyPhase(Enum):
    """Phases of emergency response"""
    DETECTION = "detection"
    VERIFICATION = "verification"
    ALERT = "alert"
    RESPONSE = "response"
    COORDINATION = "coordination"
    RESOLUTION = "resolution"
    POST_INCIDENT = "post_incident"


class EmergencyService(Enum):
    """Emergency services"""
    POLICE = "police"
    AMBULANCE = "ambulance"
    FIRE = "fire"
    RESCUE = "rescue"
    COAST_GUARD = "coast_guard"
    MOUNTAIN_REScue = "mountain_rescue"


@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    relationship: str
    phone: str
    email: Optional[str] = None
    priority: int = 1
    response_time: Optional[float] = None
    available: bool = True


class EmergencyResponseCoordinator:
    """
    Emergency response coordinator for critical situations
    Manages multi-agency coordination and emergency protocols
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'emergency_threshold': 0.85,
            'verification_timeout_seconds': 30,
            'max_emergency_contacts': 5,
            'police_response_timeout': 180,
            'ambulance_response_timeout': 120,
            'multi_agency_coordination': True,
            'incident_report_required': True
        }
        
        self.logger = setup_logger(__name__)
        
        # Active emergencies
        self.active_emergencies: Dict[str, Dict] = {}
        
        # Emergency contacts database
        self.emergency_contacts: Dict[str, List[EmergencyContact]] = {}
        
        # Emergency services coordination
        self.emergency_services = self._initialize_services()
        
        # Incident history
        self.incident_history: List[Dict] = []
        
        # Response protocols
        self.response_protocols = self._initialize_protocols()
        
        # Communication templates
        self.communication_templates = self._initialize_templates()
    
    def _initialize_services(self) -> Dict[str, Dict]:
        """Initialize emergency services information"""
        return {
            'police': {
                'contact_number': '100',
                'response_time_minutes': 15,
                'capabilities': ['crime_response', 'rescue', 'traffic_control'],
                'jurisdiction': 'local',
                'available': True
            },
            'ambulance': {
                'contact_number': '102',
                'response_time_minutes': 10,
                'capabilities': ['medical_emergency', 'accident_response'],
                'jurisdiction': 'local',
                'available': True
            },
            'fire': {
                'contact_number': '101',
                'response_time_minutes': 8,
                'capabilities': ['fire', 'rescue', 'hazardous_materials'],
                'jurisdiction': 'local',
                'available': True
            },
            'women_helpline': {
                'contact_number': '1091',
                'response_time_minutes': 20,
                'capabilities': ['women_safety', 'harassment', 'stalking'],
                'jurisdiction': 'national',
                'available': True
            },
            'disaster_management': {
                'contact_number': '108',
                'response_time_minutes': 30,
                'capabilities': ['disaster_response', 'coordination'],
                'jurisdiction': 'national',
                'available': True
            }
        }
    
    def _initialize_protocols(self) -> Dict[str, Dict]:
        """Initialize emergency response protocols"""
        return {
            'medical_emergency': {
                'services': ['ambulance'],
                'priority': 1,
                'actions': [
                    'activate_medical_protocol',
                    'notify_emergency_contacts',
                    'provide_first_aid_instructions',
                    'coordinate_hospital_admission'
                ],
                'checklist': [
                    'assess_vitals',
                    'secure_area',
                    'gather_medical_history',
                    'prepare_transport'
                ]
            },
            'crime_incident': {
                'services': ['police', 'women_helpline'],
                'priority': 1,
                'actions': [
                    'secure_evidence',
                    'notify_law_enforcement',
                    'protect_victim',
                    'document_incident'
                ],
                'checklist': [
                    'ensure_safety',
                    'gather_witness_info',
                    'preserve_scene',
                    'coordinate_with_authorities'
                ]
            },
            'stalking_emergency': {
                'services': ['police', 'women_helpline'],
                'priority': 1,
                'actions': [
                    'activate_stalking_protocol',
                    'establish_safe_location',
                    'notify_trusted_contacts',
                    'coordinate_police_response'
                ],
                'checklist': [
                    'verify_stalking_pattern',
                    'secure_current_location',
                    'document_evidence',
                    'plan_safe_extraction'
                ]
            },
            'accident_response': {
                'services': ['ambulance', 'police'],
                'priority': 2,
                'actions': [
                    'assess_injuries',
                    'secure_accident_scene',
                    'notify_emergency_services',
                    'coordinate_traffic_control'
                ],
                'checklist': [
                    'check_for_danger',
                    'attend_to_injured',
                    'call_emergency_services',
                    'gather_accident_details'
                ]
            },
            'natural_disaster': {
                'services': ['disaster_management', 'ambulance', 'fire'],
                'priority': 1,
                'actions': [
                    'activate_disaster_protocol',
                    'establish_communication',
                    'coordinate_evacuation',
                    'provide_safety_instructions'
                ],
                'checklist': [
                    'assess_threat_level',
                    'identify_safe_zones',
                    'coordinate_resources',
                    'maintain_communication'
                ]
            }
        }
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize communication templates"""
        return {
            'police_alert': """
            EMERGENCY ALERT - SHIELD Safety System
            
            User: {user_name}
            User ID: {user_id}
            Emergency Type: {emergency_type}
            
            Location Details:
            Latitude: {latitude}
            Longitude: {longitude}
            Address: {address}
            Timestamp: {timestamp}
            
            Situation Details:
            {situation_details}
            
            Risk Assessment:
            Risk Score: {risk_score}
            Primary Risk Factors: {risk_factors}
            
            User Information:
            Age: {age}
            Medical Conditions: {medical_info}
            Emergency Contacts: {emergency_contacts}
            
            Required Response: {required_response}
            
            This is an automated alert from SHIELD Safety System.
            Please respond urgently.
            """,
            
            'guardian_alert': """
            URGENT: Safety Alert for {user_name}
            
            {user_name} is in a potentially dangerous situation.
            
            Current Status:
            Location: {location}
            Risk Level: {risk_level}
            Time: {timestamp}
            
            Actions Taken:
            1. Emergency services notified: {services_notified}
            2. Live location sharing activated
            3. Safety protocols engaged
            
            Recommended Actions:
            1. Attempt to contact {user_name}
            2. Share this information with trusted contacts
            3. Proceed to location if safe to do so
            4. Contact emergency services: {emergency_numbers}
            
            Live tracking available at: {tracking_url}
            
            This is an automated alert from SHIELD Safety System.
            """,
            
            'user_instructions': """
            SAFETY INSTRUCTIONS - SHIELD Emergency Protocol
            
            Current Status: {emergency_type} detected
            
            IMMEDIATE ACTIONS:
            1. Move to a safe location if possible
            2. Keep your phone accessible
            3. Do not panic - help is on the way
            
            Safety Instructions:
            {safety_instructions}
            
            Emergency Services ETA: {eta_minutes} minutes
            
            Your location is being shared with:
            - Emergency services
            - Trusted contacts
            
            Stay on the line for further instructions.
            This message will update as the situation develops.
            """
        }
    
    def activate_emergency_protocol(self, user_id: str, 
                                   emergency_data: Dict) -> Dict:
        """
        Activate emergency response protocol
        """
        try:
            self.logger.critical(f"Activating emergency protocol for user {user_id}")
            
            # Determine emergency type
            emergency_type = self._determine_emergency_type(emergency_data)
            
            # Create emergency record
            emergency_id = f"emerg_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            emergency_record = {
                'emergency_id': emergency_id,
                'user_id': user_id,
                'emergency_type': emergency_type,
                'phase': EmergencyPhase.DETECTION.value,
                'start_time': datetime.now().isoformat(),
                'emergency_data': emergency_data,
                'activated_protocols': [],
                'notified_services': [],
                'notified_contacts': [],
                'actions_taken': [],
                'status': 'active',
                'coordinates': {
                    'latitude': emergency_data.get('latitude', 0),
                    'longitude': emergency_data.get('longitude', 0)
                }
            }
            
            # Store emergency
            self.active_emergencies[emergency_id] = emergency_record
            
            # Execute emergency protocol
            protocol_result = self._execute_emergency_protocol(
                emergency_id, emergency_type, emergency_data
            )
            
            # Update record
            emergency_record.update(protocol_result)
            
            self.logger.critical(
                f"Emergency protocol activated: {emergency_id} "
                f"Type: {emergency_type} "
                f"Services notified: {len(emergency_record['notified_services'])}"
            )
            
            # Start monitoring thread
            self._start_emergency_monitoring(emergency_id)
            
            return {
                'emergency_id': emergency_id,
                'emergency_type': emergency_type,
                'protocol_activated': True,
                'actions_initiated': emergency_record['actions_taken'],
                'services_notified': emergency_record['notified_services'],
                'next_steps': self._get_next_steps(emergency_type, EmergencyPhase.DETECTION)
            }
            
        except Exception as e:
            self.logger.critical(f"Error activating emergency protocol: {e}")
            return {
                'error': str(e),
                'emergency_activated': False,
                'recommendation': 'Manual emergency activation required'
            }
    
    def _determine_emergency_type(self, emergency_data: Dict) -> str:
        """Determine type of emergency"""
        risk_score = emergency_data.get('risk_score', 0)
        anomalies = emergency_data.get('anomalies', {})
        
        # Check for specific emergency types
        if anomalies.get('stalking_detected', False):
            stalking_risk = anomalies.get('stalking_risk', 0)
            if stalking_risk > 0.8:
                return 'stalking_emergency'
        
        if emergency_data.get('medical_emergency', False):
            return 'medical_emergency'
        
        if emergency_data.get('accident_detected', False):
            return 'accident_response'
        
        # Check risk-based determination
        if risk_score > 0.9:
            return 'crime_incident'
        elif risk_score > 0.8:
            return 'high_risk_situation'
        else:
            return 'safety_emergency'
    
    def _execute_emergency_protocol(self, emergency_id: str,
                                   emergency_type: str,
                                   emergency_data: Dict) -> Dict:
        """Execute specific emergency protocol"""
        protocol = self.response_protocols.get(emergency_type, {})
        
        result = {
            'activated_protocols': [emergency_type],
            'notified_services': [],
            'notified_contacts': [],
            'actions_taken': [],
            'verification_required': True,
            'multi_agency': protocol.get('services', []) if protocol else []
        }
        
        # Get user information
        user_id = emergency_data.get('user_id', 'unknown')
        user_contacts = self.emergency_contacts.get(user_id, [])
        
        # Phase 1: Verification
        verification_result = self._verify_emergency(
            emergency_id, emergency_type, emergency_data
        )
        
        if verification_result.get('verified', False):
            result['actions_taken'].append('emergency_verified')
            result['verification_method'] = verification_result['method']
            
            # Phase 2: Service notification
            services_to_notify = protocol.get('services', [])
            for service in services_to_notify:
                if service in self.emergency_services:
                    notification_result = self._notify_emergency_service(
                        service, emergency_id, emergency_data
                    )
                    
                    if notification_result.get('notified', False):
                        result['notified_services'].append(service)
                        result['actions_taken'].append(f'notified_{service}')
            
            # Phase 3: Contact notification
            for contact in user_contacts[:self.config['max_emergency_contacts']]:
                contact_result = self._notify_emergency_contact(
                    contact, emergency_id, emergency_data
                )
                
                if contact_result.get('notified', False):
                    result['notified_contacts'].append(contact.name)
                    result['actions_taken'].append(f'notified_contact_{contact.name}')
            
            # Phase 4: User instructions
            instructions_result = self._provide_user_instructions(
                emergency_id, emergency_type, emergency_data
            )
            
            if instructions_result.get('delivered', False):
                result['actions_taken'].append('user_instructions_delivered')
        
        else:
            # Emergency not verified
            result['verification_required'] = True
            result['actions_taken'].append('verification_failed')
            result['verification_details'] = verification_result
        
        return result
    
    def _verify_emergency(self, emergency_id: str,
                         emergency_type: str,
                         emergency_data: Dict) -> Dict:
        """Verify emergency situation"""
        verification_methods = []
        
        # Method 1: Check risk score
        risk_score = emergency_data.get('risk_score', 0)
        if risk_score > self.config['emergency_threshold']:
            verification_methods.append('high_risk_score')
        
        # Method 2: Check multiple anomalies
        anomalies = emergency_data.get('anomalies', {})
        anomaly_count = 0
        
        if anomalies.get('stalking_detected', False):
            anomaly_count += 2  # Stalking counts double
        
        if anomalies.get('route_deviation', {}).get('score', 0) > 0.7:
            anomaly_count += 1
        
        if anomalies.get('speed_anomaly', {}).get('score', 0) > 0.8:
            anomaly_count += 1
        
        if anomaly_count >= 2:
            verification_methods.append('multiple_anomalies')
        
        # Method 3: Check user response (if available)
        user_response = emergency_data.get('user_response', {})
        if user_response.get('emergency_confirmed', False):
            verification_methods.append('user_confirmation')
        
        # Method 4: Check location risk
        location_risk = emergency_data.get('crime_density', 0)
        if location_risk > 0.8:
            verification_methods.append('high_risk_location')
        
        # Determine verification result
        if len(verification_methods) >= 2:
            return {
                'verified': True,
                'method': 'multiple_indicators',
                'indicators': verification_methods,
                'confidence': min(1.0, len(verification_methods) / 4.0)
            }
        elif len(verification_methods) == 1 and 'user_confirmation' in verification_methods:
            return {
                'verified': True,
                'method': 'user_confirmation',
                'indicators': verification_methods,
                'confidence': 0.9
            }
        else:
            return {
                'verified': False,
                'method': 'insufficient_evidence',
                'indicators': verification_methods,
                'confidence': len(verification_methods) / 4.0
            }
    
    def _notify_emergency_service(self, service: str,
                                 emergency_id: str,
                                 emergency_data: Dict) -> Dict:
        """Notify emergency service"""
        try:
            service_info = self.emergency_services.get(service, {})
            
            if not service_info.get('available', True):
                return {
                    'notified': False,
                    'error': f'Service {service} not available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Prepare notification message
            message = self._prepare_service_message(
                service, emergency_id, emergency_data
            )
            
            # In production, this would actually send the notification
            # For now, simulate sending
            
            self.logger.critical(
                f"Notifying {service} for emergency {emergency_id}"
            )
            
            # Simulate response
            response_received = np.random.random() > 0.1  # 90% success rate
            
            return {
                'notified': True,
                'service': service,
                'contact_number': service_info['contact_number'],
                'message_sent': True,
                'response_received': response_received,
                'response_time_minutes': service_info.get('response_time_minutes', 15),
                'timestamp': datetime.now().isoformat(),
                'simulated': True  # Remove in production
            }
            
        except Exception as e:
            self.logger.error(f"Error notifying service {service}: {e}")
            return {
                'notified': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_service_message(self, service: str,
                               emergency_id: str,
                               emergency_data: Dict) -> str:
        """Prepare message for emergency service"""
        template = self.communication_templates.get('police_alert', '')
        
        # Fill template with data
        message = template.format(
            user_name=emergency_data.get('user_name', 'Unknown User'),
            user_id=emergency_data.get('user_id', 'unknown'),
            emergency_type=self._determine_emergency_type(emergency_data),
            latitude=emergency_data.get('latitude', 0),
            longitude=emergency_data.get('longitude', 0),
            address=emergency_data.get('address', 'Unknown location'),
            timestamp=datetime.now().isoformat(),
            situation_details=self._get_situation_details(emergency_data),
            risk_score=emergency_data.get('risk_score', 0),
            risk_factors=', '.join(emergency_data.get('risk_factors', [])),
            age=emergency_data.get('age', 'unknown'),
            medical_info=emergency_data.get('medical_info', 'none'),
            emergency_contacts=', '.join(
                [c.name for c in self.emergency_contacts.get(
                    emergency_data.get('user_id', ''), []
                )[:3]]
            ),
            required_response=self._get_required_response(service, emergency_data)
        )
        
        return message
    
    def _get_situation_details(self, emergency_data: Dict) -> str:
        """Get detailed situation description"""
        details = []
        
        anomalies = emergency_data.get('anomalies', {})
        
        if anomalies.get('stalking_detected', False):
            details.append(f"Stalking detected (risk: {anomalies.get('stalking_risk', 0):.2f})")
        
        if anomalies.get('route_deviation', {}).get('score', 0) > 0.5:
            details.append(f"Route deviation detected")
        
        if emergency_data.get('is_night', 0) == 1:
            details.append("Night time situation")
        
        if emergency_data.get('crowd_density', 0.5) < 0.2:
            details.append("Isolated location")
        
        return '; '.join(details) if details else 'Emergency situation detected'
    
    def _get_required_response(self, service: str, emergency_data: Dict) -> str:
        """Get required response for service"""
        if service == 'police':
            return "Immediate police response required. Possible crime or safety threat."
        elif service == 'ambulance':
            return "Medical emergency response required. Possible injury or medical issue."
        elif service == 'women_helpline':
            return "Women's safety emergency. Possible harassment or threat."
        else:
            return "Emergency response required. Situation details above."
    
    def _notify_emergency_contact(self, contact: EmergencyContact,
                                 emergency_id: str,
                                 emergency_data: Dict) -> Dict:
        """Notify emergency contact"""
        try:
            # Prepare message
            message = self._prepare_contact_message(contact, emergency_data)
            
            self.logger.info(
                f"Notifying emergency contact {contact.name} "
                f"for emergency {emergency_id}"
            )
            
            # Simulate notification
            # In production, this would send SMS/email/call
            
            response_probability = 0.8 if contact.available else 0.3
            response_received = np.random.random() < response_probability
            
            # Simulate response time
            response_time = None
            if response_received:
                response_time = np.random.uniform(30, 300)  # 30s to 5 minutes
            
            return {
                'notified': True,
                'contact_name': contact.name,
                'relationship': contact.relationship,
                'contact_method': 'simulated',  # Would be SMS/call/email in production
                'message_sent': True,
                'response_received': response_received,
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error notifying contact {contact.name}: {e}")
            return {
                'notified': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_contact_message(self, contact: EmergencyContact,
                                emergency_data: Dict) -> str:
        """Prepare message for emergency contact"""
        template = self.communication_templates.get('guardian_alert', '')
        
        message = template.format(
            user_name=emergency_data.get('user_name', 'User'),
            location=self._format_location(emergency_data),
            risk_level=self._format_risk_level(emergency_data.get('risk_score', 0)),
            timestamp=datetime.now().isoformat(),
            services_notified=', '.join(
                list(self.emergency_services.keys())[:2]
            ),
            emergency_numbers=', '.join([
                self.emergency_services[s]['contact_number']
                for s in list(self.emergency_services.keys())[:2]
            ]),
            tracking_url=self._generate_tracking_url(emergency_data)
        )
        
        return message
    
    def _format_location(self, emergency_data: Dict) -> str:
        """Format location for display"""
        lat = emergency_data.get('latitude', 0)
        lon = emergency_data.get('longitude', 0)
        address = emergency_data.get('address', '')
        
        if address:
            return address
        else:
            return f"Coordinates: {lat:.4f}, {lon:.4f}"
    
    def _format_risk_level(self, risk_score: float) -> str:
        """Format risk level for display"""
        if risk_score > 0.9:
            return "CRITICAL"
        elif risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.5:
            return "MEDIUM"
        else:
            return "ELEVATED"
    
    def _generate_tracking_url(self, emergency_data: Dict) -> str:
        """Generate tracking URL (simulated)"""
        lat = emergency_data.get('latitude', 0)
        lon = emergency_data.get('longitude', 0)
        
        return f"https://shield-safety.com/track?lat={lat}&lon={lon}&emergency=true"
    
    def _provide_user_instructions(self, emergency_id: str,
                                 emergency_type: str,
                                 emergency_data: Dict) -> Dict:
        """Provide instructions to user"""
        try:
            instructions = self._get_user_instructions(emergency_type)
            
            # In production, this would push to user's device
            self.logger.info(
                f"Providing emergency instructions to user for {emergency_id}"
            )
            
            return {
                'delivered': True,
                'emergency_type': emergency_type,
                'instructions': instructions,
                'delivery_method': 'push_notification',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error providing user instructions: {e}")
            return {
                'delivered': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_user_instructions(self, emergency_type: str) -> str:
        """Get user instructions for emergency type"""
        template = self.communication_templates.get('user_instructions', '')
        
        instructions_map = {
            'stalking_emergency': """
            1. Do not confront the stalker
            2. Move to a well-lit, populated area
            3. Enter a store, restaurant, or public building
            4. Call emergency services if safe to do so
            5. Keep your phone visible but secure
            """,
            'medical_emergency': """
            1. Stay where you are if safe
            2. Do not move if injured
            3. Keep your phone accessible
            4. Describe your symptoms if possible
            5. Wait for medical assistance
            """,
            'crime_incident': """
            1. Move to a safe location immediately
            2. Do not engage with perpetrators
            3. Find a secure place to hide if necessary
            4. Call emergency services when safe
            5. Provide location details clearly
            """,
            'default': """
            1. Stay calm and assess your surroundings
            2. Move to a safe location if possible
            3. Keep your phone accessible
            4. Wait for emergency services
            5. Follow instructions from authorities
            """
        }
        
        safety_instructions = instructions_map.get(
            emergency_type, 
            instructions_map['default']
        )
        
        # Get average ETA for services
        eta_minutes = self._calculate_average_eta(emergency_type)
        
        message = template.format(
            emergency_type=emergency_type.replace('_', ' ').title(),
            safety_instructions=safety_instructions,
            eta_minutes=eta_minutes
        )
        
        return message
    
    def _calculate_average_eta(self, emergency_type: str) -> int:
        """Calculate average ETA for emergency services"""
        protocol = self.response_protocols.get(emergency_type, {})
        services = protocol.get('services', [])
        
        if not services:
            return 15  # Default
        
        etas = []
        for service in services:
            if service in self.emergency_services:
                etas.append(self.emergency_services[service].get('response_time_minutes', 15))
        
        return int(np.mean(etas)) if etas else 15
    
    def _get_next_steps(self, emergency_type: str, 
                       current_phase: EmergencyPhase) -> List[Dict]:
        """Get next steps for emergency response"""
        protocol = self.response_protocols.get(emergency_type, {})
        actions = protocol.get('actions', [])
        checklist = protocol.get('checklist', [])
        
        next_steps = []
        
        # Map phase to step indices
        phase_map = {
            EmergencyPhase.DETECTION: (0, 1),
            EmergencyPhase.VERIFICATION: (1, 2),
            EmergencyPhase.ALERT: (2, 3),
            EmergencyPhase.RESPONSE: (3, 4),
            EmergencyPhase.COORDINATION: (4, len(actions))
        }
        
        start_idx, end_idx = phase_map.get(current_phase, (0, len(actions)))
        
        for i in range(start_idx, min(end_idx, len(actions))):
            next_steps.append({
                'step': i + 1,
                'action': actions[i],
                'priority': 'high' if i < 2 else 'medium'
            })
        
        # Add checklist items
        if current_phase == EmergencyPhase.RESPONSE:
            for j, item in enumerate(checklist):
                next_steps.append({
                    'step': f'C{j+1}',
                    'action': item,
                    'priority': 'medium',
                    'type': 'checklist'
                })
        
        return next_steps
    
    def _start_emergency_monitoring(self, emergency_id: str):
        """Start monitoring emergency situation"""
        import threading
        import time
        
        def monitor_emergency():
            try:
                while emergency_id in self.active_emergencies:
                    emergency = self.active_emergencies[emergency_id]
                    
                    # Check if emergency should be escalated
                    if self._should_escalate_emergency(emergency):
                        self._escalate_emergency(emergency_id)
                    
                    # Check timeout
                    start_time = datetime.fromisoformat(emergency['start_time'])
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    if elapsed > 3600:  # 1 hour timeout
                        self.logger.warning(
                            f"Emergency {emergency_id} has been active for over 1 hour"
                        )
                        self._initiate_timeout_protocol(emergency_id)
                    
                    # Update status
                    self._update_emergency_status(emergency_id)
                    
                    time.sleep(30)  # Check every 30 seconds
                    
            except Exception as e:
                self.logger.error(f"Error in emergency monitoring: {e}")
        
        # Start monitoring thread
        thread = threading.Thread(target=monitor_emergency, daemon=True)
        thread.start()
    
    def _should_escalate_emergency(self, emergency: Dict) -> bool:
        """Check if emergency should be escalated"""
        # Check if no services have responded
        notified_services = emergency.get('notified_services', [])
        if not notified_services:
            return True
        
        # Check time since activation
        start_time = datetime.fromisoformat(emergency['start_time'])
        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        
        # Escalate if no response after 10 minutes
        if elapsed_minutes > 10:
            # Check if any service has responded
            # This would check actual responses in production
            return True
        
        return False
    
    def _escalate_emergency(self, emergency_id: str):
        """Escalate emergency response"""
        try:
            emergency = self.active_emergencies[emergency_id]
            
            self.logger.critical(
                f"Escalating emergency {emergency_id}. "
                f"Current phase: {emergency['phase']}"
            )
            
            # Move to next phase
            current_phase = EmergencyPhase(emergency['phase'])
            
            phase_order = [
                EmergencyPhase.DETECTION,
                EmergencyPhase.VERIFICATION,
                EmergencyPhase.ALERT,
                EmergencyPhase.RESPONSE,
                EmergencyPhase.COORDINATION,
                EmergencyPhase.RESOLUTION
            ]
            
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                next_phase = phase_order[current_idx + 1]
                emergency['phase'] = next_phase.value
                
                # Take escalation actions
                escalation_actions = self._get_escalation_actions(next_phase)
                emergency['actions_taken'].extend(escalation_actions)
                
                self.logger.critical(
                    f"Emergency {emergency_id} escalated to {next_phase.value}"
                )
            
        except Exception as e:
            self.logger.error(f"Error escalating emergency: {e}")
    
    def _get_escalation_actions(self, phase: EmergencyPhase) -> List[str]:
        """Get actions for escalation phase"""
        actions_map = {
            EmergencyPhase.COORDINATION: [
                'activate_multi_agency_coordination',
                'establish_incident_command',
                'deploy_additional_resources'
            ],
            EmergencyPhase.RESPONSE: [
                'dispatch_emergency_teams',
                'establish_perimeter',
                'initiate_rescue_operations'
            ]
        }
        
        return actions_map.get(phase, ['escalation_protocol_activated'])
    
    def _initiate_timeout_protocol(self, emergency_id: str):
        """Initiate timeout protocol for long-running emergency"""
        try:
            emergency = self.active_emergencies[emergency_id]
            
            self.logger.critical(
                f"Initiating timeout protocol for emergency {emergency_id}"
            )
            
            # Mark for resolution
            emergency['phase'] = EmergencyPhase.RESOLUTION.value
            emergency['status'] = 'timeout'
            emergency['actions_taken'].append('timeout_protocol_activated')
            
            # Attempt final contact
            self._attempt_final_contact(emergency)
            
            # Schedule resolution
            self._schedule_emergency_resolution(emergency_id)
            
        except Exception as e:
            self.logger.error(f"Error in timeout protocol: {e}")
    
    def _attempt_final_contact(self, emergency: Dict):
        """Attempt final contact before resolution"""
        user_id = emergency['user_id']
        contacts = self.emergency_contacts.get(user_id, [])
        
        for contact in contacts[:2]:  # Try primary contacts
            self.logger.info(
                f"Attempting final contact with {contact.name} "
                f"for emergency {emergency['emergency_id']}"
            )
    
    def _schedule_emergency_resolution(self, emergency_id: str):
        """Schedule emergency for resolution"""
        import threading
        import time
        
        def resolve_emergency():
            time.sleep(300)  # Wait 5 minutes
            
            if emergency_id in self.active_emergencies:
                self.resolve_emergency(emergency_id, 'timeout_auto_resolution')
        
        thread = threading.Thread(target=resolve_emergency, daemon=True)
        thread.start()
    
    def _update_emergency_status(self, emergency_id: str):
        """Update emergency status"""
        # This would update based on real-time information
        # For now, just log
        pass
    
    def resolve_emergency(self, emergency_id: str, resolution_type: str):
        """Resolve an emergency"""
        try:
            if emergency_id not in self.active_emergencies:
                self.logger.warning(f"Emergency {emergency_id} not found")
                return
            
            emergency = self.active_emergencies[emergency_id]
            
            # Update emergency record
            emergency['phase'] = EmergencyPhase.RESOLUTION.value
            emergency['status'] = 'resolved'
            emergency['resolution_type'] = resolution_type
            emergency['resolution_time'] = datetime.now().isoformat()
            emergency['actions_taken'].append(f'resolved_{resolution_type}')
            
            # Calculate duration
            start_time = datetime.fromisoformat(emergency['start_time'])
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            emergency['duration_seconds'] = duration
            
            # Move to history
            self.incident_history.append(emergency)
            del self.active_emergencies[emergency_id]
            
            # Generate incident report
            report = self._generate_incident_report(emergency)
            emergency['incident_report'] = report
            
            self.logger.info(
                f"Emergency {emergency_id} resolved. "
                f"Duration: {duration:.0f}s, Type: {resolution_type}"
            )
            
            # Trigger post-incident actions
            self._initiate_post_incident_actions(emergency)
            
        except Exception as e:
            self.logger.error(f"Error resolving emergency: {e}")
    
    def _generate_incident_report(self, emergency: Dict) -> Dict:
        """Generate incident report"""
        return {
            'emergency_id': emergency['emergency_id'],
            'user_id': emergency['user_id'],
            'emergency_type': emergency['emergency_type'],
            'start_time': emergency['start_time'],
            'resolution_time': emergency['resolution_time'],
            'duration_seconds': emergency['duration_seconds'],
            'resolution_type': emergency.get('resolution_type', 'unknown'),
            'services_notified': emergency.get('notified_services', []),
            'contacts_notified': emergency.get('notified_contacts', []),
            'actions_taken': emergency.get('actions_taken', []),
            'location': emergency.get('coordinates', {}),
            'risk_factors': emergency.get('emergency_data', {}).get('risk_factors', []),
            'summary': self._generate_incident_summary(emergency),
            'recommendations': self._generate_recommendations(emergency),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_incident_summary(self, emergency: Dict) -> str:
        """Generate incident summary"""
        emergency_type = emergency['emergency_type']
        duration = emergency.get('duration_seconds', 0)
        services = len(emergency.get('notified_services', []))
        contacts = len(emergency.get('notified_contacts', []))
        
        return (
            f"{emergency_type.replace('_', ' ').title()} emergency "
            f"lasting {duration:.0f} seconds. "
            f"{services} emergency services and {contacts} contacts notified. "
            f"Resolved via {emergency.get('resolution_type', 'standard protocol')}."
        )
    
    def _generate_recommendations(self, emergency: Dict) -> List[str]:
        """Generate post-incident recommendations"""
        recommendations = []
        
        emergency_type = emergency['emergency_type']
        
        if emergency_type == 'stalking_emergency':
            recommendations.extend([
                "Consider filing a police report for documentation",
                "Review and update emergency contact list",
                "Vary daily routines and routes",
                "Consider self-defense training",
                "Install additional safety apps or devices"
            ])
        elif emergency_type == 'medical_emergency':
            recommendations.extend([
                "Schedule follow-up medical examination",
                "Update medical information in profile",
                "Carry emergency medical information",
                "Consider medical alert device",
                "Review emergency response plan with doctor"
            ])
        else:
            recommendations.extend([
                "Review safety settings and preferences",
                "Update emergency contact information",
                "Test emergency features regularly",
                "Share safety plan with trusted contacts",
                "Consider additional safety measures"
            ])
        
        return recommendations
    
    def _initiate_post_incident_actions(self, emergency: Dict):
        """Initiate post-incident actions"""
        # Schedule user follow-up
        self._schedule_user_followup(emergency['user_id'])
        
        # Update user risk profile
        self._update_user_risk_profile(emergency)
        
        # Log for analysis
        self._log_incident_for_analysis(emergency)
    
    def _schedule_user_followup(self, user_id: str):
        """Schedule follow-up with user"""
        # This would schedule a check-in or survey
        self.logger.info(f"Scheduled follow-up for user {user_id}")
    
    def _update_user_risk_profile(self, emergency: Dict):
        """Update user risk profile based on incident"""
        # This would update risk assessment models
        pass
    
    def _log_incident_for_analysis(self, emergency: Dict):
        """Log incident for system analysis"""
        # This would send to analytics or ML training
        pass
    
    def get_active_emergencies(self) -> List[Dict]:
        """Get active emergencies"""
        return [
            {
                'emergency_id': e['emergency_id'],
                'user_id': e['user_id'],
                'emergency_type': e['emergency_type'],
                'phase': e['phase'],
                'start_time': e['start_time'],
                'services_notified': e.get('notified_services', []),
                'status': e['status'],
                'duration_seconds': (
                    datetime.now() - datetime.fromisoformat(e['start_time'])
                ).total_seconds() if 'start_time' in e else 0
            }
            for e in self.active_emergencies.values()
        ]
    
    def get_emergency_history(self, hours: int = 24) -> List[Dict]:
        """Get emergency history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            e for e in self.incident_history
            if datetime.fromisoformat(e['start_time']) > cutoff_time
        ]
        
        return [
            {
                'emergency_id': e['emergency_id'],
                'user_id': e['user_id'],
                'emergency_type': e['emergency_type'],
                'start_time': e['start_time'],
                'resolution_time': e.get('resolution_time'),
                'duration_seconds': e.get('duration_seconds', 0),
                'resolution_type': e.get('resolution_type', 'unknown'),
                'services_notified': e.get('notified_services', [])
            }
            for e in history[-50:]  # Last 50 emergencies
        ]
    
    def add_emergency_contacts(self, user_id: str, contacts: List[EmergencyContact]):
        """Add emergency contacts for a user"""
        self.emergency_contacts[user_id] = contacts
        self.logger.info(f"Added {len(contacts)} emergency contacts for user {user_id}")
    
    def get_emergency_contacts(self, user_id: str) -> List[Dict]:
        """Get emergency contacts for a user"""
        contacts = self.emergency_contacts.get(user_id, [])
        
        return [
            {
                'name': c.name,
                'relationship': c.relationship,
                'phone': c.phone,
                'email': c.email,
                'priority': c.priority,
                'available': c.available
            }
            for c in contacts
        ]
    
    def test_emergency_protocol(self, user_id: str, 
                               protocol_type: str = 'stalking_emergency') -> Dict:
        """Test emergency protocol without actual activation"""
        try:
            self.logger.info(f"Testing emergency protocol for user {user_id}")
            
            # Create test emergency data
            test_data = {
                'user_id': user_id,
                'user_name': 'Test User',
                'risk_score': 0.9,
                'latitude': 28.6139,
                'longitude': 77.2090,
                'address': 'Test Location, New Delhi',
                'anomalies': {
                    'stalking_detected': True,
                    'stalking_risk': 0.85,
                    'route_deviation': {'score': 0.8}
                },
                'is_night': 1,
                'crowd_density': 0.1,
                'crime_density': 0.7
            }
            
            # Run through protocol without actual notifications
            result = {
                'test_mode': True,
                'user_id': user_id,
                'protocol_type': protocol_type,
                'test_started': datetime.now().isoformat(),
                'steps': []
            }
            
            # Step 1: Determine emergency type
            emergency_type = self._determine_emergency_type(test_data)
            result['steps'].append({
                'step': 'determine_emergency_type',
                'result': emergency_type,
                'success': True
            })
            
            # Step 2: Verify emergency
            verification = self._verify_emergency('test_id', emergency_type, test_data)
            result['steps'].append({
                'step': 'verify_emergency',
                'result': verification,
                'success': verification.get('verified', False)
            })
            
            # Step 3: Get protocol
            protocol = self.response_protocols.get(emergency_type, {})
            result['steps'].append({
                'step': 'get_protocol',
                'result': {
                    'services': protocol.get('services', []),
                    'actions': protocol.get('actions', []),
                    'checklist': protocol.get('checklist', [])
                },
                'success': bool(protocol)
            })
            
            # Step 4: Generate messages
            service_message = self._prepare_service_message('police', 'test_id', test_data)
            contact_message = self._prepare_contact_message(
                EmergencyContact(
                    name='Test Contact',
                    relationship='Test',
                    phone='+911234567890'
                ),
                test_data
            )
            
            result['steps'].append({
                'step': 'generate_messages',
                'result': {
                    'service_message_length': len(service_message),
                    'contact_message_length': len(contact_message)
                },
                'success': True
            })
            
            # Step 5: Get user instructions
            instructions = self._get_user_instructions(emergency_type)
            result['steps'].append({
                'step': 'get_user_instructions',
                'result': {
                    'instructions_length': len(instructions),
                    'emergency_type': emergency_type
                },
                'success': True
            })
            
            result['test_completed'] = datetime.now().isoformat()
            result['success'] = all(step['success'] for step in result['steps'])
            
            self.logger.info(f"Emergency protocol test completed: {result['success']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error testing emergency protocol: {e}")
            return {
                'test_mode': True,
                'success': False,
                'error': str(e)
            }