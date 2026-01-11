import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import heapq
from loguru import logger
from ...utils.logger import setup_logger
from ...config.constants import RiskLevel, InterventionType


class PriorityLevel(Enum):
    """Priority levels for intervention handling"""
    CRITICAL = 10
    HIGH = 7
    MEDIUM = 5
    LOW = 3
    MONITORING = 1


@dataclass
class PriorityItem:
    """Item in priority queue"""
    priority: int
    timestamp: datetime
    user_id: str
    intervention_id: str
    risk_score: float
    data: Dict
    queue_position: int = 0
    
    def __lt__(self, other):
        # Higher priority first, then earlier timestamp
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


class PriorityHandler:
    """
    Priority handler for managing and processing interventions
    Implements priority queues and resource allocation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_concurrent_interventions': 10,
            'resource_allocation_strategy': 'dynamic',
            'priority_decay_rate': 0.1,  # per minute
            'reassessment_interval_seconds': 30,
            'emergency_capacity': 3,
            'normal_capacity': 7
        }
        
        self.logger = setup_logger(__name__)
        
        # Priority queues
        self.priority_queue = []  # Min-heap (using heapq)
        self.active_interventions: Dict[str, PriorityItem] = {}
        self.completed_interventions: List[PriorityItem] = []
        
        # Resource tracking
        self.available_resources = {
            'emergency_handlers': 2,
            'intervention_coordinators': 3,
            'monitoring_slots': 20,
            'notification_channels': 10
        }
        
        self.used_resources = {
            'emergency_handlers': 0,
            'intervention_coordinators': 0,
            'monitoring_slots': 0,
            'notification_channels': 0
        }
        
        # Statistics
        self.statistics = {
            'total_processed': 0,
            'avg_processing_time': 0,
            'max_queue_length': 0,
            'resource_utilization': {}
        }
    
    def add_intervention(self, user_id: str, intervention_id: str,
                        risk_score: float, intervention_data: Dict) -> str:
        """
        Add intervention to priority queue
        Returns queue position
        """
        try:
            # Calculate priority
            priority = self._calculate_priority(risk_score, intervention_data)
            
            # Create priority item
            item = PriorityItem(
                priority=priority,
                timestamp=datetime.now(),
                user_id=user_id,
                intervention_id=intervention_id,
                risk_score=risk_score,
                data=intervention_data
            )
            
            # Add to priority queue
            heapq.heappush(self.priority_queue, item)
            
            # Update queue position
            self._update_queue_positions()
            
            # Update statistics
            self.statistics['max_queue_length'] = max(
                self.statistics['max_queue_length'],
                len(self.priority_queue)
            )
            
            queue_position = item.queue_position
            
            self.logger.info(
                f"Added intervention {intervention_id} for user {user_id} "
                f"with priority {priority} (position: {queue_position})"
            )
            
            # Trigger processing if resources available
            self._process_queue()
            
            return f"queued_at_position_{queue_position}"
            
        except Exception as e:
            self.logger.error(f"Error adding intervention to queue: {e}")
            raise
    
    def _calculate_priority(self, risk_score: float, 
                          intervention_data: Dict) -> int:
        """Calculate priority score (1-10)"""
        base_priority = int(risk_score * 8) + 1  # Map 0-1 to 1-9
        
        # Adjust based on intervention type
        intervention_type = intervention_data.get('intervention_type', '')
        if 'emergency' in intervention_type.lower():
            base_priority += 2
        elif 'police' in intervention_type.lower():
            base_priority += 3
        
        # Adjust based on time of day
        current_hour = datetime.now().hour
        if 22 <= current_hour <= 6:  # Night
            base_priority += 1
        
        # Adjust based on user history
        user_id = intervention_data.get('user_id', '')
        if self._has_recent_emergencies(user_id):
            base_priority += 1
        
        # Cap at maximum
        return min(10, base_priority)
    
    def _has_recent_emergencies(self, user_id: str) -> bool:
        """Check if user has recent emergency interventions"""
        recent_cutoff = datetime.now() - timedelta(hours=24)
        
        recent_emergencies = [
            item for item in self.completed_interventions[-100:]  # Check last 100
            if item.user_id == user_id and 
            item.data.get('intervention_type', '').lower().contains('emergency') and
            item.timestamp > recent_cutoff
        ]
        
        return len(recent_emergencies) > 0
    
    def _update_queue_positions(self):
        """Update queue positions for all items"""
        # Create sorted list (highest priority first)
        sorted_items = sorted(self.priority_queue, reverse=True)
        
        for position, item in enumerate(sorted_items, 1):
            item.queue_position = position
    
    def _process_queue(self):
        """Process items from the priority queue"""
        try:
            available_capacity = self._get_available_capacity()
            
            while (self.priority_queue and 
                   len(self.active_interventions) < self.config['max_concurrent_interventions'] and
                   available_capacity > 0):
                
                # Get highest priority item
                item = heapq.heappop(self.priority_queue)
                
                # Check if resources available
                if self._allocate_resources(item):
                    # Add to active interventions
                    self.active_interventions[item.intervention_id] = item
                    
                    self.logger.info(
                        f"Started processing intervention {item.intervention_id} "
                        f"(priority: {item.priority})"
                    )
                    
                    # Update available capacity
                    available_capacity = self._get_available_capacity()
                    
                    # Trigger actual processing (would be async in production)
                    self._start_intervention_processing(item)
                else:
                    # Not enough resources, put back in queue
                    heapq.heappush(self.priority_queue, item)
                    break
            
            # Update queue positions
            self._update_queue_positions()
            
        except Exception as e:
            self.logger.error(f"Error processing queue: {e}")
    
    def _get_available_capacity(self) -> int:
        """Get available processing capacity"""
        max_capacity = self.config['max_concurrent_interventions']
        current_active = len(self.active_interventions)
        return max(0, max_capacity - current_active)
    
    def _allocate_resources(self, item: PriorityItem) -> bool:
        """Allocate resources for intervention"""
        required_resources = self._estimate_resource_requirements(item)
        
        # Check if resources available
        for resource, amount in required_resources.items():
            available = (self.available_resources.get(resource, 0) - 
                        self.used_resources.get(resource, 0))
            if available < amount:
                return False
        
        # Allocate resources
        for resource, amount in required_resources.items():
            self.used_resources[resource] += amount
        
        return True
    
    def _estimate_resource_requirements(self, item: PriorityItem) -> Dict[str, int]:
        """Estimate resource requirements for intervention"""
        requirements = {
            'monitoring_slots': 1,
            'notification_channels': 0
        }
        
        # Adjust based on priority
        if item.priority >= PriorityLevel.CRITICAL.value:
            requirements['emergency_handlers'] = 1
            requirements['notification_channels'] = 2
        elif item.priority >= PriorityLevel.HIGH.value:
            requirements['intervention_coordinators'] = 1
            requirements['notification_channels'] = 1
        
        # Adjust based on intervention type
        intervention_type = item.data.get('intervention_type', '')
        if 'emergency' in intervention_type.lower():
            requirements['emergency_handlers'] = requirements.get('emergency_handlers', 0) + 1
            requirements['notification_channels'] += 1
        
        return requirements
    
    def _start_intervention_processing(self, item: PriorityItem):
        """Start processing an intervention"""
        # This would trigger actual intervention processing
        # For now, simulate with a timer that will complete the intervention
        
        import threading
        import time
        
        def process_intervention():
            try:
                # Simulate processing time based on priority
                # Higher priority = faster processing
                processing_time = max(1, 10 - item.priority)  # 1-9 seconds
                time.sleep(processing_time)
                
                # Complete intervention
                self.complete_intervention(item.intervention_id, success=True)
                
            except Exception as e:
                self.logger.error(f"Error in intervention processing: {e}")
                self.complete_intervention(item.intervention_id, success=False)
        
        # Start processing thread
        thread = threading.Thread(target=process_intervention, daemon=True)
        thread.start()
    
    def complete_intervention(self, intervention_id: str, success: bool = True):
        """Complete an intervention and free resources"""
        try:
            if intervention_id not in self.active_interventions:
                self.logger.warning(f"Intervention {intervention_id} not found in active interventions")
                return
            
            item = self.active_interventions[intervention_id]
            
            # Free resources
            self._free_resources(item)
            
            # Update item with completion data
            item.data['completed_at'] = datetime.now().isoformat()
            item.data['success'] = success
            item.data['processing_time'] = (
                datetime.now() - item.timestamp
            ).total_seconds()
            
            # Move to completed
            self.completed_interventions.append(item)
            del self.active_interventions[intervention_id]
            
            # Update statistics
            self.statistics['total_processed'] += 1
            
            processing_time = item.data['processing_time']
            current_avg = self.statistics['avg_processing_time']
            total_processed = self.statistics['total_processed']
            
            # Update moving average
            self.statistics['avg_processing_time'] = (
                (current_avg * (total_processed - 1) + processing_time) / 
                total_processed
            )
            
            self.logger.info(
                f"Completed intervention {intervention_id} "
                f"(success: {success}, time: {processing_time:.2f}s)"
            )
            
            # Process next items in queue
            self._process_queue()
            
        except Exception as e:
            self.logger.error(f"Error completing intervention: {e}")
    
    def _free_resources(self, item: PriorityItem):
        """Free resources used by intervention"""
        allocated_resources = self._estimate_resource_requirements(item)
        
        for resource, amount in allocated_resources.items():
            self.used_resources[resource] = max(
                0, self.used_resources.get(resource, 0) - amount
            )
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        queue_items = []
        
        # Create sorted view of queue
        sorted_queue = sorted(self.priority_queue, reverse=True)
        
        for item in sorted_queue[:20]:  # Top 20 items
            queue_items.append({
                'intervention_id': item.intervention_id,
                'user_id': item.user_id,
                'priority': item.priority,
                'queue_position': item.queue_position,
                'waiting_time': (datetime.now() - item.timestamp).total_seconds(),
                'risk_score': item.risk_score,
                'intervention_type': item.data.get('intervention_type', 'unknown')
            })
        
        return {
            'queue_length': len(self.priority_queue),
            'active_interventions': len(self.active_interventions),
            'completed_today': len([i for i in self.completed_interventions 
                                  if i.timestamp.date() == datetime.now().date()]),
            'queue_items': queue_items,
            'estimated_wait_times': self._estimate_wait_times(),
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    def _estimate_wait_times(self) -> Dict[str, float]:
        """Estimate wait times for different priority levels"""
        wait_times = {}
        
        # Calculate average processing time by priority
        priority_groups = {}
        for item in self.completed_interventions[-100:]:  # Last 100 completed
            priority = item.priority
            processing_time = item.data.get('processing_time', 0)
            
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(processing_time)
        
        # Calculate averages
        for priority in range(1, 11):
            times = priority_groups.get(priority, [])
            if times:
                avg_time = np.mean(times)
                wait_times[f'priority_{priority}'] = avg_time
            else:
                wait_times[f'priority_{priority}'] = 5.0  # Default
        
        return wait_times
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization percentages"""
        utilization = {}
        
        for resource, total in self.available_resources.items():
            used = self.used_resources.get(resource, 0)
            if total > 0:
                utilization[resource] = (used / total) * 100
            else:
                utilization[resource] = 0.0
        
        return utilization
    
    def reassess_priorities(self):
        """Reassess priorities of items in queue"""
        try:
            self.logger.info("Reassessing queue priorities")
            
            # Create new queue with reassessed priorities
            new_queue = []
            
            for item in self.priority_queue:
                # Apply priority decay based on wait time
                wait_time_minutes = (datetime.now() - item.timestamp).total_seconds() / 60
                decay_factor = 1 - (self.config['priority_decay_rate'] * wait_time_minutes)
                
                # Apply decay, but ensure minimum priority
                new_priority = max(1, int(item.priority * decay_factor))
                
                # Update item priority
                item.priority = new_priority
                
                # Add to new queue
                heapq.heappush(new_queue, item)
            
            # Replace queue
            self.priority_queue = new_queue
            
            # Update queue positions
            self._update_queue_positions()
            
            self.logger.info(f"Priority reassessment complete. Queue size: {len(self.priority_queue)}")
            
        except Exception as e:
            self.logger.error(f"Error reassessing priorities: {e}")
    
    def get_intervention_status(self, intervention_id: str) -> Dict:
        """Get status of a specific intervention"""
        # Check active interventions
        if intervention_id in self.active_interventions:
            item = self.active_interventions[intervention_id]
            status = "active"
            position = "being_processed"
            wait_time = (datetime.now() - item.timestamp).total_seconds()
        
        # Check queue
        else:
            item_in_queue = None
            for item in self.priority_queue:
                if item.intervention_id == intervention_id:
                    item_in_queue = item
                    break
            
            if item_in_queue:
                item = item_in_queue
                status = "queued"
                position = item.queue_position
                wait_time = (datetime.now() - item.timestamp).total_seconds()
            
            # Check completed
            else:
                completed_item = None
                for comp_item in self.completed_interventions[-100:]:
                    if comp_item.intervention_id == intervention_id:
                        completed_item = comp_item
                        break
                
                if completed_item:
                    item = completed_item
                    status = "completed"
                    position = "completed"
                    wait_time = item.data.get('processing_time', 0)
                else:
                    return {
                        'intervention_id': intervention_id,
                        'status': 'not_found',
                        'error': 'Intervention ID not found in system'
                    }
        
        return {
            'intervention_id': intervention_id,
            'user_id': item.user_id,
            'status': status,
            'queue_position': position,
            'priority': item.priority,
            'risk_score': item.risk_score,
            'wait_time_seconds': wait_time,
            'timestamp': item.timestamp.isoformat(),
            'intervention_type': item.data.get('intervention_type', 'unknown'),
            'estimated_completion': self._estimate_completion_time(item) if status == 'active' else None
        }
    
    def _estimate_completion_time(self, item: PriorityItem) -> str:
        """Estimate completion time for active intervention"""
        # Simple estimation based on priority
        estimated_seconds = max(1, 10 - item.priority)
        completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return completion_time.isoformat()
    
    def cancel_intervention(self, intervention_id: str) -> bool:
        """Cancel an intervention"""
        try:
            # Check active interventions
            if intervention_id in self.active_interventions:
                item = self.active_interventions[intervention_id]
                self._free_resources(item)
                del self.active_interventions[intervention_id]
                
                self.logger.info(f"Cancelled active intervention {intervention_id}")
                return True
            
            # Check queue
            for i, item in enumerate(self.priority_queue):
                if item.intervention_id == intervention_id:
                    # Remove from queue
                    self.priority_queue.pop(i)
                    heapq.heapify(self.priority_queue)  # Re-heapify
                    
                    self._update_queue_positions()
                    
                    self.logger.info(f"Cancelled queued intervention {intervention_id}")
                    return True
            
            self.logger.warning(f"Intervention {intervention_id} not found for cancellation")
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling intervention: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get handler statistics"""
        # Update resource utilization in statistics
        self.statistics['resource_utilization'] = self._calculate_resource_utilization()
        
        # Add current metrics
        current_stats = {
            'current_queue_length': len(self.priority_queue),
            'current_active': len(self.active_interventions),
            'current_resource_usage': self.used_resources,
            'available_resources': self.available_resources,
            'timestamp': datetime.now().isoformat()
        }
        
        return {**self.statistics, **current_stats}
    
    def adjust_resources(self, resource_updates: Dict[str, int]):
        """Adjust available resources"""
        for resource, change in resource_updates.items():
            if resource in self.available_resources:
                new_amount = self.available_resources[resource] + change
                if new_amount >= 0:
                    self.available_resources[resource] = new_amount
                    self.logger.info(f"Adjusted {resource} by {change}. New total: {new_amount}")
                else:
                    self.logger.warning(f"Cannot reduce {resource} below 0")
            else:
                self.logger.warning(f"Unknown resource: {resource}")
        
        # Re-process queue with new resources
        self._process_queue()
    
    def clear_old_interventions(self, hours: int = 24):
        """Clear old interventions from history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Keep only recent interventions
        self.completed_interventions = [
            item for item in self.completed_interventions
            if item.timestamp > cutoff_time
        ]
        
        self.logger.info(f"Cleared old interventions. History now has {len(self.completed_interventions)} items")