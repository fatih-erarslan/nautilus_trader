"""
Audit Trail - Comprehensive logging and compliance tracking
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

class AuditTrail:
    """
    Comprehensive audit trail for regulatory compliance
    Records all trading activities and system events
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.events = []
        self.event_count = 0
        self.max_events = 100000  # Keep last 100k events in memory
        
    async def initialize(self):
        """Initialize audit trail"""
        self.logger.info("Audit Trail initialized")
        
    async def record_event(self, event: Dict[str, Any]):
        """Record audit event"""
        try:
            # Create comprehensive audit record
            audit_record = {
                'event_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'event_type': event.get('event', 'UNKNOWN'),
                'component': event.get('component', 'SYSTEM'),
                'data': event.get('data', {}),
                'user_id': event.get('user_id'),
                'session_id': event.get('session_id'),
                'ip_address': event.get('ip_address'),
                'sequence_number': self.event_count
            }
            
            # Add to memory store
            self.events.append(audit_record)
            self.event_count += 1
            
            # Maintain memory limit
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
                
            # Log critical events
            if event.get('event') in ['KILL_SWITCH_ACTIVATED', 'EMERGENCY_STOP', 'ORDER_REJECTED']:
                self.logger.critical(f"Critical audit event: {json.dumps(audit_record)}")
            else:
                self.logger.debug(f"Audit event recorded: {audit_record['event_id']}")
                
        except Exception as e:
            self.logger.error(f"Failed to record audit event: {e}")
            
    async def get_records(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         event_type: Optional[str] = None,
                         component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit records with filtering"""
        filtered_events = self.events
        
        # Apply time filter
        if start_time:
            filtered_events = [
                e for e in filtered_events 
                if datetime.fromisoformat(e['timestamp']) >= start_time
            ]
            
        if end_time:
            filtered_events = [
                e for e in filtered_events 
                if datetime.fromisoformat(e['timestamp']) <= end_time
            ]
            
        # Apply event type filter
        if event_type:
            filtered_events = [
                e for e in filtered_events 
                if e['event_type'] == event_type
            ]
            
        # Apply component filter
        if component:
            filtered_events = [
                e for e in filtered_events 
                if e['component'] == component
            ]
            
        return filtered_events
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        if not self.events:
            return {'total_events': 0}
            
        # Calculate statistics
        event_types = {}
        components = {}
        
        for event in self.events:
            event_type = event['event_type']
            component = event['component']
            
            event_types[event_type] = event_types.get(event_type, 0) + 1
            components[component] = components.get(component, 0) + 1
            
        return {
            'total_events': len(self.events),
            'event_types': event_types,
            'components': components,
            'latest_event': self.events[-1]['timestamp'] if self.events else None
        }