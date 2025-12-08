#!/usr/bin/env python3
"""
Incident Response Playbook
Comprehensive incident response procedures and automation for disaster scenarios
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
import requests
from pathlib import Path

class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IncidentStatus(Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentType(Enum):
    SYSTEM_OUTAGE = "system_outage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    NETWORK_FAILURE = "network_failure"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_BUG = "software_bug"
    CAPACITY_OVERLOAD = "capacity_overload"

@dataclass
class IncidentRecord:
    """Record of an incident"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    incident_type: IncidentType
    status: IncidentStatus
    detected_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    affected_components: List[str] = None
    impact_assessment: str = ""
    root_cause: str = ""
    resolution_steps: List[str] = None
    lessons_learned: str = ""
    assigned_to: str = ""
    stakeholder_notifications: List[str] = None

@dataclass
class EscalationContact:
    """Contact information for escalation"""
    name: str
    role: str
    email: str
    phone: str
    slack_id: str = ""
    primary_contact: bool = False

class IncidentResponseManager:
    """Comprehensive incident response management system"""
    
    def __init__(self, config_file: str = "disaster_recovery/incident_response_config.json"):
        self.incidents: Dict[str, IncidentRecord] = {}
        self.escalation_contacts: Dict[str, List[EscalationContact]] = {}
        self.playbooks: Dict[str, Dict] = {}
        self.logger = self._setup_logging()
        
        # Load configuration
        self._load_config(config_file)
        
        # Initialize communication channels
        self._initialize_communication()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup incident response logging"""
        logger = logging.getLogger("incident_response")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("disaster_recovery/logs").mkdir(parents=True, exist_ok=True)
        
        # Create incident-specific log handler
        handler = logging.FileHandler("disaster_recovery/logs/incident_response.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_file: str) -> None:
        """Load incident response configuration"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Load escalation contacts
                for severity, contacts in config_data.get('escalation_contacts', {}).items():
                    self.escalation_contacts[severity] = [
                        EscalationContact(**contact) for contact in contacts
                    ]
                
                # Load playbooks
                self.playbooks = config_data.get('playbooks', {})
            else:
                self._create_default_config(config_file)
                
        except Exception as e:
            self.logger.error(f"Failed to load incident response config: {e}")
            self._create_default_config(config_file)
    
    def _create_default_config(self, config_file: str) -> None:
        """Create default incident response configuration"""
        default_config = {
            "escalation_contacts": {
                "critical": [
                    {
                        "name": "CTO",
                        "role": "Chief Technology Officer",
                        "email": "cto@company.com",
                        "phone": "+1-555-0001",
                        "slack_id": "@cto",
                        "primary_contact": True
                    },
                    {
                        "name": "Head of Trading",
                        "role": "Trading Operations Lead",
                        "email": "trading@company.com", 
                        "phone": "+1-555-0002",
                        "slack_id": "@trading_lead",
                        "primary_contact": True
                    },
                    {
                        "name": "CEO",
                        "role": "Chief Executive Officer",
                        "email": "ceo@company.com",
                        "phone": "+1-555-0003",
                        "slack_id": "@ceo",
                        "primary_contact": False
                    }
                ],
                "high": [
                    {
                        "name": "Engineering Manager",
                        "role": "Engineering Team Lead",
                        "email": "engineering@company.com",
                        "phone": "+1-555-0004",
                        "slack_id": "@eng_manager",
                        "primary_contact": True
                    },
                    {
                        "name": "Head of Risk",
                        "role": "Risk Management Lead",
                        "email": "risk@company.com",
                        "phone": "+1-555-0005",
                        "slack_id": "@risk_lead",
                        "primary_contact": True
                    }
                ],
                "medium": [
                    {
                        "name": "Technical Lead",
                        "role": "Senior Technical Lead",
                        "email": "tech@company.com",
                        "phone": "+1-555-0006",
                        "slack_id": "@tech_lead",
                        "primary_contact": True
                    }
                ],
                "low": [
                    {
                        "name": "On-Call Engineer",
                        "role": "Duty Engineer",
                        "email": "oncall@company.com",
                        "phone": "+1-555-0007",
                        "slack_id": "@oncall",
                        "primary_contact": True
                    }
                ]
            },
            "playbooks": {
                "system_outage": {
                    "description": "Complete system outage affecting trading operations",
                    "immediate_actions": [
                        "Confirm outage scope and impact",
                        "Activate incident response team",
                        "Notify critical stakeholders",
                        "Implement emergency procedures",
                        "Begin system recovery procedures"
                    ],
                    "investigation_steps": [
                        "Check system logs for error patterns",
                        "Verify infrastructure status",
                        "Test component connectivity",
                        "Identify root cause",
                        "Document findings"
                    ],
                    "recovery_procedures": [
                        "Execute failover procedures",
                        "Verify backup systems",
                        "Restore service functionality",
                        "Conduct system health checks",
                        "Monitor for additional issues"
                    ],
                    "communication_templates": {
                        "initial_alert": "CRITICAL: System outage detected. Trading operations affected. Response team activated.",
                        "status_update": "UPDATE: Investigation ongoing. ETA for resolution: {eta}",
                        "resolution": "RESOLVED: Systems restored. Normal operations resumed. Post-incident review scheduled."
                    }
                },
                "performance_degradation": {
                    "description": "System performance below acceptable thresholds",
                    "immediate_actions": [
                        "Measure current performance metrics",
                        "Identify affected components",
                        "Assess business impact",
                        "Implement temporary mitigations"
                    ],
                    "investigation_steps": [
                        "Analyze performance trends",
                        "Check resource utilization",
                        "Review recent changes",
                        "Identify bottlenecks",
                        "Test scaling options"
                    ],
                    "recovery_procedures": [
                        "Scale affected resources",
                        "Optimize configurations",
                        "Implement performance improvements",
                        "Monitor metrics",
                        "Validate resolution"
                    ]
                },
                "security_breach": {
                    "description": "Suspected or confirmed security incident",
                    "immediate_actions": [
                        "Isolate affected systems",
                        "Preserve evidence",
                        "Notify security team",
                        "Implement containment measures",
                        "Begin forensic analysis"
                    ],
                    "investigation_steps": [
                        "Analyze attack vectors",
                        "Assess data exposure",
                        "Review access logs",
                        "Identify compromised accounts",
                        "Document security timeline"
                    ],
                    "recovery_procedures": [
                        "Remove malicious elements",
                        "Patch vulnerabilities",
                        "Reset compromised credentials",
                        "Restore clean systems",
                        "Implement additional security measures"
                    ],
                    "notification_requirements": [
                        "Legal team within 1 hour",
                        "Regulatory authorities within 24 hours",
                        "Affected customers within 72 hours"
                    ]
                }
            }
        }
        
        # Save default configuration
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Load the default configuration
        self._load_config(config_file)
    
    def _initialize_communication(self) -> None:
        """Initialize communication channels"""
        # Placeholder for communication setup
        # In real implementation, this would initialize:
        # - Slack API connections
        # - Email SMTP configuration
        # - PagerDuty integration
        # - Phone/SMS services
        pass
    
    async def create_incident(self,
                             title: str,
                             description: str,
                             severity: IncidentSeverity,
                             incident_type: IncidentType,
                             affected_components: List[str] = None,
                             detected_by: str = "system") -> str:
        """Create a new incident record"""
        try:
            # Generate unique incident ID
            incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create incident record
            incident = IncidentRecord(
                incident_id=incident_id,
                title=title,
                description=description,
                severity=severity,
                incident_type=incident_type,
                status=IncidentStatus.DETECTED,
                detected_at=datetime.now(),
                affected_components=affected_components or [],
                resolution_steps=[],
                stakeholder_notifications=[]
            )
            
            # Store incident
            self.incidents[incident_id] = incident
            
            # Log incident creation
            self.logger.critical(f"Incident created: {incident_id} - {title} ({severity.value})")
            
            # Trigger immediate response
            await self._trigger_immediate_response(incident)
            
            return incident_id
            
        except Exception as e:
            self.logger.error(f"Failed to create incident: {e}")
            raise
    
    async def _trigger_immediate_response(self, incident: IncidentRecord) -> None:
        """Trigger immediate response procedures for incident"""
        try:
            # Send immediate notifications
            await self._send_incident_notifications(incident)
            
            # Execute playbook immediate actions
            if incident.incident_type.value in self.playbooks:
                playbook = self.playbooks[incident.incident_type.value]
                immediate_actions = playbook.get('immediate_actions', [])
                
                self.logger.info(f"Executing immediate actions for {incident.incident_id}")
                for action in immediate_actions:
                    incident.resolution_steps.append(f"[{datetime.now().isoformat()}] {action}")
                    # In real implementation, these would trigger automated actions
            
            # Update incident status
            incident.status = IncidentStatus.ACKNOWLEDGED
            incident.acknowledged_at = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to trigger immediate response for {incident.incident_id}: {e}")
    
    async def _send_incident_notifications(self, incident: IncidentRecord) -> None:
        """Send notifications to appropriate stakeholders"""
        try:
            severity_key = incident.severity.value
            contacts = self.escalation_contacts.get(severity_key, [])
            
            # Prepare notification message
            message = self._format_incident_notification(incident)
            
            # Send notifications to all contacts for this severity level
            for contact in contacts:
                try:
                    # Send email notification
                    await self._send_email_notification(contact.email, 
                                                       f"INCIDENT: {incident.title}",
                                                       message)
                    
                    # Send Slack notification
                    if contact.slack_id:
                        await self._send_slack_notification(contact.slack_id, message)
                    
                    # For critical incidents, also try phone/SMS
                    if incident.severity == IncidentSeverity.CRITICAL and contact.primary_contact:
                        await self._send_sms_notification(contact.phone, 
                                                         f"CRITICAL INCIDENT: {incident.title}")
                    
                    incident.stakeholder_notifications.append(
                        f"Notified {contact.name} ({contact.role}) at {datetime.now().isoformat()}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to notify {contact.name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to send incident notifications: {e}")
    
    def _format_incident_notification(self, incident: IncidentRecord) -> str:
        """Format incident notification message"""
        message = f"""
🚨 INCIDENT ALERT 🚨

Incident ID: {incident.incident_id}
Title: {incident.title}
Severity: {incident.severity.value.upper()}
Type: {incident.incident_type.value}
Status: {incident.status.value}
Detected: {incident.detected_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{incident.description}

Affected Components:
{', '.join(incident.affected_components) if incident.affected_components else 'Unknown'}

This incident requires immediate attention.
Please acknowledge receipt and begin response procedures.

Incident Dashboard: https://dashboard.company.com/incidents/{incident.incident_id}
        """
        return message.strip()
    
    async def _send_email_notification(self, email: str, subject: str, message: str) -> None:
        """Send email notification"""
        try:
            # Placeholder for email sending
            # In real implementation, this would use SMTP or email service API
            self.logger.info(f"Email sent to {email}: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send email to {email}: {e}")
    
    async def _send_slack_notification(self, slack_id: str, message: str) -> None:
        """Send Slack notification"""
        try:
            # Placeholder for Slack API
            # In real implementation, this would use Slack SDK
            self.logger.info(f"Slack message sent to {slack_id}")
        except Exception as e:
            self.logger.error(f"Failed to send Slack message to {slack_id}: {e}")
    
    async def _send_sms_notification(self, phone: str, message: str) -> None:
        """Send SMS notification"""
        try:
            # Placeholder for SMS service
            # In real implementation, this would use Twilio or similar
            self.logger.info(f"SMS sent to {phone}")
        except Exception as e:
            self.logger.error(f"Failed to send SMS to {phone}: {e}")
    
    async def update_incident_status(self, 
                                   incident_id: str, 
                                   status: IncidentStatus,
                                   update_message: str = "") -> None:
        """Update incident status and notify stakeholders"""
        try:
            if incident_id not in self.incidents:
                raise ValueError(f"Incident {incident_id} not found")
            
            incident = self.incidents[incident_id]
            old_status = incident.status
            incident.status = status
            
            # Add resolution step
            if update_message:
                incident.resolution_steps.append(
                    f"[{datetime.now().isoformat()}] Status: {status.value} - {update_message}"
                )
            
            # Set resolution timestamp if resolved
            if status == IncidentStatus.RESOLVED:
                incident.resolved_at = datetime.now()
            
            self.logger.info(f"Incident {incident_id} status updated: {old_status.value} -> {status.value}")
            
            # Send status update notifications
            await self._send_status_update_notifications(incident, update_message)
            
        except Exception as e:
            self.logger.error(f"Failed to update incident status: {e}")
            raise
    
    async def _send_status_update_notifications(self, 
                                              incident: IncidentRecord, 
                                              update_message: str) -> None:
        """Send status update notifications"""
        try:
            severity_key = incident.severity.value
            contacts = self.escalation_contacts.get(severity_key, [])
            
            # Prepare update message
            message = f"""
📊 INCIDENT UPDATE

Incident ID: {incident.incident_id}
Title: {incident.title}
Status: {incident.status.value.upper()}
Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

Update:
{update_message}

Duration: {self._calculate_incident_duration(incident)}

Incident Dashboard: https://dashboard.company.com/incidents/{incident.incident_id}
            """
            
            # Send to primary contacts only for updates (not everyone)
            primary_contacts = [c for c in contacts if c.primary_contact]
            
            for contact in primary_contacts:
                try:
                    await self._send_email_notification(
                        contact.email,
                        f"UPDATE: {incident.title}",
                        message
                    )
                    
                    if contact.slack_id:
                        await self._send_slack_notification(contact.slack_id, message)
                        
                except Exception as e:
                    self.logger.error(f"Failed to send update to {contact.name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to send status update notifications: {e}")
    
    def _calculate_incident_duration(self, incident: IncidentRecord) -> str:
        """Calculate incident duration"""
        end_time = incident.resolved_at or datetime.now()
        duration = end_time - incident.detected_at
        
        hours = duration.total_seconds() // 3600
        minutes = (duration.total_seconds() % 3600) // 60
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        else:
            return f"{int(minutes)}m"
    
    async def resolve_incident(self, 
                             incident_id: str,
                             resolution_summary: str,
                             root_cause: str = "",
                             lessons_learned: str = "") -> None:
        """Resolve an incident and conduct post-incident activities"""
        try:
            if incident_id not in self.incidents:
                raise ValueError(f"Incident {incident_id} not found")
            
            incident = self.incidents[incident_id]
            
            # Update incident details
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = datetime.now()
            incident.root_cause = root_cause
            incident.lessons_learned = lessons_learned
            
            # Add final resolution step
            incident.resolution_steps.append(
                f"[{datetime.now().isoformat()}] RESOLVED: {resolution_summary}"
            )
            
            self.logger.info(f"Incident {incident_id} resolved: {resolution_summary}")
            
            # Send resolution notifications
            await self._send_resolution_notifications(incident, resolution_summary)
            
            # Schedule post-incident review
            await self._schedule_post_incident_review(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to resolve incident {incident_id}: {e}")
            raise
    
    async def _send_resolution_notifications(self, 
                                           incident: IncidentRecord, 
                                           resolution_summary: str) -> None:
        """Send incident resolution notifications"""
        try:
            severity_key = incident.severity.value
            contacts = self.escalation_contacts.get(severity_key, [])
            
            duration = self._calculate_incident_duration(incident)
            
            message = f"""
✅ INCIDENT RESOLVED

Incident ID: {incident.incident_id}
Title: {incident.title}
Status: RESOLVED
Duration: {duration}
Resolved: {incident.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Resolution Summary:
{resolution_summary}

Root Cause:
{incident.root_cause or 'Under investigation'}

A post-incident review will be scheduled to discuss:
- Timeline analysis
- Improvement opportunities
- Process refinements

Thank you for your response and attention to this incident.

Incident Dashboard: https://dashboard.company.com/incidents/{incident.incident_id}
            """
            
            for contact in contacts:
                try:
                    await self._send_email_notification(
                        contact.email,
                        f"RESOLVED: {incident.title}",
                        message
                    )
                    
                    if contact.slack_id:
                        await self._send_slack_notification(contact.slack_id, message)
                        
                except Exception as e:
                    self.logger.error(f"Failed to send resolution notification to {contact.name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to send resolution notifications: {e}")
    
    async def _schedule_post_incident_review(self, incident: IncidentRecord) -> None:
        """Schedule post-incident review meeting"""
        try:
            # For critical and high severity incidents, schedule immediate review
            if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                review_time = incident.resolved_at + timedelta(days=1)
            else:
                review_time = incident.resolved_at + timedelta(days=7)
            
            self.logger.info(f"Post-incident review scheduled for {incident.incident_id} at {review_time}")
            
            # In real implementation, this would:
            # - Create calendar event
            # - Invite stakeholders
            # - Prepare review materials
            # - Set reminders
            
        except Exception as e:
            self.logger.error(f"Failed to schedule post-incident review: {e}")
    
    def get_incident_status(self, incident_id: str) -> Optional[Dict]:
        """Get current status of an incident"""
        if incident_id not in self.incidents:
            return None
        
        incident = self.incidents[incident_id]
        return {
            'incident': asdict(incident),
            'duration': self._calculate_incident_duration(incident),
            'timeline': incident.resolution_steps
        }
    
    def get_active_incidents(self) -> List[Dict]:
        """Get all active incidents"""
        active_statuses = [
            IncidentStatus.DETECTED,
            IncidentStatus.ACKNOWLEDGED,
            IncidentStatus.INVESTIGATING,
            IncidentStatus.MITIGATING
        ]
        
        active_incidents = [
            asdict(incident) for incident in self.incidents.values()
            if incident.status in active_statuses
        ]
        
        return active_incidents
    
    def get_incident_metrics(self, days: int = 30) -> Dict:
        """Get incident metrics for the specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_incidents = [
            incident for incident in self.incidents.values()
            if incident.detected_at >= cutoff_date
        ]
        
        if not recent_incidents:
            return {"message": "No incidents in the specified period"}
        
        # Calculate metrics
        total_incidents = len(recent_incidents)
        resolved_incidents = [i for i in recent_incidents if i.status == IncidentStatus.RESOLVED]
        
        # Average resolution time
        resolution_times = [
            (i.resolved_at - i.detected_at).total_seconds() / 3600
            for i in resolved_incidents if i.resolved_at
        ]
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Severity breakdown
        severity_counts = {}
        for severity in IncidentSeverity:
            severity_counts[severity.value] = len([
                i for i in recent_incidents if i.severity == severity
            ])
        
        # Type breakdown
        type_counts = {}
        for inc_type in IncidentType:
            type_counts[inc_type.value] = len([
                i for i in recent_incidents if i.incident_type == inc_type
            ])
        
        return {
            'period_days': days,
            'total_incidents': total_incidents,
            'resolved_incidents': len(resolved_incidents),
            'resolution_rate': len(resolved_incidents) / total_incidents if total_incidents > 0 else 0,
            'average_resolution_time_hours': round(avg_resolution_time, 2),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts
        }

if __name__ == "__main__":
    # Example usage
    async def main():
        irm = IncidentResponseManager()
        
        # Create a critical incident
        incident_id = await irm.create_incident(
            title="Trading Engine Complete Outage",
            description="Primary trading engine is completely unresponsive. All trading operations halted.",
            severity=IncidentSeverity.CRITICAL,
            incident_type=IncidentType.SYSTEM_OUTAGE,
            affected_components=["trading_engine", "order_management", "risk_manager"]
        )
        
        print(f"Created incident: {incident_id}")
        
        # Simulate incident progression
        await asyncio.sleep(2)
        
        await irm.update_incident_status(
            incident_id,
            IncidentStatus.INVESTIGATING,
            "Response team activated. Investigating root cause."
        )
        
        await asyncio.sleep(2)
        
        await irm.update_incident_status(
            incident_id,
            IncidentStatus.MITIGATING,
            "Root cause identified: database connection pool exhausted. Implementing failover."
        )
        
        await asyncio.sleep(2)
        
        await irm.resolve_incident(
            incident_id,
            "Failover to backup trading engine successful. All systems operational.",
            "Database connection pool configuration insufficient for peak load",
            "Need to implement better connection pool monitoring and auto-scaling"
        )
        
        # Get incident metrics
        metrics = irm.get_incident_metrics()
        print(f"Incident metrics: {metrics}")
    
    # asyncio.run(main())