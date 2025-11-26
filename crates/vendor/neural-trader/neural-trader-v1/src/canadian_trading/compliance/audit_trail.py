"""
Comprehensive Audit Trail Module for CIRO Compliance
Implements detailed logging and audit trail requirements for Canadian regulatory compliance.
Ensures all trading decisions, executions, and system events are properly recorded.
"""

import json
import logging
import hashlib
import gzip
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import threading
import queue
import pickle

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of events that must be audited"""
    # Trading events
    ORDER_PLACED = "order_placed"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXECUTED = "order_executed"
    ORDER_REJECTED = "order_rejected"
    
    # Decision events
    TRADING_SIGNAL = "trading_signal"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_DECISION = "strategy_decision"
    MANUAL_OVERRIDE = "manual_override"
    
    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    BEST_EXECUTION = "best_execution"
    CONFLICT_CHECK = "conflict_check"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_EVENT = "error_event"
    
    # Client events
    CLIENT_LOGIN = "client_login"
    CLIENT_LOGOUT = "client_logout"
    CLIENT_ONBOARDING = "client_onboarding"
    KYC_UPDATE = "kyc_update"
    
    # Market data events
    MARKET_DATA_RECEIVED = "market_data_received"
    MARKET_DATA_ERROR = "market_data_error"
    NEWS_EVENT = "news_event"
    
    # Administrative events
    USER_ACTION = "user_action"
    PERMISSION_CHANGE = "permission_change"
    AUDIT_ACCESS = "audit_access"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """Immutable audit record structure"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.ORDER_PLACED
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    account_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event details
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance fields
    regulatory_requirement: Optional[str] = None
    approval_status: Optional[str] = None
    approver_id: Optional[str] = None
    
    # System context
    system_version: Optional[str] = None
    component: Optional[str] = None
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Audit metadata
    retention_years: int = 7  # CIRO requirement
    encrypted: bool = False
    compressed: bool = False
    
    def calculate_hash(self) -> str:
        """Calculate cryptographic hash for integrity verification"""
        record_dict = asdict(self)
        # Remove hash from calculation to avoid recursion
        record_dict.pop('hash', None)
        
        # Sort for consistent hashing
        record_str = json.dumps(record_dict, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def to_json(self) -> str:
        """Convert to JSON format"""
        return json.dumps(asdict(self), default=str)
    
    def __post_init__(self):
        """Add calculated fields after initialization"""
        self.hash = self.calculate_hash()
        self.retention_until = self.timestamp + timedelta(days=365 * self.retention_years)


class AuditDatabase:
    """SQLite-based audit database for reliable storage"""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self._init_database()
        self._lock = threading.Lock()
        
    def _init_database(self):
        """Initialize audit database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    client_id TEXT,
                    account_id TEXT,
                    session_id TEXT,
                    event_data TEXT NOT NULL,
                    regulatory_requirement TEXT,
                    approval_status TEXT,
                    approver_id TEXT,
                    system_version TEXT,
                    component TEXT,
                    hostname TEXT,
                    ip_address TEXT,
                    retention_until TIMESTAMP NOT NULL,
                    hash TEXT NOT NULL,
                    compressed BOOLEAN DEFAULT FALSE,
                    encrypted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_client_id ON audit_records(client_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_account_id ON audit_records(account_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_records(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_records(severity)")
            
            # Create integrity verification table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    check_id TEXT PRIMARY KEY,
                    check_timestamp TIMESTAMP NOT NULL,
                    records_verified INTEGER NOT NULL,
                    integrity_failures INTEGER NOT NULL,
                    details TEXT
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper handling"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_record(self, record: AuditRecord) -> bool:
        """Store audit record in database"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Compress event data if large
                    event_data_str = json.dumps(record.event_data, default=str)
                    if len(event_data_str) > 1024:  # Compress if > 1KB
                        event_data_str = gzip.compress(event_data_str.encode()).hex()
                        record.compressed = True
                    
                    conn.execute("""
                        INSERT INTO audit_records (
                            record_id, timestamp, event_type, severity,
                            user_id, client_id, account_id, session_id,
                            event_data, regulatory_requirement, approval_status,
                            approver_id, system_version, component, hostname,
                            ip_address, retention_until, hash, compressed, encrypted
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.record_id, record.timestamp, record.event_type.value,
                        record.severity.value, record.user_id, record.client_id,
                        record.account_id, record.session_id, event_data_str,
                        record.regulatory_requirement, record.approval_status,
                        record.approver_id, record.system_version, record.component,
                        record.hostname, record.ip_address, record.retention_until,
                        record.hash, record.compressed, record.encrypted
                    ))
                    conn.commit()
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to store audit record: {e}")
                return False
    
    def retrieve_records(self, start_date: datetime, end_date: datetime,
                        filters: Optional[Dict[str, Any]] = None) -> List[AuditRecord]:
        """Retrieve audit records within date range"""
        with self._lock:
            query = """
                SELECT * FROM audit_records 
                WHERE timestamp >= ? AND timestamp <= ?
            """
            params = [start_date, end_date]
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        query += f" AND {key} = ?"
                        params.append(value)
            
            query += " ORDER BY timestamp DESC"
            
            records = []
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                for row in cursor:
                    # Reconstruct AuditRecord
                    event_data = row['event_data']
                    if row['compressed']:
                        event_data = gzip.decompress(bytes.fromhex(event_data)).decode()
                    event_data = json.loads(event_data)
                    
                    record = AuditRecord(
                        record_id=row['record_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=AuditEventType(row['event_type']),
                        severity=AuditSeverity(row['severity']),
                        user_id=row['user_id'],
                        client_id=row['client_id'],
                        account_id=row['account_id'],
                        session_id=row['session_id'],
                        event_data=event_data,
                        regulatory_requirement=row['regulatory_requirement'],
                        approval_status=row['approval_status'],
                        approver_id=row['approver_id'],
                        system_version=row['system_version'],
                        component=row['component'],
                        hostname=row['hostname'],
                        ip_address=row['ip_address'],
                        compressed=row['compressed'],
                        encrypted=row['encrypted']
                    )
                    records.append(record)
            
            return records
    
    def verify_integrity(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Verify integrity of audit records"""
        with self._lock:
            records = self.retrieve_records(start_date, end_date)
            
            integrity_check = {
                'check_id': str(uuid.uuid4()),
                'check_timestamp': datetime.utcnow(),
                'records_verified': len(records),
                'integrity_failures': 0,
                'failed_records': []
            }
            
            for record in records:
                calculated_hash = record.calculate_hash()
                if calculated_hash != record.hash:
                    integrity_check['integrity_failures'] += 1
                    integrity_check['failed_records'].append({
                        'record_id': record.record_id,
                        'timestamp': record.timestamp,
                        'expected_hash': record.hash,
                        'calculated_hash': calculated_hash
                    })
            
            # Store integrity check result
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO audit_integrity (
                        check_id, check_timestamp, records_verified,
                        integrity_failures, details
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    integrity_check['check_id'],
                    integrity_check['check_timestamp'],
                    integrity_check['records_verified'],
                    integrity_check['integrity_failures'],
                    json.dumps(integrity_check['failed_records'])
                ))
                conn.commit()
            
            return integrity_check


class AuditTrail:
    """Main audit trail implementation with async support"""
    
    def __init__(self, db_path: str = "audit_trail.db", 
                 buffer_size: int = 1000,
                 flush_interval: int = 5):
        self.database = AuditDatabase(db_path)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.flush_interval = flush_interval
        self.running = False
        self.flush_thread = None
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for audit context"""
        import platform
        import socket
        
        return {
            'hostname': socket.gethostname(),
            'system_version': platform.version(),
            'python_version': platform.python_version()
        }
    
    def start(self):
        """Start the audit trail service"""
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        
        # Log system start
        self.log_event(
            event_type=AuditEventType.SYSTEM_START,
            event_data={'service': 'audit_trail', 'config': {'flush_interval': self.flush_interval}},
            severity=AuditSeverity.INFO
        )
    
    def stop(self):
        """Stop the audit trail service"""
        # Log system stop
        self.log_event(
            event_type=AuditEventType.SYSTEM_STOP,
            event_data={'service': 'audit_trail'},
            severity=AuditSeverity.INFO
        )
        
        self.running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=10)
        
        # Flush remaining records
        self._flush_buffer()
    
    def _flush_worker(self):
        """Background worker to flush buffer periodically"""
        while self.running:
            try:
                asyncio.run(asyncio.sleep(self.flush_interval))
                self._flush_buffer()
            except Exception as e:
                logger.error(f"Flush worker error: {e}")
    
    def _flush_buffer(self):
        """Flush buffered records to database"""
        records_to_store = []
        
        while not self.buffer.empty() and len(records_to_store) < 100:
            try:
                record = self.buffer.get_nowait()
                records_to_store.append(record)
            except queue.Empty:
                break
        
        for record in records_to_store:
            self.database.store_record(record)
    
    def log_event(self, event_type: AuditEventType, 
                  event_data: Dict[str, Any],
                  severity: AuditSeverity = AuditSeverity.INFO,
                  user_id: Optional[str] = None,
                  client_id: Optional[str] = None,
                  account_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  regulatory_requirement: Optional[str] = None) -> str:
        """Log an audit event"""
        record = AuditRecord(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            client_id=client_id,
            account_id=account_id,
            session_id=session_id,
            event_data=event_data,
            regulatory_requirement=regulatory_requirement,
            system_version=self.system_info['system_version'],
            hostname=self.system_info['hostname']
        )
        
        # Add to buffer for async processing
        try:
            self.buffer.put_nowait(record)
        except queue.Full:
            # If buffer is full, store directly
            self.database.store_record(record)
        
        # Log critical events immediately
        if severity == AuditSeverity.CRITICAL:
            self._flush_buffer()
        
        return record.record_id
    
    def log_order_event(self, order: Dict[str, Any], event_type: AuditEventType,
                       user_id: str, client_id: str, account_id: str) -> str:
        """Specialized method for logging order events"""
        event_data = {
            'order_id': order.get('order_id'),
            'symbol': order.get('symbol'),
            'side': order.get('side'),
            'quantity': order.get('quantity'),
            'order_type': order.get('order_type'),
            'price': str(order.get('price', 0)),
            'time_in_force': order.get('time_in_force'),
            'execution_venue': order.get('execution_venue'),
            'status': order.get('status')
        }
        
        # Add execution details if available
        if event_type == AuditEventType.ORDER_EXECUTED:
            event_data.update({
                'execution_price': str(order.get('execution_price', 0)),
                'execution_quantity': order.get('execution_quantity'),
                'commission': str(order.get('commission', 0)),
                'execution_timestamp': order.get('execution_timestamp')
            })
        
        return self.log_event(
            event_type=event_type,
            event_data=event_data,
            user_id=user_id,
            client_id=client_id,
            account_id=account_id,
            regulatory_requirement='CIRO_ORDER_AUDIT'
        )
    
    def log_compliance_event(self, compliance_check: Dict[str, Any],
                            user_id: str, client_id: str) -> str:
        """Log compliance-related events"""
        event_type = (AuditEventType.COMPLIANCE_VIOLATION 
                     if compliance_check.get('violations') 
                     else AuditEventType.COMPLIANCE_CHECK)
        
        severity = (AuditSeverity.WARNING 
                   if compliance_check.get('violations')
                   else AuditSeverity.INFO)
        
        return self.log_event(
            event_type=event_type,
            event_data=compliance_check,
            severity=severity,
            user_id=user_id,
            client_id=client_id,
            regulatory_requirement='CIRO_COMPLIANCE'
        )
    
    def log_risk_assessment(self, risk_data: Dict[str, Any],
                           account_id: str) -> str:
        """Log risk assessment events"""
        return self.log_event(
            event_type=AuditEventType.RISK_ASSESSMENT,
            event_data=risk_data,
            account_id=account_id,
            regulatory_requirement='CIRO_RISK_MANAGEMENT'
        )
    
    def log_trading_signal(self, signal_data: Dict[str, Any],
                          strategy: str, account_id: str) -> str:
        """Log trading signal generation"""
        event_data = {
            'strategy': strategy,
            'signal': signal_data
        }
        
        return self.log_event(
            event_type=AuditEventType.TRADING_SIGNAL,
            event_data=event_data,
            account_id=account_id
        )
    
    def log_manual_override(self, original_action: Dict[str, Any],
                           override_action: Dict[str, Any],
                           reason: str, user_id: str, approver_id: str) -> str:
        """Log manual overrides of automated decisions"""
        event_data = {
            'original_action': original_action,
            'override_action': override_action,
            'override_reason': reason,
            'override_timestamp': datetime.utcnow().isoformat()
        }
        
        return self.log_event(
            event_type=AuditEventType.MANUAL_OVERRIDE,
            event_data=event_data,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            approval_status='approved',
            approver_id=approver_id,
            regulatory_requirement='CIRO_MANUAL_OVERRIDE'
        )
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime,
                             report_type: str = 'comprehensive') -> Dict[str, Any]:
        """Generate audit reports for compliance reviews"""
        records = self.database.retrieve_records(start_date, end_date)
        
        report = {
            'report_id': str(uuid.uuid4()),
            'report_type': report_type,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generation_timestamp': datetime.utcnow().isoformat(),
            'total_records': len(records),
            'summary': self._generate_summary(records),
            'compliance_metrics': self._calculate_compliance_metrics(records),
            'integrity_check': self.database.verify_integrity(start_date, end_date)
        }
        
        if report_type == 'comprehensive':
            report['detailed_analysis'] = self._generate_detailed_analysis(records)
        
        # Log audit report access
        self.log_event(
            event_type=AuditEventType.AUDIT_ACCESS,
            event_data={'report_id': report['report_id'], 'report_type': report_type},
            regulatory_requirement='CIRO_AUDIT_ACCESS'
        )
        
        return report
    
    def _generate_summary(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """Generate summary statistics from audit records"""
        summary = {
            'by_event_type': {},
            'by_severity': {},
            'by_user': {},
            'by_client': {}
        }
        
        for record in records:
            # Count by event type
            event_type = record.event_type.value
            summary['by_event_type'][event_type] = summary['by_event_type'].get(event_type, 0) + 1
            
            # Count by severity
            severity = record.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by user
            if record.user_id:
                summary['by_user'][record.user_id] = summary['by_user'].get(record.user_id, 0) + 1
            
            # Count by client
            if record.client_id:
                summary['by_client'][record.client_id] = summary['by_client'].get(record.client_id, 0) + 1
        
        return summary
    
    def _calculate_compliance_metrics(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """Calculate compliance-specific metrics"""
        compliance_records = [r for r in records if r.event_type in [
            AuditEventType.COMPLIANCE_CHECK,
            AuditEventType.COMPLIANCE_VIOLATION,
            AuditEventType.BEST_EXECUTION,
            AuditEventType.CONFLICT_CHECK
        ]]
        
        violations = [r for r in compliance_records if r.event_type == AuditEventType.COMPLIANCE_VIOLATION]
        
        return {
            'total_compliance_checks': len(compliance_records),
            'violations_count': len(violations),
            'violation_rate': len(violations) / len(compliance_records) if compliance_records else 0,
            'best_execution_reviews': len([r for r in records if r.event_type == AuditEventType.BEST_EXECUTION]),
            'manual_overrides': len([r for r in records if r.event_type == AuditEventType.MANUAL_OVERRIDE])
        }
    
    def _generate_detailed_analysis(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """Generate detailed analysis for comprehensive reports"""
        # Group records by hour for activity patterns
        hourly_activity = {}
        
        for record in records:
            hour_key = record.timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_activity:
                hourly_activity[hour_key] = {
                    'total': 0,
                    'orders': 0,
                    'violations': 0,
                    'errors': 0
                }
            
            hourly_activity[hour_key]['total'] += 1
            
            if 'ORDER' in record.event_type.value:
                hourly_activity[hour_key]['orders'] += 1
            if record.event_type == AuditEventType.COMPLIANCE_VIOLATION:
                hourly_activity[hour_key]['violations'] += 1
            if record.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                hourly_activity[hour_key]['errors'] += 1
        
        return {
            'hourly_activity': hourly_activity,
            'peak_activity_hour': max(hourly_activity.items(), key=lambda x: x[1]['total'])[0] if hourly_activity else None,
            'error_analysis': self._analyze_errors(records),
            'user_activity_patterns': self._analyze_user_patterns(records)
        }
    
    def _analyze_errors(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """Analyze error patterns in audit records"""
        error_records = [r for r in records if r.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]]
        
        error_patterns = {}
        for record in error_records:
            component = record.component or 'unknown'
            if component not in error_patterns:
                error_patterns[component] = []
            
            error_patterns[component].append({
                'timestamp': record.timestamp,
                'event_type': record.event_type.value,
                'message': record.event_data.get('error_message', 'No message')
            })
        
        return {
            'total_errors': len(error_records),
            'critical_errors': len([r for r in error_records if r.severity == AuditSeverity.CRITICAL]),
            'errors_by_component': {k: len(v) for k, v in error_patterns.items()},
            'recent_errors': sorted(error_records, key=lambda x: x.timestamp, reverse=True)[:10]
        }
    
    def _analyze_user_patterns(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        user_activities = {}
        
        for record in records:
            if record.user_id:
                if record.user_id not in user_activities:
                    user_activities[record.user_id] = {
                        'total_actions': 0,
                        'orders_placed': 0,
                        'manual_overrides': 0,
                        'compliance_violations': 0,
                        'first_activity': record.timestamp,
                        'last_activity': record.timestamp
                    }
                
                user_activities[record.user_id]['total_actions'] += 1
                
                if record.event_type == AuditEventType.ORDER_PLACED:
                    user_activities[record.user_id]['orders_placed'] += 1
                elif record.event_type == AuditEventType.MANUAL_OVERRIDE:
                    user_activities[record.user_id]['manual_overrides'] += 1
                elif record.event_type == AuditEventType.COMPLIANCE_VIOLATION:
                    user_activities[record.user_id]['compliance_violations'] += 1
                
                user_activities[record.user_id]['last_activity'] = max(
                    user_activities[record.user_id]['last_activity'],
                    record.timestamp
                )
        
        return user_activities