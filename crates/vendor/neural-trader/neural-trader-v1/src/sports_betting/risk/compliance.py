"""
Compliance Framework for Sports Betting Operations

Comprehensive compliance management including:
- KYC/AML integration
- Regulatory reporting
- Jurisdiction-specific rules
- Audit trail maintenance
- Responsible gambling controls
"""

import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING = "pending"
    EXEMPT = "exempt"


class DocumentType(Enum):
    """KYC document types"""
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"


class RiskLevel(Enum):
    """AML risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


class JurisdictionType(Enum):
    """Jurisdiction types"""
    US_STATE = "us_state"
    EU_COUNTRY = "eu_country"
    UK = "uk"
    CANADA_PROVINCE = "canada_province"
    AUSTRALIA_STATE = "australia_state"
    OTHER = "other"


class ReportType(Enum):
    """Regulatory report types"""
    DAILY_ACTIVITY = "daily_activity"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_REVENUE = "monthly_revenue"
    QUARTERLY_TAX = "quarterly_tax"
    AML_SUSPICIOUS = "aml_suspicious"
    LARGE_TRANSACTION = "large_transaction"
    PROBLEM_GAMBLING = "problem_gambling"


@dataclass
class KYCDocument:
    """KYC document information"""
    document_id: str
    document_type: DocumentType
    document_number: str
    issue_date: datetime
    expiry_date: Optional[datetime]
    issuing_authority: str
    verified: bool = False
    verification_date: Optional[datetime] = None
    verification_method: str = ""
    document_hash: str = ""
    
    def __post_init__(self):
        """Generate document hash for integrity"""
        if not self.document_hash:
            hash_input = f"{self.document_type.value}_{self.document_number}_{self.issue_date.isoformat()}"
            self.document_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


@dataclass
class Customer:
    """Customer profile for compliance"""
    customer_id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    nationality: str
    residence_country: str
    residence_state: Optional[str]
    kyc_documents: List[KYCDocument] = field(default_factory=list)
    kyc_status: ComplianceStatus = ComplianceStatus.PENDING
    aml_risk_level: RiskLevel = RiskLevel.MEDIUM
    registration_date: datetime = field(default_factory=datetime.now)
    last_verification_date: Optional[datetime] = None
    
    def is_kyc_complete(self) -> bool:
        """Check if KYC is complete"""
        return self.kyc_status == ComplianceStatus.COMPLIANT
    
    def get_age(self) -> int:
        """Get customer age"""
        return (datetime.now() - self.date_of_birth).days // 365


@dataclass
class JurisdictionRule:
    """Jurisdiction-specific compliance rule"""
    rule_id: str
    jurisdiction: str
    jurisdiction_type: JurisdictionType
    rule_type: str
    description: str
    max_bet_amount: Optional[float] = None
    max_daily_loss: Optional[float] = None
    min_age: int = 18
    restricted_sports: List[str] = field(default_factory=list)
    restricted_bet_types: List[str] = field(default_factory=list)
    cooling_off_periods: Dict[str, int] = field(default_factory=dict)  # Hours
    requires_additional_verification: bool = False
    tax_reporting_threshold: Optional[float] = None
    
    def check_compliance(self, 
                        customer: Customer,
                        bet_amount: float,
                        sport: str,
                        bet_type: str,
                        daily_loss: float) -> Tuple[ComplianceStatus, List[str]]:
        """Check if transaction complies with jurisdiction rules"""
        violations = []
        
        # Age check
        if customer.get_age() < self.min_age:
            violations.append(f"Customer age ({customer.get_age()}) below minimum ({self.min_age})")
        
        # Bet amount check
        if self.max_bet_amount and bet_amount > self.max_bet_amount:
            violations.append(f"Bet amount (${bet_amount:.2f}) exceeds limit (${self.max_bet_amount:.2f})")
        
        # Daily loss check
        if self.max_daily_loss and daily_loss > self.max_daily_loss:
            violations.append(f"Daily loss (${daily_loss:.2f}) exceeds limit (${self.max_daily_loss:.2f})")
        
        # Sport restrictions
        if sport in self.restricted_sports:
            violations.append(f"Sport '{sport}' is restricted in this jurisdiction")
        
        # Bet type restrictions
        if bet_type in self.restricted_bet_types:
            violations.append(f"Bet type '{bet_type}' is restricted in this jurisdiction")
        
        # Additional verification requirement
        if self.requires_additional_verification and not customer.is_kyc_complete():
            violations.append("Additional verification required for this jurisdiction")
        
        if violations:
            return ComplianceStatus.VIOLATION, violations
        else:
            return ComplianceStatus.COMPLIANT, []


@dataclass
class ComplianceEvent:
    """Compliance monitoring event"""
    event_id: str
    event_type: str
    timestamp: datetime
    customer_id: str
    severity: str  # low, medium, high, critical
    description: str
    status: ComplianceStatus
    auto_resolved: bool = False
    resolution_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Audit trail entry"""
    audit_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    success: bool = True
    error_message: str = ""


class ResponsibleGamblingMonitor:
    """Monitor for responsible gambling indicators"""
    
    def __init__(self,
                 max_session_time_hours: int = 6,
                 max_daily_deposits: float = 5000,
                 max_chase_loss_ratio: float = 3.0,
                 velocity_threshold: int = 20):  # Bets per hour
        self.max_session_time_hours = max_session_time_hours
        self.max_daily_deposits = max_daily_deposits
        self.max_chase_loss_ratio = max_chase_loss_ratio
        self.velocity_threshold = velocity_threshold
        
        # Customer session tracking
        self.active_sessions: Dict[str, datetime] = {}
        self.daily_activity: Dict[str, Dict] = {}
    
    def check_session_time(self, customer_id: str) -> Tuple[ComplianceStatus, str]:
        """Check if customer session time is concerning"""
        if customer_id not in self.active_sessions:
            self.active_sessions[customer_id] = datetime.now()
            return ComplianceStatus.COMPLIANT, ""
        
        session_duration = datetime.now() - self.active_sessions[customer_id]
        hours = session_duration.total_seconds() / 3600
        
        if hours > self.max_session_time_hours:
            return ComplianceStatus.WARNING, f"Extended session detected: {hours:.1f} hours"
        elif hours > self.max_session_time_hours * 0.8:
            return ComplianceStatus.WARNING, f"Long session detected: {hours:.1f} hours"
        
        return ComplianceStatus.COMPLIANT, ""
    
    def check_chase_behavior(self, 
                           customer_id: str,
                           recent_losses: List[float],
                           current_bet: float) -> Tuple[ComplianceStatus, str]:
        """Check for loss-chasing behavior"""
        if not recent_losses:
            return ComplianceStatus.COMPLIANT, ""
        
        total_recent_losses = sum(abs(loss) for loss in recent_losses if loss < 0)
        
        if total_recent_losses > 0:
            chase_ratio = current_bet / total_recent_losses
            
            if chase_ratio > self.max_chase_loss_ratio:
                return ComplianceStatus.WARNING, f"Potential loss-chasing detected (ratio: {chase_ratio:.2f})"
        
        return ComplianceStatus.COMPLIANT, ""
    
    def check_betting_velocity(self,
                             customer_id: str,
                             recent_bet_times: List[datetime]) -> Tuple[ComplianceStatus, str]:
        """Check betting velocity"""
        if len(recent_bet_times) < 2:
            return ComplianceStatus.COMPLIANT, ""
        
        # Check bets in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_bets = [t for t in recent_bet_times if t > one_hour_ago]
        
        bets_per_hour = len(recent_bets)
        
        if bets_per_hour > self.velocity_threshold:
            return ComplianceStatus.WARNING, f"High betting velocity: {bets_per_hour} bets/hour"
        
        return ComplianceStatus.COMPLIANT, ""


class ComplianceFramework:
    """
    Comprehensive compliance framework for sports betting operations
    """
    
    def __init__(self):
        # Customers and KYC
        self.customers: Dict[str, Customer] = {}
        self.jurisdiction_rules: Dict[str, JurisdictionRule] = {}
        
        # Compliance monitoring
        self.compliance_events: List[ComplianceEvent] = []
        self.audit_trail: List[AuditEntry] = []
        
        # Responsible gambling
        self.rg_monitor = ResponsibleGamblingMonitor()
        
        # Reporting
        self.pending_reports: List[Dict] = []
        self.report_schedules: Dict[str, datetime] = {}
        
        # Initialize default rules
        self._initialize_default_jurisdiction_rules()
        
        logger.info("Compliance framework initialized")
    
    def _initialize_default_jurisdiction_rules(self):
        """Initialize default jurisdiction rules"""
        
        # Nevada (US)
        nevada_rule = JurisdictionRule(
            rule_id="nevada_us",
            jurisdiction="Nevada",
            jurisdiction_type=JurisdictionType.US_STATE,
            rule_type="state_regulation",
            description="Nevada Gaming Commission regulations",
            max_bet_amount=50000,
            max_daily_loss=100000,
            min_age=21,
            restricted_sports=["college_local"],
            requires_additional_verification=True,
            tax_reporting_threshold=5000
        )
        self.add_jurisdiction_rule(nevada_rule)
        
        # New Jersey (US)
        nj_rule = JurisdictionRule(
            rule_id="new_jersey_us",
            jurisdiction="New Jersey",
            jurisdiction_type=JurisdictionType.US_STATE,
            rule_type="state_regulation",
            description="New Jersey Division of Gaming Enforcement regulations",
            max_bet_amount=25000,
            max_daily_loss=50000,
            min_age=21,
            restricted_sports=["college_in_state"],
            restricted_bet_types=["player_props_college"],
            cooling_off_periods={"self_exclusion": 24},
            tax_reporting_threshold=600
        )
        self.add_jurisdiction_rule(nj_rule)
        
        # UK
        uk_rule = JurisdictionRule(
            rule_id="uk_gambling",
            jurisdiction="United Kingdom",
            jurisdiction_type=JurisdictionType.UK,
            rule_type="national_regulation",
            description="UK Gambling Commission regulations",
            max_bet_amount=None,  # No statutory limit
            max_daily_loss=None,  # Customer set limits
            min_age=18,
            restricted_sports=[],
            requires_additional_verification=False,
            cooling_off_periods={"cooling_off": 24, "self_exclusion": 24 * 7},
            tax_reporting_threshold=10000
        )
        self.add_jurisdiction_rule(uk_rule)
    
    def add_jurisdiction_rule(self, rule: JurisdictionRule):
        """Add jurisdiction-specific rule"""
        self.jurisdiction_rules[rule.rule_id] = rule
        logger.info(f"Jurisdiction rule added: {rule.jurisdiction}")
    
    def register_customer(self, customer: Customer) -> bool:
        """Register new customer"""
        if customer.customer_id in self.customers:
            logger.warning(f"Customer already registered: {customer.customer_id}")
            return False
        
        self.customers[customer.customer_id] = customer
        
        # Create audit entry
        self._create_audit_entry(
            user_id="system",
            action="customer_registration",
            resource_type="customer",
            resource_id=customer.customer_id,
            new_values={"registration_date": customer.registration_date.isoformat()}
        )
        
        logger.info(f"Customer registered: {customer.customer_id}")
        return True
    
    def verify_kyc_document(self,
                           customer_id: str,
                           document: KYCDocument,
                           verification_method: str = "manual") -> bool:
        """Verify KYC document"""
        if customer_id not in self.customers:
            logger.error(f"Customer not found: {customer_id}")
            return False
        
        customer = self.customers[customer_id]
        
        # Add document to customer
        document.verified = True
        document.verification_date = datetime.now()
        document.verification_method = verification_method
        
        customer.kyc_documents.append(document)
        
        # Check if KYC is now complete
        if self._check_kyc_completeness(customer):
            customer.kyc_status = ComplianceStatus.COMPLIANT
            customer.last_verification_date = datetime.now()
        
        # Create audit entry
        self._create_audit_entry(
            user_id="kyc_system",
            action="document_verification",
            resource_type="kyc_document",
            resource_id=document.document_id,
            new_values={
                "customer_id": customer_id,
                "document_type": document.document_type.value,
                "verified": True
            }
        )
        
        logger.info(f"KYC document verified: {document.document_id} for customer {customer_id}")
        return True
    
    def _check_kyc_completeness(self, customer: Customer) -> bool:
        """Check if customer KYC is complete"""
        verified_docs = [doc for doc in customer.kyc_documents if doc.verified]
        
        # Require at least one ID document and one proof of address
        has_id = any(doc.document_type in [DocumentType.PASSPORT, DocumentType.DRIVERS_LICENSE, 
                                          DocumentType.NATIONAL_ID] for doc in verified_docs)
        has_address = any(doc.document_type in [DocumentType.UTILITY_BILL, DocumentType.BANK_STATEMENT] 
                         for doc in verified_docs)
        
        return has_id and has_address
    
    def check_transaction_compliance(self,
                                   customer_id: str,
                                   bet_amount: float,
                                   sport: str,
                                   bet_type: str,
                                   jurisdiction: str,
                                   daily_loss: float = 0.0) -> Tuple[ComplianceStatus, List[str]]:
        """Check transaction compliance across all rules"""
        if customer_id not in self.customers:
            return ComplianceStatus.VIOLATION, ["Customer not found"]
        
        customer = self.customers[customer_id]
        violations = []
        
        # Check KYC status
        if not customer.is_kyc_complete():
            violations.append("KYC verification incomplete")
        
        # Check jurisdiction rules
        jurisdiction_rule = self._find_applicable_jurisdiction_rule(jurisdiction)
        if jurisdiction_rule:
            status, rule_violations = jurisdiction_rule.check_compliance(
                customer, bet_amount, sport, bet_type, daily_loss
            )
            violations.extend(rule_violations)
        
        # Check AML risk level
        if customer.aml_risk_level == RiskLevel.PROHIBITED:
            violations.append("Customer AML risk level: PROHIBITED")
        elif customer.aml_risk_level == RiskLevel.HIGH and bet_amount > 1000:
            violations.append("High-risk customer requires additional monitoring for large bets")
        
        # Responsible gambling checks
        rg_violations = self._check_responsible_gambling(customer_id, bet_amount)
        violations.extend(rg_violations)
        
        # Determine overall status
        if violations:
            status = ComplianceStatus.VIOLATION
            
            # Create compliance event
            self._create_compliance_event(
                event_type="transaction_compliance_violation",
                customer_id=customer_id,
                severity="high",
                description=f"Transaction compliance violations: {'; '.join(violations)}",
                status=ComplianceStatus.VIOLATION,
                metadata={
                    "bet_amount": bet_amount,
                    "sport": sport,
                    "bet_type": bet_type,
                    "jurisdiction": jurisdiction
                }
            )
        else:
            status = ComplianceStatus.COMPLIANT
        
        return status, violations
    
    def _find_applicable_jurisdiction_rule(self, jurisdiction: str) -> Optional[JurisdictionRule]:
        """Find applicable jurisdiction rule"""
        for rule in self.jurisdiction_rules.values():
            if rule.jurisdiction.lower() == jurisdiction.lower():
                return rule
        return None
    
    def _check_responsible_gambling(self, customer_id: str, bet_amount: float) -> List[str]:
        """Check responsible gambling indicators"""
        violations = []
        
        # Session time check
        status, message = self.rg_monitor.check_session_time(customer_id)
        if status != ComplianceStatus.COMPLIANT:
            violations.append(message)
        
        # Additional RG checks would go here
        # (chase behavior, betting velocity, etc.)
        
        return violations
    
    def generate_report(self, report_type: ReportType, period_start: datetime, period_end: datetime) -> Dict:
        """Generate regulatory report"""
        if report_type == ReportType.DAILY_ACTIVITY:
            return self._generate_daily_activity_report(period_start, period_end)
        elif report_type == ReportType.AML_SUSPICIOUS:
            return self._generate_aml_report(period_start, period_end)
        elif report_type == ReportType.LARGE_TRANSACTION:
            return self._generate_large_transaction_report(period_start, period_end)
        else:
            return {"error": f"Report type {report_type.value} not implemented"}
    
    def _generate_daily_activity_report(self, start: datetime, end: datetime) -> Dict:
        """Generate daily activity report"""
        # Filter relevant audit entries
        period_audits = [
            audit for audit in self.audit_trail
            if start <= audit.timestamp <= end
        ]
        
        # Aggregate by action type
        action_counts = {}
        for audit in period_audits:
            action_counts[audit.action] = action_counts.get(audit.action, 0) + 1
        
        return {
            "report_type": "daily_activity",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "total_actions": len(period_audits),
            "action_breakdown": action_counts,
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_aml_report(self, start: datetime, end: datetime) -> Dict:
        """Generate AML suspicious activity report"""
        # Find high-risk customers and events
        suspicious_events = [
            event for event in self.compliance_events
            if start <= event.timestamp <= end and event.severity in ["high", "critical"]
        ]
        
        high_risk_customers = [
            customer for customer in self.customers.values()
            if customer.aml_risk_level in [RiskLevel.HIGH, RiskLevel.PROHIBITED]
        ]
        
        return {
            "report_type": "aml_suspicious",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "suspicious_events": len(suspicious_events),
            "high_risk_customers": len(high_risk_customers),
            "events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "customer_id": event.customer_id,
                    "description": event.description,
                    "severity": event.severity
                }
                for event in suspicious_events
            ],
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_large_transaction_report(self, start: datetime, end: datetime) -> Dict:
        """Generate large transaction report"""
        # This would integrate with transaction tracking
        # For now, return placeholder
        return {
            "report_type": "large_transaction",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "large_transactions": [],
            "generated_at": datetime.now().isoformat()
        }
    
    def _create_compliance_event(self,
                                event_type: str,
                                customer_id: str,
                                severity: str,
                                description: str,
                                status: ComplianceStatus,
                                metadata: Dict[str, Any] = None) -> str:
        """Create compliance monitoring event"""
        event_id = str(uuid.uuid4())
        
        event = ComplianceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            customer_id=customer_id,
            severity=severity,
            description=description,
            status=status,
            metadata=metadata or {}
        )
        
        self.compliance_events.append(event)
        
        logger.warning(f"Compliance event created: {event_id} - {description}")
        
        return event_id
    
    def _create_audit_entry(self,
                           user_id: str,
                           action: str,
                           resource_type: str,
                           resource_id: str,
                           old_values: Dict[str, Any] = None,
                           new_values: Dict[str, Any] = None,
                           ip_address: str = "",
                           user_agent: str = "",
                           success: bool = True,
                           error_message: str = "") -> str:
        """Create audit trail entry"""
        audit_id = str(uuid.uuid4())
        
        audit = AuditEntry(
            audit_id=audit_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values or {},
            new_values=new_values or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        self.audit_trail.append(audit)
        
        return audit_id
    
    def get_compliance_dashboard(self) -> Dict:
        """Get compliance monitoring dashboard"""
        # Recent events
        recent_events = [e for e in self.compliance_events 
                        if e.timestamp > datetime.now() - timedelta(days=7)]
        
        # KYC status summary
        kyc_summary = {
            "compliant": len([c for c in self.customers.values() if c.kyc_status == ComplianceStatus.COMPLIANT]),
            "pending": len([c for c in self.customers.values() if c.kyc_status == ComplianceStatus.PENDING]),
            "total": len(self.customers)
        }
        
        # AML risk distribution
        aml_summary = {
            "low": len([c for c in self.customers.values() if c.aml_risk_level == RiskLevel.LOW]),
            "medium": len([c for c in self.customers.values() if c.aml_risk_level == RiskLevel.MEDIUM]),
            "high": len([c for c in self.customers.values() if c.aml_risk_level == RiskLevel.HIGH]),
            "prohibited": len([c for c in self.customers.values() if c.aml_risk_level == RiskLevel.PROHIBITED])
        }
        
        return {
            "customers": {
                "total": len(self.customers),
                "kyc_summary": kyc_summary,
                "aml_summary": aml_summary
            },
            "compliance_events": {
                "total": len(self.compliance_events),
                "recent": len(recent_events),
                "by_severity": {
                    "critical": len([e for e in recent_events if e.severity == "critical"]),
                    "high": len([e for e in recent_events if e.severity == "high"]),
                    "medium": len([e for e in recent_events if e.severity == "medium"]),
                    "low": len([e for e in recent_events if e.severity == "low"])
                }
            },
            "audit_trail": {
                "total_entries": len(self.audit_trail),
                "recent_entries": len([a for a in self.audit_trail 
                                     if a.timestamp > datetime.now() - timedelta(days=1)])
            },
            "jurisdiction_rules": {
                "total": len(self.jurisdiction_rules),
                "by_type": {
                    jtype.value: len([r for r in self.jurisdiction_rules.values() 
                                    if r.jurisdiction_type == jtype])
                    for jtype in JurisdictionType
                }
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize compliance framework
    compliance = ComplianceFramework()
    
    # Register a customer
    customer = Customer(
        customer_id="cust_001",
        first_name="John",
        last_name="Doe",
        date_of_birth=datetime(1990, 5, 15),
        nationality="US",
        residence_country="US",
        residence_state="Nevada"
    )
    compliance.register_customer(customer)
    
    # Add KYC document
    kyc_doc = KYCDocument(
        document_id="doc_001",
        document_type=DocumentType.DRIVERS_LICENSE,
        document_number="D1234567",
        issue_date=datetime(2020, 1, 1),
        expiry_date=datetime(2025, 1, 1),
        issuing_authority="Nevada DMV"
    )
    compliance.verify_kyc_document("cust_001", kyc_doc)
    
    # Check transaction compliance
    status, violations = compliance.check_transaction_compliance(
        customer_id="cust_001",
        bet_amount=1000,
        sport="NFL",
        bet_type="spread",
        jurisdiction="Nevada",
        daily_loss=500
    )
    
    print(f"Compliance Status: {status.value}")
    if violations:
        print("Violations:", violations)
    
    # Get dashboard
    dashboard = compliance.get_compliance_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))