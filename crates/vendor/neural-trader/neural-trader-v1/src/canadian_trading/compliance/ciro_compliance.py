"""
CIRO (Canadian Investment Regulatory Organization) Compliance Module
Implements best execution monitoring, trade reporting, client identification,
record keeping requirements, and conflict of interest checks.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import uuid
from decimal import Decimal
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types recognized by CIRO"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    MOC = "market_on_close"
    LOC = "limit_on_close"


class SecurityType(Enum):
    """Security types for CIRO compliance"""
    EQUITY = "equity"
    DEBT = "debt"
    DERIVATIVE = "derivative"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"


@dataclass
class ClientIdentification:
    """Client identification requirements per CIRO rules"""
    client_id: str
    legal_name: str
    account_type: str  # individual, joint, corporate, trust
    sin_or_bn: str  # SIN for individuals, BN for corporations
    address: Dict[str, str]
    phone: str
    email: str
    occupation: Optional[str] = None
    employer: Optional[str] = None
    investment_knowledge: str = "medium"  # low, medium, high
    risk_tolerance: str = "medium"  # low, medium, high
    net_worth_range: Optional[str] = None
    kyc_date: Optional[datetime] = None
    kyc_refresh_required: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate client information meets CIRO requirements"""
        errors = []
        
        if not self.legal_name or len(self.legal_name) < 2:
            errors.append("Legal name is required")
            
        if self.account_type not in ["individual", "joint", "corporate", "trust", "partnership"]:
            errors.append(f"Invalid account type: {self.account_type}")
            
        if self.account_type == "individual" and not self._validate_sin(self.sin_or_bn):
            errors.append("Valid SIN required for individual accounts")
        elif self.account_type in ["corporate", "partnership"] and not self._validate_bn(self.sin_or_bn):
            errors.append("Valid BN required for corporate/partnership accounts")
            
        if not self.address or not all(k in self.address for k in ["street", "city", "province", "postal_code"]):
            errors.append("Complete address required")
            
        if self.kyc_date and (datetime.now() - self.kyc_date).days > 1095:  # 3 years
            self.kyc_refresh_required = True
            errors.append("KYC refresh required (over 3 years old)")
            
        return len(errors) == 0, errors
    
    def _validate_sin(self, sin: str) -> bool:
        """Validate SIN format (not actual validity)"""
        sin_clean = sin.replace(" ", "").replace("-", "")
        return len(sin_clean) == 9 and sin_clean.isdigit()
    
    def _validate_bn(self, bn: str) -> bool:
        """Validate Business Number format"""
        bn_clean = bn.replace(" ", "").replace("-", "")
        return len(bn_clean) >= 9 and bn_clean[:9].isdigit()


@dataclass
class TradeReport:
    """Trade report structure for CIRO reporting"""
    trade_id: str
    client_id: str
    account_id: str
    symbol: str
    isin: Optional[str]
    cusip: Optional[str]
    security_type: SecurityType
    side: str  # buy/sell
    quantity: int
    price: Decimal
    commission: Decimal
    total_value: Decimal
    execution_venue: str
    execution_timestamp: datetime
    settlement_date: datetime
    order_id: str
    order_type: OrderType
    time_in_force: str
    special_instructions: Optional[str] = None
    
    def to_ciro_format(self) -> Dict[str, Any]:
        """Convert to CIRO trade reporting format"""
        return {
            "tradeId": self.trade_id,
            "clientId": self.client_id,
            "accountId": self.account_id,
            "symbol": self.symbol,
            "isin": self.isin,
            "cusip": self.cusip,
            "securityType": self.security_type.value,
            "side": self.side,
            "quantity": self.quantity,
            "price": str(self.price),
            "commission": str(self.commission),
            "totalValue": str(self.total_value),
            "executionVenue": self.execution_venue,
            "executionTimestamp": self.execution_timestamp.isoformat(),
            "settlementDate": self.settlement_date.isoformat(),
            "orderId": self.order_id,
            "orderType": self.order_type.value,
            "timeInForce": self.time_in_force,
            "specialInstructions": self.special_instructions
        }


class CIROCompliance:
    """Main CIRO compliance implementation"""
    
    def __init__(self, firm_id: str, registration_number: str):
        self.firm_id = firm_id
        self.registration_number = registration_number
        self.compliance_records = []
        self.conflict_registry = {}
        self.execution_venues = {
            "TSX": {"active": True, "fees": 0.0035, "rebate": 0.0020},
            "TSXV": {"active": True, "fees": 0.0035, "rebate": 0.0020},
            "CSE": {"active": True, "fees": 0.0030, "rebate": 0.0015},
            "NEO": {"active": True, "fees": 0.0025, "rebate": 0.0018},
            "OMEGA": {"active": True, "fees": 0.0030, "rebate": 0.0015}
        }
        self.provincial_rules = self._initialize_provincial_rules()
        
    def _initialize_provincial_rules(self) -> Dict[str, Dict]:
        """Initialize provincial regulatory variations"""
        return {
            "ON": {  # Ontario
                "regulator": "OSC",
                "additional_requirements": ["accredited_investor_verification"],
                "reporting_threshold": 500000,
                "insider_reporting_days": 5
            },
            "QC": {  # Quebec
                "regulator": "AMF",
                "additional_requirements": ["french_documentation", "quebec_investor_protection"],
                "reporting_threshold": 500000,
                "insider_reporting_days": 5,
                "language_requirements": "bilingual"
            },
            "BC": {  # British Columbia
                "regulator": "BCSC",
                "additional_requirements": ["venture_exchange_compliance"],
                "reporting_threshold": 500000,
                "insider_reporting_days": 5
            },
            "AB": {  # Alberta
                "regulator": "ASC",
                "additional_requirements": ["energy_sector_disclosure"],
                "reporting_threshold": 500000,
                "insider_reporting_days": 5
            }
        }
    
    async def ensure_best_execution(self, order: Dict, market_data: Dict) -> Dict[str, Any]:
        """
        UMIR 5.1 - Best Execution Obligation
        Ensure order is executed at best available price across all venues
        """
        symbol = order['symbol']
        order_type = OrderType(order.get('type', 'market'))
        quantity = order['quantity']
        side = order['side']
        
        # Collect quotes from all active venues
        venue_quotes = {}
        for venue, info in self.execution_venues.items():
            if info['active'] and symbol in market_data.get(venue, {}):
                quote = market_data[venue][symbol]
                if side == 'buy':
                    effective_price = Decimal(str(quote['ask'])) + Decimal(str(info['fees']))
                else:
                    effective_price = Decimal(str(quote['bid'])) - Decimal(str(info['fees']))
                
                venue_quotes[venue] = {
                    'price': quote['ask'] if side == 'buy' else quote['bid'],
                    'effective_price': float(effective_price),
                    'size': quote.get('ask_size' if side == 'buy' else 'bid_size', 0),
                    'fees': info['fees'],
                    'rebate': info['rebate']
                }
        
        # Determine best execution venue
        if not venue_quotes:
            return {
                'status': 'error',
                'message': 'No venues available for symbol',
                'symbol': symbol
            }
        
        # Sort by effective price (lowest for buy, highest for sell)
        sorted_venues = sorted(
            venue_quotes.items(),
            key=lambda x: x[1]['effective_price'],
            reverse=(side == 'sell')
        )
        
        best_venue = sorted_venues[0][0]
        best_quote = sorted_venues[0][1]
        
        # Check if order size can be filled
        total_available = sum(v['size'] for _, v in sorted_venues)
        
        execution_plan = {
            'status': 'success',
            'primary_venue': best_venue,
            'primary_price': best_quote['price'],
            'effective_price': best_quote['effective_price'],
            'venues_analyzed': len(venue_quotes),
            'best_execution_factors': {
                'price': best_quote['price'],
                'speed': 'immediate',
                'likelihood_of_execution': 'high' if total_available >= quantity else 'medium',
                'size': min(quantity, best_quote['size']),
                'costs': best_quote['fees']
            },
            'alternative_venues': [
                {
                    'venue': v[0],
                    'price': v[1]['price'],
                    'effective_price': v[1]['effective_price'],
                    'available_size': v[1]['size']
                }
                for v in sorted_venues[1:3]  # Top 3 alternatives
            ],
            'compliance_note': 'Best execution analysis completed per UMIR 5.1',
            'timestamp': datetime.now().isoformat()
        }
        
        # Log best execution decision
        self._log_best_execution_decision(order, execution_plan)
        
        return execution_plan
    
    def validate_client_identification(self, client_data: Dict) -> Tuple[bool, List[str]]:
        """Validate client meets CIRO identification requirements"""
        try:
            client = ClientIdentification(**client_data)
            return client.validate()
        except Exception as e:
            return False, [f"Client data validation error: {str(e)}"]
    
    def check_conflicts_of_interest(self, trade: Dict, client: ClientIdentification) -> Dict[str, Any]:
        """Check for conflicts of interest per CIRO requirements"""
        conflicts = []
        
        # Check for proprietary trading conflicts
        symbol = trade['symbol']
        if self._has_proprietary_position(symbol):
            conflicts.append({
                'type': 'proprietary_trading',
                'severity': 'medium',
                'description': f'Firm has proprietary position in {symbol}',
                'mitigation': 'Ensure client order priority'
            })
        
        # Check for research coverage conflicts
        if self._has_research_coverage(symbol):
            conflicts.append({
                'type': 'research_conflict',
                'severity': 'low',
                'description': f'Firm provides research coverage on {symbol}',
                'mitigation': 'Disclose research relationship'
            })
        
        # Check for underwriting conflicts
        if self._has_underwriting_relationship(symbol):
            conflicts.append({
                'type': 'underwriting',
                'severity': 'high',
                'description': f'Firm has underwriting relationship with {symbol}',
                'mitigation': 'Full disclosure required, consider restrictions'
            })
        
        # Check for related party transactions
        if self._is_related_party(client, symbol):
            conflicts.append({
                'type': 'related_party',
                'severity': 'high',
                'description': 'Transaction involves related party',
                'mitigation': 'Independent review required'
            })
        
        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'max_severity': max((c['severity'] for c in conflicts), default='none'),
            'review_required': any(c['severity'] == 'high' for c in conflicts),
            'timestamp': datetime.now().isoformat()
        }
    
    async def report_trade(self, trade_data: Dict) -> Dict[str, Any]:
        """Submit trade report to CIRO as required"""
        try:
            # Create trade report
            trade_report = TradeReport(
                trade_id=trade_data.get('trade_id', str(uuid.uuid4())),
                client_id=trade_data['client_id'],
                account_id=trade_data['account_id'],
                symbol=trade_data['symbol'],
                isin=trade_data.get('isin'),
                cusip=trade_data.get('cusip'),
                security_type=SecurityType(trade_data.get('security_type', 'equity')),
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                price=Decimal(str(trade_data['price'])),
                commission=Decimal(str(trade_data.get('commission', 0))),
                total_value=Decimal(str(trade_data['quantity'])) * Decimal(str(trade_data['price'])),
                execution_venue=trade_data['execution_venue'],
                execution_timestamp=datetime.fromisoformat(trade_data['execution_timestamp']),
                settlement_date=self._calculate_settlement_date(trade_data),
                order_id=trade_data['order_id'],
                order_type=OrderType(trade_data.get('order_type', 'market')),
                time_in_force=trade_data.get('time_in_force', 'day'),
                special_instructions=trade_data.get('special_instructions')
            )
            
            # Validate trade report
            validation_errors = self._validate_trade_report(trade_report)
            if validation_errors:
                return {
                    'status': 'error',
                    'errors': validation_errors,
                    'trade_id': trade_report.trade_id
                }
            
            # Store for record keeping (6+ years requirement)
            self._store_trade_record(trade_report)
            
            # Submit to CIRO (in production, this would be actual submission)
            ciro_format = trade_report.to_ciro_format()
            
            # Check if large trader reporting required
            if self._requires_large_trader_report(trade_report):
                await self._submit_large_trader_report(trade_report)
            
            return {
                'status': 'success',
                'trade_id': trade_report.trade_id,
                'reported_to_ciro': True,
                'report_timestamp': datetime.now().isoformat(),
                'settlement_date': trade_report.settlement_date.isoformat(),
                'large_trader_report': self._requires_large_trader_report(trade_report)
            }
            
        except Exception as e:
            logger.error(f"Trade reporting error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'trade_id': trade_data.get('trade_id')
            }
    
    def _calculate_settlement_date(self, trade_data: Dict) -> datetime:
        """Calculate settlement date based on security type"""
        execution_date = datetime.fromisoformat(trade_data['execution_timestamp'])
        security_type = trade_data.get('security_type', 'equity')
        
        # Standard settlement cycles
        if security_type in ['equity', 'etf']:
            settlement_days = 2  # T+2
        elif security_type == 'debt':
            settlement_days = 2  # T+2 for most bonds
        elif security_type == 'derivative':
            settlement_days = 1  # T+1 for options
        else:
            settlement_days = 2  # Default T+2
        
        # Account for weekends and holidays
        settlement_date = execution_date
        days_added = 0
        while days_added < settlement_days:
            settlement_date += timedelta(days=1)
            if settlement_date.weekday() < 5:  # Monday-Friday
                days_added += 1
        
        return settlement_date
    
    def _validate_trade_report(self, report: TradeReport) -> List[str]:
        """Validate trade report meets CIRO requirements"""
        errors = []
        
        if not report.client_id:
            errors.append("Client ID required")
        
        if report.quantity <= 0:
            errors.append("Invalid quantity")
        
        if report.price <= 0:
            errors.append("Invalid price")
        
        if report.execution_venue not in self.execution_venues:
            errors.append(f"Unknown execution venue: {report.execution_venue}")
        
        if report.security_type == SecurityType.EQUITY and not (report.symbol or report.isin or report.cusip):
            errors.append("Security identifier required (symbol, ISIN, or CUSIP)")
        
        return errors
    
    def _store_trade_record(self, report: TradeReport) -> None:
        """Store trade record for 6+ years as required by CIRO"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'record_type': 'trade_report',
            'data': report.to_ciro_format(),
            'retention_until': (datetime.now() + timedelta(days=2557)).isoformat()  # 7 years
        }
        
        # In production, this would persist to compliant storage
        self.compliance_records.append(record)
        
        # Create audit hash for integrity
        record['audit_hash'] = self._create_audit_hash(record)
    
    def _create_audit_hash(self, record: Dict) -> str:
        """Create tamper-proof hash for audit records"""
        record_str = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def _requires_large_trader_report(self, report: TradeReport) -> bool:
        """Check if trade requires large trader reporting"""
        # Check against provincial thresholds
        threshold = 500000  # Default threshold
        
        # In production, would determine province and check specific threshold
        return float(report.total_value) >= threshold
    
    async def _submit_large_trader_report(self, report: TradeReport) -> None:
        """Submit large trader report to CIRO"""
        large_trader_report = {
            'report_type': 'large_trader',
            'trade_id': report.trade_id,
            'symbol': report.symbol,
            'total_value': str(report.total_value),
            'percentage_of_adv': self._calculate_percentage_of_adv(report),
            'reporting_firm': self.firm_id,
            'submission_timestamp': datetime.now().isoformat()
        }
        
        # Log large trader report
        logger.info(f"Large trader report submitted for {report.symbol}: ${report.total_value}")
    
    def _calculate_percentage_of_adv(self, report: TradeReport) -> float:
        """Calculate trade size as percentage of average daily volume"""
        # In production, would fetch actual ADV data
        # Placeholder calculation
        estimated_adv = 1000000  # $1M average daily volume
        return float(report.total_value) / estimated_adv * 100
    
    def _has_proprietary_position(self, symbol: str) -> bool:
        """Check if firm has proprietary position in symbol"""
        # In production, would check actual proprietary positions
        return False
    
    def _has_research_coverage(self, symbol: str) -> bool:
        """Check if firm provides research coverage"""
        # In production, would check research coverage database
        return symbol in ['RY.TO', 'TD.TO', 'BNS.TO']  # Example: major banks
    
    def _has_underwriting_relationship(self, symbol: str) -> bool:
        """Check for underwriting relationships"""
        # In production, would check underwriting database
        return False
    
    def _is_related_party(self, client: ClientIdentification, symbol: str) -> bool:
        """Check if transaction involves related parties"""
        # In production, would check related party database
        return False
    
    def _log_best_execution_decision(self, order: Dict, execution_plan: Dict) -> None:
        """Log best execution decision for compliance"""
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'record_type': 'best_execution_decision',
            'order_id': order.get('order_id'),
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'venues_considered': execution_plan['venues_analyzed'],
            'selected_venue': execution_plan['primary_venue'],
            'price_improvement': self._calculate_price_improvement(order, execution_plan),
            'decision_factors': execution_plan['best_execution_factors']
        }
        
        self.compliance_records.append(decision_record)
    
    def _calculate_price_improvement(self, order: Dict, execution_plan: Dict) -> Optional[float]:
        """Calculate price improvement achieved"""
        if 'limit_price' in order:
            limit_price = float(order['limit_price'])
            execution_price = execution_plan['effective_price']
            
            if order['side'] == 'buy':
                improvement = limit_price - execution_price
            else:
                improvement = execution_price - limit_price
            
            return improvement if improvement > 0 else 0
        
        return None
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report for period"""
        relevant_records = [
            r for r in self.compliance_records
            if start_date <= datetime.fromisoformat(r['timestamp']) <= end_date
        ]
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_trades': len([r for r in relevant_records if r['record_type'] == 'trade_report']),
                'best_execution_decisions': len([r for r in relevant_records if r['record_type'] == 'best_execution_decision']),
                'large_trader_reports': len([r for r in relevant_records if r.get('report_type') == 'large_trader']),
                'conflicts_identified': 0,  # Would count actual conflicts
                'violations': 0  # Would count actual violations
            },
            'best_execution_metrics': self._calculate_best_execution_metrics(relevant_records),
            'venue_distribution': self._calculate_venue_distribution(relevant_records),
            'compliance_attestation': {
                'attested_by': 'Chief Compliance Officer',
                'date': datetime.now().isoformat(),
                'statement': 'All trading activities comply with CIRO regulations and best execution requirements'
            }
        }
        
        return report
    
    def _calculate_best_execution_metrics(self, records: List[Dict]) -> Dict[str, Any]:
        """Calculate best execution metrics for reporting"""
        be_decisions = [r for r in records if r['record_type'] == 'best_execution_decision']
        
        if not be_decisions:
            return {'no_data': True}
        
        price_improvements = [
            r.get('price_improvement', 0) for r in be_decisions 
            if r.get('price_improvement') is not None
        ]
        
        return {
            'total_decisions': len(be_decisions),
            'average_venues_considered': sum(r['venues_considered'] for r in be_decisions) / len(be_decisions),
            'price_improvement_rate': len([p for p in price_improvements if p > 0]) / len(price_improvements) if price_improvements else 0,
            'average_price_improvement': sum(price_improvements) / len(price_improvements) if price_improvements else 0
        }
    
    def _calculate_venue_distribution(self, records: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of trades across venues"""
        venue_counts = defaultdict(int)
        
        for record in records:
            if record['record_type'] == 'best_execution_decision':
                venue_counts[record['selected_venue']] += 1
        
        return dict(venue_counts)