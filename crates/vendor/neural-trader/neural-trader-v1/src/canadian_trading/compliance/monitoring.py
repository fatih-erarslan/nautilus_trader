"""
Real-time Compliance Monitoring Module
Implements continuous monitoring and real-time compliance checks for Canadian trading regulations.
Includes position limits, trading velocity, pattern detection, and automated alerts.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import uuid
from collections import defaultdict, deque
import threading
import time
import redis
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MonitoringAlert(Enum):
    """Types of monitoring alerts"""
    POSITION_LIMIT_WARNING = "position_limit_warning"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    TRADING_VELOCITY_HIGH = "trading_velocity_high"
    UNUSUAL_PATTERN = "unusual_pattern"
    WASH_TRADE_SUSPECTED = "wash_trade_suspected"
    LAYERING_DETECTED = "layering_detected"
    SPOOFING_DETECTED = "spoofing_detected"
    PRICE_MANIPULATION = "price_manipulation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ANOMALY = "system_anomaly"
    SETTLEMENT_RISK = "settlement_risk"
    MARGIN_CALL = "margin_call"
    REGULATORY_HALT = "regulatory_halt"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringRule:
    """Configurable monitoring rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str
    parameters: Dict[str, Any]
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.MEDIUM
    actions: List[str] = field(default_factory=list)  # Actions to take when triggered
    cooldown_minutes: int = 5  # Prevent alert spam
    
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate rule against data, return (triggered, message)"""
        # This would be implemented for each specific rule type
        return False, None


@dataclass
class Alert:
    """Monitoring alert instance"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_type: MonitoringAlert = MonitoringAlert.COMPLIANCE_VIOLATION
    severity: AlertSeverity = AlertSeverity.MEDIUM
    rule_id: Optional[str] = None
    
    # Context
    account_id: Optional[str] = None
    client_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Alert details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Actions taken
    actions_required: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Resolution
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class PositionMonitor:
    """Monitor position limits and concentrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.position_limits = config.get('position_limits', {})
        self.concentration_limits = config.get('concentration_limits', {})
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def update_position(self, account_id: str, symbol: str, 
                       quantity: int, price: Decimal) -> List[Alert]:
        """Update position and check limits"""
        alerts = []
        
        with self._lock:
            if account_id not in self.positions:
                self.positions[account_id] = {}
            
            if symbol not in self.positions[account_id]:
                self.positions[account_id][symbol] = {
                    'quantity': 0,
                    'value': Decimal('0'),
                    'last_update': datetime.utcnow()
                }
            
            # Update position
            position = self.positions[account_id][symbol]
            position['quantity'] += quantity
            position['value'] = position['quantity'] * price
            position['last_update'] = datetime.utcnow()
            
            # Check position limits
            position_alerts = self._check_position_limits(account_id, symbol, position)
            alerts.extend(position_alerts)
            
            # Check concentration limits
            concentration_alerts = self._check_concentration_limits(account_id)
            alerts.extend(concentration_alerts)
        
        return alerts
    
    def _check_position_limits(self, account_id: str, symbol: str, 
                              position: Dict[str, Any]) -> List[Alert]:
        """Check if position exceeds limits"""
        alerts = []
        
        # Symbol-specific limits
        if symbol in self.position_limits:
            limit = self.position_limits[symbol]
            
            # Quantity limit
            if 'max_quantity' in limit and position['quantity'] > limit['max_quantity']:
                alerts.append(Alert(
                    alert_type=MonitoringAlert.POSITION_LIMIT_BREACH,
                    severity=AlertSeverity.HIGH,
                    account_id=account_id,
                    symbol=symbol,
                    message=f"Position quantity {position['quantity']} exceeds limit {limit['max_quantity']}",
                    details={
                        'current_quantity': position['quantity'],
                        'limit': limit['max_quantity'],
                        'excess': position['quantity'] - limit['max_quantity']
                    },
                    actions_required=['reduce_position', 'notify_risk_management']
                ))
            
            # Value limit
            if 'max_value' in limit and position['value'] > limit['max_value']:
                alerts.append(Alert(
                    alert_type=MonitoringAlert.POSITION_LIMIT_BREACH,
                    severity=AlertSeverity.HIGH,
                    account_id=account_id,
                    symbol=symbol,
                    message=f"Position value ${position['value']} exceeds limit ${limit['max_value']}",
                    details={
                        'current_value': float(position['value']),
                        'limit': limit['max_value'],
                        'excess': float(position['value'] - limit['max_value'])
                    },
                    actions_required=['reduce_position', 'notify_risk_management']
                ))
        
        # Check warning thresholds (80% of limit)
        elif symbol in self.position_limits:
            limit = self.position_limits[symbol]
            
            if 'max_quantity' in limit and position['quantity'] > limit['max_quantity'] * 0.8:
                alerts.append(Alert(
                    alert_type=MonitoringAlert.POSITION_LIMIT_WARNING,
                    severity=AlertSeverity.MEDIUM,
                    account_id=account_id,
                    symbol=symbol,
                    message=f"Position approaching limit (80% threshold)",
                    details={
                        'current_quantity': position['quantity'],
                        'limit': limit['max_quantity'],
                        'percentage': (position['quantity'] / limit['max_quantity']) * 100
                    }
                ))
        
        return alerts
    
    def _check_concentration_limits(self, account_id: str) -> List[Alert]:
        """Check portfolio concentration limits"""
        alerts = []
        
        if account_id not in self.positions:
            return alerts
        
        # Calculate total portfolio value
        total_value = sum(pos['value'] for pos in self.positions[account_id].values())
        
        if total_value == 0:
            return alerts
        
        # Check each position's concentration
        for symbol, position in self.positions[account_id].items():
            concentration = (position['value'] / total_value) * 100
            
            # Default concentration limit: 25%
            max_concentration = self.concentration_limits.get('default', 25)
            
            if concentration > max_concentration:
                alerts.append(Alert(
                    alert_type=MonitoringAlert.POSITION_LIMIT_BREACH,
                    severity=AlertSeverity.HIGH,
                    account_id=account_id,
                    symbol=symbol,
                    message=f"Position concentration {concentration:.1f}% exceeds limit {max_concentration}%",
                    details={
                        'concentration': concentration,
                        'limit': max_concentration,
                        'position_value': float(position['value']),
                        'portfolio_value': float(total_value)
                    },
                    actions_required=['rebalance_portfolio', 'notify_compliance']
                ))
        
        return alerts


class TradingPatternDetector:
    """Detect suspicious trading patterns"""
    
    def __init__(self, lookback_minutes: int = 30):
        self.lookback_minutes = lookback_minutes
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.order_book: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def add_trade(self, trade: Dict[str, Any]) -> List[Alert]:
        """Add trade and check for patterns"""
        alerts = []
        
        with self._lock:
            account_id = trade['account_id']
            self.trade_history[account_id].append({
                'timestamp': datetime.utcnow(),
                'trade': trade
            })
            
            # Check various patterns
            alerts.extend(self._check_wash_trading(account_id))
            alerts.extend(self._check_layering(account_id))
            alerts.extend(self._check_momentum_ignition(account_id))
            alerts.extend(self._check_marking_the_close(account_id))
        
        return alerts
    
    def add_order(self, order: Dict[str, Any]) -> List[Alert]:
        """Add order to book and check for spoofing"""
        alerts = []
        
        with self._lock:
            symbol = order['symbol']
            self.order_book[symbol].append({
                'timestamp': datetime.utcnow(),
                'order': order
            })
            
            # Clean old orders
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            self.order_book[symbol] = [
                o for o in self.order_book[symbol] 
                if o['timestamp'] > cutoff
            ]
            
            # Check for spoofing
            alerts.extend(self._check_spoofing(symbol))
        
        return alerts
    
    def _check_wash_trading(self, account_id: str) -> List[Alert]:
        """Check for wash trading (buying and selling to self)"""
        alerts = []
        
        recent_trades = self._get_recent_trades(account_id, minutes=5)
        
        # Group by symbol
        symbol_trades = defaultdict(list)
        for trade_record in recent_trades:
            trade = trade_record['trade']
            symbol_trades[trade['symbol']].append(trade)
        
        # Check each symbol for wash trading
        for symbol, trades in symbol_trades.items():
            buys = [t for t in trades if t['side'] == 'buy']
            sells = [t for t in trades if t['side'] == 'sell']
            
            # Check for matching quantities within short timeframe
            for buy in buys:
                for sell in sells:
                    time_diff = abs((buy['timestamp'] - sell['timestamp']).total_seconds())
                    
                    if (buy['quantity'] == sell['quantity'] and 
                        time_diff < 60 and  # Within 1 minute
                        abs(buy['price'] - sell['price']) < buy['price'] * 0.001):  # Similar price
                        
                        alerts.append(Alert(
                            alert_type=MonitoringAlert.WASH_TRADE_SUSPECTED,
                            severity=AlertSeverity.CRITICAL,
                            account_id=account_id,
                            symbol=symbol,
                            message=f"Potential wash trade detected for {symbol}",
                            details={
                                'buy_order': buy['order_id'],
                                'sell_order': sell['order_id'],
                                'quantity': buy['quantity'],
                                'time_difference': time_diff
                            },
                            actions_required=['halt_trading', 'investigate', 'notify_ciro']
                        ))
        
        return alerts
    
    def _check_layering(self, account_id: str) -> List[Alert]:
        """Check for layering (multiple orders to create false impression)"""
        alerts = []
        
        recent_trades = self._get_recent_trades(account_id, minutes=10)
        
        # Look for pattern of multiple orders followed by trades in opposite direction
        order_pattern = []
        for trade_record in recent_trades:
            trade = trade_record['trade']
            if trade['status'] in ['placed', 'cancelled']:
                order_pattern.append(trade)
            elif trade['status'] == 'executed':
                # Check if preceded by multiple cancelled orders
                cancelled_count = sum(1 for o in order_pattern[-10:] if o['status'] == 'cancelled')
                
                if cancelled_count >= 5:
                    alerts.append(Alert(
                        alert_type=MonitoringAlert.LAYERING_DETECTED,
                        severity=AlertSeverity.HIGH,
                        account_id=account_id,
                        symbol=trade['symbol'],
                        message=f"Potential layering detected - {cancelled_count} cancelled orders before execution",
                        details={
                            'cancelled_orders': cancelled_count,
                            'executed_trade': trade['order_id'],
                            'pattern_duration': '10 minutes'
                        },
                        actions_required=['review_trading_pattern', 'notify_compliance']
                    ))
        
        return alerts
    
    def _check_spoofing(self, symbol: str) -> List[Alert]:
        """Check for spoofing (placing orders with intent to cancel)"""
        alerts = []
        
        orders = self.order_book.get(symbol, [])
        if len(orders) < 10:
            return alerts
        
        # Analyze order cancellation patterns
        total_orders = len(orders)
        cancelled_orders = sum(1 for o in orders if o['order']['status'] == 'cancelled')
        
        cancellation_rate = cancelled_orders / total_orders if total_orders > 0 else 0
        
        if cancellation_rate > 0.9 and total_orders > 20:
            # High cancellation rate with significant volume
            alerts.append(Alert(
                alert_type=MonitoringAlert.SPOOFING_DETECTED,
                severity=AlertSeverity.CRITICAL,
                symbol=symbol,
                message=f"Potential spoofing detected - {cancellation_rate*100:.1f}% cancellation rate",
                details={
                    'total_orders': total_orders,
                    'cancelled_orders': cancelled_orders,
                    'cancellation_rate': cancellation_rate,
                    'time_window': '5 minutes'
                },
                actions_required=['halt_symbol_trading', 'investigate', 'notify_ciro']
            ))
        
        return alerts
    
    def _check_momentum_ignition(self, account_id: str) -> List[Alert]:
        """Check for momentum ignition attempts"""
        alerts = []
        
        recent_trades = self._get_recent_trades(account_id, minutes=15)
        
        # Group by symbol and analyze price movements
        symbol_trades = defaultdict(list)
        for trade_record in recent_trades:
            trade = trade_record['trade']
            symbol_trades[trade['symbol']].append(trade)
        
        for symbol, trades in symbol_trades.items():
            if len(trades) < 5:
                continue
            
            # Check for aggressive trading followed by opposite direction
            prices = [t['price'] for t in trades]
            
            # Calculate price momentum
            if len(prices) >= 5:
                first_half_avg = sum(prices[:len(prices)//2]) / (len(prices)//2)
                second_half_avg = sum(prices[len(prices)//2:]) / (len(prices) - len(prices)//2)
                
                price_change = (second_half_avg - first_half_avg) / first_half_avg
                
                if abs(price_change) > 0.02:  # 2% price movement
                    # Check if followed by opposite trades
                    last_side = trades[-1]['side']
                    opposite_trades = [t for t in trades[-3:] if t['side'] != last_side]
                    
                    if len(opposite_trades) >= 2:
                        alerts.append(Alert(
                            alert_type=MonitoringAlert.PRICE_MANIPULATION,
                            severity=AlertSeverity.HIGH,
                            account_id=account_id,
                            symbol=symbol,
                            message=f"Potential momentum ignition - {price_change*100:.1f}% price movement",
                            details={
                                'price_change_percent': price_change * 100,
                                'trade_count': len(trades),
                                'pattern': 'aggressive_then_reverse'
                            },
                            actions_required=['review_trades', 'notify_surveillance']
                        ))
        
        return alerts
    
    def _check_marking_the_close(self, account_id: str) -> List[Alert]:
        """Check for marking the close manipulation"""
        alerts = []
        
        now = datetime.utcnow()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM ET
        
        # Only check near market close
        if market_close - timedelta(minutes=30) <= now <= market_close + timedelta(minutes=5):
            recent_trades = self._get_recent_trades(account_id, minutes=30)
            
            # Count trades in last 5 minutes
            close_trades = [
                t for t in recent_trades 
                if (market_close - timedelta(minutes=5) <= t['timestamp'] <= market_close)
            ]
            
            if len(close_trades) >= 5:
                # High activity near close
                symbols = defaultdict(int)
                for trade_record in close_trades:
                    symbols[trade_record['trade']['symbol']] += 1
                
                for symbol, count in symbols.items():
                    if count >= 3:
                        alerts.append(Alert(
                            alert_type=MonitoringAlert.PRICE_MANIPULATION,
                            severity=AlertSeverity.HIGH,
                            account_id=account_id,
                            symbol=symbol,
                            message=f"Potential marking the close - {count} trades near market close",
                            details={
                                'trades_near_close': count,
                                'time_window': '5 minutes before close',
                                'symbol': symbol
                            },
                            actions_required=['review_closing_trades', 'notify_ciro']
                        ))
        
        return alerts
    
    def _get_recent_trades(self, account_id: str, minutes: int) -> List[Dict]:
        """Get trades from last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            t for t in self.trade_history.get(account_id, [])
            if t['timestamp'] > cutoff
        ]


class ComplianceMonitor:
    """Main real-time compliance monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_monitor = PositionMonitor(config)
        self.pattern_detector = TradingPatternDetector()
        self.rules: List[MonitoringRule] = self._load_monitoring_rules()
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        self._monitoring_thread = None
        self._last_rule_check: Dict[str, datetime] = {}
        
        # Initialize Redis for distributed monitoring (optional)
        self.redis_client = None
        if config.get('use_redis', False):
            self.redis_client = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                decode_responses=True
            )
    
    def _load_monitoring_rules(self) -> List[MonitoringRule]:
        """Load monitoring rules from configuration"""
        rules = []
        
        # Position limit rules
        rules.append(MonitoringRule(
            rule_id='position_limit_default',
            name='Default Position Limit',
            description='Monitor default position limits',
            rule_type='position_limit',
            parameters={'max_value': 1000000, 'max_concentration': 0.25},
            severity=AlertSeverity.HIGH,
            actions=['reduce_position', 'notify_risk']
        ))
        
        # Trading velocity rules
        rules.append(MonitoringRule(
            rule_id='trading_velocity',
            name='Trading Velocity Monitor',
            description='Monitor excessive trading frequency',
            rule_type='trading_velocity',
            parameters={'max_trades_per_minute': 100, 'max_orders_per_second': 10},
            severity=AlertSeverity.MEDIUM,
            actions=['throttle_trading', 'notify_ops']
        ))
        
        # Settlement risk rules
        rules.append(MonitoringRule(
            rule_id='settlement_risk',
            name='Settlement Risk Monitor',
            description='Monitor T+2 settlement obligations',
            rule_type='settlement',
            parameters={'warning_threshold_days': 1, 'critical_threshold_days': 2},
            severity=AlertSeverity.HIGH,
            actions=['flag_for_settlement', 'notify_operations']
        ))
        
        # Price deviation rules
        rules.append(MonitoringRule(
            rule_id='price_deviation',
            name='Price Deviation Monitor',
            description='Monitor orders with significant price deviation',
            rule_type='price_check',
            parameters={'max_deviation_percent': 5},
            severity=AlertSeverity.MEDIUM,
            actions=['flag_order', 'require_confirmation']
        ))
        
        return rules
    
    def start(self):
        """Start the monitoring system"""
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Compliance monitoring system started")
    
    def stop(self):
        """Stop the monitoring system"""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        self._executor.shutdown(wait=True)
        logger.info("Compliance monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Run periodic rule checks
                self._check_monitoring_rules()
                
                # Check for stale alerts
                self._check_stale_alerts()
                
                # Publish metrics
                self._publish_metrics()
                
                time.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _check_monitoring_rules(self):
        """Check all active monitoring rules"""
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_check = self._last_rule_check.get(rule.rule_id)
            if last_check and (datetime.utcnow() - last_check).total_seconds() < rule.cooldown_minutes * 60:
                continue
            
            try:
                # Rule-specific checks would be implemented here
                # This is a simplified example
                triggered, message = self._evaluate_rule(rule)
                
                if triggered:
                    alert = Alert(
                        alert_type=MonitoringAlert.COMPLIANCE_VIOLATION,
                        severity=rule.severity,
                        rule_id=rule.rule_id,
                        message=message or f"Rule {rule.name} triggered",
                        actions_required=rule.actions
                    )
                    
                    self._handle_alert(alert)
                    self._last_rule_check[rule.rule_id] = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {e}")
    
    def _evaluate_rule(self, rule: MonitoringRule) -> Tuple[bool, Optional[str]]:
        """Evaluate a specific rule"""
        # This would contain rule-specific logic
        # Simplified example
        if rule.rule_type == 'system_health':
            # Check system health metrics
            return False, None
        
        return False, None
    
    def _check_stale_alerts(self):
        """Check for alerts that need escalation"""
        stale_threshold = timedelta(minutes=30)
        now = datetime.utcnow()
        
        for alert_id, alert in self.alerts.items():
            if not alert.acknowledged and not alert.resolved:
                age = now - alert.timestamp
                
                if age > stale_threshold and alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                    # Escalate stale critical alerts
                    self._escalate_alert(alert)
    
    def _publish_metrics(self):
        """Publish monitoring metrics"""
        metrics = {
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
            'critical_alerts': len([a for a in self.alerts.values() if a.severity == AlertSeverity.CRITICAL and not a.resolved]),
            'rules_enabled': len([r for r in self.rules if r.enabled]),
            'monitoring_healthy': self._running
        }
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    'compliance:monitoring:metrics',
                    60,  # 60 second TTL
                    json.dumps(metrics)
                )
            except Exception as e:
                logger.error(f"Failed to publish metrics: {e}")
    
    def process_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade through monitoring system"""
        alerts = []
        
        # Update position monitor
        if trade['status'] == 'executed':
            position_alerts = self.position_monitor.update_position(
                trade['account_id'],
                trade['symbol'],
                trade['quantity'] if trade['side'] == 'buy' else -trade['quantity'],
                Decimal(str(trade['price']))
            )
            alerts.extend(position_alerts)
        
        # Check trading patterns
        pattern_alerts = self.pattern_detector.add_trade(trade)
        alerts.extend(pattern_alerts)
        
        # Process alerts
        for alert in alerts:
            self._handle_alert(alert)
        
        return {
            'trade_id': trade.get('trade_id'),
            'monitoring_result': 'passed' if not alerts else 'alerts_raised',
            'alerts': [a.alert_id for a in alerts],
            'critical_alerts': len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
        }
    
    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Process order through monitoring system"""
        alerts = []
        
        # Pre-trade checks
        if order['status'] == 'pending':
            # Check order parameters
            alerts.extend(self._check_order_parameters(order))
        
        # Pattern detection
        pattern_alerts = self.pattern_detector.add_order(order)
        alerts.extend(pattern_alerts)
        
        # Process alerts
        for alert in alerts:
            self._handle_alert(alert)
        
        # Determine if order should be blocked
        block_order = any(a.severity == AlertSeverity.CRITICAL for a in alerts)
        
        return {
            'order_id': order.get('order_id'),
            'monitoring_result': 'blocked' if block_order else 'passed',
            'alerts': [a.alert_id for a in alerts],
            'block_reason': alerts[0].message if block_order and alerts else None
        }
    
    def _check_order_parameters(self, order: Dict[str, Any]) -> List[Alert]:
        """Check order parameters for compliance"""
        alerts = []
        
        # Price deviation check
        if 'market_price' in order and 'limit_price' in order:
            deviation = abs(order['limit_price'] - order['market_price']) / order['market_price']
            
            if deviation > 0.05:  # 5% deviation
                alerts.append(Alert(
                    alert_type=MonitoringAlert.UNUSUAL_PATTERN,
                    severity=AlertSeverity.MEDIUM,
                    account_id=order.get('account_id'),
                    symbol=order.get('symbol'),
                    message=f"Order price deviates {deviation*100:.1f}% from market",
                    details={
                        'order_price': order['limit_price'],
                        'market_price': order['market_price'],
                        'deviation_percent': deviation * 100
                    }
                ))
        
        return alerts
    
    def _handle_alert(self, alert: Alert):
        """Handle a monitoring alert"""
        # Store alert
        self.alerts[alert.alert_id] = alert
        
        # Execute alert actions
        self._executor.submit(self._execute_alert_actions, alert)
        
        # Call registered handlers
        for handler in self.alert_handlers.get(alert.alert_type.value, []):
            self._executor.submit(handler, alert)
        
        # Log alert
        logger.warning(f"Alert raised: {alert.alert_type.value} - {alert.message}")
    
    def _execute_alert_actions(self, alert: Alert):
        """Execute required actions for an alert"""
        for action in alert.actions_required:
            try:
                if action == 'halt_trading':
                    self._halt_trading(alert.account_id)
                elif action == 'notify_compliance':
                    self._notify_compliance(alert)
                elif action == 'notify_ciro':
                    self._notify_regulatory_body(alert, 'CIRO')
                elif action == 'reduce_position':
                    self._initiate_position_reduction(alert)
                
                alert.actions_taken.append(action)
                
            except Exception as e:
                logger.error(f"Failed to execute action {action}: {e}")
    
    def _halt_trading(self, account_id: Optional[str]):
        """Halt trading for account or system-wide"""
        logger.critical(f"Trading halted for account: {account_id or 'SYSTEM-WIDE'}")
        # Implementation would interact with trading system
    
    def _notify_compliance(self, alert: Alert):
        """Notify compliance team"""
        # Implementation would send notifications
        logger.info(f"Compliance notified for alert: {alert.alert_id}")
    
    def _notify_regulatory_body(self, alert: Alert, body: str):
        """Notify regulatory body (CIRO, OSC, etc.)"""
        logger.critical(f"Regulatory notification to {body} for alert: {alert.alert_id}")
        # Implementation would submit regulatory notifications
    
    def _initiate_position_reduction(self, alert: Alert):
        """Initiate position reduction"""
        logger.warning(f"Position reduction initiated for {alert.symbol} in account {alert.account_id}")
        # Implementation would create reduction orders
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an unacknowledged alert"""
        logger.critical(f"ESCALATING ALERT: {alert.alert_id} - {alert.message}")
        # Implementation would trigger escalation procedures
    
    def register_alert_handler(self, alert_type: MonitoringAlert, handler: Callable):
        """Register a custom alert handler"""
        self.alert_handlers[alert_type.value].append(handler)
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.utcnow()
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user_id: str, notes: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_by = user_id
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = notes
            return True
        return False
    
    def get_active_alerts(self, filters: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        if filters:
            if 'severity' in filters:
                active_alerts = [a for a in active_alerts if a.severity == filters['severity']]
            if 'account_id' in filters:
                active_alerts = [a for a in active_alerts if a.account_id == filters['account_id']]
            if 'alert_type' in filters:
                active_alerts = [a for a in active_alerts if a.alert_type == filters['alert_type']]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        active_alerts = self.get_active_alerts()
        
        return {
            'status': 'running' if self._running else 'stopped',
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'high_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            'rules_active': len([r for r in self.rules if r.enabled]),
            'last_check': max(self._last_rule_check.values()) if self._last_rule_check else None,
            'position_accounts_monitored': len(self.position_monitor.positions),
            'pattern_detection_active': True
        }