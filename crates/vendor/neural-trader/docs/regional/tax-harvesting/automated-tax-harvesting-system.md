# Automated Tax Harvesting System

## Overview

This document provides detailed implementation of an automated tax harvesting system that leverages the AI News Trading Platform's MCP tools for real-time loss identification, compliance monitoring, and optimal execution. The system operates 24/7 to maximize tax efficiency while maintaining portfolio objectives.

## Core System Components

### 1. Real-Time Loss Identification Algorithm

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum

class TaxLotMethod(Enum):
    FIFO = "first_in_first_out"
    LIFO = "last_in_first_out"
    HIFO = "highest_in_first_out"
    SPECIFIC_ID = "specific_identification"

@dataclass
class TaxLot:
    symbol: str
    purchase_date: datetime
    shares: float
    cost_basis: float
    current_price: float
    
    @property
    def unrealized_gain_loss(self) -> float:
        return (self.current_price - self.cost_basis) * self.shares
    
    @property
    def holding_period_days(self) -> int:
        return (datetime.now() - self.purchase_date).days
    
    @property
    def is_long_term(self) -> bool:
        return self.holding_period_days > 365
    
    @property
    def tax_rate(self) -> float:
        # Simplified tax rates - customize based on jurisdiction
        return 0.15 if self.is_long_term else 0.35

class RealTimeLossIdentifier:
    def __init__(self, mcp_client, config: Dict):
        self.mcp_client = mcp_client
        self.config = config
        self.min_loss_threshold = config.get('min_loss_threshold', 100)
        self.scan_interval = config.get('scan_interval', 300)  # 5 minutes
        self.tax_lot_method = TaxLotMethod(config.get('tax_lot_method', 'HIFO'))
        
    async def continuous_loss_scanning(self):
        """Continuously scan portfolio for tax loss opportunities"""
        while True:
            try:
                # Get current portfolio with detailed lot information
                portfolio_data = await self._get_detailed_portfolio()
                
                # Identify loss opportunities
                opportunities = await self._identify_loss_opportunities(portfolio_data)
                
                # Rank opportunities by tax benefit
                ranked_opportunities = self._rank_opportunities(opportunities)
                
                # Process top opportunities
                for opportunity in ranked_opportunities[:5]:  # Top 5 opportunities
                    await self._process_opportunity(opportunity)
                
                # Wait before next scan
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                await self._handle_scan_error(e)
    
    async def _get_detailed_portfolio(self) -> List[TaxLot]:
        """Get detailed portfolio with tax lot information"""
        # Get portfolio status from MCP
        portfolio = await self.mcp_client.call_tool(
            "mcp__ai-news-trader__get_portfolio_status",
            {"include_analytics": True}
        )
        
        tax_lots = []
        
        for position in portfolio['positions']:
            # Get detailed lot information (simulated here)
            lots = await self._get_tax_lots(position)
            tax_lots.extend(lots)
        
        return tax_lots
    
    async def _identify_loss_opportunities(self, 
                                         tax_lots: List[TaxLot]) -> List[Dict]:
        """Identify tax loss harvesting opportunities"""
        opportunities = []
        
        for lot in tax_lots:
            if lot.unrealized_gain_loss < -self.min_loss_threshold:
                # Check wash sale compliance
                wash_sale_check = await self._check_wash_sale_window(lot)
                
                if wash_sale_check['compliant']:
                    # Find replacement security
                    replacement = await self._find_replacement_security(lot.symbol)
                    
                    if replacement:
                        opportunity = {
                            'tax_lot': lot,
                            'tax_benefit': abs(lot.unrealized_gain_loss) * lot.tax_rate,
                            'replacement_symbol': replacement['symbol'],
                            'replacement_correlation': replacement['correlation'],
                            'wash_sale_safe': wash_sale_check['safe_date'],
                            'priority_score': self._calculate_priority_score(lot, replacement)
                        }
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def _find_replacement_security(self, symbol: str) -> Optional[Dict]:
        """Find suitable replacement security maintaining market exposure"""
        # Get correlation data
        correlation_data = await self.mcp_client.call_tool(
            "mcp__ai-news-trader__correlation_analysis",
            {
                "symbols": self._get_candidate_replacements(symbol),
                "period_days": 90,
                "use_gpu": True
            }
        )
        
        # Filter for high correlation securities
        candidates = []
        min_correlation = 0.80
        
        for candidate, correlation in correlation_data['correlation_matrix'][symbol].items():
            if candidate != symbol and correlation >= min_correlation:
                # Additional checks for suitability
                suitability = await self._check_replacement_suitability(
                    symbol, candidate, correlation
                )
                
                if suitability['suitable']:
                    candidates.append({
                        'symbol': candidate,
                        'correlation': correlation,
                        'liquidity_score': suitability['liquidity_score'],
                        'cost_ratio': suitability['cost_ratio']
                    })
        
        # Return best candidate
        if candidates:
            return max(candidates, key=lambda x: x['correlation'] - x['cost_ratio'])
        
        return None
    
    def _calculate_priority_score(self, lot: TaxLot, replacement: Dict) -> float:
        """Calculate priority score for harvesting opportunity"""
        # Factors for priority scoring
        tax_benefit_score = min(abs(lot.unrealized_gain_loss) / 10000, 1.0)
        holding_period_score = 1.0 if lot.is_long_term else 0.5
        correlation_score = replacement['correlation']
        days_until_long_term = max(0, 365 - lot.holding_period_days)
        timing_score = 1.0 if days_until_long_term > 30 else 0.5
        
        # Weighted priority score
        priority = (
            tax_benefit_score * 0.4 +
            holding_period_score * 0.2 +
            correlation_score * 0.2 +
            timing_score * 0.2
        )
        
        return priority
```

### 2. Wash Sale Rule Compliance Automation

```python
class WashSaleComplianceEngine:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.wash_sale_window = 30  # days
        self.transaction_cache = {}
        self.similar_security_cache = {}
        
    async def automated_compliance_check(self, 
                                       symbol: str,
                                       proposed_action: str,
                                       shares: float) -> Dict:
        """Automated wash sale compliance checking"""
        # Get transaction history
        history = await self._get_transaction_history(symbol)
        
        # Check for wash sale violations
        violations = []
        warnings = []
        
        # Check 30 days before and after
        if proposed_action == 'sell':
            # Look for recent or planned purchases
            future_purchases = await self._check_future_purchase_risk(symbol)
            past_purchases = self._check_past_purchases(symbol, history)
            
            if past_purchases:
                violations.extend(past_purchases)
            if future_purchases:
                warnings.extend(future_purchases)
        
        # Check substantially identical securities
        similar_securities = await self._check_similar_securities(symbol)
        
        for similar in similar_securities:
            similar_history = await self._get_transaction_history(similar['symbol'])
            similar_violations = self._check_wash_sale_violations(
                symbol, similar['symbol'], similar_history
            )
            if similar_violations:
                warnings.extend(similar_violations)
        
        # Generate compliance report
        compliance_status = {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'safe_execution_date': self._calculate_safe_date(symbol, history),
            'alternative_strategies': await self._suggest_alternatives(symbol)
        }
        
        return compliance_status
    
    async def _check_similar_securities(self, symbol: str) -> List[Dict]:
        """Identify substantially identical securities"""
        if symbol in self.similar_security_cache:
            return self.similar_security_cache[symbol]
        
        # Use correlation analysis to find similar securities
        correlation_data = await self.mcp_client.call_tool(
            "mcp__ai-news-trader__cross_asset_correlation_matrix",
            {
                "assets": self._get_peer_securities(symbol),
                "lookback_days": 180,
                "include_prediction_confidence": True
            }
        )
        
        similar = []
        
        # IRS substantially identical threshold (not defined, using 0.90)
        for asset, correlation in correlation_data['correlation_matrix'][symbol].items():
            if asset != symbol and correlation > 0.90:
                # Additional checks for substantial identity
                if self._check_substantial_identity(symbol, asset):
                    similar.append({
                        'symbol': asset,
                        'correlation': correlation,
                        'identity_score': self._calculate_identity_score(symbol, asset)
                    })
        
        self.similar_security_cache[symbol] = similar
        return similar
    
    def _check_substantial_identity(self, symbol1: str, symbol2: str) -> bool:
        """Check if two securities are substantially identical"""
        # Check for common substantially identical pairs
        identical_pairs = [
            ('SPY', 'VOO'),  # S&P 500 ETFs
            ('QQQ', 'QQQM'),  # Nasdaq 100 ETFs
            ('IWM', 'VTWO'),  # Small cap ETFs
            ('VTI', 'ITOT'),  # Total market ETFs
        ]
        
        for pair in identical_pairs:
            if (symbol1, symbol2) in [pair, pair[::-1]]:
                return True
        
        # Check for same underlying index
        if self._same_underlying_index(symbol1, symbol2):
            return True
        
        # Check for convertible securities
        if self._are_convertible(symbol1, symbol2):
            return True
        
        return False
```

### 3. Portfolio Rebalancing Integration

```python
class TaxAwareRebalancer:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.rebalance_threshold = 0.05  # 5% deviation
        self.tax_cost_threshold = 0.02  # 2% tax cost limit
        
    async def tax_aware_rebalance(self, 
                                 target_allocations: Dict[str, float]) -> Dict:
        """Perform tax-aware portfolio rebalancing"""
        # Get current portfolio
        portfolio = await self.mcp_client.call_tool(
            "mcp__ai-news-trader__get_portfolio_status",
            {"include_analytics": True}
        )
        
        # Calculate current allocations
        current_allocations = self._calculate_current_allocations(portfolio)
        
        # Identify rebalancing needs
        rebalance_trades = []
        
        for symbol, target_weight in target_allocations.items():
            current_weight = current_allocations.get(symbol, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > self.rebalance_threshold:
                # Check tax implications
                tax_impact = await self._calculate_rebalance_tax_impact(
                    symbol, current_weight, target_weight, portfolio
                )
                
                if tax_impact['tax_cost_ratio'] < self.tax_cost_threshold:
                    trade = {
                        'symbol': symbol,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'shares_to_trade': tax_impact['optimal_shares'],
                        'tax_cost': tax_impact['tax_cost'],
                        'lots_to_sell': tax_impact['selected_lots']
                    }
                    rebalance_trades.append(trade)
                else:
                    # Find tax-efficient alternative
                    alternative = await self._find_tax_efficient_alternative(
                        symbol, target_weight, tax_impact
                    )
                    if alternative:
                        rebalance_trades.append(alternative)
        
        # Optimize trade execution order
        optimized_trades = self._optimize_trade_sequence(rebalance_trades)
        
        # Execute rebalancing
        execution_results = await self._execute_rebalance_trades(optimized_trades)
        
        return {
            'trades_executed': execution_results,
            'new_allocations': await self._get_post_rebalance_allocations(),
            'total_tax_cost': sum(t['tax_cost'] for t in execution_results),
            'tracking_error': self._calculate_tracking_error(target_allocations)
        }
    
    async def _calculate_rebalance_tax_impact(self, 
                                            symbol: str,
                                            current_weight: float,
                                            target_weight: float,
                                            portfolio: Dict) -> Dict:
        """Calculate tax impact of rebalancing trade"""
        position = next(p for p in portfolio['positions'] if p['symbol'] == symbol)
        
        # Get tax lots
        tax_lots = await self._get_position_tax_lots(position)
        
        # Calculate shares needed
        portfolio_value = portfolio['total_value']
        current_value = current_weight * portfolio_value
        target_value = target_weight * portfolio_value
        value_change = target_value - current_value
        
        if value_change < 0:  # Need to sell
            shares_to_sell = abs(value_change) / position['current_price']
            
            # Select optimal lots for tax efficiency
            selected_lots = self._select_tax_efficient_lots(
                tax_lots, shares_to_sell
            )
            
            tax_cost = sum(
                lot['unrealized_gain'] * lot['tax_rate'] 
                for lot in selected_lots if lot['unrealized_gain'] > 0
            )
            
            return {
                'optimal_shares': shares_to_sell,
                'selected_lots': selected_lots,
                'tax_cost': tax_cost,
                'tax_cost_ratio': tax_cost / abs(value_change)
            }
        else:
            # Buying - no immediate tax impact
            return {
                'optimal_shares': value_change / position['current_price'],
                'selected_lots': [],
                'tax_cost': 0,
                'tax_cost_ratio': 0
            }
```

### 4. Tax-Efficient Order Execution

```python
class TaxEfficientExecutor:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.execution_strategies = {
            'immediate': self._execute_immediate,
            'patient': self._execute_patient,
            'spread': self._execute_spread,
            'paired': self._execute_paired
        }
        
    async def execute_tax_harvest_trade(self, 
                                      harvest_order: Dict,
                                      strategy: str = 'paired') -> Dict:
        """Execute tax harvest trade with optimal strategy"""
        # Validate harvest order
        validation = await self._validate_harvest_order(harvest_order)
        if not validation['valid']:
            return {
                'status': 'rejected',
                'reason': validation['reason']
            }
        
        # Select execution strategy
        execution_func = self.execution_strategies.get(
            strategy, self._execute_paired
        )
        
        # Execute with monitoring
        result = await execution_func(harvest_order)
        
        # Post-execution compliance check
        compliance = await self._post_execution_compliance(result)
        
        return {
            'execution_result': result,
            'compliance_status': compliance,
            'tax_benefit_realized': self._calculate_realized_benefit(result)
        }
    
    async def _execute_paired(self, harvest_order: Dict) -> Dict:
        """Execute paired sell-buy trade for tax harvesting"""
        sell_order = harvest_order['sell_order']
        buy_order = harvest_order['buy_order']
        
        try:
            # Pre-stage buy order to minimize gap risk
            buy_staged = await self._stage_order(buy_order)
            
            # Execute sell order
            sell_result = await self.mcp_client.call_tool(
                "mcp__ai-news-trader__execute_trade",
                {
                    "strategy": "tax_harvesting",
                    "symbol": sell_order['symbol'],
                    "action": "sell",
                    "quantity": sell_order['quantity'],
                    "order_type": sell_order.get('order_type', 'market')
                }
            )
            
            if sell_result['status'] == 'success':
                # Immediately execute staged buy order
                buy_result = await self._execute_staged_order(buy_staged)
                
                # Calculate execution quality
                execution_quality = self._calculate_execution_quality(
                    sell_result, buy_result
                )
                
                return {
                    'status': 'success',
                    'sell_result': sell_result,
                    'buy_result': buy_result,
                    'execution_quality': execution_quality,
                    'gap_risk_realized': self._calculate_gap_risk(
                        sell_result['execution_price'],
                        buy_result['execution_price']
                    )
                }
            else:
                # Cancel staged buy order
                await self._cancel_staged_order(buy_staged)
                return {
                    'status': 'failed',
                    'reason': f"Sell order failed: {sell_result.get('error_message')}"
                }
                
        except Exception as e:
            # Ensure cleanup
            await self._cleanup_failed_execution(harvest_order)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _execute_spread(self, harvest_order: Dict) -> Dict:
        """Execute harvest trade spread over time to minimize market impact"""
        total_shares = harvest_order['sell_order']['quantity']
        num_slices = min(10, max(2, int(total_shares / 100)))
        slice_size = total_shares / num_slices
        
        results = []
        
        for i in range(num_slices):
            # Calculate slice timing
            delay = i * 30  # 30 seconds between slices
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Execute slice
            slice_order = {
                **harvest_order,
                'sell_order': {
                    **harvest_order['sell_order'],
                    'quantity': slice_size
                },
                'buy_order': {
                    **harvest_order['buy_order'],
                    'quantity': slice_size
                }
            }
            
            slice_result = await self._execute_paired(slice_order)
            results.append(slice_result)
            
            # Check if we should continue
            if slice_result['status'] != 'success':
                break
        
        return {
            'status': 'completed',
            'slices_executed': len(results),
            'total_executed': sum(
                r['sell_result']['quantity'] 
                for r in results if r['status'] == 'success'
            ),
            'average_prices': self._calculate_average_prices(results),
            'slice_results': results
        }
```

### 5. Year-End Tax Planning Automation

```python
class YearEndTaxPlanner:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.planning_window = 60  # Days before year end
        
    async def automated_year_end_planning(self) -> Dict:
        """Comprehensive year-end tax planning automation"""
        current_date = datetime.now()
        year_end = datetime(current_date.year, 12, 31)
        days_to_year_end = (year_end - current_date).days
        
        if days_to_year_end > self.planning_window:
            return {
                'status': 'not_yet',
                'days_until_planning': days_to_year_end - self.planning_window
            }
        
        # Comprehensive tax analysis
        tax_situation = await self._analyze_current_tax_situation()
        
        # Generate optimization strategies
        strategies = []
        
        # 1. Loss harvesting opportunities
        harvest_opportunities = await self._identify_year_end_harvests(
            tax_situation
        )
        strategies.extend(harvest_opportunities)
        
        # 2. Gain realization for low-income years
        if tax_situation['current_tax_bracket'] < tax_situation['expected_next_year_bracket']:
            gain_opportunities = await self._identify_gain_realization(
                tax_situation
            )
            strategies.extend(gain_opportunities)
        
        # 3. Charitable donation optimization
        donation_strategies = await self._optimize_charitable_donations(
            tax_situation
        )
        strategies.extend(donation_strategies)
        
        # 4. Tax lot optimization
        lot_optimization = await self._optimize_tax_lots(tax_situation)
        strategies.extend(lot_optimization)
        
        # Rank strategies by benefit
        ranked_strategies = self._rank_strategies_by_benefit(strategies)
        
        # Generate execution plan
        execution_plan = await self._create_execution_plan(
            ranked_strategies, days_to_year_end
        )
        
        return {
            'tax_situation': tax_situation,
            'recommended_strategies': ranked_strategies[:10],
            'execution_plan': execution_plan,
            'estimated_tax_savings': sum(s['tax_benefit'] for s in ranked_strategies[:10]),
            'implementation_timeline': self._create_timeline(execution_plan, days_to_year_end)
        }
    
    async def _analyze_current_tax_situation(self) -> Dict:
        """Analyze current year tax situation"""
        # Get YTD realized gains/losses
        portfolio = await self.mcp_client.call_tool(
            "mcp__ai-news-trader__get_portfolio_status",
            {"include_analytics": True}
        )
        
        # Calculate current tax position
        ytd_gains = sum(p.get('realized_gain', 0) for p in portfolio['positions'])
        ytd_losses = sum(p.get('realized_loss', 0) for p in portfolio['positions'])
        
        # Get unrealized positions
        unrealized_gains = sum(
            p['unrealized_pnl'] for p in portfolio['positions'] 
            if p['unrealized_pnl'] > 0
        )
        unrealized_losses = sum(
            p['unrealized_pnl'] for p in portfolio['positions'] 
            if p['unrealized_pnl'] < 0
        )
        
        # Estimate tax brackets
        current_bracket = self._estimate_tax_bracket(ytd_gains - ytd_losses)
        
        return {
            'ytd_realized_gains': ytd_gains,
            'ytd_realized_losses': ytd_losses,
            'net_realized': ytd_gains - ytd_losses,
            'unrealized_gains': unrealized_gains,
            'unrealized_losses': abs(unrealized_losses),
            'current_tax_bracket': current_bracket,
            'expected_next_year_bracket': self._project_next_year_bracket(),
            'carryover_losses': self._get_carryover_losses(),
            'days_to_year_end': (datetime(datetime.now().year, 12, 31) - datetime.now()).days
        }
```

### 6. Integration with Performance Tracking

```python
class TaxHarvestingPerformanceTracker:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.metrics_cache = {}
        
    async def track_harvest_performance(self, 
                                      harvest_id: str,
                                      harvest_details: Dict) -> Dict:
        """Track individual harvest performance"""
        # Record harvest execution
        execution_metrics = {
            'harvest_id': harvest_id,
            'execution_time': datetime.now(),
            'symbol_sold': harvest_details['sell_order']['symbol'],
            'symbol_bought': harvest_details['buy_order']['symbol'],
            'shares': harvest_details['sell_order']['quantity'],
            'tax_loss_realized': harvest_details['tax_loss'],
            'tax_benefit': harvest_details['tax_benefit']
        }
        
        # Monitor replacement security performance
        tracking_task = asyncio.create_task(
            self._monitor_replacement_performance(harvest_id, harvest_details)
        )
        
        # Calculate harvest efficiency
        efficiency_metrics = await self._calculate_harvest_efficiency(
            harvest_details
        )
        
        return {
            'harvest_id': harvest_id,
            'execution_metrics': execution_metrics,
            'efficiency_metrics': efficiency_metrics,
            'tracking_initiated': True
        }
    
    async def _monitor_replacement_performance(self, 
                                             harvest_id: str,
                                             harvest_details: Dict):
        """Monitor replacement security vs original"""
        original_symbol = harvest_details['sell_order']['symbol']
        replacement_symbol = harvest_details['buy_order']['symbol']
        start_date = harvest_details['execution_date']
        
        # Monitor for 31 days (wash sale period + 1)
        for day in range(32):
            await asyncio.sleep(86400)  # 24 hours
            
            # Get performance comparison
            performance = await self._compare_security_performance(
                original_symbol, replacement_symbol, start_date
            )
            
            # Store metrics
            self.metrics_cache[f"{harvest_id}_day_{day}"] = performance
            
            # Alert if significant tracking error
            if abs(performance['tracking_error']) > 0.05:  # 5%
                await self._send_tracking_error_alert(
                    harvest_id, performance
                )
    
    async def generate_harvest_report(self, period_days: int = 365) -> Dict:
        """Generate comprehensive tax harvesting performance report"""
        # Get all harvests in period
        harvests = await self._get_period_harvests(period_days)
        
        # Calculate aggregate metrics
        total_tax_benefit = sum(h['tax_benefit'] for h in harvests)
        total_trades = len(harvests)
        
        # Calculate tracking performance
        tracking_errors = []
        for harvest in harvests:
            if harvest['harvest_id'] in self.metrics_cache:
                final_metrics = self.metrics_cache[f"{harvest['harvest_id']}_day_31"]
                tracking_errors.append(final_metrics['tracking_error'])
        
        avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
        
        # Generate detailed report
        report = {
            'period_days': period_days,
            'total_harvests': total_trades,
            'total_tax_benefit': total_tax_benefit,
            'average_tax_benefit': total_tax_benefit / total_trades if total_trades > 0 else 0,
            'average_tracking_error': avg_tracking_error,
            'harvest_efficiency': self._calculate_overall_efficiency(harvests),
            'monthly_breakdown': self._generate_monthly_breakdown(harvests),
            'top_harvests': sorted(harvests, key=lambda x: x['tax_benefit'], reverse=True)[:10],
            'wash_sale_violations': self._count_wash_sale_violations(harvests),
            'replacement_performance': self._analyze_replacement_performance(harvests)
        }
        
        return report
```

## Monitoring and Alerting System

```python
class TaxHarvestingAlertSystem:
    def __init__(self, mcp_client, config: Dict):
        self.mcp_client = mcp_client
        self.alert_config = config
        self.alert_channels = config.get('channels', ['email', 'dashboard'])
        
    async def setup_automated_alerts(self):
        """Set up comprehensive alerting system"""
        alert_rules = [
            {
                'name': 'large_loss_opportunity',
                'condition': lambda opp: opp['tax_benefit'] > 1000,
                'priority': 'high',
                'message': 'Large tax loss harvesting opportunity detected'
            },
            {
                'name': 'wash_sale_warning',
                'condition': lambda pos: pos['days_until_wash_safe'] < 5,
                'priority': 'medium',
                'message': 'Approaching wash sale window expiration'
            },
            {
                'name': 'year_end_planning',
                'condition': lambda date: (datetime(date.year, 12, 31) - date).days < 30,
                'priority': 'high',
                'message': 'Year-end tax planning window opened'
            },
            {
                'name': 'tracking_error',
                'condition': lambda err: abs(err) > 0.05,
                'priority': 'medium',
                'message': 'Significant tracking error in harvest replacement'
            }
        ]
        
        # Start monitoring tasks
        for rule in alert_rules:
            asyncio.create_task(self._monitor_alert_condition(rule))
    
    async def _monitor_alert_condition(self, rule: Dict):
        """Monitor specific alert condition"""
        while True:
            try:
                # Check condition based on rule type
                if rule['name'] == 'large_loss_opportunity':
                    opportunities = await self._check_loss_opportunities()
                    for opp in opportunities:
                        if rule['condition'](opp):
                            await self._send_alert(rule, opp)
                
                elif rule['name'] == 'wash_sale_warning':
                    positions = await self._check_wash_sale_positions()
                    for pos in positions:
                        if rule['condition'](pos):
                            await self._send_alert(rule, pos)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                await self._handle_monitor_error(e, rule)
```

## Testing and Validation Framework

```python
class TaxHarvestingTester:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.test_scenarios = []
        
    async def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive test suite"""
        test_results = {
            'wash_sale_tests': await self._test_wash_sale_detection(),
            'execution_tests': await self._test_execution_strategies(),
            'compliance_tests': await self._test_compliance_engine(),
            'performance_tests': await self._test_performance_tracking(),
            'integration_tests': await self._test_mcp_integration()
        }
        
        return {
            'overall_status': 'passed' if all(
                r['passed'] for r in test_results.values()
            ) else 'failed',
            'test_results': test_results,
            'recommendations': self._generate_test_recommendations(test_results)
        }
    
    async def _test_wash_sale_detection(self) -> Dict:
        """Test wash sale detection accuracy"""
        test_cases = [
            {
                'name': 'simple_wash_sale',
                'transactions': [
                    {'date': '2024-01-01', 'symbol': 'AAPL', 'action': 'sell', 'shares': 100},
                    {'date': '2024-01-15', 'symbol': 'AAPL', 'action': 'buy', 'shares': 100}
                ],
                'expected': True
            },
            {
                'name': 'substantially_identical',
                'transactions': [
                    {'date': '2024-01-01', 'symbol': 'SPY', 'action': 'sell', 'shares': 100},
                    {'date': '2024-01-15', 'symbol': 'VOO', 'action': 'buy', 'shares': 100}
                ],
                'expected': True
            },
            {
                'name': 'outside_window',
                'transactions': [
                    {'date': '2024-01-01', 'symbol': 'AAPL', 'action': 'sell', 'shares': 100},
                    {'date': '2024-02-15', 'symbol': 'AAPL', 'action': 'buy', 'shares': 100}
                ],
                'expected': False
            }
        ]
        
        results = []
        for test in test_cases:
            result = await self._run_wash_sale_test(test)
            results.append(result)
        
        return {
            'passed': all(r['correct'] for r in results),
            'test_count': len(test_cases),
            'passed_count': sum(1 for r in results if r['correct']),
            'details': results
        }
```

## Best Practices and Optimization

### 1. Harvest Frequency Optimization
- Monitor daily but harvest strategically
- Batch small losses to reduce transaction costs
- Consider market volatility in timing decisions

### 2. Replacement Security Selection
- Maintain correlation above 0.80 for tracking
- Consider liquidity and transaction costs
- Use sector ETFs as temporary replacements

### 3. Tax Lot Management
- Use HIFO (Highest In, First Out) for maximum benefit
- Track specific lot identification
- Maintain detailed records for audit trail

### 4. Integration with Overall Strategy
- Coordinate with regular rebalancing
- Consider impact on factor exposures
- Monitor tracking error vs benchmark

### 5. Compliance Best Practices
- Automated daily wash sale checks
- Pre-trade compliance validation
- Comprehensive audit logging

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- Set up MCP client integration
- Implement tax lot tracking
- Build wash sale detection engine

### Phase 2: Automation (Weeks 3-4)
- Deploy real-time monitoring
- Implement execution strategies
- Set up compliance automation

### Phase 3: Optimization (Weeks 5-6)
- Add machine learning for timing
- Implement advanced replacement selection
- Optimize execution algorithms

### Phase 4: Testing and Deployment (Weeks 7-8)
- Comprehensive testing suite
- Performance validation
- Production deployment

## Conclusion

This automated tax harvesting system provides institutional-grade tax optimization integrated with the AI News Trading Platform. The system operates continuously to identify and execute tax-saving opportunities while maintaining investment objectives and regulatory compliance.