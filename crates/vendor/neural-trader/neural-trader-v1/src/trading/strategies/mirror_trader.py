"""Mirror Trading Strategy implementation with enhanced risk management."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExitReason(Enum):
    """Exit reason enumeration."""
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    PARTIAL_PROFIT = "partial_profit"
    INSTITUTIONAL_EXIT = "institutional_exit"
    RISK_LIMIT = "risk_limit"
    CORRELATION_LIMIT = "correlation_limit"


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    total_exposure: float
    var_95: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    sector_concentration: Dict[str, float]
    position_count: int
    avg_correlation: float


@dataclass
class PositionRisk:
    """Individual position risk metrics."""
    ticker: str
    size_pct: float
    volatility: float
    beta: float
    correlation_score: float
    stop_loss_price: float
    trailing_stop_price: Optional[float]
    risk_level: RiskLevel


class MirrorTradingEngine:
    """Engine for mirror trading strategy following institutional investors."""
    
    def __init__(self, portfolio_size: float = 100000):
        """
        Initialize mirror trading engine with enhanced risk management.
        
        Args:
            portfolio_size: Total portfolio size for position sizing
        """
        self.portfolio_size = portfolio_size
        # OPTIMIZED PARAMETERS - Improved through systematic optimization
        self.trusted_institutions = {
            "Berkshire Hathaway": 0.7052,    # Optimized from 0.95
            "Bridgewater Associates": 0.9047,  # Optimized from 0.85
            "Renaissance Technologies": 0.9307, # Optimized from 0.90
            "Soros Fund Management": 0.8437,   # Optimized from 0.80
            "Tiger Global": 0.6529,           # Optimized from 0.75
            "Third Point": 0.5062,            # Optimized from 0.70
            "Pershing Square": 0.6793,        # Optimized from 0.75
            "Appaloosa Management": 0.7975    # Optimized from 0.80
        }
        
        # Enhanced position sizing parameters
        # OPTIMIZED POSITION SIZING - Increased max position for better returns
        self.max_position_pct = 0.0329  # Optimized from 0.03 (3.29%)
        self.min_position_pct = 0.0051  # Optimized from 0.005 (0.51%)
        self.max_portfolio_exposure = 0.85  # Max 85% total exposure
        self.max_sector_exposure = 0.25  # Max 25% per sector
        self.volatility_target = 0.15  # 15% annual volatility target
        
        # Risk monitoring parameters
        self.max_correlation = 0.7  # Max correlation between positions
        self.drawdown_limit = 0.20  # Max 20% drawdown before position reduction
        self.var_confidence = 0.95  # 95% VaR confidence level
        
        # Portfolio state tracking
        self.current_positions = {}
        self.daily_pnl = []
        self.risk_metrics = {}
        self.correlation_matrix = {}
        
    def parse_13f_filing(self, filing: Dict) -> List[Dict]:
        """
        Parse 13F filing and generate mirror signals.
        
        Args:
            filing: 13F filing data
            
        Returns:
            List of mirror trading signals
        """
        signals = []
        institution = filing["filer"]
        confidence = self.trusted_institutions.get(institution, 0.5)
        
        # High priority: New positions by trusted institutions
        for ticker in filing.get("new_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence,
                "priority": "high",
                "reasoning": f"{institution} initiated new position",
                "mirror_type": "new_position",
                "institution": institution,
                "filing_type": "13F"
            })
            
        # Medium priority: Increased positions
        for ticker in filing.get("increased_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence * 0.9708,  # Optimized multiplier
                "priority": "medium",
                "reasoning": f"{institution} increased position",
                "mirror_type": "add_position",
                "institution": institution,
                "filing_type": "13F"
            })
            
        # Sell signals from eliminated positions
        for ticker in filing.get("sold_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "sell",
                "confidence": confidence * 0.7726,  # Optimized multiplier
                "priority": "high",
                "reasoning": f"{institution} eliminated position",
                "mirror_type": "exit_position",
                "institution": institution,
                "filing_type": "13F"
            })
            
        # Lower priority: Reduced positions
        for ticker in filing.get("reduced_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "reduce",
                "confidence": confidence * 0.4128,  # Optimized multiplier
                "priority": "low",
                "reasoning": f"{institution} reduced position",
                "mirror_type": "trim_position",
                "institution": institution,
                "filing_type": "13F"
            })
            
        return signals
        
    def get_institution_confidence(self, institution: str, recent_trades: List[Dict] = None) -> Dict:
        """Enhanced Bayesian confidence scoring with dynamic updates."""
        # Base confidence from trusted institutions
        base_confidence = self.trusted_institutions.get(institution, 0.5)
        
        # Bayesian parameters (alpha, beta for Beta distribution)
        alpha = base_confidence * 100  # Convert to Beta parameters
        beta = (1 - base_confidence) * 100
        
        # Update with recent performance if available
        if recent_trades:
            successes = sum(1 for trade in recent_trades if trade.get("success", False))
            failures = len(recent_trades) - successes
            alpha += successes
            beta += failures
            
        # Calculate enhanced confidence
        confidence_mean = alpha / (alpha + beta)
        confidence_variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        confidence_std = math.sqrt(confidence_variance)
        
        # 95% confidence interval
        confidence_95_lower = stats.beta.ppf(0.025, alpha, beta)
        confidence_95_upper = stats.beta.ppf(0.975, alpha, beta)
        
        return {
            "confidence_score": confidence_mean,
            "confidence_std": confidence_std,
            "confidence_95_ci": [confidence_95_lower, confidence_95_upper],
            "alpha": alpha,
            "beta": beta,
            "trades_considered": len(recent_trades) if recent_trades else 0,
            "method": "Enhanced Bayesian Beta Distribution"
        }
        
    def score_insider_transaction(self, insider_filing: Dict) -> float:
        """
        Score insider transaction confidence.
        
        Args:
            insider_filing: Form 4 insider transaction data
            
        Returns:
            Confidence score for the insider transaction
        """
        role = insider_filing["role"].lower()
        transaction_type = insider_filing["transaction_type"].lower()
        shares = insider_filing["shares"]
        
        # Base scores by role
        role_scores = {
            "ceo": 0.9,
            "cfo": 0.8,
            "president": 0.85,
            "director": 0.7,
            "officer": 0.65,
            "10% owner": 0.75
        }
        
        base_score = 0.6  # Default
        for role_key, score in role_scores.items():
            if role_key in role:
                base_score = score
                break
                
        # Adjust based on transaction type
        if "purchase" in transaction_type or "buy" in transaction_type:
            confidence = base_score
        elif "sale" in transaction_type or "sell" in transaction_type:
            confidence = 1.0 - base_score  # Inverse for sales
        else:
            confidence = 0.5  # Neutral for other transactions
            
        # Adjust based on transaction size
        if shares > 100000:  # Large transaction
            confidence *= 1.1
        elif shares < 1000:  # Small transaction
            confidence *= 0.8
            
        return max(0, min(confidence, 1.0))
        
    def calculate_mirror_position(self, institutional_trade: Dict, portfolio_volatility: float = 0.15) -> Dict:
        """
        Calculate optimal position size using enhanced Kelly Criterion.
        
        Args:
            institutional_trade: Institutional trade details
            portfolio_volatility: Current portfolio volatility
            
        Returns:
            Optimized position sizing recommendation
        """
        inst_position_pct = institutional_trade["position_size_pct"]
        confidence_data = self.get_institution_confidence(
            institutional_trade["institution"], 
            institutional_trade.get("recent_trades", [])
        )
        confidence = confidence_data["confidence_score"]
        
        # Enhanced Kelly Criterion position sizing
        expected_return = institutional_trade.get("expected_return", 0.08)
        expected_volatility = institutional_trade.get("expected_volatility", 0.20)
        
        # Kelly fraction calculation
        win_probability = confidence
        loss_probability = 1 - confidence
        
        if expected_volatility > 0:
            odds = expected_return / expected_volatility
        else:
            odds = 0.5
            
        # Kelly fraction: f = (bp - q) / b
        kelly_fraction = (odds * win_probability - loss_probability) / odds
        kelly_fraction = max(0, kelly_fraction)
        
        # Risk adjustments
        vol_adjustment = 0.15 / portfolio_volatility  # Target 15% portfolio vol
        confidence_adjustment = min(1.0, confidence * 2)
        
        # Optimal position size
        optimal_fraction = kelly_fraction * vol_adjustment * confidence_adjustment
        optimal_fraction = min(optimal_fraction, self.max_position_pct)
        optimal_fraction = max(optimal_fraction, self.min_position_pct)
        
        position_dollars = self.portfolio_size * optimal_fraction
        
        return {
            "size_pct": optimal_fraction,
            "size_dollars": position_dollars,
            "reasoning": "Kelly Criterion with risk adjustments",
            "confidence": confidence,
            "confidence_ci": confidence_data["confidence_95_ci"],
            "kelly_fraction": kelly_fraction,
            "vol_adjustment": vol_adjustment,
            "expected_holding_period": self._estimate_holding_period(
                institutional_trade["institution"]
            ),
            "institutional_commitment": inst_position_pct,
            "optimization_method": "Enhanced Kelly Criterion"
        }
        
    def _estimate_holding_period(self, institution: str) -> str:
        """Estimate holding period based on institution's style."""
        long_term_investors = ["Berkshire Hathaway", "Pershing Square"]
        medium_term = ["Tiger Global", "Third Point"]
        
        if institution in long_term_investors:
            return "6-24 months"
        elif institution in medium_term:
            return "3-12 months"
        else:
            return "1-6 months"
            
    async def determine_entry_timing(self, filing_data: Dict, market_data: Dict = None) -> Dict:
        """
        Multi-factor entry timing optimization.
        
        Args:
            filing_data: Filing timing and price data
            market_data: Additional market data for factors
            
        Returns:
            Enhanced entry timing strategy
        """
        filing_date = filing_data["filing_date"]
        current_price = filing_data["current_price"]
        filing_price = filing_data["filing_price"]
        days_since_filing = filing_data.get("days_since_filing", 
                                          (datetime.now() - filing_date).days)
        
        if market_data is None:
            market_data = {}
            
        # Multi-factor analysis
        factors = {}
        
        # 1. Price momentum (simplified)
        price_change = (current_price - filing_price) / filing_price
        factors["price_deviation"] = price_change
        
        # 2. Time decay factor
        time_decay = max(0, 1.0 - (days_since_filing / 30))
        factors["time_decay"] = time_decay
        
        # 3. Volume factor (if available)
        volume_data = market_data.get("volume_data", [])
        if volume_data and len(volume_data) >= 20:
            avg_volume = sum(volume_data[-20:]) / 20
            current_volume = volume_data[-1]
            volume_ratio = current_volume / avg_volume
            factors["volume_ratio"] = min(2.0, volume_ratio)  # Cap at 2x
        else:
            factors["volume_ratio"] = 1.0
            
        # 4. Volatility factor (simplified)
        price_history = market_data.get("price_history", [])
        if len(price_history) >= 10:
            returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                      for i in range(1, min(11, len(price_history)))]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.02
            factors["volatility"] = volatility
        else:
            factors["volatility"] = 0.02
            
        # Calculate multi-factor timing score
        timing_score = self._calculate_multi_factor_timing_score(factors)
        
        # Determine strategy based on enhanced score
        if timing_score > 0.8:
            entry_strategy = "immediate"
            urgency = "high"
            max_chase_price = current_price * 1.015
        elif timing_score > 0.6:
            entry_strategy = "prompt"
            urgency = "medium"
            max_chase_price = current_price * 1.02
        elif timing_score > 0.4:
            entry_strategy = "patient"
            urgency = "low"
            max_chase_price = current_price * 1.01
        else:
            entry_strategy = "wait"
            urgency = "very_low"
            max_chase_price = current_price * 0.99
            
        return {
            "entry_strategy": entry_strategy,
            "urgency": urgency,
            "max_chase_price": max_chase_price,
            "price_deviation": price_change,
            "days_since_filing": days_since_filing,
            "timing_score": timing_score,
            "factors": factors,
            "optimization_method": "Multi-factor timing analysis",
            "basic_timing_score": self._calculate_timing_score(days_since_filing, price_change)
        }
        
    def _calculate_timing_score(self, days: float, price_change: float) -> float:
        """Calculate timing score (higher is better)."""
        # Penalty for time elapsed
        time_score = max(0, 1.0 - (days / 14))  # Decreases over 2 weeks
        
        # Penalty for price movement
        price_score = max(0, 1.0 - abs(price_change) * 5)  # Penalty for deviation
        
        return (time_score + price_score) / 2
        
    def _calculate_multi_factor_timing_score(self, factors: Dict) -> float:
        """Calculate enhanced multi-factor timing score with weighted components."""
        weights = {
            "time_decay": 0.25,
            "price_deviation": 0.20,  # Negative impact
            "volume_ratio": 0.25,
            "volatility": 0.15,  # Negative impact
            "momentum": 0.15
        }
        
        score = 0
        
        # Time decay (higher is better)
        score += weights["time_decay"] * factors.get("time_decay", 0.5)
        
        # Price deviation (closer to 0 is better)
        price_dev = abs(factors.get("price_deviation", 0))
        price_score = max(0, 1.0 - price_dev * 10)
        score += weights["price_deviation"] * price_score
        
        # Volume ratio (around 1.5 is optimal)
        volume_ratio = factors.get("volume_ratio", 1.0)
        volume_score = min(1.0, volume_ratio / 1.5) if volume_ratio <= 1.5 else max(0, 2.0 - volume_ratio)
        score += weights["volume_ratio"] * volume_score
        
        # Volatility (lower is better for entry)
        volatility = factors.get("volatility", 0.02)
        vol_score = max(0, 1.0 - (volatility / 0.05))  # Normalize around 5%
        score += weights["volatility"] * vol_score
        
        # Momentum (if available)
        momentum = factors.get("momentum_5d", 0)
        momentum_score = max(0, min(1.0, (momentum + 0.02) / 0.04))  # Normalize around 2%
        score += weights["momentum"] * momentum_score
        
        return max(0, min(1.0, score))
        
    def parse_form_4_filing(self, form_4_data: Dict) -> Dict:
        """
        Parse Form 4 insider transaction filing.
        
        Args:
            form_4_data: Form 4 filing data
            
        Returns:
            Insider trading signal
        """
        confidence = self.score_insider_transaction(form_4_data)
        transaction_type = form_4_data["transaction_type"].lower()
        role = form_4_data["role"].lower()
        
        # Determine signal strength
        if confidence > 0.8:
            signal_strength = "strong"
            position_multiplier = 1.2
        elif confidence > 0.6:
            signal_strength = "moderate"
            position_multiplier = 1.0
        else:
            signal_strength = "weak"
            position_multiplier = 0.8
            
        # Determine action
        if "purchase" in transaction_type and confidence > 0.7:
            action = "buy"
        elif "sale" in transaction_type and confidence > 0.7:
            action = "sell"
        else:
            action = "neutral"
            
        return {
            "ticker": form_4_data["ticker"],
            "action": action,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "position_size_multiplier": position_multiplier,
            "insider_role": role,
            "transaction_type": transaction_type,
            "reasoning": f"{role} {transaction_type} - confidence {confidence:.2f}"
        }
        
    def analyze_institutional_track_record(self, track_record: Dict) -> Dict:
        """
        Analyze institutional track record for confidence scoring.
        
        Args:
            track_record: Historical performance data
            
        Returns:
            Track record analysis
        """
        institution = track_record["institution"]
        annual_returns = track_record["last_5_years"]["annual_returns"]
        winning_positions = track_record["last_5_years"]["winning_positions"]
        total_positions = track_record["last_5_years"]["total_positions"]
        recent_performance = track_record["recent_performance"]["last_12_months"]
        vs_sp500 = track_record["recent_performance"]["vs_sp500"]
        
        # Calculate metrics
        avg_return = statistics.mean(annual_returns)
        return_volatility = statistics.stdev(annual_returns) if len(annual_returns) > 1 else 0.1
        win_rate = winning_positions / total_positions if total_positions > 0 else 0
        
        # Consistency score (low volatility is good)
        consistency_score = max(0, 1.0 - (return_volatility / 0.12))  # Normalize around 12% vol
        
        # Performance score
        performance_score = min((avg_return + 0.05) / 0.15, 1.0)  # Normalize around 10% returns
        
        # Recent performance factor
        recent_factor = min((recent_performance + 0.05) / 0.15, 1.0)
        
        # Alpha factor (vs S&P 500)
        alpha_factor = min((vs_sp500 + 0.02) / 0.08, 1.0)  # Normalize around 3% alpha
        
        # Combined confidence score
        confidence_score = (
            performance_score * 0.3 +
            consistency_score * 0.25 +
            (win_rate * 1.25) * 0.25 +  # Scale win rate
            recent_factor * 0.1 +
            alpha_factor * 0.1
        )
        
        # Recommended follow percentage
        if confidence_score > 0.8:
            recommended_follow_pct = 0.9
        elif confidence_score > 0.6:
            recommended_follow_pct = 0.7
        elif confidence_score > 0.4:
            recommended_follow_pct = 0.5
        else:
            recommended_follow_pct = 0.3
            
        return {
            "institution": institution,
            "confidence_score": confidence_score,
            "win_rate": win_rate,
            "avg_annual_return": avg_return,
            "consistency_score": consistency_score,
            "recent_performance": recent_performance,
            "alpha_vs_market": vs_sp500,
            "recommended_follow_pct": recommended_follow_pct
        }
        
    def analyze_portfolio_overlap(self, our_portfolio: Dict, institutional_portfolio: Dict) -> Dict:
        """
        Analyze overlap between our portfolio and institutional portfolio.
        
        Args:
            our_portfolio: Our current holdings (ticker -> weight)
            institutional_portfolio: Institutional holdings (ticker -> weight)
            
        Returns:
            Portfolio overlap analysis
        """
        our_tickers = set(our_portfolio.keys())
        inst_tickers = set(institutional_portfolio.keys())
        
        # Calculate overlap
        common_tickers = our_tickers.intersection(inst_tickers)
        overlap_value = sum(min(our_portfolio[ticker], institutional_portfolio[ticker]) 
                          for ticker in common_tickers)
        
        # Calculate overlap percentage
        our_total = sum(our_portfolio.values())
        overlap_pct = overlap_value / our_total if our_total > 0 else 0
        
        # Identify differences
        missing_positions = list(inst_tickers - our_tickers)
        our_unique_positions = list(our_tickers - inst_tickers)
        
        # Generate recommendations
        recommendations = []
        for ticker in missing_positions:
            inst_weight = institutional_portfolio[ticker]
            if inst_weight > 0.05:  # Institution has >5% position
                recommendations.append({
                    "ticker": ticker,
                    "action": "consider_buy",
                    "institutional_weight": inst_weight,
                    "suggested_weight": min(inst_weight * 0.5, 0.03),  # Scale down
                    "reasoning": f"Institution has {inst_weight:.1%} position"
                })
                
        return {
            "overlap_pct": overlap_pct,
            "common_positions": list(common_tickers),
            "missing_positions": missing_positions,
            "our_unique_positions": our_unique_positions,
            "recommendations": recommendations,
            "overlap_value": overlap_value,
            "portfolio_alignment": "high" if overlap_pct > 0.5 else "medium" if overlap_pct > 0.3 else "low"
        }
        
    def assess_mirror_risk(self, position: Dict) -> Dict:
        """
        Assess risk for mirror trading positions.
        
        Args:
            position: Current mirror position details
            
        Returns:
            Risk assessment and recommended action
        """
        institutional_status = position["institutional_status"]
        current_price = position["current_price"]
        entry_price = position["entry_price"]
        days_held = position["days_held"]
        
        # Calculate return
        position_return = (current_price - entry_price) / entry_price
        
        # Risk assessment based on institutional status
        if institutional_status == "exited":
            action = "exit"
            reason = "institutional_exit"
            urgency = "high"
        elif institutional_status == "reduced" and position_return < -0.05:
            action = "exit"
            reason = "institutional_reduction_with_loss"
            urgency = "high"
        elif institutional_status == "increased" and position_return <= -0.10:
            action = "hold"  # Institution doubling down
            reason = "institutional_confidence"
            urgency = "low"
        elif position_return > 0.1689:  # 16.89% gain - OPTIMIZED
            action = "reduce"
            reason = "take_profits"
            urgency = "medium"
        elif position_return < -0.2329:  # 23.29% loss - OPTIMIZED
            action = "exit"
            reason = "stop_loss"
            urgency = "high"
        elif days_held > 365 and position_return > 0.15:  # Long hold with gains
            action = "reduce"
            reason = "long_term_profits"
            urgency = "low"
        else:
            action = "hold"
            reason = "position_healthy"
            urgency = "low"
            
        return {
            "action": action,
            "reason": reason,
            "urgency": urgency,
            "position_return": position_return,
            "institutional_status": institutional_status,
            "risk_level": "high" if urgency == "high" else "medium" if urgency == "medium" else "low"
        }
        
    def analyze_sector_flows(self, sector_flows: Dict) -> Dict:
        """
        Analyze institutional flows by sector.
        
        Args:
            sector_flows: Sector flow data
            
        Returns:
            Sector flow analysis
        """
        # Find strongest flows
        flows = [(sector, data["net_flow"]) for sector, data in sector_flows.items()]
        flows.sort(key=lambda x: x[1], reverse=True)
        
        strongest_inflow = flows[0]
        strongest_outflow = flows[-1]
        
        # Identify recommended sectors
        recommended_sectors = []
        avoid_sectors = []
        
        for sector, data in sector_flows.items():
            net_flow = data["net_flow"]
            consistency = data["flow_consistency"]
            
            if net_flow > 1000000000 and consistency > 0.7:  # >$1B with consistency
                recommended_sectors.append(sector)
            elif net_flow < -1000000000 and consistency > 0.6:  # <-$1B outflow
                avoid_sectors.append(sector)
                
        return {
            "strongest_inflow_sector": strongest_inflow[0],
            "strongest_inflow_amount": strongest_inflow[1],
            "strongest_outflow_sector": strongest_outflow[0],
            "strongest_outflow_amount": strongest_outflow[1],
            "recommended_sectors": recommended_sectors,
            "avoid_sectors": avoid_sectors,
            "total_sectors_analyzed": len(sector_flows)
        }
        
    def track_mirror_performance(self, mirror_trades: List[Dict]) -> Dict:
        """
        Track performance of mirror trades vs institutional performance.
        
        Args:
            mirror_trades: List of mirror trades with performance data
            
        Returns:
            Performance tracking analysis
        """
        if not mirror_trades:
            return {"error": "No trades to analyze"}
            
        our_returns = []
        institutional_returns = []
        individual_trades = []
        
        for trade in mirror_trades:
            # Our return
            our_return = (trade["current_price"] - trade["entry_price"]) / trade["entry_price"]
            our_returns.append(our_return)
            
            # Institutional return
            inst_return = (trade["institutional_current"] - trade["institutional_entry"]) / trade["institutional_entry"]
            institutional_returns.append(inst_return)
            
            # Individual trade tracking
            individual_trades.append({
                "ticker": trade["ticker"],
                "institution": trade["institution"],
                "our_return": our_return,
                "institutional_return": inst_return,
                "tracking_difference": our_return - inst_return,
                "entry_timing_diff": (trade["entry_price"] - trade["institutional_entry"]) / trade["institutional_entry"]
            })
            
        # Calculate aggregate metrics
        our_avg_return = statistics.mean(our_returns)
        institutional_avg_return = statistics.mean(institutional_returns)
        
        # Tracking efficiency (how well we track institutional returns)
        tracking_differences = [abs(our - inst) for our, inst in zip(our_returns, institutional_returns)]
        tracking_efficiency = 1.0 - (statistics.mean(tracking_differences) / 0.1)  # Normalize
        tracking_efficiency = max(0, min(tracking_efficiency, 1.0))
        
        return {
            "our_avg_return": our_avg_return,
            "institutional_avg_return": institutional_avg_return,
            "tracking_efficiency": tracking_efficiency,
            "total_trades": len(mirror_trades),
            "individual_trades": individual_trades,
            "outperformance": our_avg_return - institutional_avg_return,
            "correlation": self._calculate_correlation(our_returns, institutional_returns)
        }
        
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two return series."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0
        
    async def monitor_filing_alerts(self, recent_filings: List[Dict]) -> List[Dict]:
        """
        Monitor recent filings for trading alerts.
        
        Args:
            recent_filings: List of recent institutional filings
            
        Returns:
            List of trading alerts
        """
        alerts = []
        
        for filing in recent_filings:
            institution = filing["institution"]
            confidence = self.get_institution_confidence(institution)
            filing_age_hours = (datetime.now() - filing["filed_date"]).total_seconds() / 3600
            
            # Process new positions
            for ticker in filing.get("new_positions", []):
                priority = "high" if confidence > 0.8 and filing_age_hours < 24 else "medium"
                alerts.append({
                    "ticker": ticker,
                    "action": "buy",
                    "institution": institution,
                    "confidence": confidence,
                    "priority": priority,
                    "filing_type": filing["filing_type"],
                    "alert_type": "new_position",
                    "filing_age_hours": filing_age_hours
                })
                
            # Process position changes
            for ticker, change in filing.get("position_changes", {}).items():
                if change == "increased":
                    action = "buy"
                    alert_type = "position_increase"
                elif change == "exited":
                    action = "sell"
                    alert_type = "position_exit"
                else:
                    action = "review"
                    alert_type = "position_change"
                    
                priority = "high" if action == "sell" else "medium"
                alerts.append({
                    "ticker": ticker,
                    "action": action,
                    "institution": institution,
                    "confidence": confidence,
                    "priority": priority,
                    "filing_type": filing["filing_type"],
                    "alert_type": alert_type,
                    "position_change": change,
                    "filing_age_hours": filing_age_hours
                })
                
        # Sort by priority and confidence
        alerts.sort(key=lambda x: (x["priority"] == "high", x["confidence"]), reverse=True)
        
        return alerts
        
    def determine_exit_strategy(self, position: Dict) -> Dict:
        """
        Determine exit strategy for mirror position.
        
        Args:
            position: Position details
            
        Returns:
            Exit strategy decision
        """
        ticker = position["ticker"]
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        institution_status = position.get("institution_status", "holding")
        
        # Calculate return
        position_return = (current_price - entry_price) / entry_price
        
        # Exit decision logic
        if institution_status == "exited":
            action = "exit"
            reason = "institutional_exit"
        elif position_return >= 0.1689:  # 16.89% gain - OPTIMIZED
            action = "reduce"
            reason = "take_profits"
        elif position_return <= -0.2329:  # 23.29% loss - OPTIMIZED
            action = "exit"
            reason = "stop_loss"
        else:
            action = "hold"
            reason = "normal_conditions"
            
        return {
            "ticker": ticker,
            "action": action,
            "reason": reason,
            "position_return": position_return,
            "current_price": current_price,
            "entry_price": entry_price
        }
        
    def analyze_return_correlation(self, return_data: Dict) -> Dict:
        """
        Analyze correlation between our returns and institutional returns.
        
        Args:
            return_data: Return series data
            
        Returns:
            Correlation analysis
        """
        our_returns = return_data["our_returns"]
        institutional_returns = return_data["institutional_returns"]
        market_returns = return_data["market_returns"]
        
        # Calculate correlations
        our_vs_institutional = self._calculate_correlation(our_returns, institutional_returns)
        our_vs_market = self._calculate_correlation(our_returns, market_returns)
        institutional_vs_market = self._calculate_correlation(institutional_returns, market_returns)
        
        # Calculate tracking error
        tracking_differences = [abs(our - inst) for our, inst in zip(our_returns, institutional_returns)]
        tracking_error = statistics.stdev(tracking_differences) if len(tracking_differences) > 1 else 0
        
        # Mirror effectiveness score
        mirror_effectiveness = our_vs_institutional * (1 - tracking_error / 0.05)  # Penalize high tracking error
        mirror_effectiveness = max(0, min(mirror_effectiveness, 1.0))
        
        return {
            "our_vs_institutional": our_vs_institutional,
            "our_vs_market": our_vs_market,
            "institutional_vs_market": institutional_vs_market,
            "tracking_error": tracking_error,
            "mirror_effectiveness": mirror_effectiveness,
            "avg_our_return": statistics.mean(our_returns),
            "avg_institutional_return": statistics.mean(institutional_returns),
            "return_consistency": 1.0 - (statistics.stdev(our_returns) / abs(statistics.mean(our_returns)) if statistics.mean(our_returns) != 0 else 1.0)
        }