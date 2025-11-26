"""Optimized Mirror Trading Strategy implementation with performance enhancements."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
import pandas as pd
import numpy as np
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import threading


def cache_with_ttl(seconds: int = 300):
    """Cache decorator with time-to-live."""
    def decorator(func):
        cache = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = datetime.now()
            
            with lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if (now - timestamp).total_seconds() < seconds:
                        return result
                
                result = func(*args, **kwargs)
                cache[key] = (result, now)
                
                # Clean old entries (simple cleanup)
                if len(cache) > 1000:
                    cutoff = now - timedelta(seconds=seconds)
                    cache.clear()  # Simple cleanup - clear all if too big
                
                return result
        return wrapper
    return decorator


class OptimizedMirrorTradingEngine:
    """Optimized engine for mirror trading strategy with performance enhancements."""
    
    def __init__(self, portfolio_size: float = 100000):
        """
        Initialize optimized mirror trading engine.
        
        Args:
            portfolio_size: Total portfolio size for position sizing
        """
        self.portfolio_size = portfolio_size
        
        # Optimized institution confidence scores with better granularity
        self.trusted_institutions = {
            "Berkshire Hathaway": 0.98,      # Increased from 0.95
            "Renaissance Technologies": 0.95,  # Increased from 0.90  
            "Bridgewater Associates": 0.88,   # Increased from 0.85
            "Tiger Global": 0.82,            # Increased from 0.75
            "Appaloosa Management": 0.85,    # Increased from 0.80
            "Soros Fund Management": 0.78,   # Decreased from 0.80 (more volatile)
            "Pershing Square": 0.80,         # Increased from 0.75
            "Third Point": 0.75,             # Increased from 0.70
            "Elliott Management": 0.90,      # New high-performance fund
            "Icahn Enterprises": 0.72,       # New activist fund
        }
        
        # Optimized position sizing parameters
        self.max_position_pct = 0.035       # Increased from 0.03 (3.5% max)
        self.min_position_pct = 0.003       # Decreased from 0.005 (0.3% min)
        self.position_size_multiplier = 0.25 # Increased from 0.2
        
        # Enhanced risk management parameters
        self.stop_loss_threshold = -0.12    # Improved from -0.15 (12% stop loss)
        self.profit_taking_threshold = 0.35 # Improved from 0.30 (35% profit taking)
        self.trailing_stop_pct = 0.08       # New: 8% trailing stop
        
        # Performance caching
        self._confidence_cache = {}
        self._correlation_cache = {}
        self._timing_cache = {}
        
        # Vectorized operations setup
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @lru_cache(maxsize=256)
    def get_institution_confidence(self, institution: str) -> float:
        """Get confidence score for institution with caching."""
        return self.trusted_institutions.get(institution, 0.5)
    
    def parse_13f_filing_optimized(self, filing: Dict) -> List[Dict]:
        """
        Optimized 13F filing parsing with vectorized operations.
        
        Args:
            filing: 13F filing data
            
        Returns:
            List of mirror trading signals
        """
        signals = []
        institution = filing["filer"]
        base_confidence = self.get_institution_confidence(institution)
        
        # Batch process different position types
        position_types = [
            ("new_positions", "buy", 1.0, "high", "new_position"),
            ("increased_positions", "buy", 0.85, "medium", "add_position"),  # Increased from 0.8
            ("sold_positions", "sell", 0.95, "high", "exit_position"),      # Increased from 0.9
            ("reduced_positions", "reduce", 0.65, "low", "trim_position")   # Increased from 0.6
        ]
        
        for position_key, action, confidence_mult, priority, mirror_type in position_types:
            tickers = filing.get(position_key, [])
            if tickers:
                # Vectorized signal generation
                batch_signals = [
                    {
                        "ticker": ticker,
                        "action": action,
                        "confidence": base_confidence * confidence_mult,
                        "priority": priority,
                        "reasoning": f"{institution} {mirror_type.replace('_', ' ')}",
                        "mirror_type": mirror_type,
                        "institution": institution,
                        "filing_type": "13F",
                        "signal_strength": self._calculate_signal_strength(base_confidence * confidence_mult)
                    }
                    for ticker in tickers
                ]
                signals.extend(batch_signals)
        
        return signals
    
    @cache_with_ttl(seconds=600)  # 10-minute cache for signal strength
    def _calculate_signal_strength(self, confidence: float) -> str:
        """Calculate signal strength based on confidence with caching."""
        if confidence >= 0.85:
            return "very_strong"
        elif confidence >= 0.75:
            return "strong"
        elif confidence >= 0.65:
            return "moderate"
        elif confidence >= 0.50:
            return "weak"
        else:
            return "very_weak"
    
    def score_insider_transaction_optimized(self, insider_filing: Dict) -> float:
        """
        Optimized insider transaction scoring with enhanced logic.
        
        Args:
            insider_filing: Form 4 insider transaction data
            
        Returns:
            Enhanced confidence score for the insider transaction
        """
        role = insider_filing["role"].lower()
        transaction_type = insider_filing["transaction_type"].lower()
        shares = insider_filing["shares"]
        
        # Enhanced role scoring with more granular weights
        role_scores = {
            "ceo": 0.95,              # Increased from 0.9
            "founder": 0.93,          # New role
            "chairman": 0.90,         # New role
            "president": 0.88,        # Increased from 0.85
            "cfo": 0.85,              # Increased from 0.8
            "coo": 0.82,              # New role
            "cto": 0.80,              # New role
            "director": 0.72,         # Increased from 0.7
            "officer": 0.68,          # Increased from 0.65
            "10% owner": 0.78,        # Increased from 0.75
            "insider": 0.60           # Default insider
        }
        
        base_score = 0.6  # Default
        for role_key, score in role_scores.items():
            if role_key in role:
                base_score = score
                break
        
        # Enhanced transaction type analysis
        confidence = base_score
        if "purchase" in transaction_type or "buy" in transaction_type:
            confidence = base_score
        elif "sale" in transaction_type or "sell" in transaction_type:
            # More nuanced selling analysis
            if "exercise" in transaction_type:
                confidence = base_score * 0.7  # Exercise + sale less bearish
            else:
                confidence = 1.0 - base_score * 0.8  # Regular sale more bearish
        elif "gift" in transaction_type:
            confidence = 0.5  # Neutral
        else:
            confidence = 0.5
        
        # Enhanced size-based adjustments
        if shares > 500000:  # Very large transaction
            confidence *= 1.15
        elif shares > 100000:  # Large transaction
            confidence *= 1.1
        elif shares > 50000:   # Medium transaction
            confidence *= 1.05
        elif shares < 5000:    # Very small transaction
            confidence *= 0.85
        elif shares < 1000:    # Tiny transaction
            confidence *= 0.75
        
        return max(0, min(confidence, 1.0))
    
    def calculate_mirror_position_optimized(self, institutional_trade: Dict) -> Dict:
        """
        Optimized position sizing with enhanced risk management.
        
        Args:
            institutional_trade: Institutional trade details
            
        Returns:
            Enhanced position sizing recommendation
        """
        inst_position_pct = institutional_trade["position_size_pct"]
        institution = institutional_trade["institution"]
        confidence = self.get_institution_confidence(institution)
        
        # Enhanced position sizing with confidence and volatility adjustments
        base_position_pct = min(
            inst_position_pct * self.position_size_multiplier * confidence,
            self.max_position_pct
        )
        
        # Volatility adjustment (if available)
        volatility_adj = institutional_trade.get("volatility_factor", 1.0)
        adjusted_position_pct = base_position_pct / volatility_adj
        
        # Ensure minimum position size
        our_position_pct = max(adjusted_position_pct, self.min_position_pct)
        
        # Enhanced risk metrics
        position_dollars = self.portfolio_size * our_position_pct
        
        # Calculate enhanced metrics
        risk_score = self._calculate_position_risk_score(
            our_position_pct, confidence, volatility_adj
        )
        
        return {
            "size_pct": our_position_pct,
            "size_dollars": position_dollars,
            "reasoning": "Enhanced position sizing with volatility adjustment",
            "confidence": confidence,
            "risk_score": risk_score,
            "volatility_factor": volatility_adj,
            "expected_holding_period": self._estimate_holding_period_enhanced(institution),
            "institutional_commitment": inst_position_pct,
            "stop_loss_price": self._calculate_stop_loss_price(institutional_trade),
            "target_price": self._calculate_target_price(institutional_trade)
        }
    
    def _calculate_position_risk_score(self, position_pct: float, confidence: float, volatility: float) -> float:
        """Calculate comprehensive position risk score."""
        size_risk = min(position_pct / self.max_position_pct, 1.0)
        confidence_risk = 1.0 - confidence
        volatility_risk = min(volatility / 2.0, 1.0)
        
        return (size_risk * 0.4 + confidence_risk * 0.4 + volatility_risk * 0.2)
    
    def _calculate_stop_loss_price(self, trade: Dict) -> float:
        """Calculate dynamic stop loss price."""
        entry_price = trade.get("entry_price", trade.get("current_price", 100))
        volatility = trade.get("volatility_factor", 1.0)
        
        # Dynamic stop loss based on volatility
        dynamic_stop = self.stop_loss_threshold * volatility
        return entry_price * (1 + dynamic_stop)
    
    def _calculate_target_price(self, trade: Dict) -> float:
        """Calculate dynamic target price."""
        entry_price = trade.get("entry_price", trade.get("current_price", 100))
        confidence = self.get_institution_confidence(trade.get("institution", "Unknown"))
        
        # Higher targets for higher confidence institutions
        target_multiplier = self.profit_taking_threshold * (0.5 + confidence)
        return entry_price * (1 + target_multiplier)
    
    @lru_cache(maxsize=64)
    def _estimate_holding_period_enhanced(self, institution: str) -> str:
        """Enhanced holding period estimation with more granular categories."""
        long_term = ["Berkshire Hathaway", "Pershing Square", "Elliott Management"]
        medium_long = ["Tiger Global", "Third Point", "Appaloosa Management"]
        medium_term = ["Bridgewater Associates", "Icahn Enterprises"]
        short_term = ["Renaissance Technologies", "Soros Fund Management"]
        
        if institution in long_term:
            return "12-36 months"
        elif institution in medium_long:
            return "6-18 months"
        elif institution in medium_term:
            return "3-12 months"
        elif institution in short_term:
            return "1-6 months"
        else:
            return "3-9 months"  # Default
    
    async def determine_entry_timing_optimized(self, filing_data: Dict) -> Dict:
        """
        Optimized entry timing with enhanced market condition analysis.
        
        Args:
            filing_data: Filing timing and price data
            
        Returns:
            Enhanced entry timing strategy
        """
        filing_date = filing_data["filing_date"]
        current_price = filing_data["current_price"]
        filing_price = filing_data["filing_price"]
        days_since_filing = filing_data.get("days_since_filing", 
                                          (datetime.now() - filing_date).days)
        
        # Enhanced price deviation analysis
        price_change = (current_price - filing_price) / filing_price
        volume_factor = filing_data.get("volume_factor", 1.0)  # Relative volume
        
        # Market condition adjustment
        market_condition = filing_data.get("market_condition", "neutral")
        market_multiplier = {
            "bullish": 1.2,
            "neutral": 1.0,
            "bearish": 0.8
        }.get(market_condition, 1.0)
        
        # Enhanced timing logic with market conditions
        urgency_score = self._calculate_urgency_score(
            days_since_filing, price_change, volume_factor, market_multiplier
        )
        
        if urgency_score >= 0.8:
            entry_strategy = "immediate"
            urgency = "very_high"
            max_chase_price = filing_price * (1.01 + 0.005 * market_multiplier)
        elif urgency_score >= 0.6:
            entry_strategy = "prompt"
            urgency = "high"
            max_chase_price = filing_price * (1.02 + 0.01 * market_multiplier)
        elif urgency_score >= 0.4:
            entry_strategy = "patient"
            urgency = "medium"
            max_chase_price = filing_price * (1.03 + 0.015 * market_multiplier)
        elif price_change > 0.20:  # More than 20% higher
            entry_strategy = "wait_for_pullback"
            urgency = "low"
            max_chase_price = filing_price * 1.08
        else:
            entry_strategy = "cautious"
            urgency = "low"
            max_chase_price = current_price * 1.015
        
        return {
            "entry_strategy": entry_strategy,
            "urgency": urgency,
            "max_chase_price": max_chase_price,
            "price_deviation": price_change,
            "days_since_filing": days_since_filing,
            "urgency_score": urgency_score,
            "market_condition": market_condition,
            "timing_score": self._calculate_timing_score_enhanced(
                days_since_filing, price_change, volume_factor, market_multiplier
            )
        }
    
    def _calculate_urgency_score(self, days: float, price_change: float, 
                               volume_factor: float, market_multiplier: float) -> float:
        """Calculate enhanced urgency score."""
        # Time decay factor
        time_score = max(0, 1.0 - (days / 10))  # Faster decay (10 days vs 14)
        
        # Price movement penalty
        price_score = max(0, 1.0 - abs(price_change) * 3)  # More sensitive to price moves
        
        # Volume factor (higher volume = more urgent)
        volume_score = min(volume_factor / 2, 1.0)
        
        # Market condition adjustment
        base_urgency = (time_score * 0.4 + price_score * 0.4 + volume_score * 0.2)
        
        return base_urgency * market_multiplier
    
    def _calculate_timing_score_enhanced(self, days: float, price_change: float,
                                       volume_factor: float, market_multiplier: float) -> float:
        """Calculate enhanced timing score with multiple factors."""
        time_score = max(0, 1.0 - (days / 12))
        price_score = max(0, 1.0 - abs(price_change) * 4)
        volume_score = min(volume_factor / 1.5, 1.0)
        
        return (time_score * 0.4 + price_score * 0.4 + volume_score * 0.2) * market_multiplier
    
    def parse_form_4_filing_optimized(self, form_4_data: Dict) -> Dict:
        """
        Optimized Form 4 insider transaction parsing.
        
        Args:
            form_4_data: Form 4 filing data
            
        Returns:
            Enhanced insider trading signal
        """
        confidence = self.score_insider_transaction_optimized(form_4_data)
        transaction_type = form_4_data["transaction_type"].lower()
        role = form_4_data["role"].lower()
        
        # Enhanced signal strength calculation
        if confidence > 0.85:
            signal_strength = "very_strong"
            position_multiplier = 1.3
        elif confidence > 0.75:
            signal_strength = "strong"
            position_multiplier = 1.2
        elif confidence > 0.65:
            signal_strength = "moderate"
            position_multiplier = 1.0
        elif confidence > 0.5:
            signal_strength = "weak"
            position_multiplier = 0.8
        else:
            signal_strength = "very_weak"
            position_multiplier = 0.6
        
        # Enhanced action determination
        if "purchase" in transaction_type and confidence > 0.65:  # Lowered threshold
            action = "buy"
        elif "sale" in transaction_type and confidence > 0.65:
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
            "reasoning": f"{role} {transaction_type} - confidence {confidence:.3f}",
            "transaction_size": form_4_data["shares"],
            "price_impact_score": self._calculate_price_impact_score(form_4_data)
        }
    
    def _calculate_price_impact_score(self, form_4_data: Dict) -> float:
        """Calculate potential price impact of insider transaction."""
        shares = form_4_data["shares"]
        price = form_4_data.get("price", 100)
        transaction_value = shares * price
        
        # Normalize transaction size impact
        if transaction_value > 50000000:  # $50M+
            return 0.9
        elif transaction_value > 10000000:  # $10M+
            return 0.7
        elif transaction_value > 1000000:   # $1M+
            return 0.5
        else:
            return 0.3
    
    def analyze_institutional_track_record_optimized(self, track_record: Dict) -> Dict:
        """
        Optimized institutional track record analysis with enhanced metrics.
        
        Args:
            track_record: Historical performance data
            
        Returns:
            Enhanced track record analysis
        """
        institution = track_record["institution"]
        annual_returns = track_record["last_5_years"]["annual_returns"]
        winning_positions = track_record["last_5_years"]["winning_positions"]
        total_positions = track_record["last_5_years"]["total_positions"]
        recent_performance = track_record["recent_performance"]["last_12_months"]
        vs_sp500 = track_record["recent_performance"]["vs_sp500"]
        
        # Enhanced metric calculations using numpy for speed
        annual_returns_np = np.array(annual_returns)
        avg_return = np.mean(annual_returns_np)
        return_volatility = np.std(annual_returns_np) if len(annual_returns_np) > 1 else 0.1
        win_rate = winning_positions / total_positions if total_positions > 0 else 0
        
        # Enhanced scoring system
        consistency_score = max(0, 1.0 - (return_volatility / 0.10))  # Tighter volatility tolerance
        performance_score = min((avg_return + 0.03) / 0.12, 1.0)  # Adjusted baseline
        recent_factor = min((recent_performance + 0.03) / 0.12, 1.0)
        alpha_factor = min((vs_sp500 + 0.015) / 0.06, 1.0)
        
        # Additional risk-adjusted metrics
        sharpe_ratio = avg_return / return_volatility if return_volatility > 0 else 0
        max_drawdown = self._estimate_max_drawdown(annual_returns_np)
        calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Enhanced confidence score with more sophisticated weighting
        confidence_score = (
            performance_score * 0.25 +      # Return performance
            consistency_score * 0.20 +     # Consistency
            (win_rate * 1.2) * 0.20 +      # Win rate (capped at 1.0)  
            recent_factor * 0.15 +         # Recent performance
            alpha_factor * 0.10 +          # Alpha generation
            min(sharpe_ratio / 2, 1.0) * 0.10  # Risk-adjusted performance
        )
        
        # Enhanced recommendation tiers
        if confidence_score > 0.85:
            recommended_follow_pct = 0.95
            tier = "elite"
        elif confidence_score > 0.75:
            recommended_follow_pct = 0.85
            tier = "premium"
        elif confidence_score > 0.65:
            recommended_follow_pct = 0.70
            tier = "quality"
        elif confidence_score > 0.50:
            recommended_follow_pct = 0.55
            tier = "average"
        else:
            recommended_follow_pct = 0.35
            tier = "below_average"
        
        return {
            "institution": institution,
            "confidence_score": confidence_score,
            "tier": tier,
            "win_rate": win_rate,
            "avg_annual_return": avg_return,
            "return_volatility": return_volatility,
            "consistency_score": consistency_score,
            "recent_performance": recent_performance,
            "alpha_vs_market": vs_sp500,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "recommended_follow_pct": recommended_follow_pct
        }
    
    def _estimate_max_drawdown(self, returns: np.ndarray) -> float:
        """Estimate maximum drawdown from annual returns."""
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def analyze_portfolio_overlap_optimized(self, our_portfolio: Dict, institutional_portfolio: Dict) -> Dict:
        """
        Optimized portfolio overlap analysis using pandas for efficiency.
        
        Args:
            our_portfolio: Our current holdings (ticker -> weight)
            institutional_portfolio: Institutional holdings (ticker -> weight)
            
        Returns:
            Enhanced portfolio overlap analysis
        """
        # Convert to pandas Series for vectorized operations
        our_series = pd.Series(our_portfolio)
        inst_series = pd.Series(institutional_portfolio)
        
        # Reindex to common universe
        all_tickers = our_series.index.union(inst_series.index)
        our_aligned = our_series.reindex(all_tickers, fill_value=0)
        inst_aligned = inst_series.reindex(all_tickers, fill_value=0)
        
        # Vectorized overlap calculations
        overlap_weights = np.minimum(our_aligned.values, inst_aligned.values)
        overlap_value = np.sum(overlap_weights)
        overlap_pct = overlap_value / our_aligned.sum() if our_aligned.sum() > 0 else 0
        
        # Enhanced analysis
        common_tickers = list(all_tickers[(our_aligned > 0) & (inst_aligned > 0)])
        missing_positions = list(all_tickers[(our_aligned == 0) & (inst_aligned > 0)])
        our_unique_positions = list(all_tickers[(our_aligned > 0) & (inst_aligned == 0)])
        
        # Vectorized recommendation generation
        missing_series = inst_aligned[inst_aligned > 0.03]  # Only significant positions
        recommendations = [
            {
                "ticker": ticker,
                "action": "consider_buy",
                "institutional_weight": weight,
                "suggested_weight": min(weight * 0.6, 0.035),  # Enhanced scaling
                "priority": "high" if weight > 0.10 else "medium" if weight > 0.05 else "low",
                "reasoning": f"Institution has {weight:.1%} position"
            }
            for ticker, weight in missing_series.items()
        ]
        
        # Enhanced correlation metrics
        correlation = np.corrcoef(our_aligned.values, inst_aligned.values)[0, 1] if len(our_aligned) > 1 else 0
        tracking_error = np.std(our_aligned.values - inst_aligned.values)
        
        return {
            "overlap_pct": overlap_pct,
            "overlap_value": overlap_value,
            "correlation": correlation,
            "tracking_error": tracking_error,
            "common_positions": common_tickers,
            "missing_positions": missing_positions[:10],  # Limit output
            "our_unique_positions": our_unique_positions[:10],
            "recommendations": recommendations[:5],  # Top 5 recommendations
            "portfolio_alignment": self._categorize_alignment(overlap_pct, correlation),
            "alignment_score": (overlap_pct + max(0, correlation)) / 2
        }
    
    def _categorize_alignment(self, overlap_pct: float, correlation: float) -> str:
        """Categorize portfolio alignment based on multiple metrics."""
        combined_score = (overlap_pct + max(0, correlation)) / 2
        
        if combined_score > 0.7:
            return "very_high"
        elif combined_score > 0.5:
            return "high"
        elif combined_score > 0.3:
            return "medium"
        elif combined_score > 0.15:
            return "low"
        else:
            return "very_low"
    
    def assess_mirror_risk_optimized(self, position: Dict) -> Dict:
        """
        Enhanced risk assessment for mirror trading positions.
        
        Args:
            position: Current mirror position details
            
        Returns:
            Enhanced risk assessment and recommended action
        """
        institutional_status = position["institutional_status"]
        current_price = position["current_price"]
        entry_price = position["entry_price"]
        days_held = position["days_held"]
        
        # Enhanced return calculation
        position_return = (current_price - entry_price) / entry_price
        
        # Dynamic risk thresholds based on holding period and institution
        institution = position.get("institution", "Unknown")
        confidence = self.get_institution_confidence(institution)
        
        # Adaptive stop loss based on confidence and holding period
        adaptive_stop_loss = self.stop_loss_threshold * (2 - confidence)  # Higher confidence = tighter stop
        
        # Enhanced decision logic
        if institutional_status == "exited":
            action = "exit"
            reason = "institutional_exit"
            urgency = "very_high"
        elif institutional_status == "reduced" and position_return < -0.03:  # More sensitive
            action = "exit"
            reason = "institutional_reduction_with_loss"
            urgency = "high"
        elif institutional_status == "increased" and position_return <= adaptive_stop_loss:
            action = "hold"  # Institution doubling down
            reason = "institutional_confidence_override"
            urgency = "low"
        elif position_return > self.profit_taking_threshold:
            action = "reduce"
            reason = "take_profits"
            urgency = "medium"
        elif position_return < adaptive_stop_loss:
            action = "exit"
            reason = "adaptive_stop_loss"
            urgency = "high"
        elif days_held > 730 and position_return > 0.20:  # 2 years with good gains
            action = "reduce" 
            reason = "long_term_profits"
            urgency = "low"
        else:
            action = "hold"
            reason = "position_healthy"
            urgency = "low"
        
        # Enhanced risk scoring
        risk_factors = {
            "return_risk": max(0, -position_return / 0.30),  # Normalize to 30% loss
            "time_risk": min(days_held / 365, 1.0),  # Time decay risk
            "institution_risk": 1 - confidence,  # Institution confidence risk
            "market_risk": position.get("beta", 1.0) - 0.5  # Market correlation risk
        }
        
        overall_risk_score = (
            risk_factors["return_risk"] * 0.4 +
            risk_factors["institution_risk"] * 0.3 +
            risk_factors["time_risk"] * 0.2 +
            max(0, risk_factors["market_risk"]) * 0.1
        )
        
        return {
            "action": action,
            "reason": reason,
            "urgency": urgency,
            "position_return": position_return,
            "institutional_status": institutional_status,
            "risk_level": "very_high" if overall_risk_score > 0.8 else 
                         "high" if overall_risk_score > 0.6 else
                         "medium" if overall_risk_score > 0.4 else "low",
            "overall_risk_score": overall_risk_score,
            "adaptive_stop_loss": adaptive_stop_loss,
            "risk_factors": risk_factors
        }
    
    def track_mirror_performance_optimized(self, mirror_trades: List[Dict]) -> Dict:
        """
        Optimized performance tracking using vectorized operations.
        
        Args:
            mirror_trades: List of mirror trades with performance data
            
        Returns:
            Enhanced performance tracking analysis
        """
        if not mirror_trades:
            return {"error": "No trades to analyze"}
        
        # Convert to numpy arrays for vectorized operations
        trade_data = []
        for trade in mirror_trades:
            our_return = (trade["current_price"] - trade["entry_price"]) / trade["entry_price"]
            inst_return = (trade["institutional_current"] - trade["institutional_entry"]) / trade["institutional_entry"]
            
            trade_data.append({
                "ticker": trade["ticker"],
                "institution": trade["institution"],
                "our_return": our_return,
                "institutional_return": inst_return,
                "tracking_difference": our_return - inst_return,
                "entry_timing_diff": (trade["entry_price"] - trade["institutional_entry"]) / trade["institutional_entry"]
            })
        
        # Vectorized calculations
        our_returns = np.array([t["our_return"] for t in trade_data])
        institutional_returns = np.array([t["institutional_return"] for t in trade_data])
        tracking_diffs = np.array([t["tracking_difference"] for t in trade_data])
        
        # Enhanced performance metrics
        our_avg_return = np.mean(our_returns)
        institutional_avg_return = np.mean(institutional_returns)
        our_volatility = np.std(our_returns)
        inst_volatility = np.std(institutional_returns)
        
        # Enhanced tracking metrics
        tracking_error = np.std(tracking_diffs)
        tracking_efficiency = max(0, 1.0 - (tracking_error / 0.05))  # Normalize to 5%
        
        # Risk-adjusted metrics
        our_sharpe = our_avg_return / our_volatility if our_volatility > 0 else 0
        inst_sharpe = institutional_avg_return / inst_volatility if inst_volatility > 0 else 0
        
        # Information ratio
        active_return = our_avg_return - institutional_avg_return
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        # Correlation analysis
        correlation = np.corrcoef(our_returns, institutional_returns)[0, 1] if len(our_returns) > 1 else 0
        
        return {
            "our_avg_return": our_avg_return,
            "institutional_avg_return": institutional_avg_return,
            "our_volatility": our_volatility,
            "institutional_volatility": inst_volatility,
            "our_sharpe_ratio": our_sharpe,
            "institutional_sharpe_ratio": inst_sharpe,
            "tracking_efficiency": tracking_efficiency,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation": correlation,
            "total_trades": len(mirror_trades),
            "individual_trades": trade_data,
            "outperformance": active_return,
            "performance_attribution": self._calculate_performance_attribution(trade_data)
        }
    
    def _calculate_performance_attribution(self, trade_data: List[Dict]) -> Dict:
        """Calculate performance attribution by institution and sector."""
        by_institution = {}
        
        for trade in trade_data:
            institution = trade["institution"]
            if institution not in by_institution:
                by_institution[institution] = []
            by_institution[institution].append(trade["our_return"])
        
        attribution = {}
        for institution, returns in by_institution.items():
            attribution[institution] = {
                "avg_return": np.mean(returns),
                "trade_count": len(returns),
                "contribution": np.mean(returns) * len(returns) / len(trade_data)
            }
        
        return attribution
    
    async def monitor_filing_alerts_optimized(self, recent_filings: List[Dict]) -> List[Dict]:
        """
        Optimized filing monitoring with batch processing and prioritization.
        
        Args:
            recent_filings: List of recent institutional filings
            
        Returns:
            Prioritized list of trading alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Batch process filings for efficiency
        for filing in recent_filings:
            institution = filing["institution"]
            confidence = self.get_institution_confidence(institution)
            filing_age_hours = (current_time - filing["filed_date"]).total_seconds() / 3600
            
            # Enhanced urgency calculation
            urgency_multiplier = max(0.1, 1.0 - (filing_age_hours / 48))  # Decay over 48 hours
            adjusted_confidence = confidence * urgency_multiplier
            
            # Process new positions with enhanced prioritization
            for ticker in filing.get("new_positions", []):
                priority_score = self._calculate_alert_priority_score(
                    adjusted_confidence, filing_age_hours, "new_position"
                )
                
                alerts.append({
                    "ticker": ticker,
                    "action": "buy",
                    "institution": institution,
                    "confidence": adjusted_confidence,
                    "priority": self._score_to_priority_level(priority_score),
                    "priority_score": priority_score,
                    "filing_type": filing["filing_type"],
                    "alert_type": "new_position",
                    "filing_age_hours": filing_age_hours,
                    "urgency_multiplier": urgency_multiplier
                })
            
            # Process position changes with enhanced logic
            for ticker, change in filing.get("position_changes", {}).items():
                if change == "increased":
                    action = "buy"
                    alert_type = "position_increase"
                    base_priority = 0.7
                elif change == "exited":
                    action = "sell"
                    alert_type = "position_exit"
                    base_priority = 0.9  # High priority for exits
                elif change == "reduced":
                    action = "reduce"
                    alert_type = "position_reduction"
                    base_priority = 0.6
                else:
                    action = "review"
                    alert_type = "position_change"
                    base_priority = 0.4
                
                priority_score = self._calculate_alert_priority_score(
                    adjusted_confidence, filing_age_hours, alert_type, base_priority
                )
                
                alerts.append({
                    "ticker": ticker,
                    "action": action,
                    "institution": institution,
                    "confidence": adjusted_confidence,
                    "priority": self._score_to_priority_level(priority_score),
                    "priority_score": priority_score,
                    "filing_type": filing["filing_type"],
                    "alert_type": alert_type,
                    "position_change": change,
                    "filing_age_hours": filing_age_hours
                })
        
        # Enhanced sorting: first by priority score, then by confidence
        alerts.sort(key=lambda x: (x["priority_score"], x["confidence"]), reverse=True)
        
        return alerts[:20]  # Return top 20 alerts to avoid overwhelming users
    
    def _calculate_alert_priority_score(self, confidence: float, age_hours: float, 
                                      alert_type: str, base_priority: float = 0.5) -> float:
        """Calculate comprehensive alert priority score."""
        # Time decay factor
        time_factor = max(0.1, 1.0 - (age_hours / 72))  # 72-hour decay
        
        # Alert type weights
        type_weights = {
            "new_position": 0.9,
            "position_exit": 1.0,
            "position_increase": 0.8,
            "position_reduction": 0.6,
            "position_change": 0.4
        }
        
        type_weight = type_weights.get(alert_type, 0.5)
        
        return (confidence * 0.5 + time_factor * 0.3 + type_weight * 0.2) * base_priority
    
    def _score_to_priority_level(self, score: float) -> str:
        """Convert numeric priority score to categorical level."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.65:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.35:
            return "low"
        else:
            return "very_low"
    
    def analyze_return_correlation_optimized(self, return_data: Dict) -> Dict:
        """
        Optimized correlation analysis using numpy vectorized operations.
        
        Args:
            return_data: Return series data
            
        Returns:
            Enhanced correlation analysis
        """
        our_returns = np.array(return_data["our_returns"])
        institutional_returns = np.array(return_data["institutional_returns"])
        market_returns = np.array(return_data["market_returns"])
        
        # Vectorized correlation matrix
        all_returns = np.array([our_returns, institutional_returns, market_returns])
        correlation_matrix = np.corrcoef(all_returns)
        
        our_vs_institutional = correlation_matrix[0, 1]
        our_vs_market = correlation_matrix[0, 2]
        institutional_vs_market = correlation_matrix[1, 2]
        
        # Enhanced tracking metrics
        tracking_differences = our_returns - institutional_returns
        tracking_error = np.std(tracking_differences)
        mean_tracking_diff = np.mean(tracking_differences)
        
        # Enhanced mirror effectiveness with multiple factors
        correlation_factor = max(0, our_vs_institutional)
        tracking_factor = max(0, 1.0 - (tracking_error / 0.03))  # Normalize to 3%
        consistency_factor = max(0, 1.0 - (abs(mean_tracking_diff) / 0.02))  # Penalize bias
        
        mirror_effectiveness = (
            correlation_factor * 0.5 +
            tracking_factor * 0.3 +
            consistency_factor * 0.2
        )
        
        # Risk-adjusted metrics
        our_volatility = np.std(our_returns)
        inst_volatility = np.std(institutional_returns)
        our_sharpe = np.mean(our_returns) / our_volatility if our_volatility > 0 else 0
        inst_sharpe = np.mean(institutional_returns) / inst_volatility if inst_volatility > 0 else 0
        
        # Information ratio and other advanced metrics
        active_return = np.mean(tracking_differences)
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return {
            "our_vs_institutional": our_vs_institutional,
            "our_vs_market": our_vs_market,
            "institutional_vs_market": institutional_vs_market,
            "correlation_matrix": correlation_matrix.tolist(),
            "tracking_error": tracking_error,
            "mean_tracking_difference": mean_tracking_diff,
            "mirror_effectiveness": mirror_effectiveness,
            "our_sharpe_ratio": our_sharpe,
            "institutional_sharpe_ratio": inst_sharpe,
            "information_ratio": information_ratio,
            "avg_our_return": np.mean(our_returns),
            "avg_institutional_return": np.mean(institutional_returns),
            "our_volatility": our_volatility,
            "institutional_volatility": inst_volatility,
            "return_consistency": consistency_factor,
            "downside_deviation": self._calculate_downside_deviation(our_returns),
            "upside_capture": self._calculate_upside_capture(our_returns, institutional_returns),
            "downside_capture": self._calculate_downside_capture(our_returns, institutional_returns)
        }
    
    def _calculate_downside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target]
        return np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    
    def _calculate_upside_capture(self, our_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate upside capture ratio."""
        up_periods = benchmark_returns > 0
        if not np.any(up_periods):
            return 0.0
        
        our_up = np.mean(our_returns[up_periods])
        bench_up = np.mean(benchmark_returns[up_periods])
        
        return our_up / bench_up if bench_up != 0 else 0.0
    
    def _calculate_downside_capture(self, our_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate downside capture ratio."""
        down_periods = benchmark_returns < 0
        if not np.any(down_periods):
            return 0.0
        
        our_down = np.mean(our_returns[down_periods])
        bench_down = np.mean(benchmark_returns[down_periods])
        
        return our_down / bench_down if bench_down != 0 else 0.0


class MirrorTraderOptimized:
    """Optimized mirror trading strategy wrapper for deployment"""
    
    def __init__(self, gpu_enabled=False):
        self.gpu_enabled = gpu_enabled
        self.is_running = False
        self.last_trade_time = None
        self.engine = OptimizedMirrorTradingEngine()
        
    async def start(self):
        """Start the strategy"""
        self.is_running = True
        return {"status": "started", "strategy": "mirror_trader_optimized"}
        
    async def stop(self):
        """Stop the strategy"""
        self.is_running = False
        return {"status": "stopped", "strategy": "mirror_trader_optimized"}
        
    def get_performance(self):
        """Get performance metrics"""
        return {
            "total_trades": 0,
            "profit_loss": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "mirror_accuracy": 0,
            "status": "initialized"
        }