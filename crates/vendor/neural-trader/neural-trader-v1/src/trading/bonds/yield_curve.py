"""Yield curve shape analysis and trading signals"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YieldCurveAnalyzer:
    """Analyze yield curve shapes and generate trading signals"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.curve_regimes = {
            "bear_flattener": "Short end rising faster than long end",
            "bull_flattener": "Long end falling faster than short end", 
            "bear_steepener": "Long end rising faster than short end",
            "bull_steepener": "Short end falling faster than long end"
        }
        
    def analyze_curve_shape(self, yields: Dict[str, float]) -> Dict[str, any]:
        """
        Analyze yield curve shape and implications
        
        Args:
            yields: Dictionary of maturity/yield pairs
            
        Returns:
            Dictionary with curve analysis
        """
        # Calculate key spreads
        spreads = self._calculate_spreads(yields)
        
        # Determine curve shape
        curve_type = self._determine_curve_type(spreads)
        
        # Calculate steepness
        steepness = self._calculate_steepness(yields, spreads)
        
        # Estimate recession probability
        recession_prob = self._estimate_recession_probability(spreads)
        
        # Trading bias based on curve
        trading_bias = self._get_trading_bias(curve_type, spreads)
        
        result = {
            "type": curve_type,
            "2s10s_spread": spreads.get("2s10s", 0),
            "2s30s_spread": spreads.get("2s30s", 0),
            "5s30s_spread": spreads.get("5s30s", 0),
            "recession_probability": recession_prob,
            "steepness": steepness,
            "trading_bias": trading_bias
        }
        
        # Add inversion details if applicable
        if curve_type == "inverted":
            result["inversion_duration"] = self._calculate_inversion_duration()
            result["inversion_depth"] = abs(spreads.get("2s10s", 0))
            
        return result
    
    def analyze_curve_dynamics(self, prev_yields: Dict[str, float], 
                             curr_yields: Dict[str, float]) -> Dict[str, any]:
        """
        Analyze changes in yield curve over time
        
        Args:
            prev_yields: Previous yield curve
            curr_yields: Current yield curve
            
        Returns:
            Dictionary with curve dynamics
        """
        # Calculate changes
        changes = {}
        for maturity in curr_yields:
            if maturity in prev_yields:
                changes[maturity] = curr_yields[maturity] - prev_yields[maturity]
                
        # Identify movement type
        movement = self._identify_curve_movement(prev_yields, curr_yields, changes)
        
        # Calculate twist
        twist = self._calculate_curve_twist(changes)
        
        # Front vs back end changes
        front_end_change = np.mean([changes.get("3M", 0), changes.get("2Y", 0)])
        long_end_change = np.mean([changes.get("10Y", 0), changes.get("30Y", 0)])
        
        return {
            "movement": movement,
            "changes": changes,
            "front_end_change": front_end_change,
            "long_end_change": long_end_change,
            "twist": twist,
            "velocity": self._calculate_movement_velocity(changes)
        }
    
    def calculate_butterfly_spread(self, yields: Dict[str, float]) -> Dict[str, any]:
        """
        Calculate butterfly spread for relative value
        
        Args:
            yields: Dictionary with at least 2Y, 5Y, 10Y yields
            
        Returns:
            Dictionary with butterfly analysis
        """
        if not all(mat in yields for mat in ["2Y", "5Y", "10Y"]):
            return {"spread": 0, "signal": "insufficient_data"}
            
        # Butterfly = 2*5Y - 2Y - 10Y
        butterfly = 2 * yields["5Y"] - yields["2Y"] - yields["10Y"]
        
        # Determine signal
        if butterfly > 0.2:
            signal = "cheap"  # 5Y cheap relative to wings
        elif butterfly < -0.2:
            signal = "rich"   # 5Y rich relative to wings
        else:
            signal = "neutral"
            
        return {
            "spread": butterfly,
            "signal": signal,
            "5y_richness": -butterfly,  # Negative = rich
            "trade_recommendation": self._get_butterfly_trade(signal)
        }
    
    def generate_trading_signals(self, market_data: Dict) -> List[Dict]:
        """
        Generate trading signals based on curve analysis
        
        Args:
            market_data: Dictionary with current and previous yields, Fed policy, etc.
            
        Returns:
            List of trading signals
        """
        signals = []
        
        current_yields = market_data.get("current_yields", {})
        prev_yields = market_data.get("prev_yields", {})
        fed_policy = market_data.get("fed_policy", "neutral")
        
        # Analyze curve dynamics
        if prev_yields:
            dynamics = self.analyze_curve_dynamics(prev_yields, current_yields)
            
            # Steepener/Flattener trades
            if dynamics["movement"] == "steepening":
                signals.append({
                    "trade": "steepener",
                    "position": "long_10Y_short_2Y",
                    "rationale": "Curve steepening in progress",
                    "confidence": 0.7,
                    "entry_spread": current_yields.get("10Y", 0) - current_yields.get("2Y", 0),
                    "target_spread": dynamics["velocity"] * 10  # Project movement
                })
                
            elif dynamics["movement"] == "flattening":
                signals.append({
                    "trade": "flattener", 
                    "position": "short_10Y_long_2Y",
                    "rationale": "Curve flattening in progress",
                    "confidence": 0.7,
                    "entry_spread": current_yields.get("10Y", 0) - current_yields.get("2Y", 0),
                    "stop_spread": dynamics["velocity"] * -5
                })
                
        # Duration trades based on Fed policy
        if fed_policy == "easing":
            signals.append({
                "trade": "long_duration",
                "position": "long_TLT",
                "rationale": "Fed easing supports duration",
                "confidence": 0.8,
                "entry_yield": current_yields.get("30Y", 0),
                "target_yield": current_yields.get("30Y", 0) - 0.5
            })
            
        # Butterfly trades
        butterfly = self.calculate_butterfly_spread(current_yields)
        if butterfly["signal"] != "neutral":
            signals.append({
                "trade": "butterfly",
                "position": butterfly["trade_recommendation"],
                "rationale": f"5Y {butterfly['signal']} on butterfly",
                "confidence": 0.6,
                "entry_spread": butterfly["spread"]
            })
            
        return signals
    
    def get_duration_recommendations(self, market_conditions: Dict) -> Dict:
        """
        Get duration recommendations based on market conditions
        
        Args:
            market_conditions: Dictionary with curve shape, Fed policy, etc.
            
        Returns:
            Duration recommendations
        """
        curve_shape = market_conditions.get("curve_shape", "normal")
        curve_trend = market_conditions.get("curve_trend", "stable")
        fed_policy = market_conditions.get("fed_policy", "neutral")
        inflation_trend = market_conditions.get("inflation_trend", "stable")
        
        # Base recommendation on conditions
        if fed_policy == "tightening" and inflation_trend == "rising":
            return {
                "overall_duration": "underweight",
                "preferred_maturity": "short",
                "recommended_instruments": ["2Y", "SHY", "BSV", "BIL"],
                "avoid": ["TLT", "EDV"],
                "rationale": "Rising rate environment favors short duration"
            }
            
        elif fed_policy == "easing" or curve_shape == "inverted":
            return {
                "overall_duration": "overweight",
                "preferred_maturity": "long",
                "recommended_instruments": ["TLT", "IEF", "TLH"],
                "avoid": ["SHY", "BIL"],
                "rationale": "Rate cuts expected, long duration attractive"
            }
            
        else:
            return {
                "overall_duration": "neutral",
                "preferred_maturity": "intermediate",
                "recommended_instruments": ["IEF", "AGG", "BND"],
                "avoid": [],
                "rationale": "Balanced approach in uncertain environment"
            }
    
    def detect_regime(self, historical_data: pd.DataFrame) -> Dict:
        """
        Detect current yield curve regime
        
        Args:
            historical_data: DataFrame with historical yield data
            
        Returns:
            Dictionary with regime information
        """
        if len(historical_data) < 5:
            return {
                "current_regime": "insufficient_data",
                "regime_duration": 0,
                "regime_strength": 0
            }
            
        # Calculate recent changes
        recent_data = historical_data.tail(10)
        
        # Front and back end changes
        if '2Y' in recent_data.columns and '10Y' in recent_data.columns:
            front_change = recent_data['2Y'].iloc[-1] - recent_data['2Y'].iloc[0]
            back_change = recent_data['10Y'].iloc[-1] - recent_data['10Y'].iloc[0]
            
            # Determine regime
            if front_change > 0 and back_change > 0:
                if front_change > back_change:
                    regime = "bear_flattener"
                else:
                    regime = "bear_steepener"
            elif front_change < 0 and back_change < 0:
                if abs(front_change) > abs(back_change):
                    regime = "bull_steepener"
                else:
                    regime = "bull_flattener"
            elif front_change > 0 and back_change < 0:
                # Front rising, back falling = bear flattener
                regime = "bear_flattener"
            elif front_change < 0 and back_change > 0:
                # Front falling, back rising = bull steepener
                regime = "bull_steepener"
            else:
                # Fallback to most likely regime based on direction
                if abs(front_change) > abs(back_change):
                    regime = "bear_flattener" if front_change > 0 else "bull_steepener"
                else:
                    regime = "bear_steepener" if back_change > 0 else "bull_flattener"
                
            # Calculate regime strength
            total_movement = abs(front_change) + abs(back_change)
            strength = min(total_movement / 0.5, 1.0)  # Normalize to 0-1
            
            # Estimate duration (simplified)
            duration = self._estimate_regime_duration(historical_data, regime)
            
            return {
                "current_regime": regime,
                "regime_duration": duration,
                "regime_strength": strength,
                "description": self.curve_regimes.get(regime, "Unknown regime")
            }
            
        return {
            "current_regime": "undefined",
            "regime_duration": 0,
            "regime_strength": 0
        }
    
    def analyze_relative_value(self, current_yields: Dict[str, float]) -> Dict:
        """
        Analyze relative value opportunities across the curve
        
        Args:
            current_yields: Current yield curve
            
        Returns:
            Dictionary with relative value analysis
        """
        # Fit a smooth curve to identify outliers
        maturities = []
        yields = []
        
        maturity_map = {
            "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
            "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30
        }
        
        for mat, years in maturity_map.items():
            if mat in current_yields:
                maturities.append(years)
                yields.append(current_yields[mat])
                
        if len(maturities) < 4:
            return {
                "rich_points": [],
                "cheap_points": [],
                "trades": []
            }
            
        # Fit polynomial curve
        z = np.polyfit(maturities, yields, 3)
        p = np.poly1d(z)
        
        # Find rich/cheap points
        rich_points = []
        cheap_points = []
        
        for i, (mat_years, actual_yield) in enumerate(zip(maturities, yields)):
            fitted_yield = p(mat_years)
            diff = actual_yield - fitted_yield
            
            maturity_label = next(k for k, v in maturity_map.items() if v == mat_years)
            
            if diff > 0.1:  # 10bp cheap
                cheap_points.append({
                    "maturity": maturity_label,
                    "richness": diff,
                    "actual_yield": actual_yield,
                    "fair_yield": fitted_yield
                })
            elif diff < -0.1:  # 10bp rich
                rich_points.append({
                    "maturity": maturity_label,
                    "richness": diff,
                    "actual_yield": actual_yield,
                    "fair_yield": fitted_yield
                })
                
        # Generate trades
        trades = self._generate_rv_trades(rich_points, cheap_points)
        
        return {
            "rich_points": rich_points,
            "cheap_points": cheap_points,
            "trades": trades,
            "curve_fit_quality": self._assess_curve_fit(maturities, yields, p)
        }
    
    def _calculate_spreads(self, yields: Dict[str, float]) -> Dict[str, float]:
        """Calculate various yield spreads"""
        spreads = {}
        
        if "2Y" in yields and "10Y" in yields:
            spreads["2s10s"] = yields["10Y"] - yields["2Y"]
            
        if "2Y" in yields and "30Y" in yields:
            spreads["2s30s"] = yields["30Y"] - yields["2Y"]
            
        if "5Y" in yields and "30Y" in yields:
            spreads["5s30s"] = yields["30Y"] - yields["5Y"]
            
        if "3M" in yields and "10Y" in yields:
            spreads["3m10y"] = yields["10Y"] - yields["3M"]
            
        return spreads
    
    def _determine_curve_type(self, spreads: Dict[str, float]) -> str:
        """Determine curve shape type"""
        s2s10s = spreads.get("2s10s", 0)
        
        if s2s10s < -0.1:
            return "inverted"
        elif -0.1 <= s2s10s <= 0.3:
            return "flat"
        else:
            return "normal"
    
    def _calculate_steepness(self, yields: Dict[str, float], 
                           spreads: Dict[str, float]) -> str:
        """Calculate curve steepness"""
        s2s10s = abs(spreads.get("2s10s", 0))
        
        if s2s10s < 0.25:
            return "flat"
        elif s2s10s < 0.75:
            return "moderate"
        else:
            return "steep"
    
    def _estimate_recession_probability(self, spreads: Dict[str, float]) -> float:
        """Estimate recession probability based on curve shape"""
        s2s10s = spreads.get("2s10s", 0)
        s3m10y = spreads.get("3m10y", 0)
        
        # Simple model based on spread
        if s2s10s < -0.5:
            base_prob = 0.8
        elif s2s10s < -0.2:
            base_prob = 0.6
        elif s2s10s < 0:
            base_prob = 0.4
        elif s2s10s < 0.3:
            base_prob = 0.35  # Flat curve moderate recession risk
        elif s2s10s < 1.0:
            base_prob = 0.15  # Healthy normal curve
        else:
            base_prob = 0.1   # Very steep curve, low recession risk
            
        # Adjust for 3m10y spread
        if s3m10y < -0.5:
            base_prob = min(base_prob + 0.2, 0.9)
            
        return base_prob
    
    def _get_trading_bias(self, curve_type: str, spreads: Dict[str, float]) -> str:
        """Determine trading bias based on curve"""
        if curve_type == "inverted":
            return "long_duration"  # Expect rate cuts
        elif curve_type == "normal" and spreads.get("2s10s", 0) > 1.5:
            return "short_duration"  # Steep curve, rates may rise
        else:
            return "neutral"
    
    def _calculate_inversion_duration(self) -> int:
        """Calculate how long curve has been inverted (placeholder)"""
        # In production, would track historical inversions
        return 30  # days
    
    def _identify_curve_movement(self, prev_yields: Dict[str, float],
                               curr_yields: Dict[str, float],
                               changes: Dict[str, float]) -> str:
        """Identify type of curve movement"""
        prev_spread = prev_yields.get("10Y", 0) - prev_yields.get("2Y", 0)
        curr_spread = curr_yields.get("10Y", 0) - curr_yields.get("2Y", 0)
        
        spread_change = curr_spread - prev_spread
        
        if abs(spread_change) < 0.05:
            return "parallel"
        elif spread_change > 0:
            return "steepening"
        else:
            return "flattening"
    
    def _calculate_curve_twist(self, changes: Dict[str, float]) -> Optional[float]:
        """Calculate curve twist metric"""
        # Try with 2Y, 5Y, 10Y first
        if all(mat in changes for mat in ["2Y", "5Y", "10Y"]):
            # Twist = change in curvature
            twist = changes["5Y"] - (changes["2Y"] + changes["10Y"]) / 2
            return twist
        
        # Fallback: use available intermediate points
        elif all(mat in changes for mat in ["3M", "2Y", "10Y"]):
            # Use 2Y as the belly point
            twist = changes["2Y"] - (changes["3M"] + changes["10Y"]) / 2
            return twist
        
        elif all(mat in changes for mat in ["2Y", "10Y", "30Y"]):
            # Use 10Y as the belly point
            twist = changes["10Y"] - (changes["2Y"] + changes["30Y"]) / 2
            return twist
            
        return None
    
    def _calculate_movement_velocity(self, changes: Dict[str, float]) -> float:
        """Calculate speed of curve movement"""
        if not changes:
            return 0.0
            
        # RMS of changes
        return np.sqrt(np.mean([c**2 for c in changes.values()]))
    
    def _get_butterfly_trade(self, signal: str) -> str:
        """Get butterfly trade recommendation"""
        if signal == "cheap":
            return "long_5Y_short_2Y_10Y"
        elif signal == "rich":
            return "short_5Y_long_2Y_10Y"
        else:
            return "no_trade"
    
    def _estimate_regime_duration(self, data: pd.DataFrame, regime: str) -> int:
        """Estimate how long current regime has persisted"""
        # Simplified - in production would track regime changes
        return 15  # days
    
    def _generate_rv_trades(self, rich_points: List[Dict], 
                          cheap_points: List[Dict]) -> List[Dict]:
        """Generate relative value trades"""
        trades = []
        
        # Pair rich and cheap points
        for cheap in cheap_points:
            for rich in rich_points:
                # Look for good pairs (similar duration)
                cheap_mat = self._maturity_to_years(cheap["maturity"])
                rich_mat = self._maturity_to_years(rich["maturity"])
                
                if 0.5 <= cheap_mat / rich_mat <= 2.0:  # Reasonable pair
                    expected_profit = (cheap["richness"] - rich["richness"]) * 100
                    
                    trades.append({
                        "long": cheap["maturity"],
                        "short": rich["maturity"],
                        "expected_profit_bps": expected_profit,
                        "confidence": min(expected_profit / 20, 1.0)
                    })
                    
        # Sort by expected profit
        trades.sort(key=lambda x: x["expected_profit_bps"], reverse=True)
        
        return trades[:3]  # Top 3 trades
    
    def _maturity_to_years(self, maturity: str) -> float:
        """Convert maturity label to years"""
        maturity_map = {
            "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
            "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30
        }
        return maturity_map.get(maturity, 5)
    
    def _assess_curve_fit(self, maturities: List[float], 
                         yields: List[float], p) -> float:
        """Assess quality of curve fit"""
        fitted = [p(m) for m in maturities]
        residuals = [y - f for y, f in zip(yields, fitted)]
        rmse = np.sqrt(np.mean([r**2 for r in residuals]))
        
        # Quality score (inverse of RMSE, capped at 1)
        return min(1.0, 1.0 / (1 + rmse * 10))