"""Credit spread trading strategies."""

from typing import Dict, List, Tuple


class CreditSpreadTrader:
    """Trades corporate bond spreads vs treasuries."""
    
    def __init__(self):
        """Initialize the credit spread trader."""
        self.spread_regimes = {
            "tight": {"action": "short_credit", "confidence": 0.7},
            "normal": {"action": "neutral", "confidence": 0.5},
            "wide": {"action": "long_credit", "confidence": 0.8},
            "distressed": {"action": "selective_long", "confidence": 0.6},
        }
        
    def analyze_credit_opportunity(self, spread_data: Dict) -> Dict:
        """Analyze credit spread trading opportunities.
        
        Args:
            spread_data: Credit spread information
            
        Returns:
            Trading signal
        """
        ig_spread = spread_data.get("ig_spread", 100)  # Investment grade
        hy_spread = spread_data.get("hy_spread", 400)  # High yield
        ig_avg = spread_data.get("historical_ig_avg", 100)
        hy_avg = spread_data.get("historical_hy_avg", 400)
        vix = spread_data.get("vix", 20)
        
        # Analyze spread levels
        ig_percentile = self._calculate_spread_percentile(ig_spread, ig_avg)
        hy_percentile = self._calculate_spread_percentile(hy_spread, hy_avg)
        
        # Determine regime
        regime = self._determine_credit_regime(ig_spread, ig_avg, vix)
        
        # Generate signal based on regime
        if regime == "wide":
            return self._create_long_credit_signal(spread_data)
        elif regime == "tight":
            return self._create_short_credit_signal(spread_data)
        else:
            return {
                "trade": "neutral",
                "rationale": "Credit spreads at fair value",
                "ig_percentile": ig_percentile,
                "hy_percentile": hy_percentile,
            }
    
    def _calculate_spread_percentile(self, current: float, average: float) -> float:
        """Calculate where current spread is vs historical average.
        
        Args:
            current: Current spread
            average: Historical average
            
        Returns:
            Percentile (0-100)
        """
        # Simplified calculation
        ratio = current / average if average > 0 else 1
        
        if ratio < 0.7:
            return 10  # Very tight
        elif ratio < 0.9:
            return 25  # Tight
        elif ratio < 1.1:
            return 50  # Normal
        elif ratio < 1.3:
            return 75  # Wide
        else:
            return 90  # Very wide
    
    def _determine_credit_regime(
        self,
        current_spread: float,
        avg_spread: float,
        vix: float
    ) -> str:
        """Determine credit market regime.
        
        Args:
            current_spread: Current IG spread
            avg_spread: Historical average IG spread
            vix: Market volatility
            
        Returns:
            Regime type
        """
        spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1
        
        if spread_ratio < 0.8 and vix < 20:
            return "tight"
        elif spread_ratio > 1.3 or vix > 30:
            return "wide"
        elif spread_ratio > 2.0 or vix > 40:
            return "distressed"
        else:
            return "normal"
    
    def _create_long_credit_signal(self, spread_data: Dict) -> Dict:
        """Create long credit position signal.
        
        Args:
            spread_data: Spread data
            
        Returns:
            Long credit signal
        """
        ig_spread = spread_data.get("ig_spread", 100)
        vix = spread_data.get("vix", 20)
        
        # Choose instrument based on risk level
        if vix > 35:
            instrument = "LQD"  # Investment grade only in high vol
            hedge = "IEF"       # Treasury hedge
        else:
            instrument = "LQD"  # Primary position
            hedge = "TLT"       # Duration-matched hedge
        
        # Size ratio based on spread levels
        if ig_spread > 200:
            size_ratio = 3.0  # 3:1 credit to treasury
        else:
            size_ratio = 2.0  # 2:1 credit to treasury
        
        return {
            "trade": "long_credit",
            "instrument": instrument,
            "hedge": hedge,
            "size_ratio": size_ratio,
            "rationale": "Credit spreads wide vs historical average",
            "entry_spread": ig_spread,
            "target_spread": spread_data.get("historical_ig_avg", 100),
            "stop_spread": ig_spread * 1.2,  # 20% wider
            "confidence": 0.7,
        }
    
    def _create_short_credit_signal(self, spread_data: Dict) -> Dict:
        """Create short credit position signal.
        
        Args:
            spread_data: Spread data
            
        Returns:
            Short credit signal
        """
        ig_spread = spread_data.get("ig_spread", 100)
        
        return {
            "trade": "short_credit",
            "instrument": "LQD",  # Short investment grade
            "hedge": "IEF",       # Long treasuries
            "size_ratio": 1.0,    # Equal weight
            "rationale": "Credit spreads tight, limited upside",
            "entry_spread": ig_spread,
            "target_spread": spread_data.get("historical_ig_avg", 100),
            "stop_spread": ig_spread * 0.8,  # 20% tighter
            "confidence": 0.6,
        }
    
    def analyze_credit_sectors(self, sector_spreads: Dict[str, float]) -> List[Dict]:
        """Analyze credit opportunities by sector.
        
        Args:
            sector_spreads: Sector spread data
            
        Returns:
            List of sector opportunities
        """
        opportunities = []
        
        # Historical average spreads by sector
        sector_averages = {
            "financials": 80,
            "energy": 120,
            "technology": 70,
            "utilities": 60,
            "consumer": 90,
            "industrials": 85,
        }
        
        for sector, current_spread in sector_spreads.items():
            if sector in sector_averages:
                avg_spread = sector_averages[sector]
                ratio = current_spread / avg_spread
                
                if ratio > 1.5:
                    opportunities.append({
                        "sector": sector,
                        "action": "overweight",
                        "spread": current_spread,
                        "vs_average": f"+{int((ratio - 1) * 100)}%",
                        "confidence": min(0.8, 0.5 + (ratio - 1) * 0.3),
                    })
                elif ratio < 0.7:
                    opportunities.append({
                        "sector": sector,
                        "action": "underweight",
                        "spread": current_spread,
                        "vs_average": f"{int((ratio - 1) * 100)}%",
                        "confidence": 0.6,
                    })
        
        # Sort by opportunity
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        
        return opportunities
    
    def calculate_credit_hedge_ratio(
        self,
        credit_duration: float,
        treasury_duration: float,
        credit_spread_duration: float
    ) -> float:
        """Calculate hedge ratio for credit positions.
        
        Args:
            credit_duration: Duration of credit position
            treasury_duration: Duration of treasury hedge
            credit_spread_duration: Spread duration of credit
            
        Returns:
            Hedge ratio
        """
        # Total credit risk = interest rate risk + spread risk
        # Only hedge interest rate risk with treasuries
        interest_rate_duration = credit_duration - credit_spread_duration
        
        if treasury_duration == 0:
            return 0
        
        hedge_ratio = interest_rate_duration / treasury_duration
        
        return round(hedge_ratio, 2)
    
    def monitor_spread_trade(
        self,
        entry_spread: float,
        current_spread: float,
        position_type: str
    ) -> Dict[str, any]:
        """Monitor active credit spread trade.
        
        Args:
            entry_spread: Spread at entry
            current_spread: Current spread
            position_type: "long_credit" or "short_credit"
            
        Returns:
            Trade management recommendation
        """
        spread_change = current_spread - entry_spread
        spread_change_pct = (spread_change / entry_spread) * 100
        
        if position_type == "long_credit":
            # Long credit benefits from tightening spreads
            if spread_change < -20:  # Spreads tightened 20bp
                return {
                    "action": "take_profit",
                    "reason": "Target reached",
                    "pnl_estimate": abs(spread_change) * 0.01,  # Rough estimate
                }
            elif spread_change > 30:  # Spreads widened 30bp
                return {
                    "action": "stop_loss",
                    "reason": "Spreads widening",
                    "pnl_estimate": -abs(spread_change) * 0.01,
                }
        else:  # short_credit
            # Short credit benefits from widening spreads
            if spread_change > 20:
                return {
                    "action": "take_profit",
                    "reason": "Spreads widened as expected",
                    "pnl_estimate": spread_change * 0.01,
                }
            elif spread_change < -20:
                return {
                    "action": "stop_loss",
                    "reason": "Spreads tightening",
                    "pnl_estimate": -abs(spread_change) * 0.01,
                }
        
        return {
            "action": "hold",
            "spread_change": spread_change,
            "spread_change_pct": round(spread_change_pct, 1),
        }