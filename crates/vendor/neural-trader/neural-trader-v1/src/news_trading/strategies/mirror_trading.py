"""Mirror Trading Strategy implementation for following institutional investors."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class MirrorTradingStrategy:
    """Mirror trading strategy for following institutional investors."""
    
    def __init__(self, portfolio_size: float = 100000):
        """
        Initialize mirror trading strategy.
        
        Args:
            portfolio_size: Total portfolio size for position sizing
        """
        self.portfolio_size = portfolio_size
        self.max_position_pct = 0.03  # Maximum 3% per position
        
        # Trusted institutions with confidence scores
        self.trusted_institutions = {
            "Berkshire Hathaway": 0.95,
            "Bridgewater Associates": 0.85,
            "Renaissance Technologies": 0.90,
            "Soros Fund Management": 0.80,
            "Tiger Global": 0.75,
            "Third Point": 0.70,
            "Pershing Square": 0.75,
            "Appaloosa Management": 0.80,
            "Greenlight Capital": 0.75,
            "Baupost Group": 0.85
        }
        
    def get_institution_confidence(self, institution: str) -> float:
        """
        Get confidence score for an institution.
        
        Args:
            institution: Institution name
            
        Returns:
            Confidence score between 0 and 1
        """
        return self.trusted_institutions.get(institution, 0.50)  # Default 0.50
        
    def parse_13f_filing(self, filing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse 13F filing and generate mirror signals.
        
        Args:
            filing: 13F filing data
            
        Returns:
            List of mirror trading signals
        """
        signals = []
        institution = filing["filer"]
        confidence = self.get_institution_confidence(institution)
        
        # High priority: New positions by trusted institutions
        for ticker in filing.get("new_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence,
                "priority": "high",
                "reasoning": f"{institution} initiated new position",
                "mirror_type": "new_position"
            })
            
        # Medium priority: Increased positions
        for ticker in filing.get("increased_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "confidence": confidence * 0.8,  # Slightly lower confidence
                "priority": "medium",
                "reasoning": f"{institution} increased position",
                "mirror_type": "add_position"
            })
            
        # Sell signals from eliminated positions
        for ticker in filing.get("sold_positions", []):
            signals.append({
                "ticker": ticker,
                "action": "sell",
                "confidence": confidence * 0.9,
                "priority": "high",
                "reasoning": f"{institution} eliminated position",
                "mirror_type": "exit_position"
            })
            
        return signals
        
    def score_insider_transaction(self, transaction: Dict[str, Any]) -> float:
        """
        Score insider transaction confidence.
        
        Args:
            transaction: Insider transaction data
            
        Returns:
            Confidence score between 0 and 1
        """
        role = transaction["role"]
        transaction_type = transaction["transaction_type"]
        
        # Base scores by role
        role_scores = {
            "CEO": 0.90,
            "CFO": 0.85,
            "President": 0.85,
            "COO": 0.80,
            "Director": 0.70,
            "VP": 0.65
        }
        
        base_score = role_scores.get(role, 0.60)
        
        # Adjust for transaction type
        if transaction_type == "Purchase":
            return base_score  # Buying is positive
        else:  # Sale
            return base_score * 0.3  # Selling is negative signal
            
    def calculate_mirror_position(self, institutional_trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate our position size based on institutional commitment.
        
        Args:
            institutional_trade: Institution's trade details
            
        Returns:
            Our position sizing details
        """
        inst_position_pct = institutional_trade["position_size_pct"]
        confidence = self.get_institution_confidence(institutional_trade["institution"])
        
        # Scale down position size
        # Max 20% of institutional position size, further scaled by confidence
        our_position_pct = min(
            inst_position_pct * 0.2 * confidence,
            self.max_position_pct  # Never more than 3%
        )
        
        # For unknown institutions (confidence 0.5), cap at 1%
        if confidence <= 0.5:
            our_position_pct = min(our_position_pct, 0.01)
        # For highly trusted institutions with big bets, use at least 2%
        elif confidence >= 0.9 and inst_position_pct >= 0.10:
            our_position_pct = max(our_position_pct, 0.02)
            
        # Dollar amount
        position_dollars = self.portfolio_size * our_position_pct
        
        return {
            "size_pct": our_position_pct,
            "size_dollars": position_dollars,
            "confidence": confidence,
            "reasoning": "Scaling down institutional position for risk management",
            "expected_holding_period": self.estimate_holding_period(
                institutional_trade["institution"]
            )
        }
        
    def estimate_holding_period(self, institution: str) -> str:
        """
        Estimate holding period based on institution's style.
        
        Args:
            institution: Institution name
            
        Returns:
            Expected holding period string
        """
        long_term_investors = ["Berkshire Hathaway", "Baupost Group", "Pershing Square"]
        medium_term = ["Tiger Global", "Third Point", "Greenlight Capital"]
        short_term = ["Renaissance Technologies", "Citadel"]
        
        if institution in long_term_investors:
            return "6-24 months"
        elif institution in medium_term:
            return "3-12 months"
        elif institution in short_term:
            return "1-6 months"
        else:
            return "3-6 months"  # Default
            
    async def generate_mirror_signal(self, filing_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Generate mirror trading signal from filing data.
        
        Args:
            filing_data: Institution filing data
            
        Returns:
            TradingSignal if valid, None otherwise
        """
        institution = filing_data["institution"]
        ticker = filing_data["ticker"]
        action = filing_data["action"]
        
        # Calculate position sizing
        institutional_trade = {
            "institution": institution,
            "ticker": ticker,
            "action": action,
            "position_size_pct": filing_data["total_value"] / (filing_data["total_value"] * 10),  # Estimate
            "dollar_value": filing_data["total_value"]
        }
        
        position = self.calculate_mirror_position(institutional_trade)
        
        # Determine signal type
        signal_type = SignalType.BUY if action == "buy" else SignalType.SELL
        
        # Get current price (use filing price as estimate)
        current_price = filing_data.get("avg_price", 100)
        
        # Set stops and targets based on institution style
        holding_period = self.estimate_holding_period(institution)
        if "24 months" in holding_period:
            stop_loss = current_price * 0.85  # Wide stop for long-term
            take_profit = current_price * 1.50  # High target
        elif "12 months" in holding_period:
            stop_loss = current_price * 0.90
            take_profit = current_price * 1.30
        else:
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.15
            
        signal = TradingSignal(
            id=f"mirror-{str(uuid.uuid4())[:8]}",
            timestamp=datetime.now(),
            asset=ticker,
            asset_type=AssetType.EQUITY,
            signal_type=signal_type,
            strategy=TradingStrategy.MIRROR,
            strength=position["confidence"],
            confidence=position["confidence"],
            risk_level=RiskLevel.LOW if position["confidence"] > 0.8 else RiskLevel.MEDIUM,
            position_size=position["size_pct"],
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=holding_period,
            source_events=[f"13F-{institution}"],
            reasoning=f"Mirroring {institution} {action} of {filing_data.get('shares', 0):,} shares",
            mirror_source=institution
        )
        
        return signal
        
    async def determine_entry_timing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal entry timing based on filing age and price movement.
        
        Args:
            filing_data: Filing and market data
            
        Returns:
            Entry timing recommendation
        """
        filing_age = datetime.now() - filing_data["filing_date"]
        price_change = (filing_data["current_price"] - filing_data["filing_price"]) / filing_data["filing_price"]
        
        # Recent filing with small price move - enter immediately
        if filing_age.days < 2 and abs(price_change) < 0.02:
            return {
                "entry_strategy": "immediate",
                "max_chase_price": filing_data["filing_price"] * 1.015,  # 1.5% chase
                "urgency": "high",
                "reasoning": "Recent filing with minimal price movement"
            }
            
        # Older filing or significant price move - wait for pullback
        elif filing_age.days > 2 or price_change > 0.03:
            return {
                "entry_strategy": "wait_for_pullback",
                "target_entry": filing_data["filing_price"] * 1.01,
                "urgency": "low",
                "reasoning": "Price has moved significantly since filing"
            }
            
        else:
            return {
                "entry_strategy": "scale_in",
                "initial_size": 0.5,  # Start with half position
                "urgency": "medium",
                "reasoning": "Moderate time passed, use scaling approach"
            }