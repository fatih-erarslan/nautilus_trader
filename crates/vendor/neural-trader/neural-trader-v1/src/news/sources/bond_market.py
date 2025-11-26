"""
Bond market utility functions for yield analysis and signal generation
"""
from typing import Dict, List, Any
from ..models import NewsItem


def detect_yield_changes(current_yields: Dict[str, float], 
                        previous_yields: Dict[str, float], 
                        significance_threshold: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Detect significant changes in bond yields
    
    Args:
        current_yields: Current yield levels by maturity
        previous_yields: Previous yield levels by maturity  
        significance_threshold: Minimum basis point change to be considered significant
        
    Returns:
        Dictionary of yield changes by maturity
    """
    changes = {}
    
    for maturity in current_yields:
        if maturity in previous_yields:
            current = current_yields[maturity]
            previous = previous_yields[maturity]
            
            # Calculate change in basis points
            change_bps = round((current - previous) * 100)
            
            changes[maturity] = {
                "change_bps": change_bps,
                "is_significant": abs(change_bps) >= significance_threshold,
                "direction": "up" if change_bps > 0 else "down" if change_bps < 0 else "unchanged",
                "current_yield": current,
                "previous_yield": previous
            }
    
    return changes


def generate_bond_signals(news_items: List[NewsItem]) -> List[Dict[str, Any]]:
    """
    Generate bond trading signals from news items
    
    Args:
        news_items: List of news items to analyze
        
    Returns:
        List of trading signals
    """
    signals = []
    
    for item in news_items:
        signal = None
        
        # Federal Reserve sentiment signals
        if item.source == "federal_reserve":
            sentiment = item.metadata.get("sentiment", "neutral")
            
            if sentiment == "dovish":
                signal = {
                    "action": "buy",
                    "instrument": "10Y_Treasury",
                    "reasoning": "Dovish Fed policy typically leads to lower rates and higher bond prices",
                    "confidence": 0.8,
                    "source": item.id
                }
            elif sentiment == "hawkish":
                signal = {
                    "action": "sell",
                    "instrument": "10Y_Treasury", 
                    "reasoning": "Hawkish Fed policy typically leads to higher rates and lower bond prices",
                    "confidence": 0.8,
                    "source": item.id
                }
        
        # Treasury auction quality signals
        elif item.source == "treasury_direct":
            if "bid_to_cover" in item.metadata:
                bid_to_cover = item.metadata["bid_to_cover"]
                demand_strength = item.metadata.get("demand_strength", "moderate")
                
                if demand_strength == "weak":
                    signal = {
                        "action": "wait",
                        "instrument": "Treasury_Bonds",
                        "reasoning": f"Weak auction demand (bid-to-cover: {bid_to_cover}) suggests limited interest",
                        "confidence": 0.6,
                        "source": item.id
                    }
                elif demand_strength == "strong":
                    signal = {
                        "action": "buy",
                        "instrument": "Treasury_Bonds",
                        "reasoning": f"Strong auction demand (bid-to-cover: {bid_to_cover}) shows healthy interest",
                        "confidence": 0.7,
                        "source": item.id
                    }
        
        if signal:
            signals.append(signal)
    
    return signals