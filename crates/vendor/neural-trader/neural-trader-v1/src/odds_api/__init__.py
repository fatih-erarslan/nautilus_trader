"""
The Odds API Integration Module
Provides access to live sports betting odds from multiple bookmakers
"""

from .client import OddsAPIClient
from .tools import (
    get_sports_list,
    get_live_odds,
    find_arbitrage_opportunities,
    get_event_odds,
    get_bookmaker_odds,
    analyze_odds_movement,
    calculate_implied_probability,
    compare_bookmaker_margins,
    get_upcoming_events
)

__all__ = [
    'OddsAPIClient',
    'get_sports_list',
    'get_live_odds',
    'find_arbitrage_opportunities',
    'get_event_odds',
    'get_bookmaker_odds',
    'analyze_odds_movement',
    'calculate_implied_probability',
    'compare_bookmaker_margins',
    'get_upcoming_events'
]

__version__ = "1.0.0"