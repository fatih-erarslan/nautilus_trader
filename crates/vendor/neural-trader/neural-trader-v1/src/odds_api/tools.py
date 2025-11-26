"""
The Odds API MCP Tools
Advanced analytics and betting tools using The Odds API
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
import json
from .client import OddsAPIClient

logger = logging.getLogger(__name__)

# Initialize client (will be set when tools are used)
_client = None

def get_client() -> OddsAPIClient:
    """Get or create The Odds API client"""
    global _client
    if _client is None:
        _client = OddsAPIClient()
    return _client

def get_sports_list() -> Dict[str, Any]:
    """
    Get list of available sports from The Odds API

    Returns:
        Dictionary with sports list and metadata
    """
    try:
        client = get_client()
        sports = client.get_sports()

        # Categorize sports
        active_sports = [s for s in sports if s.get('active', False)]
        by_group = {}
        for sport in active_sports:
            group = sport.get('group', 'other')
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(sport)

        return {
            'status': 'success',
            'total_sports': len(sports),
            'active_sports': len(active_sports),
            'sports_by_group': by_group,
            'all_sports': sports,
            'popular_sports': [
                s for s in active_sports
                if s.get('key') in ['americanfootball_nfl', 'basketball_nba', 'soccer_epl', 'baseball_mlb']
            ],
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error fetching sports list: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def get_live_odds(
    sport: str,
    regions: str = 'us',
    markets: str = 'h2h',
    bookmakers: Optional[str] = None,
    odds_format: str = 'decimal'
) -> Dict[str, Any]:
    """
    Get live odds for a specific sport

    Args:
        sport: Sport key (e.g., 'americanfootball_nfl', 'basketball_nba')
        regions: Comma-separated regions (us, uk, au, eu)
        markets: Comma-separated markets (h2h, spreads, totals)
        bookmakers: Optional comma-separated bookmaker keys
        odds_format: 'decimal' or 'american'

    Returns:
        Dictionary with odds data and analysis
    """
    try:
        client = get_client()
        odds_data = client.get_odds(
            sport=sport,
            regions=regions,
            markets=markets,
            bookmakers=bookmakers,
            odds_format=odds_format
        )

        # Analyze the odds
        analysis = analyze_odds_data(odds_data)

        return {
            'status': 'success',
            'sport': sport,
            'total_events': len(odds_data),
            'markets_requested': markets.split(','),
            'regions_requested': regions.split(','),
            'events': odds_data,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error fetching odds for {sport}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def get_event_odds(
    sport: str,
    event_id: str,
    regions: str = 'us',
    markets: str = 'h2h,spreads,totals',
    bookmakers: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed odds for a specific event

    Args:
        sport: Sport key
        event_id: Event ID from previous odds request
        regions: Comma-separated regions
        markets: Comma-separated markets
        bookmakers: Optional bookmaker filter

    Returns:
        Dictionary with event odds and analysis
    """
    try:
        client = get_client()
        event_data = client.get_event_odds(
            sport=sport,
            event_id=event_id,
            regions=regions,
            markets=markets,
            bookmakers=bookmakers
        )

        # Detailed analysis for single event
        analysis = analyze_single_event(event_data)

        return {
            'status': 'success',
            'sport': sport,
            'event_id': event_id,
            'event_data': event_data,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error fetching event odds {event_id}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'event_id': event_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def find_arbitrage_opportunities(
    sport: str,
    regions: str = 'us,uk,au',
    markets: str = 'h2h',
    min_profit_margin: float = 0.01
) -> Dict[str, Any]:
    """
    Find arbitrage opportunities across bookmakers

    Args:
        sport: Sport key
        regions: Multiple regions for more bookmaker coverage
        markets: Markets to analyze
        min_profit_margin: Minimum profit margin (0.01 = 1%)

    Returns:
        Dictionary with arbitrage opportunities
    """
    try:
        client = get_client()
        odds_data = client.get_odds(sport=sport, regions=regions, markets=markets)

        arbitrage_opportunities = []

        for event in odds_data:
            arb_analysis = find_event_arbitrage(event, min_profit_margin)
            if arb_analysis['has_arbitrage']:
                arbitrage_opportunities.append({
                    'event': {
                        'id': event.get('id'),
                        'sport_title': event.get('sport_title'),
                        'commence_time': event.get('commence_time'),
                        'home_team': event.get('home_team'),
                        'away_team': event.get('away_team')
                    },
                    'arbitrage': arb_analysis
                })

        return {
            'status': 'success',
            'sport': sport,
            'total_events_checked': len(odds_data),
            'arbitrage_opportunities': len(arbitrage_opportunities),
            'opportunities': arbitrage_opportunities,
            'min_profit_margin': min_profit_margin,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error finding arbitrage for {sport}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def get_bookmaker_odds(
    sport: str,
    bookmaker: str,
    regions: str = 'us',
    markets: str = 'h2h'
) -> Dict[str, Any]:
    """
    Get odds from a specific bookmaker

    Args:
        sport: Sport key
        bookmaker: Specific bookmaker key
        regions: Region
        markets: Markets to fetch

    Returns:
        Dictionary with bookmaker-specific odds
    """
    try:
        client = get_client()
        odds_data = client.get_odds(
            sport=sport,
            regions=regions,
            markets=markets,
            bookmakers=bookmaker
        )

        # Filter and analyze bookmaker data
        bookmaker_analysis = analyze_bookmaker_performance(odds_data, bookmaker)

        return {
            'status': 'success',
            'sport': sport,
            'bookmaker': bookmaker,
            'total_events': len(odds_data),
            'events': odds_data,
            'analysis': bookmaker_analysis,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error fetching {bookmaker} odds for {sport}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'bookmaker': bookmaker,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def analyze_odds_movement(
    sport: str,
    event_id: str,
    intervals: int = 5
) -> Dict[str, Any]:
    """
    Analyze odds movement over time (requires multiple API calls)

    Args:
        sport: Sport key
        event_id: Event ID to track
        intervals: Number of intervals to check

    Returns:
        Dictionary with odds movement analysis
    """
    try:
        client = get_client()

        # Note: This would require storing historical data or making multiple calls
        # For now, we'll provide a single snapshot with movement indicators

        current_odds = client.get_event_odds(sport=sport, event_id=event_id)
        movement_analysis = analyze_odds_trends(current_odds)

        return {
            'status': 'success',
            'sport': sport,
            'event_id': event_id,
            'current_odds': current_odds,
            'movement_analysis': movement_analysis,
            'note': 'Historical tracking requires data storage - showing current snapshot',
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error analyzing odds movement for {event_id}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'event_id': event_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def calculate_implied_probability(
    odds: float,
    odds_format: str = 'decimal'
) -> Dict[str, Any]:
    """
    Calculate implied probability from odds

    Args:
        odds: Odds value
        odds_format: 'decimal' or 'american'

    Returns:
        Dictionary with probability calculations
    """
    try:
        if odds_format == 'decimal':
            implied_prob = 1 / odds
        elif odds_format == 'american':
            if odds > 0:
                implied_prob = 100 / (odds + 100)
            else:
                implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            raise ValueError("odds_format must be 'decimal' or 'american'")

        return {
            'status': 'success',
            'odds': odds,
            'odds_format': odds_format,
            'implied_probability': implied_prob,
            'implied_probability_percent': implied_prob * 100,
            'fair_odds_decimal': 1 / implied_prob,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def compare_bookmaker_margins(
    sport: str,
    regions: str = 'us',
    markets: str = 'h2h'
) -> Dict[str, Any]:
    """
    Compare bookmaker margins across providers

    Args:
        sport: Sport key
        regions: Regions to compare
        markets: Markets to analyze

    Returns:
        Dictionary with margin comparison
    """
    try:
        client = get_client()
        odds_data = client.get_odds(sport=sport, regions=regions, markets=markets)

        bookmaker_margins = calculate_bookmaker_margins(odds_data)

        return {
            'status': 'success',
            'sport': sport,
            'total_events_analyzed': len(odds_data),
            'bookmaker_margins': bookmaker_margins,
            'best_value_bookmaker': min(bookmaker_margins.items(), key=lambda x: x[1]['average_margin'])[0] if bookmaker_margins else None,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error comparing margins for {sport}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def get_upcoming_events(
    sport: str,
    regions: str = 'us',
    markets: str = 'h2h',
    days_ahead: int = 7
) -> Dict[str, Any]:
    """
    Get upcoming events with odds

    Args:
        sport: Sport key
        regions: Regions
        markets: Markets
        days_ahead: How many days ahead to look

    Returns:
        Dictionary with upcoming events
    """
    try:
        client = get_client()
        odds_data = client.get_odds(sport=sport, regions=regions, markets=markets)

        # Filter for upcoming events within time window
        cutoff_time = datetime.now() + timedelta(days=days_ahead)
        upcoming_events = []

        for event in odds_data:
            commence_time_str = event.get('commence_time')
            if commence_time_str:
                try:
                    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                    if commence_time <= cutoff_time:
                        upcoming_events.append(event)
                except ValueError:
                    # Skip events with invalid datetime
                    continue

        # Sort by commence time
        upcoming_events.sort(key=lambda x: x.get('commence_time', ''))

        return {
            'status': 'success',
            'sport': sport,
            'days_ahead': days_ahead,
            'total_upcoming_events': len(upcoming_events),
            'events': upcoming_events,
            'next_event': upcoming_events[0] if upcoming_events else None,
            'timestamp': datetime.now().isoformat(),
            'usage': client.get_usage_info()
        }
    except Exception as e:
        logger.error(f"Error fetching upcoming events for {sport}: {e}")
        return {
            'status': 'error',
            'sport': sport,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Helper functions for analysis

def analyze_odds_data(odds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze odds data for insights"""
    if not odds_data:
        return {'total_events': 0}

    total_bookmakers = set()
    market_coverage = {}

    for event in odds_data:
        for bookmaker in event.get('bookmakers', []):
            total_bookmakers.add(bookmaker.get('key'))
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                if market_key not in market_coverage:
                    market_coverage[market_key] = 0
                market_coverage[market_key] += 1

    return {
        'total_events': len(odds_data),
        'unique_bookmakers': len(total_bookmakers),
        'bookmakers': list(total_bookmakers),
        'market_coverage': market_coverage,
        'avg_bookmakers_per_event': sum(len(event.get('bookmakers', [])) for event in odds_data) / len(odds_data)
    }

def analyze_single_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed analysis for a single event"""
    if not event_data:
        return {}

    bookmakers = event_data.get('bookmakers', [])
    analysis = {
        'total_bookmakers': len(bookmakers),
        'markets_available': [],
        'best_odds': {},
        'worst_odds': {},
        'odds_variance': {}
    }

    # Analyze each market
    market_odds = {}
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            market_key = market.get('key')
            if market_key not in analysis['markets_available']:
                analysis['markets_available'].append(market_key)

            if market_key not in market_odds:
                market_odds[market_key] = {}

            for outcome in market.get('outcomes', []):
                outcome_name = outcome.get('name')
                odds = outcome.get('price', 0)

                if outcome_name not in market_odds[market_key]:
                    market_odds[market_key][outcome_name] = []
                market_odds[market_key][outcome_name].append(odds)

    # Calculate best/worst odds and variance
    for market, outcomes in market_odds.items():
        analysis['best_odds'][market] = {}
        analysis['worst_odds'][market] = {}
        analysis['odds_variance'][market] = {}

        for outcome, odds_list in outcomes.items():
            if odds_list:
                analysis['best_odds'][market][outcome] = max(odds_list)
                analysis['worst_odds'][market][outcome] = min(odds_list)
                analysis['odds_variance'][market][outcome] = statistics.variance(odds_list) if len(odds_list) > 1 else 0

    return analysis

def find_event_arbitrage(event: Dict[str, Any], min_margin: float) -> Dict[str, Any]:
    """Find arbitrage opportunities for a single event"""
    bookmakers = event.get('bookmakers', [])

    # Focus on h2h market for simplicity
    h2h_odds = {}
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market.get('key') == 'h2h':
                bookmaker_key = bookmaker.get('key')
                h2h_odds[bookmaker_key] = {}

                for outcome in market.get('outcomes', []):
                    outcome_name = outcome.get('name')
                    odds = outcome.get('price', 0)
                    h2h_odds[bookmaker_key][outcome_name] = odds

    if len(h2h_odds) < 2:
        return {'has_arbitrage': False, 'reason': 'Not enough bookmakers'}

    # Find best odds for each outcome
    best_odds = {}
    best_bookmakers = {}

    all_outcomes = set()
    for bookmaker_odds in h2h_odds.values():
        all_outcomes.update(bookmaker_odds.keys())

    for outcome in all_outcomes:
        best_odds[outcome] = 0
        best_bookmakers[outcome] = None

        for bookmaker, odds_dict in h2h_odds.items():
            if outcome in odds_dict:
                odds = odds_dict[outcome]
                if odds > best_odds[outcome]:
                    best_odds[outcome] = odds
                    best_bookmakers[outcome] = bookmaker

    # Calculate arbitrage
    if len(best_odds) >= 2:
        implied_probs = [1/odds for odds in best_odds.values() if odds > 0]
        total_implied_prob = sum(implied_probs)

        if total_implied_prob < 1:
            profit_margin = 1 - total_implied_prob
            if profit_margin >= min_margin:
                return {
                    'has_arbitrage': True,
                    'profit_margin': profit_margin,
                    'profit_percentage': profit_margin * 100,
                    'best_odds': best_odds,
                    'best_bookmakers': best_bookmakers,
                    'total_implied_probability': total_implied_prob
                }

    return {'has_arbitrage': False, 'reason': 'No profitable arbitrage found'}

def analyze_bookmaker_performance(odds_data: List[Dict[str, Any]], target_bookmaker: str) -> Dict[str, Any]:
    """Analyze performance of a specific bookmaker"""
    events_with_bookmaker = 0
    total_events = len(odds_data)

    for event in odds_data:
        for bookmaker in event.get('bookmakers', []):
            if bookmaker.get('key') == target_bookmaker:
                events_with_bookmaker += 1
                break

    coverage = events_with_bookmaker / total_events if total_events > 0 else 0

    return {
        'bookmaker': target_bookmaker,
        'events_covered': events_with_bookmaker,
        'total_events': total_events,
        'coverage_percentage': coverage * 100,
        'availability': 'high' if coverage > 0.8 else 'medium' if coverage > 0.5 else 'low'
    }

def analyze_odds_trends(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze odds trends (placeholder for historical analysis)"""
    return {
        'trend_analysis': 'Historical tracking not yet implemented',
        'current_snapshot': datetime.now().isoformat(),
        'recommendation': 'Store odds data over time to enable trend analysis'
    }

def calculate_bookmaker_margins(odds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate margins for each bookmaker"""
    bookmaker_margins = {}

    for event in odds_data:
        for bookmaker in event.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key')

            if bookmaker_key not in bookmaker_margins:
                bookmaker_margins[bookmaker_key] = {'margins': [], 'events': 0}

            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])
                    if len(outcomes) >= 2:
                        implied_probs = [1/outcome.get('price', 1) for outcome in outcomes if outcome.get('price', 0) > 0]
                        if implied_probs:
                            total_implied_prob = sum(implied_probs)
                            margin = total_implied_prob - 1
                            bookmaker_margins[bookmaker_key]['margins'].append(margin)

            bookmaker_margins[bookmaker_key]['events'] += 1

    # Calculate averages
    for bookmaker, data in bookmaker_margins.items():
        margins = data['margins']
        if margins:
            data['average_margin'] = statistics.mean(margins)
            data['margin_variance'] = statistics.variance(margins) if len(margins) > 1 else 0
        else:
            data['average_margin'] = 0
            data['margin_variance'] = 0

    return bookmaker_margins