#!/usr/bin/env python3
"""
The Odds API - Example Usage
Demonstrates how to use The Odds API integration with the neural trading system
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from odds_api.client import OddsAPIClient
from odds_api.tools import (
    get_sports_list,
    get_live_odds,
    find_arbitrage_opportunities,
    calculate_implied_probability,
    compare_bookmaker_margins
)

def test_connection():
    """Test connection to The Odds API"""
    print("🔗 Testing connection to The Odds API...")

    try:
        client = OddsAPIClient()
        result = client.validate_connection()

        if result['status'] == 'success':
            print("✅ Connection successful!")
            print(f"   Sports available: {result['sports_available']}")
            print(f"   Requests remaining: {result['requests_remaining']}")
        else:
            print("❌ Connection failed!")
            print(f"   Error: {result['error']}")

        return result['status'] == 'success'

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def explore_sports():
    """Explore available sports"""
    print("\n🏈 Exploring available sports...")

    try:
        sports_data = get_sports_list()

        if sports_data['status'] == 'success':
            print(f"✅ Found {sports_data['total_sports']} sports")
            print(f"   Active sports: {sports_data['active_sports']}")

            print("\n🔥 Popular Sports:")
            for sport in sports_data.get('popular_sports', [])[:5]:
                print(f"   • {sport.get('title', 'Unknown')} ({sport.get('key', 'N/A')})")

            print("\n📊 Sports by Group:")
            for group, sports in sports_data.get('sports_by_group', {}).items():
                print(f"   {group}: {len(sports)} sports")

            return sports_data['popular_sports']
        else:
            print(f"❌ Failed to get sports: {sports_data.get('error', 'Unknown error')}")
            return []

    except Exception as e:
        print(f"❌ Error exploring sports: {e}")
        return []

def get_sample_odds(sport_key: str = "americanfootball_nfl"):
    """Get sample odds for a sport"""
    print(f"\n📊 Getting odds for {sport_key}...")

    try:
        odds_data = get_live_odds(
            sport=sport_key,
            regions="us",
            markets="h2h,spreads",
            odds_format="decimal"
        )

        if odds_data['status'] == 'success':
            events = odds_data.get('events', [])
            print(f"✅ Found {len(events)} events")

            if events:
                # Show first event as example
                event = events[0]
                print(f"\n🎯 Sample Event:")
                print(f"   {event.get('home_team', 'Home')} vs {event.get('away_team', 'Away')}")
                print(f"   Start: {event.get('commence_time', 'TBD')}")

                bookmakers = event.get('bookmakers', [])
                print(f"   Bookmakers: {len(bookmakers)}")

                if bookmakers:
                    bm = bookmakers[0]
                    print(f"   Sample odds from {bm.get('title', 'Unknown')}:")
                    for market in bm.get('markets', []):
                        print(f"     {market.get('key', 'Unknown market')}:")
                        for outcome in market.get('outcomes', []):
                            print(f"       {outcome.get('name', 'Unknown')}: {outcome.get('price', 'N/A')}")

            analysis = odds_data.get('analysis', {})
            print(f"\n📈 Analysis:")
            print(f"   Unique bookmakers: {analysis.get('unique_bookmakers', 0)}")
            print(f"   Avg bookmakers per event: {analysis.get('avg_bookmakers_per_event', 0):.1f}")

            return odds_data
        else:
            print(f"❌ Failed to get odds: {odds_data.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"❌ Error getting odds: {e}")
        return None

def find_arbitrage_example(sport_key: str = "basketball_nba"):
    """Find arbitrage opportunities"""
    print(f"\n⚡ Searching for arbitrage in {sport_key}...")

    try:
        arb_data = find_arbitrage_opportunities(
            sport=sport_key,
            regions="us,uk",
            markets="h2h",
            min_profit_margin=0.01  # 1% minimum
        )

        if arb_data['status'] == 'success':
            opportunities = arb_data.get('arbitrage_opportunities', 0)
            print(f"✅ Found {opportunities} arbitrage opportunities")

            if opportunities > 0:
                print("\n💰 Arbitrage Opportunities:")
                for i, opp in enumerate(arb_data.get('opportunities', [])[:3]):
                    event = opp.get('event', {})
                    arbitrage = opp.get('arbitrage', {})

                    print(f"\n   {i+1}. {event.get('home_team', 'Home')} vs {event.get('away_team', 'Away')}")
                    print(f"      Profit margin: {arbitrage.get('profit_percentage', 0):.2f}%")

                    best_odds = arbitrage.get('best_odds', {})
                    best_bookmakers = arbitrage.get('best_bookmakers', {})

                    for outcome, odds in best_odds.items():
                        bookmaker = best_bookmakers.get(outcome, 'Unknown')
                        print(f"      {outcome}: {odds} at {bookmaker}")
            else:
                print("   No profitable arbitrage found at this time")

            return arb_data
        else:
            print(f"❌ Failed to find arbitrage: {arb_data.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"❌ Error finding arbitrage: {e}")
        return None

def probability_examples():
    """Calculate probability examples"""
    print("\n🎲 Probability Calculations:")

    test_odds = [
        (2.50, "decimal"),
        (1.80, "decimal"),
        (150, "american"),
        (-110, "american")
    ]

    for odds, format_type in test_odds:
        try:
            prob_data = calculate_implied_probability(odds, format_type)

            if prob_data['status'] == 'success':
                print(f"   {format_type.title()} odds {odds}:")
                print(f"     Implied probability: {prob_data['implied_probability_percent']:.1f}%")
                print(f"     Fair decimal odds: {prob_data['fair_odds_decimal']:.2f}")
            else:
                print(f"   Error calculating {odds}: {prob_data.get('error', 'Unknown')}")

        except Exception as e:
            print(f"   Error with {odds}: {e}")

def compare_margins_example(sport_key: str = "americanfootball_nfl"):
    """Compare bookmaker margins"""
    print(f"\n📊 Comparing margins for {sport_key}...")

    try:
        margins_data = compare_bookmaker_margins(
            sport=sport_key,
            regions="us",
            markets="h2h"
        )

        if margins_data['status'] == 'success':
            bookmaker_margins = margins_data.get('bookmaker_margins', {})
            best_value = margins_data.get('best_value_bookmaker')

            print(f"✅ Analyzed {margins_data.get('total_events_analyzed', 0)} events")

            if best_value:
                print(f"🏆 Best value bookmaker: {best_value}")

            if bookmaker_margins:
                print("\n📈 Bookmaker Margins:")
                sorted_margins = sorted(
                    bookmaker_margins.items(),
                    key=lambda x: x[1].get('average_margin', 999)
                )

                for bookmaker, data in sorted_margins[:5]:
                    avg_margin = data.get('average_margin', 0) * 100
                    events = data.get('events', 0)
                    print(f"   {bookmaker}: {avg_margin:.2f}% (from {events} events)")

            return margins_data
        else:
            print(f"❌ Failed to compare margins: {margins_data.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"❌ Error comparing margins: {e}")
        return None

def kelly_criterion_example():
    """Demonstrate Kelly Criterion betting"""
    print("\n💡 Kelly Criterion Betting Example:")

    # Example: You think a team has 60% chance to win, but bookmaker offers 2.0 odds (50% implied)
    true_probability = 0.60
    bookmaker_odds = 2.0
    bankroll = 1000

    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win probability, q = lose probability
    b = bookmaker_odds - 1
    p = true_probability
    q = 1 - p

    kelly_fraction = (b * p - q) / b

    if kelly_fraction > 0:
        bet_amount = bankroll * kelly_fraction
        print(f"   📈 Positive Expected Value!")
        print(f"   True probability: {p*100:.1f}%")
        print(f"   Bookmaker odds: {bookmaker_odds}")
        print(f"   Kelly fraction: {kelly_fraction:.3f}")
        print(f"   Recommended bet: ${bet_amount:.2f} ({kelly_fraction*100:.1f}% of bankroll)")
    else:
        print(f"   📉 Negative Expected Value - Don't bet!")
        print(f"   Kelly fraction: {kelly_fraction:.3f}")

def usage_monitoring():
    """Monitor API usage"""
    print("\n📊 API Usage Monitoring:")

    try:
        client = OddsAPIClient()
        usage = client.get_usage_info()

        print(f"   Requests remaining: {usage.get('requests_remaining', 'Unknown')}")
        print(f"   Requests used: {usage.get('requests_used', 'Unknown')}")
        print(f"   Last updated: {usage.get('last_updated', 'Unknown')}")

    except Exception as e:
        print(f"   Error checking usage: {e}")

def main():
    """Main example function"""
    print("🎯 The Odds API Integration - Example Usage")
    print("=" * 50)

    # Test connection
    if not test_connection():
        print("\n❌ Cannot continue without valid API connection")
        print("Please check your THE_ODDS_API_KEY environment variable")
        return

    # Explore available sports
    popular_sports = explore_sports()

    if popular_sports:
        # Use the first popular sport for examples
        sport_key = popular_sports[0].get('key', 'americanfootball_nfl')

        # Get sample odds
        get_sample_odds(sport_key)

        # Find arbitrage opportunities
        find_arbitrage_example(sport_key)

        # Compare margins
        compare_margins_example(sport_key)

    # Probability calculations
    probability_examples()

    # Kelly Criterion example
    kelly_criterion_example()

    # Monitor usage
    usage_monitoring()

    print("\n🎉 Example completed!")
    print("\nNext steps:")
    print("1. Set up your THE_ODDS_API_KEY environment variable")
    print("2. Run this script to test your integration")
    print("3. Check the documentation in docs/integrations/THE_ODDS_API_INTEGRATION.md")
    print("4. Start building your betting strategies!")

if __name__ == "__main__":
    main()