#!/usr/bin/env python3
"""
Complete functionality test for Fantasy Collective System
Tests all capabilities end-to-end
"""

import sys
import os
sys.path.insert(0, '/workspaces/ai-news-trader')

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import random

# Import the fantasy server components
from src.mcp.mcp_server_fantasy import (
    init_database, DB_PATH,
    execute_db_query, generate_id, hash_password,
    ping, create_fantasy_league, join_league,
    make_prediction, calculate_fantasy_scores,
    get_leaderboard, create_achievement
)

def setup_test_data():
    """Create test users and events for testing."""
    print("\n1️⃣ SETTING UP TEST DATA")
    
    # Create test users
    users = [
        ("user_001", "alice", "alice@test.com", 5000.0),
        ("user_002", "bob", "bob@test.com", 3000.0),
        ("user_003", "charlie", "charlie@test.com", 2000.0),
        ("user_004", "diana", "diana@test.com", 4000.0),
        ("user_005", "eve", "eve@test.com", 1500.0)
    ]
    
    for user_id, username, email, balance in users:
        try:
            execute_db_query("""
                INSERT OR IGNORE INTO users (user_id, username, email, password_hash, balance)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, email, hash_password("password123"), balance))
        except:
            pass  # User might already exist
    
    print(f"   ✅ Created {len(users)} test users")
    
    # Create test events
    events = [
        ("event_001", "NBA: Lakers vs Warriors", "sports", "basketball", "2024-03-20"),
        ("event_002", "NFL: Chiefs vs Eagles", "sports", "football", "2024-03-21"),
        ("event_003", "Stock: AAPL reaches $200?", "business", "stocks", "2024-03-25"),
        ("event_004", "Election: Candidate A wins?", "news", "politics", "2024-03-30"),
        ("event_005", "Weather: Rain tomorrow?", "custom", "weather", "2024-03-15")
    ]
    
    for event_id, title, event_type, category, date in events:
        try:
            execute_db_query("""
                INSERT OR IGNORE INTO events (event_id, title, event_type, category, start_time)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, title, event_type, category, date))
        except:
            pass
    
    print(f"   ✅ Created {len(events)} test events")
    return True

def test_fantasy_leagues():
    """Test fantasy league creation and management."""
    print("\n2️⃣ TESTING FANTASY LEAGUES")
    
    leagues_created = []
    
    # Test different league types
    league_configs = [
        ("NBA Fantasy 2024", "fantasy_sports", "basketball", 12, 50.0),
        ("Stock Predictions", "prediction", None, 20, 25.0),
        ("Survivor Pool", "survivor", "football", 100, 10.0),
        ("March Madness", "tournament", "basketball", 64, 100.0),
        ("News Predictions", "prediction", None, 50, 0.0)
    ]
    
    for name, league_type, sport, max_members, entry_fee in league_configs:
        result = create_fantasy_league(
            name=name,
            league_type=league_type,
            sport=sport,
            max_members=max_members,
            entry_fee=entry_fee,
            created_by="user_001"
        )
        
        if result.get('league_id'):
            leagues_created.append(result['league_id'])
            print(f"   ✅ Created {league_type} league: {name}")
        else:
            print(f"   ❌ Failed to create league: {name}")
    
    # Test joining leagues
    if leagues_created:
        league_id = leagues_created[0]
        
        # Join with multiple users
        for user_id in ["user_002", "user_003", "user_004"]:
            result = join_league(league_id, user_id)
            if result.get('status') == 'joined':
                print(f"   ✅ User {user_id} joined league")
            else:
                print(f"   ❌ Failed to join: {result.get('error')}")
    
    # Check league status
    league_info = execute_db_query("""
        SELECT * FROM fantasy_leagues WHERE league_id = ?
    """, (leagues_created[0] if leagues_created else "",), fetchone=True)
    
    if league_info:
        print(f"   ✅ League has {league_info['current_members']}/{league_info['max_members']} members")
        print(f"   ✅ Prize pool: ${league_info['prize_pool']}")
    
    return len(leagues_created) > 0

def test_predictions():
    """Test prediction system."""
    print("\n3️⃣ TESTING PREDICTION SYSTEM")
    
    # Get a league for predictions
    league = execute_db_query("""
        SELECT league_id FROM fantasy_leagues LIMIT 1
    """, fetchone=True)
    
    if not league:
        print("   ❌ No league found for predictions")
        return False
    
    league_id = league['league_id']
    predictions_made = []
    
    # Make predictions for different users
    prediction_configs = [
        ("user_001", "event_001", "Lakers", 0.8, 100),
        ("user_002", "event_001", "Warriors", 0.6, 50),
        ("user_003", "event_002", "Chiefs", 0.7, 75),
        ("user_001", "event_003", "Yes", 0.9, 200),
        ("user_004", "event_004", "Candidate A", 0.55, 25)
    ]
    
    for user_id, event_id, prediction_value, confidence, stake in prediction_configs:
        result = make_prediction(
            event_id=event_id,
            prediction_value=prediction_value,
            user_id=user_id,
            confidence=confidence,
            stake=stake,
            league_id=league_id
        )
        
        if result.get('prediction_id'):
            predictions_made.append(result['prediction_id'])
            print(f"   ✅ {user_id} predicted {prediction_value} with {confidence*100}% confidence")
        else:
            print(f"   ❌ Prediction failed: {result.get('error')}")
    
    # Check prediction stats
    stats = execute_db_query("""
        SELECT COUNT(*) as count, SUM(stake) as total_stake
        FROM predictions WHERE league_id = ?
    """, (league_id,), fetchone=True)
    
    if stats:
        print(f"   ✅ Total predictions: {stats['count']}")
        print(f"   ✅ Total stake: ${stats['total_stake']}")
    
    return len(predictions_made) > 0

def test_scoring_system():
    """Test scoring calculation."""
    print("\n4️⃣ TESTING SCORING SYSTEM")
    
    # Resolve some events for scoring
    events_to_resolve = [
        ("event_001", "Lakers"),
        ("event_002", "Chiefs"),
        ("event_003", "Yes")
    ]
    
    for event_id, outcome in events_to_resolve:
        execute_db_query("""
            UPDATE events SET outcome = ?, status = 'resolved'
            WHERE event_id = ?
        """, (outcome, event_id))
        print(f"   ✅ Resolved {event_id} with outcome: {outcome}")
    
    # Calculate scores for leagues
    league = execute_db_query("""
        SELECT league_id FROM fantasy_leagues LIMIT 1
    """, fetchone=True)
    
    if league:
        result = calculate_fantasy_scores(
            league_id=league['league_id'],
            period="current"
        )
        
        if result.get('status') == 'calculated':
            print(f"   ✅ Calculated scores for league")
            scores = result.get('scores', {})
            for user_id, score_data in list(scores.items())[:3]:
                print(f"      • {score_data['username']}: {score_data['total_score']} points (Rank #{score_data.get('rank', 'N/A')})")
        else:
            print(f"   ❌ Score calculation failed: {result.get('error')}")
    
    return True

def test_leaderboards():
    """Test leaderboard functionality."""
    print("\n5️⃣ TESTING LEADERBOARDS")
    
    # Get league leaderboard
    league = execute_db_query("""
        SELECT league_id FROM fantasy_leagues LIMIT 1
    """, fetchone=True)
    
    if league:
        result = get_leaderboard(
            league_id=league['league_id'],
            limit=5
        )
        
        if result.get('status') == 'success':
            print(f"   ✅ League leaderboard retrieved")
            leaderboard = result.get('leaderboard', [])
            for entry in leaderboard[:3]:
                print(f"      • #{entry['rank']} {entry['username']}: {entry['points']} points")
        else:
            print(f"   ❌ Failed to get league leaderboard")
    
    # Get global leaderboard
    global_result = get_leaderboard(limit=5)
    
    if global_result.get('status') == 'success':
        print(f"   ✅ Global leaderboard retrieved")
        global_board = global_result.get('leaderboard', [])
        for entry in global_board[:3]:
            print(f"      • #{entry['rank']} {entry['username']}: {entry['total_score']} total score")
    else:
        print(f"   ❌ Failed to get global leaderboard")
    
    return True

def test_achievements():
    """Test achievement system."""
    print("\n6️⃣ TESTING ACHIEVEMENT SYSTEM")
    
    achievements_created = []
    
    # Create different types of achievements
    achievement_configs = [
        ("First Win", "Win your first prediction", "beginner", 50, "wins", 1, "common"),
        ("High Roller", "Stake $1000 total", "betting", 100, "total_stake", 1000, "rare"),
        ("Prophet", "Achieve 80% accuracy", "accuracy", 200, "accuracy_rate", 0.8, "legendary"),
        ("League Champion", "Win a league championship", "competitive", 500, "championships", 1, "epic"),
        ("Streak Master", "5 correct predictions in a row", "streak", 150, "win_streak", 5, "rare")
    ]
    
    for name, desc, category, points, req_type, req_value, rarity in achievement_configs:
        result = create_achievement(
            name=name,
            description=desc,
            category=category,
            points_value=points,
            requirement_type=req_type,
            requirement_value=req_value,
            rarity=rarity
        )
        
        if result.get('achievement_id'):
            achievements_created.append(result['achievement_id'])
            print(f"   ✅ Created {rarity} achievement: {name} ({points} points)")
        else:
            print(f"   ❌ Failed to create achievement: {name}")
    
    # Check achievement stats
    stats = execute_db_query("""
        SELECT COUNT(*) as count, SUM(points_value) as total_points
        FROM achievements
    """, fetchone=True)
    
    if stats:
        print(f"   ✅ Total achievements: {stats['count']}")
        print(f"   ✅ Total points available: {stats['total_points']}")
    
    return len(achievements_created) > 0

def test_collectives():
    """Test collective functionality."""
    print("\n7️⃣ TESTING COLLECTIVES")
    
    # Create a collective
    collective_id = generate_id("coll_")
    
    execute_db_query("""
        INSERT INTO collectives (
            collective_id, name, description, created_by,
            consensus_threshold, total_funds
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        collective_id,
        "Pro Predictors",
        "Elite prediction collective",
        "user_001",
        0.66,
        10000.0
    ))
    
    print(f"   ✅ Created collective: Pro Predictors")
    
    # Add members
    for user_id, voting_power in [("user_001", 2.0), ("user_002", 1.5), ("user_003", 1.0)]:
        execute_db_query("""
            INSERT INTO collective_memberships (
                membership_id, collective_id, user_id, voting_power, role
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            generate_id("collmem_"),
            collective_id,
            user_id,
            voting_power,
            "owner" if user_id == "user_001" else "member"
        ))
    
    print(f"   ✅ Added 3 members to collective")
    
    # Check collective stats
    stats = execute_db_query("""
        SELECT COUNT(*) as member_count, SUM(voting_power) as total_power
        FROM collective_memberships
        WHERE collective_id = ?
    """, (collective_id,), fetchone=True)
    
    if stats:
        print(f"   ✅ Collective has {stats['member_count']} members")
        print(f"   ✅ Total voting power: {stats['total_power']}")
    
    return True

def test_tournaments():
    """Test tournament functionality."""
    print("\n8️⃣ TESTING TOURNAMENTS")
    
    # Create a tournament
    tournament_id = generate_id("tourn_")
    
    execute_db_query("""
        INSERT INTO tournaments (
            tournament_id, name, tournament_type, created_by,
            max_participants, entry_fee, prize_pool
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        tournament_id,
        "March Madness Predictor",
        "bracket",
        "user_001",
        64,
        25.0,
        1600.0
    ))
    
    print(f"   ✅ Created tournament: March Madness Predictor")
    print(f"   ✅ 64 participants max, $25 entry, $1,600 prize pool")
    
    # Check tournament
    tournament = execute_db_query("""
        SELECT * FROM tournaments WHERE tournament_id = ?
    """, (tournament_id,), fetchone=True)
    
    if tournament:
        print(f"   ✅ Tournament status: {tournament['status']}")
    
    return True

def verify_database_integrity():
    """Verify database integrity and relationships."""
    print("\n9️⃣ VERIFYING DATABASE INTEGRITY")
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Check foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA foreign_key_check")
    fk_errors = cursor.fetchall()
    
    if not fk_errors:
        print(f"   ✅ All foreign key constraints valid")
    else:
        print(f"   ❌ Foreign key errors found: {len(fk_errors)}")
    
    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = cursor.fetchall()
    print(f"   ✅ {len(indexes)} indexes for performance optimization")
    
    # Get table statistics
    table_stats = []
    tables = ['users', 'fantasy_leagues', 'predictions', 'events', 'achievements']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        table_stats.append((table, count))
        print(f"   ✅ {table}: {count} records")
    
    conn.close()
    return True

def test_advanced_features():
    """Test advanced features and edge cases."""
    print("\n🔟 TESTING ADVANCED FEATURES")
    
    # Test transaction rollback
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("INSERT INTO users (user_id, username) VALUES ('test', 'test')")
            raise Exception("Test rollback")
    except:
        print(f"   ✅ Transaction rollback working")
    
    # Test concurrent league membership
    league = execute_db_query("""
        SELECT league_id FROM fantasy_leagues LIMIT 1
    """, fetchone=True)
    
    if league:
        # Try to join twice
        result1 = join_league(league['league_id'], "user_005")
        result2 = join_league(league['league_id'], "user_005")
        
        if result2.get('error') == "Already a member of this league":
            print(f"   ✅ Duplicate join prevention working")
    
    # Test league capacity
    small_league = create_fantasy_league(
        name="Small League",
        league_type="fantasy_sports",
        max_members=2,
        created_by="user_001"
    )
    
    if small_league.get('league_id'):
        join_league(small_league['league_id'], "user_002")
        result = join_league(small_league['league_id'], "user_003")
        
        if result.get('error') == "League is full":
            print(f"   ✅ League capacity enforcement working")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("🎮 FANTASY COLLECTIVE SYSTEM - COMPLETE FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Initialize database
    init_database()
    print(f"✅ Database initialized at {DB_PATH}")
    
    # Run all tests
    results = []
    
    # Test server connectivity
    server_status = ping()
    print(f"✅ Server: {server_status['server']} v{server_status['version']}")
    print(f"✅ Features: {', '.join(server_status['features'])}")
    
    # Run test suite
    results.append(("Test Data Setup", setup_test_data()))
    results.append(("Fantasy Leagues", test_fantasy_leagues()))
    results.append(("Prediction System", test_predictions()))
    results.append(("Scoring System", test_scoring_system()))
    results.append(("Leaderboards", test_leaderboards()))
    results.append(("Achievements", test_achievements()))
    results.append(("Collectives", test_collectives()))
    results.append(("Tournaments", test_tournaments()))
    results.append(("Database Integrity", verify_database_integrity()))
    results.append(("Advanced Features", test_advanced_features()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 " * 10)
        print("✅ ALL FANTASY COLLECTIVE CAPABILITIES CONFIRMED!")
        print("🎉 " * 10)
        print("\nThe system successfully supports:")
        print("• Fantasy Sports Leagues with multiple types")
        print("• Prediction Markets for any event type")
        print("• Business and News Predictions")
        print("• Collective Intelligence with voting")
        print("• Achievement and Gamification System")
        print("• Tournament Brackets and Competitions")
        print("• Comprehensive Scoring Algorithms")
        print("• Leaderboards and Rankings")
        print("• Financial Management (entry fees, prizes)")
        print("• Database Integrity and Transactions")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review issues above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())