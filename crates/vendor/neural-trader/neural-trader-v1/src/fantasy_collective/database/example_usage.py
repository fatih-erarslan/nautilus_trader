"""Example Usage of Fantasy Collective Database Manager

This script demonstrates how to use the Fantasy Collective Database Manager
for common operations like creating users, managing leagues, handling predictions,
and processing transactions.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from manager import (
    FantasyCollectiveDBManager,
    init_database,
    get_db_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_user_operations(db: FantasyCollectiveDBManager):
    """Demonstrate user CRUD operations"""
    logger.info("=== User Operations Demo ===")
    
    # Create users
    users = []
    user_data = [
        ('alice_trader', 'alice@example.com', 'Alice Johnson', 'admin'),
        ('bob_analyst', 'bob@example.com', 'Bob Smith', 'member'),
        ('charlie_predictor', 'charlie@example.com', 'Charlie Brown', 'member'),
        ('diana_expert', 'diana@example.com', 'Diana Prince', 'moderator')
    ]
    
    for username, email, full_name, role in user_data:
        user = db.create_user(
            username=username,
            email=email,
            password_hash=f'hashed_password_for_{username}',
            full_name=full_name,
            role=role
        )
        users.append(user)
        logger.info(f"Created user: {user}")
    
    # Add initial balances
    for user in users:
        if user.role == 'admin':
            db.create_transaction(user.id, 'deposit', 1000.0, 'Admin initial balance')
        else:
            db.create_transaction(user.id, 'deposit', 500.0, 'Member initial balance')
    
    # Demonstrate user lookup
    alice = db.get_user_by_username('alice_trader')
    logger.info(f"Found user by username: {alice}")
    
    bob = db.get_user_by_email('bob@example.com')
    logger.info(f"Found user by email: {bob}")
    
    # Update user information
    updated_user = db.update_user(bob.id, total_points=150)
    logger.info(f"Updated user points: {updated_user.total_points}")
    
    return users

def demonstrate_league_operations(db: FantasyCollectiveDBManager, users):
    """Demonstrate league management"""
    logger.info("=== League Operations Demo ===")
    
    admin_user = users[0]  # Alice is admin
    
    # Create leagues with different configurations
    leagues = []
    league_configs = [
        {
            'name': 'Premier Prediction League',
            'entry_fee': 50.0,
            'max_members': 20,
            'description': 'Elite prediction league for experienced members'
        },
        {
            'name': 'Rookie League',
            'entry_fee': 10.0,
            'max_members': 100,
            'description': 'Beginner-friendly league for new members'
        },
        {
            'name': 'Championship Series',
            'entry_fee': 100.0,
            'max_members': 10,
            'description': 'High-stakes championship tournament'
        }
    ]
    
    for config in league_configs:
        league = db.create_league(
            name=config['name'],
            created_by=admin_user.id,
            entry_fee=config['entry_fee'],
            max_members=config['max_members'],
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=30),
            description=config['description']
        )
        leagues.append(league)
        logger.info(f"Created league: {league}")
    
    # Members join leagues
    for i, user in enumerate(users[1:], 1):  # Skip admin user
        # Join different leagues based on user
        if i == 1:  # Bob joins all leagues
            for league in leagues:
                success = db.join_league(user.id, league.id)
                logger.info(f"User {user.username} joined {league.name}: {success}")
        elif i == 2:  # Charlie joins rookie and premier
            for league in leagues[:2]:
                success = db.join_league(user.id, league.id)
                logger.info(f"User {user.username} joined {league.name}: {success}")
        else:  # Diana joins only championship
            success = db.join_league(user.id, leagues[2].id)
            logger.info(f"User {user.username} joined {leagues[2].name}: {success}")
    
    return leagues

def demonstrate_prediction_operations(db: FantasyCollectiveDBManager, users, leagues):
    """Demonstrate prediction management"""
    logger.info("=== Prediction Operations Demo ===")
    
    # Sample prediction data for different events
    prediction_scenarios = [
        {
            'event_name': 'NFL Super Bowl 2024',
            'event_date': datetime.utcnow() + timedelta(days=7),
            'predictions': [
                {'user_idx': 1, 'data': {'winner': 'Team A', 'score': '28-21', 'mvp': 'Player X'}, 'confidence': 0.8},
                {'user_idx': 2, 'data': {'winner': 'Team B', 'score': '24-17', 'mvp': 'Player Y'}, 'confidence': 0.7},
                {'user_idx': 3, 'data': {'winner': 'Team A', 'score': '31-14', 'mvp': 'Player X'}, 'confidence': 0.9}
            ]
        },
        {
            'event_name': 'NBA Finals Game 7',
            'event_date': datetime.utcnow() + timedelta(days=14),
            'predictions': [
                {'user_idx': 1, 'data': {'winner': 'Lakers', 'total_points': 215, 'overtime': False}, 'confidence': 0.6},
                {'user_idx': 2, 'data': {'winner': 'Celtics', 'total_points': 208, 'overtime': True}, 'confidence': 0.8}
            ]
        },
        {
            'event_name': 'World Cup Final',
            'event_date': datetime.utcnow() + timedelta(days=21),
            'predictions': [
                {'user_idx': 1, 'data': {'winner': 'Brazil', 'score': '2-1', 'goals_scored': 3}, 'confidence': 0.75},
                {'user_idx': 2, 'data': {'winner': 'Argentina', 'score': '3-2', 'goals_scored': 5}, 'confidence': 0.85},
                {'user_idx': 3, 'data': {'winner': 'Brazil', 'score': '1-0', 'goals_scored': 1}, 'confidence': 0.65}
            ]
        }
    ]
    
    predictions = []
    for scenario in prediction_scenarios:
        for pred_data in scenario['predictions']:
            user = users[pred_data['user_idx']]
            # Use first league the user is in
            user_leagues = [league for league in leagues 
                          if any(m.user_id == user.id for m in league.memberships)]
            if user_leagues:
                prediction = db.create_prediction(
                    user_id=user.id,
                    league_id=user_leagues[0].id,
                    event_name=scenario['event_name'],
                    event_date=scenario['event_date'],
                    prediction_data=pred_data['data'],
                    confidence_level=pred_data['confidence']
                )
                predictions.append(prediction)
                logger.info(f"Created prediction: {prediction}")
    
    # Resolve some predictions and award points
    resolution_results = [
        {'prediction_idx': 0, 'points': 100, 'reason': 'Correct winner and close score'},
        {'prediction_idx': 1, 'points': 25, 'reason': 'Wrong winner but good analysis'},
        {'prediction_idx': 2, 'points': 150, 'reason': 'Perfect prediction'},
        {'prediction_idx': 3, 'points': 75, 'reason': 'Correct winner'},
        {'prediction_idx': 4, 'points': 50, 'reason': 'Partial credit for overtime call'}
    ]
    
    for result in resolution_results:
        if result['prediction_idx'] < len(predictions):
            prediction = predictions[result['prediction_idx']]
            success = db.resolve_prediction(prediction.id, result['points'])
            logger.info(f"Resolved prediction {prediction.id}: {result['points']} points - {result['reason']}")
    
    return predictions

def demonstrate_ranking_and_leaderboards(db: FantasyCollectiveDBManager, leagues):
    """Demonstrate ranking calculations"""
    logger.info("=== Ranking and Leaderboard Demo ===")
    
    for league in leagues:
        logger.info(f"\\nLeague: {league.name}")
        
        # Calculate rankings
        rankings = db.calculate_league_rankings(league.id)
        for rank_data in rankings:
            logger.info(f"  Rank {rank_data['rank']}: {rank_data['username']} - {rank_data['points']} points")
        
        # Get leaderboard
        leaderboard = db.get_leaderboard(league.id, limit=5)
        logger.info(f"  Top 5 Leaderboard:")
        for entry in leaderboard:
            logger.info(f"    {entry['rank']}. {entry['username']}: {entry['points']} points")

def demonstrate_transaction_management(db: FantasyCollectiveDBManager, users):
    """Demonstrate financial transactions"""
    logger.info("=== Transaction Management Demo ===")
    
    # Award rewards to top performers
    reward_data = [
        {'user_idx': 1, 'amount': 200.0, 'description': 'Weekly champion bonus'},
        {'user_idx': 2, 'amount': 100.0, 'description': 'Top predictor reward'},
        {'user_idx': 3, 'amount': 50.0, 'description': 'Participation bonus'}
    ]
    
    for reward in reward_data:
        user = users[reward['user_idx']]
        transaction = db.create_transaction(
            user_id=user.id,
            transaction_type='reward',
            amount=reward['amount'],
            description=reward['description']
        )
        logger.info(f"Awarded reward: {transaction}")
    
    # Simulate some penalties for rule violations
    penalty_data = [
        {'user_idx': 1, 'amount': 25.0, 'description': 'Late prediction penalty'},
        {'user_idx': 2, 'amount': 10.0, 'description': 'Minor rule violation'}
    ]
    
    for penalty in penalty_data:
        user = users[penalty['user_idx']]
        try:
            transaction = db.create_transaction(
                user_id=user.id,
                transaction_type='penalty',
                amount=penalty['amount'],
                description=penalty['description']
            )
            logger.info(f"Applied penalty: {transaction}")
        except ValueError as e:
            logger.warning(f"Penalty failed for user {user.username}: {e}")
    
    # Show transaction history for each user
    for user in users[:3]:  # First 3 users
        transactions = db.get_user_transactions(user.id, limit=5)
        logger.info(f"\\nTransaction history for {user.username}:")
        for txn in transactions:
            logger.info(f"  {txn.created_at.strftime('%Y-%m-%d %H:%M')} - "
                       f"{txn.transaction_type.upper()}: ${txn.amount:.2f} - {txn.description}")
        
        # Show current balance
        current_user = db.get_user_by_id(user.id)
        logger.info(f"  Current balance: ${current_user.balance:.2f}")

def demonstrate_analytics_and_performance(db: FantasyCollectiveDBManager):
    """Demonstrate analytics and performance features"""
    logger.info("=== Analytics and Performance Demo ===")
    
    # Get comprehensive database statistics
    stats = db.get_database_stats()
    logger.info("Database Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Demonstrate caching by calling cached methods multiple times
    logger.info("\\nTesting query caching:")
    import time
    
    start_time = time.time()
    active_users_1 = db.get_active_users()
    time_1 = time.time() - start_time
    logger.info(f"First call to get_active_users: {time_1:.4f}s ({len(active_users_1)} users)")
    
    start_time = time.time()
    active_users_2 = db.get_active_users()
    time_2 = time.time() - start_time
    logger.info(f"Second call to get_active_users: {time_2:.4f}s (cached)")
    
    if time_2 < time_1:
        logger.info("✓ Caching is working effectively!")
    
    # Test database optimization
    logger.info("\\nRunning database optimization...")
    db.optimize_database()
    logger.info("✓ Database optimization completed")

def main():
    """Main demonstration function"""
    logger.info("Starting Fantasy Collective Database Manager Demo")
    
    # Initialize database
    db_path = Path(__file__).parent / 'demo_fantasy_collective.db'
    
    # Remove existing demo database
    if db_path.exists():
        db_path.unlink()
    
    # Initialize fresh database
    db = init_database(str(db_path))
    
    try:
        # Run all demonstrations
        users = demonstrate_user_operations(db)
        leagues = demonstrate_league_operations(db, users)
        predictions = demonstrate_prediction_operations(db, users, leagues)
        demonstrate_ranking_and_leaderboards(db, leagues)
        demonstrate_transaction_management(db, users)
        demonstrate_analytics_and_performance(db)
        
        logger.info("\\n=== Demo Summary ===")
        final_stats = db.get_database_stats()
        logger.info(f"Final Statistics:")
        logger.info(f"  Total Users: {final_stats['total_users']}")
        logger.info(f"  Active Users: {final_stats['active_users']}")
        logger.info(f"  Total Leagues: {final_stats['total_leagues']}")
        logger.info(f"  Total Predictions: {final_stats['total_predictions']}")
        logger.info(f"  Total Transactions: {final_stats['total_transactions']}")
        logger.info(f"  Total Volume: ${final_stats['total_volume']:.2f}")
        
        logger.info("\\n✓ Fantasy Collective Database Manager Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise
    finally:
        # Clean up
        db.close()
        logger.info(f"Demo database saved to: {db_path}")

if __name__ == '__main__':
    main()