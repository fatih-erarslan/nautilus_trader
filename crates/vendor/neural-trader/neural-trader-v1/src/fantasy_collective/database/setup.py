#!/usr/bin/env python3
"""
Fantasy Collective Database Setup Script

This script initializes the database, runs migrations, and can populate it with test data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fantasy_collective.database.connection import DatabaseConnection, init_db
from fantasy_collective.database.migrations import MigrationManager, create_initial_migration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database_setup.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_database(db_path: str = None, fresh_install: bool = False):
    """Set up the database with schema and initial data."""
    logger.info("Starting database setup...")
    
    try:
        # Initialize database connection
        if db_path:
            db = init_db(db_path)
        else:
            db = init_db()
        
        logger.info(f"Using database: {db.db_path}")
        
        # If fresh install, remove existing database
        if fresh_install and os.path.exists(db.db_path):
            logger.info("Removing existing database for fresh install")
            os.remove(db.db_path)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db.db_path), exist_ok=True)
        
        # Create migration manager
        migration_manager = MigrationManager(db)
        
        # Create migrations directory if it doesn't exist
        migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Create initial migration if it doesn't exist
        migration_files = [f for f in os.listdir(migrations_dir) if f.endswith('.sql')]
        if not migration_files:
            logger.info("Creating initial migration...")
            initial_migration_file = create_initial_migration(migrations_dir)
            logger.info(f"Created initial migration: {initial_migration_file}")
        
        # Load migrations from directory
        migration_manager.load_migrations_from_directory(migrations_dir)
        
        # Run migrations
        logger.info("Running database migrations...")
        applied = migration_manager.migrate()
        
        if applied:
            logger.info(f"Applied {len(applied)} migrations: {', '.join(applied)}")
        else:
            logger.info("Database is up to date")
        
        # Validate database
        logger.info("Validating database...")
        health = db.health_check()
        
        if health['status'] == 'healthy':
            logger.info("Database setup completed successfully")
            logger.info(f"Database size: {health.get('database_size_mb', 0)} MB")
            return True
        else:
            logger.error(f"Database validation failed: {health}")
            return False
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


def populate_test_data(db: DatabaseConnection):
    """Populate database with test data for development."""
    logger.info("Populating database with test data...")
    
    try:
        from fantasy_collective.database.models import UserRepository, LeagueRepository, PredictionRepository
        from decimal import Decimal
        from datetime import datetime, timedelta
        import random
        import hashlib
        import secrets
        
        # Create repositories
        user_repo = UserRepository(db)
        league_repo = LeagueRepository(db)
        prediction_repo = PredictionRepository(db)
        
        # Create test users
        test_users = [
            {
                'username': 'testuser1',
                'email': 'test1@example.com',
                'display_name': 'Test User 1',
                'password_hash': hashlib.sha256('password123'.encode()).hexdigest(),
                'salt': secrets.token_hex(16),
                'account_status': 'active',
                'email_verified': True,
                'referral_code': 'REF001'
            },
            {
                'username': 'testuser2',
                'email': 'test2@example.com',
                'display_name': 'Test User 2',
                'password_hash': hashlib.sha256('password123'.encode()).hexdigest(),
                'salt': secrets.token_hex(16),
                'account_status': 'active',
                'email_verified': True,
                'referral_code': 'REF002'
            },
            {
                'username': 'testuser3',
                'email': 'test3@example.com',
                'display_name': 'Test User 3',
                'password_hash': hashlib.sha256('password123'.encode()).hexdigest(),
                'salt': secrets.token_hex(16),
                'account_status': 'active',
                'email_verified': True,
                'referral_code': 'REF003'
            }
        ]
        
        user_ids = []
        for user_data in test_users:
            # Check if user already exists
            existing = user_repo.find_by_username(user_data['username'])
            if not existing:
                user_id = user_repo.create(user_data)
                user_ids.append(user_id)
                logger.info(f"Created test user: {user_data['username']} (ID: {user_id})")
            else:
                user_ids.append(existing['user_id'])
                logger.info(f"Test user already exists: {user_data['username']}")
        
        # Create test leagues
        test_leagues = [
            {
                'league_name': 'Test NFL League',
                'league_type': 'fantasy_sports',
                'category': 'NFL Football',
                'max_participants': 12,
                'entry_fee': Decimal('25.00'),
                'prize_pool': Decimal('300.00'),
                'scoring_system': {'touchdown': 6, 'field_goal': 3, 'safety': 2},
                'league_rules': {'prediction_deadline': 'game_start', 'late_predictions': False},
                'status': 'active',
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(days=120),
                'creator_id': user_ids[0],
                'is_public': True,
                'invite_code': 'NFL001'
            },
            {
                'league_name': 'Stock Prediction Challenge',
                'league_type': 'prediction_market',
                'category': 'Stock Market',
                'max_participants': 20,
                'entry_fee': Decimal('50.00'),
                'prize_pool': Decimal('1000.00'),
                'scoring_system': {'correct_direction': 10, 'exact_range': 25},
                'league_rules': {'minimum_confidence': 0.1, 'max_predictions_per_stock': 5},
                'status': 'active',
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(days=90),
                'creator_id': user_ids[1],
                'is_public': True,
                'invite_code': 'STOCK001'
            }
        ]
        
        league_ids = []
        for league_data in test_leagues:
            # Check if league already exists
            existing = league_repo.find_by_invite_code(league_data['invite_code'])
            if not existing:
                league_id = league_repo.create(league_data)
                league_ids.append(league_id)
                logger.info(f"Created test league: {league_data['league_name']} (ID: {league_id})")
            else:
                league_ids.append(existing['league_id'])
                logger.info(f"Test league already exists: {league_data['league_name']}")
        
        # Create league participants
        for league_id in league_ids:
            for user_id in user_ids:
                # Check if participation already exists
                existing = db.execute_single(
                    "SELECT * FROM league_participants WHERE league_id = ? AND user_id = ?",
                    (league_id, user_id)
                )
                
                if not existing:
                    participant_data = {
                        'league_id': league_id,
                        'user_id': user_id,
                        'team_name': f'Team User {user_id}',
                        'status': 'active',
                        'entry_fee_paid': True,
                        'total_points': Decimal(str(random.randint(50, 200))),
                        'current_rank': random.randint(1, 3),
                        'prediction_accuracy': Decimal(str(round(random.uniform(0.3, 0.8), 4)))
                    }
                    
                    fields = list(participant_data.keys())
                    placeholders = ['?' for _ in fields]
                    values = list(participant_data.values())
                    
                    query = f"""
                    INSERT INTO league_participants ({', '.join(fields)})
                    VALUES ({', '.join(placeholders)})
                    """
                    
                    db.execute_modify(query, tuple(values))
                    logger.info(f"Added user {user_id} to league {league_id}")
        
        # Create test event categories and events
        categories = [
            {'category_name': 'NFL Football', 'category_type': 'sports', 'description': 'NFL games and events'},
            {'category_name': 'Stock Market', 'category_type': 'financial', 'description': 'Stock movements'}
        ]
        
        category_ids = []
        for category in categories:
            existing = db.execute_single(
                "SELECT * FROM event_categories WHERE category_name = ?",
                (category['category_name'],)
            )
            
            if not existing:
                fields = list(category.keys())
                placeholders = ['?' for _ in fields]
                values = list(category.values())
                
                query = f"""
                INSERT INTO event_categories ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                """
                
                db.execute_modify(query, tuple(values))
                category_id = db.get_last_insert_id()
                category_ids.append(category_id)
                logger.info(f"Created event category: {category['category_name']}")
            else:
                category_ids.append(existing['category_id'])
        
        # Create test events
        test_events = [
            {
                'category_id': category_ids[0],
                'event_name': 'Chiefs vs Bills',
                'event_description': 'NFL Regular Season Game',
                'event_type': 'game',
                'participants': ['Kansas City Chiefs', 'Buffalo Bills'],
                'scheduled_start': datetime.now() + timedelta(days=7),
                'prediction_deadline': datetime.now() + timedelta(days=6),
                'status': 'upcoming'
            },
            {
                'category_id': category_ids[1],
                'event_name': 'AAPL Q4 Earnings',
                'event_description': 'Apple Q4 Earnings Report',
                'event_type': 'earnings',
                'participants': ['Apple Inc.'],
                'scheduled_start': datetime.now() + timedelta(days=14),
                'prediction_deadline': datetime.now() + timedelta(days=13),
                'status': 'upcoming'
            }
        ]
        
        event_ids = []
        for event_data in test_events:
            # Serialize JSON fields
            event_data['participants'] = str(event_data['participants'])
            
            fields = list(event_data.keys())
            placeholders = ['?' for _ in fields]
            values = list(event_data.values())
            
            query = f"""
            INSERT INTO events ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            """
            
            db.execute_modify(query, tuple(values))
            event_id = db.get_last_insert_id()
            event_ids.append(event_id)
            logger.info(f"Created test event: {event_data['event_name']}")
        
        # Create prediction markets
        test_markets = [
            {
                'event_id': event_ids[0],
                'league_id': league_ids[0],
                'market_name': 'Game Winner',
                'market_type': 'binary',
                'question': 'Who will win the game?',
                'options': ['Kansas City Chiefs', 'Buffalo Bills'],
                'status': 'open',
                'minimum_bet': Decimal('1.00'),
                'maximum_bet': Decimal('100.00')
            },
            {
                'event_id': event_ids[1],
                'league_id': league_ids[1],
                'market_name': 'Earnings Beat',
                'market_type': 'binary',
                'question': 'Will Apple beat earnings expectations?',
                'options': ['Yes', 'No'],
                'status': 'open',
                'minimum_bet': Decimal('5.00'),
                'maximum_bet': Decimal('500.00')
            }
        ]
        
        market_ids = []
        for market_data in test_markets:
            # Serialize JSON fields
            market_data['options'] = str(market_data['options'])
            
            fields = list(market_data.keys())
            placeholders = ['?' for _ in fields]
            values = list(market_data.values())
            
            query = f"""
            INSERT INTO prediction_markets ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            """
            
            db.execute_modify(query, tuple(values))
            market_id = db.get_last_insert_id()
            market_ids.append(market_id)
            logger.info(f"Created prediction market: {market_data['market_name']}")
        
        # Create some test predictions
        for i, user_id in enumerate(user_ids[:2]):  # Only first 2 users
            for j, market_id in enumerate(market_ids):
                prediction_data = {
                    'user_id': user_id,
                    'market_id': market_id,
                    'league_id': league_ids[j],
                    'predicted_outcome': ['Kansas City Chiefs', 'Yes'][j],
                    'confidence_level': Decimal(str(round(random.uniform(0.5, 0.9), 4))),
                    'stake_amount': Decimal(str(random.randint(10, 50))),
                    'odds_when_placed': Decimal(str(round(random.uniform(1.5, 3.0), 4))),
                    'bet_type': 'straight',
                    'status': 'active'
                }
                
                fields = list(prediction_data.keys())
                placeholders = ['?' for _ in fields]
                values = list(prediction_data.values())
                
                query = f"""
                INSERT INTO predictions ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                """
                
                db.execute_modify(query, tuple(values))
                logger.info(f"Created prediction for user {user_id} on market {market_id}")
        
        # Create user wallets
        for user_id in user_ids:
            existing = db.execute_single(
                "SELECT * FROM user_wallets WHERE user_id = ? AND wallet_type = 'main'",
                (user_id,)
            )
            
            if not existing:
                wallet_data = {
                    'user_id': user_id,
                    'wallet_type': 'main',
                    'currency': 'USD',
                    'current_balance': Decimal(str(random.randint(100, 1000))),
                    'available_balance': Decimal(str(random.randint(50, 500))),
                    'is_active': True
                }
                
                fields = list(wallet_data.keys())
                placeholders = ['?' for _ in fields]
                values = list(wallet_data.values())
                
                query = f"""
                INSERT INTO user_wallets ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                """
                
                db.execute_modify(query, tuple(values))
                logger.info(f"Created wallet for user {user_id}")
        
        logger.info("Test data population completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate test data: {e}")
        raise


def verify_setup(db: DatabaseConnection):
    """Verify database setup by running basic queries."""
    logger.info("Verifying database setup...")
    
    try:
        # Check tables exist
        tables = [
            'users', 'leagues', 'league_participants', 'events', 
            'prediction_markets', 'predictions', 'user_wallets'
        ]
        
        for table in tables:
            if not db.table_exists(table):
                logger.error(f"Table '{table}' does not exist")
                return False
        
        # Check basic counts
        counts = {}
        for table in tables:
            result = db.execute_single(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = result['count'] if result else 0
            logger.info(f"{table}: {counts[table]} records")
        
        # Test relationships
        if counts['league_participants'] > 0 and counts['predictions'] > 0:
            # Test user performance query
            from fantasy_collective.database.models import PredictionRepository
            
            prediction_repo = PredictionRepository(db)
            
            # Get first user's performance
            first_user = db.execute_single("SELECT user_id FROM users LIMIT 1")
            if first_user:
                performance = prediction_repo.get_user_performance(first_user['user_id'])
                logger.info(f"User performance test: {performance}")
        
        logger.info("Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def main():
    """Main entry point for database setup."""
    parser = argparse.ArgumentParser(description='Fantasy Collective Database Setup')
    parser.add_argument('--db-path', help='Database file path')
    parser.add_argument('--fresh', action='store_true', help='Fresh install (removes existing database)')
    parser.add_argument('--test-data', action='store_true', help='Populate with test data')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing setup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.verify_only:
            # Only verify existing database
            if args.db_path:
                db = init_db(args.db_path)
            else:
                db = init_db()
            
            success = verify_setup(db)
            sys.exit(0 if success else 1)
        
        # Setup database
        success = setup_database(args.db_path, args.fresh)
        if not success:
            logger.error("Database setup failed")
            sys.exit(1)
        
        # Get database connection
        if args.db_path:
            db = init_db(args.db_path)
        else:
            db = init_db()
        
        # Populate test data if requested
        if args.test_data:
            populate_test_data(db)
        
        # Verify setup
        success = verify_setup(db)
        if not success:
            logger.error("Database verification failed")
            sys.exit(1)
        
        logger.info("Database setup completed successfully!")
        
        # Print summary
        size_info = db.get_database_size()
        table_sizes = db.get_table_sizes()
        
        print("\n" + "="*50)
        print("DATABASE SETUP SUMMARY")
        print("="*50)
        print(f"Database Path: {db.db_path}")
        print(f"Database Size: {size_info.get('total_mb', 0)} MB")
        print(f"Tables Created: {len(table_sizes)}")
        print("\nTable Summary:")
        for table_name, info in table_sizes.items():
            print(f"  {table_name}: {info['row_count']} records")
        print("="*50)
        
        # Show health check
        health = db.health_check()
        print(f"Health Status: {health['status'].upper()}")
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()