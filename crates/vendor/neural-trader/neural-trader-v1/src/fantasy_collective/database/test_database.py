#!/usr/bin/env python3
"""
Fantasy Collective Database Test Suite

Comprehensive tests for database functionality, models, and performance.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fantasy_collective.database.connection import DatabaseConnection, init_db
from fantasy_collective.database.migrations import MigrationManager, create_initial_migration
from fantasy_collective.database.models import UserRepository, LeagueRepository, PredictionRepository


class TestDatabaseConnection(unittest.TestCase):
    """Test database connection and basic operations."""
    
    def setUp(self):
        """Set up test database."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
    
    def tearDown(self):
        """Clean up test database."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_connection_creation(self):
        """Test database connection creation."""
        self.assertIsNotNone(self.db)
        self.assertEqual(self.db.db_path, self.db_path)
        self.assertTrue(os.path.exists(self.db_path))
    
    def test_basic_operations(self):
        """Test basic database operations."""
        # Create a simple table
        self.db.execute_modify("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                value INTEGER
            )
        """)
        
        # Insert data
        self.db.execute_modify(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ('test1', 100)
        )
        
        # Query data
        result = self.db.execute_single(
            "SELECT * FROM test_table WHERE name = ?",
            ('test1',)
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'test1')
        self.assertEqual(result['value'], 100)
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        # Create test table
        self.db.execute_modify("""
            CREATE TABLE test_rollback (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) UNIQUE
            )
        """)
        
        # Insert initial data
        self.db.execute_modify(
            "INSERT INTO test_rollback (name) VALUES (?)",
            ('initial',)
        )
        
        # Test rollback on constraint violation
        try:
            with self.db.transaction():
                self.db.execute_modify(
                    "INSERT INTO test_rollback (name) VALUES (?)",
                    ('valid',)
                )
                # This should cause a constraint violation
                self.db.execute_modify(
                    "INSERT INTO test_rollback (name) VALUES (?)",
                    ('initial',)  # Duplicate
                )
        except Exception:
            pass  # Expected to fail
        
        # Verify rollback - should only have initial record
        count = self.db.execute_single(
            "SELECT COUNT(*) as count FROM test_rollback"
        )
        self.assertEqual(count['count'], 1)
    
    def test_health_check(self):
        """Test database health check."""
        health = self.db.health_check()
        
        self.assertIn('status', health)
        self.assertEqual(health['status'], 'healthy')
        self.assertTrue(health['connectivity'])
        self.assertTrue(health['foreign_keys_ok'])


class TestMigrations(unittest.TestCase):
    """Test database migration system."""
    
    def setUp(self):
        """Set up test database and migration manager."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
        self.migration_manager = MigrationManager(self.db)
        self.migrations_dir = os.path.join(self.test_dir, 'migrations')
        os.makedirs(self.migrations_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_initial_migration_creation(self):
        """Test creation of initial migration."""
        migration_file = create_initial_migration(self.migrations_dir)
        
        self.assertTrue(os.path.exists(migration_file))
        
        # Load and apply migrations
        self.migration_manager.load_migrations_from_directory(self.migrations_dir)
        applied = self.migration_manager.migrate()
        
        self.assertGreater(len(applied), 0)
        
        # Verify tables were created
        self.assertTrue(self.db.table_exists('users'))
        self.assertTrue(self.db.table_exists('leagues'))
        self.assertTrue(self.db.table_exists('predictions'))
    
    def test_migration_status(self):
        """Test migration status reporting."""
        create_initial_migration(self.migrations_dir)
        self.migration_manager.load_migrations_from_directory(self.migrations_dir)
        
        # Before migration
        status = self.migration_manager.status()
        self.assertEqual(status['applied_count'], 0)
        self.assertGreater(status['pending_count'], 0)
        
        # After migration
        self.migration_manager.migrate()
        status = self.migration_manager.status()
        self.assertGreater(status['applied_count'], 0)
        self.assertEqual(status['pending_count'], 0)


class TestUserRepository(unittest.TestCase):
    """Test User repository operations."""
    
    def setUp(self):
        """Set up test database with schema."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
        
        # Apply schema
        migration_manager = MigrationManager(self.db)
        migrations_dir = os.path.join(self.test_dir, 'migrations')
        os.makedirs(migrations_dir)
        create_initial_migration(migrations_dir)
        migration_manager.load_migrations_from_directory(migrations_dir)
        migration_manager.migrate()
        
        self.user_repo = UserRepository(self.db)
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_create_user(self):
        """Test user creation."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password_hash': 'hashed_password',
            'salt': 'salt123',
            'display_name': 'Test User',
            'account_status': 'active'
        }
        
        user_id = self.user_repo.create(user_data)
        self.assertIsNotNone(user_id)
        self.assertGreater(user_id, 0)
        
        # Verify user was created
        user = self.user_repo.find_by_id(user_id)
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], 'testuser')
        self.assertEqual(user['email'], 'test@example.com')
    
    def test_find_by_username(self):
        """Test finding user by username."""
        user_data = {
            'username': 'findme',
            'email': 'findme@example.com',
            'password_hash': 'hash',
            'salt': 'salt'
        }
        
        user_id = self.user_repo.create(user_data)
        user = self.user_repo.find_by_username('findme')
        
        self.assertIsNotNone(user)
        self.assertEqual(user['user_id'], user_id)
        self.assertEqual(user['username'], 'findme')
    
    def test_find_by_email(self):
        """Test finding user by email."""
        user_data = {
            'username': 'emailtest',
            'email': 'emailtest@example.com',
            'password_hash': 'hash',
            'salt': 'salt'
        }
        
        user_id = self.user_repo.create(user_data)
        user = self.user_repo.find_by_email('emailtest@example.com')
        
        self.assertIsNotNone(user)
        self.assertEqual(user['user_id'], user_id)
        self.assertEqual(user['email'], 'emailtest@example.com')
    
    def test_update_user(self):
        """Test user update."""
        user_data = {
            'username': 'updateme',
            'email': 'updateme@example.com',
            'password_hash': 'hash',
            'salt': 'salt',
            'display_name': 'Original Name'
        }
        
        user_id = self.user_repo.create(user_data)
        
        # Update user
        updated = self.user_repo.update(user_id, {
            'display_name': 'Updated Name',
            'bio': 'New bio'
        })
        
        self.assertEqual(updated, 1)  # One row affected
        
        # Verify update
        user = self.user_repo.find_by_id(user_id)
        self.assertEqual(user['display_name'], 'Updated Name')
        self.assertEqual(user['bio'], 'New bio')


class TestLeagueRepository(unittest.TestCase):
    """Test League repository operations."""
    
    def setUp(self):
        """Set up test database with schema."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
        
        # Apply schema
        migration_manager = MigrationManager(self.db)
        migrations_dir = os.path.join(self.test_dir, 'migrations')
        os.makedirs(migrations_dir)
        create_initial_migration(migrations_dir)
        migration_manager.load_migrations_from_directory(migrations_dir)
        migration_manager.migrate()
        
        self.user_repo = UserRepository(self.db)
        self.league_repo = LeagueRepository(self.db)
        
        # Create test user
        self.user_id = self.user_repo.create({
            'username': 'creator',
            'email': 'creator@example.com',
            'password_hash': 'hash',
            'salt': 'salt'
        })
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_create_league(self):
        """Test league creation."""
        league_data = {
            'league_name': 'Test League',
            'league_type': 'fantasy_sports',
            'category': 'NFL',
            'creator_id': self.user_id,
            'max_participants': 10,
            'entry_fee': Decimal('25.00'),
            'scoring_system': {'touchdown': 6, 'field_goal': 3},
            'league_rules': {'deadline': 'game_start'},
            'invite_code': 'TEST123'
        }
        
        league_id = self.league_repo.create(league_data)
        self.assertIsNotNone(league_id)
        self.assertGreater(league_id, 0)
        
        # Verify league was created
        league = self.league_repo.find_by_id(league_id)
        self.assertIsNotNone(league)
        self.assertEqual(league['league_name'], 'Test League')
        self.assertEqual(league['creator_id'], self.user_id)
    
    def test_find_by_invite_code(self):
        """Test finding league by invite code."""
        league_data = {
            'league_name': 'Invite Test League',
            'league_type': 'fantasy_sports',
            'creator_id': self.user_id,
            'scoring_system': {},
            'league_rules': {},
            'invite_code': 'INVITE123'
        }
        
        league_id = self.league_repo.create(league_data)
        league = self.league_repo.find_by_invite_code('INVITE123')
        
        self.assertIsNotNone(league)
        self.assertEqual(league['league_id'], league_id)
        self.assertEqual(league['invite_code'], 'INVITE123')
    
    def test_find_public_leagues(self):
        """Test finding public leagues."""
        # Create public league
        public_league = {
            'league_name': 'Public League',
            'league_type': 'fantasy_sports',
            'creator_id': self.user_id,
            'scoring_system': {},
            'league_rules': {},
            'is_public': True,
            'status': 'active'
        }
        
        # Create private league
        private_league = {
            'league_name': 'Private League',
            'league_type': 'fantasy_sports',
            'creator_id': self.user_id,
            'scoring_system': {},
            'league_rules': {},
            'is_public': False,
            'status': 'active'
        }
        
        self.league_repo.create(public_league)
        self.league_repo.create(private_league)
        
        public_leagues = self.league_repo.find_public_leagues()
        
        self.assertEqual(len(public_leagues), 1)
        self.assertEqual(public_leagues[0]['league_name'], 'Public League')
        self.assertTrue(public_leagues[0]['is_public'])


class TestPredictionRepository(unittest.TestCase):
    """Test Prediction repository operations."""
    
    def setUp(self):
        """Set up test database with schema and test data."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
        
        # Apply schema
        migration_manager = MigrationManager(self.db)
        migrations_dir = os.path.join(self.test_dir, 'migrations')
        os.makedirs(migrations_dir)
        create_initial_migration(migrations_dir)
        migration_manager.load_migrations_from_directory(migrations_dir)
        migration_manager.migrate()
        
        # Create repositories
        self.user_repo = UserRepository(self.db)
        self.league_repo = LeagueRepository(self.db)
        self.prediction_repo = PredictionRepository(self.db)
        
        # Create test user and league
        self.user_id = self.user_repo.create({
            'username': 'predictor',
            'email': 'predictor@example.com',
            'password_hash': 'hash',
            'salt': 'salt'
        })
        
        self.league_id = self.league_repo.create({
            'league_name': 'Prediction League',
            'league_type': 'fantasy_sports',
            'creator_id': self.user_id,
            'scoring_system': {},
            'league_rules': {}
        })
        
        # Create event category and event
        self.db.execute_modify("""
            INSERT INTO event_categories (category_name, category_type, description)
            VALUES ('Test Category', 'sports', 'Test category')
        """)
        category_id = self.db.get_last_insert_id()
        
        self.db.execute_modify("""
            INSERT INTO events (category_id, event_name, scheduled_start, prediction_deadline, status)
            VALUES (?, 'Test Event', datetime('now', '+1 day'), datetime('now', '+1 hour'), 'upcoming')
        """, (category_id,))
        event_id = self.db.get_last_insert_id()
        
        # Create prediction market
        self.db.execute_modify("""
            INSERT INTO prediction_markets (event_id, league_id, market_name, market_type, question, options, status)
            VALUES (?, ?, 'Test Market', 'binary', 'Who wins?', '["Team A", "Team B"]', 'open')
        """, (event_id, self.league_id))
        self.market_id = self.db.get_last_insert_id()
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_create_prediction(self):
        """Test prediction creation."""
        prediction_data = {
            'user_id': self.user_id,
            'market_id': self.market_id,
            'league_id': self.league_id,
            'predicted_outcome': 'Team A',
            'confidence_level': Decimal('0.7500'),
            'stake_amount': Decimal('25.00'),
            'odds_when_placed': Decimal('2.0000'),
            'status': 'active'
        }
        
        prediction_id = self.prediction_repo.create(prediction_data)
        self.assertIsNotNone(prediction_id)
        self.assertGreater(prediction_id, 0)
        
        # Verify prediction was created
        prediction = self.prediction_repo.find_by_id(prediction_id)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction['user_id'], self.user_id)
        self.assertEqual(prediction['predicted_outcome'], 'Team A')
    
    def test_find_by_user(self):
        """Test finding predictions by user."""
        # Create multiple predictions
        for i in range(3):
            self.prediction_repo.create({
                'user_id': self.user_id,
                'market_id': self.market_id,
                'predicted_outcome': f'Outcome {i}',
                'confidence_level': Decimal('0.5000'),
                'stake_amount': Decimal('10.00'),
                'odds_when_placed': Decimal('2.0000'),
                'status': 'active' if i < 2 else 'won'
            })
        
        # Find all predictions
        all_predictions = self.prediction_repo.find_by_user(self.user_id)
        self.assertEqual(len(all_predictions), 3)
        
        # Find only active predictions
        active_predictions = self.prediction_repo.find_by_user(self.user_id, status='active')
        self.assertEqual(len(active_predictions), 2)
    
    def test_user_performance(self):
        """Test user performance calculation."""
        # Create predictions with different outcomes
        predictions = [
            {'status': 'won', 'profit_loss': Decimal('10.00'), 'stake': Decimal('10.00')},
            {'status': 'won', 'profit_loss': Decimal('15.00'), 'stake': Decimal('10.00')},
            {'status': 'lost', 'profit_loss': Decimal('-10.00'), 'stake': Decimal('10.00')},
            {'status': 'lost', 'profit_loss': Decimal('-10.00'), 'stake': Decimal('10.00')},
        ]
        
        for pred in predictions:
            self.prediction_repo.create({
                'user_id': self.user_id,
                'market_id': self.market_id,
                'predicted_outcome': 'Team A',
                'confidence_level': Decimal('0.6000'),
                'stake_amount': pred['stake'],
                'odds_when_placed': Decimal('2.0000'),
                'status': pred['status'],
                'profit_loss': pred['profit_loss']
            })
        
        performance = self.prediction_repo.get_user_performance(self.user_id)
        
        self.assertEqual(performance['total_predictions'], 4)
        self.assertEqual(performance['correct_predictions'], 2)
        self.assertEqual(performance['accuracy_rate'], 0.5)
        self.assertEqual(float(performance['total_profit_loss']), 5.0)


class TestPerformance(unittest.TestCase):
    """Test database performance with larger datasets."""
    
    def setUp(self):
        """Set up test database."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.db = init_db(self.db_path)
        
        # Apply schema
        migration_manager = MigrationManager(self.db)
        migrations_dir = os.path.join(self.test_dir, 'migrations')
        os.makedirs(migrations_dir)
        create_initial_migration(migrations_dir)
        migration_manager.load_migrations_from_directory(migrations_dir)
        migration_manager.migrate()
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close_all_connections()
        shutil.rmtree(self.test_dir)
    
    def test_bulk_insert_performance(self):
        """Test performance of bulk inserts."""
        import time
        
        # Create test data
        user_data = []
        for i in range(1000):
            user_data.append((
                f'user{i}',
                f'user{i}@example.com',
                'password_hash',
                'salt',
                f'User {i}',
                'active'
            ))
        
        # Time bulk insert
        start_time = time.time()
        
        self.db.execute_many("""
            INSERT INTO users (username, email, password_hash, salt, display_name, account_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, user_data)
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Verify all users were created
        count = self.db.execute_single("SELECT COUNT(*) as count FROM users")
        self.assertEqual(count['count'], 1000)
        
        print(f"\nBulk insert of 1000 users: {insert_time:.3f} seconds")
        self.assertLess(insert_time, 5.0)  # Should complete within 5 seconds
    
    def test_complex_query_performance(self):
        """Test performance of complex queries with joins."""
        import time
        
        user_repo = UserRepository(self.db)
        league_repo = LeagueRepository(self.db)
        prediction_repo = PredictionRepository(self.db)
        
        # Create test data
        user_ids = []
        for i in range(100):
            user_id = user_repo.create({
                'username': f'perfuser{i}',
                'email': f'perfuser{i}@example.com',
                'password_hash': 'hash',
                'salt': 'salt'
            })
            user_ids.append(user_id)
        
        league_id = league_repo.create({
            'league_name': 'Performance League',
            'league_type': 'fantasy_sports',
            'creator_id': user_ids[0],
            'scoring_system': {},
            'league_rules': {}
        })
        
        # Create event and market
        self.db.execute_modify("""
            INSERT INTO event_categories (category_name, category_type, description)
            VALUES ('Perf Category', 'sports', 'Performance test category')
        """)
        category_id = self.db.get_last_insert_id()
        
        self.db.execute_modify("""
            INSERT INTO events (category_id, event_name, scheduled_start, prediction_deadline, status)
            VALUES (?, 'Perf Event', datetime('now', '+1 day'), datetime('now', '+1 hour'), 'upcoming')
        """, (category_id,))
        event_id = self.db.get_last_insert_id()
        
        self.db.execute_modify("""
            INSERT INTO prediction_markets (event_id, league_id, market_name, market_type, question, options, status)
            VALUES (?, ?, 'Perf Market', 'binary', 'Who wins?', '["A", "B"]', 'open')
        """, (event_id, league_id))
        market_id = self.db.get_last_insert_id()
        
        # Create predictions for each user
        for user_id in user_ids:
            prediction_repo.create({
                'user_id': user_id,
                'market_id': market_id,
                'league_id': league_id,
                'predicted_outcome': 'A',
                'confidence_level': Decimal('0.7500'),
                'stake_amount': Decimal('25.00'),
                'odds_when_placed': Decimal('2.0000'),
                'status': 'won',
                'profit_loss': Decimal('25.00')
            })
        
        # Time complex query
        start_time = time.time()
        
        complex_query = """
        SELECT 
            u.username,
            u.display_name,
            l.league_name,
            COUNT(p.prediction_id) as total_predictions,
            SUM(p.profit_loss) as total_profit,
            AVG(p.confidence_level) as avg_confidence
        FROM users u
        JOIN predictions p ON u.user_id = p.user_id
        JOIN leagues l ON p.league_id = l.league_id
        WHERE p.status IN ('won', 'lost')
        GROUP BY u.user_id, l.league_id
        ORDER BY total_profit DESC
        LIMIT 50
        """
        
        results = self.db.execute_query(complex_query)
        
        end_time = time.time()
        query_time = end_time - start_time
        
        self.assertGreater(len(results), 0)
        print(f"Complex query with joins (100 users): {query_time:.3f} seconds")
        self.assertLess(query_time, 1.0)  # Should complete within 1 second


def run_tests():
    """Run all database tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatabaseConnection,
        TestMigrations,
        TestUserRepository,
        TestLeagueRepository,
        TestPredictionRepository,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*50)
    print("DATABASE TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*50)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)