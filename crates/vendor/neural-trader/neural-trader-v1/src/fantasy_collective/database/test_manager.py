"""Test Suite for Fantasy Collective Database Manager

Comprehensive tests for all database operations including:
- User CRUD operations
- League management
- Prediction handling
- Transaction management
- Query optimization
- Error handling
- Thread safety
"""

import pytest
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from manager import (
    FantasyCollectiveDBManager,
    User,
    League,
    LeagueMembership,
    Prediction,
    Transaction,
    UserRole,
    LeagueStatus,
    PredictionStatus,
    TransactionType,
    init_database,
    get_db_manager
)

class TestFantasyCollectiveDBManager:
    """Test suite for FantasyCollectiveDBManager"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db_manager = FantasyCollectiveDBManager(db_path=db_path)
        db_manager.create_tables()
        
        yield db_manager
        
        # Cleanup
        db_manager.close()
        Path(db_path).unlink(missing_ok=True)
    
    def test_database_initialization(self, temp_db):
        """Test database initialization and table creation"""
        stats = temp_db.get_database_stats()
        
        assert stats['total_users'] == 0
        assert stats['total_leagues'] == 0
        assert stats['total_predictions'] == 0
        assert stats['total_transactions'] == 0
    
    def test_user_crud_operations(self, temp_db):
        """Test user create, read, update, delete operations"""
        # Create user
        user = temp_db.create_user(
            username='testuser',
            email='test@example.com',
            password_hash='hashed_password',
            full_name='Test User'
        )
        
        assert user.id is not None
        assert user.username == 'testuser'
        assert user.email == 'test@example.com'
        assert user.role == 'member'
        assert user.balance == 0.0
        assert user.active is True
        
        # Read user
        retrieved_user = temp_db.get_user_by_id(user.id)
        assert retrieved_user is not None
        assert retrieved_user.username == 'testuser'
        
        # Get by username
        user_by_username = temp_db.get_user_by_username('testuser')
        assert user_by_username.id == user.id
        
        # Get by email
        user_by_email = temp_db.get_user_by_email('test@example.com')
        assert user_by_email.id == user.id
        
        # Update user
        updated_user = temp_db.update_user(user.id, balance=100.0, role='admin')
        assert updated_user.balance == 100.0
        assert updated_user.role == 'admin'
        
        # Delete user (soft delete)
        assert temp_db.delete_user(user.id) is True
        deleted_user = temp_db.get_user_by_id(user.id)
        assert deleted_user.active is False
    
    def test_user_validation(self, temp_db):
        """Test user input validation"""
        # Test invalid email
        with pytest.raises(ValueError, match="Invalid email format"):
            temp_db.create_user('test', 'invalid_email', 'hash')
        
        # Test short username
        with pytest.raises(ValueError, match="Username must be at least 3 characters"):
            temp_db.create_user('xy', 'test@example.com', 'hash')
        
        # Test invalid characters in username
        with pytest.raises(ValueError, match="Username must contain only alphanumeric characters and underscores"):
            temp_db.create_user('test@user', 'test@example.com', 'hash')
    
    def test_league_operations(self, temp_db):
        """Test league creation and management"""
        # Create a user first
        user = temp_db.create_user('creator', 'creator@example.com', 'hash')
        
        # Create league
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=30)
        
        league = temp_db.create_league(
            name='Test League',
            created_by=user.id,
            entry_fee=10.0,
            max_members=50,
            start_date=start_date,
            end_date=end_date,
            description='A test league'
        )
        
        assert league.id is not None
        assert league.name == 'Test League'
        assert league.created_by == user.id
        assert league.entry_fee == 10.0
        assert league.max_members == 50
        assert league.current_members == 0
        assert league.status == 'active'
        assert league.slots_available == 50
        
        # Get league by ID
        retrieved_league = temp_db.get_league_by_id(league.id)
        assert retrieved_league.id == league.id
        
        # Test active leagues
        active_leagues = temp_db.get_active_leagues()
        assert len(active_leagues) == 1
        assert active_leagues[0].id == league.id
    
    def test_league_membership(self, temp_db):
        """Test joining and leaving leagues"""
        # Create user and league
        user = temp_db.create_user('testuser', 'test@example.com', 'hash')
        temp_db.update_user(user.id, balance=50.0)  # Add balance for entry fee
        
        creator = temp_db.create_user('creator', 'creator@example.com', 'hash')
        
        league = temp_db.create_league(
            name='Test League',
            created_by=creator.id,
            entry_fee=10.0,
            max_members=2
        )
        
        # Join league
        assert temp_db.join_league(user.id, league.id) is True
        
        # Check league membership
        updated_league = temp_db.get_league_by_id(league.id)
        assert updated_league.current_members == 1
        assert updated_league.prize_pool == 10.0
        
        # Check user balance was deducted
        updated_user = temp_db.get_user_by_id(user.id)
        assert updated_user.balance == 40.0
        
        # Try to join again (should fail)
        assert temp_db.join_league(user.id, league.id) is False
        
        # Create another user and fill league
        user2 = temp_db.create_user('testuser2', 'test2@example.com', 'hash')
        temp_db.update_user(user2.id, balance=50.0)
        assert temp_db.join_league(user2.id, league.id) is True
        
        # Try to join full league (should fail)
        user3 = temp_db.create_user('testuser3', 'test3@example.com', 'hash')
        temp_db.update_user(user3.id, balance=50.0)
        assert temp_db.join_league(user3.id, league.id) is False
        
        # Leave league
        assert temp_db.leave_league(user.id, league.id) is True
        updated_league = temp_db.get_league_by_id(league.id)
        assert updated_league.current_members == 1
    
    def test_prediction_operations(self, temp_db):
        """Test prediction creation and resolution"""
        # Setup user and league
        user = temp_db.create_user('testuser', 'test@example.com', 'hash')
        creator = temp_db.create_user('creator', 'creator@example.com', 'hash')
        league = temp_db.create_league('Test League', creator.id)
        temp_db.join_league(user.id, league.id)
        
        # Create prediction
        event_date = datetime.utcnow() + timedelta(days=1)
        prediction_data = {
            'team_a': 'Team Alpha',
            'team_b': 'Team Beta',
            'predicted_winner': 'Team Alpha',
            'predicted_score': '2-1'
        }
        
        prediction = temp_db.create_prediction(
            user_id=user.id,
            league_id=league.id,
            event_name='Championship Match',
            event_date=event_date,
            prediction_data=prediction_data,
            confidence_level=0.8
        )
        
        assert prediction.id is not None
        assert prediction.status == 'pending'
        assert prediction.confidence_level == 0.8
        assert prediction.prediction_dict == prediction_data
        
        # Get predictions
        user_predictions = temp_db.get_predictions_by_user(user.id)
        assert len(user_predictions) == 1
        assert user_predictions[0].id == prediction.id
        
        league_predictions = temp_db.get_predictions_by_league(league.id)
        assert len(league_predictions) == 1
        
        # Resolve prediction
        assert temp_db.resolve_prediction(prediction.id, 100) is True
        
        # Check resolution
        resolved_prediction = temp_db.get_predictions_by_user(user.id)[0]
        assert resolved_prediction.status == 'resolved'
        assert resolved_prediction.points_awarded == 100
        assert resolved_prediction.resolved_at is not None
        
        # Check user points updated
        updated_user = temp_db.get_user_by_id(user.id)
        assert updated_user.total_points == 100
    
    def test_transaction_operations(self, temp_db):
        """Test financial transaction operations"""
        user = temp_db.create_user('testuser', 'test@example.com', 'hash')
        
        # Test deposit
        deposit = temp_db.create_transaction(
            user_id=user.id,
            transaction_type='deposit',
            amount=100.0,
            description='Initial deposit'
        )
        
        assert deposit.transaction_type == 'deposit'
        assert deposit.amount == 100.0
        assert deposit.balance_before == 0.0
        assert deposit.balance_after == 100.0
        
        # Check user balance updated
        updated_user = temp_db.get_user_by_id(user.id)
        assert updated_user.balance == 100.0
        
        # Test withdrawal
        withdrawal = temp_db.create_transaction(
            user_id=user.id,
            transaction_type='withdrawal',
            amount=30.0,
            description='Test withdrawal'
        )
        
        assert withdrawal.balance_before == 100.0
        assert withdrawal.balance_after == 70.0
        
        # Test insufficient balance
        with pytest.raises(ValueError, match="Insufficient balance"):
            temp_db.create_transaction(user.id, 'withdrawal', 100.0)
        
        # Test transaction history
        transactions = temp_db.get_user_transactions(user.id)
        assert len(transactions) == 2
        assert transactions[0].transaction_type == 'withdrawal'  # Most recent first
        assert transactions[1].transaction_type == 'deposit'
    
    def test_scoring_and_rankings(self, temp_db):
        """Test scoring calculations and league rankings"""
        # Setup users and league
        creator = temp_db.create_user('creator', 'creator@example.com', 'hash')
        league = temp_db.create_league('Test League', creator.id)
        
        users = []
        for i in range(3):
            user = temp_db.create_user(f'user{i}', f'user{i}@example.com', 'hash')
            users.append(user)
            temp_db.join_league(user.id, league.id)
        
        # Award different points to users
        points = [150, 100, 75]
        for user, point_value in zip(users, points):
            temp_db.create_prediction(
                user_id=user.id,
                league_id=league.id,
                event_name=f'Event for {user.username}',
                event_date=datetime.utcnow() + timedelta(days=1),
                prediction_data={'test': 'data'},
                confidence_level=0.5
            )
            prediction = temp_db.get_predictions_by_user(user.id)[0]
            temp_db.resolve_prediction(prediction.id, point_value)
        
        # Calculate rankings
        rankings = temp_db.calculate_league_rankings(league.id)
        
        assert len(rankings) == 3
        assert rankings[0]['rank'] == 1
        assert rankings[0]['points'] == 150
        assert rankings[0]['username'] == 'user0'
        
        assert rankings[1]['rank'] == 2
        assert rankings[1]['points'] == 100
        
        assert rankings[2]['rank'] == 3
        assert rankings[2]['points'] == 75
        
        # Test leaderboard
        leaderboard = temp_db.get_leaderboard(league.id)
        assert len(leaderboard) == 3
        assert leaderboard[0]['points'] == 150
    
    def test_database_statistics(self, temp_db):
        """Test database statistics collection"""
        # Add some data
        user = temp_db.create_user('testuser', 'test@example.com', 'hash')
        creator = temp_db.create_user('creator', 'creator@example.com', 'hash')
        league = temp_db.create_league('Test League', creator.id)
        temp_db.join_league(user.id, league.id)
        temp_db.create_prediction(
            user.id, league.id, 'Test Event',
            datetime.utcnow() + timedelta(days=1),
            {'test': 'data'}, 0.5
        )
        temp_db.create_transaction(user.id, 'deposit', 100.0)
        
        stats = temp_db.get_database_stats()
        
        assert stats['total_users'] == 2
        assert stats['active_users'] == 2
        assert stats['total_leagues'] == 1
        assert stats['active_leagues'] == 1
        assert stats['total_predictions'] == 1
        assert stats['pending_predictions'] == 1
        assert stats['total_transactions'] == 1
        assert stats['total_volume'] == 100.0
    
    def test_caching(self, temp_db):
        """Test query result caching"""
        # Create some users
        for i in range(5):
            temp_db.create_user(f'user{i}', f'user{i}@example.com', 'hash')
        
        # First call should be a cache miss
        start_time = time.time()
        users1 = temp_db.get_active_users()
        time1 = time.time() - start_time
        
        # Second call should be a cache hit (faster)
        start_time = time.time()
        users2 = temp_db.get_active_users()
        time2 = time.time() - start_time
        
        assert len(users1) == len(users2) == 5
        assert time2 < time1  # Cache hit should be faster
    
    def test_thread_safety(self, temp_db):
        """Test thread safety of database operations"""
        results = []
        errors = []
        
        def create_user_thread(thread_id):
            try:
                user = temp_db.create_user(
                    f'user{thread_id}',
                    f'user{thread_id}@example.com',
                    'hash'
                )
                results.append(user.id)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_user_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique IDs
    
    def test_database_backup(self, temp_db):
        """Test database backup functionality"""
        # Add some data
        user = temp_db.create_user('testuser', 'test@example.com', 'hash')
        
        # Create backup
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            backup_path = f.name
        
        temp_db.backup_database(backup_path)
        
        # Verify backup exists and has data
        backup_db = FantasyCollectiveDBManager(db_path=backup_path)
        backup_stats = backup_db.get_database_stats()
        
        assert backup_stats['total_users'] == 1
        
        # Cleanup
        backup_db.close()
        Path(backup_path).unlink(missing_ok=True)
    
    def test_database_optimization(self, temp_db):
        """Test database optimization"""
        # This should run without errors
        temp_db.optimize_database()
    
    def test_migration_support(self, temp_db):
        """Test data migration functionality"""
        # Create a simple migration script
        migration_script = """
        CREATE TABLE IF NOT EXISTS test_migration (
            id INTEGER PRIMARY KEY,
            name VARCHAR(50)
        );
        INSERT INTO test_migration (name) VALUES ('test_value');
        """
        
        # Run migration
        temp_db.migrate_data(migration_script)
        
        # Verify migration worked
        with temp_db.session_scope() as session:
            result = session.execute("SELECT * FROM test_migration").fetchall()
            assert len(result) == 1
            assert result[0][1] == 'test_value'
    
    def test_context_manager(self):
        """Test database manager as context manager"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Test context manager usage
        with FantasyCollectiveDBManager(db_path=db_path) as db:
            db.create_tables()
            user = db.create_user('testuser', 'test@example.com', 'hash')
            assert user.id is not None
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

def test_global_database_instance():
    """Test global database instance management"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Test initialization
    db1 = init_database(db_path)
    db2 = get_db_manager()
    
    assert db1 is db2  # Should be the same instance
    
    # Cleanup
    db1.close()
    Path(db_path).unlink(missing_ok=True)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])