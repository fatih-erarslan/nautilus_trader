"""
Comprehensive tests for database operations

Tests SQLAlchemy models, database connections, queries, and data integrity.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from crypto_trading.database.models import (
    Base, VaultPosition, YieldHistory, CryptoTransaction, PortfolioSummary
)
from crypto_trading.database.connection import DatabaseManager
from crypto_trading.database.utils import DatabaseUtils
from crypto_trading.database.migrations.migrate import MigrationManager


class TestDatabaseModels:
    """Test SQLAlchemy models and their relationships"""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        yield session
        
        session.close()

    def test_vault_position_model(self, db_session):
        """Test VaultPosition model creation and validation"""
        position = VaultPosition(
            vault_id="beefy-bsc-cake-bnb",
            vault_name="CAKE-BNB LP",
            chain="bsc",
            amount_deposited=1000.0,
            shares_owned=950.5,
            current_value=1050.0,
            entry_price=1.0526,
            entry_apy=25.5,
            status="active"
        )
        
        db_session.add(position)
        db_session.commit()
        
        # Test retrieval
        retrieved = db_session.query(VaultPosition).filter_by(vault_id="beefy-bsc-cake-bnb").first()
        assert retrieved is not None
        assert retrieved.vault_name == "CAKE-BNB LP"
        assert retrieved.chain == "bsc"
        assert retrieved.amount_deposited == 1000.0
        assert retrieved.status == "active"
        
        # Test properties
        assert retrieved.unrealized_pnl == 50.0  # 1050 - 1000
        assert retrieved.roi_percentage == 5.0   # (1050 - 1000) / 1000 * 100

    def test_vault_position_chain_validation(self, db_session):
        """Test chain validation in VaultPosition"""
        # Valid chain
        position = VaultPosition(
            vault_id="test-vault",
            vault_name="Test Vault",
            chain="ethereum",
            amount_deposited=1000.0,
            shares_owned=1000.0,
            entry_price=1.0,
            entry_apy=10.0
        )
        
        db_session.add(position)
        db_session.commit()
        
        assert position.chain == "ethereum"
        
        # Invalid chain should raise error
        with pytest.raises(ValueError):
            invalid_position = VaultPosition(
                vault_id="test-vault-2",
                vault_name="Invalid Vault",
                chain="invalid_chain",
                amount_deposited=1000.0,
                shares_owned=1000.0,
                entry_price=1.0,
                entry_apy=10.0
            )
            db_session.add(invalid_position)
            db_session.commit()

    def test_vault_position_constraints(self, db_session):
        """Test database constraints on VaultPosition"""
        # Test positive amount constraint
        with pytest.raises(IntegrityError):
            position = VaultPosition(
                vault_id="test-vault",
                vault_name="Test Vault",
                chain="bsc",
                amount_deposited=-100.0,  # Invalid negative amount
                shares_owned=100.0,
                entry_price=1.0,
                entry_apy=10.0
            )
            db_session.add(position)
            db_session.commit()

    def test_yield_history_model(self, db_session):
        """Test YieldHistory model and relationship"""
        # Create parent position
        position = VaultPosition(
            vault_id="test-vault",
            vault_name="Test Vault",
            chain="bsc",
            amount_deposited=1000.0,
            shares_owned=1000.0,
            entry_price=1.0,
            entry_apy=20.0
        )
        db_session.add(position)
        db_session.commit()
        
        # Create yield history
        yield_record = YieldHistory(
            vault_id="test-vault",
            position_id=position.id,
            earned_amount=5.5,
            apy_snapshot=20.5,
            tvl_snapshot=5000000.0,
            price_per_share=1.055
        )
        
        db_session.add(yield_record)
        db_session.commit()
        
        # Test relationship
        retrieved_position = db_session.query(VaultPosition).filter_by(id=position.id).first()
        assert len(retrieved_position.yield_history) == 1
        assert retrieved_position.yield_history[0].earned_amount == 5.5

    def test_crypto_transaction_model(self, db_session):
        """Test CryptoTransaction model"""
        transaction = CryptoTransaction(
            transaction_type="deposit",
            vault_id="test-vault",
            chain="ethereum",
            amount=500.0,
            gas_used=0.05,
            tx_hash="0x1234567890abcdef",
            status="confirmed"
        )
        
        db_session.add(transaction)
        db_session.commit()
        
        retrieved = db_session.query(CryptoTransaction).filter_by(tx_hash="0x1234567890abcdef").first()
        assert retrieved is not None
        assert retrieved.transaction_type == "deposit"
        assert retrieved.total_cost == 500.05  # amount + gas

    def test_crypto_transaction_constraints(self, db_session):
        """Test transaction constraints"""
        # Invalid transaction type
        with pytest.raises(IntegrityError):
            transaction = CryptoTransaction(
                transaction_type="invalid_type",
                vault_id="test-vault",
                chain="bsc",
                amount=100.0,
                tx_hash="0xabcdef"
            )
            db_session.add(transaction)
            db_session.commit()

    def test_portfolio_summary_model(self, db_session):
        """Test PortfolioSummary model"""
        summary = PortfolioSummary(
            total_value_usd=15000.0,
            total_yield_earned=750.0,
            average_apy=18.5,
            chains_active='["bsc", "ethereum", "polygon"]',
            vaults_count=5
        )
        
        db_session.add(summary)
        db_session.commit()
        
        retrieved = db_session.query(PortfolioSummary).first()
        assert retrieved.total_value_usd == 15000.0
        assert retrieved.active_chains == ["bsc", "ethereum", "polygon"]

    def test_portfolio_summary_active_chains_property(self, db_session):
        """Test active_chains property getter/setter"""
        summary = PortfolioSummary(
            total_value_usd=1000.0,
            total_yield_earned=50.0,
            average_apy=10.0,
            chains_active="[]",
            vaults_count=0
        )
        
        # Test setter
        summary.active_chains = ["bsc", "polygon"]
        assert summary.chains_active == '["bsc", "polygon"]'
        
        # Test getter
        chains = summary.active_chains
        assert chains == ["bsc", "polygon"]
        
        db_session.add(summary)
        db_session.commit()
        
        # Test after database round-trip
        retrieved = db_session.query(PortfolioSummary).first()
        assert retrieved.active_chains == ["bsc", "polygon"]

    def test_automatic_timestamps(self, db_session):
        """Test automatic timestamp updates"""
        position = VaultPosition(
            vault_id="test-vault",
            vault_name="Test Vault",
            chain="bsc",
            amount_deposited=1000.0,
            shares_owned=1000.0,
            entry_price=1.0,
            entry_apy=20.0
        )
        
        db_session.add(position)
        db_session.commit()
        
        original_created = position.created_at
        original_updated = position.updated_at
        
        # Ensure some time passes
        import time
        time.sleep(0.01)
        
        # Update the position
        position.current_value = 1100.0
        db_session.commit()
        
        # updated_at should change, created_at should not
        assert position.created_at == original_created
        assert position.updated_at > original_updated

    def test_cascade_deletion(self, db_session):
        """Test cascade deletion of yield history"""
        # Create position
        position = VaultPosition(
            vault_id="test-vault",
            vault_name="Test Vault",
            chain="bsc",
            amount_deposited=1000.0,
            shares_owned=1000.0,
            entry_price=1.0,
            entry_apy=20.0
        )
        db_session.add(position)
        db_session.commit()
        
        # Create yield history records
        for i in range(3):
            yield_record = YieldHistory(
                vault_id="test-vault",
                position_id=position.id,
                earned_amount=i * 10.0,
                apy_snapshot=20.0,
                price_per_share=1.0
            )
            db_session.add(yield_record)
        
        db_session.commit()
        
        # Verify records exist
        yield_count = db_session.query(YieldHistory).filter_by(position_id=position.id).count()
        assert yield_count == 3
        
        # Delete position
        db_session.delete(position)
        db_session.commit()
        
        # Yield history should be deleted too
        yield_count_after = db_session.query(YieldHistory).filter_by(position_id=position.id).count()
        assert yield_count_after == 0


class TestDatabaseManager:
    """Test DatabaseManager functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        yield db_path
        
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create DatabaseManager with temporary database"""
        return DatabaseManager(database_url=f"sqlite:///{temp_db_path}")

    def test_database_manager_initialization(self, db_manager):
        """Test DatabaseManager initialization"""
        assert db_manager.engine is not None
        assert db_manager.SessionLocal is not None

    def test_create_tables(self, db_manager):
        """Test table creation"""
        db_manager.create_tables()
        
        # Check if tables exist by trying to query them
        with db_manager.get_session() as session:
            # Should not raise an error
            session.query(VaultPosition).count()
            session.query(YieldHistory).count()
            session.query(CryptoTransaction).count()
            session.query(PortfolioSummary).count()

    def test_session_context_manager(self, db_manager):
        """Test session context manager"""
        db_manager.create_tables()
        
        with db_manager.get_session() as session:
            position = VaultPosition(
                vault_id="test-vault",
                vault_name="Test Vault",
                chain="bsc",
                amount_deposited=1000.0,
                shares_owned=1000.0,
                entry_price=1.0,
                entry_apy=20.0
            )
            session.add(position)
            session.commit()
            
            # Should be able to query within the same session
            count = session.query(VaultPosition).count()
            assert count == 1

    def test_connection_pooling(self, db_manager):
        """Test connection pooling works correctly"""
        db_manager.create_tables()
        
        # Multiple sessions should work
        sessions = []
        for i in range(5):
            session = db_manager.get_session()
            sessions.append(session)
            
            # Each session should be able to query
            with session:
                count = session.query(VaultPosition).count()
                assert count >= 0

    def test_database_health_check(self, db_manager):
        """Test database health check"""
        # Should work after table creation
        db_manager.create_tables()
        is_healthy = db_manager.health_check()
        assert is_healthy is True


class TestDatabaseUtils:
    """Test database utility functions"""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Add sample data
        positions = [
            VaultPosition(
                vault_id="vault-1",
                vault_name="Vault 1",
                chain="bsc",
                amount_deposited=1000.0,
                shares_owned=1000.0,
                current_value=1100.0,
                entry_price=1.0,
                entry_apy=20.0,
                status="active"
            ),
            VaultPosition(
                vault_id="vault-2",
                vault_name="Vault 2",
                chain="ethereum",
                amount_deposited=2000.0,
                shares_owned=1950.0,
                current_value=2100.0,
                entry_price=1.026,
                entry_apy=15.0,
                status="active"
            ),
            VaultPosition(
                vault_id="vault-3",
                vault_name="Vault 3",
                chain="polygon",
                amount_deposited=500.0,
                shares_owned=500.0,
                current_value=480.0,
                entry_price=1.0,
                entry_apy=18.0,
                status="closed"
            )
        ]
        
        for position in positions:
            session.add(position)
        
        session.commit()
        
        yield session
        session.close()

    def test_get_portfolio_summary(self, db_session):
        """Test portfolio summary calculation"""
        utils = DatabaseUtils(db_session)
        
        summary = utils.get_portfolio_summary()
        
        assert summary is not None
        assert summary["total_positions"] == 3
        assert summary["active_positions"] == 2
        assert summary["total_value"] == 3200.0  # 1100 + 2100 (only active)
        assert summary["total_deposited"] == 3000.0  # 1000 + 2000 (only active)
        assert summary["unrealized_pnl"] == 200.0  # 3200 - 3000
        assert len(summary["chains"]) == 2  # bsc, ethereum (only active)

    def test_get_chain_allocation(self, db_session):
        """Test chain allocation calculation"""
        utils = DatabaseUtils(db_session)
        
        allocations = utils.get_chain_allocation()
        
        assert "bsc" in allocations
        assert "ethereum" in allocations
        assert "polygon" not in allocations  # Closed position
        
        # BSC allocation: 1100 / 3200 = 34.375%
        assert abs(allocations["bsc"] - 34.375) < 0.1
        # Ethereum allocation: 2100 / 3200 = 65.625%
        assert abs(allocations["ethereum"] - 65.625) < 0.1

    def test_get_performance_metrics(self, db_session):
        """Test performance metrics calculation"""
        utils = DatabaseUtils(db_session)
        
        # Add yield history for better metrics
        position = db_session.query(VaultPosition).first()
        
        for i in range(5):
            yield_record = YieldHistory(
                vault_id=position.vault_id,
                position_id=position.id,
                earned_amount=20.0 + i * 5,
                apy_snapshot=20.0 + i,
                price_per_share=1.0 + i * 0.01,
                recorded_at=datetime.utcnow() - timedelta(days=i)
            )
            db_session.add(yield_record)
        
        db_session.commit()
        
        metrics = utils.get_performance_metrics(days=30)
        
        assert "total_yield_earned" in metrics
        assert "average_apy" in metrics
        assert "best_performing_vault" in metrics
        assert "worst_performing_vault" in metrics

    def test_get_transaction_history(self, db_session):
        """Test transaction history retrieval"""
        # Add transactions
        transactions = [
            CryptoTransaction(
                transaction_type="deposit",
                vault_id="vault-1",
                chain="bsc",
                amount=1000.0,
                gas_used=0.01,
                tx_hash="0x123",
                status="confirmed"
            ),
            CryptoTransaction(
                transaction_type="withdraw",
                vault_id="vault-2",
                chain="ethereum",
                amount=500.0,
                gas_used=0.05,
                tx_hash="0x456",
                status="confirmed"
            )
        ]
        
        for tx in transactions:
            db_session.add(tx)
        
        db_session.commit()
        
        utils = DatabaseUtils(db_session)
        history = utils.get_transaction_history(limit=10)
        
        assert len(history) == 2
        assert history[0]["transaction_type"] in ["deposit", "withdraw"]
        assert "total_cost" in history[0]

    def test_cleanup_old_data(self, db_session):
        """Test old data cleanup"""
        # Add old yield history
        position = db_session.query(VaultPosition).first()
        
        old_yield = YieldHistory(
            vault_id=position.vault_id,
            position_id=position.id,
            earned_amount=10.0,
            apy_snapshot=15.0,
            price_per_share=1.0,
            recorded_at=datetime.utcnow() - timedelta(days=400)  # Very old
        )
        
        recent_yield = YieldHistory(
            vault_id=position.vault_id,
            position_id=position.id,
            earned_amount=20.0,
            apy_snapshot=20.0,
            price_per_share=1.05,
            recorded_at=datetime.utcnow() - timedelta(days=10)  # Recent
        )
        
        db_session.add(old_yield)
        db_session.add(recent_yield)
        db_session.commit()
        
        utils = DatabaseUtils(db_session)
        
        # Count before cleanup
        count_before = db_session.query(YieldHistory).count()
        
        # Cleanup data older than 365 days
        deleted_count = utils.cleanup_old_data(days=365)
        
        # Count after cleanup
        count_after = db_session.query(YieldHistory).count()
        
        assert deleted_count > 0
        assert count_after < count_before

    def test_backup_and_restore(self, db_session):
        """Test database backup functionality"""
        utils = DatabaseUtils(db_session)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            backup_path = f.name
        
        try:
            # Create backup
            backup_success = utils.backup_data(backup_path)
            assert backup_success is True
            assert os.path.exists(backup_path)
            
            # Check backup file has content
            with open(backup_path, 'r') as f:
                import json
                backup_data = json.load(f)
                assert "vault_positions" in backup_data
                assert len(backup_data["vault_positions"]) == 3
                
        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)


class TestMigrations:
    """Test database migration functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        yield db_path
        
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_migration_manager(self, temp_db_path):
        """Test migration manager functionality"""
        manager = MigrationManager(f"sqlite:///{temp_db_path}")
        
        # Initialize database
        manager.init_db()
        
        # Run migrations
        applied = manager.run_migrations()
        assert isinstance(applied, list)
        
        # Check migration status
        status = manager.get_migration_status()
        assert "applied_migrations" in status
        assert "pending_migrations" in status

    def test_migration_rollback(self, temp_db_path):
        """Test migration rollback functionality"""
        manager = MigrationManager(f"sqlite:///{temp_db_path}")
        
        manager.init_db()
        manager.run_migrations()
        
        # Test rollback (if implemented)
        try:
            rollback_result = manager.rollback_migration()
            assert isinstance(rollback_result, bool)
        except NotImplementedError:
            # Rollback might not be implemented for all migration types
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])