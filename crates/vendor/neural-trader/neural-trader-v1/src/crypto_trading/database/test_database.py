"""Test script for crypto trading database

Verifies all database functionality is working correctly.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.crypto_trading.database import (
    init_database,
    get_db_session,
    VaultPosition,
    YieldHistory,
    CryptoTransaction,
    PortfolioSummary,
    get_active_positions,
    calculate_total_portfolio_value,
    batch_insert_yield_history,
    create_portfolio_snapshot,
    get_chain_allocation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_operations():
    """Test all database operations"""
    logger.info("Starting database tests...")
    
    # Initialize database
    db = init_database("test_trading.db")
    
    try:
        # Test 1: Create vault positions
        logger.info("\n1. Testing vault position creation...")
        with get_db_session() as session:
            positions = [
                VaultPosition(
                    vault_id="beefy-eth-wbtc",
                    vault_name="WBTC-ETH LP",
                    chain="ethereum",
                    amount_deposited=10000.0,
                    shares_owned=95.5,
                    current_value=10500.0,
                    entry_price=104.71,
                    entry_apy=15.5
                ),
                VaultPosition(
                    vault_id="beefy-bnb-cake",
                    vault_name="CAKE-BNB LP",
                    chain="bsc",
                    amount_deposited=5000.0,
                    shares_owned=450.0,
                    current_value=5200.0,
                    entry_price=11.11,
                    entry_apy=25.3
                ),
                VaultPosition(
                    vault_id="beefy-matic-usdc",
                    vault_name="USDC-MATIC LP",
                    chain="polygon",
                    amount_deposited=3000.0,
                    shares_owned=2950.0,
                    current_value=3100.0,
                    entry_price=1.02,
                    entry_apy=8.7
                )
            ]
            
            for pos in positions:
                session.add(pos)
            session.commit()
            
            created_count = session.query(VaultPosition).count()
            logger.info(f"✓ Created {created_count} vault positions")
        
        # Test 2: Query active positions
        logger.info("\n2. Testing position queries...")
        active = get_active_positions()
        logger.info(f"✓ Found {len(active)} active positions")
        
        eth_positions = get_active_positions(chain="ethereum")
        logger.info(f"✓ Found {len(eth_positions)} Ethereum positions")
        
        # Test 3: Insert yield history
        logger.info("\n3. Testing yield history...")
        yield_data = []
        for position in active:
            for i in range(5):  # 5 days of history
                yield_data.append({
                    'vault_id': position.vault_id,
                    'position_id': position.id,
                    'earned_amount': 10.5 + i * 2,
                    'apy_snapshot': position.entry_apy + i * 0.1,
                    'tvl_snapshot': 1000000 + i * 50000,
                    'price_per_share': position.entry_price * (1 + i * 0.001),
                    'recorded_at': datetime.utcnow() - timedelta(days=4-i)
                })
        
        inserted = batch_insert_yield_history(yield_data)
        logger.info(f"✓ Inserted {inserted} yield history records")
        
        # Test 4: Create transactions
        logger.info("\n4. Testing transaction creation...")
        with get_db_session() as session:
            transactions = [
                CryptoTransaction(
                    transaction_type="deposit",
                    vault_id="beefy-eth-wbtc",
                    chain="ethereum",
                    amount=10000.0,
                    gas_used=0.05,
                    tx_hash="0x123abc456def789",
                    status="confirmed"
                ),
                CryptoTransaction(
                    transaction_type="deposit",
                    vault_id="beefy-bnb-cake",
                    chain="bsc",
                    amount=5000.0,
                    gas_used=0.002,
                    tx_hash="0x456def789abc123",
                    status="confirmed"
                ),
                CryptoTransaction(
                    transaction_type="claim",
                    vault_id="beefy-eth-wbtc",
                    chain="ethereum",
                    amount=52.5,
                    gas_used=0.03,
                    tx_hash="0x789abc123def456",
                    status="pending"
                )
            ]
            
            for tx in transactions:
                session.add(tx)
            session.commit()
            logger.info(f"✓ Created {len(transactions)} transactions")
        
        # Test 5: Portfolio calculations
        logger.info("\n5. Testing portfolio calculations...")
        total_value = calculate_total_portfolio_value()
        logger.info(f"✓ Total portfolio value: ${total_value:,.2f}")
        
        chain_allocation = get_chain_allocation()
        logger.info("✓ Chain allocation:")
        for chain, percentage in chain_allocation.items():
            logger.info(f"  - {chain}: {percentage:.1f}%")
        
        # Test 6: Create portfolio snapshot
        logger.info("\n6. Testing portfolio snapshot...")
        snapshot = create_portfolio_snapshot()
        if snapshot:
            logger.info(f"✓ Created portfolio snapshot:")
            logger.info(f"  - Total value: ${snapshot.total_value_usd:,.2f}")
            logger.info(f"  - Average APY: {snapshot.average_apy:.2f}%")
            logger.info(f"  - Active chains: {snapshot.active_chains}")
            logger.info(f"  - Vault count: {snapshot.vaults_count}")
        
        # Test 7: Database statistics
        logger.info("\n7. Testing database statistics...")
        stats = db.get_table_stats()
        logger.info("✓ Table statistics:")
        for table, count in stats.items():
            logger.info(f"  - {table}: {count} records")
        
        # Test 8: Performance queries
        logger.info("\n8. Testing performance queries...")
        from src.crypto_trading.database.utils import (
            get_positions_by_performance,
            get_transaction_summary,
            get_top_performing_vaults
        )
        
        top_positions = get_positions_by_performance(limit=3)
        logger.info("✓ Top performing positions:")
        for pos, roi in top_positions:
            logger.info(f"  - {pos.vault_name}: {roi:.2f}% ROI")
        
        tx_summary = get_transaction_summary()
        logger.info(f"✓ Transaction summary: {tx_summary}")
        
        top_vaults = get_top_performing_vaults(limit=3)
        logger.info("✓ Top vaults by yield:")
        for vault in top_vaults:
            logger.info(f"  - {vault['vault_id']}: ${vault['total_earned']:.2f} earned")
        
        logger.info("\n✅ All database tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup test database
        db.close()
        test_db_path = Path("test_trading.db")
        if test_db_path.exists():
            test_db_path.unlink()
            logger.info("✓ Cleaned up test database")


if __name__ == "__main__":
    test_database_operations()