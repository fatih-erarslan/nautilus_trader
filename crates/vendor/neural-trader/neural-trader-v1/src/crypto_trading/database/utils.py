"""Database Utility Functions for Crypto Trading

Provides high-level query functions and batch operations for the trading database.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session
from .models import VaultPosition, YieldHistory, CryptoTransaction, PortfolioSummary
from .connection import get_db_session

logger = logging.getLogger(__name__)


# Vault Position Utilities
def get_active_positions(
    chain: Optional[str] = None,
    vault_id: Optional[str] = None
) -> List[VaultPosition]:
    """
    Get all active vault positions with optional filtering
    
    Args:
        chain: Filter by blockchain (optional)
        vault_id: Filter by specific vault (optional)
        
    Returns:
        List of active VaultPosition objects
    """
    with get_db_session() as session:
        query = session.query(VaultPosition).filter(VaultPosition.status == 'active')
        
        if chain:
            query = query.filter(VaultPosition.chain == chain.lower())
        if vault_id:
            query = query.filter(VaultPosition.vault_id == vault_id)
        
        return query.all()


def calculate_total_portfolio_value() -> float:
    """Calculate total value of all active positions"""
    with get_db_session() as session:
        result = session.query(
            func.sum(VaultPosition.current_value)
        ).filter(
            VaultPosition.status == 'active'
        ).scalar()
        
        return result or 0.0


def get_positions_by_performance(
    limit: int = 10,
    order: str = 'desc'
) -> List[Tuple[VaultPosition, float]]:
    """
    Get positions ordered by ROI performance
    
    Args:
        limit: Number of positions to return
        order: 'desc' for best performers, 'asc' for worst
        
    Returns:
        List of tuples (position, roi_percentage)
    """
    with get_db_session() as session:
        positions = session.query(VaultPosition).filter(
            VaultPosition.status == 'active'
        ).all()
        
        # Calculate ROI for each position
        position_rois = [
            (pos, pos.roi_percentage) for pos in positions
        ]
        
        # Sort by ROI
        reverse = (order == 'desc')
        position_rois.sort(key=lambda x: x[1], reverse=reverse)
        
        return position_rois[:limit]


def update_position_value(position_id: int, new_value: float) -> bool:
    """
    Update the current value of a position
    
    Args:
        position_id: ID of the position to update
        new_value: New current value in USD
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_session() as session:
            position = session.query(VaultPosition).get(position_id)
            if position:
                position.current_value = new_value
                session.commit()
                return True
            return False
    except Exception as e:
        logger.error(f"Error updating position value: {e}")
        return False


# Yield History Utilities
def get_yield_history_by_vault(
    vault_id: str,
    days: int = 30
) -> List[YieldHistory]:
    """
    Get yield history for a specific vault
    
    Args:
        vault_id: Vault identifier
        days: Number of days of history to retrieve
        
    Returns:
        List of YieldHistory records
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as session:
        return session.query(YieldHistory).filter(
            and_(
                YieldHistory.vault_id == vault_id,
                YieldHistory.recorded_at >= cutoff_date
            )
        ).order_by(desc(YieldHistory.recorded_at)).all()


def calculate_average_apy(
    vault_id: Optional[str] = None,
    days: int = 7
) -> float:
    """
    Calculate average APY over a period
    
    Args:
        vault_id: Specific vault or None for all vaults
        days: Period to calculate average over
        
    Returns:
        Average APY as percentage
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as session:
        query = session.query(func.avg(YieldHistory.apy_snapshot))
        query = query.filter(YieldHistory.recorded_at >= cutoff_date)
        
        if vault_id:
            query = query.filter(YieldHistory.vault_id == vault_id)
        
        result = query.scalar()
        return result or 0.0


def batch_insert_yield_history(
    yield_records: List[Dict[str, Any]]
) -> int:
    """
    Batch insert yield history records
    
    Args:
        yield_records: List of dictionaries with yield data
        
    Returns:
        Number of records inserted
    """
    try:
        with get_db_session() as session:
            yield_objects = [
                YieldHistory(**record) for record in yield_records
            ]
            session.bulk_save_objects(yield_objects)
            session.commit()
            return len(yield_objects)
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        return 0


# Transaction Utilities
def get_recent_transactions(
    limit: int = 50,
    status: Optional[str] = None
) -> List[CryptoTransaction]:
    """
    Get recent crypto transactions
    
    Args:
        limit: Maximum number of transactions
        status: Filter by status (pending, confirmed, failed)
        
    Returns:
        List of CryptoTransaction objects
    """
    with get_db_session() as session:
        query = session.query(CryptoTransaction)
        
        if status:
            query = query.filter(CryptoTransaction.status == status)
        
        return query.order_by(
            desc(CryptoTransaction.created_at)
        ).limit(limit).all()


def get_transaction_summary() -> Dict[str, Any]:
    """Get summary statistics for transactions"""
    with get_db_session() as session:
        # Count by status
        status_counts = session.query(
            CryptoTransaction.status,
            func.count(CryptoTransaction.id)
        ).group_by(CryptoTransaction.status).all()
        
        # Count by type
        type_counts = session.query(
            CryptoTransaction.transaction_type,
            func.count(CryptoTransaction.id)
        ).group_by(CryptoTransaction.transaction_type).all()
        
        # Total gas spent
        total_gas = session.query(
            func.sum(CryptoTransaction.gas_used)
        ).scalar() or 0
        
        return {
            'status_breakdown': dict(status_counts),
            'type_breakdown': dict(type_counts),
            'total_gas_spent': total_gas,
            'total_transactions': sum(count for _, count in status_counts)
        }


def update_transaction_status(
    tx_hash: str,
    new_status: str
) -> bool:
    """
    Update transaction status by hash
    
    Args:
        tx_hash: Transaction hash
        new_status: New status (confirmed, failed)
        
    Returns:
        True if updated, False otherwise
    """
    try:
        with get_db_session() as session:
            tx = session.query(CryptoTransaction).filter(
                CryptoTransaction.tx_hash == tx_hash
            ).first()
            
            if tx:
                tx.status = new_status
                session.commit()
                return True
            return False
    except Exception as e:
        logger.error(f"Error updating transaction: {e}")
        return False


# Portfolio Summary Utilities
def create_portfolio_snapshot() -> PortfolioSummary:
    """Create a new portfolio summary snapshot"""
    with get_db_session() as session:
        # Get active positions
        active_positions = get_active_positions()
        
        if not active_positions:
            logger.warning("No active positions for portfolio snapshot")
            return None
        
        # Calculate metrics
        total_value = sum(pos.current_value for pos in active_positions)
        
        # Get total yield earned
        total_yield = session.query(
            func.sum(YieldHistory.earned_amount)
        ).scalar() or 0.0
        
        # Calculate average APY from active positions
        avg_apy = calculate_average_apy()
        
        # Get unique chains
        chains = list(set(pos.chain for pos in active_positions))
        
        # Create snapshot
        snapshot = PortfolioSummary(
            total_value_usd=total_value,
            total_yield_earned=total_yield,
            average_apy=avg_apy,
            vaults_count=len(active_positions)
        )
        snapshot.active_chains = chains
        
        session.add(snapshot)
        session.commit()
        
        return snapshot


def get_portfolio_history(
    days: int = 30
) -> List[PortfolioSummary]:
    """Get portfolio history for the specified period"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as session:
        return session.query(PortfolioSummary).filter(
            PortfolioSummary.recorded_at >= cutoff_date
        ).order_by(desc(PortfolioSummary.recorded_at)).all()


def get_portfolio_performance_metrics() -> Dict[str, Any]:
    """Calculate various portfolio performance metrics"""
    with get_db_session() as session:
        # Get current and 30-day old snapshots
        current = session.query(PortfolioSummary).order_by(
            desc(PortfolioSummary.recorded_at)
        ).first()
        
        month_ago = datetime.utcnow() - timedelta(days=30)
        old_snapshot = session.query(PortfolioSummary).filter(
            PortfolioSummary.recorded_at <= month_ago
        ).order_by(desc(PortfolioSummary.recorded_at)).first()
        
        if not current:
            return {}
        
        metrics = {
            'current_value': current.total_value_usd,
            'total_yield': current.total_yield_earned,
            'current_apy': current.average_apy,
            'active_vaults': current.vaults_count,
            'active_chains': current.active_chains
        }
        
        # Calculate 30-day changes if we have historical data
        if old_snapshot:
            metrics['value_change_30d'] = current.total_value_usd - old_snapshot.total_value_usd
            metrics['value_change_30d_pct'] = (
                (current.total_value_usd - old_snapshot.total_value_usd) / 
                old_snapshot.total_value_usd * 100
            ) if old_snapshot.total_value_usd > 0 else 0
            metrics['yield_earned_30d'] = current.total_yield_earned - old_snapshot.total_yield_earned
        
        return metrics


# Chain-specific utilities
def get_chain_allocation() -> Dict[str, float]:
    """Get portfolio allocation by blockchain"""
    with get_db_session() as session:
        results = session.query(
            VaultPosition.chain,
            func.sum(VaultPosition.current_value)
        ).filter(
            VaultPosition.status == 'active'
        ).group_by(VaultPosition.chain).all()
        
        total_value = sum(value for _, value in results)
        
        if total_value == 0:
            return {}
        
        return {
            chain: (value / total_value * 100)
            for chain, value in results
        }


def get_top_performing_vaults(limit: int = 5) -> List[Dict[str, Any]]:
    """Get top performing vaults by total yield"""
    with get_db_session() as session:
        results = session.query(
            YieldHistory.vault_id,
            func.sum(YieldHistory.earned_amount).label('total_earned'),
            func.avg(YieldHistory.apy_snapshot).label('avg_apy')
        ).group_by(
            YieldHistory.vault_id
        ).order_by(
            desc('total_earned')
        ).limit(limit).all()
        
        return [
            {
                'vault_id': vault_id,
                'total_earned': total_earned,
                'average_apy': avg_apy
            }
            for vault_id, total_earned, avg_apy in results
        ]


# Maintenance utilities
def cleanup_old_yield_history(days_to_keep: int = 90) -> int:
    """
    Remove yield history older than specified days
    
    Args:
        days_to_keep: Number of days of history to keep
        
    Returns:
        Number of records deleted
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    try:
        with get_db_session() as session:
            deleted = session.query(YieldHistory).filter(
                YieldHistory.recorded_at < cutoff_date
            ).delete()
            session.commit()
            logger.info(f"Cleaned up {deleted} old yield history records")
            return deleted
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0


def validate_database_integrity() -> Dict[str, List[str]]:
    """Check database integrity and return any issues found"""
    issues = {
        'orphaned_yields': [],
        'invalid_values': [],
        'missing_updates': []
    }
    
    with get_db_session() as session:
        # Check for orphaned yield history
        orphaned = session.query(YieldHistory).filter(
            ~YieldHistory.position_id.in_(
                session.query(VaultPosition.id)
            )
        ).all()
        
        issues['orphaned_yields'] = [
            f"Yield record {y.id} references non-existent position {y.position_id}"
            for y in orphaned
        ]
        
        # Check for negative values
        negative_positions = session.query(VaultPosition).filter(
            or_(
                VaultPosition.current_value < 0,
                VaultPosition.amount_deposited < 0
            )
        ).all()
        
        issues['invalid_values'] = [
            f"Position {p.id} has negative values"
            for p in negative_positions
        ]
        
    return issues