"""SQLAlchemy Models for Crypto Trading Database

Defines all database models with proper relationships and constraints.
"""

from datetime import datetime
from typing import Optional, List
import json
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint,
    event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


class VaultPosition(Base):
    """Model for tracking individual vault positions"""
    __tablename__ = 'vault_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vault_id = Column(String(100), nullable=False, index=True)
    vault_name = Column(String(200), nullable=False)
    chain = Column(String(50), nullable=False, index=True)
    amount_deposited = Column(Float, nullable=False)
    shares_owned = Column(Float, nullable=False)
    current_value = Column(Float, default=0)
    entry_price = Column(Float, nullable=False)
    entry_apy = Column(Float, nullable=False)
    status = Column(String(20), default='active', index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    yield_history = relationship('YieldHistory', back_populates='position', cascade='all, delete-orphan')
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('active', 'closed', 'pending')", name='check_status'),
        CheckConstraint('amount_deposited > 0', name='check_amount_positive'),
        CheckConstraint('shares_owned >= 0', name='check_shares_non_negative'),
        Index('idx_vault_chain', 'vault_id', 'chain'),
    )
    
    @validates('chain')
    def validate_chain(self, key, chain):
        """Validate blockchain network"""
        valid_chains = ['ethereum', 'bsc', 'polygon', 'avalanche', 'fantom', 'arbitrum', 'optimism']
        if chain.lower() not in valid_chains:
            raise ValueError(f"Invalid chain: {chain}. Must be one of {valid_chains}")
        return chain.lower()
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss"""
        return self.current_value - self.amount_deposited
    
    @property
    def roi_percentage(self) -> float:
        """Calculate return on investment percentage"""
        if self.amount_deposited == 0:
            return 0
        return ((self.current_value - self.amount_deposited) / self.amount_deposited) * 100
    
    def __repr__(self):
        return f"<VaultPosition(id={self.id}, vault={self.vault_name}, chain={self.chain}, value=${self.current_value:.2f})>"


class YieldHistory(Base):
    """Model for tracking yield history over time"""
    __tablename__ = 'yield_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vault_id = Column(String(100), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('vault_positions.id', ondelete='CASCADE'), nullable=False, index=True)
    earned_amount = Column(Float, nullable=False)
    apy_snapshot = Column(Float, nullable=False)
    tvl_snapshot = Column(Float)
    price_per_share = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    position = relationship('VaultPosition', back_populates='yield_history')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('earned_amount >= 0', name='check_earned_non_negative'),
        CheckConstraint('apy_snapshot >= 0', name='check_apy_non_negative'),
        CheckConstraint('price_per_share > 0', name='check_price_positive'),
        Index('idx_yield_composite', 'vault_id', 'recorded_at'),
    )
    
    def __repr__(self):
        return f"<YieldHistory(id={self.id}, vault={self.vault_id}, earned=${self.earned_amount:.4f}, apy={self.apy_snapshot:.2f}%)>"


class CryptoTransaction(Base):
    """Model for tracking blockchain transactions"""
    __tablename__ = 'crypto_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_type = Column(String(20), nullable=False)
    vault_id = Column(String(100), nullable=False, index=True)
    chain = Column(String(50), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    gas_used = Column(Float)
    tx_hash = Column(String(100), unique=True, index=True)
    status = Column(String(20), default='pending', index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("transaction_type IN ('deposit', 'withdraw', 'claim', 'compound')", name='check_tx_type'),
        CheckConstraint("status IN ('pending', 'confirmed', 'failed')", name='check_tx_status'),
        CheckConstraint('amount > 0', name='check_tx_amount_positive'),
    )
    
    @validates('chain')
    def validate_chain(self, key, chain):
        """Validate blockchain network"""
        valid_chains = ['ethereum', 'bsc', 'polygon', 'avalanche', 'fantom', 'arbitrum', 'optimism']
        if chain.lower() not in valid_chains:
            raise ValueError(f"Invalid chain: {chain}")
        return chain.lower()
    
    @property
    def total_cost(self) -> float:
        """Calculate total transaction cost including gas"""
        return self.amount + (self.gas_used or 0)
    
    def __repr__(self):
        return f"<CryptoTransaction(id={self.id}, type={self.transaction_type}, vault={self.vault_id}, status={self.status})>"


class PortfolioSummary(Base):
    """Model for portfolio aggregate metrics"""
    __tablename__ = 'portfolio_summary'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    total_value_usd = Column(Float, nullable=False)
    total_yield_earned = Column(Float, nullable=False)
    average_apy = Column(Float, nullable=False)
    chains_active = Column(Text, nullable=False)  # JSON array
    vaults_count = Column(Integer, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_value_usd >= 0', name='check_value_non_negative'),
        CheckConstraint('total_yield_earned >= 0', name='check_yield_non_negative'),
        CheckConstraint('average_apy >= 0', name='check_apy_non_negative'),
        CheckConstraint('vaults_count >= 0', name='check_vaults_non_negative'),
    )
    
    @property
    def active_chains(self) -> List[str]:
        """Get list of active chains from JSON"""
        try:
            return json.loads(self.chains_active)
        except:
            return []
    
    @active_chains.setter
    def active_chains(self, chains: List[str]):
        """Set active chains as JSON"""
        self.chains_active = json.dumps(chains)
    
    def __repr__(self):
        return f"<PortfolioSummary(id={self.id}, value=${self.total_value_usd:.2f}, apy={self.average_apy:.2f}%, vaults={self.vaults_count})>"


# Event listeners for automatic timestamp updates
@event.listens_for(VaultPosition, 'before_update')
def update_vault_position_timestamp(mapper, connection, target):
    """Automatically update the updated_at timestamp"""
    target.updated_at = datetime.utcnow()