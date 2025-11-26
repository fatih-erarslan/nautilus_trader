"""
Capital Management System for Sports Betting Syndicates

This module provides comprehensive capital management functionality including:
- Pooled fund allocation and tracking
- Automated profit/loss distribution
- Member contribution management
- Withdrawal and deposit handling
- Performance-based capital allocation
- Risk-based capital reserves
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Types of capital transactions"""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    PROFIT_DISTRIBUTION = "profit_distribution"
    LOSS_ALLOCATION = "loss_allocation"
    BET_ALLOCATION = "bet_allocation"
    BET_RETURN = "bet_return"
    FEE_DEDUCTION = "fee_deduction"
    INTEREST_CREDIT = "interest_credit"


class AllocationMethod(Enum):
    """Methods for allocating capital"""
    EQUAL_SHARE = "equal_share"
    PROPORTIONAL = "proportional"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    RISK_ADJUSTED = "risk_adjusted"
    HYBRID = "hybrid"


@dataclass
class Transaction:
    """Represents a capital transaction"""
    transaction_id: str
    member_id: str
    transaction_type: TransactionType
    amount: Decimal
    timestamp: datetime
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confirmed: bool = False
    confirmation_timestamp: Optional[datetime] = None


@dataclass
class MemberCapital:
    """Tracks capital position for a syndicate member"""
    member_id: str
    total_contributed: Decimal = Decimal('0')
    current_balance: Decimal = Decimal('0')
    allocated_capital: Decimal = Decimal('0')
    available_capital: Decimal = Decimal('0')
    lifetime_profit_loss: Decimal = Decimal('0')
    performance_score: float = 1.0
    last_activity: datetime = field(default_factory=datetime.now)
    withdrawal_limit: Decimal = Decimal('0')
    pending_transactions: List[Transaction] = field(default_factory=list)


@dataclass
class CapitalPool:
    """Represents the syndicate's capital pool"""
    pool_id: str
    total_capital: Decimal = Decimal('0')
    available_capital: Decimal = Decimal('0')
    allocated_capital: Decimal = Decimal('0')
    reserved_capital: Decimal = Decimal('0')
    total_profit_loss: Decimal = Decimal('0')
    inception_date: datetime = field(default_factory=datetime.now)
    last_rebalance: datetime = field(default_factory=datetime.now)
    management_fee_rate: Decimal = Decimal('0.02')  # 2% annual
    performance_fee_rate: Decimal = Decimal('0.20')  # 20% of profits


class CapitalManager:
    """
    Advanced capital management system for sports betting syndicates
    
    Features:
    - Pooled fund management with multiple allocation strategies
    - Automated profit/loss distribution
    - Performance-based capital allocation
    - Risk-adjusted position sizing
    - Transparent transaction tracking
    - Compliance and audit trails
    """
    
    def __init__(self, pool_id: str, min_contribution: Decimal = Decimal('100')):
        self.pool_id = pool_id
        self.min_contribution = min_contribution
        self.capital_pool = CapitalPool(pool_id=pool_id)
        self.members: Dict[str, MemberCapital] = {}
        self.transactions: List[Transaction] = []
        self.allocation_method = AllocationMethod.PROPORTIONAL
        self.risk_reserve_ratio = Decimal('0.10')  # 10% risk reserve
        self.max_allocation_per_bet = Decimal('0.05')  # 5% max per bet
        
        # Performance tracking
        self.daily_returns: List[Tuple[datetime, Decimal]] = []
        self.monthly_performance: Dict[str, Decimal] = {}
        
        logger.info(f"Initialized CapitalManager for pool {pool_id}")

    async def add_member(self, member_id: str, initial_contribution: Decimal = Decimal('0')) -> bool:
        """Add a new member to the syndicate"""
        try:
            if member_id in self.members:
                logger.warning(f"Member {member_id} already exists")
                return False
            
            if initial_contribution < self.min_contribution:
                logger.error(f"Initial contribution {initial_contribution} below minimum {self.min_contribution}")
                return False
            
            member_capital = MemberCapital(
                member_id=member_id,
                total_contributed=initial_contribution,
                current_balance=initial_contribution,
                available_capital=initial_contribution
            )
            
            self.members[member_id] = member_capital
            
            if initial_contribution > 0:
                await self._process_deposit(member_id, initial_contribution, "Initial contribution")
            
            logger.info(f"Added member {member_id} with contribution {initial_contribution}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding member {member_id}: {e}")
            return False

    async def deposit_funds(self, member_id: str, amount: Decimal, description: str = "") -> str:
        """Process a member's fund deposit"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            if amount <= 0:
                raise ValueError("Deposit amount must be positive")
            
            transaction_id = await self._process_deposit(member_id, amount, description)
            
            logger.info(f"Processed deposit of {amount} for member {member_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error processing deposit for {member_id}: {e}")
            raise

    async def _process_deposit(self, member_id: str, amount: Decimal, description: str) -> str:
        """Internal method to process deposits"""
        transaction_id = f"DEP_{member_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            member_id=member_id,
            transaction_type=TransactionType.DEPOSIT,
            amount=amount,
            timestamp=datetime.now(),
            description=description,
            confirmed=True,
            confirmation_timestamp=datetime.now()
        )
        
        # Update member capital
        member = self.members[member_id]
        member.total_contributed += amount
        member.current_balance += amount
        member.available_capital += amount
        member.last_activity = datetime.now()
        
        # Update pool
        self.capital_pool.total_capital += amount
        self.capital_pool.available_capital += amount
        
        # Record transaction
        self.transactions.append(transaction)
        
        return transaction_id

    async def request_withdrawal(self, member_id: str, amount: Decimal, reason: str = "") -> str:
        """Process a member's withdrawal request"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            member = self.members[member_id]
            
            # Check withdrawal limits
            if amount > member.available_capital:
                raise ValueError(f"Insufficient available capital. Available: {member.available_capital}")
            
            if amount > member.withdrawal_limit and member.withdrawal_limit > 0:
                raise ValueError(f"Amount exceeds withdrawal limit: {member.withdrawal_limit}")
            
            # Process withdrawal
            transaction_id = await self._process_withdrawal(member_id, amount, reason)
            
            logger.info(f"Processed withdrawal of {amount} for member {member_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error processing withdrawal for {member_id}: {e}")
            raise

    async def _process_withdrawal(self, member_id: str, amount: Decimal, reason: str) -> str:
        """Internal method to process withdrawals"""
        transaction_id = f"WITH_{member_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            member_id=member_id,
            transaction_type=TransactionType.WITHDRAWAL,
            amount=amount,
            timestamp=datetime.now(),
            description=reason,
            confirmed=True,
            confirmation_timestamp=datetime.now()
        )
        
        # Update member capital
        member = self.members[member_id]
        member.current_balance -= amount
        member.available_capital -= amount
        member.last_activity = datetime.now()
        
        # Update pool
        self.capital_pool.total_capital -= amount
        self.capital_pool.available_capital -= amount
        
        # Record transaction
        self.transactions.append(transaction)
        
        return transaction_id

    async def allocate_capital_for_bet(self, bet_id: str, required_amount: Decimal, 
                                     allocation_method: Optional[AllocationMethod] = None) -> Dict[str, Decimal]:
        """Allocate capital from the pool for a specific bet"""
        try:
            method = allocation_method or self.allocation_method
            
            # Check if enough capital is available
            if required_amount > self.capital_pool.available_capital:
                raise ValueError(f"Insufficient capital. Required: {required_amount}, Available: {self.capital_pool.available_capital}")
            
            # Check maximum allocation per bet
            max_allowed = self.capital_pool.total_capital * self.max_allocation_per_bet
            if required_amount > max_allowed:
                raise ValueError(f"Bet amount exceeds maximum allocation limit: {max_allowed}")
            
            # Calculate allocations based on method
            allocations = await self._calculate_allocations(required_amount, method)
            
            # Process allocations
            for member_id, allocation in allocations.items():
                await self._allocate_member_capital(member_id, allocation, bet_id)
            
            # Update pool
            self.capital_pool.available_capital -= required_amount
            self.capital_pool.allocated_capital += required_amount
            
            logger.info(f"Allocated {required_amount} for bet {bet_id} across {len(allocations)} members")
            return allocations
            
        except Exception as e:
            logger.error(f"Error allocating capital for bet {bet_id}: {e}")
            raise

    async def _calculate_allocations(self, amount: Decimal, method: AllocationMethod) -> Dict[str, Decimal]:
        """Calculate capital allocations based on the specified method"""
        allocations = {}
        
        if method == AllocationMethod.EQUAL_SHARE:
            # Equal allocation among all members
            per_member = amount / len(self.members)
            for member_id in self.members:
                allocations[member_id] = per_member
                
        elif method == AllocationMethod.PROPORTIONAL:
            # Proportional to member's capital contribution
            total_contributions = sum(member.current_balance for member in self.members.values())
            for member_id, member in self.members.items():
                proportion = member.current_balance / total_contributions
                allocations[member_id] = amount * proportion
                
        elif method == AllocationMethod.PERFORMANCE_WEIGHTED:
            # Weighted by member performance scores
            total_performance = sum(member.performance_score for member in self.members.values())
            for member_id, member in self.members.items():
                weight = member.performance_score / total_performance
                allocations[member_id] = amount * Decimal(str(weight))
                
        elif method == AllocationMethod.RISK_ADJUSTED:
            # Adjust for member risk tolerance and capital
            total_risk_capital = sum(
                member.available_capital * Decimal(str(member.performance_score))
                for member in self.members.values()
            )
            for member_id, member in self.members.items():
                risk_weight = (member.available_capital * Decimal(str(member.performance_score))) / total_risk_capital
                allocations[member_id] = amount * risk_weight
                
        elif method == AllocationMethod.HYBRID:
            # Combination of proportional and performance-weighted
            prop_allocations = await self._calculate_allocations(amount * Decimal('0.7'), AllocationMethod.PROPORTIONAL)
            perf_allocations = await self._calculate_allocations(amount * Decimal('0.3'), AllocationMethod.PERFORMANCE_WEIGHTED)
            
            for member_id in self.members:
                allocations[member_id] = prop_allocations.get(member_id, Decimal('0')) + perf_allocations.get(member_id, Decimal('0'))
        
        return allocations

    async def _allocate_member_capital(self, member_id: str, amount: Decimal, bet_id: str):
        """Allocate capital from a specific member"""
        member = self.members[member_id]
        
        if amount > member.available_capital:
            raise ValueError(f"Member {member_id} has insufficient available capital")
        
        # Create allocation transaction
        transaction_id = f"ALLOC_{member_id}_{bet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            member_id=member_id,
            transaction_type=TransactionType.BET_ALLOCATION,
            amount=amount,
            timestamp=datetime.now(),
            description=f"Capital allocation for bet {bet_id}",
            metadata={"bet_id": bet_id},
            confirmed=True,
            confirmation_timestamp=datetime.now()
        )
        
        # Update member capital
        member.available_capital -= amount
        member.allocated_capital += amount
        member.last_activity = datetime.now()
        
        # Record transaction
        self.transactions.append(transaction)

    async def process_bet_settlement(self, bet_id: str, result: str, payout: Decimal = Decimal('0')):
        """Process the settlement of a bet and distribute returns"""
        try:
            # Find all allocations for this bet
            bet_allocations = [
                t for t in self.transactions 
                if t.transaction_type == TransactionType.BET_ALLOCATION and 
                t.metadata.get('bet_id') == bet_id
            ]
            
            if not bet_allocations:
                raise ValueError(f"No allocations found for bet {bet_id}")
            
            total_allocated = sum(t.amount for t in bet_allocations)
            
            # Calculate profit/loss
            net_result = payout - total_allocated
            
            # Distribute returns proportionally
            for allocation in bet_allocations:
                member_id = allocation.member_id
                allocated_amount = allocation.amount
                
                # Calculate member's share of the payout
                proportion = allocated_amount / total_allocated
                member_payout = payout * proportion
                
                # Process return transaction
                await self._process_bet_return(member_id, bet_id, allocated_amount, member_payout)
            
            # Update pool statistics
            self.capital_pool.allocated_capital -= total_allocated
            self.capital_pool.available_capital += payout
            self.capital_pool.total_profit_loss += net_result
            
            # Record daily performance
            self.daily_returns.append((datetime.now(), net_result))
            
            logger.info(f"Settled bet {bet_id}: Result={result}, Payout={payout}, P&L={net_result}")
            
        except Exception as e:
            logger.error(f"Error settling bet {bet_id}: {e}")
            raise

    async def _process_bet_return(self, member_id: str, bet_id: str, allocated_amount: Decimal, payout: Decimal):
        """Process the return of capital from a settled bet"""
        transaction_id = f"RETURN_{member_id}_{bet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        net_result = payout - allocated_amount
        
        transaction = Transaction(
            transaction_id=transaction_id,
            member_id=member_id,
            transaction_type=TransactionType.BET_RETURN,
            amount=payout,
            timestamp=datetime.now(),
            description=f"Return from bet {bet_id} (P&L: {net_result})",
            metadata={"bet_id": bet_id, "allocated_amount": str(allocated_amount), "net_result": str(net_result)},
            confirmed=True,
            confirmation_timestamp=datetime.now()
        )
        
        # Update member capital
        member = self.members[member_id]
        member.allocated_capital -= allocated_amount
        member.available_capital += payout
        member.current_balance += net_result
        member.lifetime_profit_loss += net_result
        member.last_activity = datetime.now()
        
        # Update performance score based on result
        if net_result > 0:
            member.performance_score = min(2.0, member.performance_score * 1.01)  # Small positive adjustment
        elif net_result < 0:
            member.performance_score = max(0.5, member.performance_score * 0.99)  # Small negative adjustment
        
        # Record transaction
        self.transactions.append(transaction)

    async def distribute_profits(self, profit_amount: Decimal, distribution_method: AllocationMethod = AllocationMethod.PROPORTIONAL):
        """Distribute profits to syndicate members"""
        try:
            if profit_amount <= 0:
                raise ValueError("Profit amount must be positive")
            
            # Calculate management and performance fees
            management_fee = self.capital_pool.total_capital * self.capital_pool.management_fee_rate / Decimal('365')  # Daily fee
            performance_fee = profit_amount * self.capital_pool.performance_fee_rate
            total_fees = management_fee + performance_fee
            
            # Net profit after fees
            net_profit = profit_amount - total_fees
            
            # Calculate distribution
            distributions = await self._calculate_allocations(net_profit, distribution_method)
            
            # Process distributions
            for member_id, distribution in distributions.items():
                await self._process_profit_distribution(member_id, distribution)
            
            logger.info(f"Distributed profit: Total={profit_amount}, Net={net_profit}, Fees={total_fees}")
            
        except Exception as e:
            logger.error(f"Error distributing profits: {e}")
            raise

    async def _process_profit_distribution(self, member_id: str, amount: Decimal):
        """Process profit distribution to a member"""
        transaction_id = f"PROFIT_{member_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            member_id=member_id,
            transaction_type=TransactionType.PROFIT_DISTRIBUTION,
            amount=amount,
            timestamp=datetime.now(),
            description="Profit distribution",
            confirmed=True,
            confirmation_timestamp=datetime.now()
        )
        
        # Update member capital
        member = self.members[member_id]
        member.current_balance += amount
        member.available_capital += amount
        member.lifetime_profit_loss += amount
        member.last_activity = datetime.now()
        
        # Record transaction
        self.transactions.append(transaction)

    def get_member_summary(self, member_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a member's capital position"""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")
        
        member = self.members[member_id]
        
        # Calculate member's share of total pool
        pool_share = (member.current_balance / self.capital_pool.total_capital * 100) if self.capital_pool.total_capital > 0 else 0
        
        # Calculate ROI
        roi = (member.lifetime_profit_loss / member.total_contributed * 100) if member.total_contributed > 0 else 0
        
        return {
            "member_id": member_id,
            "total_contributed": str(member.total_contributed),
            "current_balance": str(member.current_balance),
            "available_capital": str(member.available_capital),
            "allocated_capital": str(member.allocated_capital),
            "lifetime_profit_loss": str(member.lifetime_profit_loss),
            "performance_score": member.performance_score,
            "pool_share_percentage": float(pool_share),
            "roi_percentage": float(roi),
            "last_activity": member.last_activity.isoformat(),
            "withdrawal_limit": str(member.withdrawal_limit)
        }

    def get_pool_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the capital pool"""
        # Calculate pool performance metrics
        total_roi = (self.capital_pool.total_profit_loss / self.capital_pool.total_capital * 100) if self.capital_pool.total_capital > 0 else 0
        
        # Calculate recent performance (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_returns = [r[1] for r in self.daily_returns if r[0] >= thirty_days_ago]
        recent_performance = sum(recent_returns) if recent_returns else Decimal('0')
        
        return {
            "pool_id": self.pool_id,
            "total_capital": str(self.capital_pool.total_capital),
            "available_capital": str(self.capital_pool.available_capital),
            "allocated_capital": str(self.capital_pool.allocated_capital),
            "reserved_capital": str(self.capital_pool.reserved_capital),
            "total_profit_loss": str(self.capital_pool.total_profit_loss),
            "total_roi_percentage": float(total_roi),
            "recent_30d_performance": str(recent_performance),
            "member_count": len(self.members),
            "inception_date": self.capital_pool.inception_date.isoformat(),
            "last_rebalance": self.capital_pool.last_rebalance.isoformat(),
            "management_fee_rate": str(self.capital_pool.management_fee_rate),
            "performance_fee_rate": str(self.capital_pool.performance_fee_rate),
            "allocation_method": self.allocation_method.value,
            "risk_reserve_ratio": str(self.risk_reserve_ratio),
            "max_allocation_per_bet": str(self.max_allocation_per_bet)
        }

    def get_transaction_history(self, member_id: Optional[str] = None, 
                              transaction_type: Optional[TransactionType] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history with optional filtering"""
        filtered_transactions = self.transactions
        
        if member_id:
            filtered_transactions = [t for t in filtered_transactions if t.member_id == member_id]
        
        if transaction_type:
            filtered_transactions = [t for t in filtered_transactions if t.transaction_type == transaction_type]
        
        # Sort by timestamp (most recent first) and limit
        filtered_transactions.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_transactions = filtered_transactions[:limit]
        
        return [
            {
                "transaction_id": t.transaction_id,
                "member_id": t.member_id,
                "transaction_type": t.transaction_type.value,
                "amount": str(t.amount),
                "timestamp": t.timestamp.isoformat(),
                "description": t.description,
                "metadata": t.metadata,
                "confirmed": t.confirmed,
                "confirmation_timestamp": t.confirmation_timestamp.isoformat() if t.confirmation_timestamp else None
            }
            for t in filtered_transactions
        ]

    async def rebalance_pool(self):
        """Rebalance the capital pool and update allocations"""
        try:
            # Update risk reserves
            target_reserve = self.capital_pool.total_capital * self.risk_reserve_ratio
            self.capital_pool.reserved_capital = target_reserve
            
            # Adjust available capital
            committed_capital = self.capital_pool.allocated_capital + self.capital_pool.reserved_capital
            self.capital_pool.available_capital = self.capital_pool.total_capital - committed_capital
            
            # Update withdrawal limits for members
            for member in self.members.values():
                # Set withdrawal limit to 80% of available capital to maintain liquidity
                member.withdrawal_limit = member.available_capital * Decimal('0.8')
            
            self.capital_pool.last_rebalance = datetime.now()
            
            logger.info("Pool rebalancing completed")
            
        except Exception as e:
            logger.error(f"Error during pool rebalancing: {e}")
            raise

    def export_capital_report(self) -> Dict[str, Any]:
        """Export comprehensive capital management report"""
        return {
            "report_timestamp": datetime.now().isoformat(),
            "pool_summary": self.get_pool_summary(),
            "member_summaries": {
                member_id: self.get_member_summary(member_id)
                for member_id in self.members.keys()
            },
            "recent_transactions": self.get_transaction_history(limit=50),
            "performance_metrics": {
                "daily_returns_count": len(self.daily_returns),
                "total_transactions": len(self.transactions),
                "active_members": len([m for m in self.members.values() if m.current_balance > 0]),
                "capital_utilization": float(
                    (self.capital_pool.allocated_capital / self.capital_pool.total_capital * 100)
                    if self.capital_pool.total_capital > 0 else 0
                )
            }
        }