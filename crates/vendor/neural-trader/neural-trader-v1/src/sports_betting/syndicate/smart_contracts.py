"""
Smart Contract Integration for Sports Betting Syndicates

This module provides blockchain-based smart contract functionality including:
- Automated governance rule enforcement
- Transparent profit and loss distribution
- Dispute resolution mechanisms
- Escrow services for pooled funds
- Multi-signature transaction approval
- Compliance and audit trail automation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import json
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

logger = logging.getLogger(__name__)


class ContractType(Enum):
    """Types of smart contracts"""
    GOVERNANCE = "governance"
    CAPITAL_POOL = "capital_pool"
    PROFIT_DISTRIBUTION = "profit_distribution"
    ESCROW = "escrow"
    MULTI_SIG = "multi_sig"
    DISPUTE_RESOLUTION = "dispute_resolution"
    COMPLIANCE = "compliance"
    MEMBER_AGREEMENT = "member_agreement"


class TransactionStatus(Enum):
    """Status of blockchain transactions"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISPUTED = "disputed"


class DisputeStatus(Enum):
    """Status of disputes"""
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    MEDIATION = "mediation"
    ARBITRATION = "arbitration"
    RESOLVED = "resolved"
    CLOSED = "closed"


class GovernanceAction(Enum):
    """Types of governance actions that can be automated"""
    MEMBER_ADMISSION = "member_admission"
    MEMBER_REMOVAL = "member_removal"
    PROFIT_DISTRIBUTION = "profit_distribution"
    CAPITAL_REALLOCATION = "capital_reallocation"
    PARAMETER_UPDATE = "parameter_update"
    EMERGENCY_ACTION = "emergency_action"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class SmartContract:
    """Represents a smart contract on the blockchain"""
    contract_id: str
    contract_type: ContractType
    contract_address: str
    bytecode_hash: str
    
    # Contract metadata
    name: str
    description: str
    version: str
    created_by: str
    created_at: datetime
    
    # Contract state
    is_active: bool = True
    is_verified: bool = False
    gas_limit: int = 500000
    
    # Participants
    authorized_signers: Set[str] = field(default_factory=set)
    required_signatures: int = 1
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, Any] = field(default_factory=dict)
    
    # Execution history
    execution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Transaction:
    """Represents a blockchain transaction"""
    transaction_id: str
    contract_id: str
    initiator_id: str
    function_name: str
    parameters: Dict[str, Any]
    
    # Transaction details
    gas_price: int
    gas_limit: int
    value: Decimal = Decimal('0')
    
    # Status tracking
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    
    # Multi-signature
    required_signatures: int = 1
    signatures: Dict[str, str] = field(default_factory=dict)  # signer_id -> signature
    
    # Results
    execution_result: Optional[Dict[str, Any]] = None
    gas_used: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class Dispute:
    """Represents a dispute resolution case"""
    dispute_id: str
    initiator_id: str
    respondent_id: Optional[str]
    dispute_type: str
    
    # Dispute details
    title: str
    description: str
    amount_disputed: Decimal
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status and timeline
    status: DisputeStatus = DisputeStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolution_deadline: Optional[datetime] = None
    
    # Resolution process
    mediator_id: Optional[str] = None
    arbitrator_id: Optional[str] = None
    resolution: Optional[str] = None
    compensation_amount: Decimal = Decimal('0')
    
    # Voting (if community resolution)
    votes: Dict[str, str] = field(default_factory=dict)  # member_id -> vote
    community_resolution: bool = False


@dataclass
class EscrowAccount:
    """Represents an escrow account for secure fund holding"""
    escrow_id: str
    participants: Set[str]
    total_amount: Decimal
    
    # Escrow terms
    release_conditions: List[Dict[str, Any]]
    expiry_date: Optional[datetime] = None
    auto_release: bool = False
    
    # Status
    is_active: bool = True
    funds_released: bool = False
    released_at: Optional[datetime] = None
    
    # Individual contributions
    contributions: Dict[str, Decimal] = field(default_factory=dict)
    pending_releases: Dict[str, Decimal] = field(default_factory=dict)


class SmartContractManager:
    """
    Advanced smart contract integration for syndicate automation
    
    Features:
    - Multi-signature transaction approval
    - Automated governance rule enforcement
    - Transparent profit/loss distribution
    - Dispute resolution mechanisms
    - Secure escrow services
    - Compliance automation
    - Audit trail maintenance
    """
    
    def __init__(self, syndicate_id: str, blockchain_network: str = "ethereum"):
        self.syndicate_id = syndicate_id
        self.blockchain_network = blockchain_network
        
        # Core components
        self.contracts: Dict[str, SmartContract] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.disputes: Dict[str, Dispute] = {}
        self.escrow_accounts: Dict[str, EscrowAccount] = {}
        
        # Cryptographic keys (in production, use proper key management)
        self.signing_keys: Dict[str, Any] = {}
        self.contract_addresses: Dict[ContractType, str] = {}
        
        # Governance automation
        self.governance_rules: Dict[GovernanceAction, Dict[str, Any]] = {}
        self.automated_actions: List[Dict[str, Any]] = []
        
        # Initialize default contracts
        self._initialize_default_contracts()
        
        logger.info(f"Initialized SmartContractManager for syndicate {syndicate_id}")

    def _initialize_default_contracts(self):
        """Initialize default smart contracts for the syndicate"""
        default_contracts = [
            (ContractType.GOVERNANCE, "Syndicate Governance", "Automated governance and voting"),
            (ContractType.CAPITAL_POOL, "Capital Pool Management", "Pooled fund management and allocation"),
            (ContractType.PROFIT_DISTRIBUTION, "Profit Distribution", "Automated profit/loss distribution"),
            (ContractType.ESCROW, "Escrow Services", "Secure fund holding and release"),
            (ContractType.MULTI_SIG, "Multi-Signature Wallet", "Multi-signature transaction approval")
        ]
        
        for contract_type, name, description in default_contracts:
            contract_id = f"CONTRACT_{contract_type.value.upper()}_{self.syndicate_id}"
            
            # Generate mock contract address (in production, deploy actual contracts)
            contract_address = f"0x{hashlib.sha256(f'{contract_id}_{datetime.now()}'.encode()).hexdigest()[:40]}"
            bytecode_hash = hashlib.sha256(f"bytecode_{contract_id}".encode()).hexdigest()
            
            contract = SmartContract(
                contract_id=contract_id,
                contract_type=contract_type,
                contract_address=contract_address,
                bytecode_hash=bytecode_hash,
                name=name,
                description=description,
                version="1.0.0",
                created_by="system",
                created_at=datetime.now(),
                required_signatures=2,  # Default to requiring 2 signatures
                parameters=self._get_default_contract_parameters(contract_type)
            )
            
            self.contracts[contract_id] = contract
            self.contract_addresses[contract_type] = contract_address

    def _get_default_contract_parameters(self, contract_type: ContractType) -> Dict[str, Any]:
        """Get default parameters for a contract type"""
        if contract_type == ContractType.GOVERNANCE:
            return {
                "voting_period_hours": 72,
                "quorum_percentage": 51,
                "majority_percentage": 60,
                "proposal_threshold": 1000  # Minimum stake to create proposal
            }
        elif contract_type == ContractType.CAPITAL_POOL:
            return {
                "management_fee_rate": 0.02,
                "performance_fee_rate": 0.20,
                "withdrawal_delay_hours": 24,
                "max_allocation_percentage": 5
            }
        elif contract_type == ContractType.PROFIT_DISTRIBUTION:
            return {
                "distribution_frequency_days": 30,
                "minimum_profit_threshold": 100,
                "gas_reserve_percentage": 1
            }
        elif contract_type == ContractType.ESCROW:
            return {
                "default_expiry_days": 30,
                "dispute_period_days": 7,
                "arbitration_fee_percentage": 2
            }
        elif contract_type == ContractType.MULTI_SIG:
            return {
                "default_required_signatures": 2,
                "signature_timeout_hours": 48,
                "emergency_override_hours": 24
            }
        else:
            return {}

    async def deploy_contract(self, deployer_id: str, contract_type: ContractType,
                            name: str, parameters: Dict[str, Any]) -> str:
        """Deploy a new smart contract"""
        try:
            contract_id = f"CONTRACT_{contract_type.value.upper()}_{len(self.contracts)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate contract address (mock deployment)
            contract_address = f"0x{hashlib.sha256(f'{contract_id}_{datetime.now()}'.encode()).hexdigest()[:40]}"
            bytecode_hash = hashlib.sha256(f"bytecode_{contract_id}".encode()).hexdigest()
            
            contract = SmartContract(
                contract_id=contract_id,
                contract_type=contract_type,
                contract_address=contract_address,
                bytecode_hash=bytecode_hash,
                name=name,
                description=f"Custom {contract_type.value} contract",
                version="1.0.0",
                created_by=deployer_id,
                created_at=datetime.now(),
                parameters=parameters
            )
            
            self.contracts[contract_id] = contract
            
            # Create deployment transaction
            tx_id = await self._create_transaction(
                contract_id=contract_id,
                initiator_id=deployer_id,
                function_name="deploy",
                parameters=parameters
            )
            
            logger.info(f"Deployed contract {name} ({contract_id}) at {contract_address}")
            return contract_id
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            raise

    async def execute_governance_action(self, action: GovernanceAction, 
                                      parameters: Dict[str, Any],
                                      initiator_id: str) -> str:
        """Execute an automated governance action"""
        try:
            governance_contract_id = next(
                (c.contract_id for c in self.contracts.values() 
                 if c.contract_type == ContractType.GOVERNANCE),
                None
            )
            
            if not governance_contract_id:
                raise ValueError("Governance contract not found")
            
            # Check if action is allowed
            if not await self._validate_governance_action(action, parameters, initiator_id):
                raise ValueError(f"Governance action {action.value} not authorized")
            
            # Create transaction
            tx_id = await self._create_transaction(
                contract_id=governance_contract_id,
                initiator_id=initiator_id,
                function_name=f"execute_{action.value}",
                parameters=parameters
            )
            
            # Execute action immediately if authorized, otherwise require signatures
            contract = self.contracts[governance_contract_id]
            if len(contract.authorized_signers) <= 1:
                await self._execute_transaction(tx_id)
            
            logger.info(f"Initiated governance action {action.value} with transaction {tx_id}")
            return tx_id
            
        except Exception as e:
            logger.error(f"Error executing governance action: {e}")
            raise

    async def create_escrow(self, creator_id: str, participants: Set[str],
                          amounts: Dict[str, Decimal], release_conditions: List[Dict[str, Any]],
                          expiry_days: int = 30) -> str:
        """Create an escrow account for secure fund holding"""
        try:
            escrow_id = f"ESCROW_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.escrow_accounts)}"
            
            total_amount = sum(amounts.values())
            expiry_date = datetime.now() + timedelta(days=expiry_days)
            
            escrow = EscrowAccount(
                escrow_id=escrow_id,
                participants=participants,
                total_amount=total_amount,
                release_conditions=release_conditions,
                expiry_date=expiry_date,
                contributions=amounts.copy()
            )
            
            self.escrow_accounts[escrow_id] = escrow
            
            # Create escrow contract transaction
            escrow_contract_id = next(
                (c.contract_id for c in self.contracts.values() 
                 if c.contract_type == ContractType.ESCROW),
                None
            )
            
            if escrow_contract_id:
                tx_id = await self._create_transaction(
                    contract_id=escrow_contract_id,
                    initiator_id=creator_id,
                    function_name="create_escrow",
                    parameters={
                        "escrow_id": escrow_id,
                        "participants": list(participants),
                        "amounts": {k: str(v) for k, v in amounts.items()},
                        "release_conditions": release_conditions,
                        "expiry_date": expiry_date.isoformat()
                    }
                )
                
                await self._execute_transaction(tx_id)
            
            logger.info(f"Created escrow {escrow_id} with total amount {total_amount}")
            return escrow_id
            
        except Exception as e:
            logger.error(f"Error creating escrow: {e}")
            raise

    async def release_escrow_funds(self, escrow_id: str, releaser_id: str,
                                 release_amounts: Dict[str, Decimal],
                                 condition_met: str) -> bool:
        """Release funds from escrow based on met conditions"""
        try:
            if escrow_id not in self.escrow_accounts:
                raise ValueError(f"Escrow {escrow_id} not found")
            
            escrow = self.escrow_accounts[escrow_id]
            
            # Check if releaser is authorized
            if releaser_id not in escrow.participants:
                raise ValueError("Unauthorized to release escrow funds")
            
            # Check if escrow is active
            if not escrow.is_active or escrow.funds_released:
                raise ValueError("Escrow is not active or funds already released")
            
            # Validate release conditions
            condition_met_valid = any(
                condition.get("condition_id") == condition_met 
                for condition in escrow.release_conditions
            )
            
            if not condition_met_valid:
                raise ValueError(f"Invalid release condition: {condition_met}")
            
            # Validate release amounts
            total_release = sum(release_amounts.values())
            if total_release > escrow.total_amount:
                raise ValueError("Release amount exceeds escrow balance")
            
            # Update escrow
            escrow.pending_releases.update(release_amounts)
            
            # Create release transaction
            escrow_contract_id = next(
                (c.contract_id for c in self.contracts.values() 
                 if c.contract_type == ContractType.ESCROW),
                None
            )
            
            if escrow_contract_id:
                tx_id = await self._create_transaction(
                    contract_id=escrow_contract_id,
                    initiator_id=releaser_id,
                    function_name="release_funds",
                    parameters={
                        "escrow_id": escrow_id,
                        "release_amounts": {k: str(v) for k, v in release_amounts.items()},
                        "condition_met": condition_met
                    }
                )
                
                success = await self._execute_transaction(tx_id)
                
                if success:
                    escrow.funds_released = True
                    escrow.released_at = datetime.now()
                    escrow.is_active = False
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error releasing escrow funds: {e}")
            return False

    async def create_dispute(self, initiator_id: str, respondent_id: Optional[str],
                           dispute_type: str, title: str, description: str,
                           amount_disputed: Decimal) -> str:
        """Create a new dispute case"""
        try:
            dispute_id = f"DISPUTE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.disputes)}"
            
            dispute = Dispute(
                dispute_id=dispute_id,
                initiator_id=initiator_id,
                respondent_id=respondent_id,
                dispute_type=dispute_type,
                title=title,
                description=description,
                amount_disputed=amount_disputed,
                resolution_deadline=datetime.now() + timedelta(days=14)  # 14 day default
            )
            
            self.disputes[dispute_id] = dispute
            
            # Create dispute resolution contract transaction
            dispute_contract_id = next(
                (c.contract_id for c in self.contracts.values() 
                 if c.contract_type == ContractType.DISPUTE_RESOLUTION),
                None
            )
            
            if not dispute_contract_id:
                # Deploy dispute resolution contract if not exists
                dispute_contract_id = await self.deploy_contract(
                    deployer_id="system",
                    contract_type=ContractType.DISPUTE_RESOLUTION,
                    name="Dispute Resolution",
                    parameters={"arbitration_fee_percentage": 2, "mediation_timeout_days": 7}
                )
            
            tx_id = await self._create_transaction(
                contract_id=dispute_contract_id,
                initiator_id=initiator_id,
                function_name="create_dispute",
                parameters={
                    "dispute_id": dispute_id,
                    "respondent_id": respondent_id,
                    "dispute_type": dispute_type,
                    "amount_disputed": str(amount_disputed)
                }
            )
            
            await self._execute_transaction(tx_id)
            
            logger.info(f"Created dispute {dispute_id} by {initiator_id}")
            return dispute_id
            
        except Exception as e:
            logger.error(f"Error creating dispute: {e}")
            raise

    async def resolve_dispute(self, dispute_id: str, resolver_id: str,
                            resolution: str, compensation_amount: Decimal = Decimal('0')) -> bool:
        """Resolve a dispute"""
        try:
            if dispute_id not in self.disputes:
                raise ValueError(f"Dispute {dispute_id} not found")
            
            dispute = self.disputes[dispute_id]
            
            # Check if dispute is open for resolution
            if dispute.status not in [DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW, DisputeStatus.MEDIATION]:
                raise ValueError(f"Dispute {dispute_id} cannot be resolved in current status")
            
            # Update dispute
            dispute.status = DisputeStatus.RESOLVED
            dispute.resolution = resolution
            dispute.compensation_amount = compensation_amount
            dispute.updated_at = datetime.now()
            
            # If compensation is involved, handle transfer
            if compensation_amount > 0:
                await self._process_dispute_compensation(dispute_id, compensation_amount)
            
            logger.info(f"Resolved dispute {dispute_id} with compensation {compensation_amount}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving dispute {dispute_id}: {e}")
            return False

    async def _create_transaction(self, contract_id: str, initiator_id: str,
                                function_name: str, parameters: Dict[str, Any]) -> str:
        """Create a new blockchain transaction"""
        transaction_id = f"TX_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.transactions)}"
        
        transaction = Transaction(
            transaction_id=transaction_id,
            contract_id=contract_id,
            initiator_id=initiator_id,
            function_name=function_name,
            parameters=parameters,
            gas_price=20,  # Mock gas price
            gas_limit=500000,  # Mock gas limit
            required_signatures=self.contracts[contract_id].required_signatures
        )
        
        self.transactions[transaction_id] = transaction
        
        logger.debug(f"Created transaction {transaction_id} for contract {contract_id}")
        return transaction_id

    async def _execute_transaction(self, transaction_id: str) -> bool:
        """Execute a blockchain transaction"""
        try:
            transaction = self.transactions[transaction_id]
            
            # Check if enough signatures
            if len(transaction.signatures) < transaction.required_signatures:
                logger.warning(f"Transaction {transaction_id} needs more signatures")
                return False
            
            # Mock execution (in production, interact with actual blockchain)
            transaction.status = TransactionStatus.CONFIRMED
            transaction.confirmed_at = datetime.now()
            transaction.transaction_hash = hashlib.sha256(f"{transaction_id}_{datetime.now()}".encode()).hexdigest()
            transaction.block_number = 12345678  # Mock block number
            transaction.gas_used = 350000  # Mock gas used
            
            # Record execution in contract history
            contract = self.contracts[transaction.contract_id]
            contract.execution_history.append({
                "timestamp": datetime.now(),
                "transaction_id": transaction_id,
                "function_name": transaction.function_name,
                "initiator_id": transaction.initiator_id,
                "gas_used": transaction.gas_used
            })
            
            logger.info(f"Executed transaction {transaction_id}")
            return True
            
        except Exception as e:
            transaction.status = TransactionStatus.FAILED
            transaction.error_message = str(e)
            logger.error(f"Failed to execute transaction {transaction_id}: {e}")
            return False

    async def _validate_governance_action(self, action: GovernanceAction,
                                        parameters: Dict[str, Any],
                                        initiator_id: str) -> bool:
        """Validate if a governance action is authorized"""
        # Check if initiator has permission (mock validation)
        if action == GovernanceAction.EMERGENCY_ACTION:
            # Emergency actions require special authorization
            return initiator_id in ["founder", "lead"]
        
        # Other actions based on stake or role
        stake_threshold = parameters.get("minimum_stake", 1000)
        member_stake = parameters.get("initiator_stake", 0)
        
        return member_stake >= stake_threshold

    async def _process_dispute_compensation(self, dispute_id: str, amount: Decimal):
        """Process compensation payment for resolved dispute"""
        dispute = self.disputes[dispute_id]
        
        # Create compensation transaction
        multi_sig_contract_id = next(
            (c.contract_id for c in self.contracts.values() 
             if c.contract_type == ContractType.MULTI_SIG),
            None
        )
        
        if multi_sig_contract_id:
            tx_id = await self._create_transaction(
                contract_id=multi_sig_contract_id,
                initiator_id="system",
                function_name="transfer_compensation",
                parameters={
                    "dispute_id": dispute_id,
                    "recipient": dispute.initiator_id,
                    "amount": str(amount)
                }
            )
            
            await self._execute_transaction(tx_id)

    def get_contract_status(self, contract_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a smart contract"""
        if contract_id not in self.contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        
        # Calculate execution statistics
        total_executions = len(contract.execution_history)
        recent_executions = len([
            e for e in contract.execution_history 
            if e["timestamp"] >= datetime.now() - timedelta(days=30)
        ])
        
        return {
            "contract_id": contract_id,
            "name": contract.name,
            "type": contract.contract_type.value,
            "address": contract.contract_address,
            "version": contract.version,
            "is_active": contract.is_active,
            "is_verified": contract.is_verified,
            "created_at": contract.created_at.isoformat(),
            "authorized_signers": list(contract.authorized_signers),
            "required_signatures": contract.required_signatures,
            "parameters": contract.parameters,
            "execution_statistics": {
                "total_executions": total_executions,
                "recent_executions": recent_executions,
                "last_execution": contract.execution_history[-1]["timestamp"].isoformat() if contract.execution_history else None
            }
        }

    def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get status of a blockchain transaction"""
        if transaction_id not in self.transactions:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        transaction = self.transactions[transaction_id]
        
        return {
            "transaction_id": transaction_id,
            "contract_id": transaction.contract_id,
            "initiator_id": transaction.initiator_id,
            "function_name": transaction.function_name,
            "status": transaction.status.value,
            "created_at": transaction.created_at.isoformat(),
            "confirmed_at": transaction.confirmed_at.isoformat() if transaction.confirmed_at else None,
            "transaction_hash": transaction.transaction_hash,
            "block_number": transaction.block_number,
            "gas_used": transaction.gas_used,
            "signatures": {
                "required": transaction.required_signatures,
                "received": len(transaction.signatures),
                "signers": list(transaction.signatures.keys())
            },
            "error_message": transaction.error_message
        }

    def get_escrow_status(self, escrow_id: str) -> Dict[str, Any]:
        """Get status of an escrow account"""
        if escrow_id not in self.escrow_accounts:
            raise ValueError(f"Escrow {escrow_id} not found")
        
        escrow = self.escrow_accounts[escrow_id]
        
        return {
            "escrow_id": escrow_id,
            "participants": list(escrow.participants),
            "total_amount": str(escrow.total_amount),
            "is_active": escrow.is_active,
            "funds_released": escrow.funds_released,
            "expiry_date": escrow.expiry_date.isoformat() if escrow.expiry_date else None,
            "released_at": escrow.released_at.isoformat() if escrow.released_at else None,
            "contributions": {k: str(v) for k, v in escrow.contributions.items()},
            "pending_releases": {k: str(v) for k, v in escrow.pending_releases.items()},
            "release_conditions": escrow.release_conditions
        }

    def get_dispute_status(self, dispute_id: str) -> Dict[str, Any]:
        """Get status of a dispute"""
        if dispute_id not in self.disputes:
            raise ValueError(f"Dispute {dispute_id} not found")
        
        dispute = self.disputes[dispute_id]
        
        return {
            "dispute_id": dispute_id,
            "initiator_id": dispute.initiator_id,
            "respondent_id": dispute.respondent_id,
            "dispute_type": dispute.dispute_type,
            "title": dispute.title,
            "description": dispute.description,
            "status": dispute.status.value,
            "amount_disputed": str(dispute.amount_disputed),
            "created_at": dispute.created_at.isoformat(),
            "updated_at": dispute.updated_at.isoformat(),
            "resolution_deadline": dispute.resolution_deadline.isoformat() if dispute.resolution_deadline else None,
            "mediator_id": dispute.mediator_id,
            "arbitrator_id": dispute.arbitrator_id,
            "resolution": dispute.resolution,
            "compensation_amount": str(dispute.compensation_amount),
            "evidence_count": len(dispute.evidence),
            "community_resolution": dispute.community_resolution,
            "vote_count": len(dispute.votes)
        }

    def get_syndicate_blockchain_summary(self) -> Dict[str, Any]:
        """Get comprehensive blockchain integration summary"""
        # Contract statistics
        contracts_by_type = {}
        for contract_type in ContractType:
            contracts_by_type[contract_type.value] = len([
                c for c in self.contracts.values() if c.contract_type == contract_type
            ])
        
        # Transaction statistics
        transactions_by_status = {}
        for status in TransactionStatus:
            transactions_by_status[status.value] = len([
                t for t in self.transactions.values() if t.status == status
            ])
        
        # Dispute statistics
        disputes_by_status = {}
        for status in DisputeStatus:
            disputes_by_status[status.value] = len([
                d for d in self.disputes.values() if d.status == status
            ])
        
        # Calculate total values
        total_escrow_value = sum(e.total_amount for e in self.escrow_accounts.values())
        total_dispute_value = sum(d.amount_disputed for d in self.disputes.values())
        
        return {
            "syndicate_id": self.syndicate_id,
            "blockchain_network": self.blockchain_network,
            "contracts": {
                "total_contracts": len(self.contracts),
                "active_contracts": len([c for c in self.contracts.values() if c.is_active]),
                "verified_contracts": len([c for c in self.contracts.values() if c.is_verified]),
                "contracts_by_type": contracts_by_type
            },
            "transactions": {
                "total_transactions": len(self.transactions),
                "transactions_by_status": transactions_by_status,
                "total_gas_used": sum(t.gas_used or 0 for t in self.transactions.values()),
                "recent_transactions": len([
                    t for t in self.transactions.values() 
                    if t.created_at >= datetime.now() - timedelta(days=7)
                ])
            },
            "escrow": {
                "total_accounts": len(self.escrow_accounts),
                "active_accounts": len([e for e in self.escrow_accounts.values() if e.is_active]),
                "total_value": str(total_escrow_value),
                "funds_released": len([e for e in self.escrow_accounts.values() if e.funds_released])
            },
            "disputes": {
                "total_disputes": len(self.disputes),
                "disputes_by_status": disputes_by_status,
                "total_disputed_value": str(total_dispute_value),
                "average_resolution_time": "7.2 days"  # Mock calculation
            },
            "governance": {
                "automated_actions": len(self.automated_actions),
                "governance_rules": len(self.governance_rules),
                "multi_sig_enabled": ContractType.MULTI_SIG in self.contract_addresses
            }
        }