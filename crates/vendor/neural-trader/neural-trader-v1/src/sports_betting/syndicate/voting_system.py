"""
Voting and Consensus System for Sports Betting Syndicates

This module provides comprehensive voting and governance functionality including:
- Proposal creation and management
- Weighted voting based on expertise and capital
- Consensus mechanisms for large decisions
- Time-limited voting periods
- Automated proposal execution
- Governance analytics and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Types of proposals that can be voted on"""
    LARGE_BET = "large_bet"
    STRATEGY_CHANGE = "strategy_change"
    MEMBER_ADMISSION = "member_admission"
    MEMBER_REMOVAL = "member_removal"
    CAPITAL_REALLOCATION = "capital_reallocation"
    RISK_PARAMETER_CHANGE = "risk_parameter_change"
    FEE_ADJUSTMENT = "fee_adjustment"
    EMERGENCY_ACTION = "emergency_action"
    PARTNERSHIP = "partnership"
    INVESTMENT_OPPORTUNITY = "investment_opportunity"


class ProposalStatus(Enum):
    """Status of a proposal"""
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class VoteType(Enum):
    """Types of votes"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class VotingMethod(Enum):
    """Methods for calculating voting power"""
    ONE_MEMBER_ONE_VOTE = "one_member_one_vote"
    CAPITAL_WEIGHTED = "capital_weighted"
    EXPERTISE_WEIGHTED = "expertise_weighted"
    HYBRID_WEIGHTED = "hybrid_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"


@dataclass
class Vote:
    """Represents a single vote on a proposal"""
    voter_id: str
    proposal_id: str
    vote_type: VoteType
    voting_power: Decimal
    timestamp: datetime
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """Represents a governance proposal"""
    proposal_id: str
    proposer_id: str
    proposal_type: ProposalType
    title: str
    description: str
    details: Dict[str, Any]
    
    # Voting parameters
    voting_method: VotingMethod
    required_quorum: Decimal  # Percentage of total voting power needed
    required_majority: Decimal  # Percentage of votes needed to pass
    voting_deadline: datetime
    
    # Status tracking
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    
    # Voting results
    votes: List[Vote] = field(default_factory=list)
    total_voting_power: Decimal = Decimal('0')
    yes_voting_power: Decimal = Decimal('0')
    no_voting_power: Decimal = Decimal('0')
    abstain_voting_power: Decimal = Decimal('0')
    
    # Execution
    execution_callback: Optional[Callable] = None
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class VoterProfile:
    """Profile of a syndicate member for voting purposes"""
    member_id: str
    capital_contribution: Decimal = Decimal('0')
    expertise_score: float = 1.0
    performance_score: float = 1.0
    voting_history: List[str] = field(default_factory=list)  # Proposal IDs
    delegate_to: Optional[str] = None  # Delegate voting power to another member
    voting_restrictions: Set[ProposalType] = field(default_factory=set)


class VotingSystem:
    """
    Advanced voting and consensus system for syndicate governance
    
    Features:
    - Multiple voting methods (capital-weighted, expertise-based, hybrid)
    - Flexible quorum and majority requirements
    - Time-limited voting periods
    - Delegation and proxy voting
    - Automated proposal execution
    - Governance analytics and insights
    """
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        self.proposals: Dict[str, Proposal] = {}
        self.voter_profiles: Dict[str, VoterProfile] = {}
        self.governance_settings = self._initialize_governance_settings()
        
        # Analytics
        self.proposal_statistics: Dict[str, Any] = {}
        self.voting_analytics: Dict[str, Any] = {}
        
        logger.info(f"Initialized VotingSystem for syndicate {syndicate_id}")

    def _initialize_governance_settings(self) -> Dict[str, Any]:
        """Initialize default governance settings"""
        return {
            "default_voting_period_hours": 72,  # 3 days
            "emergency_voting_period_hours": 24,  # 1 day for emergency proposals
            "default_quorum": Decimal('0.51'),  # 51% of voting power
            "default_majority": Decimal('0.60'),  # 60% majority to pass
            "large_bet_threshold": Decimal('10000'),  # Threshold for requiring vote
            "delegation_allowed": True,
            "max_active_proposals": 10,
            "proposal_types_requiring_supermajority": {
                ProposalType.MEMBER_REMOVAL,
                ProposalType.FEE_ADJUSTMENT,
                ProposalType.RISK_PARAMETER_CHANGE
            }
        }

    async def register_voter(self, member_id: str, capital_contribution: Decimal, 
                           expertise_score: float = 1.0, performance_score: float = 1.0) -> bool:
        """Register a member as a voter in the syndicate"""
        try:
            if member_id in self.voter_profiles:
                logger.warning(f"Voter {member_id} already registered")
                return False
            
            voter_profile = VoterProfile(
                member_id=member_id,
                capital_contribution=capital_contribution,
                expertise_score=expertise_score,
                performance_score=performance_score
            )
            
            self.voter_profiles[member_id] = voter_profile
            
            logger.info(f"Registered voter {member_id} with capital {capital_contribution}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering voter {member_id}: {e}")
            return False

    async def create_proposal(self, proposer_id: str, proposal_type: ProposalType,
                            title: str, description: str, details: Dict[str, Any],
                            custom_voting_params: Optional[Dict[str, Any]] = None) -> str:
        """Create a new governance proposal"""
        try:
            if proposer_id not in self.voter_profiles:
                raise ValueError(f"Proposer {proposer_id} is not a registered voter")
            
            # Check if proposer can create this type of proposal
            voter = self.voter_profiles[proposer_id]
            if proposal_type in voter.voting_restrictions:
                raise ValueError(f"Proposer {proposer_id} cannot create {proposal_type.value} proposals")
            
            # Check maximum active proposals
            active_proposals = len([p for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE])
            if active_proposals >= self.governance_settings["max_active_proposals"]:
                raise ValueError("Maximum number of active proposals reached")
            
            # Generate proposal ID
            proposal_id = f"PROP_{proposal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Set voting parameters
            voting_params = self._get_voting_parameters(proposal_type, custom_voting_params)
            
            # Create proposal
            proposal = Proposal(
                proposal_id=proposal_id,
                proposer_id=proposer_id,
                proposal_type=proposal_type,
                title=title,
                description=description,
                details=details,
                **voting_params
            )
            
            self.proposals[proposal_id] = proposal
            
            logger.info(f"Created proposal {proposal_id} by {proposer_id}")
            return proposal_id
            
        except Exception as e:
            logger.error(f"Error creating proposal: {e}")
            raise

    def _get_voting_parameters(self, proposal_type: ProposalType, 
                             custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get voting parameters for a specific proposal type"""
        # Default parameters
        params = {
            "voting_method": VotingMethod.HYBRID_WEIGHTED,
            "required_quorum": self.governance_settings["default_quorum"],
            "required_majority": self.governance_settings["default_majority"],
            "voting_deadline": datetime.now() + timedelta(
                hours=self.governance_settings["default_voting_period_hours"]
            )
        }
        
        # Adjust for proposal type
        if proposal_type == ProposalType.EMERGENCY_ACTION:
            params["voting_deadline"] = datetime.now() + timedelta(
                hours=self.governance_settings["emergency_voting_period_hours"]
            )
            params["required_majority"] = Decimal('0.75')  # 75% for emergency actions
        
        elif proposal_type in self.governance_settings["proposal_types_requiring_supermajority"]:
            params["required_majority"] = Decimal('0.67')  # 67% supermajority
        
        elif proposal_type == ProposalType.LARGE_BET:
            params["voting_method"] = VotingMethod.CAPITAL_WEIGHTED
            params["required_majority"] = Decimal('0.55')  # 55% for betting decisions
        
        # Apply custom parameters
        if custom_params:
            params.update(custom_params)
        
        return params

    async def start_voting(self, proposal_id: str) -> bool:
        """Start the voting period for a proposal"""
        try:
            if proposal_id not in self.proposals:
                raise ValueError(f"Proposal {proposal_id} not found")
            
            proposal = self.proposals[proposal_id]
            
            if proposal.status != ProposalStatus.DRAFT:
                raise ValueError(f"Proposal {proposal_id} is not in draft status")
            
            # Calculate total voting power
            proposal.total_voting_power = await self._calculate_total_voting_power(proposal.voting_method)
            
            # Update proposal status
            proposal.status = ProposalStatus.ACTIVE
            proposal.started_at = datetime.now()
            
            logger.info(f"Started voting for proposal {proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voting for proposal {proposal_id}: {e}")
            return False

    async def cast_vote(self, proposal_id: str, voter_id: str, vote_type: VoteType,
                       rationale: str = "") -> bool:
        """Cast a vote on an active proposal"""
        try:
            if proposal_id not in self.proposals:
                raise ValueError(f"Proposal {proposal_id} not found")
            
            if voter_id not in self.voter_profiles:
                raise ValueError(f"Voter {voter_id} not registered")
            
            proposal = self.proposals[proposal_id]
            
            # Check if proposal is active
            if proposal.status != ProposalStatus.ACTIVE:
                raise ValueError(f"Proposal {proposal_id} is not active")
            
            # Check if voting period has expired
            if datetime.now() > proposal.voting_deadline:
                await self._expire_proposal(proposal_id)
                raise ValueError(f"Voting period for proposal {proposal_id} has expired")
            
            # Check if voter has already voted
            existing_vote = next((v for v in proposal.votes if v.voter_id == voter_id), None)
            if existing_vote:
                # Update existing vote
                await self._update_vote(proposal, existing_vote, vote_type, rationale)
            else:
                # Cast new vote
                await self._cast_new_vote(proposal, voter_id, vote_type, rationale)
            
            # Check if proposal should be finalized
            await self._check_proposal_completion(proposal_id)
            
            logger.info(f"Vote cast by {voter_id} on proposal {proposal_id}: {vote_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error casting vote for proposal {proposal_id}: {e}")
            return False

    async def _cast_new_vote(self, proposal: Proposal, voter_id: str, 
                           vote_type: VoteType, rationale: str):
        """Cast a new vote"""
        # Calculate voting power
        voting_power = await self._calculate_voting_power(voter_id, proposal.voting_method)
        
        # Create vote
        vote = Vote(
            voter_id=voter_id,
            proposal_id=proposal.proposal_id,
            vote_type=vote_type,
            voting_power=voting_power,
            timestamp=datetime.now(),
            rationale=rationale
        )
        
        # Add vote to proposal
        proposal.votes.append(vote)
        
        # Update vote tallies
        if vote_type == VoteType.YES:
            proposal.yes_voting_power += voting_power
        elif vote_type == VoteType.NO:
            proposal.no_voting_power += voting_power
        elif vote_type == VoteType.ABSTAIN:
            proposal.abstain_voting_power += voting_power
        
        # Update voter history
        self.voter_profiles[voter_id].voting_history.append(proposal.proposal_id)

    async def _update_vote(self, proposal: Proposal, existing_vote: Vote, 
                         new_vote_type: VoteType, rationale: str):
        """Update an existing vote"""
        # Remove old vote from tallies
        old_voting_power = existing_vote.voting_power
        if existing_vote.vote_type == VoteType.YES:
            proposal.yes_voting_power -= old_voting_power
        elif existing_vote.vote_type == VoteType.NO:
            proposal.no_voting_power -= old_voting_power
        elif existing_vote.vote_type == VoteType.ABSTAIN:
            proposal.abstain_voting_power -= old_voting_power
        
        # Update vote
        existing_vote.vote_type = new_vote_type
        existing_vote.timestamp = datetime.now()
        existing_vote.rationale = rationale
        
        # Add new vote to tallies
        if new_vote_type == VoteType.YES:
            proposal.yes_voting_power += old_voting_power
        elif new_vote_type == VoteType.NO:
            proposal.no_voting_power += old_voting_power
        elif new_vote_type == VoteType.ABSTAIN:
            proposal.abstain_voting_power += old_voting_power

    async def _calculate_voting_power(self, voter_id: str, voting_method: VotingMethod) -> Decimal:
        """Calculate voting power for a voter based on the voting method"""
        voter = self.voter_profiles[voter_id]
        
        # Check for delegation
        if voter.delegate_to and voter.delegate_to in self.voter_profiles:
            return Decimal('0')  # Delegated votes are handled separately
        
        if voting_method == VotingMethod.ONE_MEMBER_ONE_VOTE:
            return Decimal('1')
        
        elif voting_method == VotingMethod.CAPITAL_WEIGHTED:
            total_capital = sum(v.capital_contribution for v in self.voter_profiles.values())
            if total_capital == 0:
                return Decimal('1')
            return voter.capital_contribution / total_capital * Decimal('100')
        
        elif voting_method == VotingMethod.EXPERTISE_WEIGHTED:
            total_expertise = sum(v.expertise_score for v in self.voter_profiles.values())
            if total_expertise == 0:
                return Decimal('1')
            return Decimal(str(voter.expertise_score)) / Decimal(str(total_expertise)) * Decimal('100')
        
        elif voting_method == VotingMethod.PERFORMANCE_WEIGHTED:
            total_performance = sum(v.performance_score for v in self.voter_profiles.values())
            if total_performance == 0:
                return Decimal('1')
            return Decimal(str(voter.performance_score)) / Decimal(str(total_performance)) * Decimal('100')
        
        elif voting_method == VotingMethod.HYBRID_WEIGHTED:
            # 40% capital, 30% expertise, 30% performance
            capital_weight = await self._calculate_voting_power(voter_id, VotingMethod.CAPITAL_WEIGHTED)
            expertise_weight = await self._calculate_voting_power(voter_id, VotingMethod.EXPERTISE_WEIGHTED)
            performance_weight = await self._calculate_voting_power(voter_id, VotingMethod.PERFORMANCE_WEIGHTED)
            
            return (capital_weight * Decimal('0.4') + 
                   expertise_weight * Decimal('0.3') + 
                   performance_weight * Decimal('0.3'))
        
        return Decimal('1')

    async def _calculate_total_voting_power(self, voting_method: VotingMethod) -> Decimal:
        """Calculate total voting power for all eligible voters"""
        total_power = Decimal('0')
        
        for voter_id in self.voter_profiles:
            power = await self._calculate_voting_power(voter_id, voting_method)
            total_power += power
        
        return total_power

    async def _check_proposal_completion(self, proposal_id: str):
        """Check if a proposal should be finalized"""
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.ACTIVE:
            return
        
        # Calculate participation rate
        votes_cast = proposal.yes_voting_power + proposal.no_voting_power + proposal.abstain_voting_power
        participation_rate = votes_cast / proposal.total_voting_power if proposal.total_voting_power > 0 else Decimal('0')
        
        # Check if quorum is met
        if participation_rate >= proposal.required_quorum:
            # Calculate majority (excluding abstentions)
            total_decisive_votes = proposal.yes_voting_power + proposal.no_voting_power
            yes_percentage = proposal.yes_voting_power / total_decisive_votes if total_decisive_votes > 0 else Decimal('0')
            
            # Determine outcome
            if yes_percentage >= proposal.required_majority:
                await self._finalize_proposal(proposal_id, ProposalStatus.PASSED)
            else:
                await self._finalize_proposal(proposal_id, ProposalStatus.REJECTED)
        
        # Check if deadline has passed
        elif datetime.now() > proposal.voting_deadline:
            await self._expire_proposal(proposal_id)

    async def _finalize_proposal(self, proposal_id: str, status: ProposalStatus):
        """Finalize a proposal with the given status"""
        proposal = self.proposals[proposal_id]
        proposal.status = status
        proposal.ended_at = datetime.now()
        
        # Execute proposal if passed and has callback
        if status == ProposalStatus.PASSED and proposal.execution_callback:
            try:
                result = await proposal.execution_callback(proposal.details)
                proposal.execution_result = result
                proposal.executed_at = datetime.now()
                proposal.status = ProposalStatus.EXECUTED
                logger.info(f"Executed proposal {proposal_id}")
            except Exception as e:
                logger.error(f"Error executing proposal {proposal_id}: {e}")
                proposal.execution_result = {"error": str(e)}
        
        logger.info(f"Finalized proposal {proposal_id} with status {status.value}")

    async def _expire_proposal(self, proposal_id: str):
        """Handle proposal expiration"""
        proposal = self.proposals[proposal_id]
        
        # Check if quorum was met
        votes_cast = proposal.yes_voting_power + proposal.no_voting_power + proposal.abstain_voting_power
        participation_rate = votes_cast / proposal.total_voting_power if proposal.total_voting_power > 0 else Decimal('0')
        
        if participation_rate >= proposal.required_quorum:
            # Decide based on votes
            total_decisive_votes = proposal.yes_voting_power + proposal.no_voting_power
            yes_percentage = proposal.yes_voting_power / total_decisive_votes if total_decisive_votes > 0 else Decimal('0')
            
            status = ProposalStatus.PASSED if yes_percentage >= proposal.required_majority else ProposalStatus.REJECTED
            await self._finalize_proposal(proposal_id, status)
        else:
            # Expired due to lack of quorum
            proposal.status = ProposalStatus.EXPIRED
            proposal.ended_at = datetime.now()
            logger.info(f"Proposal {proposal_id} expired due to insufficient participation")

    async def delegate_voting_power(self, delegator_id: str, delegate_id: str) -> bool:
        """Delegate voting power from one member to another"""
        try:
            if delegator_id not in self.voter_profiles:
                raise ValueError(f"Delegator {delegator_id} not found")
            
            if delegate_id not in self.voter_profiles:
                raise ValueError(f"Delegate {delegate_id} not found")
            
            if not self.governance_settings["delegation_allowed"]:
                raise ValueError("Delegation is not allowed in this syndicate")
            
            # Set delegation
            self.voter_profiles[delegator_id].delegate_to = delegate_id
            
            logger.info(f"Voting power delegated from {delegator_id} to {delegate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error delegating voting power: {e}")
            return False

    async def revoke_delegation(self, delegator_id: str) -> bool:
        """Revoke a voting power delegation"""
        try:
            if delegator_id not in self.voter_profiles:
                raise ValueError(f"Delegator {delegator_id} not found")
            
            self.voter_profiles[delegator_id].delegate_to = None
            
            logger.info(f"Voting delegation revoked for {delegator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking delegation: {e}")
            return False

    def get_proposal_summary(self, proposal_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        # Calculate voting statistics
        votes_cast = proposal.yes_voting_power + proposal.no_voting_power + proposal.abstain_voting_power
        participation_rate = float(votes_cast / proposal.total_voting_power) if proposal.total_voting_power > 0 else 0
        
        total_decisive_votes = proposal.yes_voting_power + proposal.no_voting_power
        yes_percentage = float(proposal.yes_voting_power / total_decisive_votes) if total_decisive_votes > 0 else 0
        no_percentage = float(proposal.no_voting_power / total_decisive_votes) if total_decisive_votes > 0 else 0
        
        return {
            "proposal_id": proposal.proposal_id,
            "proposer_id": proposal.proposer_id,
            "proposal_type": proposal.proposal_type.value,
            "title": proposal.title,
            "description": proposal.description,
            "details": proposal.details,
            "status": proposal.status.value,
            "created_at": proposal.created_at.isoformat(),
            "started_at": proposal.started_at.isoformat() if proposal.started_at else None,
            "voting_deadline": proposal.voting_deadline.isoformat(),
            "ended_at": proposal.ended_at.isoformat() if proposal.ended_at else None,
            "voting_method": proposal.voting_method.value,
            "required_quorum": str(proposal.required_quorum),
            "required_majority": str(proposal.required_majority),
            "voting_statistics": {
                "total_voting_power": str(proposal.total_voting_power),
                "votes_cast": str(votes_cast),
                "participation_rate": participation_rate,
                "yes_voting_power": str(proposal.yes_voting_power),
                "no_voting_power": str(proposal.no_voting_power),
                "abstain_voting_power": str(proposal.abstain_voting_power),
                "yes_percentage": yes_percentage,
                "no_percentage": no_percentage,
                "vote_count": len(proposal.votes)
            },
            "execution_result": proposal.execution_result
        }

    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get all active proposals"""
        active_proposals = [
            p for p in self.proposals.values() 
            if p.status == ProposalStatus.ACTIVE
        ]
        
        return [self.get_proposal_summary(p.proposal_id) for p in active_proposals]

    def get_voting_history(self, member_id: str) -> List[Dict[str, Any]]:
        """Get voting history for a member"""
        if member_id not in self.voter_profiles:
            raise ValueError(f"Member {member_id} not found")
        
        member_votes = []
        for proposal in self.proposals.values():
            vote = next((v for v in proposal.votes if v.voter_id == member_id), None)
            if vote:
                member_votes.append({
                    "proposal_id": proposal.proposal_id,
                    "proposal_title": proposal.title,
                    "proposal_type": proposal.proposal_type.value,
                    "vote_type": vote.vote_type.value,
                    "voting_power": str(vote.voting_power),
                    "timestamp": vote.timestamp.isoformat(),
                    "rationale": vote.rationale,
                    "proposal_outcome": proposal.status.value
                })
        
        # Sort by timestamp (most recent first)
        member_votes.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return member_votes

    def get_governance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive governance analytics"""
        total_proposals = len(self.proposals)
        
        # Proposal statistics by status
        status_counts = {}
        for status in ProposalStatus:
            status_counts[status.value] = len([p for p in self.proposals.values() if p.status == status])
        
        # Proposal statistics by type
        type_counts = {}
        for prop_type in ProposalType:
            type_counts[prop_type.value] = len([p for p in self.proposals.values() if p.proposal_type == prop_type])
        
        # Voter participation statistics
        voter_stats = {}
        for voter_id, voter in self.voter_profiles.items():
            participation_rate = len(voter.voting_history) / total_proposals if total_proposals > 0 else 0
            voter_stats[voter_id] = {
                "total_votes": len(voter.voting_history),
                "participation_rate": participation_rate,
                "capital_contribution": str(voter.capital_contribution),
                "expertise_score": voter.expertise_score,
                "performance_score": voter.performance_score,
                "has_delegation": voter.delegate_to is not None
            }
        
        # Calculate average participation
        avg_participation = sum(v["participation_rate"] for v in voter_stats.values()) / len(voter_stats) if voter_stats else 0
        
        return {
            "syndicate_id": self.syndicate_id,
            "total_proposals": total_proposals,
            "active_proposals": status_counts.get("active", 0),
            "total_voters": len(self.voter_profiles),
            "average_participation_rate": avg_participation,
            "governance_settings": self.governance_settings,
            "proposal_statistics": {
                "by_status": status_counts,
                "by_type": type_counts
            },
            "voter_statistics": voter_stats,
            "recent_activity": {
                "proposals_this_month": len([
                    p for p in self.proposals.values() 
                    if p.created_at >= datetime.now() - timedelta(days=30)
                ]),
                "votes_this_month": len([
                    v for p in self.proposals.values() 
                    for v in p.votes 
                    if v.timestamp >= datetime.now() - timedelta(days=30)
                ])
            }
        }