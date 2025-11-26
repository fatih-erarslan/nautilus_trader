"""
Syndicate Risk Controls for Sports Betting

Implements member-level betting limits, consensus requirements,
risk allocation by expertise, and emergency shutdown procedures.
"""

import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


class MemberRole(Enum):
    """Syndicate member roles"""
    ADMIN = "admin"
    SENIOR_TRADER = "senior_trader"
    TRADER = "trader"
    ANALYST = "analyst"
    OBSERVER = "observer"


class ExpertiseLevel(Enum):
    """Member expertise levels"""
    EXPERT = "expert"
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"


class ConsensusType(Enum):
    """Types of consensus requirements"""
    MAJORITY = "majority"  # >50%
    SUPER_MAJORITY = "super_majority"  # >66%
    UNANIMOUS = "unanimous"  # 100%
    WEIGHTED = "weighted"  # Based on expertise


class EmergencyStatus(Enum):
    """Emergency status levels"""
    NORMAL = "normal"
    ALERT = "alert"
    WARNING = "warning"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class SyndicateMember:
    """Represents a syndicate member"""
    member_id: str
    name: str
    role: MemberRole
    expertise_areas: Dict[str, ExpertiseLevel] = field(default_factory=dict)
    betting_limit: float = 0.0
    daily_limit: float = 0.0
    current_exposure: float = 0.0
    daily_volume: float = 0.0
    performance_score: float = 0.0
    active: bool = True
    joined_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_activity: Optional[datetime.datetime] = None


@dataclass
class ConsensusRequirement:
    """Consensus requirement for different bet sizes"""
    min_amount: float
    max_amount: float
    consensus_type: ConsensusType
    required_roles: List[MemberRole] = field(default_factory=list)
    min_participants: int = 2
    timeout_minutes: int = 30


@dataclass
class BettingProposal:
    """Betting proposal requiring consensus"""
    proposal_id: str
    proposer_id: str
    sport: str
    event: str
    selection: str
    odds: float
    proposed_stake: float
    rationale: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    votes: Dict[str, bool] = field(default_factory=dict)
    comments: List[Dict] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, expired


@dataclass
class EmergencyProtocol:
    """Emergency shutdown protocol"""
    trigger_condition: str
    trigger_value: float
    action: str  # 'alert', 'restrict', 'shutdown'
    notification_list: List[str] = field(default_factory=list)
    cooldown_hours: int = 24


class SyndicateRiskController:
    """
    Controls risk at the syndicate level including member limits,
    consensus mechanisms, and emergency procedures.
    """
    
    def __init__(self,
                 syndicate_name: str,
                 total_bankroll: float,
                 max_member_allocation: float = 0.20):
        """
        Initialize Syndicate Risk Controller
        
        Args:
            syndicate_name: Name of the syndicate
            total_bankroll: Total syndicate bankroll
            max_member_allocation: Maximum allocation per member
        """
        self.syndicate_name = syndicate_name
        self.total_bankroll = total_bankroll
        self.max_member_allocation = max_member_allocation
        
        # Member management
        self.members: Dict[str, SyndicateMember] = {}
        self.member_permissions: Dict[str, Set[str]] = defaultdict(set)
        
        # Consensus requirements
        self.consensus_requirements = self._initialize_consensus_requirements()
        self.active_proposals: Dict[str, BettingProposal] = {}
        self.proposal_history: List[BettingProposal] = []
        
        # Risk allocation
        self.expertise_weights = {
            ExpertiseLevel.EXPERT: 1.5,
            ExpertiseLevel.ADVANCED: 1.2,
            ExpertiseLevel.INTERMEDIATE: 1.0,
            ExpertiseLevel.BEGINNER: 0.5
        }
        
        # Emergency protocols
        self.emergency_status = EmergencyStatus.NORMAL
        self.emergency_protocols = self._initialize_emergency_protocols()
        self.emergency_log: List[Dict] = []
        
        # Activity tracking
        self.daily_activity = defaultdict(lambda: {
            'bets': 0,
            'volume': 0,
            'proposals': 0,
            'approvals': 0
        })
        
    def _initialize_consensus_requirements(self) -> List[ConsensusRequirement]:
        """Initialize default consensus requirements"""
        return [
            # Small bets - single member decision
            ConsensusRequirement(
                min_amount=0,
                max_amount=1000,
                consensus_type=ConsensusType.MAJORITY,
                min_participants=1
            ),
            # Medium bets - majority vote
            ConsensusRequirement(
                min_amount=1000,
                max_amount=5000,
                consensus_type=ConsensusType.MAJORITY,
                required_roles=[MemberRole.TRADER, MemberRole.SENIOR_TRADER],
                min_participants=3,
                timeout_minutes=30
            ),
            # Large bets - super majority with senior involvement
            ConsensusRequirement(
                min_amount=5000,
                max_amount=20000,
                consensus_type=ConsensusType.SUPER_MAJORITY,
                required_roles=[MemberRole.SENIOR_TRADER],
                min_participants=5,
                timeout_minutes=60
            ),
            # Very large bets - weighted consensus with admin approval
            ConsensusRequirement(
                min_amount=20000,
                max_amount=float('inf'),
                consensus_type=ConsensusType.WEIGHTED,
                required_roles=[MemberRole.ADMIN, MemberRole.SENIOR_TRADER],
                min_participants=7,
                timeout_minutes=120
            )
        ]
        
    def _initialize_emergency_protocols(self) -> List[EmergencyProtocol]:
        """Initialize emergency shutdown protocols"""
        return [
            # Daily loss alert
            EmergencyProtocol(
                trigger_condition='daily_loss_percentage',
                trigger_value=0.05,  # 5%
                action='alert',
                notification_list=['all_admins', 'senior_traders']
            ),
            # Significant loss warning
            EmergencyProtocol(
                trigger_condition='daily_loss_percentage',
                trigger_value=0.10,  # 10%
                action='restrict',
                notification_list=['all_members'],
                cooldown_hours=12
            ),
            # Emergency shutdown
            EmergencyProtocol(
                trigger_condition='daily_loss_percentage',
                trigger_value=0.15,  # 15%
                action='shutdown',
                notification_list=['all_members', 'external_monitors'],
                cooldown_hours=24
            ),
            # Rapid loss protection
            EmergencyProtocol(
                trigger_condition='hourly_loss_rate',
                trigger_value=0.05,  # 5% per hour
                action='shutdown',
                notification_list=['all_members'],
                cooldown_hours=6
            )
        ]
        
    def add_member(self, member: SyndicateMember) -> bool:
        """Add a new syndicate member"""
        if member.member_id in self.members:
            logger.warning(f"Member {member.member_id} already exists")
            return False
            
        # Set default limits based on role
        if member.betting_limit == 0:
            role_limits = {
                MemberRole.ADMIN: self.total_bankroll * 0.10,
                MemberRole.SENIOR_TRADER: self.total_bankroll * 0.07,
                MemberRole.TRADER: self.total_bankroll * 0.05,
                MemberRole.ANALYST: self.total_bankroll * 0.02,
                MemberRole.OBSERVER: 0
            }
            member.betting_limit = role_limits.get(member.role, 0)
            
        if member.daily_limit == 0:
            member.daily_limit = member.betting_limit * 3
            
        self.members[member.member_id] = member
        
        # Set permissions based on role
        self._set_member_permissions(member)
        
        logger.info(f"Added member {member.name} ({member.role.value})")
        return True
        
    def _set_member_permissions(self, member: SyndicateMember):
        """Set permissions based on member role"""
        base_permissions = {'view_proposals', 'comment'}
        
        role_permissions = {
            MemberRole.ADMIN: {
                'create_proposal', 'vote', 'emergency_shutdown',
                'modify_limits', 'add_members', 'remove_members'
            },
            MemberRole.SENIOR_TRADER: {
                'create_proposal', 'vote', 'emergency_alert',
                'modify_own_limits'
            },
            MemberRole.TRADER: {
                'create_proposal', 'vote'
            },
            MemberRole.ANALYST: {
                'create_proposal', 'vote_limited'
            },
            MemberRole.OBSERVER: set()
        }
        
        self.member_permissions[member.member_id] = (
            base_permissions | role_permissions.get(member.role, set())
        )
        
    def check_member_limit(self,
                           member_id: str,
                           proposed_amount: float,
                           sport: str
                           ) -> Tuple[bool, List[str]]:
        """
        Check if member can place bet within limits
        
        Args:
            member_id: Member identifier
            proposed_amount: Proposed bet amount
            sport: Sport category
            
        Returns:
            Tuple of (is_allowed, violations)
        """
        if member_id not in self.members:
            return False, ["Member not found"]
            
        member = self.members[member_id]
        violations = []
        
        # Check if member is active
        if not member.active:
            violations.append("Member account is inactive")
            
        # Check single bet limit
        if proposed_amount > member.betting_limit:
            violations.append(
                f"Exceeds betting limit: ${proposed_amount:.2f} > ${member.betting_limit:.2f}"
            )
            
        # Check daily limit
        if member.daily_volume + proposed_amount > member.daily_limit:
            remaining = member.daily_limit - member.daily_volume
            violations.append(
                f"Exceeds daily limit: ${proposed_amount:.2f} > ${remaining:.2f} remaining"
            )
            
        # Check expertise requirement for large bets
        if proposed_amount > member.betting_limit * 0.5:
            expertise = member.expertise_areas.get(sport, ExpertiseLevel.BEGINNER)
            if expertise not in [ExpertiseLevel.EXPERT, ExpertiseLevel.ADVANCED]:
                violations.append(
                    f"Insufficient expertise in {sport} for large bet"
                )
                
        return len(violations) == 0, violations
        
    def create_betting_proposal(self,
                                proposer_id: str,
                                sport: str,
                                event: str,
                                selection: str,
                                odds: float,
                                proposed_stake: float,
                                rationale: str
                                ) -> Optional[BettingProposal]:
        """
        Create a betting proposal that requires consensus
        
        Args:
            proposer_id: Member creating the proposal
            sport: Sport type
            event: Event description
            selection: Betting selection
            odds: Proposed odds
            proposed_stake: Proposed stake amount
            rationale: Reasoning for the bet
            
        Returns:
            BettingProposal if created, None if not allowed
        """
        # Check permissions
        if 'create_proposal' not in self.member_permissions.get(proposer_id, set()):
            logger.warning(f"Member {proposer_id} lacks permission to create proposals")
            return None
            
        # Find applicable consensus requirement
        consensus_req = None
        for req in self.consensus_requirements:
            if req.min_amount <= proposed_stake <= req.max_amount:
                consensus_req = req
                break
                
        if not consensus_req:
            logger.error(f"No consensus requirement found for stake ${proposed_stake}")
            return None
            
        # Create proposal
        proposal = BettingProposal(
            proposal_id=f"PROP_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            proposer_id=proposer_id,
            sport=sport,
            event=event,
            selection=selection,
            odds=odds,
            proposed_stake=proposed_stake,
            rationale=rationale,
            created_at=datetime.datetime.now(),
            expires_at=datetime.datetime.now() + datetime.timedelta(
                minutes=consensus_req.timeout_minutes
            )
        )
        
        # Auto-approve if proposer has sufficient authority
        member = self.members[proposer_id]
        if (member.role == MemberRole.ADMIN and 
            proposed_stake <= member.betting_limit):
            proposal.votes[proposer_id] = True
            proposal.status = "approved"
        else:
            proposal.votes[proposer_id] = True  # Proposer automatically votes yes
            
        self.active_proposals[proposal.proposal_id] = proposal
        
        # Update daily activity
        date_key = datetime.datetime.now().date().isoformat()
        self.daily_activity[date_key]['proposals'] += 1
        
        logger.info(
            f"Created betting proposal {proposal.proposal_id} for "
            f"${proposed_stake:.2f} on {selection}"
        )
        
        return proposal
        
    def vote_on_proposal(self,
                         proposal_id: str,
                         member_id: str,
                         vote: bool,
                         comment: Optional[str] = None
                         ) -> bool:
        """
        Vote on a betting proposal
        
        Args:
            proposal_id: Proposal identifier
            member_id: Voting member
            vote: True for approve, False for reject
            comment: Optional comment
            
        Returns:
            True if vote recorded successfully
        """
        if proposal_id not in self.active_proposals:
            logger.warning(f"Proposal {proposal_id} not found")
            return False
            
        if 'vote' not in self.member_permissions.get(member_id, set()):
            logger.warning(f"Member {member_id} lacks voting permission")
            return False
            
        proposal = self.active_proposals[proposal_id]
        
        # Check if proposal has expired
        if datetime.datetime.now() > proposal.expires_at:
            proposal.status = "expired"
            self._finalize_proposal(proposal_id)
            return False
            
        # Record vote
        proposal.votes[member_id] = vote
        
        # Add comment if provided
        if comment:
            proposal.comments.append({
                'member_id': member_id,
                'timestamp': datetime.datetime.now(),
                'comment': comment
            })
            
        # Check if consensus is reached
        if self._check_consensus(proposal_id):
            self._finalize_proposal(proposal_id)
            
        return True
        
    def _check_consensus(self, proposal_id: str) -> bool:
        """Check if consensus has been reached for a proposal"""
        proposal = self.active_proposals[proposal_id]
        
        # Find applicable consensus requirement
        consensus_req = None
        for req in self.consensus_requirements:
            if req.min_amount <= proposal.proposed_stake <= req.max_amount:
                consensus_req = req
                break
                
        if not consensus_req:
            return False
            
        # Check minimum participants
        if len(proposal.votes) < consensus_req.min_participants:
            return False
            
        # Check required roles
        voters = set(proposal.votes.keys())
        required_roles_met = True
        
        for required_role in consensus_req.required_roles:
            role_members = {
                m_id for m_id, m in self.members.items()
                if m.role == required_role
            }
            if not (voters & role_members):
                required_roles_met = False
                break
                
        if not required_roles_met:
            return False
            
        # Calculate consensus based on type
        if consensus_req.consensus_type == ConsensusType.UNANIMOUS:
            return all(proposal.votes.values())
            
        elif consensus_req.consensus_type == ConsensusType.MAJORITY:
            yes_votes = sum(1 for v in proposal.votes.values() if v)
            return yes_votes > len(proposal.votes) / 2
            
        elif consensus_req.consensus_type == ConsensusType.SUPER_MAJORITY:
            yes_votes = sum(1 for v in proposal.votes.values() if v)
            return yes_votes > len(proposal.votes) * 0.66
            
        elif consensus_req.consensus_type == ConsensusType.WEIGHTED:
            # Weight votes by expertise
            weighted_yes = 0
            weighted_total = 0
            
            for member_id, vote in proposal.votes.items():
                member = self.members[member_id]
                expertise = member.expertise_areas.get(
                    proposal.sport,
                    ExpertiseLevel.BEGINNER
                )
                weight = self.expertise_weights[expertise]
                
                weighted_total += weight
                if vote:
                    weighted_yes += weight
                    
            return weighted_yes > weighted_total * 0.66
            
        return False
        
    def _finalize_proposal(self, proposal_id: str):
        """Finalize a proposal (approve/reject)"""
        proposal = self.active_proposals[proposal_id]
        
        if proposal.status == "pending":
            if self._check_consensus(proposal_id):
                proposal.status = "approved"
                # Update daily activity
                date_key = datetime.datetime.now().date().isoformat()
                self.daily_activity[date_key]['approvals'] += 1
                logger.info(f"Proposal {proposal_id} approved")
            else:
                proposal.status = "rejected"
                logger.info(f"Proposal {proposal_id} rejected")
                
        # Move to history
        self.proposal_history.append(proposal)
        del self.active_proposals[proposal_id]
        
    def allocate_risk_by_expertise(self,
                                   total_stake: float,
                                   sport: str,
                                   participating_members: List[str]
                                   ) -> Dict[str, float]:
        """
        Allocate risk among members based on expertise
        
        Args:
            total_stake: Total stake to allocate
            sport: Sport type
            participating_members: List of participating member IDs
            
        Returns:
            Dictionary of member_id to allocated amount
        """
        allocations = {}
        total_weight = 0
        member_weights = {}
        
        # Calculate weights based on expertise
        for member_id in participating_members:
            if member_id not in self.members:
                continue
                
            member = self.members[member_id]
            expertise = member.expertise_areas.get(sport, ExpertiseLevel.BEGINNER)
            
            # Base weight on expertise
            weight = self.expertise_weights[expertise]
            
            # Adjust by performance score
            weight *= (0.5 + member.performance_score)
            
            # Adjust by available limit
            available = min(
                member.betting_limit - member.current_exposure,
                member.daily_limit - member.daily_volume
            )
            
            if available > 0:
                member_weights[member_id] = weight
                total_weight += weight
                
        # Allocate proportionally
        if total_weight > 0:
            for member_id, weight in member_weights.items():
                member = self.members[member_id]
                
                # Calculate allocation
                allocation = total_stake * (weight / total_weight)
                
                # Apply member limits
                max_allocation = min(
                    member.betting_limit - member.current_exposure,
                    member.daily_limit - member.daily_volume
                )
                
                allocation = min(allocation, max_allocation)
                allocations[member_id] = allocation
                
        # Handle any unallocated amount
        allocated_total = sum(allocations.values())
        if allocated_total < total_stake:
            unallocated = total_stake - allocated_total
            logger.warning(
                f"Could not fully allocate ${unallocated:.2f} of "
                f"${total_stake:.2f} stake"
            )
            
        return allocations
        
    def check_emergency_conditions(self) -> Tuple[EmergencyStatus, List[str]]:
        """
        Check emergency conditions and trigger protocols if needed
        
        Returns:
            Tuple of (emergency_status, triggered_protocols)
        """
        triggered = []
        current_time = datetime.datetime.now()
        
        # Calculate current metrics
        daily_pnl = self._calculate_daily_pnl()
        hourly_pnl = self._calculate_hourly_pnl()
        
        daily_loss_pct = abs(daily_pnl / self.total_bankroll) if daily_pnl < 0 else 0
        hourly_loss_rate = abs(hourly_pnl / self.total_bankroll) if hourly_pnl < 0 else 0
        
        # Check each protocol
        for protocol in self.emergency_protocols:
            triggered_value = None
            
            if protocol.trigger_condition == 'daily_loss_percentage':
                if daily_loss_pct >= protocol.trigger_value:
                    triggered_value = daily_loss_pct
                    
            elif protocol.trigger_condition == 'hourly_loss_rate':
                if hourly_loss_rate >= protocol.trigger_value:
                    triggered_value = hourly_loss_rate
                    
            if triggered_value is not None:
                triggered.append(
                    f"{protocol.action}: {protocol.trigger_condition} = "
                    f"{triggered_value:.2%} (trigger: {protocol.trigger_value:.2%})"
                )
                
                # Log emergency event
                self.emergency_log.append({
                    'timestamp': current_time,
                    'protocol': protocol.trigger_condition,
                    'value': triggered_value,
                    'action': protocol.action
                })
                
                # Update emergency status
                if protocol.action == 'alert':
                    self.emergency_status = EmergencyStatus.ALERT
                elif protocol.action == 'restrict':
                    self.emergency_status = EmergencyStatus.WARNING
                elif protocol.action == 'shutdown':
                    self.emergency_status = EmergencyStatus.SHUTDOWN
                    
                # Send notifications
                self._send_emergency_notifications(protocol, triggered_value)
                
        return self.emergency_status, triggered
        
    def execute_emergency_shutdown(self, reason: str):
        """Execute emergency shutdown procedures"""
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        
        # Set status
        self.emergency_status = EmergencyStatus.SHUTDOWN
        
        # Deactivate all members
        for member in self.members.values():
            member.active = False
            
        # Cancel all active proposals
        for proposal_id in list(self.active_proposals.keys()):
            proposal = self.active_proposals[proposal_id]
            proposal.status = "cancelled"
            self._finalize_proposal(proposal_id)
            
        # Log shutdown
        self.emergency_log.append({
            'timestamp': datetime.datetime.now(),
            'action': 'shutdown',
            'reason': reason
        })
        
        # Send emergency notifications
        self._send_emergency_notifications(
            EmergencyProtocol(
                trigger_condition='manual_shutdown',
                trigger_value=0,
                action='shutdown',
                notification_list=['all_members', 'external_monitors']
            ),
            0
        )
        
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        date_key = datetime.datetime.now().date().isoformat()
        # This would integrate with actual betting records
        return 0.0  # Placeholder
        
    def _calculate_hourly_pnl(self) -> float:
        """Calculate last hour's P&L"""
        # This would integrate with actual betting records
        return 0.0  # Placeholder
        
    def _send_emergency_notifications(self,
                                      protocol: EmergencyProtocol,
                                      triggered_value: float):
        """Send emergency notifications"""
        notification_targets = []
        
        for target in protocol.notification_list:
            if target == 'all_members':
                notification_targets.extend(self.members.keys())
            elif target == 'all_admins':
                notification_targets.extend([
                    m_id for m_id, m in self.members.items()
                    if m.role == MemberRole.ADMIN
                ])
            elif target == 'senior_traders':
                notification_targets.extend([
                    m_id for m_id, m in self.members.items()
                    if m.role == MemberRole.SENIOR_TRADER
                ])
                
        # Log notification (actual implementation would send alerts)
        logger.critical(
            f"Emergency notification sent to {len(notification_targets)} recipients: "
            f"{protocol.action} - {protocol.trigger_condition} = {triggered_value:.2%}"
        )
        
    def get_syndicate_status(self) -> Dict:
        """Get comprehensive syndicate status"""
        active_members = sum(1 for m in self.members.values() if m.active)
        total_exposure = sum(m.current_exposure for m in self.members.values())
        
        return {
            'syndicate_name': self.syndicate_name,
            'emergency_status': self.emergency_status.value,
            'total_bankroll': self.total_bankroll,
            'total_exposure': total_exposure,
            'exposure_percentage': f"{total_exposure / self.total_bankroll:.1%}",
            'active_members': active_members,
            'total_members': len(self.members),
            'active_proposals': len(self.active_proposals),
            'daily_stats': dict(self.daily_activity[datetime.datetime.now().date().isoformat()]),
            'recent_emergency_events': self.emergency_log[-5:]
        }