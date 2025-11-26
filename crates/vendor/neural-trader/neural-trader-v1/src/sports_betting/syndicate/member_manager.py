"""
Member Management System for Sports Betting Syndicates

This module provides comprehensive member management functionality including:
- Role-based permissions and access control
- Performance tracking and analytics
- Expertise scoring and verification
- Member onboarding and KYC processes
- Activity monitoring and engagement metrics
- Reputation and trust scoring
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

logger = logging.getLogger(__name__)


class MemberRole(Enum):
    """Member roles with different permission levels"""
    FOUNDER = "founder"  # Full control, can modify core settings
    LEAD = "lead"  # Strategic decisions, major bet approvals
    ANALYST = "analyst"  # Research, analysis, moderate betting
    CONTRIBUTOR = "contributor"  # Basic betting, limited permissions
    OBSERVER = "observer"  # Read-only access, no betting
    SUSPENDED = "suspended"  # Temporarily restricted access


class PermissionLevel(Enum):
    """Permission levels for different actions"""
    NONE = 0
    READ = 1
    WRITE = 2
    MODERATE = 3
    ADMIN = 4
    OWNER = 5


class MemberStatus(Enum):
    """Member account status"""
    PENDING = "pending"  # Awaiting approval/verification
    ACTIVE = "active"  # Full access
    INACTIVE = "inactive"  # Temporarily inactive
    SUSPENDED = "suspended"  # Access restricted
    BANNED = "banned"  # Permanently banned
    ALUMNI = "alumni"  # Former member with read access


class ExpertiseArea(Enum):
    """Areas of betting expertise"""
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    BASEBALL = "baseball"
    HOCKEY = "hockey"
    SOCCER = "soccer"
    TENNIS = "tennis"
    GOLF = "golf"
    MMA = "mma"
    ESPORTS = "esports"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MARKET_ANALYSIS = "market_analysis"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class PerformanceMetrics:
    """Performance tracking for a member"""
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_staked: Decimal = Decimal('0')
    total_profit_loss: Decimal = Decimal('0')
    longest_winning_streak: int = 0
    longest_losing_streak: int = 0
    current_streak: int = 0
    current_streak_type: str = "none"  # "winning", "losing", "none"
    
    # Advanced metrics
    roi_percentage: float = 0.0
    win_rate: float = 0.0
    average_odds: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: Decimal = Decimal('0')
    
    # Time-based performance
    last_30_days_roi: float = 0.0
    last_90_days_roi: float = 0.0
    monthly_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertiseProfile:
    """Expertise scoring and verification"""
    areas: Set[ExpertiseArea] = field(default_factory=set)
    scores: Dict[ExpertiseArea, float] = field(default_factory=dict)  # 0-100 scale
    verification_status: Dict[ExpertiseArea, bool] = field(default_factory=dict)
    certifications: List[str] = field(default_factory=list)
    years_experience: int = 0
    external_references: List[str] = field(default_factory=list)
    peer_endorsements: Dict[str, float] = field(default_factory=dict)  # member_id -> score


@dataclass
class ActivityMetrics:
    """Member activity and engagement tracking"""
    last_login: datetime = field(default_factory=datetime.now)
    total_logins: int = 0
    days_active: int = 0
    bets_placed_last_week: int = 0
    research_contributions: int = 0
    votes_cast: int = 0
    proposals_created: int = 0
    messages_sent: int = 0
    reputation_score: float = 50.0  # 0-100 scale
    trust_score: float = 50.0  # 0-100 scale


@dataclass
class KYCInfo:
    """Know Your Customer information"""
    identity_verified: bool = False
    document_type: str = ""
    document_number: str = ""
    verification_date: Optional[datetime] = None
    address_verified: bool = False
    phone_verified: bool = False
    email_verified: bool = False
    financial_verification: bool = False
    compliance_status: str = "pending"
    risk_assessment: str = "low"  # low, medium, high


@dataclass
class Member:
    """Comprehensive member profile"""
    member_id: str
    username: str
    email: str
    
    # Role and permissions
    role: MemberRole
    status: MemberStatus
    permissions: Dict[str, PermissionLevel] = field(default_factory=dict)
    
    # Profile information
    full_name: str = ""
    bio: str = ""
    location: str = ""
    timezone: str = "UTC"
    preferred_language: str = "en"
    
    # Dates
    joined_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    status_changed_date: datetime = field(default_factory=datetime.now)
    
    # Performance and expertise
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    expertise: ExpertiseProfile = field(default_factory=ExpertiseProfile)
    activity: ActivityMetrics = field(default_factory=ActivityMetrics)
    
    # Compliance
    kyc_info: KYCInfo = field(default_factory=KYCInfo)
    
    # Social features
    followers: Set[str] = field(default_factory=set)
    following: Set[str] = field(default_factory=set)
    blocked_members: Set[str] = field(default_factory=set)
    
    # Settings
    notifications_enabled: bool = True
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    
    # Internal tracking
    warning_count: int = 0
    suspension_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)


class MemberManager:
    """
    Advanced member management system for syndicate operations
    
    Features:
    - Role-based access control with granular permissions
    - Performance tracking and analytics
    - Expertise verification and scoring
    - KYC/AML compliance management
    - Activity monitoring and engagement metrics
    - Social features and reputation systems
    """
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        self.members: Dict[str, Member] = {}
        self.role_permissions = self._initialize_role_permissions()
        self.invitation_codes: Dict[str, Dict[str, Any]] = {}
        
        # Analytics and tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.activity_logs: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized MemberManager for syndicate {syndicate_id}")

    def _initialize_role_permissions(self) -> Dict[MemberRole, Dict[str, PermissionLevel]]:
        """Initialize default permissions for each role"""
        return {
            MemberRole.FOUNDER: {
                "manage_members": PermissionLevel.OWNER,
                "modify_settings": PermissionLevel.OWNER,
                "financial_operations": PermissionLevel.OWNER,
                "create_proposals": PermissionLevel.OWNER,
                "vote": PermissionLevel.OWNER,
                "place_bets": PermissionLevel.OWNER,
                "view_analytics": PermissionLevel.OWNER,
                "manage_funds": PermissionLevel.OWNER,
                "moderate_content": PermissionLevel.OWNER
            },
            MemberRole.LEAD: {
                "manage_members": PermissionLevel.MODERATE,
                "modify_settings": PermissionLevel.WRITE,
                "financial_operations": PermissionLevel.ADMIN,
                "create_proposals": PermissionLevel.ADMIN,
                "vote": PermissionLevel.ADMIN,
                "place_bets": PermissionLevel.ADMIN,
                "view_analytics": PermissionLevel.ADMIN,
                "manage_funds": PermissionLevel.ADMIN,
                "moderate_content": PermissionLevel.MODERATE
            },
            MemberRole.ANALYST: {
                "manage_members": PermissionLevel.READ,
                "modify_settings": PermissionLevel.READ,
                "financial_operations": PermissionLevel.WRITE,
                "create_proposals": PermissionLevel.WRITE,
                "vote": PermissionLevel.WRITE,
                "place_bets": PermissionLevel.WRITE,
                "view_analytics": PermissionLevel.WRITE,
                "manage_funds": PermissionLevel.WRITE,
                "moderate_content": PermissionLevel.READ
            },
            MemberRole.CONTRIBUTOR: {
                "manage_members": PermissionLevel.NONE,
                "modify_settings": PermissionLevel.NONE,
                "financial_operations": PermissionLevel.READ,
                "create_proposals": PermissionLevel.READ,
                "vote": PermissionLevel.WRITE,
                "place_bets": PermissionLevel.WRITE,
                "view_analytics": PermissionLevel.READ,
                "manage_funds": PermissionLevel.READ,
                "moderate_content": PermissionLevel.NONE
            },
            MemberRole.OBSERVER: {
                "manage_members": PermissionLevel.NONE,
                "modify_settings": PermissionLevel.NONE,
                "financial_operations": PermissionLevel.READ,
                "create_proposals": PermissionLevel.NONE,
                "vote": PermissionLevel.NONE,
                "place_bets": PermissionLevel.NONE,
                "view_analytics": PermissionLevel.READ,
                "manage_funds": PermissionLevel.NONE,
                "moderate_content": PermissionLevel.NONE
            },
            MemberRole.SUSPENDED: {
                "manage_members": PermissionLevel.NONE,
                "modify_settings": PermissionLevel.NONE,
                "financial_operations": PermissionLevel.NONE,
                "create_proposals": PermissionLevel.NONE,
                "vote": PermissionLevel.NONE,
                "place_bets": PermissionLevel.NONE,
                "view_analytics": PermissionLevel.NONE,
                "manage_funds": PermissionLevel.NONE,
                "moderate_content": PermissionLevel.NONE
            }
        }

    async def create_invitation(self, inviter_id: str, role: MemberRole, 
                               expiry_hours: int = 168) -> str:  # Default 7 days
        """Create an invitation code for new member registration"""
        try:
            if inviter_id not in self.members:
                raise ValueError(f"Inviter {inviter_id} not found")
            
            inviter = self.members[inviter_id]
            
            # Check if inviter has permission to invite
            if not self._check_permission(inviter, "manage_members", PermissionLevel.MODERATE):
                raise ValueError(f"Inviter {inviter_id} lacks permission to invite members")
            
            # Generate invitation code
            code_data = f"{self.syndicate_id}_{inviter_id}_{datetime.now().isoformat()}_{role.value}"
            invitation_code = hashlib.sha256(code_data.encode()).hexdigest()[:16].upper()
            
            # Store invitation
            self.invitation_codes[invitation_code] = {
                "inviter_id": inviter_id,
                "role": role,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=expiry_hours),
                "used": False,
                "used_by": None,
                "used_at": None
            }
            
            logger.info(f"Created invitation code {invitation_code} by {inviter_id}")
            return invitation_code
            
        except Exception as e:
            logger.error(f"Error creating invitation: {e}")
            raise

    async def register_member(self, username: str, email: str, invitation_code: str,
                            full_name: str = "", bio: str = "") -> str:
        """Register a new member using an invitation code"""
        try:
            # Validate invitation code
            if invitation_code not in self.invitation_codes:
                raise ValueError("Invalid invitation code")
            
            invitation = self.invitation_codes[invitation_code]
            
            if invitation["used"]:
                raise ValueError("Invitation code already used")
            
            if datetime.now() > invitation["expires_at"]:
                raise ValueError("Invitation code expired")
            
            # Check if username/email already exists
            for member in self.members.values():
                if member.username.lower() == username.lower():
                    raise ValueError(f"Username {username} already exists")
                if member.email.lower() == email.lower():
                    raise ValueError(f"Email {email} already registered")
            
            # Generate member ID
            member_id = f"MEM_{hashlib.sha256(f'{username}_{email}_{datetime.now()}'.encode()).hexdigest()[:12].upper()}"
            
            # Create member
            member = Member(
                member_id=member_id,
                username=username,
                email=email,
                role=invitation["role"],
                status=MemberStatus.PENDING,
                full_name=full_name,
                bio=bio,
                permissions=self.role_permissions[invitation["role"]].copy()
            )
            
            # Initialize privacy settings
            member.privacy_settings = {
                "show_performance": True,
                "show_activity": True,
                "allow_messages": True,
                "show_real_name": False
            }
            
            self.members[member_id] = member
            
            # Mark invitation as used
            invitation["used"] = True
            invitation["used_by"] = member_id
            invitation["used_at"] = datetime.now()
            
            # Log activity
            await self._log_activity("member_registered", {
                "member_id": member_id,
                "username": username,
                "inviter_id": invitation["inviter_id"]
            })
            
            logger.info(f"Registered new member {member_id} ({username})")
            return member_id
            
        except Exception as e:
            logger.error(f"Error registering member: {e}")
            raise

    async def approve_member(self, member_id: str, approver_id: str) -> bool:
        """Approve a pending member"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            if approver_id not in self.members:
                raise ValueError(f"Approver {approver_id} not found")
            
            member = self.members[member_id]
            approver = self.members[approver_id]
            
            # Check permissions
            if not self._check_permission(approver, "manage_members", PermissionLevel.MODERATE):
                raise ValueError(f"Approver {approver_id} lacks permission")
            
            if member.status != MemberStatus.PENDING:
                raise ValueError(f"Member {member_id} is not pending approval")
            
            # Approve member
            member.status = MemberStatus.ACTIVE
            member.status_changed_date = datetime.now()
            
            # Log activity
            await self._log_activity("member_approved", {
                "member_id": member_id,
                "approver_id": approver_id
            })
            
            logger.info(f"Approved member {member_id} by {approver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving member {member_id}: {e}")
            return False

    async def update_member_role(self, member_id: str, new_role: MemberRole, 
                               updater_id: str, reason: str = "") -> bool:
        """Update a member's role"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            if updater_id not in self.members:
                raise ValueError(f"Updater {updater_id} not found")
            
            member = self.members[member_id]
            updater = self.members[updater_id]
            
            # Check permissions
            if not self._check_permission(updater, "manage_members", PermissionLevel.ADMIN):
                raise ValueError(f"Updater {updater_id} lacks permission")
            
            # Prevent founder from being demoted unless by another founder
            if member.role == MemberRole.FOUNDER and updater.role != MemberRole.FOUNDER:
                raise ValueError("Only founders can modify founder roles")
            
            old_role = member.role
            
            # Update role and permissions
            member.role = new_role
            member.permissions = self.role_permissions[new_role].copy()
            member.status_changed_date = datetime.now()
            
            # Log activity
            await self._log_activity("role_updated", {
                "member_id": member_id,
                "old_role": old_role.value,
                "new_role": new_role.value,
                "updater_id": updater_id,
                "reason": reason
            })
            
            logger.info(f"Updated role for {member_id} from {old_role.value} to {new_role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating role for {member_id}: {e}")
            return False

    async def suspend_member(self, member_id: str, suspender_id: str, 
                           duration_hours: int, reason: str) -> bool:
        """Suspend a member for a specified duration"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            if suspender_id not in self.members:
                raise ValueError(f"Suspender {suspender_id} not found")
            
            member = self.members[member_id]
            suspender = self.members[suspender_id]
            
            # Check permissions
            if not self._check_permission(suspender, "manage_members", PermissionLevel.ADMIN):
                raise ValueError(f"Suspender {suspender_id} lacks permission")
            
            # Cannot suspend founder unless by another founder
            if member.role == MemberRole.FOUNDER and suspender.role != MemberRole.FOUNDER:
                raise ValueError("Only founders can suspend founders")
            
            # Update status
            old_status = member.status
            member.status = MemberStatus.SUSPENDED
            member.status_changed_date = datetime.now()
            
            # Record suspension
            suspension_record = {
                "suspended_by": suspender_id,
                "reason": reason,
                "start_date": datetime.now(),
                "end_date": datetime.now() + timedelta(hours=duration_hours),
                "duration_hours": duration_hours,
                "active": True
            }
            
            member.suspension_history.append(suspension_record)
            
            # Log activity
            await self._log_activity("member_suspended", {
                "member_id": member_id,
                "suspender_id": suspender_id,
                "duration_hours": duration_hours,
                "reason": reason
            })
            
            logger.info(f"Suspended member {member_id} for {duration_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Error suspending member {member_id}: {e}")
            return False

    async def update_performance(self, member_id: str, bet_result: Dict[str, Any]) -> bool:
        """Update member performance metrics based on bet result"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            member = self.members[member_id]
            perf = member.performance
            
            # Extract bet information
            stake = Decimal(str(bet_result.get("stake", 0)))
            payout = Decimal(str(bet_result.get("payout", 0)))
            won = bet_result.get("won", False)
            odds = float(bet_result.get("odds", 1.0))
            
            # Update basic metrics
            perf.total_bets += 1
            perf.total_staked += stake
            
            profit_loss = payout - stake
            perf.total_profit_loss += profit_loss
            
            if won:
                perf.winning_bets += 1
                # Update winning streak
                if perf.current_streak_type == "winning":
                    perf.current_streak += 1
                else:
                    perf.current_streak = 1
                    perf.current_streak_type = "winning"
                perf.longest_winning_streak = max(perf.longest_winning_streak, perf.current_streak)
            else:
                perf.losing_bets += 1
                # Update losing streak
                if perf.current_streak_type == "losing":
                    perf.current_streak += 1
                else:
                    perf.current_streak = 1
                    perf.current_streak_type = "losing"
                perf.longest_losing_streak = max(perf.longest_losing_streak, perf.current_streak)
            
            # Calculate derived metrics
            perf.win_rate = (perf.winning_bets / perf.total_bets) * 100 if perf.total_bets > 0 else 0
            perf.roi_percentage = float((perf.total_profit_loss / perf.total_staked) * 100) if perf.total_staked > 0 else 0
            perf.average_odds = (perf.average_odds * (perf.total_bets - 1) + odds) / perf.total_bets
            
            # Calculate profit factor
            total_wins = sum(t.get("profit", 0) for t in self.performance_history.get(member_id, []) if t.get("won", False))
            total_losses = abs(sum(t.get("profit", 0) for t in self.performance_history.get(member_id, []) if not t.get("won", False)))
            perf.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
            
            # Update max drawdown
            if profit_loss < 0:
                perf.max_drawdown = max(perf.max_drawdown, abs(profit_loss))
            
            # Store performance history
            if member_id not in self.performance_history:
                self.performance_history[member_id] = []
            
            self.performance_history[member_id].append({
                "timestamp": datetime.now(),
                "stake": float(stake),
                "payout": float(payout),
                "profit": float(profit_loss),
                "won": won,
                "odds": odds,
                "cumulative_profit": float(perf.total_profit_loss),
                "bet_count": perf.total_bets
            })
            
            # Update activity metrics
            member.activity.last_login = datetime.now()
            member.last_active = datetime.now()
            
            logger.info(f"Updated performance for member {member_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance for {member_id}: {e}")
            return False

    async def update_expertise_score(self, member_id: str, area: ExpertiseArea, 
                                   score: float, verifier_id: Optional[str] = None) -> bool:
        """Update expertise score for a member in a specific area"""
        try:
            if member_id not in self.members:
                raise ValueError(f"Member {member_id} not found")
            
            if not 0 <= score <= 100:
                raise ValueError("Score must be between 0 and 100")
            
            member = self.members[member_id]
            
            # If verifier provided, check permissions
            if verifier_id:
                if verifier_id not in self.members:
                    raise ValueError(f"Verifier {verifier_id} not found")
                
                verifier = self.members[verifier_id]
                if not self._check_permission(verifier, "moderate_content", PermissionLevel.MODERATE):
                    raise ValueError("Verifier lacks permission to update expertise scores")
                
                # Mark as verified if score is high enough and verifier is qualified
                if score >= 70 and verifier.role in [MemberRole.FOUNDER, MemberRole.LEAD]:
                    member.expertise.verification_status[area] = True
            
            # Update score
            member.expertise.areas.add(area)
            member.expertise.scores[area] = score
            
            # Log activity
            await self._log_activity("expertise_updated", {
                "member_id": member_id,
                "area": area.value,
                "score": score,
                "verifier_id": verifier_id
            })
            
            logger.info(f"Updated expertise score for {member_id} in {area.value}: {score}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating expertise score: {e}")
            return False

    def _check_permission(self, member: Member, action: str, required_level: PermissionLevel) -> bool:
        """Check if a member has sufficient permission for an action"""
        if member.status not in [MemberStatus.ACTIVE]:
            return False
        
        member_level = member.permissions.get(action, PermissionLevel.NONE)
        return member_level.value >= required_level.value

    async def _log_activity(self, activity_type: str, data: Dict[str, Any]):
        """Log member activity"""
        activity_log = {
            "timestamp": datetime.now(),
            "activity_type": activity_type,
            "data": data,
            "syndicate_id": self.syndicate_id
        }
        
        self.activity_logs.append(activity_log)
        
        # Keep only last 10000 activity logs
        if len(self.activity_logs) > 10000:
            self.activity_logs = self.activity_logs[-10000:]

    def get_member_profile(self, member_id: str, viewer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive member profile"""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")
        
        member = self.members[member_id]
        
        # Check privacy settings if viewer is different
        show_performance = member.privacy_settings.get("show_performance", True)
        show_activity = member.privacy_settings.get("show_activity", True)
        show_real_name = member.privacy_settings.get("show_real_name", False)
        
        if viewer_id and viewer_id != member_id and viewer_id in self.members:
            viewer = self.members[viewer_id]
            # Admins can see everything
            if self._check_permission(viewer, "manage_members", PermissionLevel.ADMIN):
                show_performance = True
                show_activity = True
                show_real_name = True
        
        profile = {
            "member_id": member_id,
            "username": member.username,
            "role": member.role.value,
            "status": member.status.value,
            "bio": member.bio,
            "location": member.location,
            "joined_date": member.joined_date.isoformat(),
            "last_active": member.last_active.isoformat(),
            "followers_count": len(member.followers),
            "following_count": len(member.following)
        }
        
        # Add name if privacy allows
        if show_real_name:
            profile["full_name"] = member.full_name
        
        # Add performance if privacy allows
        if show_performance:
            profile["performance"] = {
                "total_bets": member.performance.total_bets,
                "win_rate": member.performance.win_rate,
                "roi_percentage": member.performance.roi_percentage,
                "total_profit_loss": str(member.performance.total_profit_loss),
                "current_streak": member.performance.current_streak,
                "current_streak_type": member.performance.current_streak_type,
                "longest_winning_streak": member.performance.longest_winning_streak,
                "profit_factor": member.performance.profit_factor
            }
        
        # Add activity if privacy allows
        if show_activity:
            profile["activity"] = {
                "reputation_score": member.activity.reputation_score,
                "trust_score": member.activity.trust_score,
                "votes_cast": member.activity.votes_cast,
                "proposals_created": member.activity.proposals_created,
                "research_contributions": member.activity.research_contributions
            }
        
        # Add expertise
        profile["expertise"] = {
            "areas": [area.value for area in member.expertise.areas],
            "scores": {area.value: score for area, score in member.expertise.scores.items()},
            "verified_areas": [area.value for area, verified in member.expertise.verification_status.items() if verified],
            "years_experience": member.expertise.years_experience
        }
        
        return profile

    def get_member_rankings(self, sort_by: str = "roi", limit: int = 10) -> List[Dict[str, Any]]:
        """Get member rankings by various metrics"""
        active_members = [m for m in self.members.values() if m.status == MemberStatus.ACTIVE]
        
        if sort_by == "roi":
            active_members.sort(key=lambda m: m.performance.roi_percentage, reverse=True)
        elif sort_by == "win_rate":
            active_members.sort(key=lambda m: m.performance.win_rate, reverse=True)
        elif sort_by == "total_profit":
            active_members.sort(key=lambda m: m.performance.total_profit_loss, reverse=True)
        elif sort_by == "reputation":
            active_members.sort(key=lambda m: m.activity.reputation_score, reverse=True)
        elif sort_by == "activity":
            active_members.sort(key=lambda m: m.activity.votes_cast + m.activity.research_contributions, reverse=True)
        
        rankings = []
        for i, member in enumerate(active_members[:limit]):
            rankings.append({
                "rank": i + 1,
                "member_id": member.member_id,
                "username": member.username,
                "role": member.role.value,
                "metric_value": getattr(member.performance, sort_by, 0) if hasattr(member.performance, sort_by) else getattr(member.activity, sort_by, 0)
            })
        
        return rankings

    def get_syndicate_analytics(self) -> Dict[str, Any]:
        """Get comprehensive syndicate member analytics"""
        total_members = len(self.members)
        
        # Status distribution
        status_counts = {}
        for status in MemberStatus:
            status_counts[status.value] = len([m for m in self.members.values() if m.status == status])
        
        # Role distribution
        role_counts = {}
        for role in MemberRole:
            role_counts[role.value] = len([m for m in self.members.values() if m.role == role])
        
        # Performance aggregates
        active_members = [m for m in self.members.values() if m.status == MemberStatus.ACTIVE]
        
        if active_members:
            avg_roi = sum(m.performance.roi_percentage for m in active_members) / len(active_members)
            avg_win_rate = sum(m.performance.win_rate for m in active_members) / len(active_members)
            total_bets = sum(m.performance.total_bets for m in active_members)
            total_profit_loss = sum(m.performance.total_profit_loss for m in active_members)
        else:
            avg_roi = avg_win_rate = total_bets = total_profit_loss = 0
        
        # Activity metrics
        total_votes = sum(m.activity.votes_cast for m in active_members)
        total_proposals = sum(m.activity.proposals_created for m in active_members)
        total_research = sum(m.activity.research_contributions for m in active_members)
        
        return {
            "syndicate_id": self.syndicate_id,
            "total_members": total_members,
            "active_members": len(active_members),
            "member_distribution": {
                "by_status": status_counts,
                "by_role": role_counts
            },
            "performance_summary": {
                "average_roi": avg_roi,
                "average_win_rate": avg_win_rate,
                "total_bets": total_bets,
                "total_profit_loss": str(total_profit_loss)
            },
            "activity_summary": {
                "total_votes": total_votes,
                "total_proposals": total_proposals,
                "total_research_contributions": total_research,
                "total_activity_logs": len(self.activity_logs)
            },
            "engagement_metrics": {
                "average_reputation": sum(m.activity.reputation_score for m in active_members) / len(active_members) if active_members else 0,
                "average_trust": sum(m.activity.trust_score for m in active_members) / len(active_members) if active_members else 0
            }
        }