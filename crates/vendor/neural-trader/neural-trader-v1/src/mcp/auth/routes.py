"""
Authentication Routes for Syndicate Operations
Handles login, logout, token refresh, and member management
"""

from typing import Dict, Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from decimal import Decimal

from src.syndicate.member_management import MemberRole, SyndicateMemberManager
from .syndicate_auth import (
    SyndicateAuthenticator, AuthToken, AuditAction,
    get_authenticator, get_audit_logger, get_current_member,
    require_permission, require_role
)
from .encryption import SecureDataStore


# Pydantic models for request/response
class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str
    syndicate_id: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    member_id: str
    role: str
    syndicate_id: str


class RegisterRequest(BaseModel):
    """Member registration request"""
    name: str
    email: EmailStr
    password: str
    syndicate_id: str
    initial_contribution: Decimal
    role: MemberRole = MemberRole.CONTRIBUTING_MEMBER
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('initial_contribution')
    def validate_contribution(cls, v):
        """Validate initial contribution"""
        if v < Decimal('1000'):
            raise ValueError('Minimum initial contribution is $1000')
        return v


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password(cls, v, values):
        """Validate new password"""
        if 'current_password' in values and v == values['current_password']:
            raise ValueError('New password must be different from current password')
        
        # Apply same validation as registration
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Initialize components
authenticator = get_authenticator()
audit_logger = get_audit_logger()
secure_store = SecureDataStore()
security = HTTPBearer()

# In-memory storage (replace with database in production)
member_credentials = {}
member_managers = {}


@router.post("/register", response_model=Dict[str, Any])
async def register_member(request: RegisterRequest, req: Request):
    """Register a new syndicate member"""
    
    # Check if email already exists
    if request.email in member_credentials:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Get or create member manager for syndicate
    if request.syndicate_id not in member_managers:
        member_managers[request.syndicate_id] = SyndicateMemberManager(request.syndicate_id)
    
    member_manager = member_managers[request.syndicate_id]
    
    # Hash password
    hashed_password, salt = authenticator.hash_password(request.password)
    
    # Create member
    member = member_manager.add_member(
        name=request.name,
        email=request.email,
        role=request.role,
        initial_contribution=request.initial_contribution
    )
    
    # Store credentials securely
    credentials = {
        'member_id': member.id,
        'email': request.email,
        'password': hashed_password,
        'salt': salt,
        'syndicate_id': request.syndicate_id
    }
    
    encrypted_creds = secure_store.store_member_credentials(member.id, credentials)
    member_credentials[request.email] = encrypted_creds
    
    # Log registration
    audit_logger.log(
        AuditAction.MEMBER_ADDED,
        member_id=member.id,
        syndicate_id=request.syndicate_id,
        details={
            'email': request.email,
            'role': request.role.value,
            'ip_address': req.client.host
        }
    )
    
    return {
        "message": "Member registered successfully",
        "member_id": member.id,
        "role": request.role.value
    }


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, req: Request):
    """Authenticate member and return JWT token"""
    
    # Check rate limit
    if not authenticator.rate_limiter.check_rate_limit(request.email, "login"):
        retry_after = authenticator.rate_limiter.get_remaining_time(request.email, "login")
        
        # Log failed attempt
        audit_logger.log(
            AuditAction.FAILED_LOGIN,
            member_id="unknown",
            syndicate_id=request.syndicate_id,
            details={
                'email': request.email,
                'reason': 'rate_limit_exceeded',
                'ip_address': req.client.host
            }
        )
        
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Please try again in {retry_after} seconds"
        )
    
    # Retrieve encrypted credentials
    if request.email not in member_credentials:
        # Log failed attempt
        audit_logger.log(
            AuditAction.FAILED_LOGIN,
            member_id="unknown",
            syndicate_id=request.syndicate_id,
            details={
                'email': request.email,
                'reason': 'invalid_credentials',
                'ip_address': req.client.host
            }
        )
        
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Decrypt credentials
    encrypted_creds = member_credentials[request.email]
    creds = secure_store.retrieve_member_credentials(encrypted_creds)
    
    # Verify password
    if not authenticator.verify_password(request.password, creds['password'], creds['salt']):
        # Log failed attempt
        audit_logger.log(
            AuditAction.FAILED_LOGIN,
            member_id=creds['member_id'],
            syndicate_id=request.syndicate_id,
            details={
                'email': request.email,
                'reason': 'invalid_password',
                'ip_address': req.client.host
            }
        )
        
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify syndicate ID
    if creds['syndicate_id'] != request.syndicate_id:
        raise HTTPException(status_code=401, detail="Invalid syndicate")
    
    # Get member details
    member_manager = member_managers.get(request.syndicate_id)
    if not member_manager:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    member = member_manager.members.get(creds['member_id'])
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if not member.is_active:
        raise HTTPException(status_code=403, detail="Account suspended")
    
    # Generate token
    token = authenticator.generate_token(
        member_id=member.id,
        syndicate_id=request.syndicate_id,
        role=member.role,
        permissions=member.permissions
    )
    
    # Create session
    session = authenticator.create_session(
        member_id=member.id,
        syndicate_id=request.syndicate_id,
        role=member.role,
        ip_address=req.client.host,
        user_agent=req.headers.get("User-Agent", "unknown")
    )
    
    # Log successful login
    audit_logger.log(
        AuditAction.LOGIN,
        member_id=member.id,
        syndicate_id=request.syndicate_id,
        details={
            'ip_address': req.client.host,
            'session_id': session.session_id
        }
    )
    
    return LoginResponse(
        access_token=token,
        expires_in=authenticator.token_expiry_hours * 3600,
        member_id=member.id,
        role=member.role.value,
        syndicate_id=request.syndicate_id
    )


@router.post("/logout")
async def logout(
    current_member: AuthToken = Depends(get_current_member),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Logout member and invalidate session"""
    
    # TODO: In production, add token to blacklist
    
    # Log logout
    audit_logger.log(
        AuditAction.LOGOUT,
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        details={'token_id': current_member.token_id}
    )
    
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    current_member: AuthToken = Depends(get_current_member)
):
    """Refresh authentication token"""
    
    # Check if token is about to expire (within 1 hour)
    if current_member.remaining_time.total_seconds() > 3600:
        raise HTTPException(
            status_code=400,
            detail="Token is not eligible for refresh yet"
        )
    
    # Get member details
    member_manager = member_managers.get(current_member.syndicate_id)
    if not member_manager:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    member = member_manager.members.get(current_member.member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    if not member.is_active:
        raise HTTPException(status_code=403, detail="Account suspended")
    
    # Generate new token
    new_token = authenticator.generate_token(
        member_id=member.id,
        syndicate_id=current_member.syndicate_id,
        role=member.role,
        permissions=member.permissions
    )
    
    return LoginResponse(
        access_token=new_token,
        expires_in=authenticator.token_expiry_hours * 3600,
        member_id=member.id,
        role=member.role.value,
        syndicate_id=current_member.syndicate_id
    )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_member: AuthToken = Depends(get_current_member)
):
    """Change member password"""
    
    # Get member credentials
    member_manager = member_managers.get(current_member.syndicate_id)
    if not member_manager:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    member = member_manager.members.get(current_member.member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    # Retrieve encrypted credentials
    if member.email not in member_credentials:
        raise HTTPException(status_code=404, detail="Credentials not found")
    
    # Decrypt credentials
    encrypted_creds = member_credentials[member.email]
    creds = secure_store.retrieve_member_credentials(encrypted_creds)
    
    # Verify current password
    if not authenticator.verify_password(request.current_password, creds['password'], creds['salt']):
        raise HTTPException(status_code=401, detail="Invalid current password")
    
    # Hash new password
    new_hashed_password, new_salt = authenticator.hash_password(request.new_password)
    
    # Update credentials
    creds['password'] = new_hashed_password
    creds['salt'] = new_salt
    
    # Re-encrypt and store
    encrypted_creds = secure_store.store_member_credentials(current_member.member_id, creds)
    member_credentials[member.email] = encrypted_creds
    
    # Log password change
    audit_logger.log(
        AuditAction.SETTINGS_CHANGED,
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        details={'change_type': 'password'}
    )
    
    return {"message": "Password changed successfully"}


@router.get("/me")
async def get_current_member_info(
    current_member: AuthToken = Depends(get_current_member)
):
    """Get current member information"""
    
    member_manager = member_managers.get(current_member.syndicate_id)
    if not member_manager:
        raise HTTPException(status_code=404, detail="Syndicate not found")
    
    member = member_manager.members.get(current_member.member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    
    return {
        "member_id": member.id,
        "name": member.name,
        "email": member.email,
        "role": member.role.value,
        "tier": member.tier.value,
        "syndicate_id": current_member.syndicate_id,
        "joined_date": member.joined_date.isoformat(),
        "is_active": member.is_active,
        "permissions": {
            "create_syndicate": member.permissions.create_syndicate,
            "modify_strategy": member.permissions.modify_strategy,
            "approve_large_bets": member.permissions.approve_large_bets,
            "manage_members": member.permissions.manage_members,
            "distribute_profits": member.permissions.distribute_profits,
            "access_all_analytics": member.permissions.access_all_analytics,
            "veto_power": member.permissions.veto_power,
            "propose_bets": member.permissions.propose_bets,
            "access_advanced_analytics": member.permissions.access_advanced_analytics,
            "create_models": member.permissions.create_models,
            "vote_on_strategy": member.permissions.vote_on_strategy,
            "manage_junior_analysts": member.permissions.manage_junior_analysts,
            "view_bets": member.permissions.view_bets,
            "vote_on_major_decisions": member.permissions.vote_on_major_decisions,
            "access_basic_analytics": member.permissions.access_basic_analytics,
            "propose_ideas": member.permissions.propose_ideas,
            "withdraw_own_funds": member.permissions.withdraw_own_funds
        }
    }


@router.get("/audit-log")
@require_role(MemberRole.LEAD_INVESTOR)
async def get_audit_log(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    member_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    current_member: AuthToken = Depends(get_current_member)
):
    """Get audit log entries (Lead Investor only)"""
    
    # TODO: Implement audit log retrieval from persistent storage
    
    return {
        "entries": [],
        "total": 0,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "member_id": member_id,
            "action": action
        }
    }