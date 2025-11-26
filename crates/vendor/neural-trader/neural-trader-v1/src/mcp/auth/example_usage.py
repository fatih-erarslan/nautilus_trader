"""
Example usage of Syndicate Authentication System
Demonstrates integration with FastAPI application
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from decimal import Decimal
from typing import Dict, Any

# Import authentication components
from src.mcp.auth import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    AuditLoggingMiddleware,
    auth_router,
    get_current_member,
    require_permission,
    require_role,
    AuthToken,
    get_audit_logger,
    AuditAction
)

from src.syndicate.member_management import MemberRole

# Create FastAPI app
app = FastAPI(title="Syndicate Trading Platform", version="1.0.0")

# Add authentication middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Include auth routes
app.include_router(auth_router)

# Example protected endpoints

@app.get("/api/dashboard")
async def get_dashboard(current_member: AuthToken = Depends(get_current_member)):
    """Get member dashboard - available to all authenticated members"""
    return {
        "member_id": current_member.member_id,
        "syndicate_id": current_member.syndicate_id,
        "role": current_member.role.value,
        "message": "Welcome to your dashboard!"
    }


@app.post("/api/funds/transfer")
@require_permission("distribute_profits")
async def transfer_funds(
    amount: Decimal,
    recipient_member_id: str,
    reason: str,
    current_member: AuthToken = Depends(get_current_member)
):
    """Transfer funds between members - requires distribute_profits permission"""
    
    # Log financial operation
    get_audit_logger().log_financial_operation(
        operation_type="transfer",
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        amount=float(amount),
        currency="USD",
        details={
            "recipient": recipient_member_id,
            "reason": reason
        }
    )
    
    # Implement actual transfer logic here
    
    return {
        "status": "success",
        "transaction_id": "TXN123456",
        "amount": str(amount),
        "recipient": recipient_member_id
    }


@app.post("/api/strategy/modify")
@require_permission("modify_strategy")
async def modify_strategy(
    strategy_updates: Dict[str, Any],
    current_member: AuthToken = Depends(get_current_member)
):
    """Modify trading strategy - requires modify_strategy permission"""
    
    # Log strategy modification
    get_audit_logger().log(
        AuditAction.STRATEGY_MODIFIED,
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        details=strategy_updates
    )
    
    # Implement strategy modification logic here
    
    return {
        "status": "success",
        "message": "Strategy updated successfully",
        "updates": strategy_updates
    }


@app.get("/api/analytics/advanced")
@require_permission("access_advanced_analytics")
async def get_advanced_analytics(
    current_member: AuthToken = Depends(get_current_member)
):
    """Get advanced analytics - requires access_advanced_analytics permission"""
    
    # Log data access
    get_audit_logger().log(
        AuditAction.DATA_ACCESS,
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        details={"data_type": "advanced_analytics"}
    )
    
    # Return analytics data
    return {
        "performance_metrics": {
            "total_roi": 15.7,
            "sharpe_ratio": 1.85,
            "max_drawdown": -8.3,
            "win_rate": 0.62
        },
        "risk_analysis": {
            "var_95": -5000,
            "cvar_95": -7500,
            "beta": 0.85
        }
    }


@app.post("/api/members/manage")
@require_role(MemberRole.LEAD_INVESTOR)
async def manage_members(
    action: str,
    target_member_id: str,
    details: Dict[str, Any],
    current_member: AuthToken = Depends(get_current_member)
):
    """Manage syndicate members - requires LEAD_INVESTOR role"""
    
    allowed_actions = ["suspend", "activate", "change_role", "update_tier"]
    
    if action not in allowed_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Allowed: {allowed_actions}")
    
    # Log member management action
    get_audit_logger().log(
        AuditAction.ROLE_CHANGED if action == "change_role" else AuditAction.SETTINGS_CHANGED,
        member_id=current_member.member_id,
        syndicate_id=current_member.syndicate_id,
        details={
            "action": action,
            "target_member": target_member_id,
            **details
        }
    )
    
    # Implement member management logic here
    
    return {
        "status": "success",
        "action": action,
        "target_member": target_member_id,
        "details": details
    }


@app.get("/api/audit/summary")
@require_role(MemberRole.LEAD_INVESTOR)
async def get_audit_summary(
    current_member: AuthToken = Depends(get_current_member)
):
    """Get audit summary - requires LEAD_INVESTOR role"""
    
    # This would fetch from the actual audit log
    return {
        "total_logins": 156,
        "failed_logins": 12,
        "financial_operations": 45,
        "strategy_modifications": 8,
        "member_changes": 3,
        "last_24h_activity": {
            "logins": 23,
            "trades": 67,
            "votes": 5
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper logging"""
    
    # Log 4xx and 5xx errors
    if exc.status_code >= 400:
        details = {
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
        
        # Try to get member info if authenticated
        member_id = "anonymous"
        syndicate_id = "unknown"
        
        if hasattr(request.state, 'auth_token'):
            member_id = request.state.auth_token.member_id
            syndicate_id = request.state.auth_token.syndicate_id
        
        get_audit_logger().log(
            AuditAction.PERMISSION_DENIED if exc.status_code == 403 else AuditAction.DATA_ACCESS,
            member_id=member_id,
            syndicate_id=syndicate_id,
            details=details
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "syndicate-platform"}


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )