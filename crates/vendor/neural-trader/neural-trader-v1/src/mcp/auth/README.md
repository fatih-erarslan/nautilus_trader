# Syndicate Authentication and Access Control System

This module provides comprehensive authentication and security features for syndicate operations, including JWT-based authentication, role-based access control (RBAC), audit logging, and data encryption.

## Features

### 1. **JWT-Based Authentication**
- Secure token generation and validation
- Configurable token expiry
- Session management
- Token refresh capability

### 2. **Role-Based Access Control (RBAC)**
- Five predefined roles with hierarchical permissions:
  - **Lead Investor**: Full system access
  - **Senior Analyst**: Advanced analytics and strategy management
  - **Junior Analyst**: Basic analytics and bet proposals
  - **Contributing Member**: Voting rights and basic access
  - **Observer**: Read-only access

### 3. **Security Middleware**
- **Authentication Middleware**: Validates JWT tokens on protected routes
- **Rate Limiting**: Prevents abuse with configurable limits
- **Security Headers**: Adds security headers to all responses
- **Audit Logging**: Tracks all sensitive operations
- **CORS Handling**: Configurable cross-origin resource sharing

### 4. **Data Encryption**
- Field-level encryption for sensitive data
- Secure storage of member credentials
- Encrypted financial transaction data
- Backup and restore with encryption

### 5. **Audit Trail**
- Comprehensive logging of all operations
- Financial transaction tracking
- Failed login attempts
- Permission denied events
- Data access logging

## Installation

```bash
# Install required dependencies
pip install cryptography pyjwt fastapi

# Set environment variables
export JWT_SECRET_KEY="your-secret-key"
export SYNDICATE_MASTER_KEY="your-master-encryption-key"
```

## Quick Start

```python
from fastapi import FastAPI, Depends
from src.mcp.auth import (
    AuthenticationMiddleware,
    auth_router,
    get_current_member,
    require_permission
)

# Create FastAPI app
app = FastAPI()

# Add authentication middleware
app.add_middleware(AuthenticationMiddleware)

# Include auth routes
app.include_router(auth_router)

# Protected endpoint example
@app.get("/api/protected")
@require_permission("view_bets")
async def protected_endpoint(current_member = Depends(get_current_member)):
    return {"member_id": current_member.member_id}
```

## API Endpoints

### Authentication Endpoints

- `POST /auth/register` - Register new member
- `POST /auth/login` - Login and receive JWT token
- `POST /auth/logout` - Logout and invalidate session
- `POST /auth/refresh` - Refresh expiring token
- `POST /auth/change-password` - Change password
- `GET /auth/me` - Get current member info
- `GET /auth/audit-log` - View audit logs (Lead Investor only)

## Permission Decorators

### Role-Based Access
```python
from src.mcp.auth import require_role
from src.syndicate.member_management import MemberRole

@app.post("/api/admin-action")
@require_role(MemberRole.LEAD_INVESTOR)
async def admin_only_action():
    # Only Lead Investors can access
    pass
```

### Permission-Based Access
```python
from src.mcp.auth import require_permission

@app.post("/api/distribute-profits")
@require_permission("distribute_profits")
async def distribute_profits():
    # Only members with distribute_profits permission
    pass
```

## Security Best Practices

1. **Environment Variables**: Never hardcode secrets
2. **HTTPS Only**: Always use HTTPS in production
3. **Token Expiry**: Set appropriate token expiration times
4. **Rate Limiting**: Configure limits based on your needs
5. **Audit Logs**: Regularly review audit logs
6. **Password Policy**: Enforce strong password requirements

## Rate Limits

Default rate limits:
- Login: 5 attempts per 5 minutes
- Fund transfers: 10 per hour
- API general: 100 per minute
- Bet placement: 20 per minute

## Audit Log Actions

The system tracks these actions:
- `LOGIN` - Successful login
- `LOGOUT` - User logout
- `FAILED_LOGIN` - Failed login attempt
- `TOKEN_REFRESH` - Token refresh
- `PERMISSION_DENIED` - Access denied
- `FUND_TRANSFER` - Financial operations
- `VOTE_CAST` - Voting actions
- `MEMBER_ADDED` - New member registration
- `MEMBER_REMOVED` - Member removal
- `ROLE_CHANGED` - Role modifications
- `STRATEGY_MODIFIED` - Strategy changes
- `BET_PLACED` - Bet placement
- `PROFIT_DISTRIBUTED` - Profit distribution
- `DATA_ACCESS` - Sensitive data access
- `SETTINGS_CHANGED` - Settings modifications

## Configuration

### JWT Configuration
```python
authenticator = SyndicateAuthenticator(
    secret_key="your-secret-key",
    token_expiry_hours=24  # Default: 24 hours
)
```

### Middleware Configuration
```python
# Exclude paths from authentication
auth_middleware = AuthenticationMiddleware(
    app,
    exclude_paths=["/health", "/docs", "/auth/login"]
)
```

## Error Handling

The system provides specific error responses:
- `401 Unauthorized` - Invalid or missing token
- `403 Forbidden` - Insufficient permissions
- `429 Too Many Requests` - Rate limit exceeded

## Security Considerations

1. **Token Storage**: Store tokens securely on client side
2. **Refresh Strategy**: Implement automatic token refresh
3. **Logout**: Implement proper logout with token blacklisting
4. **Monitoring**: Monitor failed login attempts
5. **Encryption**: Enable encryption for sensitive data
6. **Backup**: Regular encrypted backups of credentials

## Support

For issues or questions about the authentication system:
1. Check audit logs for detailed error information
2. Verify permissions and roles are correctly configured
3. Ensure environment variables are properly set
4. Review rate limiting if experiencing 429 errors