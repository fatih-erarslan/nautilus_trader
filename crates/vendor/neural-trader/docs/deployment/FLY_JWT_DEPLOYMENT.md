# Deploying with JWT Authentication to Fly.io

## Overview
This guide explains how to securely deploy the AI News Trader API with JWT authentication enabled on Fly.io.

## Quick Setup

### 1. Set JWT Secret and Enable Authentication
```bash
# Generate and set a secure JWT secret
fly secrets set \
  JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  AUTH_ENABLED="true" \
  JWT_ALGORITHM="HS256" \
  JWT_EXPIRATION_HOURS="24" \
  --app ruvtrade
```

### 2. Set Admin Credentials
```bash
# Set admin username and password
fly secrets set \
  AUTH_USERNAME="admin" \
  AUTH_PASSWORD="YourSecurePassword123!" \
  --app ruvtrade
```

### 3. Set Trading API Keys (Optional)
```bash
# Set your actual trading API keys
fly secrets set \
  ALPACA_API_KEY="your-alpaca-key" \
  ALPACA_SECRET_KEY="your-alpaca-secret" \
  NEWS_API_KEY="your-news-api-key" \
  FINNHUB_API_KEY="your-finnhub-key" \
  --app ruvtrade
```

## Verification

### Check Secrets Are Set
```bash
fly secrets list --app ruvtrade
```

Expected output:
```
NAME                    DIGEST                  CREATED AT
AUTH_ENABLED           abc123...               1m ago
AUTH_PASSWORD          def456...               1m ago  
AUTH_USERNAME          ghi789...               1m ago
JWT_ALGORITHM          jkl012...               1m ago
JWT_EXPIRATION_HOURS   mno345...               1m ago
JWT_SECRET_KEY         pqr678...               1m ago
```

### Test Authentication

1. **Get JWT Token**:
```bash
curl -X POST https://ruvtrade.fly.dev/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=YourSecurePassword123!"
```

2. **Use Token for Protected Endpoints**:
```bash
TOKEN="your-jwt-token-here"
curl -X GET https://ruvtrade.fly.dev/strategies/list \
  -H "Authorization: Bearer $TOKEN"
```

## Security Best Practices

### JWT Secret Key
- **Generate securely**: Use `openssl rand -hex 32` for 256-bit keys
- **Never commit**: Keep out of version control
- **Rotate regularly**: Change every 90 days
- **Unique per environment**: Different keys for dev/staging/prod

### Password Requirements
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, symbols
- No dictionary words
- Unique per deployment

### Environment-Specific Settings

#### Development (Local)
```env
AUTH_ENABLED=false
JWT_SECRET_KEY=dev-only-not-secure
```

#### Staging
```env
AUTH_ENABLED=true
JWT_SECRET_KEY=<staging-secret>
JWT_EXPIRATION_HOURS=24
```

#### Production
```env
AUTH_ENABLED=true
JWT_SECRET_KEY=<production-secret>
JWT_EXPIRATION_HOURS=12
```

## Deployment Commands

### Full Deployment with Auth
```bash
# 1. Generate new JWT secret
JWT_SECRET=$(openssl rand -hex 32)

# 2. Set all secrets at once
fly secrets set \
  JWT_SECRET_KEY="$JWT_SECRET" \
  AUTH_ENABLED="true" \
  JWT_ALGORITHM="HS256" \
  JWT_EXPIRATION_HOURS="24" \
  AUTH_USERNAME="admin" \
  AUTH_PASSWORD="SecurePass123!" \
  --app ruvtrade

# 3. Deploy application
fly deploy --app ruvtrade

# 4. Verify deployment
fly status --app ruvtrade
```

### Update Existing Deployment
```bash
# Just update the secrets (no redeploy needed)
fly secrets set JWT_SECRET_KEY="$(openssl rand -hex 32)" --app ruvtrade

# The app will automatically restart with new secrets
```

## Troubleshooting

### Authentication Not Working
1. Check if `AUTH_ENABLED` is set to `"true"` (as string)
2. Verify JWT_SECRET_KEY is set: `fly secrets list --app ruvtrade`
3. Check logs: `fly logs --app ruvtrade`

### Token Expired
- Default expiration is 24 hours
- Adjust with `JWT_EXPIRATION_HOURS`
- Implement token refresh endpoint if needed

### 401 Unauthorized Errors
- Ensure token format: `Authorization: Bearer <token>`
- Check token hasn't expired
- Verify credentials are correct

## API Endpoints with Auth

### Public Endpoints (No Auth Required)
- `GET /` - Root
- `GET /health` - Health check
- `POST /auth/token` - Get JWT token

### Protected Endpoints (Auth Required when enabled)
All other endpoints require JWT token when `AUTH_ENABLED=true`:
- `/strategies/*`
- `/analysis/*`
- `/neural/*`
- `/prediction/*`
- `/sports/*`
- `/syndicate/*`
- `/portfolio/*`
- `/news/*`
- `/system/*`

## Example: Full Authentication Flow

```python
import requests

# 1. Get token
auth_response = requests.post(
    "https://ruvtrade.fly.dev/auth/token",
    data={"username": "admin", "password": "YourPassword123!"}
)
token = auth_response.json()["access_token"]

# 2. Use token for protected endpoints
headers = {"Authorization": f"Bearer {token}"}

# Get strategies
strategies = requests.get(
    "https://ruvtrade.fly.dev/strategies/list",
    headers=headers
)

# Execute trade
trade = requests.post(
    "https://ruvtrade.fly.dev/execute",
    headers=headers,
    json={
        "strategy": "momentum",
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 10
    }
)
```

## Removing Authentication

To disable authentication temporarily:
```bash
fly secrets set AUTH_ENABLED="false" --app ruvtrade
```

## Security Checklist

- [ ] JWT secret is at least 256 bits (32 bytes hex)
- [ ] Passwords meet complexity requirements
- [ ] Secrets are set via Fly CLI, not in code
- [ ] Different secrets for each environment
- [ ] HTTPS is enforced (Fly.io default)
- [ ] Token expiration is appropriate
- [ ] Regular secret rotation scheduled
- [ ] Monitoring for failed auth attempts
- [ ] Rate limiting implemented
- [ ] Audit logging enabled

## Support

For issues or questions:
- Fly.io Status: https://status.fly.io/
- API Docs: https://ruvtrade.fly.dev/docs
- GitHub Issues: https://github.com/ruvnet/ai-news-trader/issues