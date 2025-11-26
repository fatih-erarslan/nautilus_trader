# Neural Trader by rUv - JWT Authentication Guide

## Overview

The Neural Trader platform includes **optional JWT authentication** that can be enabled to secure API endpoints. By default, authentication is **disabled** to allow easy access, but it can be enabled at any time for production deployments.

## Features

- 🔐 **JWT Bearer Token Authentication**
- 🔑 **Optional API Key Support**
- ⚡ **Environment-based Configuration**
- 🛡️ **Secure Password Hashing (bcrypt)**
- ⏱️ **Configurable Token Expiration**
- 🎯 **Per-Endpoint Protection Control**

## Quick Start

### 1. Enable Authentication

Set the environment variable on Fly.io:
```bash
fly secrets set AUTH_ENABLED=true --app ai-news-trader
```

### 2. Configure Credentials

```bash
fly secrets set \
  AUTH_ENABLED=true \
  JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  AUTH_USERNAME="your-username" \
  AUTH_PASSWORD="your-secure-password" \
  --app ai-news-trader
```

### 3. Obtain JWT Token

```bash
curl -X POST https://neural-trader.ruv.io/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your-username","password":"your-password"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "auth_enabled": true
}
```

### 4. Use Token in Requests

```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  https://neural-trader.ruv.io/trading/status
```

## Authentication Endpoints

### POST `/auth/login`
Login and obtain JWT token.

**Request:**
```json
{
  "username": "admin",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 86400,
  "auth_enabled": true
}
```

### GET `/auth/status`
Check authentication status and current user.

**Response:**
```json
{
  "enabled": true,
  "authenticated": true,
  "username": "admin",
  "auth_type": "jwt"
}
```

### POST `/auth/verify`
Verify a JWT token (requires authentication).

**Request:**
```json
{
  "token": "eyJhbGc..."
}
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH_ENABLED` | Enable/disable authentication | `false` |
| `JWT_SECRET_KEY` | Secret key for JWT signing | `your-secret-key-change-in-production` |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `JWT_EXPIRATION_HOURS` | Token expiration time | `24` |
| `AUTH_USERNAME` | Default username | `admin` |
| `AUTH_PASSWORD` | Default password | `changeme` |
| `API_KEY` | Optional API key (alternative to JWT) | `""` |

### Generate Secure Secret Key

```bash
# Generate a secure 32-byte hex key
openssl rand -hex 32
```

## Authentication Modes

### 1. Disabled (Default)
- All endpoints accessible without authentication
- `AUTH_ENABLED=false`

### 2. Optional Authentication
- Endpoints work with or without authentication
- Authenticated users get additional features/logging
- Current implementation mode

### 3. Required Authentication
- Can be configured per-endpoint
- Use `check_auth_required` dependency

## API Key Authentication

As an alternative to JWT, you can use a simple API key:

```bash
# Set API key
fly secrets set API_KEY="your-api-key-here" --app ai-news-trader

# Use API key in requests
curl -H "Authorization: Bearer your-api-key-here" \
  https://neural-trader.ruv.io/trading/status
```

## Security Best Practices

1. **Always use HTTPS** in production
2. **Generate strong secret keys** using cryptographic methods
3. **Rotate keys regularly** (monthly recommended)
4. **Use strong passwords** (min 12 chars, mixed case, numbers, symbols)
5. **Set appropriate token expiration** (24 hours default)
6. **Monitor authentication logs** for suspicious activity

## Testing Authentication

### Test Script
```bash
# Run the included test script
./scripts/test-auth.sh
```

### Manual Testing
```bash
# 1. Check if auth is enabled
curl https://neural-trader.ruv.io/auth/status

# 2. Login
curl -X POST https://neural-trader.ruv.io/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your-password"}'

# 3. Use token
TOKEN="your-jwt-token"
curl -H "Authorization: Bearer $TOKEN" \
  https://neural-trader.ruv.io/trading/start \
  -H "Content-Type: application/json" \
  -d '{"strategies":["momentum_trader"],"symbols":["SPY"]}'
```

## Troubleshooting

### Token Expired
- Tokens expire after 24 hours by default
- Login again to get a new token

### Invalid Credentials
- Check username and password are correct
- Ensure environment variables are set properly

### Authentication Not Working
- Verify `AUTH_ENABLED=true` is set
- Check JWT_SECRET_KEY is configured
- Restart the application after changing secrets

## Local Development

### .env File
```env
AUTH_ENABLED=true
JWT_SECRET_KEY=dev-secret-key-change-in-production
AUTH_USERNAME=admin
AUTH_PASSWORD=admin123
```

### Run Locally
```bash
# Load environment variables
source .env

# Start the application
python -m uvicorn src.main:app --reload
```

## Integration Examples

### Python
```python
import requests

# Login
response = requests.post(
    "https://neural-trader.ruv.io/auth/login",
    json={"username": "admin", "password": "password"}
)
token = response.json()["access_token"]

# Use token
headers = {"Authorization": f"Bearer {token}"}
trading_status = requests.get(
    "https://neural-trader.ruv.io/trading/status",
    headers=headers
)
```

### JavaScript
```javascript
// Login
const loginResponse = await fetch('https://neural-trader.ruv.io/auth/login', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({username: 'admin', password: 'password'})
});
const {access_token} = await loginResponse.json();

// Use token
const tradingResponse = await fetch('https://neural-trader.ruv.io/trading/status', {
  headers: {'Authorization': `Bearer ${access_token}`}
});
```

## Support

For issues or questions about authentication:
- Check the logs: `fly logs --app ai-news-trader`
- Review configuration: `fly secrets list --app ai-news-trader`
- Test endpoints: `./scripts/test-auth.sh`

---

**Note:** Authentication is **disabled by default** to allow easy API access. Enable it for production deployments requiring security.