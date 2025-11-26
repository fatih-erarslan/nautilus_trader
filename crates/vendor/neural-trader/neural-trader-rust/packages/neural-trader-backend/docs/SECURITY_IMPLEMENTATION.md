# Security Implementation - Neural Trader Backend

## Overview

Comprehensive security infrastructure has been implemented for the Neural Trader backend to protect financial trading operations. This document describes all security features, usage, and best practices.

## 🔐 Security Components

### 1. Authentication System (`src/auth.rs`)

**Features:**
- **API Key Authentication**: Generate and validate API keys with `ntk_` prefix
- **JWT Token Support**: Generate and validate JSON Web Tokens for session management
- **Role-Based Access Control (RBAC)**: Four permission levels
  - `ReadOnly`: View-only access
  - `User`: Standard trading capabilities
  - `Admin`: Full administrative access
  - `Service`: System service accounts

**API Functions:**

```javascript
// Initialize authentication
await initAuth(jwtSecret?)

// Create API key
const apiKey = await createApiKey(username, role, rateLimit?, expiresInDays?)

// Validate API key
const user = await validateApiKey(apiKey)

// Generate JWT token
const token = await generateToken(apiKey)

// Validate JWT token
const user = await validateToken(token)

// Check authorization
const allowed = await checkAuthorization(apiKey, operation, requiredRole)

// Revoke API key
await revokeApiKey(apiKey)
```

**Example Usage:**

```javascript
// Create admin API key
const adminKey = await createApiKey('admin_user', 'admin', 1000, 365);

// Validate before operations
const user = await validateApiKey(adminKey);
console.log(`Authenticated as: ${user.username}, role: ${user.role}`);

// Check permissions
const canTrade = await checkAuthorization(adminKey, 'execute_trade', 'user');
if (canTrade) {
  // Execute trade
}
```

### 2. Rate Limiting (`src/rate_limit.rs`)

**Features:**
- **Token Bucket Algorithm**: Smooth rate limiting with burst support
- **Per-Identifier Limits**: Rate limit by API key, IP, or user ID
- **DDoS Protection**: Automatic detection and blocking of suspicious activity
- **Configurable Limits**: Set custom limits per identifier

**API Functions:**

```javascript
// Initialize rate limiter
await initRateLimiter({
  maxRequestsPerMinute: 100,
  burstSize: 10,
  windowDurationSecs: 60
})

// Check rate limit
const allowed = await checkRateLimit(identifier, tokens?)

// Get statistics
const stats = await getRateLimitStats(identifier)

// Reset rate limit (admin)
await resetRateLimit(identifier)

// DDoS protection
const safe = await checkDdosProtection(ipAddress, requestCount?)

// Block/unblock IPs (admin)
await blockIp(ipAddress)
await unblockIp(ipAddress)
const blockedIps = await getBlockedIps()

// Cleanup old entries
await cleanupRateLimiter()
```

**Example Usage:**

```javascript
// Check rate limit before processing
try {
  await checkRateLimit(apiKey, 1.0); // 1 token
  await executeTradeOperation();
} catch (error) {
  // Rate limit exceeded
  console.error('Rate limit exceeded:', error.message);
}

// Check DDoS protection
try {
  await checkDdosProtection(clientIp, 1);
  // Process request
} catch (error) {
  // Suspicious activity detected
  console.error('DDoS protection:', error.message);
}
```

### 3. Audit Logging (`src/audit.rs`)

**Features:**
- **Structured Logging**: Comprehensive event tracking
- **Sensitive Data Masking**: Automatic masking of passwords, API keys, etc.
- **Event Categories**: Authentication, authorization, trading, portfolio, etc.
- **Severity Levels**: Info, Warning, Security, Error, Critical
- **Compliance Ready**: Full audit trail for regulatory requirements

**API Functions:**

```javascript
// Initialize audit logger
await initAuditLogger(maxEvents?, logToConsole?, logToFile?)

// Log audit event
const eventId = await logAuditEvent(
  level,        // 'info' | 'warning' | 'security' | 'error' | 'critical'
  category,     // 'authentication' | 'trading' | 'portfolio' | etc.
  action,       // 'execute_trade' | 'create_portfolio' | etc.
  outcome,      // 'success' | 'failure'
  userId?,
  username?,
  ipAddress?,
  resource?,
  details?      // JSON string
)

// Get recent events
const events = await getAuditEvents(limit?)

// Get statistics
const stats = await getAuditStatistics()

// Clear log (admin only)
await clearAuditLog()
```

**Example Usage:**

```javascript
// Log successful trade
await logAuditEvent(
  'info',
  'trading',
  'execute_trade',
  'success',
  user.userId,
  user.username,
  clientIp,
  'AAPL',
  JSON.stringify({ quantity: 100, price: 150.00 })
);

// Log failed authentication
await logAuditEvent(
  'security',
  'authentication',
  'login_attempt',
  'failure',
  null,
  null,
  clientIp,
  null,
  JSON.stringify({ reason: 'Invalid API key' })
);

// Get audit statistics
const stats = await getAuditStatistics();
console.log(`Success rate: ${stats.success_rate}%`);
```

### 4. Input Validation & Sanitization (`src/middleware.rs`)

**Features:**
- **SQL Injection Prevention**: Detect and sanitize SQL patterns
- **XSS Prevention**: HTML encoding of dangerous characters
- **Path Traversal Prevention**: Block directory traversal attempts
- **Input Validation**: Validate trading parameters, emails, numbers
- **Threat Detection**: Identify security threats in input

**API Functions:**

```javascript
// Sanitize input
const result = await sanitizeInput(input)
// Returns: { original, sanitized, threats_detected, is_safe }

// Validate trading parameters
const valid = await validateTradingParams(symbol, quantity, price?)

// Validate email format
const validEmail = await validateEmailFormat(email)

// Validate API key format
const validKey = await validateApiKeyFormat(key)

// Check for security threats
const threats = await checkSecurityThreats(input)
```

**Example Usage:**

```javascript
// Sanitize user input
const result = await sanitizeInput(userInput);
if (!result.is_safe) {
  console.warn('Security threats detected:', result.threats_detected);
  // Use sanitized version
  const cleanInput = result.sanitized;
}

// Validate trading request
try {
  await validateTradingParams('AAPL', 100, 150.00);
  // Parameters are valid
} catch (error) {
  console.error('Invalid parameters:', error.message);
}

// Check for threats
const threats = await checkSecurityThreats(suspiciousInput);
if (threats.length > 0) {
  console.warn('Detected threats:', threats);
}
```

### 5. Security Configuration (`src/security_config.rs`)

**Features:**
- **CORS Configuration**: Manage allowed origins, methods, headers
- **Security Headers**: HSTS, X-Frame-Options, CSP, etc.
- **IP Filtering**: Whitelist/blacklist IP addresses
- **HTTPS Enforcement**: Require secure connections

**API Functions:**

```javascript
// Initialize security config
await initSecurityConfig(corsConfig?, requireHttps?)

// Get CORS headers
const headers = await getCorsHeaders(origin?)

// Get security headers
const secHeaders = await getSecurityHeaders()

// Check IP allowed
const allowed = await checkIpAllowed(ipAddress)

// Check CORS origin
const corsAllowed = await checkCorsOrigin(origin)

// Manage IP blacklist (admin)
await addIpToBlacklist(ipAddress)
await removeIpFromBlacklist(ipAddress)
```

**Example CORS Configuration:**

```javascript
await initSecurityConfig({
  allowed_origins: ['https://app.example.com', 'https://admin.example.com'],
  allowed_methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowed_headers: ['Content-Type', 'Authorization', 'X-API-Key'],
  exposed_headers: ['X-RateLimit-Limit', 'X-RateLimit-Remaining'],
  allow_credentials: true,
  max_age: 3600
}, true); // Require HTTPS
```

## 🛡️ Security Best Practices

### 1. API Key Management

```javascript
// ✅ GOOD: Store API keys securely
const apiKey = await createApiKey('user', 'user', 100, 90);
// Store in secure environment variable or secret manager
process.env.NEURAL_TRADER_API_KEY = apiKey;

// ❌ BAD: Hardcode API keys
const apiKey = 'ntk_123456789...'; // Don't do this!
```

### 2. Rate Limiting Strategy

```javascript
// ✅ GOOD: Check rate limit before expensive operations
async function executeTrade(apiKey, params) {
  await checkRateLimit(apiKey, 5.0); // Higher cost for trades
  await checkDdosProtection(clientIp);
  return await performTrade(params);
}

// ✅ GOOD: Different limits for different operations
await checkRateLimit(apiKey, 0.1); // Low cost for reads
await checkRateLimit(apiKey, 5.0); // High cost for writes
```

### 3. Comprehensive Audit Logging

```javascript
// ✅ GOOD: Log all critical operations
async function executeTradeWithAudit(user, trade) {
  try {
    const result = await executeTrade(trade);
    await logAuditEvent('info', 'trading', 'execute_trade', 'success',
      user.userId, user.username, user.ip, trade.symbol,
      JSON.stringify({ quantity: trade.quantity, price: trade.price }));
    return result;
  } catch (error) {
    await logAuditEvent('error', 'trading', 'execute_trade', 'failure',
      user.userId, user.username, user.ip, trade.symbol,
      JSON.stringify({ error: error.message }));
    throw error;
  }
}
```

### 4. Input Validation

```javascript
// ✅ GOOD: Always sanitize and validate user input
async function handleUserInput(input) {
  // Sanitize first
  const sanitized = await sanitizeInput(input);
  if (!sanitized.is_safe) {
    throw new Error('Invalid input detected');
  }

  // Then validate
  if (sanitized.sanitized.includes('bad_pattern')) {
    throw new Error('Input validation failed');
  }

  return sanitized.sanitized;
}
```

### 5. Multi-Layer Security

```javascript
// ✅ GOOD: Apply multiple security layers
async function secureTradeExecution(apiKey, clientIp, tradeParams) {
  // 1. Authenticate
  const user = await validateApiKey(apiKey);

  // 2. Check authorization
  const authorized = await checkAuthorization(apiKey, 'execute_trade', 'user');
  if (!authorized) throw new Error('Forbidden');

  // 3. Rate limiting
  await checkRateLimit(apiKey, 5.0);
  await checkDdosProtection(clientIp);

  // 4. Input validation
  await validateTradingParams(tradeParams.symbol, tradeParams.quantity, tradeParams.price);

  // 5. Execute with audit logging
  try {
    const result = await executeTrade(tradeParams);
    await logAuditEvent('info', 'trading', 'execute_trade', 'success',
      user.userId, user.username, clientIp, tradeParams.symbol,
      JSON.stringify(tradeParams));
    return result;
  } catch (error) {
    await logAuditEvent('error', 'trading', 'execute_trade', 'failure',
      user.userId, user.username, clientIp, tradeParams.symbol,
      JSON.stringify({ error: error.message }));
    throw error;
  }
}
```

## 🔒 Production Deployment Checklist

- [ ] **JWT Secret**: Set strong JWT secret via `JWT_SECRET` environment variable
- [ ] **HTTPS Only**: Enable `requireHttps: true` in security config
- [ ] **Rate Limits**: Configure appropriate rate limits for production load
- [ ] **Audit Logging**: Enable file-based audit logging with rotation
- [ ] **IP Filtering**: Configure IP whitelist if needed
- [ ] **CORS**: Restrict allowed origins to production domains only
- [ ] **API Keys**: Generate production API keys with appropriate expiration
- [ ] **Monitoring**: Set up monitoring for security events and rate limit violations
- [ ] **Backup**: Configure audit log backup and retention
- [ ] **Testing**: Run comprehensive security tests before deployment

## 📊 Monitoring & Metrics

### Rate Limit Statistics

```javascript
const stats = await getRateLimitStats(identifier);
console.log(`
  Available tokens: ${stats.tokens_available}
  Total requests: ${stats.total_requests}
  Blocked requests: ${stats.blocked_requests}
  Success rate: ${stats.success_rate}%
`);
```

### Audit Statistics

```javascript
const stats = await getAuditStatistics();
console.log(`
  Total events: ${stats.total_events}
  Failed operations: ${stats.failed_operations}
  Success rate: ${stats.success_rate}%
  Events by level: ${JSON.stringify(stats.events_by_level)}
`);
```

### Blocked IPs

```javascript
const blockedIps = await getBlockedIps();
console.log(`Currently blocked IPs: ${blockedIps.length}`);
blockedIps.forEach(ip => console.log(`  - ${ip}`));
```

## 🚨 Incident Response

### Suspicious Activity Detected

1. **Check audit logs** for patterns:
   ```javascript
   const events = await getAuditEvents(1000);
   const failures = events.filter(e => e.outcome === 'failure');
   ```

2. **Review rate limit violations**:
   ```javascript
   const stats = await getRateLimitStats(suspiciousIdentifier);
   ```

3. **Block malicious actors**:
   ```javascript
   await blockIp(maliciousIp);
   await revokeApiKey(compromisedKey);
   ```

### Data Breach Response

1. **Immediate Actions**:
   - Revoke all API keys
   - Review audit logs for unauthorized access
   - Change JWT secret
   - Block suspicious IPs

2. **Investigation**:
   ```javascript
   const auditEvents = await getAuditEvents(10000);
   const securityEvents = auditEvents.filter(e =>
     e.level === 'security' || e.level === 'critical'
   );
   ```

3. **Recovery**:
   - Generate new API keys for legitimate users
   - Update all client applications
   - Monitor for continued suspicious activity

## 📝 Error Handling

All security operations return proper error types:

- `Unauthorized`: Invalid or missing credentials
- `Forbidden`: Valid credentials but insufficient permissions
- `RateLimited`: Rate limit exceeded
- `Authentication`: Authentication failure
- `Authorization`: Authorization failure

```javascript
try {
  await validateApiKey(apiKey);
} catch (error) {
  if (error.message.includes('Unauthorized')) {
    // Handle invalid credentials
  } else if (error.message.includes('Rate limit')) {
    // Handle rate limiting
  }
}
```

## 🔧 Maintenance

### Regular Tasks

1. **Cleanup old entries** (daily):
   ```javascript
   await cleanupRateLimiter();
   ```

2. **Review audit logs** (weekly):
   ```javascript
   const stats = await getAuditStatistics();
   // Analyze trends and anomalies
   ```

3. **Rotate API keys** (quarterly):
   ```javascript
   // Generate new keys
   // Migrate users
   // Revoke old keys
   ```

4. **Update IP blacklist** (as needed):
   ```javascript
   await addIpToBlacklist(newMaliciousIp);
   ```

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [API Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

## 🎯 Summary

The Neural Trader backend now includes:

✅ **Authentication**: API keys + JWT tokens with RBAC
✅ **Rate Limiting**: Token bucket with DDoS protection
✅ **Audit Logging**: Comprehensive audit trail with masking
✅ **Input Validation**: SQL injection, XSS, path traversal prevention
✅ **Security Config**: CORS, security headers, IP filtering
✅ **Error Handling**: Proper security error types
✅ **Production Ready**: All security features integrated

**Critical for Production**: This is real financial trading software. All security features must be properly configured and monitored in production environments.
