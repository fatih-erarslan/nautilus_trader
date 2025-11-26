# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of Neural Trader seriously. If you discover a security vulnerability, please follow these guidelines:

### Where to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, report them via:

1. **Email**: security@neural-trader.io
2. **GitHub Security Advisory**: https://github.com/ruvnet/neural-trader/security/advisories/new

### What to Include

When reporting a vulnerability, please include:

- **Description**: Detailed description of the vulnerability
- **Impact**: Potential impact and attack scenario
- **Steps to Reproduce**: Step-by-step instructions to reproduce
- **Affected Versions**: Versions affected by the vulnerability
- **Proof of Concept**: Code or commands demonstrating the issue
- **Suggested Fix**: (Optional) Potential ways to address the issue
- **Disclosure Timeline**: Your preferred timeline for disclosure

### Example Report

```
Subject: [SECURITY] SQL Injection in Portfolio Query

Description:
The portfolio query endpoint is vulnerable to SQL injection through the
'symbol' parameter, allowing unauthorized database access.

Impact:
An attacker could read, modify, or delete database records, potentially
accessing sensitive trading data or manipulating positions.

Steps to Reproduce:
1. Send POST request to /api/v1/portfolio/query
2. Include payload: {"symbol": "AAPL' OR '1'='1"}
3. Observe that all portfolio records are returned

Affected Versions:
All versions >= 0.1.0

Proof of Concept:
curl -X POST http://localhost:8080/api/v1/portfolio/query \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL'"'"' OR '"'"'1'"'"'='"'"'1"}'

Suggested Fix:
Use parameterized queries or an ORM that properly escapes input.
```

### Response Timeline

- **Initial Response**: Within 48 hours
- **Triage**: Within 1 week (severity assessment)
- **Fix Development**: Within 2-4 weeks (depending on severity)
- **Public Disclosure**: After patch release + 7 days

### Severity Levels

| Level    | Response Time | Criteria                                    |
|----------|--------------|---------------------------------------------|
| Critical | 24 hours     | Remote code execution, data breach          |
| High     | 1 week       | Authentication bypass, privilege escalation |
| Medium   | 2 weeks      | Data exposure, denial of service            |
| Low      | 4 weeks      | Information disclosure, minor issues        |

## Security Best Practices

### For Users

#### 1. API Keys and Secrets

**NEVER commit secrets to version control:**

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.*" >> .gitignore
echo "config/secrets.toml" >> .gitignore
```

**Use environment variables:**

```bash
# Set via environment
export ALPACA_API_KEY="your-key"
export ALPACA_API_SECRET="your-secret"

# Or use .env file (not committed)
cat > .env << EOF
ALPACA_API_KEY=your-key
ALPACA_API_SECRET=your-secret
EOF
```

**Rotate credentials regularly:**

```bash
# Every 90 days, generate new keys
# Update environment variables
# Verify old keys are deactivated
```

#### 2. Network Security

**Use TLS in production:**

```toml
[security]
enable_tls = true
tls_cert_path = "/path/to/cert.pem"
tls_key_path = "/path/to/key.pem"
```

**Configure firewall rules:**

```bash
# Allow only necessary ports
ufw allow 8080/tcp  # API
ufw allow 9090/tcp  # Metrics (internal only)
ufw enable
```

**Use reverse proxy with HTTPS:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.neural-trader.io;

    ssl_certificate /etc/letsencrypt/live/api.neural-trader.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.neural-trader.io/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 3. Access Control

**Require API key authentication:**

```toml
[security]
require_api_key = true
```

**Enable rate limiting:**

```toml
[security]
rate_limit_enabled = true
rate_limit_requests_per_minute = 100
```

**Configure CORS properly:**

```toml
[security]
cors_allowed_origins = ["https://app.neural-trader.io"]
```

#### 4. Database Security

**Use strong passwords:**

```bash
# Generate secure password
openssl rand -base64 32
```

**Enable SSL for PostgreSQL:**

```
# postgresql.conf
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
```

**Restrict network access:**

```
# pg_hba.conf
hostssl all all 10.0.0.0/8 md5
```

#### 5. Container Security

**Run as non-root user:**

```dockerfile
USER neural
```

**Scan images for vulnerabilities:**

```bash
docker scan neuraltrader/neural-trader-rust:latest
```

**Keep base images updated:**

```bash
docker pull rust:1.75-slim
docker pull debian:bookworm-slim
```

#### 6. Monitoring and Auditing

**Enable audit logging:**

```toml
[logging]
enable_audit_log = true
```

**Monitor suspicious activity:**

```bash
# Failed authentication attempts
grep "authentication failed" /var/log/neural-trader/audit.log

# Unusual order patterns
grep "high_risk_order" /var/log/neural-trader/audit.log
```

**Set up alerts:**

```yaml
# prometheus/alerts.yml
groups:
  - name: security
    rules:
      - alert: SuspiciousActivity
        expr: rate(failed_auth_total[5m]) > 10
        annotations:
          summary: "High rate of failed authentication"
```

### For Developers

#### 1. Dependency Management

**Audit dependencies regularly:**

```bash
cargo audit
```

**Keep dependencies updated:**

```bash
cargo update
```

**Review new dependencies:**

- Check crate popularity and maintenance
- Review source code for obvious issues
- Check for known vulnerabilities

#### 2. Code Security

**Use safe Rust practices:**

```rust
// ❌ Avoid unsafe blocks unless absolutely necessary
unsafe {
    // Dangerous operation
}

// ✅ Use safe alternatives
let result = safe_operation()?;
```

**Validate all inputs:**

```rust
// ✅ Always validate user input
fn process_order(symbol: &str) -> Result<()> {
    if !is_valid_symbol(symbol) {
        return Err(Error::InvalidSymbol);
    }
    // Process order
}
```

**Use prepared statements:**

```rust
// ✅ Use parameterized queries
sqlx::query("SELECT * FROM orders WHERE symbol = ?")
    .bind(symbol)
    .fetch_all(&pool)
    .await?;

// ❌ Never concatenate SQL
let query = format!("SELECT * FROM orders WHERE symbol = '{}'", symbol);
```

**Sanitize outputs:**

```rust
// ✅ Escape user content in responses
let safe_message = html_escape::encode_text(&user_message);
```

#### 3. Authentication & Authorization

**Implement proper authentication:**

```rust
async fn authenticate(api_key: &str) -> Result<User> {
    // Constant-time comparison
    if subtle::ConstantTimeEq::ct_eq(api_key.as_bytes(), expected.as_bytes()).into() {
        Ok(user)
    } else {
        Err(Error::Unauthorized)
    }
}
```

**Enforce authorization:**

```rust
async fn cancel_order(user: User, order_id: Uuid) -> Result<()> {
    let order = get_order(order_id).await?;

    // Verify ownership
    if order.user_id != user.id {
        return Err(Error::Forbidden);
    }

    // Proceed with cancellation
}
```

#### 4. Cryptography

**Use established libraries:**

```rust
// ✅ Use rust-crypto or ring
use ring::hmac;

// ❌ Don't roll your own crypto
```

**Store passwords securely:**

```rust
use argon2::{Argon2, PasswordHasher, PasswordVerifier};

// Hash password
let password_hash = Argon2::default()
    .hash_password(password.as_bytes(), &salt)?
    .to_string();

// Verify password
Argon2::default().verify_password(
    password.as_bytes(),
    &PasswordHash::new(&password_hash)?
)?;
```

#### 5. Testing

**Write security tests:**

```rust
#[tokio::test]
async fn test_sql_injection_protection() {
    let malicious_input = "AAPL' OR '1'='1";
    let result = query_portfolio(malicious_input).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_authorization_enforcement() {
    let user_a = create_test_user("a").await;
    let user_b = create_test_user("b").await;

    let order = create_order(&user_a).await;

    // User B should not be able to cancel User A's order
    let result = cancel_order(&user_b, order.id).await;
    assert_eq!(result.unwrap_err(), Error::Forbidden);
}
```

## Security Features

### Built-in Protections

1. **Input Validation**: All user inputs validated against schemas
2. **SQL Injection Protection**: Parameterized queries throughout
3. **XSS Prevention**: Output encoding for all user content
4. **CSRF Protection**: Token validation for state-changing operations
5. **Rate Limiting**: Configurable rate limits per endpoint
6. **Authentication**: API key or OAuth2 authentication
7. **Authorization**: Role-based access control (RBAC)
8. **TLS/SSL**: HTTPS support with configurable certificates
9. **Audit Logging**: Comprehensive audit trail
10. **Secrets Management**: Environment variable based configuration

### Security Headers

The following security headers are set by default:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

## Known Limitations

1. **API Keys in Memory**: API keys are stored in memory during runtime. Use secrets management systems (Vault, AWS Secrets Manager) for enhanced security.

2. **Local Database**: Default SQLite storage is not encrypted at rest. Use PostgreSQL with encryption for production.

3. **WebSocket Authentication**: WebSocket connections use bearer tokens which may be logged by proxies.

## Compliance

### Data Protection

- **GDPR**: User data can be exported and deleted on request
- **CCPA**: California residents have data rights
- **PCI DSS**: Not applicable (no credit card processing)

### Financial Regulations

- **SEC**: Not a registered broker-dealer
- **FINRA**: Not a FINRA member
- **MiFID II**: Not applicable (no EU operations)

**Disclaimer**: Neural Trader is a tool for algorithmic trading. Users are responsible for compliance with applicable financial regulations in their jurisdiction.

## Security Updates

Subscribe to security announcements:

- Watch the GitHub repository for security advisories
- Join our Discord security channel: https://discord.gg/neural-trader
- Follow @neuraltrader on Twitter for updates

## Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

<!-- Add researchers here after successful disclosure -->

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Rust Security Guidelines](https://anssi-fr.github.io/rust-guide/)
- [napi-rs Security](https://napi.rs/docs/security)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

## Contact

- **Security Email**: security@neural-trader.io
- **PGP Key**: [Download](https://neural-trader.io/security.asc)
- **Bug Bounty**: Coming soon

---

Last updated: 2024-11-12
