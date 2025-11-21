# ATS-Core Production Security Deployment Guide

## 🔒 Critical Security Requirements

This document outlines the mandatory security configurations for deploying ATS-Core in production environments, particularly for financial trading systems.

## ⚠️ Security Vulnerabilities Addressed

### Fixed Critical Issues:
- **CRITICAL**: Removed hardcoded JWT secret `"your-secret-key-change-this"`
- **HIGH**: Eliminated all hardcoded credentials from configuration files
- **HIGH**: Implemented secure environment variable configuration system
- **MEDIUM**: Added production configuration validation
- **MEDIUM**: Created secure configuration templates

## 🚀 Production Deployment Checklist

### 1. Secret Management (MANDATORY)

**❌ NEVER do this in production:**
```rust
// DON'T - Hardcoded secrets
jwt: JwtConfig {
    secret: "your-secret-key-change-this".to_string(),
    // ...
}
```

**✅ ALWAYS do this instead:**
```rust
// DO - Environment-based configuration
jwt: JwtConfig {
    secret: std::env::var("ATS_JWT_SECRET").expect("ATS_JWT_SECRET must be set"),
    // ...
}
```

### 2. Environment Variables Configuration

**Required Production Environment Variables:**

```bash
# CRITICAL: JWT Configuration
ATS_JWT_SECRET=<64-character-secure-base64-secret>
ATS_JWT_EXPIRY_SECONDS=3600  # 1 hour for production
ATS_JWT_ALGORITHM=RS256      # Use RS256 for production

# CRITICAL: Encryption Keys
ATS_ENCRYPTION_KEY=<256-bit-hex-key>
ATS_ENCRYPT_AT_REST=true
ATS_ENCRYPT_IN_TRANSIT=true

# CRITICAL: Database Security
ATS_DB_PASSWORD=<secure-database-password>
ATS_DB_SSL_MODE=require

# Security Controls
ATS_RATE_LIMIT_RPM=100
ATS_MAX_BODY_SIZE=1048576
ATS_SECURITY_AUDIT_ENABLED=true
```

### 3. Secret Generation Commands

**Generate Secure JWT Secret:**
```bash
openssl rand -base64 64
```

**Generate Encryption Key:**
```bash
openssl rand -hex 32
```

**Generate Database Password:**
```bash
openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
```

### 4. Secrets Management Solutions

#### Option 1: HashiCorp Vault (Recommended)
```bash
# Setup Vault environment
export VAULT_ADDR=https://vault.yourdomain.com:8200
export VAULT_TOKEN_PATH=/var/secrets/vault-token

# Store secrets in Vault
vault kv put secret/ats-core jwt_secret="<secure-secret>"
vault kv put secret/ats-core encryption_key="<encryption-key>"
```

#### Option 2: AWS Secrets Manager
```bash
# Store secrets in AWS
aws secretsmanager create-secret \
    --name "ats-core/jwt-secret" \
    --secret-string "<secure-secret>"

# Configure environment
export AWS_REGION=us-west-2
export AWS_SECRET_PREFIX=ats-core
```

#### Option 3: Kubernetes Secrets
```bash
# Create Kubernetes secrets
kubectl create secret generic ats-jwt-secret \
    --from-literal=secret="<secure-secret>"

# Mount as environment variables
kubectl apply -f k8s-secrets.yaml
```

#### Option 4: Docker Secrets
```yaml
# docker-compose.production.yml
secrets:
  jwt_secret:
    external: true
    name: ats_jwt_secret_v1
services:
  ats-core:
    secrets:
      - jwt_secret
```

### 5. Production Security Configuration

**Use the ProductionSecurityManager:**

```rust
use ats_core::security_config::{ProductionSecurityManager, SecurityTemplates};

// Load production configuration
let security_manager = ProductionSecurityManager::new()?;

// Or use predefined template
let template = SecurityTemplates::production_financial();
let security_manager = ProductionSecurityManager::from_template(&template)?;
```

## 🛡️ Security Configuration Templates

### Production Financial Template
- **Security Level**: Critical
- **JWT Algorithm**: RS256 only
- **Token Expiry**: 1 hour maximum
- **Secret Length**: Minimum 64 characters
- **Encryption**: Required at rest and in transit
- **Rate Limiting**: 60 RPM

### Staging Template  
- **Security Level**: High
- **JWT Algorithm**: HS256/RS256
- **Token Expiry**: 2 hours maximum
- **Secret Length**: Minimum 32 characters

### Development Template
- **Security Level**: Standard
- **JWT Algorithm**: HS256 allowed
- **Token Expiry**: 24 hours maximum
- **Secrets**: Optional (generates defaults)

## 🔍 Configuration Validation

The system automatically validates:

1. **JWT Secret Strength**:
   - Minimum length requirements
   - No weak/default values
   - Algorithm compatibility

2. **Encryption Requirements**:
   - Encryption enabled for production
   - Strong key requirements

3. **Environment Detection**:
   - Prevents dev config in production
   - Validates required secrets

4. **Rate Limiting**:
   - Ensures rate limiting is enabled
   - Validates reasonable limits

## 🚨 Security Monitoring

### Audit Logging
```rust
// Security events are automatically logged
SecurityAuditEntry {
    event_type: SecurityEventType::AuthenticationFailure,
    client_ip: Some(client_ip),
    severity: SecuritySeverity::High,
    action_taken: SecurityAction::Block,
}
```

### Metrics Collection
- Authentication success/failure rates
- Rate limiting statistics  
- Input validation failures
- Security violations

## 🏗️ Deployment Architecture

### Container Security
```dockerfile
# Use non-root user
USER 1001:1001

# Read-only root filesystem
--read-only --tmpfs /tmp

# Drop capabilities
--cap-drop=ALL
```

### Network Security
- Use TLS 1.3 for all communications
- Implement proper certificate management
- Configure secure headers
- Enable HSTS, CSP, and other security headers

### Infrastructure Security
- Use private networks for internal communications
- Implement proper firewall rules
- Enable logging and monitoring
- Regular security updates

## 🧪 Security Testing

### Pre-deployment Testing
```bash
# Test configuration validation
cargo test security_config::tests

# Verify no hardcoded secrets
grep -r "secret\|password\|key.*=" src/ --include="*.rs"

# Check for weak configurations
cargo run --bin security-audit
```

### Penetration Testing
- JWT token validation
- Authentication bypass attempts
- Rate limiting effectiveness
- Input validation testing

## 📋 Compliance Requirements

### Financial Industry Standards
- **SOC 2 Type II**: Audit logging and access controls
- **PCI DSS**: If handling payment data
- **ISO 27001**: Information security management
- **SOX**: Financial reporting controls

### Required Security Controls
1. Multi-factor authentication
2. Encryption at rest and in transit
3. Audit logging and monitoring
4. Access control and authorization
5. Incident response procedures
6. Regular security assessments

## 🚀 Deployment Commands

### Docker Deployment
```bash
# Create secrets
echo "$(openssl rand -base64 64)" | docker secret create ats_jwt_secret_v1 -
echo "$(openssl rand -hex 32)" | docker secret create ats_encryption_key_v1 -

# Deploy stack
docker stack deploy -c docker-compose.production.yml ats-core
```

### Kubernetes Deployment
```bash
# Create secrets
kubectl create secret generic ats-secrets \
    --from-literal=jwt-secret="$(openssl rand -base64 64)" \
    --from-literal=encryption-key="$(openssl rand -hex 32)"

# Deploy application
kubectl apply -f k8s/production/
```

### Manual Deployment
```bash
# Set environment variables
export ATS_JWT_SECRET="$(openssl rand -base64 64)"
export ATS_ENCRYPTION_KEY="$(openssl rand -hex 32)"
export ATS_ENVIRONMENT=production

# Run application
./target/release/ats-core-server
```

## 🔄 Security Maintenance

### Regular Tasks
1. **Key Rotation**: Rotate encryption keys every 30 days
2. **Secret Updates**: Update JWT secrets quarterly
3. **Security Patches**: Apply updates within 7 days
4. **Audit Reviews**: Monthly security audit reviews
5. **Penetration Testing**: Quarterly security assessments

### Incident Response
1. **Detection**: Monitor security events and alerts
2. **Containment**: Isolate affected systems
3. **Investigation**: Analyze security logs
4. **Recovery**: Restore secure operations
5. **Learning**: Update security procedures

## 🆘 Emergency Procedures

### Compromise Response
```bash
# 1. Immediately rotate all secrets
./scripts/emergency-key-rotation.sh

# 2. Revoke all active JWT tokens
kubectl delete secret ats-jwt-secret
kubectl create secret generic ats-jwt-secret --from-literal=secret="$(openssl rand -base64 64)"

# 3. Enable maximum security logging
export ATS_LOG_LEVEL=debug
export ATS_SECURITY_AUDIT_ENABLED=true

# 4. Restart all services
kubectl rollout restart deployment/ats-core
```

### Security Contacts
- **Security Team**: security@tengri.ai
- **Emergency**: +1-XXX-XXX-XXXX
- **Incident Response**: incidents@tengri.ai

---

## 📚 Additional Resources

- [OWASP Security Guidelines](https://owasp.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [AWS Secrets Manager Guide](https://docs.aws.amazon.com/secretsmanager/)
- [Kubernetes Secrets Management](https://kubernetes.io/docs/concepts/configuration/secret/)

**Remember**: Security is everyone's responsibility. When in doubt, choose the more secure option.