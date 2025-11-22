# JWT Security Configuration Guide

## 🚨 CRITICAL SECURITY NOTICE

The JWT secret configuration has been updated to eliminate security vulnerabilities. **ALL DEPLOYMENTS MUST FOLLOW THESE REQUIREMENTS.**

## Required Environment Configuration

### JWT_SECRET Environment Variable

**MANDATORY**: Set the `JWT_SECRET` environment variable with a cryptographically secure key.

```bash
# Generate a secure 64-character JWT secret
JWT_SECRET=$(openssl rand -base64 48)
export JWT_SECRET

# Or use a password generator
JWT_SECRET=$(head -c 32 /dev/urandom | base64)
export JWT_SECRET
```

### Security Requirements

The system enforces the following security policies for JWT_SECRET:

1. **Minimum Length**: 32 characters
2. **No Insecure Patterns**: Cannot contain common insecure strings like:
   - "secret", "password", "key", "test", "dev"
   - "demo", "example", "default", "changeme"
   - "admin", "root", "user", "123456"

### Production Deployment

**For Docker:**
```dockerfile
ENV JWT_SECRET=your-cryptographically-secure-64-character-secret-here
```

**For Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jwt-secret
data:
  JWT_SECRET: <base64-encoded-secret>
```

**For systemd service:**
```ini
[Service]
Environment=JWT_SECRET=your-cryptographically-secure-64-character-secret-here
```

## Error Handling

If JWT_SECRET is not properly configured, the system will:

1. **Fail to start** with clear error messages
2. **Log security warnings** for invalid secrets
3. **Reject all authentication attempts** until properly configured

## Testing

To test JWT configuration:

```bash
# Set a test secret (minimum 32 chars)
export JWT_SECRET="test-secret-that-is-exactly-32-chars-long-for-security"

# Start the service
cargo run

# Check logs for JWT validation
```

## Security Audit Compliance

This configuration addresses:
- **SEC-001**: No hardcoded secrets policy
- **SEC-003**: Environment variable security requirements
- **SEC-005**: Cryptographic key management standards

## Migration from Previous Versions

If upgrading from a version with hardcoded fallbacks:

1. **IMMEDIATELY** set JWT_SECRET environment variable
2. **RESTART** all running instances
3. **VERIFY** no hardcoded secrets remain in code
4. **ROTATE** any JWT tokens issued with old secrets

## Support

For security questions or issues:
- Review security policies in `/security/policies/`
- Run CQGS security scan: `.cqgs/hooks/post-edit.sh`
- Contact security team for production deployments