# @neural-trader/backend

High-performance Neural Trader backend with native Rust bindings via NAPI-RS.

## Features

- 🚀 **High Performance**: Native Rust implementation for maximum speed
- 📊 **Trading Algorithms**: Advanced technical indicators and strategies
- 💾 **Memory Efficient**: Optimized memory usage with Rust's zero-cost abstractions
- 🔒 **Type Safe**: Full TypeScript type definitions
- 🌍 **Multi-Platform**: Pre-built binaries for major platforms

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux | x64 | ✅ Supported |
| Linux | ARM64 | ✅ Supported |
| macOS | x64 (Intel) | ✅ Supported |
| macOS | ARM64 (Apple Silicon) | ✅ Supported |
| Windows | x64 | ✅ Supported |
| Windows | ARM64 | 🚧 Experimental |

## Installation

```bash
npm install @neural-trader/backend
```

The package will automatically install the correct platform-specific binary for your system.

## Usage

```javascript
const backend = require('@neural-trader/backend');

// Example: Calculate technical indicators
const prices = [100, 102, 101, 103, 105, 104, 106];
const indicators = backend.calculateTechnicalIndicators(prices);

console.log('Indicators:', indicators);

// Example: Run backtest
const strategy = {
  type: 'momentum',
  params: {
    shortPeriod: 10,
    longPeriod: 20
  }
};

const backtestResults = backend.runBacktest(strategy, prices);
console.log('Backtest results:', backtestResults);
```

## Building from Source

If pre-built binaries are not available for your platform:

```bash
# Clone the repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust/packages/neural-trader-backend

# Install dependencies
npm install

# Build for your platform
npm run build

# Run tests
npm test
```

## Security

### Environment Variables (REQUIRED)

Before running this package, you MUST configure these security-critical environment variables:

#### 1. JWT_SECRET (REQUIRED)
```bash
# Generate a secure 64-byte secret
openssl rand -hex 64

# Set in your environment
export JWT_SECRET="your-generated-secret-here"
```

**⚠️ CRITICAL:** The application will refuse to start if JWT_SECRET is not set. Never use default or weak secrets in production.

#### 2. API Key Management
```bash
# API keys should be generated with high entropy
export API_KEY_MASTER="your-secure-master-key"
```

### Security Features

This package includes multiple layers of security:

- **Authentication & Authorization**
  - JWT token-based authentication
  - Role-Based Access Control (RBAC)
  - API key validation with expiration
  - Session management

- **Input Validation**
  - SQL injection prevention
  - XSS sanitization
  - Path traversal prevention
  - Comprehensive input sanitization

- **Rate Limiting**
  - Token bucket algorithm
  - Per-API-key rate limiting
  - DDoS protection
  - Configurable limits

- **Audit Logging**
  - All security events logged
  - Failed authentication attempts tracked
  - API access audited

### Security Best Practices

1. **Always use HTTPS** in production
2. **Rotate secrets regularly** (JWT_SECRET, API keys)
3. **Use parameterized queries** - never concatenate SQL
4. **Enable rate limiting** for all public endpoints
5. **Monitor audit logs** for suspicious activity
6. **Keep dependencies updated** - run `npm audit` regularly

### Reporting Security Vulnerabilities

If you discover a security vulnerability, please email security@neural-trader.com instead of using the issue tracker.

## License

MIT
