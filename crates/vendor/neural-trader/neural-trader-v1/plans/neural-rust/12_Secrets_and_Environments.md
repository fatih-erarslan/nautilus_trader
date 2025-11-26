# Secrets & Environment Management

## Document Overview

**Status**: Production-Ready Planning
**Last Updated**: 2025-11-12
**Owner**: Security & Operations Team
**Related Docs**: `08_Security_Governance_AIDefence_Lean.md`, `11_CLI_and_NPM_Release.md`

## Executive Summary

This document covers:
- **Environment Variables**: Complete list of required secrets and configuration
- **Secret Storage**: Multiple storage backends (environment, files, vaults)
- **Key Rotation**: Automated rotation procedures
- **Access Control**: Least privilege patterns
- **Cloud Integration**: AWS, GCP, Azure secret managers
- **Incident Response**: Handling compromised keys

---

## 1. Complete Environment Variable List

### 1.1 Exchange API Keys

**Alpaca (Stock/Crypto Trading):**

```bash
# Paper Trading
ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxxxxx"
ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ALPACA_API_URL="https://paper-api.alpaca.markets"

# Live Trading
ALPACA_API_KEY="AKxxxxxxxxxxxxxxxxxxxxx"
ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ALPACA_API_URL="https://api.alpaca.markets"
```

**Coinbase (Crypto):**

```bash
COINBASE_API_KEY="organizations/xxxxx/apiKeys/xxxxx"
COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\nMHc...\n-----END EC PRIVATE KEY-----"
```

**Binance (Crypto):**

```bash
BINANCE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BINANCE_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BINANCE_TESTNET="true"  # Use testnet (default: false)
```

**Interactive Brokers:**

```bash
IB_GATEWAY_HOST="localhost"
IB_GATEWAY_PORT="4001"
IB_ACCOUNT_ID="DU1234567"
IB_CLIENT_ID="1"
```

### 1.2 LLM API Keys

**OpenRouter (Multi-LLM Gateway):**

```bash
OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

**OpenAI:**

```bash
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_ORGANIZATION="org-xxxxxxxxxxxxxxxx"
```

**Anthropic:**

```bash
ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Google (Gemini):**

```bash
GOOGLE_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### 1.3 Infrastructure & Cloud

**E2B (Sandboxes):**

```bash
E2B_API_KEY="e2b_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**AWS:**

```bash
AWS_ACCESS_KEY_ID="AKIAxxxxxxxxxxxxxxxxxx"
AWS_SECRET_ACCESS_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
AWS_REGION="us-east-1"
AWS_S3_BUCKET="neural-trader-artifacts"
```

**Google Cloud Platform:**

```bash
GOOGLE_CLOUD_PROJECT="neural-trader-prod"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Azure:**

```bash
AZURE_SUBSCRIPTION_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
AZURE_TENANT_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
AZURE_CLIENT_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
AZURE_CLIENT_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 1.4 Database & Storage

**PostgreSQL:**

```bash
DATABASE_URL="postgresql://user:password@localhost:5432/neural_trader"
DATABASE_MAX_CONNECTIONS="20"
DATABASE_SSL_MODE="require"
```

**Redis:**

```bash
REDIS_URL="redis://localhost:6379/0"
REDIS_PASSWORD="xxxxxxxxxxxxxxxx"
REDIS_TLS="true"
```

**TimescaleDB:**

```bash
TIMESCALE_URL="postgresql://user:password@timescale.example.com:5432/neural_trader"
```

### 1.5 Monitoring & Observability

**Sentry (Error Tracking):**

```bash
SENTRY_DSN="https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx@o123456.ingest.sentry.io/123456"
SENTRY_ENVIRONMENT="production"
SENTRY_RELEASE="neural-trader@0.2.0"
```

**Datadog:**

```bash
DD_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DD_APP_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DD_SITE="datadoghq.com"
DD_SERVICE="neural-trader"
```

**Prometheus (Metrics):**

```bash
PROMETHEUS_PUSHGATEWAY_URL="http://localhost:9091"
```

### 1.6 Notifications

**Slack:**

```bash
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
SLACK_BOT_TOKEN="xoxb-xxxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx"
```

**Telegram:**

```bash
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID="123456789"
```

**Email (SMTP):**

```bash
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USERNAME="notifications@neural-trader.io"
SMTP_PASSWORD="xxxxxxxxxxxxxxxx"
SMTP_FROM="Neural Trader <notifications@neural-trader.io>"
```

### 1.7 Security & Authentication

**JWT Secret:**

```bash
JWT_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
JWT_EXPIRATION_HOURS="24"
```

**Encryption Keys:**

```bash
# Master encryption key (AES-256)
ENCRYPTION_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# API signing secret (HMAC-SHA256)
API_SIGNING_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 1.8 Application Configuration

**General:**

```bash
# Environment (development, staging, production)
ENVIRONMENT="production"

# Logging
LOG_LEVEL="info"  # trace, debug, info, warn, error
LOG_FORMAT="json"  # json, pretty

# Server
SERVER_HOST="0.0.0.0"
SERVER_PORT="8080"

# Features
PAPER_TRADING="false"
SANDBOX_ENABLED="true"
AIDEFENCE_ENABLED="true"
```

---

## 2. Example .env Files

### 2.1 Local Development (.env.local)

```bash
# .env.local - DO NOT COMMIT
# For local development only

# Exchange (Paper Trading)
ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxxxxx"
ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ALPACA_API_URL="https://paper-api.alpaca.markets"

# LLM
OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# E2B
E2B_API_KEY="e2b_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Database (Local)
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/neural_trader_dev"
REDIS_URL="redis://localhost:6379/0"

# Config
ENVIRONMENT="development"
LOG_LEVEL="debug"
PAPER_TRADING="true"
SANDBOX_ENABLED="true"
```

### 2.2 CI/CD (.env.ci)

```bash
# .env.ci
# For CI/CD pipelines (GitHub Actions, etc.)

# Use test API keys (limited scope)
ALPACA_API_KEY="${{ secrets.ALPACA_TEST_API_KEY }}"
ALPACA_SECRET_KEY="${{ secrets.ALPACA_TEST_SECRET_KEY }}"
ALPACA_API_URL="https://paper-api.alpaca.markets"

# LLM (with rate limits)
OPENROUTER_API_KEY="${{ secrets.OPENROUTER_TEST_API_KEY }}"

# Database (Temporary)
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/neural_trader_test"

# Config
ENVIRONMENT="test"
LOG_LEVEL="warn"
PAPER_TRADING="true"
SANDBOX_ENABLED="false"
```

### 2.3 Production (.env.production)

```bash
# .env.production
# Managed by infrastructure (Kubernetes secrets, AWS Secrets Manager, etc.)
# This file should NOT exist in production - use secret managers instead

# ⚠️ NEVER store production secrets in plain text files!
# Instead, use:
# - Kubernetes: kubectl create secret
# - AWS: AWS Secrets Manager / Parameter Store
# - Azure: Azure Key Vault
# - GCP: Secret Manager
# - HashiCorp Vault

# Example showing structure (values from secret manager):
ALPACA_API_KEY="${ALPACA_API_KEY}"
ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY}"
ALPACA_API_URL="https://api.alpaca.markets"

OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
E2B_API_KEY="${E2B_API_KEY}"

DATABASE_URL="${DATABASE_URL}"
REDIS_URL="${REDIS_URL}"

ENCRYPTION_KEY="${ENCRYPTION_KEY}"
JWT_SECRET="${JWT_SECRET}"

ENVIRONMENT="production"
LOG_LEVEL="info"
PAPER_TRADING="false"
```

---

## 3. Secret Storage Options

### 3.1 Environment Variables (Development)

**Pros:**
- Simple, no dependencies
- Works everywhere (local, CI, containers)

**Cons:**
- Visible in process list (`ps aux`)
- Not encrypted at rest
- Easy to leak (logs, error messages)

**Usage:**

```bash
# Load from .env file
export $(cat .env.local | xargs)

# Or use dotenv
cargo install dotenv-cli
dotenv -f .env.local -- neural-trader backtest
```

### 3.2 Encrypted Files (dotenv-vault)

**Installation:**

```bash
npm install -g dotenv-vault
```

**Usage:**

```bash
# Create encrypted .env.vault file
dotenv-vault local build

# Push to remote vault
dotenv-vault push

# Pull in production
dotenv-vault pull production > .env.production

# Load encrypted secrets
DOTENV_KEY="dotenv://:key_xxxx@dotenv.local/vault/.env.vault?environment=production" \
  neural-trader paper
```

### 3.3 AWS Secrets Manager

**Implementation:**

```rust
// src/secrets/aws.rs

use aws_config::BehaviorVersion;
use aws_sdk_secretsmanager::Client;

pub struct AWSSecretsManager {
    client: Client,
}

impl AWSSecretsManager {
    pub async fn new() -> Self {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        Self { client }
    }

    pub async fn get_secret(&self, secret_name: &str) -> Result<String, String> {
        let response = self
            .client
            .get_secret_value()
            .secret_id(secret_name)
            .send()
            .await
            .map_err(|e| format!("Failed to get secret: {}", e))?;

        response
            .secret_string()
            .ok_or_else(|| "Secret is not a string".to_string())
            .map(|s| s.to_string())
    }

    pub async fn set_secret(&self, secret_name: &str, secret_value: &str) -> Result<(), String> {
        self.client
            .put_secret_value()
            .secret_id(secret_name)
            .secret_string(secret_value)
            .send()
            .await
            .map_err(|e| format!("Failed to set secret: {}", e))?;

        Ok(())
    }
}
```

**Usage:**

```rust
// Load secrets from AWS Secrets Manager
let secrets = AWSSecretsManager::new().await;

let alpaca_key = secrets.get_secret("neural-trader/alpaca/api-key").await?;
let alpaca_secret = secrets.get_secret("neural-trader/alpaca/secret-key").await?;
```

**Create secret via CLI:**

```bash
# Store secret
aws secretsmanager create-secret \
  --name neural-trader/alpaca/api-key \
  --secret-string "PKxxxxxxxxxxxxxxxxxxxxx"

# Retrieve secret
aws secretsmanager get-secret-value \
  --secret-id neural-trader/alpaca/api-key \
  --query SecretString \
  --output text
```

### 3.4 Google Cloud Secret Manager

**Implementation:**

```rust
// src/secrets/gcp.rs

use google_secretmanager1::{SecretManager, hyper, hyper_rustls, oauth2};

pub struct GCPSecretsManager {
    hub: SecretManager,
    project_id: String,
}

impl GCPSecretsManager {
    pub async fn new(project_id: String) -> Result<Self, String> {
        let secret = oauth2::ServiceAccountKey::from_file("service-account.json")
            .map_err(|e| e.to_string())?;

        let auth = oauth2::ServiceAccountAuthenticator::builder(secret)
            .build()
            .await
            .map_err(|e| e.to_string())?;

        let client = hyper::Client::builder().build(
            hyper_rustls::HttpsConnectorBuilder::new()
                .with_native_roots()
                .https_or_http()
                .enable_http1()
                .build()
        );

        let hub = SecretManager::new(client, auth);

        Ok(Self { hub, project_id })
    }

    pub async fn get_secret(&self, secret_name: &str) -> Result<String, String> {
        let name = format!(
            "projects/{}/secrets/{}/versions/latest",
            self.project_id, secret_name
        );

        let (_, response) = self
            .hub
            .projects()
            .secrets_versions_access(&name)
            .doit()
            .await
            .map_err(|e| format!("Failed to access secret: {}", e))?;

        let payload = response
            .payload
            .ok_or_else(|| "No payload in response".to_string())?;

        let data = payload
            .data
            .ok_or_else(|| "No data in payload".to_string())?;

        String::from_utf8(data).map_err(|e| format!("Invalid UTF-8: {}", e))
    }
}
```

### 3.5 Azure Key Vault

**Implementation:**

```rust
// src/secrets/azure.rs

use azure_identity::DefaultAzureCredential;
use azure_security_keyvault::KeyvaultClient;

pub struct AzureKeyVault {
    client: KeyvaultClient,
    vault_url: String,
}

impl AzureKeyVault {
    pub async fn new(vault_url: String) -> Self {
        let credential = DefaultAzureCredential::default();
        let client = KeyvaultClient::new(&vault_url, credential).unwrap();

        Self { client, vault_url }
    }

    pub async fn get_secret(&self, secret_name: &str) -> Result<String, String> {
        let secret = self
            .client
            .secret_client()
            .get(secret_name)
            .await
            .map_err(|e| format!("Failed to get secret: {}", e))?;

        Ok(secret.value().to_string())
    }

    pub async fn set_secret(&self, secret_name: &str, secret_value: &str) -> Result<(), String> {
        self.client
            .secret_client()
            .set(secret_name, secret_value)
            .await
            .map_err(|e| format!("Failed to set secret: {}", e))?;

        Ok(())
    }
}
```

### 3.6 HashiCorp Vault

**Implementation:**

```rust
// src/secrets/vault.rs

use vaultrs::client::{VaultClient, VaultClientSettingsBuilder};
use vaultrs::kv2;

pub struct HashiCorpVault {
    client: VaultClient,
    mount: String,
}

impl HashiCorpVault {
    pub fn new(address: &str, token: &str, mount: String) -> Result<Self, String> {
        let settings = VaultClientSettingsBuilder::default()
            .address(address)
            .token(token)
            .build()
            .map_err(|e| e.to_string())?;

        let client = VaultClient::new(settings).map_err(|e| e.to_string())?;

        Ok(Self { client, mount })
    }

    pub async fn get_secret(&self, path: &str) -> Result<String, String> {
        let secret: serde_json::Value = kv2::read(&self.client, &self.mount, path)
            .await
            .map_err(|e| format!("Failed to read secret: {}", e))?;

        secret
            .get("value")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "Secret not found".to_string())
    }

    pub async fn set_secret(&self, path: &str, key: &str, value: &str) -> Result<(), String> {
        let data = serde_json::json!({ key: value });

        kv2::set(&self.client, &self.mount, path, &data)
            .await
            .map_err(|e| format!("Failed to set secret: {}", e))?;

        Ok(())
    }
}
```

---

## 4. Secret Rotation Procedures

### 4.1 Rotation Strategy

**Rotation Schedule:**
- **API Keys**: Every 90 days
- **Database Passwords**: Every 180 days
- **JWT Secrets**: Every 365 days
- **Encryption Keys**: Every 730 days (2 years)

**Compromised Keys**: Immediate rotation

### 4.2 Automated Rotation Script

```bash
#!/bin/bash
# scripts/rotate-secrets.sh

set -e

SECRET_NAME=$1
NEW_VALUE=$2

if [ -z "$SECRET_NAME" ] || [ -z "$NEW_VALUE" ]; then
  echo "Usage: $0 <secret-name> <new-value>"
  exit 1
fi

echo "Rotating secret: $SECRET_NAME"

# 1. Store new secret with version
aws secretsmanager put-secret-value \
  --secret-id "$SECRET_NAME" \
  --secret-string "$NEW_VALUE"

# 2. Update application config
echo "Updated secret in AWS Secrets Manager"

# 3. Trigger rolling restart (Kubernetes example)
kubectl rollout restart deployment/neural-trader

# 4. Wait for rollout to complete
kubectl rollout status deployment/neural-trader

# 5. Verify new secret works
echo "Verifying new secret..."
if neural-trader status --json | jq -e '.status == "healthy"' > /dev/null; then
  echo "✓ Secret rotation successful"
else
  echo "✗ Secret rotation failed - rolling back"
  # Rollback logic here
  exit 1
fi

# 6. Delete old secret version (optional, after grace period)
echo "Old secret version retained for 30 days"
```

### 4.3 Zero-Downtime Rotation

**Strategy: Blue-Green Deployment**

```bash
# 1. Deploy new version with new secrets (green)
kubectl apply -f deployments/neural-trader-green.yaml

# 2. Wait for green to be healthy
kubectl wait --for=condition=ready pod -l version=green

# 3. Switch traffic to green
kubectl patch service neural-trader -p '{"spec":{"selector":{"version":"green"}}}'

# 4. Wait for traffic to settle
sleep 60

# 5. Delete old version (blue)
kubectl delete deployment neural-trader-blue
```

---

## 5. Least Privilege Access Patterns

### 5.1 IAM Policies (AWS)

**Read-Only Policy (for backtests):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:neural-trader/alpaca/api-key-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::neural-trader-data/*"
      ]
    }
  ]
}
```

**Full Access Policy (for live trading):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:PutSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:neural-trader/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:*"
      ],
      "Resource": [
        "arn:aws:s3:::neural-trader-artifacts/*",
        "arn:aws:s3:::neural-trader-logs/*"
      ]
    }
  ]
}
```

### 5.2 Role-Based Access (RBAC)

**User Roles:**

| Role | Secrets Access | Trading Permissions |
|------|----------------|---------------------|
| **Developer** | Read test keys | Paper trading only |
| **Trader** | Read prod keys | Paper + Live (approval) |
| **Admin** | Full access | All operations |
| **Auditor** | Read-only logs | View only |

**Implementation:**

```rust
// src/auth/rbac.rs

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    Developer,
    Trader,
    Admin,
    Auditor,
}

impl Role {
    pub fn can_access_secret(&self, secret_name: &str) -> bool {
        match self {
            Role::Developer => secret_name.contains("test") || secret_name.contains("paper"),
            Role::Trader => !secret_name.contains("admin"),
            Role::Admin => true,
            Role::Auditor => false,
        }
    }

    pub fn can_live_trade(&self) -> bool {
        matches!(self, Role::Trader | Role::Admin)
    }

    pub fn can_modify_config(&self) -> bool {
        matches!(self, Role::Admin)
    }
}
```

---

## 6. Key Derivation

**Use Case**: Generate sub-keys from master key

```rust
// src/crypto/kdf.rs

use argon2::{Argon2, PasswordHasher};
use argon2::password_hash::{SaltString, rand_core::OsRng};

pub struct KeyDerivation {
    master_key: Vec<u8>,
}

impl KeyDerivation {
    pub fn new(master_key: Vec<u8>) -> Self {
        Self { master_key }
    }

    /// Derive sub-key for specific purpose
    pub fn derive_key(&self, purpose: &str) -> Result<Vec<u8>, String> {
        let salt = SaltString::from_b64(purpose)
            .unwrap_or_else(|_| SaltString::generate(&mut OsRng));

        let argon2 = Argon2::default();

        let password_hash = argon2
            .hash_password(&self.master_key, &salt)
            .map_err(|e| e.to_string())?;

        // Extract derived key (32 bytes)
        let derived_key = password_hash.hash.unwrap().as_bytes()[..32].to_vec();

        Ok(derived_key)
    }
}
```

**Usage:**

```rust
let kdf = KeyDerivation::new(master_key);

let encryption_key = kdf.derive_key("encryption")?;
let signing_key = kdf.derive_key("signing")?;
let hmac_key = kdf.derive_key("hmac")?;
```

---

## 7. Encryption at Rest

### 7.1 Config File Encryption

```rust
// src/crypto/config_encryption.rs

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::Rng;

pub struct ConfigEncryption {
    cipher: Aes256Gcm,
}

impl ConfigEncryption {
    pub fn new(key: &[u8; 32]) -> Self {
        let cipher = Aes256Gcm::new(key.into());
        Self { cipher }
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        let nonce_bytes: [u8; 12] = rand::thread_rng().gen();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| e.to_string())?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>, String> {
        if encrypted.len() < 12 {
            return Err("Invalid ciphertext".to_string());
        }

        let (nonce_bytes, ciphertext) = encrypted.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| e.to_string())
    }
}
```

### 7.2 CLI Integration

```bash
# Encrypt config file
neural-trader config encrypt --input config.toml --output config.toml.enc

# Decrypt config file
neural-trader config decrypt --input config.toml.enc --output config.toml

# Use encrypted config
neural-trader --config config.toml.enc backtest --start 2024-01-01
```

---

## 8. Audit Logging for Secret Access

```rust
// src/secrets/audit.rs

use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SecretAccessLog {
    pub timestamp: chrono::DateTime<Utc>,
    pub user: String,
    pub action: SecretAction,
    pub secret_name: String,
    pub success: bool,
    pub ip_address: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SecretAction {
    Read,
    Write,
    Delete,
    Rotate,
}

pub struct SecretAuditLogger {
    log_file: String,
}

impl SecretAuditLogger {
    pub fn new(log_file: String) -> Self {
        Self { log_file }
    }

    pub fn log_access(&self, log: SecretAccessLog) -> Result<(), String> {
        let log_json = serde_json::to_string(&log).map_err(|e| e.to_string())?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)
            .map_err(|e| e.to_string())?;

        use std::io::Write;
        writeln!(file, "{}", log_json).map_err(|e| e.to_string())?;

        Ok(())
    }
}
```

---

## 9. Incident Response for Compromised Keys

### 9.1 Immediate Actions

**Step 1: Identify Scope (0-5 minutes)**

```bash
# Check recent API calls
aws cloudtrail lookup-events --lookup-attributes AttributeKey=Username,AttributeValue=neural-trader

# Check access logs
grep "ALPACA_API_KEY" /var/log/neural-trader/*.log
```

**Step 2: Revoke Compromised Key (5-15 minutes)**

```bash
# Delete from secrets manager
aws secretsmanager delete-secret \
  --secret-id neural-trader/alpaca/api-key \
  --force-delete-without-recovery

# Revoke API key at provider
curl -X DELETE https://api.alpaca.markets/v2/account/keys/compromised-key \
  -H "Authorization: Bearer $ALPACA_ADMIN_TOKEN"
```

**Step 3: Generate New Key (15-30 minutes)**

```bash
# Generate new API key at provider (manual)
# Store new key
aws secretsmanager create-secret \
  --name neural-trader/alpaca/api-key \
  --secret-string "PK_NEW_KEY_HERE"

# Trigger application restart
kubectl rollout restart deployment/neural-trader
```

**Step 4: Investigate Breach (30 minutes - 4 hours)**

```bash
# Review audit logs
neural-trader audit query --start "2025-11-11T00:00:00Z" --filter "secret_access"

# Check for unauthorized trades
neural-trader trades --start "2025-11-11" --format json | jq '.[] | select(.source == "unauthorized")'

# Review system logs
journalctl -u neural-trader --since "2025-11-11 00:00:00"
```

**Step 5: Document Incident (4-24 hours)**

```markdown
# Incident Report: Compromised API Key

**Date**: 2025-11-12
**Severity**: P1 (High)
**Status**: Resolved

## Summary
Alpaca API key was exposed in GitHub commit log.

## Timeline
- 10:30 AM: Key exposed in commit
- 11:15 AM: Detected by GitHub secret scanner
- 11:20 AM: Key revoked
- 11:30 AM: New key deployed
- 11:45 AM: Trading resumed
- 12:00 PM: Post-mortem started

## Impact
- No unauthorized trades executed
- 15 minutes of downtime

## Root Cause
Developer committed .env file to repository

## Remediation
1. Implemented pre-commit hook to scan for secrets
2. Added .env to .gitignore
3. Conducted security training for team
4. Enabled GitHub secret scanning

## Prevention
- [ ] All developers trained on secret management
- [ ] Pre-commit hooks installed on all machines
- [ ] Regular secret rotation (90 days)
- [ ] Monitoring for exposed secrets
```

### 9.2 Post-Incident Checklist

- [ ] **Revoked Compromised Key**: Deleted from all systems
- [ ] **Generated New Key**: Securely stored in secret manager
- [ ] **Restarted Services**: All applications using new key
- [ ] **Investigated Breach**: Root cause identified
- [ ] **Documented Incident**: Report filed
- [ ] **Notified Stakeholders**: Team and management informed
- [ ] **Implemented Prevention**: Controls added to prevent recurrence
- [ ] **Scheduled Post-Mortem**: Team meeting to discuss lessons learned

---

## 10. Security Checklist

### Pre-Production

- [ ] **No Secrets in Code**: Verified with `git log -p | grep -i "api_key"`
- [ ] **Secret Manager Configured**: AWS/GCP/Azure/Vault set up
- [ ] **Rotation Policy**: Automated rotation scheduled
- [ ] **Access Control**: RBAC policies defined and enforced
- [ ] **Encryption at Rest**: Secrets encrypted in storage
- [ ] **Encryption in Transit**: TLS 1.3 for all API calls
- [ ] **Audit Logging**: Secret access logged
- [ ] **Monitoring**: Alerts for unauthorized access
- [ ] **Incident Response**: Runbook prepared
- [ ] **Backup Keys**: Offline backup of master keys in secure location

---

## 11. Troubleshooting

### Common Issues

**Issue: Secret not found**

```bash
# Symptoms:
Error: Secret 'neural-trader/alpaca/api-key' not found

# Solutions:
1. Verify secret exists:
   aws secretsmanager list-secrets | grep neural-trader

2. Check IAM permissions:
   aws secretsmanager get-secret-value --secret-id neural-trader/alpaca/api-key

3. Verify region:
   echo $AWS_REGION  # Should match secret region
```

**Issue: Permission denied**

```bash
# Symptoms:
Error: User is not authorized to perform: secretsmanager:GetSecretValue

# Solutions:
1. Check IAM role:
   aws sts get-caller-identity

2. Attach policy:
   aws iam attach-user-policy --user-name neural-trader --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

---

## 12. References & Resources

### Secret Management
- **AWS Secrets Manager**: https://docs.aws.amazon.com/secretsmanager/
- **GCP Secret Manager**: https://cloud.google.com/secret-manager
- **Azure Key Vault**: https://docs.microsoft.com/en-us/azure/key-vault/
- **HashiCorp Vault**: https://www.vaultproject.io/

### Best Practices
- **OWASP Secrets Management**: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- **NIST Key Management**: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf

---

**Document Status**: ✅ Production-Ready
**Next Review**: 2026-02-12
**Contact**: security@neural-trader.io
