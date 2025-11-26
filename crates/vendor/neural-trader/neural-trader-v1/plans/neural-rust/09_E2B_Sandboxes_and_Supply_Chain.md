# E2B Sandboxes & Supply Chain Security

## Document Overview

**Status**: Production-Ready Planning
**Last Updated**: 2025-11-12
**Owner**: Infrastructure & Security Team
**Related Docs**: `08_Security_Governance_AIDefence_Lean.md`, `10_Federations_and_Agentic_Payments.md`

## Executive Summary

This document covers:
- **E2B Sandboxes**: Isolated execution environments for backtests, exchange sandboxes, fuzzing, and neural training
- **Supply Chain Security**: SBOM generation, dependency scanning, license compliance, vulnerability management
- **Sandbox Lifecycle**: Image building, resource quotas, artifact retention, and cleanup

---

## 1. E2B Sandbox Overview

### 1.1 What is E2B?

**E2B (Environment-as-a-Service)** provides secure, isolated cloud sandboxes for executing untrusted code, running experiments, and training models without risking the host system.

**Key Features:**
- Docker-based isolation
- Resource quotas (CPU, memory, disk, network)
- Custom images with pre-installed dependencies
- Artifact retention and retrieval
- Automatic cleanup and lifecycle management

### 1.2 Use Cases for Neural Trader

| Use Case | Description | Isolation Level | Duration |
|----------|-------------|-----------------|----------|
| **Backtests** | Run historical simulations without affecting live trading | High | Minutes-Hours |
| **Exchange Sandboxes** | Test exchange API integrations safely | Medium | Hours-Days |
| **Fuzzing** | Security fuzzing of trading logic | High | Hours |
| **Neural Training** | Train ML models on GPUs without local setup | Medium | Hours-Days |
| **Strategy Development** | Develop new strategies in isolated environments | Medium | Days-Weeks |
| **Production Isolation** | Run production agents in sandboxes for safety | High | Continuous |

### 1.3 Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Neural Trader CLI                       │
│  (User's Local Machine)                                   │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          │ E2B API (HTTPS)
                          ▼
┌───────────────────────────────────────────────────────────┐
│                    E2B Cloud Platform                      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Sandbox Orchestrator                   │  │
│  └────────┬────────────────────────────────────┬───────┘  │
│           │                                    │           │
│           ▼                                    ▼           │
│  ┌────────────────┐                  ┌────────────────┐   │
│  │   Sandbox 1    │                  │   Sandbox 2    │   │
│  │  (Backtest)    │                  │  (Training)    │   │
│  │                │                  │                │   │
│  │  - Ubuntu      │                  │  - CUDA 12.1   │   │
│  │  - Rust        │                  │  - PyTorch     │   │
│  │  - TA-Lib      │                  │  - Jupyter     │   │
│  │                │                  │                │   │
│  │  CPU: 4 cores  │                  │  GPU: 1x A100  │   │
│  │  RAM: 8GB      │                  │  RAM: 32GB     │   │
│  │  Disk: 20GB    │                  │  Disk: 100GB   │   │
│  └────────────────┘                  └────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Sandbox Image Specifications

### 2.1 Base Image: Backtesting

**Dockerfile:**

```dockerfile
# Dockerfile.backtest

FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install TA-Lib (technical analysis library)
RUN apt-get update && apt-get install -y wget && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && make install && \
    cd .. && rm -rf ta-lib*

# Install neural-trader CLI
COPY neural-trader /usr/local/bin/neural-trader
RUN chmod +x /usr/local/bin/neural-trader

# Create workspace
WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]
```

**Build & Push:**

```bash
# Build image
docker build -f Dockerfile.backtest -t neural-trader/backtest:latest .

# Push to E2B registry (or your own)
docker tag neural-trader/backtest:latest ghcr.io/your-org/neural-trader-backtest:latest
docker push ghcr.io/your-org/neural-trader-backtest:latest
```

### 2.2 GPU Image: Neural Training

**Dockerfile:**

```dockerfile
# Dockerfile.neural

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python and ML dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional ML libraries
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    jupyter==1.0.0 \
    matplotlib==3.7.2 \
    tensorboard==2.13.0

# Install Rust (for hybrid Rust+Python workflows)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install neural-trader Python bindings (if available)
# COPY neural_trader_py-0.1.0-py3-none-any.whl /tmp/
# RUN pip3 install /tmp/neural_trader_py-0.1.0-py3-none-any.whl

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

**Build:**

```bash
docker build -f Dockerfile.neural -t neural-trader/neural:latest .
docker push ghcr.io/your-org/neural-trader-neural:latest
```

### 2.3 Exchange Sandbox Image

**Dockerfile:**

```dockerfile
# Dockerfile.exchange

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install neural-trader with exchange support
COPY neural-trader /usr/local/bin/neural-trader
RUN chmod +x /usr/local/bin/neural-trader

# Install exchange API clients (example)
RUN cargo install --locked ccxt-cli  # If using CCXT

WORKDIR /workspace

CMD ["/bin/bash"]
```

---

## 3. Sandbox Lifecycle Management

### 3.1 Rust SDK Integration

**Cargo.toml:**

```toml
[dependencies]
e2b = "0.2"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Implementation:**

```rust
// src/sandbox/mod.rs

use e2b::{Sandbox, SandboxConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
pub struct SandboxRequest {
    pub image: String,
    pub cpu_count: u32,
    pub memory_mb: u64,
    pub disk_gb: u64,
    pub timeout_seconds: u64,
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SandboxResult {
    pub sandbox_id: String,
    pub status: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub artifacts: Vec<String>,
}

pub struct SandboxManager {
    api_key: String,
}

impl SandboxManager {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    /// Create a new sandbox
    pub async fn create_sandbox(
        &self,
        request: &SandboxRequest,
    ) -> Result<Sandbox, String> {
        let config = SandboxConfig {
            template_id: request.image.clone(),
            timeout: Duration::from_secs(request.timeout_seconds),
            cpu_count: request.cpu_count,
            memory_mb: request.memory_mb,
            env_vars: request.env_vars.clone(),
        };

        Sandbox::new(&self.api_key, config)
            .await
            .map_err(|e| format!("Failed to create sandbox: {}", e))
    }

    /// Execute a command in the sandbox
    pub async fn execute(
        &self,
        sandbox: &Sandbox,
        command: &str,
    ) -> Result<SandboxResult, String> {
        let output = sandbox
            .execute(command)
            .await
            .map_err(|e| format!("Execution failed: {}", e))?;

        Ok(SandboxResult {
            sandbox_id: sandbox.id().to_string(),
            status: if output.exit_code == 0 { "success" } else { "failure" }.to_string(),
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.exit_code,
            artifacts: vec![],
        })
    }

    /// Upload files to sandbox
    pub async fn upload_files(
        &self,
        sandbox: &Sandbox,
        files: HashMap<String, Vec<u8>>,
    ) -> Result<(), String> {
        for (path, content) in files {
            sandbox
                .write_file(&path, content)
                .await
                .map_err(|e| format!("Failed to upload {}: {}", path, e))?;
        }
        Ok(())
    }

    /// Download artifacts from sandbox
    pub async fn download_artifacts(
        &self,
        sandbox: &Sandbox,
        paths: &[String],
    ) -> Result<HashMap<String, Vec<u8>>, String> {
        let mut artifacts = HashMap::new();

        for path in paths {
            let content = sandbox
                .read_file(path)
                .await
                .map_err(|e| format!("Failed to download {}: {}", path, e))?;
            artifacts.insert(path.clone(), content);
        }

        Ok(artifacts)
    }

    /// Terminate sandbox
    pub async fn terminate(&self, sandbox: Sandbox) -> Result<(), String> {
        sandbox
            .close()
            .await
            .map_err(|e| format!("Failed to terminate sandbox: {}", e))
    }
}
```

### 3.2 Backtest Workflow Example

```rust
// src/sandbox/backtest.rs

use super::{SandboxManager, SandboxRequest};
use std::collections::HashMap;

pub async fn run_backtest(
    manager: &SandboxManager,
    strategy_code: String,
    start_date: String,
    end_date: String,
) -> Result<serde_json::Value, String> {
    // 1. Create sandbox
    let request = SandboxRequest {
        image: "ghcr.io/your-org/neural-trader-backtest:latest".to_string(),
        cpu_count: 4,
        memory_mb: 8192,
        disk_gb: 20,
        timeout_seconds: 3600,
        env_vars: HashMap::new(),
    };

    let sandbox = manager.create_sandbox(&request).await?;

    // 2. Upload strategy code
    let mut files = HashMap::new();
    files.insert("/workspace/strategy.rs".to_string(), strategy_code.into_bytes());

    manager.upload_files(&sandbox, files).await?;

    // 3. Run backtest
    let command = format!(
        "neural-trader backtest --strategy /workspace/strategy.rs --start {} --end {} --output /workspace/results.json",
        start_date, end_date
    );

    let result = manager.execute(&sandbox, &command).await?;

    if result.exit_code != 0 {
        manager.terminate(sandbox).await?;
        return Err(format!("Backtest failed: {}", result.stderr));
    }

    // 4. Download results
    let artifacts = manager
        .download_artifacts(&sandbox, &["/workspace/results.json".to_string()])
        .await?;

    // 5. Cleanup
    manager.terminate(sandbox).await?;

    // 6. Parse results
    let results_json = artifacts
        .get("/workspace/results.json")
        .ok_or("Results file not found")?;

    let results: serde_json::Value = serde_json::from_slice(results_json)
        .map_err(|e| format!("Failed to parse results: {}", e))?;

    Ok(results)
}
```

### 3.3 Neural Training Workflow

```rust
// src/sandbox/neural_training.rs

use super::{SandboxManager, SandboxRequest};
use std::collections::HashMap;

pub async fn train_model(
    manager: &SandboxManager,
    training_script: String,
    dataset_path: String,
) -> Result<Vec<u8>, String> {
    // 1. Create GPU sandbox
    let mut env_vars = HashMap::new();
    env_vars.insert("CUDA_VISIBLE_DEVICES".to_string(), "0".to_string());

    let request = SandboxRequest {
        image: "ghcr.io/your-org/neural-trader-neural:latest".to_string(),
        cpu_count: 8,
        memory_mb: 32768,
        disk_gb: 100,
        timeout_seconds: 14400, // 4 hours
        env_vars,
    };

    let sandbox = manager.create_sandbox(&request).await?;

    // 2. Upload training script and data
    let script_content = std::fs::read(&training_script)
        .map_err(|e| format!("Failed to read script: {}", e))?;

    let dataset_content = std::fs::read(&dataset_path)
        .map_err(|e| format!("Failed to read dataset: {}", e))?;

    let mut files = HashMap::new();
    files.insert("/workspace/train.py".to_string(), script_content);
    files.insert("/workspace/dataset.csv".to_string(), dataset_content);

    manager.upload_files(&sandbox, files).await?;

    // 3. Run training
    let command = "python3 /workspace/train.py --data /workspace/dataset.csv --output /workspace/model.pt";
    let result = manager.execute(&sandbox, command).await?;

    if result.exit_code != 0 {
        manager.terminate(sandbox).await?;
        return Err(format!("Training failed: {}", result.stderr));
    }

    // 4. Download trained model
    let artifacts = manager
        .download_artifacts(&sandbox, &["/workspace/model.pt".to_string()])
        .await?;

    // 5. Cleanup
    manager.terminate(sandbox).await?;

    Ok(artifacts
        .get("/workspace/model.pt")
        .ok_or("Model file not found")?
        .clone())
}
```

---

## 4. Resource Quotas & Cost Management

### 4.1 Resource Limits

**Configuration:**

```toml
# config/sandbox_quotas.toml

[backtest]
cpu_cores = 4
memory_gb = 8
disk_gb = 20
timeout_minutes = 60
max_concurrent = 10

[neural_training]
cpu_cores = 8
memory_gb = 32
disk_gb = 100
gpu_count = 1
gpu_type = "A100"
timeout_hours = 4
max_concurrent = 2

[exchange_sandbox]
cpu_cores = 2
memory_gb = 4
disk_gb = 10
timeout_hours = 24
max_concurrent = 5

[fuzzing]
cpu_cores = 16
memory_gb = 16
disk_gb = 50
timeout_hours = 8
max_concurrent = 4
```

### 4.2 Cost Tracking

**Implementation:**

```rust
// src/sandbox/cost_tracker.rs

use rust_decimal::Decimal;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CostConfig {
    pub cpu_cost_per_hour: Decimal,
    pub memory_cost_per_gb_hour: Decimal,
    pub disk_cost_per_gb_hour: Decimal,
    pub gpu_cost_per_hour: HashMap<String, Decimal>,
}

impl Default for CostConfig {
    fn default() -> Self {
        let mut gpu_costs = HashMap::new();
        gpu_costs.insert("A100".to_string(), Decimal::from_str_exact("2.50").unwrap());
        gpu_costs.insert("V100".to_string(), Decimal::from_str_exact("1.25").unwrap());

        Self {
            cpu_cost_per_hour: Decimal::from_str_exact("0.10").unwrap(),
            memory_cost_per_gb_hour: Decimal::from_str_exact("0.01").unwrap(),
            disk_cost_per_gb_hour: Decimal::from_str_exact("0.0001").unwrap(),
            gpu_cost_per_hour: gpu_costs,
        }
    }
}

pub struct CostTracker {
    config: CostConfig,
}

impl CostTracker {
    pub fn new(config: CostConfig) -> Self {
        Self { config }
    }

    pub fn calculate_cost(
        &self,
        cpu_cores: u32,
        memory_gb: u64,
        disk_gb: u64,
        gpu_type: Option<&str>,
        duration_hours: f64,
    ) -> Decimal {
        let cpu_cost = Decimal::from(cpu_cores) * self.config.cpu_cost_per_hour * Decimal::from_f64_retain(duration_hours).unwrap();
        let memory_cost = Decimal::from(memory_gb) * self.config.memory_cost_per_gb_hour * Decimal::from_f64_retain(duration_hours).unwrap();
        let disk_cost = Decimal::from(disk_gb) * self.config.disk_cost_per_gb_hour * Decimal::from_f64_retain(duration_hours).unwrap();

        let gpu_cost = if let Some(gpu) = gpu_type {
            self.config
                .gpu_cost_per_hour
                .get(gpu)
                .copied()
                .unwrap_or(Decimal::ZERO)
                * Decimal::from_f64_retain(duration_hours).unwrap()
        } else {
            Decimal::ZERO
        };

        cpu_cost + memory_cost + disk_cost + gpu_cost
    }

    pub fn estimate_backtest_cost(&self, duration_minutes: u64) -> Decimal {
        let hours = duration_minutes as f64 / 60.0;
        self.calculate_cost(4, 8, 20, None, hours)
    }

    pub fn estimate_training_cost(&self, duration_hours: u64) -> Decimal {
        self.calculate_cost(8, 32, 100, Some("A100"), duration_hours as f64)
    }
}
```

---

## 5. Supply Chain Security

### 5.1 SBOM Generation

**Software Bill of Materials (SBOM)** provides transparency into all dependencies.

**Installation:**

```bash
cargo install cargo-sbom
```

**Generate SBOM:**

```bash
# CycloneDX format (JSON)
cargo sbom --output-format cyclonedx-json > sbom.json

# SPDX format
cargo sbom --output-format spdx-json > sbom-spdx.json
```

**Example SBOM (CycloneDX):**

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:12345678-1234-5678-1234-567812345678",
  "version": 1,
  "metadata": {
    "timestamp": "2025-11-12T10:00:00Z",
    "component": {
      "type": "application",
      "name": "neural-trader",
      "version": "0.1.0"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "tokio",
      "version": "1.35.1",
      "purl": "pkg:cargo/tokio@1.35.1",
      "licenses": [
        {
          "license": {
            "id": "MIT"
          }
        }
      ]
    },
    {
      "type": "library",
      "name": "serde",
      "version": "1.0.193",
      "purl": "pkg:cargo/serde@1.0.193"
    }
  ]
}
```

### 5.2 Dependency Scanning

**cargo-audit**: Scan for known vulnerabilities

```bash
# Install
cargo install cargo-audit

# Run audit
cargo audit

# Example output:
# Fetching advisory database from `https://github.com/RustSec/advisory-db.git`
#       Loaded 573 security advisories (from github.com/RustSec/advisory-db)
#     Updating crates.io index
#     Scanning Cargo.lock for vulnerabilities (342 crate dependencies)
# Crate:     tokio
# Version:   1.28.0
# Warning:   memory corruption vulnerability
# ID:        RUSTSEC-2023-0072
# Solution:  Upgrade to >=1.28.1
```

**Fix vulnerabilities:**

```bash
# Update dependencies
cargo update

# Or update specific crate
cargo update -p tokio
```

### 5.3 License Compliance

**cargo-deny**: Enforce license policies

**Installation:**

```bash
cargo install cargo-deny
```

**Configuration (deny.toml):**

```toml
# deny.toml

[licenses]
# Allow only these licenses
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "ISC",
    "Unlicense",
]

# Deny these licenses
deny = [
    "GPL-3.0",
    "AGPL-3.0",
]

# Warn for these licenses (require manual review)
warn = [
    "LGPL-3.0",
]

[bans]
# Ban specific crates
deny = [
    { name = "openssl", wrappers = ["openssl-sys"] },
]

# Allow deprecated crates (with warning)
warn = [
    { name = "chrono", reason = "deprecated, migrate to time" },
]

[advisories]
# Deny crates with security advisories
vulnerability = "deny"
unmaintained = "warn"
unsound = "deny"

[sources]
# Only allow crates.io and git sources
unknown-registry = "deny"
unknown-git = "deny"
```

**Run checks:**

```bash
# Check licenses
cargo deny check licenses

# Check for banned crates
cargo deny check bans

# Check for advisories
cargo deny check advisories

# Check all
cargo deny check
```

### 5.4 Vulnerability Scanning

**Automated CI/CD Integration:**

**GitHub Actions (.github/workflows/security.yml):**

```yaml
name: Security Audit

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Run cargo-audit
        run: cargo audit

      - name: Install cargo-deny
        run: cargo install cargo-deny

      - name: Run cargo-deny
        run: cargo deny check

      - name: Generate SBOM
        run: |
          cargo install cargo-sbom
          cargo sbom --output-format cyclonedx-json > sbom.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json
```

---

## 6. Reproducible Builds

### 6.1 Lock Files

**Always commit Cargo.lock:**

```bash
git add Cargo.lock
git commit -m "Add Cargo.lock for reproducible builds"
```

### 6.2 Rust Toolchain Pinning

**rust-toolchain.toml:**

```toml
[toolchain]
channel = "1.75.0"
components = ["rustfmt", "clippy", "rust-src"]
targets = ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin", "aarch64-apple-darwin"]
profile = "minimal"
```

### 6.3 Docker Build Reproducibility

**Multi-stage build with pinned base images:**

```dockerfile
# Use pinned digest instead of tags
FROM rust:1.75.0-slim@sha256:abc123... as builder

WORKDIR /build

# Copy dependencies first (for layer caching)
COPY Cargo.toml Cargo.lock ./
RUN cargo fetch

# Copy source code
COPY src ./src

# Build with release profile
RUN cargo build --release --locked

# Production image
FROM ubuntu:22.04@sha256:def456...

COPY --from=builder /build/target/release/neural-trader /usr/local/bin/

CMD ["neural-trader"]
```

---

## 7. Code Signing & Verification

### 7.1 Binary Signing

**Using cosign:**

```bash
# Install cosign
go install github.com/sigstore/cosign/v2/cmd/cosign@latest

# Generate key pair
cosign generate-key-pair

# Sign binary
cosign sign-blob --key cosign.key neural-trader > neural-trader.sig

# Verify signature
cosign verify-blob --key cosign.pub --signature neural-trader.sig neural-trader
```

### 7.2 Container Image Signing

```bash
# Sign Docker image
cosign sign --key cosign.key ghcr.io/your-org/neural-trader:v1.0.0

# Verify image
cosign verify --key cosign.pub ghcr.io/your-org/neural-trader:v1.0.0
```

---

## 8. Artifact Retention

### 8.1 Retention Policy

**Configuration:**

```toml
# config/artifact_retention.toml

[backtest_results]
retention_days = 90
storage_location = "s3://neural-trader-artifacts/backtests/"
compression = "gzip"

[trained_models]
retention_days = 365
storage_location = "s3://neural-trader-artifacts/models/"
compression = "none"

[logs]
retention_days = 30
storage_location = "s3://neural-trader-logs/"
compression = "gzip"

[sandbox_snapshots]
retention_days = 7
storage_location = "s3://neural-trader-sandboxes/"
compression = "zstd"
```

### 8.2 Cleanup Automation

```rust
// src/sandbox/cleanup.rs

use chrono::{Duration, Utc};
use std::path::Path;

pub struct ArtifactCleaner {
    retention_days: i64,
    storage_path: String,
}

impl ArtifactCleaner {
    pub fn new(retention_days: i64, storage_path: String) -> Self {
        Self {
            retention_days,
            storage_path,
        }
    }

    pub async fn cleanup_old_artifacts(&self) -> Result<usize, String> {
        let cutoff_date = Utc::now() - Duration::days(self.retention_days);
        let mut deleted_count = 0;

        // List all artifacts in storage
        let entries = std::fs::read_dir(&self.storage_path)
            .map_err(|e| format!("Failed to read storage: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let metadata = entry.metadata().map_err(|e| e.to_string())?;

            if let Ok(modified) = metadata.modified() {
                let modified_datetime: chrono::DateTime<Utc> = modified.into();

                if modified_datetime < cutoff_date {
                    std::fs::remove_file(entry.path())
                        .map_err(|e| format!("Failed to delete file: {}", e))?;
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }
}
```

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: Sandbox creation timeout**

```bash
# Symptoms:
Error: Sandbox creation timed out after 60 seconds

# Solutions:
1. Check E2B API status: https://status.e2b.dev
2. Verify API key is valid
3. Check network connectivity
4. Try reducing resource requirements
```

**Issue: Out of memory in sandbox**

```bash
# Symptoms:
Error: OOMKilled (exit code 137)

# Solutions:
1. Increase memory limit in SandboxRequest
2. Optimize code to use less memory
3. Use streaming instead of loading all data
4. Enable swap (if allowed)
```

**Issue: Dependency build failure**

```bash
# Symptoms:
error: failed to compile `openssl-sys`

# Solutions:
1. Add build dependencies to Dockerfile:
   RUN apt-get install -y libssl-dev pkg-config

2. Or use pure-Rust alternatives:
   [dependencies]
   rustls = "0.21"  # Instead of openssl
```

### 9.2 Debugging Sandboxes

**Enable verbose logging:**

```rust
let sandbox = manager
    .create_sandbox(&request)
    .await?;

// Execute with verbose output
let result = sandbox
    .execute("bash -x /workspace/script.sh")
    .await?;

println!("STDOUT: {}", result.stdout);
println!("STDERR: {}", result.stderr);
```

**Interactive shell access:**

```rust
// Start interactive session (for debugging only)
let result = sandbox
    .execute("bash")
    .await?;

// Or use SSH (if configured)
// ssh -i sandbox_key user@sandbox_ip
```

---

## 10. Security Checklist

### Pre-Production Security Review

- [ ] **Sandbox Isolation**: All untrusted code runs in sandboxes
- [ ] **Resource Limits**: CPU, memory, disk quotas enforced
- [ ] **Network Restrictions**: Outbound traffic limited/monitored
- [ ] **Image Scanning**: All Docker images scanned for vulnerabilities
- [ ] **SBOM Generated**: Software bill of materials available
- [ ] **Dependencies Audited**: cargo-audit passes with no criticals
- [ ] **Licenses Compliant**: cargo-deny passes license checks
- [ ] **Secrets Management**: No secrets in images or code
- [ ] **Artifact Retention**: Cleanup policies configured
- [ ] **Cost Monitoring**: Budget alerts and quotas set
- [ ] **Logging Enabled**: All sandbox activity logged
- [ ] **Incident Response**: Procedures for compromised sandboxes

---

## 11. Cost Optimization Strategies

### 11.1 Right-Sizing Sandboxes

```rust
// Analyze actual usage and adjust
let usage_stats = sandbox.get_resource_usage().await?;

if usage_stats.avg_cpu_percent < 30.0 {
    println!("Consider reducing CPU allocation");
}

if usage_stats.max_memory_mb < (request.memory_mb as f64 * 0.5) {
    println!("Consider reducing memory allocation");
}
```

### 11.2 Spot Instances (if supported)

```rust
let request = SandboxRequest {
    image: "neural-trader/backtest:latest".to_string(),
    cpu_count: 4,
    memory_mb: 8192,
    disk_gb: 20,
    timeout_seconds: 3600,
    use_spot_instances: true,  // 60-80% cost savings
    env_vars: HashMap::new(),
};
```

### 11.3 Batch Processing

```rust
// Run multiple backtests in parallel in same sandbox
let sandbox = manager.create_sandbox(&request).await?;

let strategies = vec!["strategy1.rs", "strategy2.rs", "strategy3.rs"];

for strategy in strategies {
    let result = manager.execute(&sandbox, &format!("neural-trader backtest --strategy {}", strategy)).await?;
    // Process result
}

// Cleanup once
manager.terminate(sandbox).await?;
```

---

## 12. References & Resources

### E2B Platform
- **Docs**: https://e2b.dev/docs
- **API Reference**: https://e2b.dev/docs/api
- **Pricing**: https://e2b.dev/pricing

### Supply Chain Security
- **SBOM Guide**: https://www.cisa.gov/sbom
- **SLSA Framework**: https://slsa.dev
- **Sigstore**: https://www.sigstore.dev

### Rust Security Tools
- **cargo-audit**: https://crates.io/crates/cargo-audit
- **cargo-deny**: https://embarkstudios.github.io/cargo-deny/
- **cargo-geiger**: https://crates.io/crates/cargo-geiger (unsafe code detection)

---

**Document Status**: ✅ Production-Ready
**Next Review**: 2026-02-12
**Contact**: infrastructure@neural-trader.io
