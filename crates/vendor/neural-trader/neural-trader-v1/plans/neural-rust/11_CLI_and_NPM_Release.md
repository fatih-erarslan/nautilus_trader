# CLI & NPM Release Strategy

## Document Overview

**Status**: Production-Ready Planning
**Last Updated**: 2025-11-12
**Owner**: Developer Experience Team
**Related Docs**: `12_Secrets_and_Environments.md`, `09_E2B_Sandboxes_and_Supply_Chain.md`

## Executive Summary

This document covers:
- **CLI Command Specification**: Complete command-line interface for neural-trader
- **Configuration Management**: Config files, environment variables, and profiles
- **NPM Packaging**: Distribution strategy for multi-platform binaries
- **Release Automation**: Versioning, changelogs, and deployment
- **Platform Support**: Linux, macOS, Windows binaries

---

## 1. CLI Command Specification

### 1.1 Command Structure

```bash
neural-trader [GLOBAL_OPTIONS] <COMMAND> [COMMAND_OPTIONS] [ARGS]
```

### 1.2 Global Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Config file path | `~/.neural-trader/config.toml` |
| `--profile` | `-p` | Profile name | `default` |
| `--verbose` | `-v` | Verbose output | `false` |
| `--quiet` | `-q` | Suppress output | `false` |
| `--json` | | JSON output mode | `false` |
| `--pretty` | | Pretty-print JSON | `false` |
| `--help` | `-h` | Show help | |
| `--version` | `-V` | Show version | |

### 1.3 Commands

#### `init` - Initialize a new trading project

```bash
neural-trader init [OPTIONS] <PROJECT_NAME>

# Initialize with template
neural-trader init --template momentum my-strategy

# Initialize with wizard
neural-trader init --interactive my-strategy
```

**Options:**
- `--template <NAME>`: Use template (momentum, mean-reversion, arbitrage, ml)
- `--interactive`: Run interactive wizard
- `--exchange <NAME>`: Configure exchange (alpaca, coinbase, binance)
- `--paper`: Configure for paper trading (default: true)

**Output:**
```
✓ Created project directory: my-strategy/
✓ Generated config file: my-strategy/config.toml
✓ Created strategy template: my-strategy/src/strategy.rs
✓ Created tests: my-strategy/tests/
✓ Initialized Git repository

Next steps:
1. cd my-strategy
2. Edit config.toml with your API keys
3. Run: neural-trader backtest --start 2024-01-01 --end 2024-12-31
```

#### `backtest` - Run historical backtest

```bash
neural-trader backtest [OPTIONS]

# Basic backtest
neural-trader backtest --start 2024-01-01 --end 2024-12-31

# Backtest with custom strategy
neural-trader backtest --strategy ./custom_strategy.rs --start 2024-01-01 --end 2024-12-31

# Backtest with multiple symbols
neural-trader backtest --symbols BTC,ETH,SOL --start 2024-01-01

# Save results to file
neural-trader backtest --output results.json --pretty

# Run in sandbox
neural-trader backtest --sandbox --cpu 4 --memory 8
```

**Options:**
- `--strategy <PATH>`: Strategy file path (default: `./src/strategy.rs`)
- `--start <DATE>`: Start date (YYYY-MM-DD)
- `--end <DATE>`: End date (YYYY-MM-DD, default: today)
- `--symbols <LIST>`: Comma-separated symbols (default: from config)
- `--initial-capital <AMOUNT>`: Initial capital (default: 100000)
- `--output <PATH>`: Output file path
- `--sandbox`: Run in E2B sandbox
- `--cpu <CORES>`: CPU cores for sandbox
- `--memory <GB>`: Memory for sandbox

**Output (JSON):**
```json
{
  "status": "completed",
  "duration_seconds": 45.2,
  "metrics": {
    "total_return": 0.2341,
    "sharpe_ratio": 1.82,
    "max_drawdown": -0.0823,
    "win_rate": 0.64,
    "total_trades": 287,
    "profitable_trades": 184,
    "losing_trades": 103,
    "avg_profit": 125.43,
    "avg_loss": -78.21
  },
  "equity_curve": [...],
  "trades": [...]
}
```

#### `paper` - Run paper trading (simulated)

```bash
neural-trader paper [OPTIONS]

# Start paper trading
neural-trader paper --exchange alpaca

# Paper trading with custom strategy
neural-trader paper --strategy ./strategy.rs --symbols BTC,ETH

# Run as daemon
neural-trader paper --daemon --log-file paper.log
```

**Options:**
- `--strategy <PATH>`: Strategy file path
- `--exchange <NAME>`: Exchange (alpaca, coinbase, binance)
- `--symbols <LIST>`: Symbols to trade
- `--daemon`: Run as background process
- `--log-file <PATH>`: Log file path
- `--check-interval <SECONDS>`: Check interval (default: 60)

**Output:**
```
✓ Connected to Alpaca Paper Trading API
✓ Loaded strategy: MomentumStrategy
✓ Watching symbols: BTC, ETH, SOL
✓ Initial capital: $100,000

[2025-11-12 10:30:00] Checking for signals...
[2025-11-12 10:30:05] Signal: BUY BTC @ $43,250
[2025-11-12 10:30:06] Order placed: BTC-USD, qty=0.5, order_id=abc123
[2025-11-12 10:30:10] Order filled: BTC-USD @ $43,255

Press Ctrl+C to stop...
```

#### `live` - Run live trading (real money)

```bash
neural-trader live [OPTIONS]

# Start live trading (requires confirmation)
neural-trader live --exchange alpaca

# Live trading with safety limits
neural-trader live --max-position-size 5000 --max-drawdown 0.10

# Dry-run mode (shows what would be traded)
neural-trader live --dry-run
```

**Options:**
- `--strategy <PATH>`: Strategy file path
- `--exchange <NAME>`: Exchange
- `--symbols <LIST>`: Symbols to trade
- `--max-position-size <AMOUNT>`: Max position size
- `--max-drawdown <PCT>`: Max drawdown before halt (0.0-1.0)
- `--daemon`: Run as background process
- `--dry-run`: Dry-run mode (no actual trades)

**Safety Confirmation:**
```
⚠️  WARNING: You are about to start LIVE TRADING with REAL MONEY.

Exchange: Alpaca (Live)
Strategy: MomentumStrategy
Symbols: BTC, ETH, SOL
Max Position Size: $5,000 per symbol
Max Drawdown: 10%

Type 'YES' to confirm: _
```

#### `status` - Show status of running agents

```bash
neural-trader status [OPTIONS]

# Show all agents
neural-trader status

# Show detailed status
neural-trader status --verbose

# Watch status (auto-refresh)
neural-trader status --watch

# JSON output
neural-trader status --json --pretty
```

**Output:**
```
Neural Trader Status
====================

Active Agents: 2

┌──────────┬───────────┬──────────┬────────────┬──────────────┐
│ Agent ID │ Strategy  │ Exchange │ Status     │ PnL (24h)    │
├──────────┼───────────┼──────────┼────────────┼──────────────┤
│ agent-01 │ Momentum  │ Alpaca   │ Running    │ +$1,234.56   │
│ agent-02 │ Arbitrage │ Binance  │ Idle       │ +$567.89     │
└──────────┴───────────┴──────────┴────────────┴──────────────┘

Total PnL (24h): +$1,802.45
Total Trades (24h): 47
Win Rate: 63.8%

Recent Activity:
[10:45:23] agent-01: Bought 0.5 BTC @ $43,250
[10:43:12] agent-02: Detected arbitrage opportunity (1.2%)
[10:40:05] agent-01: Sold 10 ETH @ $2,315
```

#### `secrets` - Manage API keys and secrets

```bash
neural-trader secrets <SUBCOMMAND> [OPTIONS]

# Set secret
neural-trader secrets set ALPACA_API_KEY

# Get secret (masked)
neural-trader secrets get ALPACA_API_KEY

# List all secrets
neural-trader secrets list

# Delete secret
neural-trader secrets delete ALPACA_API_KEY

# Import from .env file
neural-trader secrets import .env

# Export to .env file (encrypted)
neural-trader secrets export .env.encrypted
```

**Set Secret:**
```bash
$ neural-trader secrets set ALPACA_API_KEY
Enter value for ALPACA_API_KEY: ********
✓ Secret ALPACA_API_KEY stored securely
```

**List Secrets:**
```
Stored Secrets:
- ALPACA_API_KEY: ***********2345 (set 2 days ago)
- ALPACA_SECRET_KEY: ***********6789 (set 2 days ago)
- OPENROUTER_API_KEY: ***********abcd (set 5 days ago)

Total: 3 secrets
```

#### `config` - Manage configuration

```bash
neural-trader config <SUBCOMMAND> [OPTIONS]

# Show current config
neural-trader config show

# Get specific value
neural-trader config get strategy.symbols

# Set value
neural-trader config set strategy.symbols "BTC,ETH,SOL"

# Edit config in $EDITOR
neural-trader config edit

# Validate config
neural-trader config validate

# Create new profile
neural-trader config profile create production
```

#### `report` - Generate performance reports

```bash
neural-trader report [OPTIONS]

# Generate HTML report
neural-trader report --format html --output report.html

# Generate PDF report
neural-trader report --format pdf --output report.pdf

# Email report
neural-trader report --email user@example.com

# Custom date range
neural-trader report --start 2024-01-01 --end 2024-12-31
```

#### `swarm` - Manage multi-agent swarms

```bash
neural-trader swarm <SUBCOMMAND> [OPTIONS]

# Initialize swarm
neural-trader swarm init --topology mesh --agents 5

# Add agent to swarm
neural-trader swarm add --strategy momentum

# Remove agent
neural-trader swarm remove agent-03

# Show swarm status
neural-trader swarm status

# Coordinate federated strategies
neural-trader swarm coordinate --portfolio-allocation risk-parity
```

---

## 2. Configuration File Format

### 2.1 Structure (TOML)

**~/.neural-trader/config.toml:**

```toml
[general]
profile = "default"
log_level = "info"
log_file = "~/.neural-trader/logs/neural-trader.log"

[strategy]
name = "MomentumStrategy"
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
initial_capital = 100_000.0
check_interval_seconds = 60

[exchange]
name = "alpaca"
paper_trading = true
api_url = "https://paper-api.alpaca.markets"

[risk]
max_position_size = 5000.0
max_portfolio_exposure = 0.20
max_leverage = 3.0
max_drawdown = 0.15
min_account_balance = 1000.0

[execution]
order_type = "market"  # market, limit, stop, stop_limit
time_in_force = "gtc"  # gtc, day, ioc, fok
slippage_tolerance = 0.001

[backtest]
start_date = "2024-01-01"
end_date = "2024-12-31"
commission = 0.001  # 0.1%
slippage = 0.0005   # 0.05%

[sandbox]
enabled = false
image = "ghcr.io/your-org/neural-trader-backtest:latest"
cpu_cores = 4
memory_gb = 8
timeout_minutes = 60

[agentic_flow]
enabled = false
federation_topology = "mesh"  # mesh, hierarchical, hybrid
max_agents = 10
coordinator_endpoint = "http://localhost:8080"

[payments]
enabled = false
monthly_budget = 1000.0
cost_alerts = [500.0, 750.0, 900.0]

[notifications]
enabled = true
email = "user@example.com"
slack_webhook = ""
telegram_bot_token = ""

[aidefence]
enabled = true
min_confidence = 0.85
max_hallucination_score = 0.15
policy_file = "~/.neural-trader/policy.json"
```

### 2.2 Profile Management

**Multiple profiles in config.toml:**

```toml
[profiles.default]
exchange = "alpaca"
paper_trading = true
initial_capital = 100_000.0

[profiles.production]
exchange = "alpaca"
paper_trading = false
initial_capital = 50_000.0
max_position_size = 2000.0

[profiles.crypto]
exchange = "coinbase"
symbols = ["BTC-USD", "ETH-USD"]
initial_capital = 20_000.0
```

**Usage:**

```bash
# Use production profile
neural-trader --profile production paper

# Use crypto profile
neural-trader --profile crypto backtest --start 2024-01-01
```

---

## 3. NPM Packaging Strategy

### 3.1 Package Structure

```
neural-trader/
├── package.json
├── README.md
├── LICENSE
├── bin/
│   └── neural-trader.js          # Node.js wrapper script
├── native/
│   ├── neural-trader-linux-x64
│   ├── neural-trader-darwin-x64
│   ├── neural-trader-darwin-arm64
│   └── neural-trader-win32-x64.exe
├── lib/
│   └── index.js                   # Node.js API (optional)
└── postinstall.js                 # Post-install script
```

### 3.2 package.json

```json
{
  "name": "neural-trader",
  "version": "0.1.0",
  "description": "AI-powered algorithmic trading platform with Rust + LLM integration",
  "bin": {
    "neural-trader": "./bin/neural-trader.js"
  },
  "scripts": {
    "postinstall": "node postinstall.js"
  },
  "keywords": [
    "trading",
    "algorithmic-trading",
    "crypto",
    "ai",
    "machine-learning",
    "rust"
  ],
  "author": "Your Name",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/your-org/neural-trader.git"
  },
  "engines": {
    "node": ">=16.0.0"
  },
  "os": [
    "linux",
    "darwin",
    "win32"
  ],
  "cpu": [
    "x64",
    "arm64"
  ],
  "optionalDependencies": {
    "neural-trader-linux-x64": "0.1.0",
    "neural-trader-darwin-x64": "0.1.0",
    "neural-trader-darwin-arm64": "0.1.0",
    "neural-trader-win32-x64": "0.1.0"
  }
}
```

### 3.3 Platform-Specific Packages

**@neural-trader/linux-x64/package.json:**

```json
{
  "name": "neural-trader-linux-x64",
  "version": "0.1.0",
  "description": "Neural Trader binary for Linux x64",
  "os": ["linux"],
  "cpu": ["x64"],
  "files": [
    "neural-trader"
  ]
}
```

**Similar packages:**
- `neural-trader-darwin-x64` (macOS Intel)
- `neural-trader-darwin-arm64` (macOS Apple Silicon)
- `neural-trader-win32-x64` (Windows)

### 3.4 Wrapper Script (bin/neural-trader.js)

```javascript
#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Detect platform
const platform = process.platform;
const arch = process.arch;

// Map to binary name
let binaryName = `neural-trader-${platform}-${arch}`;
if (platform === 'win32') {
  binaryName += '.exe';
}

// Try to find binary in native/ directory (development)
let binaryPath = path.join(__dirname, '..', 'native', binaryName);

// If not found, try platform-specific package (production)
if (!fs.existsSync(binaryPath)) {
  try {
    const platformPackage = require(`neural-trader-${platform}-${arch}`);
    binaryPath = platformPackage.binaryPath;
  } catch (err) {
    console.error(`Error: No binary found for ${platform}-${arch}`);
    console.error('Please ensure neural-trader is installed correctly.');
    process.exit(1);
  }
}

// Spawn binary with all arguments
const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: 'inherit',
  windowsHide: true,
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code);
  }
});
```

### 3.5 Post-Install Script (postinstall.js)

```javascript
#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const platform = process.platform;
const arch = process.arch;

console.log(`Installing neural-trader for ${platform}-${arch}...`);

// Make binary executable on Unix-like systems
if (platform !== 'win32') {
  const binaryPath = path.join(__dirname, 'native', `neural-trader-${platform}-${arch}`);

  if (fs.existsSync(binaryPath)) {
    try {
      fs.chmodSync(binaryPath, '755');
      console.log('✓ Binary made executable');
    } catch (err) {
      console.warn('Warning: Could not make binary executable:', err.message);
    }
  }
}

// Create config directory
const configDir = path.join(require('os').homedir(), '.neural-trader');
if (!fs.existsSync(configDir)) {
  fs.mkdirSync(configDir, { recursive: true });
  console.log('✓ Created config directory:', configDir);
}

// Create default config if it doesn't exist
const configFile = path.join(configDir, 'config.toml');
if (!fs.existsSync(configFile)) {
  const defaultConfig = `# Neural Trader Configuration
# Generated on ${new Date().toISOString()}

[general]
profile = "default"
log_level = "info"

[strategy]
symbols = ["BTC-USD", "ETH-USD"]
initial_capital = 100_000.0

[exchange]
name = "alpaca"
paper_trading = true
`;

  fs.writeFileSync(configFile, defaultConfig);
  console.log('✓ Created default config:', configFile);
}

console.log('\n✓ Neural Trader installed successfully!');
console.log('\nNext steps:');
console.log('1. Run: npx neural-trader init my-strategy');
console.log('2. Edit: ~/.neural-trader/config.toml');
console.log('3. Run: npx neural-trader backtest --start 2024-01-01\n');
```

---

## 4. Release Automation

### 4.1 GitHub Actions Workflow

**.github/workflows/release.yml:**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build:
    name: Build ${{ matrix.target }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            binary: neural-trader
          - os: macos-latest
            target: x86_64-apple-darwin
            binary: neural-trader
          - os: macos-latest
            target: aarch64-apple-darwin
            binary: neural-trader
          - os: windows-latest
            target: x86_64-pc-windows-msvc
            binary: neural-trader.exe

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}

      - name: Build binary
        run: cargo build --release --target ${{ matrix.target }}

      - name: Strip binary (Linux/macOS)
        if: matrix.os != 'windows-latest'
        run: strip target/${{ matrix.target }}/release/${{ matrix.binary }}

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.target }}
          path: target/${{ matrix.target }}/release/${{ matrix.binary }}

  package:
    name: Create NPM packages
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v3
        with:
          path: artifacts

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'

      - name: Create package structure
        run: |
          mkdir -p native
          cp artifacts/x86_64-unknown-linux-gnu/neural-trader native/neural-trader-linux-x64
          cp artifacts/x86_64-apple-darwin/neural-trader native/neural-trader-darwin-x64
          cp artifacts/aarch64-apple-darwin/neural-trader native/neural-trader-darwin-arm64
          cp artifacts/x86_64-pc-windows-msvc/neural-trader.exe native/neural-trader-win32-x64.exe

          chmod +x native/neural-trader-*

      - name: Update version in package.json
        run: |
          VERSION=${GITHUB_REF#refs/tags/v}
          sed -i "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" package.json

      - name: Publish to NPM
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

  github-release:
    name: Create GitHub Release
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v3
        with:
          path: artifacts

      - name: Create release archives
        run: |
          cd artifacts/x86_64-unknown-linux-gnu && tar czf ../../neural-trader-linux-x64.tar.gz neural-trader
          cd ../x86_64-apple-darwin && tar czf ../../neural-trader-darwin-x64.tar.gz neural-trader
          cd ../aarch64-apple-darwin && tar czf ../../neural-trader-darwin-arm64.tar.gz neural-trader
          cd ../x86_64-pc-windows-msvc && zip ../../neural-trader-win32-x64.zip neural-trader.exe

      - name: Generate changelog
        id: changelog
        run: |
          echo "## Changes" > RELEASE_NOTES.md
          git log $(git describe --tags --abbrev=0 HEAD^)..HEAD --oneline >> RELEASE_NOTES.md

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            neural-trader-linux-x64.tar.gz
            neural-trader-darwin-x64.tar.gz
            neural-trader-darwin-arm64.tar.gz
            neural-trader-win32-x64.zip
          body_path: RELEASE_NOTES.md
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 4.2 Versioning Strategy

**Semantic Versioning (semver):**

```
MAJOR.MINOR.PATCH

Examples:
- 0.1.0 - Initial release
- 0.2.0 - New features (backtest improvements)
- 0.2.1 - Bug fixes
- 1.0.0 - Production-ready
- 1.1.0 - New command (swarm)
- 2.0.0 - Breaking changes (API redesign)
```

**Version Bump:**

```bash
# Install cargo-release
cargo install cargo-release

# Bump version and create tag
cargo release patch  # 0.1.0 -> 0.1.1
cargo release minor  # 0.1.1 -> 0.2.0
cargo release major  # 0.2.0 -> 1.0.0
```

### 4.3 Changelog Automation

**CHANGELOG.md template:**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New swarm coordination features

### Changed
- Improved backtest performance by 40%

### Fixed
- Fixed memory leak in paper trading mode

## [0.2.0] - 2025-11-12

### Added
- E2B sandbox integration for backtests
- AIDefence guardrails
- Multi-strategy federation support
- Payment tracking and budgets

### Changed
- Refactored risk management system
- Updated CLI with new commands

### Fixed
- Fixed timezone handling in backtests
- Fixed order execution edge cases

## [0.1.0] - 2025-10-01

### Added
- Initial release
- Basic backtesting engine
- Alpaca integration
- Paper trading mode
- CLI interface
```

---

## 5. Smoke Tests

### 5.1 Platform-Specific Tests

**tests/smoke.sh:**

```bash
#!/bin/bash

set -e

echo "Running smoke tests for neural-trader..."

# Test 1: Binary exists and is executable
echo "Test 1: Binary executable check"
neural-trader --version
echo "✓ Binary is executable"

# Test 2: Config initialization
echo "Test 2: Config initialization"
neural-trader config show > /dev/null
echo "✓ Config accessible"

# Test 3: Help command
echo "Test 3: Help command"
neural-trader --help > /dev/null
echo "✓ Help command works"

# Test 4: Init project
echo "Test 4: Init project"
rm -rf /tmp/test-strategy
neural-trader init --template momentum /tmp/test-strategy
[ -d /tmp/test-strategy ] || exit 1
echo "✓ Project initialization works"

# Test 5: Validate config
echo "Test 5: Validate config"
cd /tmp/test-strategy
neural-trader config validate
echo "✓ Config validation works"

# Test 6: Backtest (quick test with small date range)
echo "Test 6: Quick backtest"
neural-trader backtest --start 2024-01-01 --end 2024-01-07 --json > /dev/null
echo "✓ Backtest works"

echo ""
echo "✅ All smoke tests passed!"
```

### 5.2 CI Smoke Test Integration

**Add to GitHub Actions:**

```yaml
  smoke-test:
    name: Smoke test ${{ matrix.target }}
    needs: build
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc

    steps:
      - uses: actions/checkout@v4

      - name: Download binary
        uses: actions/download-artifact@v3
        with:
          name: ${{ matrix.target }}
          path: bin

      - name: Make executable (Unix)
        if: matrix.os != 'windows-latest'
        run: chmod +x bin/neural-trader

      - name: Run smoke tests
        run: |
          export PATH=$PWD/bin:$PATH
          bash tests/smoke.sh
```

---

## 6. Installation Methods

### 6.1 NPM

```bash
# Global installation
npm install -g neural-trader

# Local installation (project-specific)
npm install --save-dev neural-trader

# Run without installing (npx)
npx neural-trader --help
```

### 6.2 Cargo

```bash
# Install from crates.io
cargo install neural-trader

# Install from Git
cargo install --git https://github.com/your-org/neural-trader
```

### 6.3 Prebuilt Binaries

```bash
# Download from GitHub Releases
curl -L https://github.com/your-org/neural-trader/releases/latest/download/neural-trader-linux-x64.tar.gz | tar xz

# Add to PATH
sudo mv neural-trader /usr/local/bin/
```

### 6.4 Package Managers

**Homebrew (macOS/Linux):**

```bash
brew install neural-trader
```

**Scoop (Windows):**

```powershell
scoop bucket add neural-trader https://github.com/your-org/scoop-neural-trader
scoop install neural-trader
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue: Binary not found after install**

```bash
# Symptoms:
$ npx neural-trader
Error: No binary found for darwin-arm64

# Solutions:
1. Check platform support: npm view neural-trader os cpu
2. Reinstall: npm uninstall -g neural-trader && npm install -g neural-trader
3. Manual install: Download binary from GitHub Releases
```

**Issue: Permission denied**

```bash
# Symptoms:
$ neural-trader --version
zsh: permission denied: neural-trader

# Solutions:
1. Make executable: chmod +x $(which neural-trader)
2. Check installation: ls -la $(which neural-trader)
```

**Issue: Config not found**

```bash
# Symptoms:
Error: Config file not found: ~/.neural-trader/config.toml

# Solutions:
1. Initialize config: neural-trader config show
2. Or create manually: mkdir -p ~/.neural-trader && touch ~/.neural-trader/config.toml
```

---

## 8. Release Checklist

### Pre-Release

- [ ] **Version Bump**: Update version in Cargo.toml and package.json
- [ ] **Changelog**: Update CHANGELOG.md with all changes
- [ ] **Tests**: All tests pass (`cargo test`)
- [ ] **Linting**: No clippy warnings (`cargo clippy`)
- [ ] **Formatting**: Code formatted (`cargo fmt`)
- [ ] **Documentation**: Update README.md and docs
- [ ] **Security Audit**: Run `cargo audit`
- [ ] **Dependency Update**: Update dependencies if needed

### Release

- [ ] **Create Tag**: `git tag -a v0.2.0 -m "Release v0.2.0"`
- [ ] **Push Tag**: `git push origin v0.2.0`
- [ ] **Wait for CI**: Ensure GitHub Actions completes successfully
- [ ] **Verify NPM**: Check package on npmjs.com
- [ ] **Verify GitHub Release**: Check binaries on GitHub Releases
- [ ] **Test Installation**: `npm install -g neural-trader@0.2.0`
- [ ] **Smoke Test**: Run smoke tests on fresh install

### Post-Release

- [ ] **Announcement**: Post release notes to blog/social media
- [ ] **Documentation**: Update docs site (if applicable)
- [ ] **Monitor**: Watch for installation issues and bug reports
- [ ] **Next Version**: Create v0.3.0 milestone and issues

---

## 9. References & Resources

### CLI Design
- **Command Line Interface Guidelines**: https://clig.dev
- **Rust CLI Book**: https://rust-cli.github.io/book/

### NPM Publishing
- **Publishing Binaries**: https://nodejs.dev/learn/publish-npm-package
- **Platform-Specific Packages**: https://docs.npmjs.com/cli/v9/configuring-npm/package-json#os
- **Semantic Versioning**: https://semver.org

### Release Automation
- **GitHub Actions**: https://docs.github.com/en/actions
- **cargo-release**: https://github.com/crate-ci/cargo-release

---

**Document Status**: ✅ Production-Ready
**Next Review**: 2026-02-12
**Contact**: devex@neural-trader.io
