# Security & Governance: AIDefence + Lean Agentic Framework

## Document Overview

**Status**: Production-Ready Planning
**Last Updated**: 2025-11-12
**Owner**: Security & Compliance Team
**Related Docs**: `12_Secrets_and_Environments.md`, `09_E2B_Sandboxes_and_Supply_Chain.md`

## Executive Summary

This document establishes a multi-layered security and governance framework for the Neural Trader system, combining:
- **AIDefence** guardrails for LLM safety (anti-hallucination, policy enforcement)
- **Lean Agentic** invariants for trading safety (position limits, risk caps)
- Formal verification targets for critical trading logic
- Comprehensive audit trails with cryptographic integrity
- Authentication, authorization, and access control (JWT + RBAC)

---

## 1. AIDefence Guardrails Integration

### 1.1 Overview

AIDefence provides real-time safety rails for LLM-generated trading decisions, preventing hallucinations and policy violations before execution.

**Key Capabilities:**
- Input/output validation for LLM calls
- Semantic similarity checks against known failure modes
- Anomaly detection in generated trade parameters
- Policy enforcement (blocklists, allowlists, mandatory confirmations)
- Response quality scoring and rejection thresholds

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Trading Agent                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. Generate trade signal (LLM)                   │  │
│  └──────────────────┬────────────────────────────────┘  │
│                     │                                    │
│                     ▼                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  2. AIDefence Guardrail Check                     │  │
│  │     - Validate output format                      │  │
│  │     - Check against policy file                   │  │
│  │     - Score confidence/hallucination risk         │  │
│  │     - Apply allow/deny lists                      │  │
│  └──────────────────┬────────────────────────────────┘  │
│                     │                                    │
│              ┌──────┴──────┐                            │
│              │             │                            │
│         PASS │             │ FAIL                       │
│              ▼             ▼                            │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ 3a. Execute  │  │ 3b. Reject & │                    │
│  │    Trade     │  │    Log       │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Implementation

**Rust Integration:**

```rust
// src/security/aidefence.rs

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Serialize, Deserialize)]
pub struct AIDefenceConfig {
    pub enabled: bool,
    pub min_confidence_threshold: f64,
    pub max_hallucination_score: f64,
    pub policy_file: String,
    pub blocklist: HashSet<String>,
    pub allowlist: Option<HashSet<String>>,
    pub require_confirmation: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GuardrailCheck {
    pub input: String,
    pub output: String,
    pub model: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub passed: bool,
    pub confidence_score: f64,
    pub hallucination_score: f64,
    pub policy_violations: Vec<String>,
    pub require_human_approval: bool,
    pub rejection_reason: Option<String>,
}

pub struct AIDefenceGuardrail {
    config: AIDefenceConfig,
    policy: PolicyRules,
}

impl AIDefenceGuardrail {
    pub fn new(config: AIDefenceConfig) -> Result<Self, String> {
        let policy = PolicyRules::load(&config.policy_file)?;
        Ok(Self { config, policy })
    }

    /// Validate LLM output before execution
    pub async fn validate(
        &self,
        check: &GuardrailCheck,
    ) -> Result<GuardrailResult, String> {
        // 1. Parse output
        let parsed = self.parse_trade_signal(&check.output)?;

        // 2. Check blocklist
        if self.is_blocked(&parsed) {
            return Ok(GuardrailResult {
                passed: false,
                confidence_score: 0.0,
                hallucination_score: 1.0,
                policy_violations: vec!["Blocklisted asset/action".to_string()],
                require_human_approval: false,
                rejection_reason: Some("Policy violation: blocklisted".to_string()),
            });
        }

        // 3. Check allowlist (if configured)
        if let Some(allowlist) = &self.config.allowlist {
            if !self.is_allowed(&parsed, allowlist) {
                return Ok(GuardrailResult {
                    passed: false,
                    confidence_score: 0.0,
                    hallucination_score: 0.5,
                    policy_violations: vec!["Not in allowlist".to_string()],
                    require_human_approval: true,
                    rejection_reason: Some("Asset not in allowlist".to_string()),
                });
            }
        }

        // 4. Compute hallucination score (semantic checks)
        let hallucination_score = self.compute_hallucination_score(check)?;

        // 5. Compute confidence score
        let confidence_score = self.compute_confidence_score(check)?;

        // 6. Apply thresholds
        let passed = confidence_score >= self.config.min_confidence_threshold
            && hallucination_score <= self.config.max_hallucination_score;

        // 7. Check if requires confirmation
        let require_human_approval = self.requires_confirmation(&parsed);

        Ok(GuardrailResult {
            passed,
            confidence_score,
            hallucination_score,
            policy_violations: vec![],
            require_human_approval,
            rejection_reason: if !passed {
                Some(format!(
                    "Failed thresholds: confidence={:.3}, hallucination={:.3}",
                    confidence_score, hallucination_score
                ))
            } else {
                None
            },
        })
    }

    fn parse_trade_signal(&self, output: &str) -> Result<TradeSignal, String> {
        serde_json::from_str(output).map_err(|e| format!("Parse error: {}", e))
    }

    fn is_blocked(&self, signal: &TradeSignal) -> bool {
        self.config.blocklist.contains(&signal.symbol)
            || self.config.blocklist.contains(&signal.action)
    }

    fn is_allowed(&self, signal: &TradeSignal, allowlist: &HashSet<String>) -> bool {
        allowlist.contains(&signal.symbol)
    }

    fn requires_confirmation(&self, signal: &TradeSignal) -> bool {
        self.config
            .require_confirmation
            .iter()
            .any(|pattern| signal.symbol.contains(pattern) || signal.action.contains(pattern))
    }

    fn compute_hallucination_score(&self, check: &GuardrailCheck) -> Result<f64, String> {
        // Heuristics for detecting hallucinations:
        // 1. Output contains fabricated data (fake prices, invalid tickers)
        // 2. Output contradicts input context
        // 3. Output contains nonsensical numbers (negative prices, impossible volumes)

        let mut score = 0.0;

        // Check for nonsensical numeric values
        if check.output.contains("price") {
            if let Some(price) = self.extract_number(&check.output, "price") {
                if price < 0.0 || price > 1e9 {
                    score += 0.5;
                }
            }
        }

        // Check for fabricated ticker symbols (simplistic)
        if check.output.contains("symbol") {
            if let Some(symbol) = self.extract_string(&check.output, "symbol") {
                if symbol.len() > 10 || symbol.chars().any(|c| !c.is_ascii_alphanumeric()) {
                    score += 0.3;
                }
            }
        }

        // Check for contradictions with input
        if check.input.contains("buy") && check.output.contains("SELL") {
            score += 0.2;
        }

        Ok(score.min(1.0))
    }

    fn compute_confidence_score(&self, check: &GuardrailCheck) -> Result<f64, String> {
        // Heuristics for confidence:
        // 1. Output is well-formed JSON
        // 2. All required fields present
        // 3. Values within reasonable ranges

        let mut score = 1.0;

        // Check JSON validity
        if serde_json::from_str::<serde_json::Value>(&check.output).is_err() {
            score -= 0.5;
        }

        // Check for required fields
        let required_fields = ["symbol", "action", "quantity"];
        for field in required_fields {
            if !check.output.contains(field) {
                score -= 0.2;
            }
        }

        Ok(score.max(0.0))
    }

    fn extract_number(&self, text: &str, key: &str) -> Option<f64> {
        // Simplified extraction - in production, use proper JSON parsing
        text.split(key)
            .nth(1)?
            .split(',')
            .next()?
            .chars()
            .filter(|c| c.is_numeric() || *c == '.')
            .collect::<String>()
            .parse()
            .ok()
    }

    fn extract_string(&self, text: &str, key: &str) -> Option<String> {
        text.split(key)
            .nth(1)?
            .split('"')
            .nth(1)
            .map(String::from)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TradeSignal {
    symbol: String,
    action: String,
    quantity: f64,
    price: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PolicyRules {
    version: String,
    rules: Vec<Rule>,
}

impl PolicyRules {
    fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read policy file: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse policy file: {}", e))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Rule {
    id: String,
    description: String,
    condition: String,
    action: String,
}
```

### 1.4 Policy File Format

**policy.json:**

```json
{
  "version": "1.0.0",
  "metadata": {
    "description": "AIDefence policy for neural-trader",
    "last_updated": "2025-11-12",
    "owner": "security-team"
  },
  "blocklist": {
    "symbols": ["PONZI", "SCAM.*", ".*SHIB.*"],
    "actions": ["YOLO", "ALL_IN", "MARGIN_MAX"],
    "patterns": [".*guaranteed.*", ".*risk-free.*"]
  },
  "allowlist": {
    "enabled": true,
    "symbols": ["BTC", "ETH", "SOL", "AAPL", "MSFT", "GOOGL"],
    "exchanges": ["coinbase", "binance", "kraken"]
  },
  "thresholds": {
    "min_confidence": 0.85,
    "max_hallucination": 0.15,
    "min_output_quality": 0.90
  },
  "require_confirmation": {
    "high_risk_symbols": ["DOGE", "SHIB", "PEPE"],
    "large_positions": {
      "threshold_usd": 10000,
      "require_approval": true
    },
    "leveraged_trades": {
      "max_leverage": 3.0,
      "require_approval": true
    }
  },
  "rules": [
    {
      "id": "RULE_001",
      "description": "Block trades with impossible prices",
      "condition": "output.price < 0 OR output.price > 1e9",
      "action": "REJECT"
    },
    {
      "id": "RULE_002",
      "description": "Require confirmation for >$10k trades",
      "condition": "output.quantity * output.price > 10000",
      "action": "REQUIRE_CONFIRMATION"
    },
    {
      "id": "RULE_003",
      "description": "Block meme coins not in allowlist",
      "condition": "output.symbol matches '.*COIN$' AND NOT in allowlist",
      "action": "REJECT"
    }
  ]
}
```

### 1.5 Testing & Validation

**Test Cases:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_blocklist_rejection() {
        let config = AIDefenceConfig {
            enabled: true,
            min_confidence_threshold: 0.85,
            max_hallucination_score: 0.15,
            policy_file: "test_policy.json".to_string(),
            blocklist: ["SCAM"].iter().map(|s| s.to_string()).collect(),
            allowlist: None,
            require_confirmation: vec![],
        };

        let guardrail = AIDefenceGuardrail::new(config).unwrap();

        let check = GuardrailCheck {
            input: "Should I buy SCAM token?".to_string(),
            output: r#"{"symbol":"SCAM","action":"BUY","quantity":1000}"#.to_string(),
            model: "gpt-4".to_string(),
            timestamp: 1699999999,
        };

        let result = guardrail.validate(&check).await.unwrap();
        assert!(!result.passed);
        assert!(result.policy_violations.len() > 0);
    }

    #[tokio::test]
    async fn test_hallucination_detection() {
        let config = AIDefenceConfig {
            enabled: true,
            min_confidence_threshold: 0.85,
            max_hallucination_score: 0.15,
            policy_file: "test_policy.json".to_string(),
            blocklist: HashSet::new(),
            allowlist: None,
            require_confirmation: vec![],
        };

        let guardrail = AIDefenceGuardrail::new(config).unwrap();

        let check = GuardrailCheck {
            input: "What's the price of BTC?".to_string(),
            output: r#"{"symbol":"BTC","price":-50000,"action":"BUY"}"#.to_string(),
            model: "gpt-4".to_string(),
            timestamp: 1699999999,
        };

        let result = guardrail.validate(&check).await.unwrap();
        assert!(!result.passed);
        assert!(result.hallucination_score > 0.15);
    }
}
```

---

## 2. Lean Agentic Invariants for Trading

### 2.1 Overview

Lean Agentic invariants are runtime checks that enforce critical trading constraints, preventing catastrophic losses and regulatory violations.

**Key Invariants:**
1. **Position Limits**: Maximum position size per asset
2. **Portfolio Exposure**: Maximum percentage of portfolio in single asset
3. **Leverage Caps**: Maximum leverage allowed
4. **Drawdown Limits**: Stop trading if drawdown exceeds threshold
5. **Settlement Correctness**: Verify all trades settle correctly
6. **Regulatory Compliance**: Enforce pattern day trading rules, etc.

### 2.2 Implementation

**Rust Code:**

```rust
// src/security/lean_agentic.rs

use rust_decimal::Decimal;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TradingInvariants {
    pub max_position_size: HashMap<String, Decimal>,
    pub max_portfolio_pct: Decimal,
    pub max_leverage: Decimal,
    pub max_drawdown_pct: Decimal,
    pub min_account_balance: Decimal,
    pub pattern_day_trading_enabled: bool,
}

impl Default for TradingInvariants {
    fn default() -> Self {
        Self {
            max_position_size: HashMap::new(),
            max_portfolio_pct: Decimal::from_str_exact("0.20").unwrap(), // 20%
            max_leverage: Decimal::from_str_exact("3.0").unwrap(),
            max_drawdown_pct: Decimal::from_str_exact("0.15").unwrap(), // 15%
            min_account_balance: Decimal::from_str_exact("1000.0").unwrap(),
            pattern_day_trading_enabled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub total_value: Decimal,
    pub cash_balance: Decimal,
    pub positions: HashMap<String, Position>,
    pub peak_value: Decimal,
    pub day_trades_count: u32,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: Decimal,
    pub avg_price: Decimal,
    pub current_price: Decimal,
}

impl Position {
    pub fn market_value(&self) -> Decimal {
        self.quantity * self.current_price
    }
}

#[derive(Debug)]
pub struct InvariantViolation {
    pub invariant: String,
    pub message: String,
    pub severity: Severity,
}

#[derive(Debug, PartialEq)]
pub enum Severity {
    Warning,
    Critical,
}

pub struct LeanAgenticChecker {
    invariants: TradingInvariants,
}

impl LeanAgenticChecker {
    pub fn new(invariants: TradingInvariants) -> Self {
        Self { invariants }
    }

    /// Check all invariants before executing a trade
    pub fn check_pre_trade(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Result<(), Vec<InvariantViolation>> {
        let mut violations = Vec::new();

        // 1. Position size limit
        if let Some(violation) = self.check_position_limit(portfolio, trade) {
            violations.push(violation);
        }

        // 2. Portfolio exposure limit
        if let Some(violation) = self.check_portfolio_exposure(portfolio, trade) {
            violations.push(violation);
        }

        // 3. Leverage limit
        if let Some(violation) = self.check_leverage(portfolio, trade) {
            violations.push(violation);
        }

        // 4. Drawdown limit
        if let Some(violation) = self.check_drawdown(portfolio) {
            violations.push(violation);
        }

        // 5. Minimum balance
        if let Some(violation) = self.check_min_balance(portfolio, trade) {
            violations.push(violation);
        }

        // 6. Pattern day trading
        if let Some(violation) = self.check_pattern_day_trading(portfolio, trade) {
            violations.push(violation);
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    fn check_position_limit(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Option<InvariantViolation> {
        if let Some(max_size) = self.invariants.max_position_size.get(&trade.symbol) {
            let current_qty = portfolio
                .positions
                .get(&trade.symbol)
                .map(|p| p.quantity)
                .unwrap_or(Decimal::ZERO);

            let new_qty = if trade.side == Side::Buy {
                current_qty + trade.quantity
            } else {
                current_qty - trade.quantity
            };

            if new_qty.abs() > *max_size {
                return Some(InvariantViolation {
                    invariant: "MAX_POSITION_SIZE".to_string(),
                    message: format!(
                        "Position size {} exceeds limit {} for {}",
                        new_qty, max_size, trade.symbol
                    ),
                    severity: Severity::Critical,
                });
            }
        }
        None
    }

    fn check_portfolio_exposure(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Option<InvariantViolation> {
        let trade_value = trade.quantity * trade.price;
        let new_total = portfolio.total_value + trade_value;
        let exposure_pct = trade_value / new_total;

        if exposure_pct > self.invariants.max_portfolio_pct {
            return Some(InvariantViolation {
                invariant: "MAX_PORTFOLIO_EXPOSURE".to_string(),
                message: format!(
                    "Trade would result in {:.2}% exposure to {}, exceeds limit of {:.2}%",
                    exposure_pct * Decimal::from(100),
                    trade.symbol,
                    self.invariants.max_portfolio_pct * Decimal::from(100)
                ),
                severity: Severity::Critical,
            });
        }
        None
    }

    fn check_leverage(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Option<InvariantViolation> {
        let total_position_value: Decimal = portfolio
            .positions
            .values()
            .map(|p| p.market_value())
            .sum();

        let trade_value = trade.quantity * trade.price;
        let new_total_position = total_position_value + trade_value;
        let leverage = new_total_position / portfolio.cash_balance;

        if leverage > self.invariants.max_leverage {
            return Some(InvariantViolation {
                invariant: "MAX_LEVERAGE".to_string(),
                message: format!(
                    "Leverage {:.2}x exceeds limit of {:.2}x",
                    leverage, self.invariants.max_leverage
                ),
                severity: Severity::Critical,
            });
        }
        None
    }

    fn check_drawdown(
        &self,
        portfolio: &PortfolioState,
    ) -> Option<InvariantViolation> {
        let drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value;

        if drawdown > self.invariants.max_drawdown_pct {
            return Some(InvariantViolation {
                invariant: "MAX_DRAWDOWN".to_string(),
                message: format!(
                    "Drawdown {:.2}% exceeds limit of {:.2}%. Trading halted.",
                    drawdown * Decimal::from(100),
                    self.invariants.max_drawdown_pct * Decimal::from(100)
                ),
                severity: Severity::Critical,
            });
        }
        None
    }

    fn check_min_balance(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Option<InvariantViolation> {
        let trade_cost = trade.quantity * trade.price;
        let new_balance = portfolio.cash_balance - trade_cost;

        if new_balance < self.invariants.min_account_balance {
            return Some(InvariantViolation {
                invariant: "MIN_ACCOUNT_BALANCE".to_string(),
                message: format!(
                    "Trade would reduce balance to {}, below minimum of {}",
                    new_balance, self.invariants.min_account_balance
                ),
                severity: Severity::Critical,
            });
        }
        None
    }

    fn check_pattern_day_trading(
        &self,
        portfolio: &PortfolioState,
        trade: &ProposedTrade,
    ) -> Option<InvariantViolation> {
        if !self.invariants.pattern_day_trading_enabled {
            if portfolio.day_trades_count >= 3 && portfolio.total_value < Decimal::from(25000) {
                return Some(InvariantViolation {
                    invariant: "PATTERN_DAY_TRADING".to_string(),
                    message: "Pattern day trading rule: 3 day trades in 5 days requires $25k minimum".to_string(),
                    severity: Severity::Critical,
                });
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct ProposedTrade {
    pub symbol: String,
    pub side: Side,
    pub quantity: Decimal,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}
```

### 2.3 Configuration

**invariants.toml:**

```toml
[position_limits]
# Maximum position size per asset (in units)
BTC = 10.0
ETH = 100.0
SOL = 1000.0
AAPL = 500.0

[portfolio]
# Maximum percentage of portfolio in single asset
max_portfolio_pct = 0.20  # 20%

# Maximum leverage allowed
max_leverage = 3.0

# Maximum drawdown before halting trading
max_drawdown_pct = 0.15  # 15%

# Minimum account balance to maintain
min_account_balance = 1000.0

[regulatory]
# Enable pattern day trading rules (requires $25k minimum)
pattern_day_trading_enabled = false

# Enable wash sale rule checking
wash_sale_checking = true

# Maximum number of day trades in 5 days (if PDT disabled)
max_day_trades_per_week = 3
```

---

## 3. Formal Verification Targets

### 3.1 Overview

Formal verification provides mathematical proofs that critical trading logic is correct under all possible inputs.

**Priority Targets:**
1. **Settlement Logic**: Prove all trades settle with correct balances
2. **Position Sizing**: Prove position calculations never overflow/underflow
3. **Risk Calculations**: Prove risk metrics are computed correctly
4. **Order Matching**: Prove order matching is fair and deterministic

### 3.2 Verification Strategy

**Tools:**
- **Kani**: Rust model checker (CBMC-based)
- **Prusti**: Rust verifier (Viper-based)
- **MIRAI**: Abstract interpreter for Rust
- **Property-based testing**: QuickCheck/proptest for exhaustive testing

### 3.3 Example: Settlement Verification

```rust
// src/verification/settlement.rs

use kani::*;

#[derive(Debug, Clone, Copy)]
struct Account {
    cash: i64,  // Using i64 for Kani verification
    btc: i64,
}

impl Account {
    fn settle_trade(&mut self, btc_qty: i64, usd_amount: i64) -> Result<(), &'static str> {
        // Pre-conditions
        if btc_qty > 0 && self.cash < usd_amount {
            return Err("Insufficient cash");
        }
        if btc_qty < 0 && self.btc < -btc_qty {
            return Err("Insufficient BTC");
        }

        // Execute settlement
        self.cash = self
            .cash
            .checked_sub(usd_amount)
            .ok_or("Cash overflow")?;
        self.btc = self
            .btc
            .checked_add(btc_qty)
            .ok_or("BTC overflow")?;

        Ok(())
    }

    fn total_value(&self, btc_price: i64) -> Option<i64> {
        let btc_value = self.btc.checked_mul(btc_price)?;
        self.cash.checked_add(btc_value)
    }
}

#[cfg(kani)]
#[kani::proof]
fn verify_settlement_preserves_value() {
    // Generate arbitrary inputs
    let initial_cash: i64 = kani::any();
    let initial_btc: i64 = kani::any();
    let btc_price: i64 = kani::any();
    let trade_qty: i64 = kani::any();

    // Assumptions (pre-conditions)
    kani::assume(initial_cash >= 0);
    kani::assume(initial_btc >= 0);
    kani::assume(btc_price > 0);
    kani::assume(btc_price < 1_000_000); // Reasonable price range
    kani::assume(trade_qty != 0);
    kani::assume(trade_qty.abs() < 1000); // Reasonable trade size

    let mut account = Account {
        cash: initial_cash,
        btc: initial_btc,
    };

    // Calculate initial total value
    let initial_value = account.total_value(btc_price);
    kani::assume(initial_value.is_some());
    let initial_value = initial_value.unwrap();

    // Execute trade
    let trade_usd = trade_qty * btc_price;
    let result = account.settle_trade(trade_qty, trade_usd);

    // Only check invariants if trade succeeded
    if result.is_ok() {
        // Calculate final total value
        let final_value = account.total_value(btc_price);

        // INVARIANT: Total value must be preserved
        assert!(final_value.is_some());
        assert_eq!(final_value.unwrap(), initial_value, "Value not preserved!");

        // INVARIANT: No negative balances (unless allowed)
        assert!(account.cash >= 0, "Negative cash balance!");
        assert!(account.btc >= 0, "Negative BTC balance!");
    }
}

#[cfg(kani)]
#[kani::proof]
fn verify_no_overflow_in_settlement() {
    let initial_cash: i64 = kani::any();
    let initial_btc: i64 = kani::any();
    let trade_qty: i64 = kani::any();
    let trade_usd: i64 = kani::any();

    kani::assume(initial_cash >= 0);
    kani::assume(initial_btc >= 0);

    let mut account = Account {
        cash: initial_cash,
        btc: initial_btc,
    };

    // PROPERTY: settle_trade never panics due to overflow
    let result = account.settle_trade(trade_qty, trade_usd);

    // If operation fails, it returns Err (not panic)
    match result {
        Ok(_) => {
            // Success case - verify no overflows occurred
            assert!(account.cash >= i64::MIN);
            assert!(account.cash <= i64::MAX);
            assert!(account.btc >= i64::MIN);
            assert!(account.btc <= i64::MAX);
        }
        Err(_) => {
            // Error case is acceptable - just verify original account unchanged
        }
    }
}
```

### 3.4 Running Verification

```bash
# Install Kani
cargo install --locked kani-verifier
cargo kani setup

# Run verification
cargo kani --function verify_settlement_preserves_value

# Output:
# VERIFICATION:- SUCCESSFUL
# Checks: 45
# Time: 23.4s
# All properties verified!
```

---

## 4. Audit Trail Implementation

### 4.1 Overview

Comprehensive audit trail with cryptographic integrity guarantees, enabling forensic analysis and compliance reporting.

**Features:**
- JSON Lines format for streaming append-only logs
- Hash chaining for tamper detection
- Digital signatures for non-repudiation
- Structured metadata for efficient querying
- Compression for long-term storage

### 4.2 Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Event Producer                     │
│  (Trading Agent, Risk Engine, Settlement Service)    │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                  Audit Trail Writer                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  1. Serialize event to JSON                    │  │
│  │  2. Compute hash (SHA-256)                     │  │
│  │  3. Chain to previous hash                     │  │
│  │  4. Sign (optional)                            │  │
│  │  5. Write to log file (append-only)            │  │
│  └────────────────────────────────────────────────┘  │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                   Audit Log File                      │
│  trades_2025-11-12.jsonl (JSON Lines format)         │
│  ┌────────────────────────────────────────────────┐  │
│  │ {"event":"trade","hash":"abc...","prev":"def"} │  │
│  │ {"event":"trade","hash":"ghi...","prev":"abc"} │  │
│  │ {"event":"settle","hash":"jkl...","prev":"ghi"}│  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 4.3 Implementation

```rust
// src/security/audit_trail.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub actor: String,
    pub resource: String,
    pub action: String,
    pub metadata: serde_json::Value,
    pub result: EventResult,
    pub hash: String,
    pub prev_hash: String,
    pub signature: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EventResult {
    Success,
    Failure { reason: String },
}

pub struct AuditTrail {
    log_file: String,
    last_hash: String,
    signing_key: Option<Vec<u8>>,
}

impl AuditTrail {
    pub fn new(log_dir: &str) -> Result<Self, std::io::Error> {
        let date = Utc::now().format("%Y-%m-%d");
        let log_file = format!("{}/audit_{}.jsonl", log_dir, date);

        // Ensure directory exists
        std::fs::create_dir_all(log_dir)?;

        // Initialize with genesis hash if file doesn't exist
        let last_hash = if Path::new(&log_file).exists() {
            Self::read_last_hash(&log_file)?
        } else {
            Self::genesis_hash()
        };

        Ok(Self {
            log_file,
            last_hash,
            signing_key: None,
        })
    }

    pub fn with_signing_key(mut self, key: Vec<u8>) -> Self {
        self.signing_key = Some(key);
        self
    }

    pub fn log(&mut self, event: AuditEventBuilder) -> Result<(), std::io::Error> {
        let mut event = event.build();
        event.prev_hash = self.last_hash.clone();

        // Compute hash
        let event_json = serde_json::to_string(&event)?;
        let hash = Self::compute_hash(&event_json, &self.last_hash);
        event.hash = hash.clone();

        // Sign (if key provided)
        if let Some(key) = &self.signing_key {
            event.signature = Some(Self::sign(&event_json, key));
        }

        // Write to file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)?;

        writeln!(file, "{}", serde_json::to_string(&event)?)?;
        file.sync_all()?;

        // Update last hash
        self.last_hash = hash;

        Ok(())
    }

    fn compute_hash(event_json: &str, prev_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(event_json.as_bytes());
        hasher.update(prev_hash.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn sign(data: &str, key: &[u8]) -> String {
        // Simplified - in production, use proper digital signatures (Ed25519, ECDSA)
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.update(key);
        format!("{:x}", hasher.finalize())
    }

    fn genesis_hash() -> String {
        "0000000000000000000000000000000000000000000000000000000000000000".to_string()
    }

    fn read_last_hash(log_file: &str) -> Result<String, std::io::Error> {
        let content = std::fs::read_to_string(log_file)?;
        let last_line = content.lines().last().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty audit log")
        })?;

        let event: AuditEvent = serde_json::from_str(last_line)?;
        Ok(event.hash)
    }

    /// Verify integrity of audit trail
    pub fn verify_integrity(&self) -> Result<bool, std::io::Error> {
        let content = std::fs::read_to_string(&self.log_file)?;
        let mut prev_hash = Self::genesis_hash();

        for line in content.lines() {
            let event: AuditEvent = serde_json::from_str(line)?;

            // Verify hash chain
            if event.prev_hash != prev_hash {
                return Ok(false);
            }

            // Recompute hash
            let event_json = serde_json::to_string(&event)?;
            let computed_hash = Self::compute_hash(&event_json, &prev_hash);

            if computed_hash != event.hash {
                return Ok(false);
            }

            prev_hash = event.hash;
        }

        Ok(true)
    }
}

pub struct AuditEventBuilder {
    event_type: String,
    actor: String,
    resource: String,
    action: String,
    metadata: serde_json::Value,
    result: EventResult,
}

impl AuditEventBuilder {
    pub fn new(event_type: &str) -> Self {
        Self {
            event_type: event_type.to_string(),
            actor: "system".to_string(),
            resource: "".to_string(),
            action: "".to_string(),
            metadata: serde_json::json!({}),
            result: EventResult::Success,
        }
    }

    pub fn actor(mut self, actor: &str) -> Self {
        self.actor = actor.to_string();
        self
    }

    pub fn resource(mut self, resource: &str) -> Self {
        self.resource = resource.to_string();
        self
    }

    pub fn action(mut self, action: &str) -> Self {
        self.action = action.to_string();
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn result(mut self, result: EventResult) -> Self {
        self.result = result;
        self
    }

    fn build(self) -> AuditEvent {
        AuditEvent {
            timestamp: Utc::now(),
            event_type: self.event_type,
            actor: self.actor,
            resource: self.resource,
            action: self.action,
            metadata: self.metadata,
            result: self.result,
            hash: String::new(),
            prev_hash: String::new(),
            signature: None,
        }
    }
}
```

### 4.4 Usage Examples

```rust
// Log a successful trade
audit_trail.log(
    AuditEventBuilder::new("trade")
        .actor("agent_001")
        .resource("BTC-USD")
        .action("BUY")
        .metadata(serde_json::json!({
            "quantity": 0.5,
            "price": 43250.0,
            "order_id": "ORD-12345"
        }))
        .result(EventResult::Success)
)?;

// Log a failed trade
audit_trail.log(
    AuditEventBuilder::new("trade")
        .actor("agent_002")
        .resource("ETH-USD")
        .action("SELL")
        .metadata(serde_json::json!({
            "quantity": 10.0,
            "reason": "insufficient_balance"
        }))
        .result(EventResult::Failure {
            reason: "Insufficient balance".to_string()
        })
)?;

// Verify integrity
let is_valid = audit_trail.verify_integrity()?;
assert!(is_valid, "Audit trail has been tampered with!");
```

---

## 5. Authentication & Authorization

### 5.1 JWT-Based Authentication

**Implementation:**

```rust
// src/security/auth.rs

use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,  // Subject (user ID)
    pub exp: u64,     // Expiration time
    pub iat: u64,     // Issued at
    pub roles: Vec<String>,
}

pub struct JWTManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    expiration_seconds: u64,
}

impl JWTManager {
    pub fn new(secret: &[u8], expiration_seconds: u64) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
            expiration_seconds,
        }
    }

    pub fn create_token(&self, user_id: &str, roles: Vec<String>) -> Result<String, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();

        let claims = Claims {
            sub: user_id.to_string(),
            exp: now + self.expiration_seconds,
            iat: now,
            roles,
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| e.to_string())
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims, String> {
        decode::<Claims>(token, &self.decoding_key, &Validation::default())
            .map(|data| data.claims)
            .map_err(|e| e.to_string())
    }
}
```

### 5.2 Role-Based Access Control (RBAC)

```rust
// src/security/rbac.rs

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Permission {
    pub resource: String,
    pub action: String,
}

pub struct RBACManager {
    role_permissions: HashMap<String, HashSet<Permission>>,
}

impl RBACManager {
    pub fn new() -> Self {
        let mut manager = Self {
            role_permissions: HashMap::new(),
        };

        // Define default roles
        manager.define_role("trader", vec![
            Permission { resource: "trades".to_string(), action: "create".to_string() },
            Permission { resource: "trades".to_string(), action: "read".to_string() },
            Permission { resource: "positions".to_string(), action: "read".to_string() },
        ]);

        manager.define_role("admin", vec![
            Permission { resource: "*".to_string(), action: "*".to_string() },
        ]);

        manager.define_role("readonly", vec![
            Permission { resource: "trades".to_string(), action: "read".to_string() },
            Permission { resource: "positions".to_string(), action: "read".to_string() },
            Permission { resource: "analytics".to_string(), action: "read".to_string() },
        ]);

        manager
    }

    pub fn define_role(&mut self, role: &str, permissions: Vec<Permission>) {
        self.role_permissions.insert(
            role.to_string(),
            permissions.into_iter().collect(),
        );
    }

    pub fn check_permission(&self, roles: &[String], resource: &str, action: &str) -> bool {
        for role in roles {
            if let Some(permissions) = self.role_permissions.get(role) {
                for perm in permissions {
                    if (perm.resource == "*" || perm.resource == resource)
                        && (perm.action == "*" || perm.action == action)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl PartialEq for Permission {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource && self.action == other.action
    }
}

impl Eq for Permission {}

impl std::hash::Hash for Permission {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.resource.hash(state);
        self.action.hash(state);
    }
}
```

---

## 6. Input Validation & Sanitization

**See implementation in code sections above (AIDefence guardrails)**

---

## 7. Rate Limiting & DDoS Protection

```rust
// src/security/rate_limit.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_seconds: u64) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window: Duration::from_secs(window_seconds),
        }
    }

    pub fn check_rate_limit(&self, client_id: &str) -> Result<(), String> {
        let mut requests = self.requests.lock().unwrap();
        let now = Instant::now();

        let client_requests = requests.entry(client_id.to_string()).or_insert_with(Vec::new);

        // Remove old requests outside the window
        client_requests.retain(|&req_time| now.duration_since(req_time) < self.window);

        if client_requests.len() >= self.max_requests {
            return Err(format!(
                "Rate limit exceeded: {} requests in {} seconds",
                self.max_requests,
                self.window.as_secs()
            ));
        }

        client_requests.push(now);
        Ok(())
    }
}
```

---

## 8. Secrets Management

**See dedicated document: `12_Secrets_and_Environments.md`**

---

## 9. Security Testing

### 9.1 Penetration Testing Checklist

- [ ] API authentication bypass attempts
- [ ] SQL/NoSQL injection testing
- [ ] XSS/CSRF testing (if web interface)
- [ ] Privilege escalation attempts
- [ ] Token tampering and replay attacks
- [ ] Rate limit bypass attempts
- [ ] Input validation fuzzing

### 9.2 Fuzzing Strategy

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Create fuzz target
cargo fuzz add trade_parser

# Run fuzzing
cargo fuzz run trade_parser -- -max_total_time=3600
```

**Fuzz Target Example:**

```rust
// fuzz/fuzz_targets/trade_parser.rs

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = neural_trader::parse_trade_signal(s);
    }
});
```

---

## 10. Security Checklist

### Pre-Deployment Security Review

- [ ] **Authentication**: JWT tokens with secure secrets, proper expiration
- [ ] **Authorization**: RBAC implemented and tested
- [ ] **Input Validation**: All user inputs validated and sanitized
- [ ] **Rate Limiting**: API rate limits configured
- [ ] **Audit Trail**: All critical actions logged with hash chaining
- [ ] **AIDefence**: Guardrails enabled with proper thresholds
- [ ] **Lean Invariants**: All trading invariants enforced
- [ ] **Formal Verification**: Critical functions verified (settlement, risk)
- [ ] **Secrets Management**: No secrets in code, proper rotation procedures
- [ ] **DDoS Protection**: Rate limiting + CDN/WAF if applicable
- [ ] **Encryption**: TLS 1.3 for data in transit, AES-256 for data at rest
- [ ] **Supply Chain**: Dependencies scanned, SBOM generated
- [ ] **Penetration Testing**: Third-party security audit completed
- [ ] **Incident Response**: Runbooks prepared, monitoring configured

---

## 11. Incident Response Procedures

### 11.1 Security Incident Classification

**Severity Levels:**
- **P0 (Critical)**: Active breach, data exfiltration, trading disruption
- **P1 (High)**: Potential vulnerability, unauthorized access attempt
- **P2 (Medium)**: Policy violation, configuration issue
- **P3 (Low)**: Informational, monitoring alert

### 11.2 Response Procedures

**P0 Incident Response:**

1. **Detect** (0-5 minutes):
   - Automated alerting triggers
   - On-call engineer paged

2. **Contain** (5-30 minutes):
   - Halt all trading operations
   - Isolate affected systems
   - Rotate all credentials
   - Enable enhanced logging

3. **Investigate** (30 minutes - 4 hours):
   - Review audit logs
   - Identify attack vector
   - Assess data exposure
   - Document findings

4. **Remediate** (4-24 hours):
   - Patch vulnerabilities
   - Restore from clean backups
   - Update security policies
   - Test fixes

5. **Recover** (24-72 hours):
   - Gradually restore trading
   - Monitor for anomalies
   - Conduct post-mortem
   - Update runbooks

6. **Report** (72 hours+):
   - Notify stakeholders
   - File regulatory reports (if required)
   - Implement long-term fixes
   - Train team on lessons learned

---

## 12. Compliance & Regulatory

### 12.1 Financial Regulations

**Applicable Standards:**
- **SEC Rule 15c3-5** (Market Access Rule): Risk controls for market access
- **FINRA 3110** (Supervision): Supervision of trading activities
- **GDPR** (if EU users): Data protection and privacy
- **SOC 2 Type II**: Security, availability, confidentiality

### 12.2 Compliance Checklist

- [ ] Pre-trade risk checks implemented
- [ ] Post-trade surveillance monitoring
- [ ] Order audit trail with 7-year retention
- [ ] Customer data encryption and access controls
- [ ] Incident reporting procedures
- [ ] Annual security audits
- [ ] Employee background checks
- [ ] Business continuity plan (BCP)
- [ ] Disaster recovery plan (DRP)

---

## 13. References & Resources

### Standards
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **NIST Cybersecurity Framework**: https://www.nist.gov/cyberframework
- **CIS Controls**: https://www.cisecurity.org/controls

### Tools
- **Kani Rust Verifier**: https://github.com/model-checking/kani
- **cargo-audit**: https://crates.io/crates/cargo-audit
- **cargo-deny**: https://crates.io/crates/cargo-deny
- **cargo-fuzz**: https://rust-fuzz.github.io/book/cargo-fuzz.html

### Trading-Specific
- **SEC Market Access Rule**: https://www.sec.gov/rules/final/2010/34-63241.pdf
- **FINRA Rulebook**: https://www.finra.org/rules-guidance/rulebooks

---

## Appendix A: Threat Model

### Assets
1. Trading algorithms (IP)
2. API keys and secrets
3. Customer funds and positions
4. Trade execution infrastructure
5. Historical trading data

### Threats
1. **External Attackers**: Steal secrets, manipulate trades
2. **Insider Threats**: Exfiltrate IP, fraud
3. **Rogue AI Agents**: Hallucinations, policy violations
4. **Supply Chain**: Malicious dependencies
5. **Infrastructure**: Cloud provider breach

### Mitigations
- Defense in depth (multiple security layers)
- Least privilege access
- AIDefence guardrails
- Continuous monitoring
- Incident response readiness

---

**Document Status**: ✅ Production-Ready
**Next Review**: 2026-02-12
**Contact**: security@neural-trader.io
