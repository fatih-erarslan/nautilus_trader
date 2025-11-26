# Multi-Market NAPI Integration Plan

**Version:** 2.5.1
**Date:** 2025-11-18
**Status:** Planning Document
**Estimated Effort:** 4-6 hours

---

## 📋 Overview

The `multi-market` crate provides comprehensive support for trading across multiple market types (sports betting, prediction markets, and cryptocurrency). Currently, this crate is **not exposed via NAPI**, limiting its accessibility from JavaScript/TypeScript.

This document outlines the plan to expose multi-market functionality through NAPI bindings.

---

## 🎯 Current Status

### Existing Multi-Market Crate

**Location:** `neural-trader-rust/crates/multi-market`

**Features Implemented:**

1. **Sports Betting Module**
   - `OddsApiClient` - Live odds tracking for 30+ sports
   - `KellyOptimizer` - Kelly Criterion for optimal stake sizing
   - `Syndicate` - Syndicate management for pooled betting
   - `ArbitrageDetector` - Arbitrage opportunity detection
   - `PollingOddsStreamer` & `WebSocketOddsStreamer` - Real-time odds streaming

2. **Prediction Markets Module**
   - `PolymarketClient` - Polymarket CLOB API integration
   - `SentimentAnalyzer` - Sentiment analysis for market manipulation detection
   - `ExpectedValueCalculator` - Expected value calculation
   - `OrderbookAnalyzer` - Orderbook analysis and market making
   - `MarketMakingStrategy` - Market making strategies
   - `ArbitrageDetector` - Multi-market arbitrage detection

3. **Cryptocurrency Module**
   - `DefiManager` - DeFi yield optimization
   - `ArbitrageEngine` - Cross-exchange arbitrage
   - `YieldFarmingStrategy` - Automated yield farming
   - `GasOptimizer` - Gas optimization for transactions
   - `DexArbitrageStrategy` - DEX arbitrage strategies
   - `LiquidityPoolStrategy` - Liquidity pool optimization

### Current Integration Status

- ✅ Rust crate fully implemented
- ✅ Integrated in workspace (enabled in Cargo.toml)
- ❌ **No NAPI bindings** - Not exposed to JavaScript
- ❌ **Not in CLI registry** - Not accessible via `neural-trader list`

---

## 🔌 Proposed NAPI Functions

### Sports Betting (8 functions)

```javascript
// OddsApiClient
multiMarketSportsFetchOdds(apiKey, sport, region, markets)
multiMarketSportsListSports()
multiMarketSportsStreamOdds(apiKey, sport, callback)

// Kelly Criterion
multiMarketSportsCalculateKelly(bankroll, edge, fractional)
multiMarketSportsOptimizeStakes(opportunities, bankroll)

// Arbitrage
multiMarketSportsFindArbitrage(odds, bookmakers)
multiMarketSportsSyndicateCreate(name, members, bankroll)
multiMarketSportsSyndicateDistribute(syndicateId, profits, distributionModel)
```

### Prediction Markets (7 functions)

```javascript
// Polymarket
multiMarketPredictionFetchMarkets(query, limit)
multiMarketPredictionGetOrderbook(marketId)
multiMarketPredictionPlaceOrder(marketId, side, price, size)

// Analysis
multiMarketPredictionAnalyzeSentiment(marketId, sources)
multiMarketPredictionCalculateEV(marketId, trueProb)
multiMarketPredictionFindArbitrage(markets)
multiMarketPredictionMarketMaking(marketId, spread, inventory)
```

### Cryptocurrency (9 functions)

```javascript
// DeFi
multiMarketCryptoGetYieldOpportunities(protocols, assets)
multiMarketCryptoOptimizeYield(capital, risk, protocols)
multiMarketCryptoFarmYield(protocol, pool, amount)

// Arbitrage
multiMarketCryptoFindArbitrage(asset, exchanges)
multiMarketCryptoExecuteArbitrage(opportunity, amount)
multiMarketCryptoDexArbitrage(tokens, dexes)

// Gas & Liquidity
multiMarketCryptoOptimizeGas(transaction, network)
multiMarketCryptoProvideLiquidity(pool, token0, token1, amount)
multiMarketCryptoRebalanceLiquidity(positions, targetRatios)
```

**Total New Functions:** 24

---

## 🛠️ Implementation Steps

### Step 1: Create NAPI Module Structure (1 hour)

**File:** `neural-trader-rust/crates/napi-bindings/src/multi_market.rs`

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

// Sports betting functions
#[napi]
pub async fn multi_market_sports_fetch_odds(
    api_key: String,
    sport: String,
    region: String,
    markets: Vec<String>,
) -> Result<String> {
    // Implementation
}

#[napi]
pub fn multi_market_sports_calculate_kelly(
    bankroll: f64,
    edge: f64,
    fractional: f64,
) -> Result<f64> {
    // Implementation
}

// ... more functions
```

### Step 2: Update napi-bindings/lib.rs (15 minutes)

```rust
// In neural-trader-rust/crates/napi-bindings/src/lib.rs

mod multi_market;
pub use multi_market::*;
```

### Step 3: Update napi-bindings Cargo.toml (15 minutes)

```toml
[dependencies]
multi-market = { path = "../multi-market" }
```

### Step 4: Update index.js (30 minutes)

```javascript
// In index.js

// Multi-Market - Sports Betting (8)
const {
  multiMarketSportsFetchOdds,
  multiMarketSportsListSports,
  multiMarketSportsStreamOdds,
  multiMarketSportsCalculateKelly,
  multiMarketSportsOptimizeStakes,
  multiMarketSportsFindArbitrage,
  multiMarketSportsSyndicateCreate,
  multiMarketSportsSyndicateDistribute
} = nativeBinding;

// Multi-Market - Prediction Markets (7)
const {
  multiMarketPredictionFetchMarkets,
  multiMarketPredictionGetOrderbook,
  multiMarketPredictionPlaceOrder,
  multiMarketPredictionAnalyzeSentiment,
  multiMarketPredictionCalculateEV,
  multiMarketPredictionFindArbitrage,
  multiMarketPredictionMarketMaking
} = nativeBinding;

// Multi-Market - Cryptocurrency (9)
const {
  multiMarketCryptoGetYieldOpportunities,
  multiMarketCryptoOptimizeYield,
  multiMarketCryptoFarmYield,
  multiMarketCryptoFindArbitrage,
  multiMarketCryptoExecuteArbitrage,
  multiMarketCryptoDexArbitrage,
  multiMarketCryptoOptimizeGas,
  multiMarketCryptoProvideLiquidity,
  multiMarketCryptoRebalanceLiquidity
} = nativeBinding;

// Export all new functions
module.exports = {
  // ... existing exports

  // Multi-Market - Sports Betting
  multiMarketSportsFetchOdds,
  multiMarketSportsListSports,
  multiMarketSportsStreamOdds,
  multiMarketSportsCalculateKelly,
  multiMarketSportsOptimizeStakes,
  multiMarketSportsFindArbitrage,
  multiMarketSportsSyndicateCreate,
  multiMarketSportsSyndicateDistribute,

  // Multi-Market - Prediction Markets
  multiMarketPredictionFetchMarkets,
  multiMarketPredictionGetOrderbook,
  multiMarketPredictionPlaceOrder,
  multiMarketPredictionAnalyzeSentiment,
  multiMarketPredictionCalculateEV,
  multiMarketPredictionFindArbitrage,
  multiMarketPredictionMarketMaking,

  // Multi-Market - Cryptocurrency
  multiMarketCryptoGetYieldOpportunities,
  multiMarketCryptoOptimizeYield,
  multiMarketCryptoFarmYield,
  multiMarketCryptoFindArbitrage,
  multiMarketCryptoExecuteArbitrage,
  multiMarketCryptoDexArbitrage,
  multiMarketCryptoOptimizeGas,
  multiMarketCryptoProvideLiquidity,
  multiMarketCryptoRebalanceLiquidity
};
```

### Step 5: Register in CLI (30 minutes)

**File:** `src/cli/data/packages.js`

```javascript
'multi-market': {
  name: 'Multi-Market Trading',
  description: 'Cross-market trading support for sports betting, prediction markets, and crypto',
  category: 'trading',
  version: '2.0.0',
  size: '18.5 MB',
  packages: ['@neural-trader/multi-market'],
  dependencies: ['@neural-trader/core', '@neural-trader/risk'],
  features: [
    'Sports betting with Kelly Criterion',
    'Prediction market trading',
    'Cross-exchange crypto arbitrage',
    'DeFi yield optimization',
    'Real-time odds streaming',
    'Multi-market arbitrage detection'
  ],
  hasExamples: true,
  installed: false
},
```

### Step 6: Build & Test (1 hour)

```bash
# Build NAPI bindings
cd neural-trader-rust/crates/napi-bindings
npm run build

# Test exports
node -e "const nt = require('../../../index.js'); console.log(typeof nt.multiMarketSportsFetchOdds);"

# Run regression tests
npm test

# Test CLI
neural-trader list
neural-trader info multi-market
```

### Step 7: Documentation (1 hour)

Create `docs/api/multi-market.md` documenting:
- All 24 new functions
- Parameters and return types
- Code examples
- Error handling
- API keys and configuration

---

## 📊 Impact Analysis

### Benefits

1. **Expanded Trading Capabilities**
   - Access to sports betting markets
   - Prediction market integration
   - Cross-market arbitrage opportunities

2. **Unified API**
   - Single JavaScript interface for all market types
   - Consistent error handling
   - Standardized data formats

3. **Production Ready**
   - Rust implementation already battle-tested
   - Just needs NAPI exposure layer

### Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes | Medium | Thorough regression testing before release |
| API key management | Medium | Clear documentation, secure defaults |
| Rate limiting | Low | Built-in rate limiting in Rust layer |
| Complex async flows | Low | Well-tested async patterns already in use |

---

## 🔄 Alternative Approaches

### Option 1: Full NAPI Integration (Recommended)

- **Pros:** Complete access, best performance, native types
- **Cons:** 4-6 hours implementation, need to maintain NAPI layer
- **Status:** This document

### Option 2: CLI Wrapper Only

- **Pros:** Faster implementation (2 hours)
- **Cons:** Limited flexibility, no programmatic access, CLI-only
- **Approach:** Add CLI commands without NAPI

### Option 3: Separate Package

- **Pros:** Independent release cycle, focused scope
- **Cons:** More complex distribution, split ecosystem
- **Approach:** Create `@neural-trader/multi-market` NPM package

---

## 📅 Release Plan

### Version 2.6.0 (Target for Multi-Market)

**Estimated Timeline:** 2 weeks

**Phase 1 (Week 1):**
- ✅ Planning document (this document)
- 🔲 Implement NAPI bindings (Steps 1-3)
- 🔲 Update exports and CLI registry (Steps 4-5)

**Phase 2 (Week 2):**
- 🔲 Comprehensive testing (Step 6)
- 🔲 Documentation (Step 7)
- 🔲 Create examples
- 🔲 Release v2.6.0

---

## ✅ Testing Checklist

- [ ] All 24 functions callable from JavaScript
- [ ] Type definitions correct
- [ ] Async functions return Promises
- [ ] Error handling works correctly
- [ ] CLI commands accessible
- [ ] Examples run successfully
- [ ] Zero regressions in existing functionality
- [ ] Performance benchmarks meet targets
- [ ] Documentation complete and accurate

---

## 📚 Resources

- Multi-Market Crate: `neural-trader-rust/crates/multi-market`
- NAPI-RS Docs: https://napi.rs/
- Existing NAPI Bindings: `neural-trader-rust/crates/napi-bindings`
- Similar Implementation: Sports betting crate (already has NAPI)

---

## 🎯 Success Criteria

1. ✅ All 24 functions exposed via NAPI
2. ✅ Package registered in CLI (`neural-trader list`)
3. ✅ Complete API documentation
4. ✅ Working examples for all three modules
5. ✅ Zero regressions (41+ tests passing)
6. ✅ Performance: <100ms average response time
7. ✅ 100% backward compatibility

---

## 📝 Notes

- The multi-market crate is already production-ready
- Main work is NAPI glue code, not core functionality
- This is a value-add feature, not a critical bug fix
- Can be implemented incrementally (sports → prediction → crypto)
- Should maintain consistency with existing NAPI patterns

---

**Status:** Planning Complete
**Priority:** Medium (Enhancement, not critical)
**Complexity:** Low-Medium (Well-defined scope)
**Effort:** 4-6 hours
**Target Release:** v2.6.0

---

**Generated:** 2025-11-18
**Author:** Claude Code AI
**Review Status:** Pending Technical Review
