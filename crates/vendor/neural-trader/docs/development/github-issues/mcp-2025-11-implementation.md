# MCP 2025-11 Specification Implementation

## 🎯 Objective

Implement a fully compliant Model Context Protocol (MCP) 2025-11 server for Neural Trader with 99+ trading tools, following the [MCP Developer Specification (Version 2025-11)](https://gist.github.com/ruvnet/284f199d0e0836c1b5185e30f819e052).

## 📊 Current State Analysis

### Problems Identified

1. **No Real MCP Implementation**
   - Current `/neural-trader-rust/packages/mcp/index.js` is just a stub
   - TODOs reference non-existent Rust NAPI bindings
   - MCP server doesn't actually implement the protocol

2. **Missing Core Components**
   - ❌ No JSON-RPC 2.0 message handling
   - ❌ No stdio transport protocol
   - ❌ No tool discovery mechanism (`/tools/` directory)
   - ❌ No JSON Schema 1.1 definitions for 99 tools
   - ❌ No proper request/response handling
   - ❌ No authentication (OAuth 2.1/mTLS/keypair)
   - ❌ No audit logging (JSON Lines format)
   - ❌ No schema caching (ETag/SHA-256)

3. **Architecture Mismatch**
   - Node.js wrapper expects Rust implementation that doesn't exist
   - Python MCP servers exist but not integrated with Node.js CLI
   - Remote environment can't access Rust binaries

### Tool Catalog

**99 Unique Tools Found** across 7 Python MCP files:

#### Core Trading (23 tools)
- list_strategies, get_strategy_info, get_portfolio_status
- execute_trade, simulate_trade, quick_analysis
- backtest_strategy, optimize_strategy, optimize_parameters
- run_backtest, run_benchmark, quick_backtest
- risk_analysis, monte_carlo_simulation
- get_market_analysis, get_market_status
- performance_report, correlation_analysis
- recommend_strategy, switch_active_strategy
- adaptive_strategy_selection, get_strategy_comparison
- ping

#### Neural Networks (7 tools)
- neural_train, neural_forecast, neural_evaluate
- neural_backtest, neural_optimize, neural_model_status

#### News Trading (8 tools)
- analyze_news, get_news_sentiment, get_news_trends
- control_news_collection, get_news_provider_status
- fetch_filtered_news

#### Portfolio & Risk (5 tools)
- portfolio_rebalance, cross_asset_correlation_matrix
- execute_multi_asset_trade, get_execution_analytics
- get_system_metrics, monitor_strategy_health

#### Sports Betting (13 tools)
- get_sports_events, get_sports_odds, find_sports_arbitrage
- analyze_betting_market_depth, calculate_kelly_criterion
- simulate_betting_strategy, execute_sports_bet
- get_betting_portfolio_status, get_sports_betting_performance
- compare_betting_providers
- **The Odds API Integration (9 tools)**:
  - odds_api_get_sports, odds_api_get_live_odds
  - odds_api_get_event_odds, odds_api_find_arbitrage
  - odds_api_get_bookmaker_odds, odds_api_analyze_movement
  - odds_api_calculate_probability, odds_api_compare_margins
  - odds_api_get_upcoming

#### Prediction Markets (5 tools)
- get_prediction_markets_tool, get_prediction_positions_tool
- place_prediction_order_tool, analyze_market_sentiment_tool
- calculate_expected_value_tool, get_market_orderbook_tool

#### Syndicates (15 tools)
- create_syndicate_tool, add_syndicate_member
- get_syndicate_status_tool, allocate_syndicate_funds
- distribute_syndicate_profits, process_syndicate_withdrawal
- get_syndicate_member_performance, create_syndicate_vote
- cast_syndicate_vote, get_syndicate_allocation_limits
- update_syndicate_member_contribution, get_syndicate_profit_history
- simulate_syndicate_allocation, get_syndicate_withdrawal_history
- update_syndicate_allocation_strategy, get_syndicate_member_list
- calculate_syndicate_tax_liability

#### E2B Cloud Execution (9 tools)
- create_e2b_sandbox, run_e2b_agent, execute_e2b_process
- list_e2b_sandboxes, terminate_e2b_sandbox
- get_e2b_sandbox_status, deploy_e2b_template
- scale_e2b_deployment, monitor_e2b_health
- export_e2b_template

#### Fantasy Trading (5 tools)
- create_fantasy_league, join_league, make_prediction
- calculate_fantasy_scores, get_leaderboard, create_achievement

## 🚀 Implementation Plan

### Phase 1: Core MCP Protocol ⚡ **PRIORITY: CRITICAL**

#### 1.1 JSON-RPC 2.0 Implementation
- [ ] Request/Response message handlers
- [ ] Error code standardization
- [ ] Request ID tracking
- [ ] Batch request support

#### 1.2 STDIO Transport Layer
- [ ] stdin message reader (line-delimited JSON)
- [ ] stdout message writer
- [ ] stderr logging (separate from protocol)
- [ ] Message validation

#### 1.3 Tool Discovery System
- [ ] `/tools/` directory structure
- [ ] JSON Schema 1.1 definitions for all 99 tools
- [ ] Tool metadata (cost, latency, categories)
- [ ] Registry endpoint (`/tools/list`)

#### 1.4 Schema Generation
- [ ] Generate JSON Schema for each tool
- [ ] Input parameter schemas
- [ ] Output result schemas
- [ ] Example payloads

**Deliverables:**
- `packages/mcp/src/protocol/jsonrpc.js` - JSON-RPC handler
- `packages/mcp/src/transport/stdio.js` - stdio transport
- `packages/mcp/src/discovery/` - Tool discovery system
- `packages/mcp/tools/*.json` - 99 tool schema files

### Phase 2: Tool Integration 🔧 **PRIORITY: HIGH**

#### 2.1 Tool Handler Infrastructure
- [ ] Tool registration system
- [ ] Parameter validation (JSON Schema)
- [ ] Result serialization
- [ ] Error handling & normalization

#### 2.2 Python Bridge
- [ ] Child process spawning for Python tools
- [ ] IPC communication (stdio/JSON)
- [ ] Process pool management
- [ ] Error propagation

#### 2.3 Node.js Tool Implementations
- [ ] Core trading tools (23 tools)
- [ ] Neural network tools (7 tools)
- [ ] News trading tools (8 tools)
- [ ] Portfolio & risk tools (5 tools)
- [ ] Sports betting tools (13 tools)
- [ ] Prediction markets (5 tools)
- [ ] Syndicate tools (15 tools)
- [ ] E2B tools (9 tools)
- [ ] Fantasy tools (5 tools)

#### 2.4 Progress Tracking
- [ ] Long-running operation support
- [ ] Job handles & polling
- [ ] Progress callbacks
- [ ] Cancellation support

**Deliverables:**
- `packages/mcp/src/handlers/` - Tool handler implementations
- `packages/mcp/src/bridge/python.js` - Python bridge
- `packages/mcp/src/jobs/` - Async job management

### Phase 3: Security & Compliance 🔒 **PRIORITY: MEDIUM**

#### 3.1 Authentication
- [ ] Local keypair exchange (Ed25519)
- [ ] OAuth 2.1 Bearer token support
- [ ] Mutual TLS (mTLS) certificate pinning
- [ ] API key validation

#### 3.2 Audit Logging
- [ ] JSON Lines format logs
- [ ] Request/response logging
- [ ] Error logging
- [ ] Performance metrics

#### 3.3 Sandboxing
- [ ] Resource quotas (CPU/memory)
- [ ] Timeout enforcement
- [ ] Process isolation
- [ ] Rate limiting

**Deliverables:**
- `packages/mcp/src/auth/` - Authentication modules
- `packages/mcp/logs/` - Audit log directory
- `packages/mcp/src/sandbox/` - Sandboxing utilities

### Phase 4: Performance Optimization ⚡ **PRIORITY: MEDIUM**

#### 4.1 Schema Caching
- [ ] ETag generation (SHA-256)
- [ ] Client-side cache directives
- [ ] Invalidation on schema changes
- [ ] File reference system

#### 4.2 Token Optimization
- [ ] File references instead of inline schemas
- [ ] 98% token reduction (150k → 2k tokens)
- [ ] Compact tool descriptors
- [ ] Lazy schema loading

#### 4.3 Async Job Handling
- [ ] Job ID generation
- [ ] Status polling endpoints
- [ ] Result caching
- [ ] Job cleanup

**Deliverables:**
- `packages/mcp/src/cache/` - Caching system
- `packages/mcp/src/optimization/` - Token optimization

### Phase 5: Testing & Validation ✅ **PRIORITY: HIGH**

#### 5.1 Unit Tests
- [ ] Protocol compliance tests
- [ ] Tool validation tests
- [ ] Schema validation tests
- [ ] Error handling tests

#### 5.2 Integration Tests
- [ ] End-to-end tool invocation
- [ ] Claude Desktop integration
- [ ] Python bridge tests
- [ ] Authentication tests

#### 5.3 Performance Tests
- [ ] Latency benchmarks
- [ ] Token usage measurements
- [ ] Concurrent request handling
- [ ] Memory leak detection

**Deliverables:**
- `packages/mcp/tests/` - Test suite
- `packages/mcp/benchmarks/` - Performance tests

## 📋 MCP 2025-11 Requirements Checklist

| Requirement | Status | Priority | Target |
|------------|--------|----------|--------|
| **Protocol** |
| JSON-RPC 2.0 message handling | ❌ | CRITICAL | Phase 1 |
| Request ID tracking | ❌ | CRITICAL | Phase 1 |
| Error standardization | ❌ | CRITICAL | Phase 1 |
| **Transport** |
| STDIO transport | ❌ | CRITICAL | Phase 1 |
| HTTP+SSE transport | ⏸️ | LOW | Future |
| WebSocket transport | ⏸️ | LOW | Future |
| **Discovery** |
| File-based discovery (`/tools/`) | ❌ | CRITICAL | Phase 1 |
| JSON Schema 1.1 definitions | ❌ | CRITICAL | Phase 1 |
| Tool metadata (cost/latency) | ❌ | HIGH | Phase 2 |
| Registry endpoints | ❌ | MEDIUM | Phase 2 |
| **Security** |
| OAuth 2.1 Bearer tokens | ❌ | MEDIUM | Phase 3 |
| Mutual TLS (mTLS) | ❌ | LOW | Phase 3 |
| Local keypair exchange | ❌ | MEDIUM | Phase 3 |
| Audit logging (JSON Lines) | ❌ | MEDIUM | Phase 3 |
| **Execution** |
| Synchronous tool execution | ❌ | HIGH | Phase 2 |
| Asynchronous job handling | ❌ | MEDIUM | Phase 4 |
| Progress indicators | ❌ | MEDIUM | Phase 2 |
| Resource quotas | ❌ | MEDIUM | Phase 3 |
| **Optimization** |
| Schema caching (ETag) | ❌ | MEDIUM | Phase 4 |
| File references (98% reduction) | ❌ | HIGH | Phase 4 |
| SHA-256 digests | ❌ | MEDIUM | Phase 4 |

## 🎯 Success Criteria

### Functional Requirements
- ✅ All 99 tools accessible via MCP protocol
- ✅ STDIO transport working with Claude Desktop
- ✅ JSON Schema validation for all tools
- ✅ Python tool integration functional
- ✅ Error handling & logging operational

### Performance Requirements
- ✅ Tool invocation latency < 100ms (local)
- ✅ Schema loading < 50ms with caching
- ✅ Token usage reduction by 95%+
- ✅ Support 10+ concurrent requests

### Compliance Requirements
- ✅ MCP 2025-11 specification compliant
- ✅ JSON-RPC 2.0 compliant
- ✅ JSON Schema 1.1 compliant
- ✅ Audit logs in JSON Lines format

## 📦 Deliverables

### Package Structure
```
@neural-trader/mcp@2.0.0
├── bin/
│   └── mcp-server.js           # CLI entry point
├── src/
│   ├── protocol/
│   │   ├── jsonrpc.js          # JSON-RPC 2.0 handler
│   │   └── validator.js        # Schema validation
│   ├── transport/
│   │   ├── stdio.js            # STDIO transport
│   │   ├── http.js             # HTTP+SSE (future)
│   │   └── websocket.js        # WebSocket (future)
│   ├── discovery/
│   │   ├── registry.js         # Tool registry
│   │   └── metadata.js         # Tool metadata
│   ├── handlers/
│   │   ├── trading.js          # Trading tools
│   │   ├── neural.js           # Neural tools
│   │   ├── news.js             # News tools
│   │   ├── portfolio.js        # Portfolio tools
│   │   ├── sports.js           # Sports betting tools
│   │   ├── prediction.js       # Prediction markets
│   │   ├── syndicate.js        # Syndicate tools
│   │   ├── e2b.js              # E2B tools
│   │   └── fantasy.js          # Fantasy tools
│   ├── bridge/
│   │   └── python.js           # Python tool bridge
│   ├── auth/
│   │   ├── keypair.js          # Ed25519 keypair
│   │   ├── oauth.js            # OAuth 2.1
│   │   └── mtls.js             # Mutual TLS
│   ├── cache/
│   │   └── schema.js           # Schema caching
│   ├── jobs/
│   │   └── async.js            # Async job management
│   └── logging/
│       └── audit.js            # JSON Lines logging
├── tools/                      # Tool schema directory
│   ├── list_strategies.json
│   ├── execute_trade.json
│   └── ... (99 total)
├── logs/                       # Audit logs
│   └── mcp-audit.jsonl
└── tests/
    ├── protocol.test.js
    ├── tools.test.js
    └── integration.test.js
```

### Documentation
- [ ] MCP 2025-11 compliance guide
- [ ] Tool reference documentation (99 tools)
- [ ] Integration guide (Claude Desktop)
- [ ] Python bridge documentation
- [ ] Security & authentication guide
- [ ] Performance optimization guide

## 📅 Timeline

### Sprint 1 (Week 1): Core Protocol
- Day 1-2: JSON-RPC 2.0 implementation
- Day 3-4: STDIO transport layer
- Day 5-7: Tool discovery & schema generation

### Sprint 2 (Week 2): Tool Integration
- Day 1-3: Tool handler infrastructure
- Day 4-5: Python bridge implementation
- Day 6-7: Core trading tools (23 tools)

### Sprint 3 (Week 3): Remaining Tools
- Day 1-2: Neural & news tools (15 tools)
- Day 3-4: Portfolio & sports tools (18 tools)
- Day 5-7: Prediction, syndicate, E2B, fantasy tools (34 tools)

### Sprint 4 (Week 4): Security & Testing
- Day 1-2: Authentication & audit logging
- Day 3-4: Schema caching & optimization
- Day 5-7: Testing & validation

## 🔍 Migration Path

### For Existing Users
1. Upgrade to `@neural-trader/mcp@2.0.0`
2. Update Claude Desktop configuration
3. Regenerate authentication keys
4. Test tool access via MCP Inspector

### Breaking Changes
- `npx neural-trader mcp` now implements real protocol
- Requires Node.js 18+ (for native crypto APIs)
- Python 3.9+ required for tool bridge
- Environment variables for authentication

## 📚 References

- [MCP Developer Specification 2025-11](https://gist.github.com/ruvnet/284f199d0e0836c1b5185e30f819e052)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [JSON Schema 1.1](https://json-schema.org/draft/2020-12/release-notes.html)
- [FastMCP Python Library](https://github.com/jlowin/fastmcp)

## 💬 Discussion

- **Architecture Decision**: Use Python bridge vs. pure Node.js?
  - ✅ Hybrid approach: Python for complex ML/trading logic, Node.js for protocol
- **Authentication**: Which method to implement first?
  - ✅ Start with local keypair (simplest), add OAuth later
- **Performance**: Target latency goals?
  - ✅ < 100ms for simple tools, < 1s for ML tools

---

**Issue Created**: 2025-11-14
**Target Release**: `@neural-trader/mcp@2.0.0`
**Specification**: MCP 2025-11
**Status**: 🚧 Ready for Implementation
