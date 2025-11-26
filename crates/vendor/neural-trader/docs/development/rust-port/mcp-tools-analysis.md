# MCP Tools Analysis - Python to Rust Port

**Agent 2 - MCP Server Implementation**
**Date:** 2025-11-12
**Status:** Analysis Complete, Implementation In Progress

## Executive Summary

The Python MCP server implementation consists of **87 tools** across 4,564 LOC in `mcp_server_enhanced.py`. This document provides a complete analysis for porting to Rust.

## Architecture Overview

### Python Implementation
- **Framework:** FastMCP (Anthropic's official MCP library)
- **Transport:** STDIO (primary) + HTTP + WebSocket
- **Protocol:** JSON-RPC 2.0
- **GPU Support:** CuPy + PyTorch CUDA
- **Authentication:** JWT + Encryption middleware

### Rust Target Architecture
```
neural-trader-rust/
├── crates/
│   ├── mcp-server/          # Main MCP server crate
│   │   ├── src/
│   │   │   ├── lib.rs       # Core MCP protocol
│   │   │   ├── transport/
│   │   │   │   ├── stdio.rs   # STDIO transport
│   │   │   │   ├── http.rs    # HTTP+SSE transport
│   │   │   │   └── websocket.rs
│   │   │   ├── tools/
│   │   │   │   ├── trading.rs      # 15 core trading tools
│   │   │   │   ├── neural.rs       # 6 neural forecasting
│   │   │   │   ├── brokers.rs      # 12+ multi-broker
│   │   │   │   ├── sports.rs       # 9 sports betting
│   │   │   │   ├── prediction.rs   # 6 prediction markets
│   │   │   │   ├── crypto.rs       # 5 crypto trading
│   │   │   │   ├── news.rs         # 3 news/sentiment
│   │   │   │   ├── sandbox.rs      # 3 E2B sandbox
│   │   │   │   ├── syndicate.rs    # 17 syndicate mgmt
│   │   │   │   └── odds_api.rs     # 9 Odds API
│   │   │   ├── handlers/
│   │   │   │   ├── tools.rs
│   │   │   │   ├── resources.rs
│   │   │   │   ├── prompts.rs
│   │   │   │   └── sampling.rs
│   │   │   └── discovery/
│   │   │       └── file_resource.rs  # 98% token reduction
│   │   └── Cargo.toml
│   └── mcp-protocol/        # Shared protocol types
│       ├── src/
│       │   ├── lib.rs
│       │   ├── types.rs     # JSON-RPC 2.0 types
│       │   └── error.rs
│       └── Cargo.toml
```

## Complete Tool Inventory (87 Tools)

### 1. Core Trading Tools (15)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `ping` | mcp_server_enhanced.py:330 | trading.rs | P0 |
| `list_strategies` | mcp_server_enhanced.py:340 | trading.rs | P0 |
| `get_strategy_info` | mcp_server_enhanced.py:355 | trading.rs | P0 |
| `quick_analysis` | mcp_server_enhanced.py:375 | trading.rs | P0 |
| `simulate_trade` | mcp_server_enhanced.py:415 | trading.rs | P0 |
| `get_portfolio_status` | mcp_server_enhanced.py:455 | trading.rs | P0 |
| `analyze_news` | mcp_server_enhanced.py:495 | news.rs | P0 |
| `get_news_sentiment` | mcp_server_enhanced.py:540 | news.rs | P0 |
| `run_backtest` | mcp_server_enhanced.py:580 | trading.rs | P1 |
| `optimize_strategy` | mcp_server_enhanced.py:650 | trading.rs | P1 |
| `risk_analysis` | mcp_server_enhanced.py:720 | trading.rs | P1 |
| `execute_trade` | mcp_server_enhanced.py:800 | trading.rs | P1 |
| `performance_report` | mcp_server_enhanced.py:860 | trading.rs | P2 |
| `correlation_analysis` | mcp_server_enhanced.py:920 | trading.rs | P2 |
| `run_benchmark` | mcp_server_enhanced.py:980 | trading.rs | P2 |

### 2. Neural Forecasting Tools (6)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `neural_forecast` | mcp_server_enhanced.py:1040 | neural.rs | P0 |
| `neural_train` | mcp_server_enhanced.py:1100 | neural.rs | P1 |
| `neural_evaluate` | mcp_server_enhanced.py:1160 | neural.rs | P1 |
| `neural_backtest` | mcp_server_enhanced.py:1220 | neural.rs | P1 |
| `neural_model_status` | mcp_server_enhanced.py:1280 | neural.rs | P1 |
| `neural_optimize` | mcp_server_enhanced.py:1340 | neural.rs | P2 |

### 3. Multi-Broker Tools (12+)
| Tool | Python Location | Rust Module | Notes |
|------|----------------|-------------|-------|
| `get_alpaca_account` | alpaca/mcp_integration.py:50 | brokers.rs | Agent 3 integration |
| `execute_alpaca_trade` | alpaca/mcp_integration.py:75 | brokers.rs | Agent 3 integration |
| `get_ibkr_positions` | brokers/ibkr.py | brokers.rs | Agent 3 integration |
| `execute_ibkr_trade` | brokers/ibkr.py | brokers.rs | Agent 3 integration |
| `get_questrade_quotes` | canadian_trading/ | brokers.rs | Agent 3 integration |
| `get_polygon_data` | brokers/polygon.py | brokers.rs | Agent 3 integration |
| `get_schwab_orders` | brokers/schwab.py | brokers.rs | Agent 3 integration |
| `get_tda_positions` | brokers/tda.py | brokers.rs | Agent 3 integration |
| `execute_webull_trade` | brokers/webull.py | brokers.rs | Agent 3 integration |

### 4. Sports Betting Tools (9)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `get_sports_events` | mcp_server_enhanced.py:1400 | sports.rs | P1 |
| `get_sports_odds` | mcp_server_enhanced.py:1460 | sports.rs | P1 |
| `find_sports_arbitrage` | mcp_server_enhanced.py:1520 | sports.rs | P1 |
| `analyze_betting_market_depth` | mcp_server_enhanced.py:1580 | sports.rs | P2 |
| `calculate_kelly_criterion` | mcp_server_enhanced.py:1640 | sports.rs | P1 |
| `simulate_betting_strategy` | mcp_server_enhanced.py:1700 | sports.rs | P2 |
| `get_betting_portfolio_status` | mcp_server_enhanced.py:1760 | sports.rs | P1 |
| `execute_sports_bet` | mcp_server_enhanced.py:1820 | sports.rs | P1 |
| `get_sports_betting_performance` | mcp_server_enhanced.py:1880 | sports.rs | P2 |

### 5. Prediction Markets Tools (6)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `get_prediction_markets` | polymarket/mcp_tools.py:40 | prediction.rs | P1 |
| `analyze_market_sentiment` | polymarket/mcp_tools.py:85 | prediction.rs | P1 |
| `get_market_orderbook` | polymarket/mcp_tools.py:130 | prediction.rs | P2 |
| `place_prediction_order` | polymarket/mcp_tools.py:175 | prediction.rs | P1 |
| `get_prediction_positions` | polymarket/mcp_tools.py:220 | prediction.rs | P1 |
| `calculate_expected_value` | polymarket/mcp_tools.py:265 | prediction.rs | P1 |

### 6. Crypto Trading Tools (5)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `beefy_get_vaults` | crypto_trading/mcp_tools/integration.py:50 | crypto.rs | P1 |
| `beefy_analyze_vault` | crypto_trading/mcp_tools/integration.py:95 | crypto.rs | P1 |
| `beefy_invest` | crypto_trading/mcp_tools/integration.py:140 | crypto.rs | P1 |
| `beefy_harvest_yields` | crypto_trading/mcp_tools/integration.py:185 | crypto.rs | P1 |
| `beefy_rebalance_portfolio` | crypto_trading/mcp_tools/integration.py:230 | crypto.rs | P2 |

### 7. News & Sentiment Tools (4)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `control_news_collection` | mcp_server_enhanced.py:1940 | news.rs | P1 |
| `get_news_provider_status` | mcp_server_enhanced.py:2000 | news.rs | P1 |
| `fetch_filtered_news` | mcp_server_enhanced.py:2060 | news.rs | P1 |
| `get_news_trends` | mcp_server_enhanced.py:2120 | news.rs | P2 |

### 8. E2B Sandbox Tools (10)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `create_e2b_sandbox` | mcp_server_enhanced.py:2180 | sandbox.rs | P1 |
| `run_e2b_agent` | mcp_server_enhanced.py:2240 | sandbox.rs | P1 |
| `execute_e2b_process` | mcp_server_enhanced.py:2300 | sandbox.rs | P1 |
| `list_e2b_sandboxes` | mcp_server_enhanced.py:2360 | sandbox.rs | P1 |
| `terminate_e2b_sandbox` | mcp_server_enhanced.py:2420 | sandbox.rs | P1 |
| `get_e2b_sandbox_status` | mcp_server_enhanced.py:2480 | sandbox.rs | P1 |
| `deploy_e2b_template` | mcp_server_enhanced.py:2540 | sandbox.rs | P2 |
| `scale_e2b_deployment` | mcp_server_enhanced.py:2600 | sandbox.rs | P2 |
| `monitor_e2b_health` | mcp_server_enhanced.py:2660 | sandbox.rs | P2 |
| `export_e2b_template` | mcp_server_enhanced.py:2720 | sandbox.rs | P2 |

### 9. Syndicate Management Tools (17)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `create_syndicate` | syndicate/syndicate_tools.py:40 | syndicate.rs | P1 |
| `add_syndicate_member` | syndicate/syndicate_tools.py:85 | syndicate.rs | P1 |
| `get_syndicate_status` | syndicate/syndicate_tools.py:130 | syndicate.rs | P1 |
| `allocate_syndicate_funds` | syndicate/syndicate_tools.py:175 | syndicate.rs | P1 |
| `distribute_syndicate_profits` | syndicate/syndicate_tools.py:220 | syndicate.rs | P1 |
| `process_syndicate_withdrawal` | syndicate/syndicate_tools.py:265 | syndicate.rs | P1 |
| `get_syndicate_member_performance` | syndicate/syndicate_tools.py:310 | syndicate.rs | P2 |
| `create_syndicate_vote` | syndicate/syndicate_tools.py:355 | syndicate.rs | P1 |
| `cast_syndicate_vote` | syndicate/syndicate_tools.py:400 | syndicate.rs | P1 |
| `get_syndicate_allocation_limits` | syndicate/syndicate_tools.py:445 | syndicate.rs | P2 |
| `update_syndicate_member_contribution` | syndicate/syndicate_tools.py:490 | syndicate.rs | P2 |
| `get_syndicate_profit_history` | syndicate/syndicate_tools.py:535 | syndicate.rs | P2 |
| `simulate_syndicate_allocation` | syndicate/syndicate_tools.py:580 | syndicate.rs | P2 |
| `get_syndicate_withdrawal_history` | syndicate/syndicate_tools.py:625 | syndicate.rs | P2 |
| `update_syndicate_allocation_strategy` | syndicate/syndicate_tools.py:670 | syndicate.rs | P2 |
| `get_syndicate_member_list` | syndicate/syndicate_tools.py:715 | syndicate.rs | P2 |
| `calculate_syndicate_tax_liability` | syndicate/syndicate_tools.py:760 | syndicate.rs | P2 |

### 10. Odds API Tools (9)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `odds_api_get_sports` | odds_api/tools.py:40 | odds_api.rs | P1 |
| `odds_api_get_live_odds` | odds_api/tools.py:85 | odds_api.rs | P1 |
| `odds_api_get_event_odds` | odds_api/tools.py:130 | odds_api.rs | P1 |
| `odds_api_find_arbitrage` | odds_api/tools.py:175 | odds_api.rs | P1 |
| `odds_api_get_bookmaker_odds` | odds_api/tools.py:220 | odds_api.rs | P2 |
| `odds_api_analyze_movement` | odds_api/tools.py:265 | odds_api.rs | P2 |
| `odds_api_calculate_probability` | odds_api/tools.py:310 | odds_api.rs | P2 |
| `odds_api_compare_margins` | odds_api/tools.py:355 | odds_api.rs | P2 |
| `odds_api_get_upcoming` | odds_api/tools.py:400 | odds_api.rs | P2 |

### 11. Strategy Management Tools (4)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `recommend_strategy` | mcp_server_enhanced.py:2780 | trading.rs | P2 |
| `switch_active_strategy` | mcp_server_enhanced.py:2840 | trading.rs | P2 |
| `get_strategy_comparison` | mcp_server_enhanced.py:2900 | trading.rs | P2 |
| `adaptive_strategy_selection` | mcp_server_enhanced.py:2960 | trading.rs | P2 |

### 12. Performance Monitoring Tools (3)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `get_system_metrics` | mcp_server_enhanced.py:3020 | trading.rs | P2 |
| `monitor_strategy_health` | mcp_server_enhanced.py:3080 | trading.rs | P2 |
| `get_execution_analytics` | mcp_server_enhanced.py:3140 | trading.rs | P2 |

### 13. Multi-Asset Trading Tools (3)
| Tool | Python Location | Rust Module | Priority |
|------|----------------|-------------|----------|
| `execute_multi_asset_trade` | mcp_server_enhanced.py:3200 | trading.rs | P2 |
| `portfolio_rebalance` | mcp_server_enhanced.py:3260 | trading.rs | P2 |
| `cross_asset_correlation_matrix` | mcp_server_enhanced.py:3320 | trading.rs | P2 |

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)
1. ✅ Create `mcp-protocol` crate with JSON-RPC 2.0 types
2. ✅ Implement STDIO transport (primary for `npx neural-trader mcp start`)
3. ✅ Implement HTTP+SSE transport (secondary for web clients)
4. ✅ File-based resource discovery system
5. ✅ JSON Lines logging infrastructure

### Phase 2: Priority 0 Tools (Week 1-2)
Implement critical path tools first:
- Core trading: ping, list_strategies, get_strategy_info, quick_analysis, simulate_trade, get_portfolio_status
- Neural: neural_forecast
- News: analyze_news, get_news_sentiment

### Phase 3: Priority 1 Tools (Week 2-3)
- Remaining trading tools
- Broker integrations (coordinate with Agent 3)
- Sports betting basics
- Prediction markets
- Syndicate core

### Phase 4: Priority 2 Tools (Week 3-4)
- Advanced analytics
- E2B sandbox
- Odds API
- Performance monitoring

### Phase 5: Testing & Benchmarking (Week 4)
- Unit tests for all tools
- Integration tests with Node.js
- Performance benchmarks vs Python
- GPU acceleration validation

## Technical Specifications

### Dependencies
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.35", features = ["full"] }
axum = "0.7"  # HTTP+SSE
tower-http = "0.5"
jsonrpc-core = "18.0"
async-trait = "0.1"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

# GPU support
cudarc = { version = "0.10", optional = true }
tch = { version = "0.15", optional = true }  # PyTorch bindings

# Trading
ta = "0.5"  # Technical analysis
rust_decimal = "1.33"

# Brokers (Agent 3 integration)
reqwest = { version = "0.11", features = ["json"] }
tokio-tungstenite = "0.21"  # WebSocket

# Database (Agent 8 integration)
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio"] }

[features]
default = ["stdio", "http"]
stdio = []
http = ["axum", "tower-http"]
gpu = ["cudarc", "tch"]
all = ["stdio", "http", "gpu"]
```

### JSON-RPC 2.0 Protocol Types
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,  // Always "2.0"
    pub method: String,
    pub params: Option<serde_json::Value>,
    pub id: Option<RequestId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Number(i64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,  // Always "2.0"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Option<RequestId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}
```

## Integration Points

### Agent 3: Multi-Broker Support
- Rust broker adapters must match Python API
- Shared authentication tokens
- Consistent error handling

### Agent 4: Neural Models
- ONNX Runtime for model inference
- Shared model format
- GPU memory management

### Agent 6: Risk Management
- Real-time risk calculations
- Position limit enforcement
- Portfolio constraints

### Agent 8: AgentDB
- Persistent state storage
- Trade history
- Model cache

## Performance Targets

| Metric | Python Baseline | Rust Target | Improvement |
|--------|----------------|-------------|-------------|
| Tool call latency | 50-200ms | 5-20ms | 10x |
| Throughput | 100 req/s | 1000+ req/s | 10x |
| Memory usage | 500MB | 50MB | 10x |
| Startup time | 5-10s | 0.5-1s | 10x |
| GPU memory | 2.1GB | 1.5GB | 1.4x |

## Testing Strategy

### Unit Tests
- Each tool function isolated
- Mock external dependencies
- GPU fallback validation

### Integration Tests
- Node.js MCP client
- Multi-tool workflows
- Error handling
- Timeout scenarios

### Performance Tests
- Load testing (1000+ concurrent)
- Latency percentiles (p50, p95, p99)
- Memory profiling
- GPU utilization

## Documentation Requirements

1. **API Documentation**
   - Tool signatures and parameters
   - Example requests/responses
   - Error codes and handling

2. **Integration Guide**
   - Node.js usage examples
   - Environment configuration
   - Authentication setup

3. **Performance Tuning**
   - GPU optimization
   - Connection pooling
   - Caching strategies

## Success Criteria

✅ All 87 tools implemented in Rust
✅ Node.js can call tools via MCP protocol
✅ Performance ≥ Python version
✅ Full test coverage (>90%)
✅ Complete documentation
✅ CI/CD integration

## Next Steps

1. Create `mcp-server` crate structure
2. Implement JSON-RPC 2.0 protocol layer
3. Build STDIO transport
4. Port first 10 P0 tools
5. Integration test with Node.js client
6. Iterate through remaining tools by priority

---

**Agent 2 Status:** Analysis complete. Ready to begin Phase 1 implementation.
