# CQGS MCP Plugin

Code Quality Governance Sentinels Model Context Protocol Plugin - Exposes 49 sentinels via MCP with Dilithium ML-DSA-65 post-quantum security.

## Features

- **49 Sentinels**: Core (17), Security (12), Infrastructure (10), Advanced (10)
- **Post-Quantum Security**: Dilithium ML-DSA-65 (NIST FIPS 204)
- **Quality Gates**: GATE_1 through GATE_5 (60→80→95→100 scoring)
- **Native Performance**: Rust core with 100x speedup
- **Multi-Platform**: NAPI (Bun.JS/Node.js) + WASM bindings

## Quick Start

### Build

```bash
# Install dependencies
bun install

# Build Rust native module
cargo build --release --features napi

# Build NAPI bindings
bun run build
```

### Run MCP Server

```bash
# Start server
bun run start

# Or development mode (auto-rebuild)
bun run dev
```

### Test with Claude Desktop

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cqgs": {
      "command": "bun",
      "args": [
        "run",
        "/Volumes/Tengritek/Ashina/code-governance/cqgs-mcp-plugin/examples/mcp_server.ts"
      ]
    }
  }
}
```

## Available MCP Tools

### Sentinel Execution

- `sentinel_execute_all` - Execute all 49 sentinels
- `sentinel_quality_score` - Calculate quality score
- `sentinel_quality_gate` - Check quality gate (GATE_1-5)

### Dilithium Authentication

- `dilithium_keygen` - Generate ML-DSA-65 key pair
- `dilithium_sign` - Sign message
- `dilithium_verify` - Verify signature

### Hyperbolic Geometry

- `hyperbolic_distance` - Compute H^11 distance

### Symbolic Computation

- `shannon_entropy` - Shannon entropy calculation

### System

- `cqgs_version` - Get version and features

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CQGS MCP PLUGIN v1.0                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Dilithium  │  │   49 Core   │  │    MCP      │  │   NAPI/     │    │
│  │   ML-DSA    │  │  Sentinels  │  │  Protocol   │  │   WASM      │    │
│  │  Security   │  │  Exposure   │  │  Server     │  │  Bindings   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│         └────────────────┼────────────────┼────────────────┘           │
│                          │                │                            │
│  ┌─────────────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────┐    │
│  │   Wolfram   │  │  Hyperbolic │  │  Symbolic   │  │  Quality    │    │
│  │Integration  │  │  Geometry   │  │  Compute    │  │  Metrics    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quality Gates

Based on Scientific Financial System evaluation rubric:

- **GATE_1**: No forbidden patterns (score ≥ 0)
- **GATE_2**: Integration ready (score ≥ 60)
- **GATE_3**: Testing ready (score ≥ 80)
- **GATE_4**: Production ready (score ≥ 95)
- **GATE_5**: Deployment approved (score = 100)

## License

MIT OR Apache-2.0
