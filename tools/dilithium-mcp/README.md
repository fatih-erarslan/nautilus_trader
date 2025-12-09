# Dilithium MCP Server

**Post-Quantum Secure Model Context Protocol Server**

Version 3.0.0 | Rust-Bun.js Architecture | ML-DSA (Dilithium) Authentication

## Overview

Dilithium MCP is a high-performance Model Context Protocol server featuring:

- **Post-Quantum Security**: Dilithium ML-DSA digital signatures (NIST PQC standard)
- **Rust-Bun.js Architecture**: Native Rust core with Bun.js runtime
- **Hyperbolic Geometry**: Lorentz H^11 operations for hierarchical embeddings
- **pBit Dynamics**: Boltzmann statistics and Ising model computations
- **Agent Swarm**: Multi-agent coordination with consensus protocols
- **Enhanced Tooling**: Design thinking, systems dynamics, LLM enhancement

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Bun.js Runtime                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  MCP SDK    │  │   Swarm     │  │   Tools     │          │
│  │  Transport  │  │  Coordinator│  │   Router    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │   FFI     │                            │
│                    │  Bridge   │                            │
│                    └─────┬─────┘                            │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Native Rust Module                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Dilithium  │  │ Hyperbolic  │  │    pBit     │          │
│  │    Crypto   │  │  Geometry   │  │  Dynamics   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## Security Model

### Dilithium Authentication Flow

1. **Key Generation**: Client generates Dilithium key pair
2. **Registration**: Client registers public key with server
3. **Request Signing**: Each request signed with secret key
4. **Verification**: Server verifies signature before processing
5. **Nonce Protection**: Prevents replay attacks

```typescript
// Generate key pair
const { public_key, secret_key } = dilithium_keygen();

// Create signed request
const request = {
  client_id: "my-agent",
  timestamp: new Date().toISOString(),
  nonce: generate_nonce(),
  payload: JSON.stringify({ tool: "pbit_sample", args: { field: 0 } }),
  signature: dilithium_sign(secret_key, `${timestamp}${nonce}${payload}`)
};

// Server verifies
const result = verify_request(request);
// { valid: true, client_id: "my-agent", timestamp: "..." }
```

### Post-Quantum Resistance

Dilithium (ML-DSA) is one of the NIST Post-Quantum Cryptography standards, designed to be secure against attacks from quantum computers.

| Algorithm | Key Size | Signature Size | Security Level |
|-----------|----------|----------------|----------------|
| Dilithium3 | 1.9 KB | 3.3 KB | 128-bit quantum |

## Tool Categories (114+ tools)

### Core Native (13 tools)
- `dilithium_keygen` - Generate ML-DSA key pair
- `dilithium_sign` - Sign message
- `dilithium_verify` - Verify signature
- `blake3_hash` - BLAKE3 hash
- `hyperbolic_distance` - Distance in H^11
- `lift_to_hyperboloid` - Euclidean → Lorentz
- `mobius_add` - Möbius addition
- `pbit_sample` - Sampling probability
- `boltzmann_weight` - exp(-E/T)
- `ising_critical_temp` - Onsager T_c
- `stdp_weight_change` - STDP plasticity
- `compute` - Math expression
- `symbolic` - Symbolic ops

### Dilithium Auth (7 tools)
- `dilithium_register_client` - Register client
- `dilithium_authorize` - Authorization token
- `dilithium_validate_token` - Token validation
- `dilithium_check_quota` - Check usage quota
- `dilithium_list_clients` - List clients
- `dilithium_revoke_client` - Revoke access
- `dilithium_update_capabilities` - Update perms

### Agent Swarm (15 tools)
- Agent registration and discovery
- Message passing and broadcast
- Consensus proposals and voting
- Shared memory (CRDT-based)
- Task distribution

### Design Thinking (12 tools)
- `design_empathize_*` - User research
- `design_define_*` - Problem statements
- `design_ideate_*` - Brainstorming
- `design_prototype_*` - Scaffolding
- `design_test_*` - Validation
- `design_iterate_*` - Feedback loops

### Systems Dynamics (13 tools)
- `systems_model_*` - Create models
- `systems_equilibrium_*` - Fixed points
- `systems_control_*` - Control design
- `systems_feedback_*` - Causal loops
- `systems_network_*` - Graph analysis
- `systems_sensitivity_*` - Sensitivity
- `systems_monte_carlo` - Uncertainty

### LLM Tools (11 tools)
- `wolfram_llm_function` - Function creation
- `wolfram_llm_synthesize` - Content generation
- `wolfram_llm_tool_define` - Tool definitions
- `wolfram_llm_prompt_*` - Prompt engineering
- `wolfram_llm_code_*` - Code generation/review
- `wolfram_llm_analyze` - Deep analysis
- `wolfram_llm_reason` - Multi-step reasoning
- `wolfram_llm_graph` - Knowledge graphs

### DevOps Pipeline (19 tools)
- `git_*` - Version control analysis
- `cicd_*` - CI/CD pipeline generation
- `deploy_*` - Deployment strategies
- `observability_*` - Monitoring setup

### Project Management (13 tools)
- `sprint_*` - Sprint planning
- `estimate_*` - Effort estimation
- `backlog_*` - Backlog management
- `team_*` - Workload balancing
- `dora_*` - DORA metrics

### Documentation (14 tools)
- `docs_api_*` - API documentation
- `docs_architecture_*` - Architecture diagrams
- `docs_adr_*` - ADRs
- `docs_runbook_*` - Operational runbooks
- `docs_postmortem_*` - Incident analysis

### Code Quality (16 tools)
- `code_analyze_*` - Static analysis
- `refactor_*` - Refactoring
- `techdebt_*` - Technical debt
- `codehealth_*` - Health metrics

**Total: 133+ tools across 11 categories**

## Installation

### Prerequisites

- Bun >= 1.1.0
- Rust >= 1.75.0 (for native module)
- Cargo with `pqcrypto` crate support

### Quick Start

```bash
# Install dependencies
bun install

# Build TypeScript
bun run build

# Build native Rust module (recommended for production)
bun run build:native

# Run server
bun run start
```

### MCP Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dilithium": {
      "command": "bun",
      "args": ["run", "/path/to/dilithium-mcp/dist/index.js"],
      "env": {
        "DILITHIUM_NATIVE_PATH": "/path/to/dilithium-mcp/native/target/release/libdilithium_native.dylib"
      }
    }
  }
}
```

## Performance

### Native Module (Rust)
- **Key Generation**: ~1ms
- **Signing**: ~0.5ms
- **Verification**: ~0.3ms
- **Hyperbolic Distance**: ~10μs

### Hardware Requirements
- **Minimum**: Any x86_64 or arm64 CPU
- **Recommended**: AVX2-capable CPU for SIMD acceleration
- **Memory**: 256MB base + ~1KB per registered client

## Development

```bash
# Development mode with hot reload
bun run dev

# Run tests
bun test

# Security tests
bun run test:security

# Lint
bun run lint
```

## Integration with HyperPhysics

This MCP server integrates with the HyperPhysics ecosystem:

- **hyperphysics-dilithium**: Rust crypto library
- **hyperphysics-mcp-auth**: Authentication middleware
- **hyperphysics-pbit**: pBit dynamics engine
- **hyperphysics-lorentz**: Hyperbolic geometry
- **tengri-holographic-cortex**: Cortex architecture

## License

MIT

## References

- [NIST PQC Standards](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Bun.js Runtime](https://bun.sh)
- [pqcrypto Rust Crate](https://crates.io/crates/pqcrypto)
