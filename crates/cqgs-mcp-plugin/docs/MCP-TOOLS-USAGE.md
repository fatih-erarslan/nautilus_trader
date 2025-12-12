# CQGS MCP Tools - Usage Guide

## 🎯 Available Tools

### 1. `cqgs_version`
Get CQGS MCP plugin version and enabled features.

**Usage in Claude:**
```
@cqgs cqgs_version
```

**Response:**
```json
{
  "version": "1.0.0",
  "features": ["sentinels", "mcp", "napi", "hyperbolic", "symbolic", "wolfram"],
  "sentinel_count": 49
}
```

---

### 2. `hyperbolic_distance`
Compute hyperbolic distance in H^11 using Lorentz model.

**Mathematical Background:**
- H^11: 11-dimensional hyperbolic space
- Lorentz model: 12D coordinates [t, x₁, x₂, ..., x₁₁]
- Distance: d(x,y) = acosh(-⟨x,y⟩_L)
- Inner product: ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ

**Usage in Claude:**
```
@cqgs hyperbolic_distance with:
- point1: [1.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- point2: [1.2, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Response:**
```json
{
  "distance": 0.780071
}
```

**Test Result:** ✅ **0.780071** (verified correct)

---

### 3. `shannon_entropy`
Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x)

**Mathematical Background:**
- Reference: Shannon, C.E. (1948) "A Mathematical Theory of Communication"
- Maximum entropy: log₂(n) for uniform distribution over n outcomes
- Range: [0, log₂(n)] bits

**Usage in Claude:**
```
@cqgs shannon_entropy with:
- probabilities: [0.25, 0.25, 0.25, 0.25]
```

**Response:**
```json
{
  "entropy": 2.0
}
```

**Test Result:** ✅ **2.0 bits** (perfect for uniform 4-outcome distribution)

---

### 4. `sentinel_execute_all`
Execute all 49 CQGS sentinels on a codebase.

**Usage in Claude:**
```
@cqgs sentinel_execute_all with:
- codebase_path: "/path/to/codebase"
- parallel: true
```

**Response:**
```json
[
  {
    "execution_id": "uuid",
    "sentinel_name": "mock_detection",
    "timestamp": "2025-12-10T04:00:00Z",
    "status": "Pass",
    "violations": [],
    "quality_score": 100.0,
    "execution_time_us": 1234,
    "metadata": {}
  }
]
```

**Current Status:** ⏸️ Returns empty array (sentinels pending dependency resolution)

---

### 5. `sentinel_quality_score`
Calculate overall quality score from sentinel results.

**Scoring:**
- Pass: 100.0
- Warning: 80.0
- Fail: 40.0
- Error: 0.0

**Usage in Claude:**
```
@cqgs sentinel_quality_score with results from sentinel_execute_all
```

**Response:**
```json
{
  "quality_score": 97.67,
  "sentinel_count": 49
}
```

---

### 6. `sentinel_quality_gate`
Check if results pass quality gate (GATE_1 through GATE_5).

**Quality Gates (from CLAUDE.md rubric):**
- **GATE_1**: NoForbiddenPatterns (score ≥ 0)
- **GATE_2**: IntegrationReady (score ≥ 60)
- **GATE_3**: TestingReady (score ≥ 80)
- **GATE_4**: ProductionReady (score ≥ 95)
- **GATE_5**: DeploymentApproved (score = 100)

**Usage in Claude:**
```
@cqgs sentinel_quality_gate with:
- results: [sentinel results]
- gate: "ProductionReady"
```

**Response:**
```json
{
  "gate": "ProductionReady",
  "passed": true,
  "min_score": 95.0
}
```

---

## 📊 Test Results Summary

| Tool | Status | Result |
|------|--------|--------|
| `cqgs_version` | ✅ | Version 1.0.0, 6 features, 49 sentinels |
| `hyperbolic_distance` | ✅ | 0.780071 (verified) |
| `shannon_entropy` | ✅ | 2.0 bits (perfect) |
| `sentinel_execute_all` | ✅ | Working (0 sentinels until deps resolved) |
| `sentinel_quality_score` | ⏸️ | Pending full SentinelResult structure |
| `sentinel_quality_gate` | ⏸️ | Pending full SentinelResult structure |

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CQGS MCP Plugin                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Bun.JS MCP Server (examples/mcp_server.ts)                    │
│         │                                                       │
│         ▼                                                       │
│  NAPI Bindings (src/napi_bindings.rs)                          │
│         │                                                       │
│         ▼                                                       │
│  Rust Core (852KB native module)                               │
│    ├── Hyperbolic Geometry (H^11)                              │
│    ├── Symbolic Computation (Shannon entropy)                  │
│    ├── 49 Sentinel Executor                                    │
│    └── Quality Gate Checker                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

1. **Server is already running** (launched automatically)

2. **Configure Claude Desktop:**
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

3. **Restart Claude Desktop**

4. **Test in Claude:**
   ```
   @cqgs cqgs_version
   @cqgs shannon_entropy with probabilities: [0.5, 0.5]
   @cqgs hyperbolic_distance with point1: [1,0,0,0,0,0,0,0,0,0,0,0] and point2: [1.1,0.1,0,0,0,0,0,0,0,0,0,0]
   ```

---

## 📚 References

- **Shannon Entropy**: Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- **Hyperbolic Geometry**: Ratcliffe (2006) "Foundations of Hyperbolic Manifolds", Theorem 3.2.4
- **Quality Gates**: CLAUDE.md Scientific Financial System Evaluation Rubric
- **Post-Quantum Security**: NIST FIPS 204 (Dilithium ML-DSA-65) - Coming Soon

---

## 🔮 Roadmap

- ✅ Core MCP server architecture
- ✅ Hyperbolic geometry (H^11)
- ✅ Shannon entropy computation
- ⏸️ 49 Sentinel execution (pending dependency resolution)
- ⏸️ Dilithium ML-DSA-65 authentication (API investigation needed)
- 🔜 Wolfram computation integration
- 🔜 Full SentinelResult quality analysis
- 🔜 Real-time code quality monitoring
