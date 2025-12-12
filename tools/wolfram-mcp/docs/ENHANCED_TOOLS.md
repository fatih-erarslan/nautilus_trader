# Wolfram MCP Enhanced Tools

Complete reference for all enhanced capabilities in Wolfram MCP v2.0.

## Tool Categories Overview

| Category | Tools | Description |
|----------|-------|-------------|
| **Core Wolfram** | 9 | Basic computation, LLM query, unit conversion |
| **Agent Swarm** | 15 | Multi-agent communication |
| **Design Thinking** | 12 | Cyclical development methodology |
| **Systems Dynamics** | 13 | Modeling, equilibrium, control theory |
| **LLM Tools** | 11 | Advanced LLM capabilities |
| **Dilithium Auth** | 7 | Post-quantum secure authorization |
| **Total** | **67** | Complete toolkit |

---

## 1. Core Wolfram Tools

### Query & Compute
| Tool | Description |
|------|-------------|
| `wolfram_llm_query` | Natural language query to Wolfram Alpha |
| `wolfram_compute` | Mathematical computation |
| `wolfram_validate` | Validate expressions/identities |
| `wolfram_unit_convert` | Unit conversion |
| `wolfram_data_query` | Scientific/geographic data lookup |
| `wolfram_full_query` | Full Results API with structured pods |
| `wolfram_local_eval` | Execute Wolfram Language locally |
| `wolfram_symbolic` | Symbolic math: integrate, differentiate, solve |
| `wolfram_hyperbolic` | Hyperbolic geometry computations |

---

## 2. Design Thinking Tools

### Empathize Phase
```
design_empathize_analyze
  - Analyze user research with NLP
  - Extract themes, sentiment, entities
  
design_empathize_persona  
  - Generate personas from user data
  - Clustering-based segmentation
```

### Define Phase
```
design_define_problem
  - Generate "How Might We" statements
  - Structured problem framing
  
design_define_requirements
  - Extract and prioritize requirements
  - Dependency graph analysis
```

### Ideate Phase
```
design_ideate_brainstorm
  - LLM-powered divergent thinking
  - Analogical reasoning across domains
  
design_ideate_evaluate
  - Multi-criteria decision analysis
  - Weighted scoring matrix
```

### Prototype Phase
```
design_prototype_architecture
  - Generate system architecture
  - Graph-based component modeling
  
design_prototype_code
  - Code scaffolding generation
  - Multi-language support
```

### Test Phase
```
design_test_generate
  - Property-based test generation
  - Boundary analysis
  
design_test_analyze
  - Test result analysis
  - Failure pattern detection
```

### Iterate Phase
```
design_iterate_feedback
  - Feedback sentiment/theme analysis
  - Guide next iteration
  
design_iterate_metrics
  - Track metrics across iterations
  - Progress visualization
```

---

## 3. Systems Dynamics Tools

### System Modeling
```
systems_model_create
  - Create stock-flow models
  - Define feedback loops
  
systems_model_simulate
  - Simulate system trajectories
  - NDSolve for differential equations
```

### Equilibrium Analysis
```
systems_equilibrium_find
  - Find fixed points/steady states
  - Algebraic solution of dynamics

systems_equilibrium_stability
  - Eigenvalue stability analysis
  - Classify equilibrium type
  
systems_equilibrium_bifurcation
  - Bifurcation diagrams
  - Parameter sensitivity
```

### Control Theory
```
systems_control_design
  - PID, LQR, MPC controller design
  - Specification-based tuning
  
systems_control_analyze
  - Controllability/observability
  - Pole placement analysis
```

### Feedback Analysis
```
systems_feedback_causal_loop
  - Causal loop diagram analysis
  - Identify reinforcing/balancing loops
  
systems_feedback_loop_gain
  - Loop gain calculation
  - Phase margin analysis
```

### Network Analysis
```
systems_network_analyze
  - Centrality, clustering, communities
  - Graph metrics
  
systems_network_optimize
  - Max flow, min cost, shortest path
  - Network optimization
```

### Sensitivity & Uncertainty
```
systems_sensitivity_analyze
  - Parameter sensitivity gradients
  - Elasticity analysis
  
systems_monte_carlo
  - Monte Carlo simulation
  - Uncertainty quantification
```

---

## 4. LLM Tools

### Function & Synthesis
```
wolfram_llm_function
  - Create reusable LLM functions
  - Template with placeholders
  
wolfram_llm_synthesize
  - Generate content (text, code, JSON)
  - Configurable model and tokens
  
wolfram_llm_tool_define
  - Define tools for LLM agents
  - Function calling interface
```

### Prompt Engineering
```
wolfram_llm_prompt
  - Structured prompt creation
  - Role, task, examples, constraints
  
wolfram_llm_prompt_chain
  - Multi-step prompt chains
  - Dependency-aware execution
```

### Code Generation
```
wolfram_llm_code_generate
  - Generate code with Wolfram verification
  - Multi-language: Rust, Python, Swift, TypeScript
  
wolfram_llm_code_review
  - AI code review with static analysis
  - Bug, security, style checks
  
wolfram_llm_code_explain
  - Natural language code explanation
  - Brief, detailed, or tutorial level
```

### Analysis & Reasoning
```
wolfram_llm_analyze
  - SWOT, root cause, comparative analysis
  - Deep analysis with knowledge base
  
wolfram_llm_reason
  - Chain-of-thought reasoning
  - Tree-of-thought, self-consistency
  
wolfram_llm_graph
  - Knowledge graph extraction
  - Entity and relation extraction
```

---

## 5. Dilithium Authorization

### Client Management
```
dilithium_register_client
  - Register new Sentry client
  - Assign capabilities and quotas
  
dilithium_list_clients
  - List all registered clients
  - Show status and last activity
  
dilithium_revoke_client
  - Revoke client access
  - Audit logged
  
dilithium_update_capabilities
  - Modify client permissions
```

### Authorization Flow
```
dilithium_authorize
  - Sign request with Dilithium key
  - Receive authorization token
  
dilithium_validate_token
  - Check token validity
  - Verify expiration and signature
  
dilithium_check_quota
  - Check remaining API quota
  - Daily requests and tokens
```

### Capabilities
| Capability | Access |
|------------|--------|
| `llm_query` | Basic LLM queries |
| `llm_synthesize` | Content generation |
| `compute` | Mathematical computation |
| `data_query` | Data lookups |
| `systems_model` | System modeling |
| `equilibrium` | Equilibrium analysis |
| `design_thinking` | Design tools |
| `swarm` | Agent swarm |
| `full_access` | All capabilities |

---

## 6. Agent Swarm Tools

See [AGENT_SWARM.md](./AGENT_SWARM.md) for complete documentation.

---

## Usage Examples

### Design Thinking Workflow
```javascript
// 1. Empathize - Analyze user research
design_empathize_analyze({
  userResearch: "Users find the signup process confusing...",
  stakeholders: ["end_users", "admins"]
})

// 2. Define - Frame the problem
design_define_problem({
  insights: ["Signup takes too long", "Too many fields"],
  goals: ["Reduce friction", "Increase conversion"]
})

// 3. Ideate - Generate solutions
design_ideate_brainstorm({
  problemStatement: "How might we simplify signup?",
  inspirationDomains: ["gaming", "social media"],
  ideaCount: 10
})

// 4. Prototype - Generate architecture
design_prototype_architecture({
  requirements: ["social login", "progressive disclosure"],
  style: "microservices"
})

// 5. Test - Generate test cases
design_test_generate({
  specification: "User can signup with Google OAuth in < 30 seconds",
  testTypes: ["e2e", "property"]
})
```

### Systems Dynamics Workflow
```javascript
// Model a predator-prey system
systems_model_simulate({
  equations: [
    "x'[t] == a*x[t] - b*x[t]*y[t]",
    "y'[t] == -c*y[t] + d*x[t]*y[t]"
  ],
  initialConditions: { x: 100, y: 20 },
  parameters: { a: 0.1, b: 0.02, c: 0.3, d: 0.01 },
  timeSpan: [0, 200]
})

// Find equilibrium
systems_equilibrium_find({
  equations: ["a*x - b*x*y", "-c*y + d*x*y"],
  variables: ["x", "y"]
})

// Analyze stability
systems_equilibrium_stability({
  jacobian: [["-b*y_eq", "-b*x_eq"], ["d*y_eq", "d*x_eq - c"]]
})
```

### Dilithium Authorization Flow
```javascript
// 1. Register client (admin)
dilithium_register_client({
  name: "Sentry-Node-1",
  publicKey: "abc123...",  // Dilithium public key
  capabilities: ["llm_query", "compute", "systems_model"]
})

// 2. Client requests authorization
dilithium_authorize({
  clientId: "xyz789",
  publicKey: "abc123...",
  requestedCapabilities: ["llm_query", "compute"],
  timestamp: Date.now(),
  nonce: "random123",
  signature: "dilithium_signature..."  // Signed request
})
// Returns: { token: { clientId, expiresAt, capabilities, ... } }

// 3. Validate token before API call
dilithium_validate_token({ token: {...} })
// Returns: { valid: true }

// 4. Check quota
dilithium_check_quota({ clientId: "xyz789" })
// Returns: { allowed: true, remaining: { requests: 950, tokens: 99000 } }
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Wolfram MCP Server v2.0                           │
│                         (Bun.js)                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │
│  │ Core Wolfram │ │   Swarm      │ │   Design     │ │  Systems    │ │
│  │    Tools     │ │   Tools      │ │  Thinking    │ │  Dynamics   │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ │
│         │                │                │                │         │
│  ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴──────┐ │
│  │  LLM Tools   │ │  Dilithium   │ │    Native    │ │  Wolfram    │ │
│  │              │ │    Auth      │ │    Rust      │ │   Script    │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Wolfram Engine    │
                    │  (Local or Cloud)   │
                    └─────────────────────┘
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WOLFRAM_APP_ID` | Wolfram Alpha API key | - |
| `WOLFRAMSCRIPT_PATH` | Path to wolframscript | `/usr/local/bin/wolframscript` |
| `WOLFRAM_NATIVE_PATH` | Path to Rust module | `./native/wolfram-native.*.node` |
| `WOLFRAM_AUTH_DIR` | Auth state directory | `/tmp/wolfram-auth` |
| `WOLFRAM_SERVER_SECRET` | Token signing secret | (dev default) |
| `AGENT_MESH_DIR` | Swarm state directory | `/tmp/hyperphysics-mesh` |

---

## Build

```bash
cd tools/wolfram-mcp

# Install dependencies
bun install

# Build TypeScript
bun run build

# Build native Rust module
cd native && cargo build --release

# Build Swift CLI (macOS)
cd native/swift && swift build -c release
```

## Test

```bash
# Test the server
bun run dev

# In another terminal
echo '{"method":"tools/list"}' | bun run dist/index.js
```
