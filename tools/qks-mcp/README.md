# QKS MCP Server v2.0 - 8-Layer Cognitive Architecture

**Agentic access to the Quantum Knowledge System** - A complete cognitive architecture for AI agents.

## Overview

The QKS MCP Server exposes 64 tools across 8 cognitive layers, providing Claude Code and other AI agents with access to:

- **Thermodynamic optimization** (energy, entropy, temperature)
- **Cognitive functions** (attention, memory, perception)
- **Active inference** (decision making, free energy principle)
- **Learning algorithms** (STDP, meta-learning, transfer learning)
- **Collective intelligence** (swarm coordination, consensus)
- **Consciousness metrics** (IIT Φ, global workspace theory)
- **Metacognition** (introspection, self-modeling)
- **Full agency** (homeostasis, autopoiesis, system integration)

## Quick Start

### Installation

```bash
cd /Volumes/Tengritek/Ashina/quantum_knowledge_system/tools/qks-mcp
bun install
bun run build
```

### Run Server

```bash
bun run dist/index.js
```

### Configure Claude Code

Add to `~/.config/claude-code/mcp_config.json`:

```json
{
  "mcpServers": {
    "qks": {
      "command": "bun",
      "args": ["run", "/Volumes/Tengritek/Ashina/quantum_knowledge_system/tools/qks-mcp/dist/index.js"],
      "env": {
        "QKS_NATIVE_PATH": "/Volumes/Tengritek/Ashina/quantum_knowledge_system/rust-core/target/release/libqks_core.dylib"
      }
    }
  }
}
```

## 8-Layer Architecture

### Layer 1: Thermodynamic Foundation (6 tools)

**Scientific basis**: Landauer's principle, Boltzmann statistics, free energy minimization

| Tool | Description |
|------|-------------|
| `qks_thermo_energy` | Compute system energy (Helmholtz free energy F = E - TS) |
| `qks_thermo_temperature` | Effective temperature via Boltzmann statistics |
| `qks_thermo_entropy` | Shannon entropy S = -Σ p_i log₂(p_i) |
| `qks_thermo_critical_point` | Phase transition critical points (Ising model) |
| `qks_thermo_landauer_cost` | Minimum energy cost for computation (kT ln(2) per bit) |
| `qks_thermo_free_energy` | Free energy F = E - TS (drives system evolution) |

**Example**:
```typescript
// Compute Landauer cost for 1000 bit erasures at room temperature
{
  "tool": "qks_thermo_landauer_cost",
  "args": {
    "temperature": 300,
    "num_bits": 1000,
    "operation": "erase"
  }
}
// Returns: { landauer_cost: 2.87e-18 J, ... }
```

---

### Layer 2: Cognitive Architecture (8 tools)

**Scientific basis**: Working memory (Miller's 7±2), attention mechanisms, memory consolidation

| Tool | Description |
|------|-------------|
| `qks_cognitive_attention` | Softmax attention weights over inputs |
| `qks_cognitive_memory_store` | Store in working/episodic/semantic memory |
| `qks_cognitive_memory_consolidate` | Hippocampal replay consolidation |
| `qks_cognitive_pattern_match` | Pattern matching with similarity metrics |
| `qks_cognitive_perceive` | Process sensory input through perception pipeline |
| `qks_cognitive_working_memory_capacity` | Estimate working memory capacity |
| `qks_cognitive_attention_gate` | Apply attention gating (binary mask) |
| `qks_cognitive_memory_decay` | Exponential memory decay (forgetting) |

**Example**:
```typescript
// Apply attention to inputs
{
  "tool": "qks_cognitive_attention",
  "args": {
    "inputs": [0.2, 0.8, 0.5, 0.3],
    "temperature": 1.0
  }
}
// Returns: { focus_weights: [0.12, 0.45, 0.28, 0.15], entropy: 1.73, ... }
```

---

### Layer 3: Decision Making (8 tools)

**Scientific basis**: Active inference, Free Energy Principle (Friston), Bayesian inference

| Tool | Description |
|------|-------------|
| `qks_decision_compute_efe` | Expected Free Energy (EFE = Pragmatic + Epistemic) |
| `qks_decision_select_action` | Active inference action selection |
| `qks_decision_update_beliefs` | Bayesian belief update P(s\|o) ∝ P(o\|s)P(s) |
| `qks_decision_epistemic_value` | Information gain (uncertainty reduction) |
| `qks_decision_pragmatic_value` | Expected utility (goal achievement) |
| `qks_decision_inference_step` | One active inference step |
| `qks_decision_prediction_error` | Compute surprise PE = -log P(o\|s) |
| `qks_decision_precision_weighting` | Precision-weighted prediction errors |

**Example**:
```typescript
// Compute Expected Free Energy for policy
{
  "tool": "qks_decision_compute_efe",
  "args": {
    "policy": { "actions": ["explore", "observe"], "expected_outcomes": [...] },
    "beliefs": { "beliefs": [0.4, 0.4, 0.2], "precision": [1.0, 1.0, 1.0] },
    "preferences": [0.2, 0.6, 0.2]
  }
}
// Returns: { expected_free_energy: -2.3, epistemic_value: 0.8, pragmatic_value: -3.1, ... }
```

---

### Layer 4: Learning & Reasoning (8 tools)

**Scientific basis**: STDP (Bi & Poo, 1998), MAML (Finn et al., 2017), Curriculum Learning

| Tool | Description |
|------|-------------|
| `qks_learning_stdp` | Spike-Timing Dependent Plasticity weight change |
| `qks_learning_consolidate` | Episodic → Semantic memory consolidation |
| `qks_learning_transfer` | Transfer learning efficiency estimation |
| `qks_learning_reasoning_route` | Route reasoning to LSH/symbolic backend |
| `qks_learning_curriculum` | Generate curriculum (progressive, spiral, ZPD) |
| `qks_learning_meta_adapt` | MAML fast adaptation |
| `qks_learning_catastrophic_forgetting` | EWC-based forgetting prevention |
| `qks_learning_gradient_analysis` | Detect vanishing/exploding gradients |

**Example**:
```typescript
// Compute STDP weight change
{
  "tool": "qks_learning_stdp",
  "args": {
    "pre_spike_time": 10.0,
    "post_spike_time": 12.5,
    "a_plus": 0.1,
    "tau": 20.0
  }
}
// Returns: { weight_change: 0.0887, plasticity_type: "LTP (potentiation)", ... }
```

---

### Layer 5: Collective Intelligence (8 tools)

**Scientific basis**: Swarm intelligence, Byzantine consensus, Stigmergy (Grassé, 1959)

| Tool | Description |
|------|-------------|
| `qks_collective_swarm_coordinate` | Swarm coordination with topology |
| `qks_collective_consensus` | Voting protocols (Raft, Byzantine, Quorum) |
| `qks_collective_stigmergy` | Indirect coordination via environment |
| `qks_collective_register_agent` | Agent registration with coordinator |
| `qks_collective_message_broadcast` | Broadcast to topology |
| `qks_collective_distributed_memory` | CRDT-based distributed memory |
| `qks_collective_quorum_decision` | Quorum consensus decision |
| `qks_collective_emerge` | Detect emergent collective behavior |

**Example**:
```typescript
// Reach consensus on proposal
{
  "tool": "qks_collective_consensus",
  "args": {
    "proposal": { "id": "prop_123", "content": { "action": "upgrade" } },
    "votes": [
      { "agent": "a1", "vote": true },
      { "agent": "a2", "vote": true },
      { "agent": "a3", "vote": false }
    ],
    "protocol": "simple_majority"
  }
}
// Returns: { consensus_reached: true, votes_for: 2, votes_against: 1, ... }
```

---

### Layer 6: Consciousness (8 tools)

**Scientific basis**: IIT 3.0 (Tononi et al., 2016), Global Workspace Theory (Baars, 1988)

| Tool | Description |
|------|-------------|
| `qks_consciousness_compute_phi` | Integrated Information Φ (IIT 3.0) |
| `qks_consciousness_global_workspace` | Broadcast to global workspace |
| `qks_consciousness_phase_coherence` | Phase synchrony (Kuramoto order parameter) |
| `qks_consciousness_integration` | Measure information integration |
| `qks_consciousness_complexity` | Neural complexity (Tononi et al., 1994) |
| `qks_consciousness_attention_schema` | Attention Schema Theory (Graziano) |
| `qks_consciousness_qualia_space` | Map phenomenal experience |
| `qks_consciousness_reportability` | Access vs phenomenal consciousness |

**Example**:
```typescript
// Compute integrated information Φ
{
  "tool": "qks_consciousness_compute_phi",
  "args": {
    "network_state": [1, 0, 1, 1, 0, 1, 0, 0],
    "algorithm": "greedy"
  }
}
// Returns: { phi: 1.35, interpretation: "High Φ - System exhibits integrated information", ... }
```

---

### Layer 7: Metacognition (10 tools)

**Scientific basis**: Meta-learning (MAML), Confidence calibration, Self-modeling

| Tool | Description |
|------|-------------|
| `qks_meta_introspect` | Real-time introspection of cognitive state |
| `qks_meta_self_model` | Access self-model (beliefs, goals, capabilities) |
| `qks_meta_update_beliefs` | Precision-weighted belief update |
| `qks_meta_confidence` | Compute calibrated confidence |
| `qks_meta_calibrate_confidence` | Confidence calibration via history |
| `qks_meta_detect_uncertainty` | Epistemic vs aleatoric uncertainty |
| `qks_meta_learn` | MAML meta-learning |
| `qks_meta_strategy_select` | Select metacognitive strategy |
| `qks_meta_conflict_resolution` | Resolve internal conflicts |
| `qks_meta_goal_management` | Goal hierarchy management |

**Example**:
```typescript
// Introspect internal state
{
  "tool": "qks_meta_introspect",
  "args": {
    "depth": 2
  }
}
// Returns: {
//   internal_state: { energy: 1.0, temperature: 1.0 },
//   certainty: 0.75,
//   coherence: 0.85,
//   conflicts: [],
//   depth: 2
// }
```

---

### Layer 8: Full Agency (8 tools)

**Scientific basis**: Autopoiesis (Maturana & Varela), Self-Organized Criticality (Bak et al.)

| Tool | Description |
|------|-------------|
| `qks_system_health` | Overall system health across all layers |
| `qks_cognitive_loop` | Perception → Inference → Action → Feedback |
| `qks_homeostasis` | Homeostatic balance maintenance |
| `qks_emergent_features` | Detect emergent higher-order features |
| `qks_orchestrate` | Orchestrate all 8 layers for unified agency |
| `qks_autopoiesis` | Autopoietic self-organization |
| `qks_criticality` | Self-organized criticality assessment |
| `qks_full_cycle` | Complete 8-layer processing cycle |

**Example**:
```typescript
// Get system health
{
  "tool": "qks_system_health",
  "args": {
    "detailed": true
  }
}
// Returns: {
//   overall_health: 0.95,
//   layers: {
//     L1_thermodynamic: { health: 1.0, status: "operational" },
//     L2_cognitive: { health: 0.98, status: "operational" },
//     ...
//   }
// }
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QKS MCP SERVER v2.0                             │
│                   8-Layer Cognitive Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Claude  │  │   MCP    │  │  Stdio   │  │  Rust    │            │
│  │   Code   │◄─┤ Protocol │◄─┤Transport │◄─┤  Bridge  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                     │
│  ┌───────────────────────────────────────────────────────┐          │
│  │                 64 MCP Tools (8 Layers)                │          │
│  ├───────────────────────────────────────────────────────┤          │
│  │ L8: Full Agency (Homeostasis, Autopoiesis)       [8]  │          │
│  │ L7: Metacognition (Introspection, Meta-learning) [10] │          │
│  │ L6: Consciousness (IIT Φ, Global Workspace)      [8]  │          │
│  │ L5: Collective (Swarm, Consensus)                [8]  │          │
│  │ L4: Learning (STDP, Meta-learning)               [8]  │          │
│  │ L3: Decision (Active Inference, EFE)             [8]  │          │
│  │ L2: Cognitive (Attention, Memory)                [8]  │          │
│  │ L1: Thermodynamic (Energy, Entropy)              [6]  │          │
│  └───────────────────────────────────────────────────────┘          │
│                         ▲                                           │
│                         │                                           │
│  ┌──────────────────────┴────────────────────────┐                  │
│  │       Rust Core (FFI/TypeScript Fallback)     │                  │
│  │  • libqks_core.dylib (native performance)     │                  │
│  │  • TypeScript implementations (development)    │                  │
│  └───────────────────────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Development

### Build Native Module (Optional but Recommended)

```bash
cd ../../rust-core
cargo build --release
cp target/release/libqks_core.dylib ../tools/qks-mcp/dist/
```

### Watch Mode

```bash
bun run dev
```

### Testing

```bash
bun test
```

### Linting

```bash
bun run lint
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QKS_NATIVE_PATH` | Path to Rust native module | Auto-detect |
| `QKS_MCP_PORT` | Server port (if TCP mode) | 3002 |

## Performance

- **Tool Execution**: <100ms per tool (TypeScript fallback)
- **Tool Execution**: <10ms per tool (Rust native)
- **Memory Footprint**: ~50MB (TypeScript), ~20MB (Rust)
- **Startup Time**: ~200ms

## Scientific References

### Thermodynamics
- Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process
- Bérut et al. (2012). Experimental verification of Landauer's principle

### Cognitive Architecture
- Miller, G. A. (1956). The Magical Number Seven, Plus or Minus Two
- Baddeley & Hitch (1974). Working Memory

### Active Inference
- Friston, K. (2010). The Free-Energy Principle
- Friston et al. (2017). Active Inference: A Process Theory

### Learning
- Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons
- Finn et al. (2017). Model-Agnostic Meta-Learning (MAML)

### Collective Intelligence
- Grassé, P. (1959). La reconstruction du nid et les coordinations inter-individuelles
- Raft Consensus Algorithm (Ongaro & Ousterhout, 2014)

### Consciousness
- Tononi et al. (2016). Integrated Information Theory (IIT) 3.0
- Baars, B. J. (1988). A Cognitive Theory of Consciousness

### Metacognition
- Flavell, J. H. (1979). Metacognition and Cognitive Monitoring

### Agency & Autopoiesis
- Maturana & Varela (1980). Autopoiesis and Cognition
- Bak et al. (1987). Self-Organized Criticality

## License

MIT

## Related Projects

- [Quantum Knowledge System](../../README.md)
- [QKS Rust Core](../../rust-core/)
- [Dilithium MCP](../../../HyperPhysics/tools/dilithium-mcp/)
- [HyperPhysics Plugin](../../../HyperPhysics/crates/hyperphysics-plugin/)
