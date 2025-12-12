# HyperPhysics Agency Functions - Dilithium MCP Integration

## Overview

Successfully integrated **hyperphysics-agency** crate functions into the dilithium-mcp native Rust module, exposing cybernetic agency capabilities through NAPI-RS bindings.

## Implemented Functions

### 1. `agency_create_agent(config_json: string) -> AgencyResult`

Creates a new cybernetic agent with given configuration.

**Config Fields:**
- `observation_dim`: Observation space dimensionality
- `action_dim`: Action space dimensionality
- `hidden_dim`: Hidden state dimensionality
- `learning_rate`: Belief update learning rate (default: 0.01)
- `fe_min_rate`: Free energy minimization rate
- `survival_strength`: Survival drive strength (default: 1.0)
- `impermanence_rate`: State change rate (default: 0.4)
- `branching_target`: Criticality target (default: 1.0)
- `use_dilithium`: Enable Dilithium signatures

**Returns:**
```typescript
{
  success: boolean,
  agent_id?: string,
  error?: string
}
```

### 2. `agency_agent_step(agent_id: string, observation_json: string) -> AgencyResult`

Executes one time step: observation → inference → action.

**Returns:**
```typescript
{
  success: boolean,
  agent_id: string,
  data: {
    action: number[],  // Motor commands
    state: {
      phi: number,           // Integrated information
      free_energy: number,   // Variational free energy
      survival: number,      // Survival drive [0,1]
      control: number,       // Control authority [0,1]
      model_accuracy: number // Model accuracy [0,1]
    },
    metrics: {
      phi: number,
      free_energy: number,
      survival: number,
      control: number
    }
  }
}
```

### 3. `agency_compute_free_energy(observation: number[], beliefs: number[], precision: number[]) -> AgencyResult`

Computes variational free energy: **F = D_KL[q||p] + accuracy**

**Returns:**
```typescript
{
  success: boolean,
  data: {
    F: number,           // Total free energy (nats)
    complexity: number,  // KL divergence term
    accuracy: number     // Prediction accuracy term
  }
}
```

**Note:** Currently has dimension handling issues - being investigated.

### 4. `agency_compute_survival_drive(free_energy: number, position: number[]) -> AgencyResult`

Computes survival urgency from free energy and hyperbolic position.

**Position:** 12D Lorentz coordinates in H¹¹ (hyperbolic space)

**Returns:**
```typescript
{
  success: boolean,
  data: {
    drive: number,                    // Survival urgency [0,1]
    threat_level: number,             // Threat assessment [0,1]
    homeostatic_status: string,       // "safe" | "stressed" | "critical"
    in_crisis: boolean,
    threat_assessment: {
      detected: boolean,
      free_energy_contribution: number,
      distance_contribution: number,
      rate_of_change: number
    }
  }
}
```

### 5. `agency_compute_phi(network_state: number[]) -> AgencyResult`

Computes integrated information Φ (consciousness metric).

**Returns:**
```typescript
{
  success: boolean,
  data: {
    phi: number,                   // Integrated information (bits)
    consciousness_level: string,   // "minimal" | "moderate" | "high"
    coherence: number              // Belief coherence
  }
}
```

### 6. `agency_analyze_criticality(timeseries: number[]) -> AgencyResult`

Analyzes self-organized criticality markers.

**Returns:**
```typescript
{
  success: boolean,
  data: {
    branching_ratio: number,      // σ (≈1.0 at criticality)
    at_criticality: boolean,      // True if σ ≈ 1.0 ± 0.1
    hurst_exponent: number,       // H (long-range correlations)
    criticality_distance: number  // |σ - 1.0|
  }
}
```

### 7. `agency_regulate_homeostasis(current_state_json: string, setpoints_json?: string) -> AgencyResult`

Performs homeostatic regulation with PID control + allostatic prediction.

**Current State:**
```typescript
{
  phi: number,
  free_energy: number,
  survival: number,
  control?: number,
  model_accuracy?: number
}
```

**Setpoints (optional):**
```typescript
{
  phi_setpoint: number,
  fe_setpoint: number,
  survival_setpoint: number
}
```

**Returns:**
```typescript
{
  success: boolean,
  data: {
    control_signals: {
      phi_correction: number,
      fe_correction: number,
      survival_correction: number
    },
    allostatic_adjustment: number,
    disturbance_rejection: number,
    prediction_confidence: number,
    regulated_state: {
      phi: number,
      free_energy: number,
      survival: number
    }
  }
}
```

## Test Results

✅ **Agent Creation**: Successfully creates agents with unique IDs
✅ **Agent Step**: Executes perception → inference → action cycle
⚠️ **Free Energy Computation**: Dimension handling issue (under investigation)
✅ **Survival Drive**: Computes threat levels and homeostatic status
✅ **Consciousness (Φ)**: Computes integrated information metrics
✅ **Criticality Analysis**: Detects self-organized criticality (σ ≈ 1.0)
✅ **Homeostatic Regulation**: PID control with allostatic prediction

## Files Modified

1. **`/tools/dilithium-mcp/native/Cargo.toml`**
   - Added `hyperphysics-agency` dependency
   - Added `ndarray` dependency with rayon and serde features

2. **`/tools/dilithium-mcp/native/src/lib.rs`**
   - Added 7 NAPI-RS bindings for agency functions
   - Implemented agent registry with thread-safe storage
   - Added JSON serialization for complex return types

3. **`/tools/dilithium-mcp/native/index.js`**
   - Created Node.js wrapper for native addon

4. **`/tools/dilithium-mcp/test-agency.ts`**
   - Created comprehensive test suite

## Theoretical Foundation

The implemented functions are based on:

- **Free Energy Principle** (Karl Friston): Minimizing variational free energy
- **Integrated Information Theory** (Giulio Tononi): Φ as consciousness metric
- **Self-Organized Criticality** (Bak, Tang, Wiesenfeld): Edge of chaos dynamics
- **Autopoiesis** (Maturana & Varela): Self-maintenance and adaptation
- **Hyperbolic Geometry**: H¹¹ Lorentz model for threat assessment

## Usage Example

```typescript
import {
  agencyCreateAgent,
  agencyAgentStep,
  agencyComputeSurvivalDrive,
  agencyComputePhi,
  agencyAnalyzeCriticality,
  agencyRegulateHomeostasis,
} from './native/index.js';

// Create agent
const config = JSON.stringify({
  observation_dim: 32,
  action_dim: 16,
  hidden_dim: 64,
  learning_rate: 0.01,
  fe_min_rate: 0.1,
  survival_strength: 1.0,
  impermanence_rate: 0.4,
  branching_target: 1.0,
  use_dilithium: false,
});

const { agentId } = agencyCreateAgent(config);

// Run agent step
const observation = JSON.stringify(new Array(32).fill(0.5));
const { data } = agencyAgentStep(agentId, observation);
const metrics = JSON.parse(data);

console.log('Φ (consciousness):', metrics.metrics.phi);
console.log('Survival drive:', metrics.metrics.survival);
console.log('Control authority:', metrics.metrics.control);
```

## Next Steps

1. Fix dimension handling issue in `agency_compute_free_energy`
2. Add TypeScript type definitions (`.d.ts` file)
3. Implement additional agency functions:
   - `agency_get_agent_metrics(agent_id)`
   - `agency_reset_agent(agent_id)`
   - `agency_delete_agent(agent_id)`
4. Add comprehensive error handling
5. Create integration tests with MCP server

## Performance Notes

- Agent operations are **thread-safe** using `Arc<RwLock<CyberneticAgent>>`
- Agent registry uses `DashMap` for concurrent access
- Native Rust performance with minimal JS overhead
- SIMD-optimized matrix operations via ndarray

---

**Implementation Status**: ✅ **COMPLETED** (6/7 functions fully operational)
