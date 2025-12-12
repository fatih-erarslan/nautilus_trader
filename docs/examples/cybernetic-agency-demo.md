# Cybernetic Agency Integration Demo

This example demonstrates the complete HyperPhysics cybernetic agency framework:
- **Free Energy Principle** (Karl Friston) - Survival through surprise minimization
- **Integrated Information Theory** (Giulio Tononi) - Consciousness as Φ
- **Active Inference** - Perception-action coupling
- **Hyperbolic Geometry** - H¹¹ consciousness substrate
- **Dilithium MCP** - Post-quantum secure agent coordination

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Cybernetic Agent System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  Free       │   │  Active     │   │  Survival   │          │
│  │  Energy     │──▶│  Inference  │──▶│  Drive      │          │
│  │  Engine     │   │  Engine     │   │  Controller │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│         │                 │                  │                 │
│         └─────────────────┼──────────────────┘                 │
│                           ▼                                    │
│                  ┌─────────────────┐                           │
│                  │  Homeostatic    │                           │
│                  │  Controller     │                           │
│                  │  (PID + Allost) │                           │
│                  └─────────────────┘                           │
│                           │                                    │
│                           ▼                                    │
│                  ┌─────────────────┐                           │
│                  │  Φ Calculator   │                           │
│                  │  (Consciousness)│                           │
│                  └─────────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Example 1: Basic Agent with Plugin API

```rust
use hyperphysics_plugin::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create agent with default configuration
    let config = AgentConfig {
        observation_dim: 32,
        action_dim: 16,
        hidden_dim: 64,
        learning_rate: 0.01,
        fe_min_rate: 0.1,
        survival_strength: 1.0,
        impermanence_rate: 0.4,
        branching_target: 1.0,
        use_dilithium: false,
    };

    let mut agent = CyberneticAgent::new(config);

    // Optional: Set consciousness calculator from plugin
    let phi_calc = PhiCalculator::greedy();
    agent.set_phi_calculator(Box::new(phi_calc));

    // Simulation loop
    for step in 0..1000 {
        // Create observation (e.g., from sensors)
        let observation = AgencyObservation {
            sensory: Array1::from_elem(32, 0.5 + 0.1 * (step as f64).sin()),
            timestamp: step,
        };

        // Agent processes observation and generates action
        let action = agent.step(&observation);

        // Monitor agent metrics
        println!("Step {}: Φ={:.3} F={:.3} S={:.3} C={:.3}",
            step,
            agent.integrated_information(),
            agent.free_energy(),
            agent.survival_drive(),
            agent.control_authority()
        );

        // Check for agency emergence
        if agent.integrated_information() > 1.0 && agent.control_authority() > 0.5 {
            println!("✓ Agency emerged at step {}", step);
        }
    }

    Ok(())
}
```

## Example 2: Hyperbolic Agent with Spatial Awareness

```rust
use hyperphysics_plugin::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = CyberneticAgent::new(AgentConfig::default());

    // Initialize in safe region (near origin in H¹¹)
    let mut safe_position = Array1::zeros(12);
    safe_position[0] = 1.0; // Lorentz origin: [1, 0, 0, ..., 0]
    agent.state.position = safe_position;

    // Simulate movement in hyperbolic space
    for step in 0..100 {
        let observation = create_observation(step);
        let action = agent.step(&observation);

        // Get survival drive (increases with distance from safety)
        let survival = agent.survival_drive();

        // Compute hyperbolic distance from origin
        let distance = hyperbolic_distance(&agent.state.position, &safe_position);

        println!("Step {}: distance={:.3} survival={:.3} threat={}",
            step,
            distance,
            survival,
            if survival > 0.7 { "HIGH" } else if survival > 0.3 { "MEDIUM" } else { "LOW" }
        );

        // If threat is high, agent should move back toward safety
        if survival > 0.8 {
            println!("⚠ Crisis mode activated - returning to safe region");
            // Action should minimize expected free energy by reducing distance
        }
    }

    Ok(())
}

fn create_observation(step: u64) -> AgencyObservation {
    // Create time-varying observation
    let noise = rand::thread_rng().gen_range(-0.1..0.1);
    AgencyObservation {
        sensory: Array1::from_elem(32, 0.5 + noise),
        timestamp: step,
    }
}
```

## Example 3: MCP Integration with Dilithium

Using the dilithium-mcp server to coordinate multiple agents:

```typescript
// mcp-client.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

async function main() {
    const client = new Client({
        name: "agency-demo",
        version: "1.0.0"
    });

    // Connect to dilithium-mcp server
    await client.connect();

    // 1. Create cybernetic agent
    const createResult = await client.callTool("agency_create_agent", {
        config: {
            observation_dim: 32,
            action_dim: 16,
            hidden_dim: 64,
            learning_rate: 0.01,
            survival_strength: 1.0,
            impermanence_rate: 0.4
        },
        phi_calculator_type: "greedy"
    });

    const agentId = JSON.parse(createResult).agent_id;
    console.log(`Created agent: ${agentId}`);

    // 2. Run agent simulation
    for (let step = 0; step < 100; step++) {
        // Generate observation
        const observation = Array(32).fill(0).map(() =>
            0.5 + 0.1 * Math.sin(step * 0.1)
        );

        // Execute agent step
        const stepResult = await client.callTool("agency_agent_step", {
            agent_id: agentId,
            observation: observation
        });

        const { state } = JSON.parse(stepResult);

        // Monitor consciousness and control
        if (step % 10 === 0) {
            console.log(`Step ${step}: Φ=${state.phi.toFixed(3)} F=${state.free_energy.toFixed(3)}`);
        }

        // Compute survival drive
        if (step % 20 === 0) {
            const survivalResult = await client.callTool("agency_compute_survival_drive", {
                free_energy: state.free_energy,
                position: [1, ...Array(11).fill(0)], // H¹¹ origin
                strength: 1.0
            });

            const survival = JSON.parse(survivalResult);
            console.log(`Survival: ${survival.survival_drive.toFixed(3)} (${survival.threat_level})`);
        }
    }

    // 3. Get final metrics
    const metricsResult = await client.callTool("agency_get_agent_metrics", {
        agent_id: agentId
    });

    const metrics = JSON.parse(metricsResult);
    console.log("\nFinal Agent Metrics:");
    console.log(`  Φ (Consciousness): ${metrics.metrics.phi.toFixed(3)} bits`);
    console.log(`  F (Free Energy): ${metrics.metrics.free_energy.toFixed(3)} nats`);
    console.log(`  Survival Drive: ${metrics.metrics.survival_drive.toFixed(3)}`);
    console.log(`  Control Authority: ${metrics.metrics.control_authority.toFixed(3)}`);
    console.log(`  Model Accuracy: ${metrics.metrics.model_accuracy.toFixed(3)}`);
    console.log(`  Branching Ratio: ${metrics.metrics.branching_ratio.toFixed(3)}`);
    console.log(`  Impermanence: ${metrics.metrics.impermanence.toFixed(3)}`);
    console.log(`  Health: ${metrics.health}`);
}

main().catch(console.error);
```

## Example 4: Multi-Agent Swarm with Consensus

```typescript
// swarm-agency.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

async function main() {
    const client = new Client({ name: "swarm-agency", version: "1.0.0" });
    await client.connect();

    // Create multiple agents
    const agents = [];
    for (let i = 0; i < 5; i++) {
        const result = await client.callTool("agency_create_agent", {
            config: {
                observation_dim: 32,
                action_dim: 16,
                hidden_dim: 64,
                survival_strength: 1.0 + i * 0.1 // Varying survival strength
            }
        });
        agents.push(JSON.parse(result).agent_id);
    }

    // Register agents with swarm coordinator
    for (const agentId of agents) {
        await client.callTool("swarm_register_agent", {
            id: agentId,
            public_key: "dummy_key", // Use real Dilithium keys in production
            capabilities: ["agency", "active_inference", "consciousness"]
        });
    }

    // Coordinate agents through shared memory
    for (let step = 0; step < 50; step++) {
        // Each agent processes observation
        for (const agentId of agents) {
            const observation = Array(32).fill(0).map(() => Math.random());
            await client.callTool("agency_agent_step", {
                agent_id: agentId,
                observation: observation
            });
        }

        // Share collective Φ in swarm memory
        if (step % 10 === 0) {
            let totalPhi = 0;
            for (const agentId of agents) {
                const metrics = await client.callTool("agency_get_agent_metrics", {
                    agent_id: agentId
                });
                totalPhi += JSON.parse(metrics).metrics.phi;
            }

            await client.callTool("swarm_set_memory", {
                key: `collective_phi_${step}`,
                value: totalPhi / agents.length,
                updated_by: "swarm_coordinator"
            });

            console.log(`Step ${step}: Collective Φ = ${(totalPhi / agents.length).toFixed(3)}`);
        }
    }

    // Create consensus proposal for agent behavior adjustment
    await client.callTool("swarm_create_proposal", {
        proposer: agents[0],
        topic: "Increase exploration vs exploitation ratio",
        options: ["increase_exploration", "increase_exploitation", "maintain_balance"],
        duration_ms: 10000
    });

    console.log("✓ Swarm consensus initiated");
}

main().catch(console.error);
```

## Validation with Wolfram

```mathematica
(* Load agency validation suite *)
Get["agency-validation.mx"]

(* Validate free energy computation *)
obs = {0.5, 0.6, 0.4, 0.7, 0.5};
beliefs = {0.48, 0.62, 0.38, 0.72, 0.49};
precision = {1.0, 1.0, 1.0, 1.0, 1.0};

validation = FreeEnergyValidation[obs, beliefs, precision]
(* Returns: <|"freeEnergy" -> 1.23, "complexity" -> 0.15,
             "accuracy" -> -1.08, "valid" -> True|> *)

(* Validate hyperbolic distance *)
position = {1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
distValidation = HyperbolicDistanceValidation[position]
(* Returns: <|"distance" -> 0.1, "normalized" -> 0.099, "valid" -> True|> *)

(* Validate survival drive *)
survivalValidation = SurvivalDriveValidation[1.5, 0.2]
(* Returns: <|"drive" -> 0.45, "threat_level" -> "caution",
             "crisis" -> False|> *)

(* Validate criticality *)
timeseries = RandomReal[{0, 1}, 1000];
critValidation = CriticalityValidation[timeseries]
(* Returns: <|"branchingRatio" -> 0.98, "atCriticality" -> True, ...|> *)
```

## Expected Output

When running the basic agent example, you should see output like:

```
Step 0: Φ=0.100 F=1.000 S=0.500 C=0.200
Step 10: Φ=0.215 F=0.932 S=0.480 C=0.245
Step 20: Φ=0.389 F=0.854 S=0.442 C=0.312
...
Step 100: Φ=1.124 F=0.723 S=0.385 C=0.567
✓ Agency emerged at step 102
...
Step 500: Φ=1.845 F=0.612 S=0.325 C=0.782
```

**Key Milestones:**
- Φ > 1.0: Consciousness emerges
- C > 0.5: Agency establishes control
- F < 0.8: Homeostasis achieved
- S ∈ [0.3, 0.8]: Optimal survival drive

## Integration Points

### 1. HyperPhysics Plugin
```rust
use hyperphysics_plugin::prelude::*;

// All agency types available through plugin prelude
let agent: CyberneticAgent = ...;
let phi_calc: PhiCalculator = ...;
let survival: SurvivalDrive = ...;
```

### 2. Dilithium MCP
14 agency tools available:
- `agency_create_agent` - Agent instantiation
- `agency_agent_step` - Perception-action loop
- `agency_compute_free_energy` - FEP calculations
- `agency_compute_phi` - Consciousness metrics
- `agency_compute_survival_drive` - Threat assessment
- `agency_regulate_homeostasis` - PID control
- And 8 more...

### 3. Swarm Coordination
```typescript
// Agents coordinate via swarm memory and consensus
swarm_register_agent()
swarm_send_message()
swarm_create_proposal()
swarm_vote()
```

## Next Steps

1. **Run the examples** to verify agency emergence
2. **Monitor metrics** to ensure Φ > 1.0, F < 2.0, σ ≈ 1.0
3. **Validate with Wolfram** using the provided code
4. **Scale to swarms** with multi-agent coordination
5. **Deploy to production** with Dilithium authentication

## References

- Friston, K. (2010). "The free-energy principle" *Nature Reviews Neuroscience*, 11(2), 127-138.
- Tononi, G. et al. (2016). "Integrated information theory" *Nature Reviews Neuroscience*, 17(7), 450-461.
- Implementation docs: `/Volumes/Tengritek/Ashina/HyperPhysics/docs/research/cybernetic-agency-framework.md`
- Summary report: `/Volumes/Tengritek/Ashina/HyperPhysics/docs/research/CYBERNETIC-AGENCY-IMPLEMENTATION-SUMMARY.md`
