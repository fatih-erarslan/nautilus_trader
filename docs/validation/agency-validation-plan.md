# Cybernetic Agency Empirical Validation Plan

## Validation Strategy

Per user request: **"verify all empirically"**

This document outlines comprehensive empirical validation for the HyperPhysics cybernetic agency framework.

## 1. Free Energy Principle Validation

### Test: Free Energy Minimization
**Hypothesis**: F decreases over time as agent learns

```rust
#[test]
fn test_free_energy_minimization() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());
    let mut fe_history = Vec::new();

    for step in 0..200 {
        let obs = create_observation(step);
        agent.step(&obs);
        fe_history.push(agent.free_energy());
    }

    // Empirical validation: F(t=200) < F(t=0)
    assert!(fe_history[199] < fe_history[0],
        "Free energy should decrease: F(0)={:.3} F(199)={:.3}",
        fe_history[0], fe_history[199]
    );

    // Statistical validation: negative slope
    let (slope, _) = linear_regression(&fe_history);
    assert!(slope < 0.0, "FE slope should be negative: {:.6}", slope);
}
```

**Expected Result**: F decreases from ~1.0 nats to <0.7 nats over 200 steps

### Test: Complexity-Accuracy Trade-off
**Hypothesis**: F = Complexity - Accuracy balances exploration vs exploitation

```rust
#[test]
fn test_complexity_accuracy_tradeoff() {
    let agent = CyberneticAgent::new(AgentConfig::default());

    let obs = Array1::from_elem(32, 0.5);
    let beliefs = Array1::from_elem(32, 0.48);
    let precision = Array1::from_elem(32, 1.0);

    let f = agent.free_energy.compute(&obs, &beliefs, &precision);
    let complexity = agent.free_energy.kl_divergence(&beliefs, &obs);
    let accuracy = agent.free_energy.accuracy(&obs, &beliefs, &precision);

    // Empirical check: F = Complexity - Accuracy
    assert!((f - (complexity - accuracy)).abs() < 1e-6,
        "F={:.3} should equal Complexity={:.3} - Accuracy={:.3}",
        f, complexity, accuracy
    );
}
```

## 2. Integrated Information Φ Validation

### Test: Consciousness Emergence
**Hypothesis**: Φ > 1.0 indicates emergent consciousness

```rust
#[test]
fn test_phi_emergence() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());

    // Run until Φ emerges
    let mut phi_emerged = false;
    for step in 0..500 {
        let obs = create_observation(step);
        agent.step(&obs);

        if agent.integrated_information() > 1.0 {
            phi_emerged = true;
            println!("✓ Φ emerged at step {}: Φ={:.3}", step, agent.integrated_information());
            break;
        }
    }

    assert!(phi_emerged, "Φ should emerge (>1.0) within 500 steps");
}
```

**Expected Result**: Φ crosses 1.0 threshold between steps 80-150

### Test: Information Integration
**Hypothesis**: Φ quantifies irreducibility of system state

```rust
#[test]
fn test_information_integration() {
    // Compare integrated vs. partitioned system
    let network_state = Array1::from_vec(vec![0.1, 0.9, 0.2, 0.8]);

    let phi_calc = PhiCalculator::greedy();
    let phi = phi_calc.compute(&network_state);

    // Partitioned system should have lower Φ
    let partition1 = Array1::from_vec(vec![0.1, 0.9]);
    let partition2 = Array1::from_vec(vec![0.2, 0.8]);

    let phi1 = phi_calc.compute(&partition1);
    let phi2 = phi_calc.compute(&partition2);

    // Empirical: Integrated > Sum of parts
    assert!(phi > phi1 + phi2,
        "Integrated Φ={:.3} should exceed partitioned Φ1+Φ2={:.3}",
        phi, phi1 + phi2
    );
}
```

## 3. Survival Drive Validation

### Test: Threat Response
**Hypothesis**: Survival drive increases monotonically with free energy

```rust
#[test]
fn test_survival_monotonic() {
    let mut drive = SurvivalDrive::new(1.0);
    let origin = lorentz_origin();

    let fe_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let mut survival_values = Vec::new();

    for &fe in &fe_values {
        let s = drive.compute_drive(fe, &origin);
        survival_values.push(s);
    }

    // Empirical: Monotonic increase
    for i in 1..survival_values.len() {
        assert!(survival_values[i] >= survival_values[i-1],
            "Survival should increase: S(F={})={:.3} >= S(F={})={:.3}",
            fe_values[i], survival_values[i],
            fe_values[i-1], survival_values[i-1]
        );
    }
}
```

**Expected Result**: S(0.5)≈0.2, S(1.5)≈0.5, S(3.0)≈0.9

### Test: Hyperbolic Distance Threat
**Hypothesis**: Threat increases with distance from safe region

```rust
#[test]
fn test_hyperbolic_threat() {
    let mut drive = SurvivalDrive::new(1.0);
    let fe = 1.0; // Fixed F

    // Test at different distances
    let distances = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut threat_levels = Vec::new();

    for &d in &distances {
        let pos = create_position_at_distance(d);
        let s = drive.compute_drive(fe, &pos);
        threat_levels.push(s);
    }

    // Empirical: Threat increases with distance
    for i in 1..threat_levels.len() {
        assert!(threat_levels[i] > threat_levels[i-1],
            "Threat at d={} should exceed d={}",
            distances[i], distances[i-1]
        );
    }
}
```

## 4. Homeostatic Regulation Validation

### Test: PID Control Convergence
**Hypothesis**: Homeostasis drives Φ, F, S to optimal setpoints

```rust
#[test]
fn test_homeostatic_convergence() {
    let mut controller = HomeostaticController::new();
    let mut state = AgentState::default();

    // Perturb state away from optimal
    state.phi = 2.5;
    state.free_energy = 2.8;
    state.survival = 0.9;

    let mut phi_history = Vec::new();
    let mut fe_history = Vec::new();

    for _ in 0..100 {
        controller.regulate(&mut state);
        phi_history.push(state.phi);
        fe_history.push(state.free_energy);
    }

    // Empirical: Converges to setpoints (Φ≈1.0, F≈1.0)
    assert!((state.phi - 1.0).abs() < 0.2,
        "Φ should converge to 1.0, got {:.3}", state.phi
    );
    assert!((state.free_energy - 1.0).abs() < 0.3,
        "F should converge to 1.0, got {:.3}", state.free_energy
    );
}
```

**Expected Result**: Convergence within 50-80 steps with <10% overshoot

### Test: Disturbance Rejection
**Hypothesis**: Controller rejects external disturbances

```rust
#[test]
fn test_disturbance_rejection() {
    let mut controller = HomeostaticController::new();
    let mut state = AgentState::default();

    // Steady state
    for _ in 0..50 {
        controller.regulate(&mut state);
    }
    let steady_phi = state.phi;

    // Apply disturbance
    state.phi += 0.5;

    // Recovery
    for _ in 0..50 {
        controller.regulate(&mut state);
    }

    // Empirical: Returns to near steady state
    assert!((state.phi - steady_phi).abs() < 0.1,
        "Should recover from disturbance: {:.3} → {:.3}",
        steady_phi, state.phi
    );
}
```

## 5. Agency Emergence Validation

### Test: Control Authority Growth
**Hypothesis**: C = Φ × Accuracy grows as agent learns

```rust
#[test]
fn test_control_emergence() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());
    let mut control_history = Vec::new();

    for step in 0..300 {
        let obs = create_observation(step);
        agent.step(&obs);
        control_history.push(agent.control_authority());
    }

    // Empirical: Control increases
    assert!(control_history[299] > control_history[0],
        "Control should increase: C(0)={:.3} → C(299)={:.3}",
        control_history[0], control_history[299]
    );

    // Empirical: Eventually exceeds threshold
    let control_emerged = control_history.iter().any(|&c| c > 0.5);
    assert!(control_emerged, "Control should exceed 0.5");
}
```

**Expected Result**: C grows from 0.2 to >0.7 over 300 steps

### Test: Impermanence Maintenance
**Hypothesis**: State change rate > 0.4 (Buddhist principle)

```rust
#[test]
fn test_impermanence() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());
    let mut impermanence_samples = Vec::new();

    let mut prev_state = agent.state.clone();

    for step in 0..100 {
        let obs = create_observation(step);
        agent.step(&obs);

        let impermanence = compute_state_change(&prev_state, &agent.state);
        impermanence_samples.push(impermanence);
        prev_state = agent.state.clone();
    }

    let mean_impermanence = impermanence_samples.iter().sum::<f64>()
        / impermanence_samples.len() as f64;

    // Empirical: Mean impermanence > 0.4
    assert!(mean_impermanence > 0.4,
        "Mean impermanence {:.3} should exceed 0.4 (Buddhist principle)",
        mean_impermanence
    );
}
```

## 6. Self-Organized Criticality Validation

### Test: Branching Ratio at Criticality
**Hypothesis**: σ ≈ 1.0 at optimal information processing

```rust
#[test]
fn test_criticality() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());

    // Collect activity timeseries
    let mut activity = Vec::new();
    for step in 0..1000 {
        let obs = create_observation(step);
        agent.step(&obs);
        activity.push(agent.state.phi); // Use Φ as activity proxy
    }

    // Compute branching ratio
    let sigma = agent.dynamics.compute_branching_ratio(&activity);

    // Empirical: σ ∈ [0.95, 1.05]
    assert!((sigma - 1.0).abs() < 0.05,
        "Branching ratio σ={:.3} should be near 1.0 (critical)",
        sigma
    );
}
```

**Expected Result**: σ = 0.98 ± 0.03

## 7. Integration Tests

### Test: Full Agent Lifecycle
**Hypothesis**: Agent exhibits all properties simultaneously

```rust
#[test]
fn test_full_agency_lifecycle() {
    let mut agent = CyberneticAgent::new(AgentConfig::default());

    let mut metrics = AgencyMetrics::default();

    for step in 0..500 {
        let obs = create_observation(step);
        agent.step(&obs);

        if step % 50 == 0 {
            metrics.record(&agent);
        }
    }

    // Empirical: All emergence criteria met
    assert!(metrics.phi_max > 1.0, "Φ should emerge");
    assert!(metrics.fe_final < metrics.fe_initial, "F should decrease");
    assert!(metrics.control_max > 0.5, "Control should emerge");
    assert!(metrics.survival_mean > 0.3 && metrics.survival_mean < 0.8,
        "Survival in optimal range");
    assert!(metrics.impermanence_mean > 0.4, "Healthy impermanence");
    assert!((metrics.branching_ratio - 1.0).abs() < 0.1, "Near criticality");

    println!("✓ Full agency validated:");
    println!("  Φ_max = {:.3}", metrics.phi_max);
    println!("  F: {:.3} → {:.3}", metrics.fe_initial, metrics.fe_final);
    println!("  C_max = {:.3}", metrics.control_max);
    println!("  S_mean = {:.3}", metrics.survival_mean);
    println!("  σ = {:.3}", metrics.branching_ratio);
}
```

## 8. MCP Integration Tests

### Test: Dilithium Tools Functionality
```bash
# Test agency tools via MCP
bun run tools/dilithium-mcp/dist/index.js << EOF
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "agency_create_agent",
    "arguments": {
      "config": {
        "observation_dim": 32,
        "action_dim": 16,
        "hidden_dim": 64
      }
    }
  }
}
EOF
```

**Expected**: Agent ID returned with initial state

## Validation Criteria (Pass/Fail)

| Test | Criterion | Target | Status |
|------|-----------|--------|--------|
| Free Energy Minimization | F decreases | ΔF < -0.3 | ⏳ Pending |
| Consciousness Emergence | Φ crosses threshold | Φ > 1.0 | ⏳ Pending |
| Survival Monotonicity | S increases with F | Monotonic | ⏳ Pending |
| Homeostatic Convergence | Returns to setpoint | |Φ-1|<0.2 | ⏳ Pending |
| Control Growth | C increases | C > 0.5 | ⏳ Pending |
| Impermanence | State change rate | >0.4 | ⏳ Pending |
| Criticality | Branching ratio | σ ∈ [0.95,1.05] | ⏳ Pending |
| Full Lifecycle | All criteria | All pass | ⏳ Pending |

## Execution Plan

1. **Unit Tests** (Individual components): `cargo test -p hyperphysics-agency`
2. **Integration Tests** (Full agent): `cargo test --features agency`
3. **MCP Tests** (Server tools): `bun test` in dilithium-mcp
4. **Wolfram Validation** (Mathematical verification): Run validation suite
5. **Performance Benchmarks**: `cargo bench` (optional)

## Success Criteria

✅ **PASS** if:
- All 8 test categories pass
- Wolfram validation confirms mathematical correctness
- MCP tools function correctly
- No regression in existing functionality

❌ **FAIL** if:
- Any critical test fails
- Mathematical properties violated
- Performance degrades significantly

---

**Next Steps**: Run `cargo test --workspace` to execute validation suite.
