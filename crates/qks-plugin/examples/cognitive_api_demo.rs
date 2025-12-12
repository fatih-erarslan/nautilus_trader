//! # Cognitive API Demo
//!
//! Demonstrates all 8 layers of the QKS cognitive API

use qks_plugin::api::prelude::*;
use qks_plugin::Result;

fn main() -> Result<()> {
    println!("=== QKS Cognitive API Demo ===\n");

    // ========================================================================
    // Layer 1: Thermodynamic Computing
    // ========================================================================
    println!("Layer 1: Thermodynamic Computing");
    println!("Critical temperature (Tc) = {}", ISING_CRITICAL_TEMP);

    set_temperature(ISING_CRITICAL_TEMP)?;
    let energy = get_energy()?;
    println!("Energy: {:.4}", energy);

    let phase = get_phase(ISING_CRITICAL_TEMP);
    println!("Phase: {:?}", phase);

    let weight = boltzmann_weight(1.0, ISING_CRITICAL_TEMP);
    println!("Boltzmann weight: {:.6}\n", weight);

    // ========================================================================
    // Layer 2: Cognitive Processing
    // ========================================================================
    println!("Layer 2: Cognitive Processing");

    // Attention allocation
    let saliences = vec![0.5, 0.3, 0.9, 0.2];
    let attention = focus_attention(&saliences)?;
    println!("Attention weights: {:?}", attention);

    // Working memory
    let mut memory = WorkingMemory::new();
    println!("Working memory capacity: {}", WORKING_MEMORY_CAPACITY);

    // Pattern recognition
    let input = vec![0.5, 0.3, 0.8];
    let templates = vec![
        ("pattern_A", vec![0.5, 0.3, 0.9]),
        ("pattern_B", vec![0.1, 0.2, 0.3]),
    ];
    let matches = recognize_pattern(&input, &templates)?;
    println!("Best match: {} (similarity: {:.3})\n",
             matches[0].pattern_id, matches[0].similarity);

    // ========================================================================
    // Layer 3: Decision Making (Active Inference)
    // ========================================================================
    println!("Layer 3: Decision Making (Active Inference)");

    let states = vec!["explore".to_string(), "exploit".to_string()];
    let beliefs = BeliefState::uniform(states);
    println!("Belief entropy: {:.4}", beliefs.entropy());

    let preferences = Preferences {
        observations: std::collections::HashMap::new(),
        precision: 1.0,
    };

    let action = select_action(&beliefs, &preferences)?;
    println!("Selected action: {}\n", action);

    // ========================================================================
    // Layer 4: Learning (STDP)
    // ========================================================================
    println!("Layer 4: Learning (STDP)");
    println!("Fibonacci tau: {}", FIBONACCI_TAU);

    // Pre fires 5ms before post → potentiation
    let dw_potentiation = apply_stdp(5.0, 1.0, DEFAULT_LEARNING_RATE)?;
    println!("STDP potentiation (Δt=+5ms): {:.6}", dw_potentiation);

    // Post fires 5ms before pre → depression
    let dw_depression = apply_stdp(-5.0, 1.0, DEFAULT_LEARNING_RATE)?;
    println!("STDP depression (Δt=-5ms): {:.6}", dw_depression);

    // Eligibility trace
    let trace = eligibility_trace(10.0, FIBONACCI_TAU);
    println!("Eligibility trace (t=10ms): {:.6}\n", trace);

    // ========================================================================
    // Layer 5: Collective Intelligence
    // ========================================================================
    println!("Layer 5: Collective Intelligence");

    let mut swarm = SwarmState::new();
    println!("Swarm size: {}", swarm.size());

    // Swarm cohesion
    let agent_states = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.05, 0.05],
    ];
    let cohesion = swarm_cohesion(&agent_states);
    println!("Swarm cohesion: {:.3}\n", cohesion);

    // ========================================================================
    // Layer 6: Consciousness (IIT)
    // ========================================================================
    println!("Layer 6: Consciousness (IIT)");
    println!("Φ threshold for consciousness: {}", PHI_THRESHOLD);

    let network = NeuralState {
        activations: vec![1.0, 0.5, 0.8, 0.3],
        connectivity: vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ],
        labels: vec!["n1".to_string(), "n2".to_string(), "n3".to_string(), "n4".to_string()],
    };

    let phi_result = compute_phi(&network)?;
    println!("Φ (integrated information): {:.4}", phi_result.phi);
    println!("Is conscious: {}", is_conscious(phi_result.phi));
    println!("Consciousness level: {}\n", consciousness_level(phi_result.phi));

    // ========================================================================
    // Layer 7: Meta-Cognition
    // ========================================================================
    println!("Layer 7: Meta-Cognition");

    let mut self_model = get_self_model()?;
    self_model.add_capability("pattern_recognition");
    self_model.add_capability("decision_making");
    println!("Capabilities: {:?}", self_model.capabilities);

    let report = introspect()?;
    println!("System health: {:.2}", report.health);
    println!("Confidence: {:.2}", report.confidence);

    // Confidence calibration
    let predictions = vec![0.9, 0.8, 0.6, 0.3];
    let outcomes = vec![1.0, 1.0, 0.0, 0.0];
    let ece = calibrate_confidence(&predictions, &outcomes)?;
    println!("Expected Calibration Error: {:.4}", ece);
    println!("Well calibrated: {}\n", is_well_calibrated(ece));

    // ========================================================================
    // Layer 8: Integration (Homeostasis & Orchestration)
    // ========================================================================
    println!("Layer 8: Integration (Homeostasis & Orchestration)");

    let homeostasis = get_homeostasis()?;
    println!("Homeostatic variables: {}", N_HOMEOSTATIC_VARS);
    println!("Energy setpoint: {:.2}", homeostasis.setpoints[0]);
    println!("Temperature setpoint: {:.3}", homeostasis.setpoints[1]);

    // Full cognitive cycle
    let input = SensoryInput {
        visual: vec![0.5, 0.3, 0.8],
        auditory: vec![0.2],
        proprioceptive: vec![0.7],
        timestamp: 0.0,
    };

    let output = cognitive_cycle(&input)?;
    println!("\nCognitive Cycle Output:");
    println!("  Action: {}", output.action);
    println!("  Confidence: {:.2}", output.confidence);
    println!("  Energy: {:.2}", output.internal_state.energy);
    println!("  Temperature: {:.3}", output.internal_state.temperature);

    // System health
    let health = system_health()?;
    println!("\nSystem Health Report:");
    println!("  Health score: {:.2}", health.health_score);
    println!("  Φ: {:.2}", health.phi);
    println!("  Stability: {:.2}", health.homeostatic_stability);
    println!("  Issues: {}", health.issues.len());

    println!("\n=== Demo Complete ===");
    Ok(())
}
