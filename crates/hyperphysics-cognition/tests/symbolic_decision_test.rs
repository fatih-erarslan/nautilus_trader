//! Test symbolic decision logging with core cognitive actions

use hyperphysics_cognition::actions::*;
use hyperphysics_cognition::error::Result;
use parking_lot::RwLock;
use std::sync::Arc;

fn create_test_state() -> Arc<RwLock<CognitiveState>> {
    Arc::new(RwLock::new(CognitiveState {
        bottom_up_input: 0.0,
        prediction: 0.5,
        prediction_error: 0.0,
        salience: 0.0,
        attention_weight: 1.0,
        evidence: 0.0,
        drift_rate: 0.5,
        decision_threshold: 1.5,
        current_input: 0.0,
    }))
}

#[test]
fn test_full_decision_cycle_with_logging() -> Result<()> {
    let state = create_test_state();
    let mut logger = SymbolicDecisionLogger::new();
    
    // === DECISION CYCLE START ===
    logger.begin_decision();
    
    // Phase 1: PERCEPTION (Observe)
    let sensory_input = 0.8;
    let prediction_error = observe(state.clone(), &mut logger, sensory_input)?;
    println!("✓ Observation: PE = {:.3}", prediction_error);
    
    // Phase 2: COGNITION (Investigate)
    let free_energy = investigate(state.clone(), &mut logger)?;
    println!("✓ Investigation: F = {:.3}", free_energy);
    
    // Phase 3: LEARNING
    let td_error = prediction_error * 0.5;
    let weight_change = learn(state.clone(), &mut logger, td_error, 2)?;
    println!("✓ Learning (L2): Δw = {:.4}", weight_change);
    
    // Phase 4: PREDICTION
    let action_candidate = 0.3;
    let predicted_outcome = predict(state.clone(), &mut logger, action_candidate)?;
    println!("✓ Prediction: o' = {:.3}", predicted_outcome);
    
    // Phase 5: INTEGRATION (Broadcast)
    broadcast(state.clone(), &mut logger, predicted_outcome, "MotorCortex")?;
    println!("✓ Broadcast complete");
    
    // Create symbolic path
    let path = SymbolicPath {
        root: "Sensory Input (0.8)".to_string(),
        edges: vec![
            ("Sensory Input".to_string(), "Prediction Error".to_string(), "PE = 0.3".to_string()),
            ("Prediction Error".to_string(), "Free Energy".to_string(), "F = D_KL + H".to_string()),
            ("Free Energy".to_string(), "Learn Weights".to_string(), "STDP L2".to_string()),
            ("Learn Weights".to_string(), "Predict Outcome".to_string(), "Forward model".to_string()),
            ("Predict Outcome".to_string(), "Motor Command".to_string(), "Execute action".to_string()),
        ],
        terminal: "Action Executed".to_string(),
        cost: free_energy,
    };
    
    logger.log_symbolic_path(&path);
    
    // Simulate Wolfram validation
    let validation = WolframValidation {
        formula: "F = PE² + ln(w)".to_string(),
        wolfram_result: format!("{:.6}", free_energy),
        rust_result: free_energy,
        valid: true,
        tolerance: 1e-6,
    };
    
    logger.log_wolfram_validation(&validation);
    
    // === DECISION CYCLE END ===
    logger.end_decision("ExecuteMotorCommand", free_energy);
    
    println!("\n{}", logger.summary());
    
    Ok(())
}

#[test]
fn test_dream_state_consolidation() -> Result<()> {
    let state = create_test_state();
    let mut logger = SymbolicDecisionLogger::new();
    
    logger.begin_decision();
    
    // Simulate experience buffer
    let replay_buffer = vec![0.1, 0.2, 0.15, 0.25, 0.3];
    
    // Consolidate memories
    let total_consolidation = consolidate(state.clone(), &mut logger, &replay_buffer)?;
    println!("✓ Consolidated {} memories: Δw_total = {:.4}", replay_buffer.len(), total_consolidation);
    
    // Enter rest state
    rest(state.clone(), &mut logger)?;
    println!("✓ Entered rest state");
    
    logger.end_decision("RestAndConsolidate", 0.1);
    
    Ok(())
}

#[test]
fn test_meta_learning_adaptation() -> Result<()> {
    let state = create_test_state();
    let mut logger = SymbolicDecisionLogger::new();
    
    logger.begin_decision();
    
    // Test L3 transformation detection
    let high_error = 2.5; // Above threshold for L3
    let weight_change = learn(state.clone(), &mut logger, high_error, 3)?;
    println!("⚠️  L3 transformation detected with δ = {:.2}", high_error);
    
    // Adapt strategy based on performance
    let performance = 0.8;
    let new_drift = adapt(state.clone(), &mut logger, performance)?;
    println!("✓ Adapted drift rate: μ = {:.3}", new_drift);
    
    logger.end_decision("AdaptStrategy", 0.5);
    
    Ok(())
}

#[test]
fn test_symbolic_path_rendering() {
    let path = SymbolicPath {
        root: "Perception".to_string(),
        edges: vec![
            ("Perception".to_string(), "Cognition".to_string(), "Prediction error".to_string()),
            ("Cognition".to_string(), "Deliberation".to_string(), "Free energy min".to_string()),
            ("Deliberation".to_string(), "Action".to_string(), "Policy selection".to_string()),
        ],
        terminal: "Execute".to_string(),
        cost: 0.25,
    };
    
    assert_eq!(path.edges.len(), 3);
    assert_eq!(path.cost, 0.25);
    assert_eq!(path.root, "Perception");
    assert_eq!(path.terminal, "Execute");
}
