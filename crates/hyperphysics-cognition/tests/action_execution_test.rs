//! Integration tests for cognitive action execution

use hyperphysics_cognition::actions::*;
use hyperphysics_cognition::error::Result;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;

fn create_test_state() -> Arc<RwLock<CognitiveState>> {
    Arc::new(RwLock::new(CognitiveState {
        bottom_up_input: 1.0,
        prediction: 0.8,
        prediction_error: 0.0,
        salience: 0.0,
        attention_weight: 1.0,
        evidence: 0.0,
        drift_rate: 0.5,
        decision_threshold: 1.5,
        current_input: 0.5,
    }))
}

#[test]
fn test_predict_sensory() -> Result<()> {
    let state = create_test_state();
    let executor = ActionExecutor::new(state.clone());

    let result = executor.execute(CognitiveAction::PredictSensory)?;
    
    assert!(result > Duration::ZERO);
    assert_eq!(CognitiveAction::PredictSensory.domain(), CognitiveDomain::Perception);

    Ok(())
}

#[test]
fn test_compute_prediction_error() -> Result<()> {
    let state = create_test_state();
    let executor = ActionExecutor::new(state.clone());

    let result = executor.execute(CognitiveAction::ComputePredictionError)?;
    
    assert!(result > Duration::ZERO);
    assert_eq!(CognitiveAction::ComputePredictionError.domain(), CognitiveDomain::Perception);
    
    // Check that prediction error was computed
    let state_lock = state.read();
    assert_eq!(state_lock.prediction_error, state_lock.bottom_up_input - state_lock.prediction);

    Ok(())
}

#[test]
fn test_detect_salience() -> Result<()> {
    let state = create_test_state();
    let executor = ActionExecutor::new(state.clone());

    let result = executor.execute(CognitiveAction::DetectSalience)?;
    
    assert!(result > Duration::ZERO);
    assert_eq!(CognitiveAction::DetectSalience.domain(), CognitiveDomain::Attention);
    
    // Check that salience was computed
    let state_lock = state.read();
    assert!(state_lock.salience > 0.0);

    Ok(())
}

#[test]
fn test_accumulate_evidence() -> Result<()> {
    let state = create_test_state();
    let executor = ActionExecutor::new(state.clone());

    let initial_evidence = state.read().evidence;
    
    let result = executor.execute(CognitiveAction::AccumulateEvidence)?;
    
    assert!(result > Duration::ZERO);
    assert_eq!(CognitiveAction::AccumulateEvidence.domain(), CognitiveDomain::Decision);
    
    // Check that evidence accumulated
    let final_evidence = state.read().evidence;
    assert!(final_evidence > initial_evidence);

    Ok(())
}

#[test]
fn test_domain_classification() {
    assert_eq!(CognitiveAction::PredictSensory.domain(), CognitiveDomain::Perception);
    assert_eq!(CognitiveAction::GroundSymbol.domain(), CognitiveDomain::Cognition);
    assert_eq!(CognitiveAction::AppraiseEvent.domain(), CognitiveDomain::Emotion);
    assert_eq!(CognitiveAction::SeparatePatterns.domain(), CognitiveDomain::Memory);
    assert_eq!(CognitiveAction::DetectSalience.domain(), CognitiveDomain::Attention);
    assert_eq!(CognitiveAction::UpdateBelief.domain(), CognitiveDomain::Decision);
    assert_eq!(CognitiveAction::ComputeControl.domain(), CognitiveDomain::Action);
    assert_eq!(CognitiveAction::UpdateTD.domain(), CognitiveDomain::Learning);
}

#[test]
fn test_algorithm_mapping() {
    assert_eq!(
        CognitiveAction::PredictSensory.algorithm(),
        BiomimeticAlgorithm::PredictiveCoding
    );
    assert_eq!(
        CognitiveAction::AccumulateEvidence.algorithm(),
        BiomimeticAlgorithm::DriftDiffusion
    );
    assert_eq!(
        CognitiveAction::UpdateSTDP.algorithm(),
        BiomimeticAlgorithm::STDP
    );
}

#[test]
fn test_temporal_costs() {
    // Perception actions should be fast (< 10μs)
    assert!(CognitiveAction::PredictSensory.temporal_cost_ns() < 10_000);
    assert!(CognitiveAction::ComputePredictionError.temporal_cost_ns() < 10_000);
    
    // Learning actions can be slower (> 10μs)
    assert!(CognitiveAction::UpdateSTDP.temporal_cost_ns() > 10_000);
    assert!(CognitiveAction::UpdateMetaParameters.temporal_cost_ns() > 10_000);
    
    // All should be under 1ms for 40Hz gamma cycle
    assert!(CognitiveAction::UpdateMetaParameters.temporal_cost_ns() < 1_000_000);
}

#[test]
fn test_unimplemented_actions_return_error() {
    let state = create_test_state();
    let executor = ActionExecutor::new(state);

    // These actions are not yet implemented
    let unimplemented = [
        CognitiveAction::GroundSymbol,
        CognitiveAction::AppraiseEvent,
        CognitiveAction::UpdateBelief,
        CognitiveAction::ComputeControl,
    ];

    for action in &unimplemented {
        let result = executor.execute(*action);
        assert!(result.is_err());
    }
}

#[test]
fn test_sequential_action_execution() -> Result<()> {
    let state = create_test_state();
    let executor = ActionExecutor::new(state);

    // Execute a sequence of implemented actions
    executor.execute(CognitiveAction::PredictSensory)?;
    executor.execute(CognitiveAction::ComputePredictionError)?;
    executor.execute(CognitiveAction::DetectSalience)?;
    executor.execute(CognitiveAction::AccumulateEvidence)?;

    Ok(())
}

#[test]
fn test_zero_copy_shared_state() -> Result<()> {
    let state = create_test_state();
    let executor1 = ActionExecutor::new(state.clone());
    let executor2 = ActionExecutor::new(state.clone());

    // Modify state via executor1
    executor1.execute(CognitiveAction::ComputePredictionError)?;
    
    let prediction_error = state.read().prediction_error;
    
    // Verify both executors see the same state
    assert_eq!(state.read().prediction_error, prediction_error);

    // Modify via executor2
    executor2.execute(CognitiveAction::DetectSalience)?;
    
    // Both should see the changes
    assert!(state.read().salience > 0.0);

    Ok(())
}
