//! # PADS Integration Tests
//!
//! Comprehensive integration tests for the Panarchy Adaptive Decision System.

use std::time::Duration;
use tokio::time::sleep;
use swarm_intelligence::pads::{
    PadsSystem, PadsConfig, DecisionContext, DecisionLayer, AdaptiveCyclePhase,
    init_pads, init_pads_with_config
};

#[tokio::test]
async fn test_pads_system_lifecycle() {
    // Test complete PADS system lifecycle
    let config = PadsConfig::builder()
        .with_system_id("test-pads-001".to_string())
        .with_decision_layers(4)
        .with_adaptive_cycles(true)
        .with_real_time_monitoring(true)
        .with_thread_pool_size(4)
        .build();
    
    // Initialize PADS system
    let mut pads = PadsSystem::new(config).await.expect("Failed to create PADS system");
    
    // Check initial health
    assert!(pads.is_healthy().await);
    
    // Start the system
    pads.start().await.expect("Failed to start PADS system");
    
    // Verify system is running
    assert!(pads.is_healthy().await);
    
    // Stop the system
    pads.stop().await.expect("Failed to stop PADS system");
}

#[tokio::test]
async fn test_pads_decision_making() {
    // Test decision-making capabilities
    let pads = init_pads().await.expect("Failed to initialize PADS");
    
    // Create decision context
    let context = DecisionContext::new(
        "decision-001".to_string(),
        DecisionLayer::Tactical,
        AdaptiveCyclePhase::Growth,
    );
    
    // Make a decision
    let response = pads.make_decision(context).await.expect("Failed to make decision");
    
    // Verify response
    assert!(!response.action.is_empty());
    assert!(response.confidence > 0.0);
    assert!(response.confidence <= 1.0);
    assert!(!response.reasoning.is_empty());
}

#[tokio::test]
async fn test_multiple_decision_layers() {
    // Test decision-making across all layers
    let pads = init_pads().await.expect("Failed to initialize PADS");
    
    let layers = vec![
        DecisionLayer::Tactical,
        DecisionLayer::Operational,
        DecisionLayer::Strategic,
        DecisionLayer::MetaStrategic,
    ];
    
    for layer in layers {
        let context = DecisionContext::new(
            format!("decision-{:?}", layer),
            layer,
            AdaptiveCyclePhase::Growth,
        );
        
        let response = pads.make_decision(context).await
            .expect(&format!("Failed to make decision for layer {:?}", layer));
        
        assert_eq!(response.layer, layer);
        assert!(response.confidence > 0.0);
    }
}

#[tokio::test]
async fn test_adaptive_cycle_phases() {
    // Test decision-making across all adaptive cycle phases
    let pads = init_pads().await.expect("Failed to initialize PADS");
    
    let phases = vec![
        AdaptiveCyclePhase::Growth,
        AdaptiveCyclePhase::Conservation,
        AdaptiveCyclePhase::Release,
        AdaptiveCyclePhase::Reorganization,
    ];
    
    for phase in phases {
        let context = DecisionContext::new(
            format!("decision-{:?}", phase),
            DecisionLayer::Operational,
            phase,
        );
        
        let response = pads.make_decision(context).await
            .expect(&format!("Failed to make decision for phase {:?}", phase));
        
        assert!(!response.action.is_empty());
        assert!(response.reasoning.len() > 0);
    }
}

#[tokio::test]
async fn test_pads_with_custom_config() {
    // Test PADS with custom configuration
    let config = PadsConfig::builder()
        .with_system_id("custom-pads-test".to_string())
        .with_decision_layers(2) // Only tactical and operational
        .with_adaptive_cycles(true)
        .with_real_time_monitoring(false) // Disable monitoring for this test
        .with_thread_pool_size(2)
        .build();
    
    let pads = init_pads_with_config(config).await
        .expect("Failed to initialize PADS with custom config");
    
    assert!(pads.is_healthy().await);
    
    // Test that configured layers work
    let tactical_context = DecisionContext::new(
        "tactical-test".to_string(),
        DecisionLayer::Tactical,
        AdaptiveCyclePhase::Growth,
    );
    
    let tactical_response = pads.make_decision(tactical_context).await;
    assert!(tactical_response.is_ok());
}

#[tokio::test]
async fn test_pads_state_management() {
    // Test PADS state management
    let pads = init_pads().await.expect("Failed to initialize PADS");
    
    // Get initial state
    let initial_state = pads.get_state().await;
    assert!(initial_state.health.is_operational());
    
    // Get initial metrics
    let initial_metrics = pads.get_metrics().await;
    
    // Make a decision to change state
    let context = DecisionContext::new(
        "state-test-001".to_string(),
        DecisionLayer::Operational,
        AdaptiveCyclePhase::Growth,
    );
    
    let _response = pads.make_decision(context).await
        .expect("Failed to make decision");
    
    // Get updated metrics
    let updated_metrics = pads.get_metrics().await;
    
    // Verify metrics changed (decision count should increase)
    if let (Some(initial_count), Some(updated_count)) = (
        initial_metrics.get("decisions_processed"),
        updated_metrics.get("decisions_processed")
    ) {
        assert!(updated_count > initial_count);
    }
}

#[tokio::test]
async fn test_pads_error_handling() {
    // Test PADS error handling with invalid contexts
    let pads = init_pads().await.expect("Failed to initialize PADS");
    
    // Create context with very short timeout (should expire)
    let mut context = DecisionContext::new(
        "timeout-test".to_string(),
        DecisionLayer::Tactical,
        AdaptiveCyclePhase::Growth,
    );
    context.time_budget = Duration::from_millis(1); // Very short timeout
    
    // Wait for context to expire
    sleep(Duration::from_millis(10)).await;
    
    // Try to make decision with expired context
    let result = pads.make_decision(context).await;
    assert!(result.is_err()); // Should fail due to expired context
}

#[tokio::test]
async fn test_decision_context_validity() {
    // Test decision context validity checks
    let context = DecisionContext::new(
        "validity-test".to_string(),
        DecisionLayer::Strategic,
        AdaptiveCyclePhase::Conservation,
    );
    
    // Context should be valid initially
    assert!(context.is_valid());
    
    // Remaining time should be positive
    assert!(context.remaining_time() > Duration::from_secs(0));
    
    // Check time budget matches layer
    assert_eq!(context.time_budget, DecisionLayer::Strategic.time_horizon());
}

#[tokio::test]
async fn test_decision_layer_properties() {
    // Test decision layer properties
    assert!(DecisionLayer::Tactical.priority() > DecisionLayer::Operational.priority());
    assert!(DecisionLayer::Operational.priority() > DecisionLayer::Strategic.priority());
    assert!(DecisionLayer::Strategic.priority() > DecisionLayer::MetaStrategic.priority());
    
    // Check time horizons
    assert!(DecisionLayer::Tactical.time_horizon() < DecisionLayer::Operational.time_horizon());
    assert!(DecisionLayer::Operational.time_horizon() < DecisionLayer::Strategic.time_horizon());
    assert!(DecisionLayer::Strategic.time_horizon() < DecisionLayer::MetaStrategic.time_horizon());
}

#[tokio::test]
async fn test_adaptive_cycle_transitions() {
    // Test adaptive cycle phase transitions
    let growth = AdaptiveCyclePhase::Growth;
    let conservation = AdaptiveCyclePhase::Conservation;
    let release = AdaptiveCyclePhase::Release;
    let reorganization = AdaptiveCyclePhase::Reorganization;
    
    // Test phase transitions
    assert_eq!(growth.next_phase(), conservation);
    assert_eq!(conservation.next_phase(), release);
    assert_eq!(release.next_phase(), reorganization);
    assert_eq!(reorganization.next_phase(), growth);
    
    // Test phase characteristics
    let growth_chars = growth.characteristics();
    let conservation_chars = conservation.characteristics();
    
    // Growth should have higher potential than conservation
    assert!(growth_chars.potential > conservation_chars.potential);
    
    // Conservation should have higher connectedness than growth
    assert!(conservation_chars.connectedness > growth_chars.connectedness);
}

#[tokio::test] 
async fn test_pads_integration_basic() {
    // Basic integration test to ensure all components work together
    let config = PadsConfig::default();
    
    // Validate config
    assert!(config.validate().is_ok());
    
    // Create PADS system
    let mut pads = PadsSystem::new(config).await
        .expect("Failed to create PADS system");
    
    // Start system
    pads.start().await.expect("Failed to start PADS");
    
    // Check health
    assert!(pads.is_healthy().await);
    
    // Make a simple decision
    let context = DecisionContext::new(
        "integration-test".to_string(),
        DecisionLayer::Operational,
        AdaptiveCyclePhase::Growth,
    );
    
    let response = pads.make_decision(context).await
        .expect("Failed to make decision in integration test");
    
    // Verify response structure
    assert!(!response.decision_id.is_empty());
    assert!(!response.action.is_empty());
    assert!(response.confidence > 0.0);
    assert!(!response.reasoning.is_empty());
    
    // Stop system
    pads.stop().await.expect("Failed to stop PADS");
}