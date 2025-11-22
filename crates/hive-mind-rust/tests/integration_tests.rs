//! Integration tests for the Hive Mind system

use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use hive_mind_rust::{
    config::HiveMindConfig,
    core::{HiveMind, HiveMindBuilder, OperationalMode, HealthStatus},
    error::Result,
};

/// Test basic system startup and shutdown
#[tokio::test]
async fn test_system_lifecycle() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    // Test startup
    hive_mind.start().await?;
    
    let state = hive_mind.get_state().await;
    assert!(state.is_running);
    assert_eq!(state.mode, OperationalMode::Normal);
    
    // Test shutdown
    hive_mind.stop().await?;
    
    let state = hive_mind.get_state().await;
    assert!(!state.is_running);
    
    Ok(())
}

/// Test fault tolerance and recovery
#[tokio::test]
async fn test_fault_tolerance() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Wait for system to stabilize
    sleep(Duration::from_secs(2)).await;
    
    // Test emergency shutdown
    hive_mind.emergency_shutdown().await?;
    
    let state = hive_mind.get_state().await;
    assert!(!state.is_running);
    assert_eq!(state.mode, OperationalMode::Emergency);
    
    Ok(())
}

/// Test consensus proposal submission
#[tokio::test]
async fn test_consensus_proposal() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Wait for consensus engine to be ready
    sleep(Duration::from_secs(1)).await;
    
    let proposal = serde_json::json!({
        "action": "test_trade",
        "symbol": "BTC/USDT",
        "quantity": 1.0
    });
    
    let proposal_id = hive_mind.submit_proposal(proposal).await?;
    assert!(!proposal_id.is_nil());
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test collective memory operations
#[tokio::test]
async fn test_collective_memory() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Store knowledge
    let test_data = serde_json::json!({
        "trend": "bullish",
        "confidence": 0.85,
        "timestamp": chrono::Utc::now()
    });
    
    hive_mind.store_knowledge("test_market_analysis", test_data).await?;
    
    // Query knowledge
    let results = hive_mind.query_memory("test_market_analysis").await?;
    assert!(!results.is_empty());
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test neural pattern recognition
#[tokio::test]
async fn test_neural_insights() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Test neural insights
    let market_data = vec![100.0, 101.5, 102.0, 101.8, 103.2, 102.5, 104.0];
    let insights = hive_mind.get_neural_insights(&market_data).await?;
    
    // Should return some form of analysis
    assert!(insights.is_object());
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test agent management
#[tokio::test]
async fn test_agent_management() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Spawn an agent
    let capabilities = vec![
        "market_analysis".to_string(),
        "risk_assessment".to_string(),
    ];
    
    let agent_id = hive_mind.spawn_agent(capabilities).await?;
    assert!(!agent_id.is_nil());
    
    // Get active agents
    let active_agents = hive_mind.get_active_agents().await?;
    assert!(active_agents.contains(&agent_id));
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test system health monitoring
#[tokio::test]
async fn test_health_monitoring() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Wait for health monitoring to kick in
    sleep(Duration::from_secs(3)).await;
    
    let state = hive_mind.get_state().await;
    
    // Check that health monitoring is working
    assert!(state.last_health_check.elapsed() < Duration::from_secs(60));
    
    // System should be healthy after startup
    assert_eq!(state.health.overall_status, HealthStatus::Healthy);
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test system recovery
#[tokio::test]
async fn test_system_recovery() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Test recovery from previous state
    hive_mind.recover_from_previous_state().await?;
    
    let state = hive_mind.get_state().await;
    assert!(state.is_running);
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test performance under load
#[tokio::test]
async fn test_performance_load() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Submit multiple proposals concurrently
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let hive_mind_clone = &hive_mind;
        let handle = tokio::spawn(async move {
            let proposal = serde_json::json!({
                "action": "load_test",
                "id": i,
                "timestamp": chrono::Utc::now()
            });
            
            hive_mind_clone.submit_proposal(proposal).await
        });
        handles.push(handle);
    }
    
    // Wait for all proposals to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test configuration validation
#[tokio::test]
async fn test_configuration_validation() -> Result<()> {
    // Test with default configuration
    let config = HiveMindConfig::default();
    assert!(config.validate().is_ok());
    
    // Test building with valid configuration
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    assert!(hive_mind.start().await.is_ok());
    
    hive_mind.stop().await?;
    Ok(())
}

/// Test integration with trading engine simulation
#[tokio::test]
async fn test_trading_integration() -> Result<()> {
    let config = HiveMindConfig::default();
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    
    hive_mind.start().await?;
    
    // Simulate trading decision flow
    let market_data = vec![100.0, 99.5, 98.0, 99.0, 101.0];
    let insights = hive_mind.get_neural_insights(&market_data).await?;
    
    // Store market analysis
    hive_mind.store_knowledge("btc_analysis", insights.clone()).await?;
    
    // Submit trading proposal based on insights
    let proposal = serde_json::json!({
        "action": "trade",
        "symbol": "BTC/USDT",
        "decision": insights,
        "timestamp": chrono::Utc::now()
    });
    
    let proposal_id = hive_mind.submit_proposal(proposal).await?;
    assert!(!proposal_id.is_nil());
    
    hive_mind.stop().await?;
    Ok(())
}