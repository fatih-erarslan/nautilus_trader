//! Basic usage example for PADS connector

use pads_connector::*;
use std::time::Duration;
use tracing::{info, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    info!("Starting PADS connector example");
    
    // Create configuration
    let mut config = PadsConfig::default();
    
    // Customize scale parameters
    config.scale_config.micro_scale.time_horizon_ms = 50;
    config.scale_config.micro_scale.exploitation_weight = 0.9;
    config.scale_config.meso_scale.balance_factor = 0.5;
    config.scale_config.macro_scale.exploration_weight = 0.8;
    
    // Adjust routing settings
    config.routing_config.max_queue_size = 5000;
    config.routing_config.load_balancing = LoadBalancingStrategy::Adaptive;
    
    // Configure resilience
    config.resilience_config.circuit_breaker.failure_threshold = 3;
    config.resilience_config.recovery_strategies.auto_recovery = true;
    
    // Initialize PADS connector
    let pads = PadsConnector::new(config).await?;
    pads.initialize().await?;
    
    info!("PADS connector initialized successfully");
    
    // Example 1: Process a high-urgency micro-scale decision
    example_micro_scale_decision(&pads).await?;
    
    // Example 2: Process a balanced meso-scale decision
    example_meso_scale_decision(&pads).await?;
    
    // Example 3: Process a strategic macro-scale decision
    example_macro_scale_decision(&pads).await?;
    
    // Example 4: Trigger scale transition
    example_scale_transition(&pads).await?;
    
    // Example 5: Test resilience
    example_resilience_test(&pads).await?;
    
    // Get system status
    let status = pads.get_status().await?;
    info!("System status: {:?}", status);
    
    Ok(())
}

async fn example_micro_scale_decision(pads: &PadsConnector) -> Result<()> {
    info!("Example 1: High-urgency micro-scale decision");
    
    let decision = PanarchyDecision {
        id: "micro-001".to_string(),
        timestamp: chrono::Utc::now(),
        context: create_context(0.2, 0.8, AdaptiveCyclePhase::Growth),
        objectives: vec![
            Objective {
                name: "quick_profit".to_string(),
                weight: 0.8,
                target_value: 0.01,
                optimization_direction: OptimizationDirection::Maximize,
            },
        ],
        constraints: vec![
            Constraint {
                name: "max_risk".to_string(),
                constraint_type: ConstraintType::LessThan,
                value: 0.05,
            },
        ],
        urgency: 0.9,  // High urgency
        impact: 0.3,   // Low impact
        uncertainty: 0.1, // Low uncertainty
    };
    
    let result = pads.process_decision(decision).await?;
    
    info!("Micro-scale result: scale={:?}, success={}, latency={}ms",
          result.scale_level, result.success, result.metrics.processing_time_ms);
    
    Ok(())
}

async fn example_meso_scale_decision(pads: &PadsConnector) -> Result<()> {
    info!("Example 2: Balanced meso-scale decision");
    
    let decision = PanarchyDecision {
        id: "meso-001".to_string(),
        timestamp: chrono::Utc::now(),
        context: create_context(0.5, 0.5, AdaptiveCyclePhase::Conservation),
        objectives: vec![
            Objective {
                name: "balance_portfolio".to_string(),
                weight: 0.5,
                target_value: 0.0,
                optimization_direction: OptimizationDirection::Target,
            },
            Objective {
                name: "manage_risk".to_string(),
                weight: 0.5,
                target_value: 0.1,
                optimization_direction: OptimizationDirection::Minimize,
            },
        ],
        constraints: vec![],
        urgency: 0.5,
        impact: 0.5,
        uncertainty: 0.5,
    };
    
    let result = pads.process_decision(decision).await?;
    
    info!("Meso-scale result: scale={:?}, confidence={}",
          result.scale_level, result.metrics.confidence_score);
    
    // Check for cross-scale effects
    if !result.cross_scale_effects.upward_effects.is_empty() {
        info!("Upward effects detected: {:?}", result.cross_scale_effects.upward_effects);
    }
    if !result.cross_scale_effects.downward_effects.is_empty() {
        info!("Downward effects detected: {:?}", result.cross_scale_effects.downward_effects);
    }
    
    Ok(())
}

async fn example_macro_scale_decision(pads: &PadsConnector) -> Result<()> {
    info!("Example 3: Strategic macro-scale decision");
    
    let decision = PanarchyDecision {
        id: "macro-001".to_string(),
        timestamp: chrono::Utc::now(),
        context: create_context(0.8, 0.3, AdaptiveCyclePhase::Reorganization),
        objectives: vec![
            Objective {
                name: "strategic_positioning".to_string(),
                weight: 0.7,
                target_value: 0.0,
                optimization_direction: OptimizationDirection::Maximize,
            },
            Objective {
                name: "explore_opportunities".to_string(),
                weight: 0.3,
                target_value: 0.0,
                optimization_direction: OptimizationDirection::Maximize,
            },
        ],
        constraints: vec![],
        urgency: 0.2,  // Low urgency
        impact: 0.8,   // High impact
        uncertainty: 0.7, // High uncertainty
    };
    
    let result = pads.process_decision(decision).await?;
    
    info!("Macro-scale result: scale={:?}, adaptation_rate={}",
          result.scale_level, result.metrics.adaptation_rate);
    
    Ok(())
}

async fn example_scale_transition(pads: &PadsConnector) -> Result<()> {
    info!("Example 4: Triggering scale transition");
    
    // Process multiple decisions to trigger transition
    for i in 0..5 {
        let decision = PanarchyDecision {
            id: format!("transition-{}", i),
            timestamp: chrono::Utc::now(),
            context: create_context(0.9, 0.2, AdaptiveCyclePhase::Release),
            objectives: vec![
                Objective {
                    name: "adapt_strategy".to_string(),
                    weight: 1.0,
                    target_value: 0.0,
                    optimization_direction: OptimizationDirection::Maximize,
                },
            ],
            constraints: vec![],
            urgency: 0.1 + (i as f64 * 0.2), // Increasing urgency
            impact: 0.6,
            uncertainty: 0.8 - (i as f64 * 0.1), // Decreasing uncertainty
        };
        
        let result = pads.process_decision(decision).await?;
        info!("Decision {} processed at scale {:?}", i, result.scale_level);
        
        // Check for transition actions
        for action in &result.actions {
            if let ActionType::Transition(transition) = &action.action_type {
                info!("Scale transition detected: {:?} -> {:?}",
                      transition.from_scale, transition.to_scale);
            }
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    Ok(())
}

async fn example_resilience_test(pads: &PadsConnector) -> Result<()> {
    info!("Example 5: Testing resilience mechanisms");
    
    // Simulate failures
    for i in 0..10 {
        let decision = PanarchyDecision {
            id: format!("resilience-{}", i),
            timestamp: chrono::Utc::now(),
            context: create_context(0.1, 0.9, AdaptiveCyclePhase::Growth),
            objectives: vec![
                Objective {
                    name: "stress_test".to_string(),
                    weight: 1.0,
                    target_value: 0.0,
                    optimization_direction: OptimizationDirection::Maximize,
                },
            ],
            constraints: vec![
                Constraint {
                    name: "impossible_constraint".to_string(),
                    constraint_type: ConstraintType::Equal,
                    value: -1.0, // Impossible to satisfy
                },
            ],
            urgency: 1.0,
            impact: 1.0,
            uncertainty: 1.0,
        };
        
        match pads.process_decision(decision).await {
            Ok(result) => {
                if !result.success {
                    info!("Decision failed as expected: {:?}", result.errors);
                }
            }
            Err(e) => {
                error!("Error processing decision: {}", e);
            }
        }
    }
    
    // Test recovery
    info!("Testing recovery mechanism");
    pads.recover().await?;
    
    // Verify system is operational
    let status = pads.get_status().await?;
    info!("Post-recovery status: health_score={}, circuit_breaker={:?}",
          status.resilience_status.health_score,
          status.resilience_status.circuit_breaker_state);
    
    Ok(())
}

fn create_context(volatility: f64, success_rate: f64, phase: AdaptiveCyclePhase) -> DecisionContext {
    DecisionContext {
        market_state: MarketContext {
            volatility,
            trend_strength: 0.5,
            liquidity: 0.8,
            regime: "testing".to_string(),
        },
        system_state: SystemContext {
            resource_utilization: 0.5,
            active_scales: vec![ScaleLevel::Micro, ScaleLevel::Meso],
            current_phase: phase,
        },
        historical_performance: PerformanceContext {
            recent_success_rate: success_rate,
            adaptive_capacity_used: 0.4,
            resilience_score: 0.7,
        },
    }
}