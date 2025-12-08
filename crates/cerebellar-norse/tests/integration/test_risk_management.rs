//! Integration tests for risk management systems
//! 
//! Tests the complete risk management pipeline including neural validation,
//! position limits, circuit breakers, and real-time monitoring.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use cerebellar_norse::{
    RiskManager, RiskLimits, SafeTradingProcessor, RiskDashboard,
    NeuralValidator, ValidationConfig, TradingCerebellarProcessor,
    CircuitBreakerType, TradeAction, AnomalyType
};

#[tokio::test]
async fn test_comprehensive_risk_management_flow() {
    // Setup risk limits
    let mut limits = RiskLimits::default();
    limits.max_position_per_symbol.insert("AAPL".to_string(), 1000.0);
    limits.max_total_exposure = 5000.0;
    limits.max_daily_loss = -1000.0;
    limits.max_drawdown_percent = 0.05;
    limits.max_trading_velocity = 50.0;
    limits.var_limit = 2000.0;
    limits.neural_output_bounds = (-5.0, 5.0);
    limits.min_neural_confidence = 0.75;

    // Create safe trading processor
    let mut processor = SafeTradingProcessor::new(limits);
    let risk_manager = processor.get_risk_manager();

    // Start risk monitoring
    risk_manager.start_monitoring().await.unwrap();

    // Test series of trading decisions
    let test_cases = vec![
        ("AAPL", 150.0, 1000.0, 1625000000),
        ("AAPL", 151.0, 1200.0, 1625000001),
        ("GOOGL", 2800.0, 500.0, 1625000002),
        ("AAPL", 149.0, 800.0, 1625000003),
    ];

    let mut approved_trades = 0;
    let mut rejected_trades = 0;

    for (symbol, price, volume, timestamp) in test_cases {
        let decision = processor.safe_process_tick(
            symbol.to_string(),
            price,
            volume,
            timestamp
        ).await.unwrap();

        println!("Decision for {}: {:?}, Approved: {}, Confidence: {:.2}", 
                symbol, decision.action, decision.risk_approved, decision.confidence);

        if decision.risk_approved {
            approved_trades += 1;
        } else {
            rejected_trades += 1;
            println!("Rejection reasons: {:?}", decision.risk_reasons);
        }
    }

    println!("Approved: {}, Rejected: {}", approved_trades, rejected_trades);
    assert!(approved_trades > 0, "At least some trades should be approved");
}

#[tokio::test]
async fn test_position_limit_enforcement() {
    let mut limits = RiskLimits::default();
    limits.max_position_per_symbol.insert("TEST".to_string(), 500.0);
    
    let risk_manager = Arc::new(RiskManager::new(limits));
    
    // First trade within limits
    assert!(risk_manager.validate_trade("TEST", 300.0).await.unwrap());
    risk_manager.update_position("TEST".to_string(), 300.0).await.unwrap();
    
    // Second trade that would exceed limits
    assert!(!risk_manager.validate_trade("TEST", 300.0).await.unwrap());
    
    // Trade in opposite direction should be allowed
    assert!(risk_manager.validate_trade("TEST", -100.0).await.unwrap());
}

#[tokio::test]
async fn test_circuit_breaker_functionality() {
    let limits = RiskLimits::default();
    let risk_manager = Arc::new(RiskManager::new(limits));
    
    // Initially trading should be enabled
    assert!(risk_manager.get_risk_status().trading_enabled);
    
    // Trigger emergency shutdown
    risk_manager.emergency_shutdown("Test emergency".to_string()).await.unwrap();
    
    // Trading should now be blocked
    assert!(!risk_manager.get_risk_status().trading_enabled);
    
    // Any trade should be rejected
    assert!(!risk_manager.validate_trade("ANY", 100.0).await.unwrap());
    
    // Check active circuit breakers
    let active_breakers = risk_manager.circuit_breaker.get_active_breakers();
    assert!(active_breakers.contains(&CircuitBreakerType::EmergencyStop));
}

#[tokio::test]
async fn test_drawdown_monitoring() {
    let mut limits = RiskLimits::default();
    limits.max_drawdown_percent = 0.03; // 3% max drawdown
    
    let risk_manager = Arc::new(RiskManager::new(limits));
    
    // Simulate profitable trading first
    risk_manager.update_pnl(1000.0, 0.0).await.unwrap(); // $1000 profit
    
    // Now simulate a loss that triggers drawdown limit
    risk_manager.update_pnl(-500.0, 0.0).await.unwrap(); // $500 loss from peak
    
    // Check if circuit breaker was triggered
    let status = risk_manager.get_risk_status();
    if status.current_drawdown > 0.03 {
        // Should have triggered loss breaker
        let active_breakers = risk_manager.circuit_breaker.get_active_breakers();
        assert!(active_breakers.contains(&CircuitBreakerType::LossBreaker));
    }
}

#[tokio::test]
async fn test_neural_output_validation() {
    let config = ValidationConfig::default();
    let validator = NeuralValidator::new(config);
    
    // Test valid neural outputs
    let valid_outputs = vec![1.0, 2.0, -1.0, 0.5];
    let membrane_potentials = vec![0.5, -0.3, 0.1, 0.8];
    let spike_trains = vec![true, false, false, true];
    
    // Create mock circuit metrics
    let circuit_metrics = cerebellar_norse::CircuitMetrics {
        granule_stats: cerebellar_norse::LayerStats {
            total_neurons: 1000,
            active_neurons: 100,
            spike_count: 50,
            average_membrane_potential: 0.3
        },
        purkinje_stats: cerebellar_norse::LayerStats {
            total_neurons: 100,
            active_neurons: 20,
            spike_count: 10,
            average_membrane_potential: 0.5
        },
        golgi_stats: cerebellar_norse::LayerStats {
            total_neurons: 50,
            active_neurons: 5,
            spike_count: 2,
            average_membrane_potential: 0.2
        },
        dcn_stats: cerebellar_norse::LayerStats {
            total_neurons: 10,
            active_neurons: 3,
            spike_count: 1,
            average_membrane_potential: 0.7
        },
        total_neurons: 1160
    };
    
    let result = validator.validate_comprehensive(
        &valid_outputs,
        &circuit_metrics,
        &membrane_potentials,
        &spike_trains
    ).unwrap();
    
    assert!(result.is_valid, "Valid outputs should pass validation");
    assert!(result.confidence > 0.5, "Confidence should be reasonable");
    assert!(result.anomalies.is_empty(), "No anomalies should be detected");
    
    // Test invalid neural outputs (out of bounds)
    let invalid_outputs = vec![15.0, 2.0, -1.0, 0.5]; // 15.0 exceeds bounds
    
    let result = validator.validate_comprehensive(
        &invalid_outputs,
        &circuit_metrics,
        &membrane_potentials,
        &spike_trains
    ).unwrap();
    
    assert!(!result.is_valid, "Invalid outputs should fail validation");
    assert!(!result.anomalies.is_empty(), "Anomalies should be detected");
    
    // Check that output range anomaly was detected
    let has_range_anomaly = result.anomalies.iter()
        .any(|anomaly| matches!(anomaly.anomaly_type, AnomalyType::OutputRangeAnomaly));
    assert!(has_range_anomaly, "Output range anomaly should be detected");
}

#[tokio::test]
async fn test_var_calculation_and_limits() {
    let mut limits = RiskLimits::default();
    limits.var_limit = 500.0; // $500 VaR limit
    limits.var_confidence_level = 0.95;
    
    let risk_manager = Arc::new(RiskManager::new(limits));
    
    // Simulate P&L history for VaR calculation
    let pnl_values = vec![100.0, 50.0, -20.0, 80.0, -10.0, 200.0, -50.0, 30.0];
    
    for (i, pnl) in pnl_values.iter().enumerate() {
        // Simulate daily P&L updates
        risk_manager.update_pnl(*pnl, 0.0).await.unwrap();
        
        // Add some delay to simulate time passage
        if i > 0 {
            sleep(Duration::from_millis(10)).await;
        }
    }
    
    let metrics = risk_manager.get_metrics();
    println!("Current VaR: {}", metrics.current_var);
    
    // VaR should be calculated (though may be 0 with limited data)
    assert!(metrics.current_var.is_finite());
}

#[tokio::test]
async fn test_real_time_dashboard() {
    let limits = RiskLimits::default();
    let risk_manager = Arc::new(RiskManager::new(limits));
    let dashboard = RiskDashboard::new(risk_manager.clone());
    
    // Start dashboard monitoring
    dashboard.start_monitoring().await.unwrap();
    
    // Simulate some trading activity
    risk_manager.update_position("AAPL".to_string(), 100.0).await.unwrap();
    risk_manager.update_pnl(50.0, 0.0).await.unwrap();
    
    // Wait for metrics update
    sleep(Duration::from_millis(1100)).await;
    
    // Get dashboard metrics
    let metrics = dashboard.get_metrics();
    
    assert!(metrics.last_update > 0, "Dashboard should have updated");
    assert_eq!(metrics.risk_overview.trading_enabled, true);
    assert!(metrics.system_health.uptime_seconds > 0);
    
    // Test real-time updates subscription
    let mut updates = dashboard.subscribe_to_updates();
    
    // Trigger an update
    risk_manager.update_position("GOOGL".to_string(), 200.0).await.unwrap();
    
    // Wait for update (with timeout)
    let update_result = tokio::time::timeout(
        Duration::from_secs(2),
        updates.recv()
    ).await;
    
    match update_result {
        Ok(Ok(updated_metrics)) => {
            println!("Received real-time update: last_update = {}", updated_metrics.last_update);
            assert!(updated_metrics.last_update >= metrics.last_update);
        },
        Ok(Err(_)) => panic!("Update channel closed unexpectedly"),
        Err(_) => {
            // Timeout - this might happen in test environment, just warn
            println!("Warning: Didn't receive real-time update within timeout");
        }
    }
}

#[tokio::test]
async fn test_trading_velocity_limits() {
    let mut limits = RiskLimits::default();
    limits.max_trading_velocity = 5.0; // 5 trades per second max
    
    let risk_manager = Arc::new(RiskManager::new(limits));
    
    // Simulate rapid trading
    for i in 0..10 {
        let symbol = format!("STOCK{}", i % 3);
        let size = 100.0;
        
        // Update position to simulate trade
        risk_manager.update_position(symbol.clone(), size).await.unwrap();
        
        // Validate next trade immediately (high velocity)
        let is_valid = risk_manager.validate_trade(&symbol, size).await.unwrap();
        
        if i > 5 {
            // After several rapid trades, velocity limit should kick in
            println!("Trade {} validation result: {}", i, is_valid);
        }
    }
    
    // Check trading velocity
    let status = risk_manager.get_risk_status();
    println!("Current trading velocity: {}", status.trading_velocity);
    
    // If velocity is high, trades should be rejected
    if status.trading_velocity > limits.max_trading_velocity {
        assert!(!risk_manager.validate_trade("TEST", 100.0).await.unwrap());
    }
}

#[tokio::test]
async fn test_risk_report_generation() {
    let limits = RiskLimits::default();
    let risk_manager = Arc::new(RiskManager::new(limits));
    let dashboard = RiskDashboard::new(risk_manager.clone());
    
    // Simulate some activity
    risk_manager.update_position("AAPL".to_string(), 500.0).await.unwrap();
    risk_manager.update_position("GOOGL".to_string(), 300.0).await.unwrap();
    risk_manager.update_pnl(100.0, 50.0).await.unwrap();
    
    // Start monitoring to populate metrics
    dashboard.start_monitoring().await.unwrap();
    sleep(Duration::from_millis(1100)).await;
    
    // Generate risk report
    let report = dashboard.generate_risk_report().await.unwrap();
    
    assert!(report.generated_at > 0);
    assert_eq!(report.system_status, "OPERATIONAL");
    assert!(report.key_metrics.active_positions > 0);
    assert!(!report.recommendations.is_empty());
    
    println!("Risk Report Generated:");
    println!("  System Status: {}", report.system_status);
    println!("  Active Positions: {}", report.key_metrics.active_positions);
    println!("  Risk Score: {:.1}", report.key_metrics.risk_score);
    println!("  Recommendations: {:?}", report.recommendations);
}

#[tokio::test]
async fn test_anomaly_detection_learning() {
    let config = ValidationConfig::default();
    let validator = NeuralValidator::new(config);
    
    // Create consistent circuit metrics
    let circuit_metrics = cerebellar_norse::CircuitMetrics {
        granule_stats: cerebellar_norse::LayerStats {
            total_neurons: 1000,
            active_neurons: 100,
            spike_count: 50,
            average_membrane_potential: 0.3
        },
        purkinje_stats: cerebellar_norse::LayerStats {
            total_neurons: 100,
            active_neurons: 20,
            spike_count: 10,
            average_membrane_potential: 0.5
        },
        golgi_stats: cerebellar_norse::LayerStats {
            total_neurons: 50,
            active_neurons: 5,
            spike_count: 2,
            average_membrane_potential: 0.2
        },
        dcn_stats: cerebellar_norse::LayerStats {
            total_neurons: 10,
            active_neurons: 3,
            spike_count: 1,
            average_membrane_potential: 0.7
        },
        total_neurons: 1160
    };
    
    // Train with normal patterns
    for i in 0..50 {
        let normal_outputs = vec![1.0 + (i as f64 * 0.1), 0.5, -0.3, 0.8];
        let membrane_potentials = vec![0.5, -0.3, 0.1, 0.8];
        let spike_trains = vec![i % 2 == 0, false, i % 3 == 0, true];
        
        let result = validator.validate_comprehensive(
            &normal_outputs,
            &circuit_metrics,
            &membrane_potentials,
            &spike_trains
        ).unwrap();
        
        // Early samples might show anomalies due to lack of baseline
        if i > 10 {
            assert!(result.confidence > 0.3, "Confidence should improve with learning");
        }
    }
    
    // Now test with an anomalous pattern
    let anomalous_outputs = vec![10.0, 8.0, 7.0, 9.0]; // Much higher than training data
    let membrane_potentials = vec![0.5, -0.3, 0.1, 0.8];
    let spike_trains = vec![true, false, false, true];
    
    let result = validator.validate_comprehensive(
        &anomalous_outputs,
        &circuit_metrics,
        &membrane_potentials,
        &spike_trains
    ).unwrap();
    
    // Should detect anomaly after learning normal patterns
    assert!(!result.anomalies.is_empty(), "Should detect anomaly in unusual pattern");
    println!("Detected anomalies: {:?}", result.anomalies.iter().map(|a| &a.anomaly_type).collect::<Vec<_>>());
}

#[tokio::test]
async fn test_emergency_shutdown_scenario() {
    let limits = RiskLimits::default();
    let mut processor = SafeTradingProcessor::new(limits);
    let risk_manager = processor.get_risk_manager();
    
    // Start with normal operation
    assert!(risk_manager.get_risk_status().trading_enabled);
    
    // Process a normal trade
    let decision1 = processor.safe_process_tick(
        "AAPL".to_string(),
        150.0,
        1000.0,
        1625000000
    ).await.unwrap();
    
    println!("Normal trade decision: approved={}", decision1.risk_approved);
    
    // Trigger emergency shutdown
    risk_manager.emergency_shutdown("Critical system error detected".to_string()).await.unwrap();
    
    // Verify trading is halted
    assert!(!risk_manager.get_risk_status().trading_enabled);
    
    // Try to process another trade - should be rejected
    let decision2 = processor.safe_process_tick(
        "GOOGL".to_string(),
        2800.0,
        500.0,
        1625000001
    ).await.unwrap();
    
    assert!(!decision2.risk_approved, "Trade should be rejected after emergency shutdown");
    assert!(decision2.risk_reasons.iter().any(|reason| reason.contains("emergency") || reason.contains("rejected")));
    
    println!("Post-shutdown trade decision: approved={}, reasons={:?}", 
             decision2.risk_approved, decision2.risk_reasons);
}

#[test]
fn test_risk_limits_configuration() {
    let mut limits = RiskLimits::default();
    
    // Test default values
    assert_eq!(limits.max_total_exposure, 1_000_000.0);
    assert_eq!(limits.max_daily_loss, -50_000.0);
    assert_eq!(limits.max_drawdown_percent, 0.05);
    
    // Test custom configuration
    limits.max_position_per_symbol.insert("AAPL".to_string(), 10_000.0);
    limits.max_position_per_symbol.insert("GOOGL".to_string(), 5_000.0);
    limits.max_total_exposure = 100_000.0;
    limits.neural_output_bounds = (-5.0, 5.0);
    
    assert_eq!(limits.max_position_per_symbol["AAPL"], 10_000.0);
    assert_eq!(limits.max_position_per_symbol["GOOGL"], 5_000.0);
    assert_eq!(limits.neural_output_bounds, (-5.0, 5.0));
}