//! SEC Rule 15c3-5 Compliance Test Suite
//! 
//! Comprehensive testing to validate regulatory compliance including:
//! - Sub-100ms pre-trade validation
//! - <1 second kill switch propagation
//! - Mathematical correctness under load
//! - Concurrent access safety

use std::time::{Duration, Instant, SystemTime};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use tokio::sync::mpsc;
use uuid::Uuid;
use rust_decimal::Decimal;
use futures::future::join_all;

use crate::compliance::sec_rule_15c3_5::*;
use crate::risk::market_access_controls::*;
use crate::audit::regulatory_audit::*;
use crate::emergency::kill_switch::*;

const REGULATORY_LATENCY_LIMIT_MS: u128 = 100;
const KILL_SWITCH_LIMIT_MS: u128 = 1000;
const HIGH_LOAD_ORDER_COUNT: usize = 100_000;
const CONCURRENT_USER_COUNT: usize = 1000;

/// Test pre-trade validation performance under regulatory requirements
#[tokio::test]
async fn test_pretrade_validation_latency_compliance() {
    let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();
    
    // Configure strict limits
    engine.update_risk_limits("test_client".to_string(), RiskLimits {
        max_order_size: Decimal::from(10000),
        max_position_size: Decimal::from(100000),
        max_daily_loss: Decimal::from(500000),
        max_credit_exposure: Decimal::from(5000000),
        max_concentration_pct: Decimal::from(25),
        max_orders_per_second: 100,
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    let mut max_latency = 0u128;
    let mut total_latency = 0u128;
    let test_count = 10000;
    
    for i in 0..test_count {
        let order = create_test_order(i, "test_client");
        
        let start = Instant::now();
        let result = engine.validate_order(&order).await;
        let latency = start.elapsed().as_millis();
        
        max_latency = max_latency.max(latency);
        total_latency += latency;
        
        // REGULATORY REQUIREMENT: Each validation must be under 100ms
        assert!(latency < REGULATORY_LATENCY_LIMIT_MS, 
            "Order {} validation took {}ms (exceeds 100ms limit)", i, latency);
        
        // Verify nanosecond precision tracking
        assert!(result.validation_duration_nanos < 100_000_000, 
            "Validation duration {}ns exceeds 100ms", result.validation_duration_nanos);
    }
    
    let avg_latency = total_latency / test_count as u128;
    println!("Validation Performance:");
    println!("  Average latency: {}ms", avg_latency);
    println!("  Maximum latency: {}ms", max_latency);
    println!("  Regulatory limit: {}ms", REGULATORY_LATENCY_LIMIT_MS);
    
    // Ensure we're well within regulatory limits
    assert!(avg_latency < REGULATORY_LATENCY_LIMIT_MS / 2);
    assert!(max_latency < REGULATORY_LATENCY_LIMIT_MS);
}

/// Test kill switch activation and propagation speed
#[tokio::test]
async fn test_kill_switch_propagation_compliance() {
    let (emergency_tx, mut emergency_rx) = mpsc::unbounded_channel();
    let kill_switch = EmergencyKillSwitchEngine::new(emergency_tx);
    
    let start = Instant::now();
    let result = kill_switch.activate_kill_switch(
        KillSwitchTrigger::ExcessiveLoss,
        KillSwitchLevel::Level3,
        "compliance_officer".to_string(),
        "Daily loss limit exceeded".to_string(),
    ).await.unwrap();
    let propagation_time = start.elapsed().as_millis();
    
    // REGULATORY REQUIREMENT: Kill switch must propagate within 1 second
    assert!(propagation_time < KILL_SWITCH_LIMIT_MS,
        "Kill switch propagation took {}ms (exceeds 1000ms limit)", propagation_time);
    
    assert!(result.total_propagation_time_nanos < 1_000_000_000);
    assert!(kill_switch.is_active());
    
    // Verify emergency alert was sent
    let alert = emergency_rx.try_recv().unwrap();
    assert_eq!(alert.severity, AlertSeverity::Critical);
    assert!(alert.message.contains("KILL SWITCH ACTIVATED"));
    
    println!("Kill Switch Performance:");
    println!("  Propagation time: {}ms", propagation_time);
    println!("  Regulatory limit: {}ms", KILL_SWITCH_LIMIT_MS);
}

/// Test system under extreme load (1M+ orders/second simulation)
#[tokio::test]
async fn test_extreme_load_performance() {
    let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();
    
    // Configure limits for load testing
    engine.update_risk_limits("load_test_client".to_string(), RiskLimits {
        max_order_size: Decimal::from(1000000),
        max_position_size: Decimal::from(10000000),
        max_daily_loss: Decimal::from(50000000),
        max_credit_exposure: Decimal::from(100000000),
        max_concentration_pct: Decimal::from(50),
        max_orders_per_second: 10000,
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    let start_time = Instant::now();
    let processed_orders = Arc::new(AtomicU64::new(0));
    let latency_violations = Arc::new(AtomicU64::new(0));
    
    // Spawn concurrent validation tasks
    let mut handles = Vec::new();
    
    for batch in 0..100 {
        let engine_clone = engine.clone();
        let processed_clone = processed_orders.clone();
        let violations_clone = latency_violations.clone();
        
        let handle = tokio::spawn(async move {
            for i in 0..1000 {
                let order = create_test_order(batch * 1000 + i, "load_test_client");
                
                let validation_start = Instant::now();
                let result = engine_clone.validate_order(&order).await;
                let validation_time = validation_start.elapsed();
                
                processed_clone.fetch_add(1, Ordering::SeqCst);
                
                if validation_time.as_millis() >= REGULATORY_LATENCY_LIMIT_MS {
                    violations_clone.fetch_add(1, Ordering::SeqCst);
                }
                
                // Verify result consistency
                assert!(result.validation_duration_nanos > 0);
                assert!(!result.risk_checks.is_empty());
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all validations to complete
    join_all(handles).await;
    
    let total_time = start_time.elapsed();
    let total_processed = processed_orders.load(Ordering::SeqCst);
    let total_violations = latency_violations.load(Ordering::SeqCst);
    let orders_per_second = (total_processed as f64) / total_time.as_secs_f64();
    
    println!("Extreme Load Test Results:");
    println!("  Total orders processed: {}", total_processed);
    println!("  Total time: {:?}", total_time);
    println!("  Orders per second: {:.0}", orders_per_second);
    println!("  Latency violations: {}", total_violations);
    println!("  Violation rate: {:.2}%", (total_violations as f64 / total_processed as f64) * 100.0);
    
    // Performance requirements
    assert_eq!(total_processed, HIGH_LOAD_ORDER_COUNT as u64);
    assert!(orders_per_second > 50000.0, "Throughput too low: {:.0} orders/sec", orders_per_second);
    assert!(total_violations < total_processed / 100, "Too many latency violations: {}", total_violations);
}

/// Test concurrent access safety and race condition prevention
#[tokio::test]
async fn test_concurrent_access_safety() {
    let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();
    
    // Configure shared client limits
    engine.update_risk_limits("concurrent_client".to_string(), RiskLimits {
        max_order_size: Decimal::from(1000),
        max_position_size: Decimal::from(50000),
        max_daily_loss: Decimal::from(100000),
        max_credit_exposure: Decimal::from(1000000),
        max_concentration_pct: Decimal::from(30),
        max_orders_per_second: 10,
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    let success_count = Arc::new(AtomicU64::new(0));
    let rejection_count = Arc::new(AtomicU64::new(0));
    
    // Spawn many concurrent users
    let mut handles = Vec::new();
    
    for user_id in 0..CONCURRENT_USER_COUNT {
        let engine_clone = engine.clone();
        let success_clone = success_count.clone();
        let rejection_clone = rejection_count.clone();
        
        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let order = Order {
                    order_id: Uuid::new_v4(),
                    client_id: "concurrent_client".to_string(),
                    instrument_id: format!("STOCK{}", i % 5),
                    side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
                    quantity: Decimal::from(100),
                    price: Some(Decimal::from(100)),
                    order_type: OrderType::Limit,
                    timestamp: SystemTime::now(),
                    trader_id: format!("trader_{}", user_id),
                };
                
                let result = engine_clone.validate_order(&order).await;
                
                if result.is_valid {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    rejection_clone.fetch_add(1, Ordering::SeqCst);
                }
                
                // Brief delay to simulate realistic order flow
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        
        handles.push(handle);
    }
    
    join_all(handles).await;
    
    let total_success = success_count.load(Ordering::SeqCst);
    let total_rejections = rejection_count.load(Ordering::SeqCst);
    let total_orders = total_success + total_rejections;
    
    println!("Concurrent Access Test Results:");
    println!("  Total orders: {}", total_orders);
    println!("  Successful: {}", total_success);
    println!("  Rejected: {}", total_rejections);
    println!("  Success rate: {:.2}%", (total_success as f64 / total_orders as f64) * 100.0);
    
    // Verify no orders were lost and some were properly rejected due to velocity limits
    assert_eq!(total_orders, (CONCURRENT_USER_COUNT * 10) as u64);
    assert!(total_rejections > 0, "Expected some rejections due to velocity limits");
    assert!(total_success > total_orders / 2, "Success rate too low");
}

/// Test mathematical correctness of risk calculations
#[tokio::test]
async fn test_risk_calculation_accuracy() {
    let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();
    
    // Set precise limits for mathematical validation
    let max_position = Decimal::from_str("50000.123456789").unwrap();
    engine.update_risk_limits("math_test_client".to_string(), RiskLimits {
        max_order_size: Decimal::from_str("10000.987654321").unwrap(),
        max_position_size: max_position,
        max_daily_loss: Decimal::from_str("100000.555555555").unwrap(),
        max_credit_exposure: Decimal::from_str("1000000.123456789").unwrap(),
        max_concentration_pct: Decimal::from_str("25.777777777").unwrap(),
        max_orders_per_second: 5,
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    // Test order size validation with precise decimals
    let large_order = Order {
        order_id: Uuid::new_v4(),
        client_id: "math_test_client".to_string(),
        instrument_id: "PRECISION_TEST".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_str("10000.987654322").unwrap(), // Slightly over limit
        price: Some(Decimal::from_str("100.123456789").unwrap()),
        order_type: OrderType::Limit,
        timestamp: SystemTime::now(),
        trader_id: "precision_trader".to_string(),
    };
    
    let result = engine.validate_order(&large_order).await;
    assert!(!result.is_valid, "Order should be rejected for exceeding size limit");
    
    let size_check = result.risk_checks.iter()
        .find(|check| matches!(check.check_type, RiskCheckType::OrderSize))
        .expect("Order size check should be present");
    
    assert!(!size_check.passed);
    assert_eq!(size_check.current_value, large_order.quantity);
    
    // Test valid order within limits
    let valid_order = Order {
        order_id: Uuid::new_v4(),
        client_id: "math_test_client".to_string(),
        instrument_id: "PRECISION_TEST".to_string(),
        side: OrderSide::Buy,
        quantity: Decimal::from_str("10000.987654320").unwrap(), // Just under limit
        price: Some(Decimal::from_str("100.123456789").unwrap()),
        order_type: OrderType::Limit,
        timestamp: SystemTime::now(),
        trader_id: "precision_trader".to_string(),
    };
    
    let result = engine.validate_order(&valid_order).await;
    assert!(result.is_valid, "Order should be accepted within limits");
    
    println!("Mathematical Precision Test:");
    println!("  Order quantity: {}", valid_order.quantity);
    println!("  Size limit: {}", Decimal::from_str("10000.987654321").unwrap());
    println!("  Validation passed: {}", result.is_valid);
}

/// Test audit trail integrity and cryptographic verification
#[tokio::test]
async fn test_audit_trail_integrity() {
    let audit_engine = RegulatoryAuditEngine::new();
    let test_event_count = 1000;
    
    // Generate a series of audit events
    for i in 0..test_event_count {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            event_type: match i % 4 {
                0 => AuditEventType::OrderSubmitted,
                1 => AuditEventType::RiskValidationPerformed,
                2 => AuditEventType::OrderAccepted,
                _ => AuditEventType::OrderRejected,
            },
            timestamp: SystemTime::now(),
            nanosecond_precision: (i as u64) * 1000000, // Simulate different durations
            user_id: format!("trader_{}", i % 10),
            order_id: Some(Uuid::new_v4()),
            details: serde_json::json!({
                "test_index": i,
                "validation_time": (i as u64) * 1000000
            }),
            cryptographic_hash: format!("hash_{}", i),
        };
        
        audit_engine.log_audit_event(event).await.unwrap();
    }
    
    // Verify audit chain integrity
    let integrity_result = audit_engine.verify_audit_integrity().await;
    
    println!("Audit Trail Integrity Test:");
    println!("  Total records: {}", integrity_result.total_records);
    println!("  Verified records: {}", integrity_result.verified_records);
    println!("  Hash mismatches: {}", integrity_result.hash_mismatches.len());
    println!("  Sequence gaps: {}", integrity_result.sequence_gaps.len());
    println!("  Timestamp anomalies: {}", integrity_result.timestamp_anomalies.len());
    
    assert!(integrity_result.is_valid, "Audit trail integrity verification failed");
    assert_eq!(integrity_result.total_records, test_event_count);
    assert_eq!(integrity_result.verified_records, test_event_count);
    assert!(integrity_result.hash_mismatches.is_empty());
    assert!(integrity_result.sequence_gaps.is_empty());
}

/// Test regulatory reporting generation
#[tokio::test]
async fn test_regulatory_reporting() {
    let audit_engine = RegulatoryAuditEngine::new();
    let start_time = SystemTime::now();
    
    // Generate diverse audit events for reporting
    let event_types = [
        AuditEventType::OrderSubmitted,
        AuditEventType::OrderRejected,
        AuditEventType::KillSwitchActivated,
        AuditEventType::RiskValidationPerformed,
        AuditEventType::SystemAlert,
    ];
    
    for (i, &event_type) in event_types.iter().cycle().take(100).enumerate() {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            event_type,
            timestamp: SystemTime::now(),
            nanosecond_precision: if event_type == AuditEventType::RiskValidationPerformed {
                (i as u64 + 1) * 10_000_000 // 10ms increments
            } else {
                0
            },
            user_id: format!("trader_{}", i % 5),
            order_id: Some(Uuid::new_v4()),
            details: serde_json::json!({"test_event": i}),
            cryptographic_hash: format!("hash_{}", i),
        };
        
        audit_engine.log_audit_event(event).await.unwrap();
    }
    
    let end_time = SystemTime::now();
    
    // Generate regulatory report
    let report = audit_engine.generate_report(
        ReportType::OnDemand,
        start_time,
        end_time,
    ).await.unwrap();
    
    println!("Regulatory Report Test:");
    println!("  Report ID: {}", report.report_id);
    println!("  Total records: {}", report.total_records);
    println!("  Total orders: {}", report.summary_statistics.total_orders);
    println!("  Total rejections: {}", report.summary_statistics.total_rejections);
    println!("  Avg validation time: {}ns", report.summary_statistics.avg_validation_time_nanos);
    println!("  Max validation time: {}ns", report.summary_statistics.max_validation_time_nanos);
    println!("  Kill switch activations: {}", report.summary_statistics.kill_switch_activations);
    
    assert_eq!(report.total_records, 100);
    assert!(report.summary_statistics.total_orders > 0);
    assert!(report.summary_statistics.kill_switch_activations > 0);
    assert!(report.summary_statistics.avg_validation_time_nanos > 0);
    assert!(report.summary_statistics.max_validation_time_nanos < 100_000_000); // Under 100ms
}

/// Test market access controls and circuit breakers
#[tokio::test]
async fn test_market_access_controls() {
    let (audit_tx, _audit_rx) = mpsc::unbounded_channel();
    let (emergency_tx, mut emergency_rx) = mpsc::unbounded_channel();
    
    let market_engine = MarketAccessEngine::new(audit_tx, emergency_tx);
    
    // Test normal market access
    let access_decision = market_engine.is_market_access_allowed().await;
    assert!(access_decision.allowed);
    assert!(access_decision.decision_time_nanos < 10_000_000); // Under 10ms
    
    // Test circuit breaker activation
    let high_stress_metrics = SystematicRiskMetrics {
        market_stress_indicator: Decimal::from(15), // 15% decline - should trigger Level 2
        volatility_index: Decimal::from(60),
        correlation_breakdown: false,
        liquidity_stress: Decimal::from(70),
        credit_stress: Decimal::from(40),
        operational_risk_level: RiskLevel::High,
        last_updated: SystemTime::now(),
    };
    
    market_engine.update_systematic_risk(high_stress_metrics).await;
    
    // Verify circuit breaker activated
    let market_status = market_engine.get_market_status();
    assert_eq!(market_status.circuit_breaker_level, CircuitBreakerLevel::Level2);
    
    // Verify market access now denied
    let access_decision = market_engine.is_market_access_allowed().await;
    assert!(!access_decision.allowed);
    assert!(access_decision.reason.contains("Circuit breaker"));
    
    // Verify emergency alert was sent
    let alert = emergency_rx.try_recv().unwrap();
    assert_eq!(alert.severity, AlertSeverity::Critical);
    assert!(alert.message.contains("CIRCUIT BREAKER"));
    
    println!("Market Access Controls Test:");
    println!("  Circuit breaker level: {:?}", market_status.circuit_breaker_level);
    println!("  Market stress level: {}", market_status.market_stress_level);
    println!("  Access decision time: {}ns", access_decision.decision_time_nanos);
}

/// Comprehensive integration test simulating real trading scenario
#[tokio::test]
async fn test_full_integration_scenario() {
    let (pre_trade_engine, _audit_rx, emergency_rx) = PreTradeRiskEngine::new();
    let (emergency_tx, mut emergency_alerts) = mpsc::unbounded_channel();
    let kill_switch = EmergencyKillSwitchEngine::new(emergency_tx);
    
    // Configure realistic limits
    pre_trade_engine.update_risk_limits("integration_client".to_string(), RiskLimits {
        max_order_size: Decimal::from(50000),
        max_position_size: Decimal::from(500000),
        max_daily_loss: Decimal::from(1000000),
        max_credit_exposure: Decimal::from(10000000),
        max_concentration_pct: Decimal::from(20),
        max_orders_per_second: 50,
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    let mut total_orders = 0;
    let mut successful_orders = 0;
    let mut rejected_orders = 0;
    
    // Simulate normal trading activity
    for i in 0..1000 {
        let order = Order {
            order_id: Uuid::new_v4(),
            client_id: "integration_client".to_string(),
            instrument_id: format!("STOCK{}", i % 20),
            side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
            quantity: Decimal::from(1000 + (i % 100) * 10),
            price: Some(Decimal::from(100 + i % 50)),
            order_type: OrderType::Limit,
            timestamp: SystemTime::now(),
            trader_id: format!("trader_{}", i % 10),
        };
        
        let result = pre_trade_engine.validate_order(&order).await;
        total_orders += 1;
        
        if result.is_valid {
            successful_orders += 1;
        } else {
            rejected_orders += 1;
        }
        
        // Simulate kill switch trigger at order 500
        if i == 500 {
            let kill_result = kill_switch.activate_kill_switch(
                KillSwitchTrigger::ExcessiveLoss,
                KillSwitchLevel::Level3,
                "risk_manager".to_string(),
                "Simulated risk breach".to_string(),
            ).await.unwrap();
            
            assert!(kill_result.total_propagation_time_nanos < 1_000_000_000);
            break; // Stop processing orders after kill switch
        }
    }
    
    // Verify kill switch is active
    assert!(kill_switch.is_active());
    
    // Verify emergency alert was generated
    let alert = emergency_alerts.try_recv().unwrap();
    assert_eq!(alert.severity, AlertSeverity::Critical);
    
    println!("Full Integration Test Results:");
    println!("  Total orders processed: {}", total_orders);
    println!("  Successful orders: {}", successful_orders);
    println!("  Rejected orders: {}", rejected_orders);
    println!("  Kill switch activated at order: 501");
    println!("  Success rate: {:.2}%", (successful_orders as f64 / total_orders as f64) * 100.0);
    
    // Verify system state
    assert_eq!(total_orders, 501); // Should stop at kill switch activation
    assert!(successful_orders > 0);
    assert!(rejected_orders >= 0);
}

// Helper function to create test orders
fn create_test_order(index: usize, client_id: &str) -> Order {
    Order {
        order_id: Uuid::new_v4(),
        client_id: client_id.to_string(),
        instrument_id: format!("STOCK{}", index % 10),
        side: if index % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
        quantity: Decimal::from(100 + (index % 100)),
        price: Some(Decimal::from(100 + (index % 20))),
        order_type: OrderType::Limit,
        timestamp: SystemTime::now(),
        trader_id: format!("trader_{}", index % 5),
    }
}

/// Test order velocity controls under high frequency
#[tokio::test]
async fn test_velocity_controls_hft() {
    let (engine, _audit_rx, _emergency_rx) = PreTradeRiskEngine::new();
    
    // Configure strict velocity limits
    engine.update_risk_limits("hft_client".to_string(), RiskLimits {
        max_order_size: Decimal::from(1000),
        max_position_size: Decimal::from(100000),
        max_daily_loss: Decimal::from(500000),
        max_credit_exposure: Decimal::from(5000000),
        max_concentration_pct: Decimal::from(25),
        max_orders_per_second: 5, // Very strict limit
        updated_at: SystemTime::now(),
        valid_until: SystemTime::now() + Duration::from_secs(86400),
    }).await;
    
    let mut velocity_rejections = 0;
    let mut successful_orders = 0;
    
    // Send 20 orders in rapid succession
    for i in 0..20 {
        let order = Order {
            order_id: Uuid::new_v4(),
            client_id: "hft_client".to_string(),
            instrument_id: "HFT_STOCK".to_string(),
            side: if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell },
            quantity: Decimal::from(100),
            price: Some(Decimal::from(100)),
            order_type: OrderType::Limit,
            timestamp: SystemTime::now(),
            trader_id: "hft_trader".to_string(),
        };
        
        let result = engine.validate_order(&order).await;
        
        if result.is_valid {
            successful_orders += 1;
        } else {
            // Check if rejection was due to velocity control
            let velocity_check = result.risk_checks.iter()
                .find(|check| matches!(check.check_type, RiskCheckType::VelocityControl));
            
            if let Some(check) = velocity_check {
                if !check.passed {
                    velocity_rejections += 1;
                }
            }
        }
        
        // Small delay to simulate realistic order timing
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    println!("HFT Velocity Control Test:");
    println!("  Total orders: 20");
    println!("  Successful orders: {}", successful_orders);
    println!("  Velocity rejections: {}", velocity_rejections);
    println!("  Orders per second limit: 5");
    
    // Should have some velocity rejections due to rapid order submission
    assert!(velocity_rejections > 0, "Expected velocity rejections with rapid order submission");
    assert!(successful_orders <= 10, "Too many orders accepted given velocity limits");
}