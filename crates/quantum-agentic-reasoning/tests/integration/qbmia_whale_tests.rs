//! Integration tests for QBMIA & Whale Defense hive-mind coordination
//! 
//! Tests the specialized hive-mind system that coordinates:
//! - QBMIA quantum-biological intelligence acceleration
//! - Whale defense sub-microsecond threat assessment
//! - Cross-system coordination and consensus
//! - Performance optimization recommendations

use quantum_agentic_reasoning::qbmia_whale_integration::*;
use quantum_agentic_reasoning::{MarketData, execution_context::ExecutionContext, QARConfig};
use std::time::Instant;

#[tokio::test]
async fn test_qbmia_whale_hive_mind_initialization() {
    let config = QBMIAWhaleConfig::default();
    let hive_mind = QBMIAWhaleHiveMind::new(config);
    
    assert!(hive_mind.is_ok());
    let hive_mind = hive_mind.unwrap();
    
    let metrics = hive_mind.get_hive_mind_metrics();
    assert_eq!(metrics.total_coordinations, 0);
    assert_eq!(metrics.coordination_success_rate, 1.0);
    assert!(metrics.threat_detection_accuracy >= 0.9);
}

#[tokio::test]
async fn test_integrated_enhancement_performance() {
    let config = QBMIAWhaleConfig {
        max_response_time_ns: 500, // 500ns target
        coordination_level: CoordinationLevel::Intensive,
        ..Default::default()
    };
    
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    
    let start = Instant::now();
    let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await;
    let elapsed = start.elapsed();
    
    assert!(enhancement.is_ok());
    let enhancement = enhancement.unwrap();
    
    // Verify performance constraint
    assert!(elapsed.as_nanos() < 2000); // Allow some overhead for test environment
    
    // Verify enhancement structure
    assert!(enhancement.threat_assessment >= 0.0 && enhancement.threat_assessment <= 1.0);
    assert!(!enhancement.risk_adjustments.is_empty());
    assert!(!enhancement.performance_optimizations.is_empty());
    
    // QBMIA intelligence should be present
    assert!(enhancement.qbmia_intelligence.biological_confidence >= 0.0);
    assert!(enhancement.qbmia_intelligence.biological_confidence <= 1.0);
    assert!(enhancement.qbmia_intelligence.market_intelligence >= 0.0);
    assert!(enhancement.qbmia_intelligence.quantum_enhancement >= 0.0);
    
    // Whale defense should be active
    assert!(enhancement.whale_defense.threat_level >= 0.0);
    assert!(enhancement.whale_defense.threat_level <= 1.0);
    assert!(enhancement.whale_defense.realtime_metrics.response_time_ns > 0);
}

#[tokio::test]
async fn test_qbmia_biological_intelligence() {
    let config = QBMIAWhaleConfig {
        enable_qbmia: true,
        enable_whale_defense: false, // Test QBMIA in isolation
        ..Default::default()
    };
    
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![55000.0, 45000.0],
        buy_probabilities: vec![0.8, 0.2], // Strong bullish signal
        sell_probabilities: vec![0.2, 0.8],
        hold_probabilities: vec![0.3, 0.7],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Gain,
            emphasis: 0.9,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await.unwrap();
    
    // QBMIA should be active
    assert!(enhancement.qbmia_intelligence.biological_confidence > 0.5);
    assert!(enhancement.qbmia_intelligence.market_intelligence > 0.5);
    assert!(!enhancement.qbmia_intelligence.learning_insights.is_empty());
    
    // Performance metrics should be realistic
    assert!(enhancement.qbmia_intelligence.acceleration_factors.gpu_utilization >= 0.0);
    assert!(enhancement.qbmia_intelligence.acceleration_factors.gpu_utilization <= 1.0);
    assert!(enhancement.qbmia_intelligence.acceleration_factors.simd_efficiency >= 0.0);
    assert!(enhancement.qbmia_intelligence.acceleration_factors.simd_efficiency <= 1.0);
    
    // Should have performance optimizations
    assert!(!enhancement.performance_optimizations.is_empty());
    
    // Whale defense should be in safe mode
    assert!(!enhancement.whale_defense.whale_detected);
    assert_eq!(enhancement.whale_defense.threat_level, 0.0);
}

#[tokio::test]
async fn test_whale_defense_threat_detection() {
    let config = QBMIAWhaleConfig {
        enable_qbmia: false, // Test whale defense in isolation
        enable_whale_defense: true,
        ..Default::default()
    };
    
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    // Simulate potential whale activity with unusual market data
    let suspicious_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![65000.0, 35000.0], // Extreme volatility
        buy_probabilities: vec![0.05, 0.95], // Extreme bearish signal
        sell_probabilities: vec![0.95, 0.05],
        hold_probabilities: vec![0.1, 0.9],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Loss,
            emphasis: 1.0,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    let enhancement = hive_mind.get_integrated_enhancement(&suspicious_data, &context).await.unwrap();
    
    // Whale defense should be analyzing the situation
    assert!(enhancement.whale_defense.realtime_metrics.response_time_ns < 1000); // Sub-microsecond
    assert!(enhancement.whale_defense.ml_analysis.confidence >= 0.0);
    assert!(enhancement.whale_defense.ml_analysis.confidence <= 1.0);
    
    // Should have timing recommendations
    assert!(enhancement.timing_recommendations.optimal_delay_ns > 0);
    
    // QBMIA should be in neutral mode
    assert_eq!(enhancement.qbmia_intelligence.biological_confidence, 0.5);
    assert_eq!(enhancement.qbmia_intelligence.market_intelligence, 0.5);
}

#[tokio::test]
async fn test_coordination_levels() {
    let coordination_levels = vec![
        CoordinationLevel::Minimal,
        CoordinationLevel::Standard,
        CoordinationLevel::Intensive,
        CoordinationLevel::Symbiotic,
    ];
    
    for level in coordination_levels {
        let config = QBMIAWhaleConfig {
            coordination_level: level,
            ..Default::default()
        };
        
        let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
        
        let market_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            current_price: 50000.0,
            possible_outcomes: vec![52000.0, 48000.0],
            buy_probabilities: vec![0.6, 0.4],
            sell_probabilities: vec![0.4, 0.6],
            hold_probabilities: vec![0.5, 0.5],
            frame: prospect_theory::FramingContext {
                frame_type: prospect_theory::FrameType::Neutral,
                emphasis: 0.5,
            },
            timestamp: 1640995200000,
        };
        
        let context = ExecutionContext::new(&QARConfig::default()).unwrap();
        let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await;
        
        assert!(enhancement.is_ok());
        let enhancement = enhancement.unwrap();
        
        // All coordination levels should produce valid results
        assert!(enhancement.threat_assessment >= 0.0 && enhancement.threat_assessment <= 1.0);
        assert!(!enhancement.risk_adjustments.is_empty());
    }
}

#[tokio::test]
async fn test_defense_mechanism_activation() {
    let config = QBMIAWhaleConfig::default();
    let hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    // Test different threat levels
    let threat_levels = vec![0.0, 0.3, 0.5, 0.7, 0.9, 1.0];
    
    for threat_level in threat_levels {
        let defenses = hive_mind.select_defense_mechanisms(threat_level);
        
        match threat_level {
            t if t <= 0.3 => {
                // Low threat - minimal or no defenses
                assert!(defenses.len() <= 1);
            },
            t if t <= 0.5 => {
                // Medium threat - some defenses
                assert!(!defenses.is_empty());
                assert!(defenses.contains(&DefenseMechanism::TimingRandomization));
            },
            t if t <= 0.7 => {
                // High threat - multiple defenses
                assert!(defenses.len() >= 2);
                assert!(defenses.contains(&DefenseMechanism::OrderFragmentation));
            },
            _ => {
                // Maximum threat - all defenses
                assert!(defenses.len() >= 3);
                assert!(defenses.contains(&DefenseMechanism::Steganography));
                assert!(defenses.contains(&DefenseMechanism::VolumeObfuscation));
            }
        }
    }
}

#[tokio::test]
async fn test_performance_optimization_recommendations() {
    let config = QBMIAWhaleConfig::default();
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await.unwrap();
    
    // Should have performance optimizations
    assert!(!enhancement.performance_optimizations.is_empty());
    
    // Check that optimizations are actionable
    for optimization in &enhancement.performance_optimizations {
        assert!(!optimization.is_empty());
        assert!(optimization.len() > 10); // Should be descriptive
    }
    
    // Should include cross-system optimizations
    let has_cross_system = enhancement.performance_optimizations.iter()
        .any(|opt| opt.contains("cross-agent") || opt.contains("coordination"));
    assert!(has_cross_system);
}

#[tokio::test]
async fn test_risk_adjustments() {
    let config = QBMIAWhaleConfig::default();
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    let enhancement = hive_mind.get_integrated_enhancement(&market_data, &context).await.unwrap();
    
    // Should have risk adjustments
    assert!(!enhancement.risk_adjustments.is_empty());
    
    // Check specific risk factors
    assert!(enhancement.risk_adjustments.contains_key("biological_risk_factor"));
    assert!(enhancement.risk_adjustments.contains_key("quantum_uncertainty"));
    assert!(enhancement.risk_adjustments.contains_key("whale_threat_adjustment"));
    assert!(enhancement.risk_adjustments.contains_key("integrated_risk_multiplier"));
    
    // All risk factors should be reasonable
    for (factor_name, &factor_value) in &enhancement.risk_adjustments {
        assert!(factor_value >= 0.0, "Risk factor {} = {} should be non-negative", factor_name, factor_value);
        assert!(factor_value <= 2.0, "Risk factor {} = {} should be reasonable", factor_name, factor_value);
    }
    
    // Integrated risk multiplier should be close to 1.0 for neutral scenarios
    let integrated_risk = enhancement.risk_adjustments.get("integrated_risk_multiplier").unwrap();
    assert!(*integrated_risk >= 0.8);
    assert!(*integrated_risk <= 1.5);
}

#[tokio::test]
async fn test_timing_recommendations() {
    let config = QBMIAWhaleConfig::default();
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    // Test different threat scenarios
    let scenarios = vec![
        ("Low threat", 0.2),
        ("Medium threat", 0.6),
        ("High threat", 0.9),
    ];
    
    for (scenario_name, threat_level) in scenarios {
        // Create whale defense status with specific threat level
        let whale_defense = WhaleDefenseStatus {
            threat_level,
            whale_detected: threat_level > 0.7,
            defenses_active: hive_mind.select_defense_mechanisms(threat_level),
            realtime_metrics: RealtimeMetrics {
                response_time_ns: 150,
                throughput_ops_per_sec: 60000,
                memory_usage_mb: 50.0,
                cpu_utilization: 0.2,
            },
            ml_analysis: MLAnalysisResult {
                anomaly_score: threat_level * 0.5,
                behavioral_classification: "Test".to_string(),
                confidence: 0.9,
                patterns_detected: vec!["test_pattern".to_string()],
            },
        };
        
        let timing = hive_mind.generate_timing_recommendations(&whale_defense);
        
        println!("{}: delay={}ns, window={}ns, steg={}, frag={}", 
                scenario_name, 
                timing.optimal_delay_ns, 
                timing.randomization_window_ns,
                timing.use_steganography,
                timing.fragment_orders);
        
        // Higher threat should lead to more conservative timing
        assert!(timing.optimal_delay_ns > 0);
        assert!(timing.randomization_window_ns > 0);
        
        if threat_level > 0.6 {
            assert!(timing.use_steganography);
        }
        
        if threat_level > 0.8 {
            assert!(timing.fragment_orders);
        }
    }
}

#[tokio::test]
async fn test_coordination_metrics_tracking() {
    let config = QBMIAWhaleConfig {
        max_response_time_ns: 1000,
        ..Default::default()
    };
    
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    
    // Perform multiple coordinations
    for _ in 0..5 {
        hive_mind.get_integrated_enhancement(&market_data, &context).await.unwrap();
    }
    
    let metrics = hive_mind.get_hive_mind_metrics();
    
    // Should track all coordinations
    assert_eq!(metrics.total_coordinations, 5);
    assert!(metrics.average_response_time_ns > 0);
    assert!(metrics.average_response_time_ns < 5000); // Should be fast
    assert!(metrics.coordination_success_rate >= 0.8);
    assert!(metrics.threat_detection_accuracy >= 0.9);
    assert!(metrics.qbmia_intelligence_quality >= 0.8);
}

#[tokio::test]
async fn test_hive_mind_under_load() {
    let config = QBMIAWhaleConfig {
        max_response_time_ns: 500,
        coordination_level: CoordinationLevel::Intensive,
        ..Default::default()
    };
    
    let mut hive_mind = QBMIAWhaleHiveMind::new(config).unwrap();
    
    let market_data = MarketData {
        symbol: "BTC/USDT".to_string(),
        current_price: 50000.0,
        possible_outcomes: vec![52000.0, 48000.0],
        buy_probabilities: vec![0.6, 0.4],
        sell_probabilities: vec![0.4, 0.6],
        hold_probabilities: vec![0.5, 0.5],
        frame: prospect_theory::FramingContext {
            frame_type: prospect_theory::FrameType::Neutral,
            emphasis: 0.5,
        },
        timestamp: 1640995200000,
    };
    
    let context = ExecutionContext::new(&QARConfig::default()).unwrap();
    
    // Rapid-fire coordinations to test under load
    let num_coordinations = 100;
    let start = Instant::now();
    
    for i in 0..num_coordinations {
        let mut data = market_data.clone();
        data.current_price += i as f64; // Slight variations
        data.timestamp += i as u64;
        
        let enhancement = hive_mind.get_integrated_enhancement(&data, &context).await.unwrap();
        
        // Each coordination should be valid
        assert!(enhancement.threat_assessment >= 0.0 && enhancement.threat_assessment <= 1.0);
        assert!(!enhancement.risk_adjustments.is_empty());
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time.as_nanos() as f64 / num_coordinations as f64;
    
    println!("Hive-mind under load: {} coordinations in {:.2}ms (avg: {:.0}ns)", 
             num_coordinations, total_time.as_millis(), avg_time);
    
    // Should maintain performance under load
    assert!(avg_time < 2000.0); // 2μs average allowing for test overhead
    
    let final_metrics = hive_mind.get_hive_mind_metrics();
    assert_eq!(final_metrics.total_coordinations, num_coordinations);
    assert!(final_metrics.coordination_success_rate >= 0.95);
}