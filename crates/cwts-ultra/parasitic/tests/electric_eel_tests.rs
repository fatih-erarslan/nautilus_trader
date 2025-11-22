//! Comprehensive tests for Electric Eel Shocker organism
//! Following TDD methodology with extensive test coverage
//! CQGS Compliance: Zero mocks, real implementations only

use chrono::Utc;
use parasitic::organisms::electric_eel::{
    ElectricEelShocker, HiddenLiquidityDetector, HiddenLiquidityPool, MarketCondition,
    MarketDisruptor, PoolType, ShockRevealationResult, ShockTimingOptimizer,
};
use parasitic::traits::{Adaptive, MarketData, Organism, OrganismMetrics, PerformanceMonitor};
use parasitic::{Error, Result};
use std::sync::Arc;
use std::thread;

/// Create test market data with specified parameters
fn create_test_market_data(price: f64, volatility: f64, liquidity: f64) -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: Utc::now(),
        price,
        volume: 1000.0,
        volatility,
        bid: price * 0.999,
        ask: price * 1.001,
        spread_percent: 0.002,
        market_cap: Some(1_000_000_000.0),
        liquidity_score: liquidity,
    }
}

/// Create test market sequence for timing analysis
fn create_market_sequence(base_price: f64, volatility_trend: f64, count: usize) -> Vec<MarketData> {
    let mut sequence = Vec::with_capacity(count);

    for i in 0..count {
        let price_drift = (i as f64) * 10.0; // $10 drift per step
        let volatility = 0.05 + volatility_trend * (i as f64) * 0.02;
        let liquidity = 0.8 - (i as f64) * 0.05; // Decreasing liquidity

        sequence.push(create_test_market_data(
            base_price + price_drift,
            volatility.max(0.01).min(0.5),
            liquidity.max(0.3).min(1.0),
        ));
    }

    sequence
}

#[cfg(test)]
mod electric_eel_tests {
    use super::*;

    /// Test 1: Basic organism creation and initialization
    #[test]
    fn test_electric_eel_creation() {
        let shocker = ElectricEelShocker::new();
        assert!(
            shocker.is_ok(),
            "ElectricEelShocker creation should succeed"
        );

        let shocker = shocker.unwrap();
        assert_eq!(shocker.name(), "ElectricEelShocker");
        assert_eq!(shocker.organism_type(), "BioelectricPredator");
        assert!(shocker.is_active());
        assert_eq!(shocker.get_charge_level(), 1.0); // Should start fully charged
    }

    /// Test 2: Bioelectric shock generation - core functionality
    #[test]
    fn test_bioelectric_shock_generation() {
        let shocker = ElectricEelShocker::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.15, 0.8);

        let result = shocker.generate_bioelectric_shock(&market_data, 0.8);
        assert!(
            result.is_ok(),
            "Bioelectric shock generation should succeed"
        );

        let shock_result = result.unwrap();

        // Validate disruption result
        assert!(shock_result.disruption_result.shock_intensity > 0.0);
        assert!(shock_result.disruption_result.voltage_generated > 0.0);
        assert!(shock_result.disruption_result.disruption_radius > 0.0);
        assert!(!shock_result
            .disruption_result
            .affected_price_levels
            .is_empty());

        // Validate bioelectric charge depletion
        assert!(
            shock_result.bioelectric_charge_remaining < 1.0,
            "Charge should be depleted after shock"
        );

        // Validate timing window
        assert!(shock_result.next_optimal_window.duration_ms > 0);
        assert!(
            shock_result.next_optimal_window.expected_effectiveness >= 0.0
                && shock_result.next_optimal_window.expected_effectiveness <= 1.0
        );

        // Validate information gain
        assert!(shock_result.information_gain >= 0.0);
    }

    /// Test 3: Hidden liquidity detection capabilities
    #[test]
    fn test_hidden_liquidity_detection() {
        let shocker = ElectricEelShocker::new().unwrap();

        // High volatility, medium liquidity should reveal hidden pools
        let volatile_market = create_test_market_data(50000.0, 0.25, 0.6);
        let result = shocker
            .generate_bioelectric_shock(&volatile_market, 0.9)
            .unwrap();

        // Should discover some hidden liquidity in volatile conditions
        assert!(
            !result.hidden_liquidity_pools.is_empty(),
            "Volatile market should reveal hidden liquidity"
        );

        for pool in &result.hidden_liquidity_pools {
            assert!(pool.confidence_score > 0.0 && pool.confidence_score <= 1.0);
            assert!(pool.estimated_volume > 0.0);
            assert!(pool.price_level > 0.0);
            assert_ne!(pool.pool_type, PoolType::Unknown); // Should classify pool type
        }
    }

    /// Test 4: Market condition classification
    #[test]
    fn test_market_condition_classification() {
        let shocker = ElectricEelShocker::new().unwrap();

        // Test different market conditions
        let calm_market = create_test_market_data(50000.0, 0.02, 0.9);
        let volatile_market = create_test_market_data(50000.0, 0.3, 0.5);
        let chaotic_market = create_test_market_data(50000.0, 0.4, 0.3);

        let calm_result = shocker
            .generate_bioelectric_shock(&calm_market, 0.5)
            .unwrap();
        let volatile_result = shocker
            .generate_bioelectric_shock(&volatile_market, 0.8)
            .unwrap();

        // Recharge before chaotic test
        shocker.recharge_bioelectric_system(0.5).unwrap();
        let chaotic_result = shocker
            .generate_bioelectric_shock(&chaotic_market, 0.9)
            .unwrap();

        // Different market conditions should produce different timing strategies
        assert!(
            calm_result.next_optimal_window.duration_ms
                >= volatile_result.next_optimal_window.duration_ms
        );

        // Chaotic markets should have shorter optimal windows
        assert!(
            chaotic_result.next_optimal_window.duration_ms
                <= calm_result.next_optimal_window.duration_ms
        );
    }

    /// Test 5: Bioelectric charge management
    #[test]
    fn test_bioelectric_charge_management() {
        let shocker = ElectricEelShocker::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.2, 0.7);

        // Initial charge should be full
        assert_eq!(shocker.get_charge_level(), 1.0);

        // Generate multiple shocks to test charge depletion
        let initial_charge = shocker.get_charge_level();
        let result1 = shocker
            .generate_bioelectric_shock(&market_data, 0.8)
            .unwrap();
        assert!(result1.bioelectric_charge_remaining < initial_charge);

        let charge_after_first = result1.bioelectric_charge_remaining;
        let result2 = shocker
            .generate_bioelectric_shock(&market_data, 0.6)
            .unwrap();
        assert!(result2.bioelectric_charge_remaining < charge_after_first);

        // Test recharging
        let depleted_charge = shocker.get_charge_level();
        let new_charge = shocker.recharge_bioelectric_system(0.4).unwrap();
        assert!(new_charge > depleted_charge);
        assert!(new_charge <= 1.0);
    }

    /// Test 6: Insufficient charge error handling
    #[test]
    fn test_insufficient_charge_error() {
        let shocker = ElectricEelShocker::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.1, 0.8);

        // Manually set low charge
        *shocker.charge_level.write() = 0.05;

        // High-intensity shock should fail with low charge
        let result = shocker.generate_bioelectric_shock(&market_data, 0.9);
        assert!(
            result.is_err(),
            "High intensity shock with low charge should fail"
        );

        // Low-intensity shock should work
        let low_intensity_result = shocker.generate_bioelectric_shock(&market_data, 0.1);
        assert!(
            low_intensity_result.is_ok(),
            "Low intensity shock should work with low charge"
        );
    }

    /// Test 7: Shock sequence generation and batch processing
    #[test]
    fn test_shock_sequence_generation() {
        let shocker = ElectricEelShocker::new().unwrap();
        let market_sequence = create_market_sequence(50000.0, 0.1, 5);

        let results = shocker
            .generate_shock_sequence(&market_sequence, 0.4)
            .unwrap();

        // Should process at least some of the sequence
        assert!(
            !results.is_empty(),
            "Should process at least part of the sequence"
        );

        // Each result should have decreasing or equal charge
        for i in 1..results.len() {
            assert!(
                results[i].bioelectric_charge_remaining
                    <= results[i - 1].bioelectric_charge_remaining,
                "Charge should decrease or stay same through sequence"
            );
        }

        // Each result should be valid
        for (i, result) in results.iter().enumerate() {
            assert!(result.disruption_result.shock_intensity > 0.0);
            assert!(!result.disruption_result.affected_price_levels.is_empty());
            println!(
                "Sequence {}: charge={:.2}, pools={}",
                i,
                result.bioelectric_charge_remaining,
                result.hidden_liquidity_pools.len()
            );
        }
    }

    /// Test 8: Performance requirements validation
    #[test]
    fn test_performance_requirements() {
        let shocker = ElectricEelShocker::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.1, 0.8);

        let start = std::time::Instant::now();
        let mut successful_operations = 0;

        for _ in 0..50 {
            // Reduced count due to charge depletion
            if shocker.get_charge_level() > 0.2 {
                match shocker.generate_bioelectric_shock(&test_data, 0.3) {
                    Ok(_) => successful_operations += 1,
                    Err(_) => break, // Stop on insufficient charge
                }
            } else {
                break;
            }
        }

        let duration = start.elapsed();

        // Performance should be sub-millisecond per operation
        if successful_operations > 0 {
            let avg_time_per_op = duration.as_millis() as f64 / successful_operations as f64;
            assert!(
                avg_time_per_op < 1.0,
                "Average time per operation should be under 1ms"
            );
        }

        println!(
            "Performance test: {} operations in {}ms",
            successful_operations,
            duration.as_millis()
        );
    }

    /// Test 9: Individual component testing
    #[test]
    fn test_individual_components() {
        // Test Market Disruptor
        let disruptor = MarketDisruptor::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.15, 0.7);

        let disruption = disruptor.generate_shock(&market_data, 0.7).unwrap();
        assert!(disruption.voltage_generated > 0.0);
        assert!(disruption.disruption_radius > 0.0);
        assert!(!disruption.affected_price_levels.is_empty());

        // Test Hidden Liquidity Detector
        let detector = HiddenLiquidityDetector::new().unwrap();
        let _pools = detector
            .scan_for_hidden_liquidity(&market_data, &disruption)
            .unwrap();

        for pool in _pools {
            assert!(pool.confidence_score >= 0.0 && pool.confidence_score <= 1.0);
            assert!(pool.estimated_volume > 0.0);
        }

        // Test Shock Timing Optimizer
        let optimizer = ShockTimingOptimizer::new().unwrap();
        let timing = optimizer
            .calculate_optimal_timing(&market_data, &[market_data.clone()])
            .unwrap();

        assert!(timing.duration_ms > 0);
        assert!(timing.expected_effectiveness >= 0.0 && timing.expected_effectiveness <= 1.0);
        assert!(timing.start_time > chrono::Utc::now().timestamp_millis() as u64);
    }

    /// Test 10: Adaptation capabilities
    #[test]
    fn test_adaptation_capabilities() {
        let mut shocker = ElectricEelShocker::new().unwrap();

        // Test adaptation to different market conditions
        let high_vol_data = create_test_market_data(50000.0, 0.3, 0.5);
        let low_vol_data = create_test_market_data(50000.0, 0.05, 0.9);

        let _initial_sensitivity = shocker.get_sensitivity_level();

        // Adapt to high volatility
        shocker.adapt_to_conditions(&high_vol_data, 1000).unwrap();
        let high_vol_sensitivity = shocker.get_sensitivity_level();

        // Adapt to low volatility
        shocker.adapt_to_conditions(&low_vol_data, 2000).unwrap();
        let low_vol_sensitivity = shocker.get_sensitivity_level();

        // All sensitivities should be in valid range
        assert!(initial_sensitivity >= 0.1 && initial_sensitivity <= 1.0);
        assert!(high_vol_sensitivity >= 0.1 && high_vol_sensitivity <= 1.0);
        assert!(low_vol_sensitivity >= 0.1 && low_vol_sensitivity <= 1.0);

        println!(
            "Sensitivity adaptation: initial={:.2}, high_vol={:.2}, low_vol={:.2}",
            initial_sensitivity, high_vol_sensitivity, low_vol_sensitivity
        );
    }

    /// Test 11: Metrics collection and validation
    #[test]
    fn test_metrics_collection() {
        let shocker = ElectricEelShocker::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.1, 0.8);

        // Perform operations to generate metrics
        let mut operations_performed = 0;
        for _ in 0..10 {
            if shocker.get_charge_level() > 0.1 {
                match shocker.generate_bioelectric_shock(&test_data, 0.3) {
                    Ok(_) => operations_performed += 1,
                    Err(_) => break,
                }
            }
        }

        let metrics = shocker.get_metrics().unwrap();

        assert_eq!(metrics.total_operations, operations_performed as u64);
        assert!(metrics.average_processing_time_ns > 0);
        assert!(metrics.accuracy_rate >= 0.0 && metrics.accuracy_rate <= 1.0);

        // Validate custom metrics
        assert!(metrics
            .custom_metrics
            .contains_key("bioelectric_charge_level"));
        assert!(metrics
            .custom_metrics
            .contains_key("hidden_pools_discovered"));
        assert!(metrics.custom_metrics.contains_key("shock_intensity"));

        let charge_level = metrics.custom_metrics["bioelectric_charge_level"];
        assert!(charge_level >= 0.0 && charge_level <= 1.0);

        println!(
            "Metrics: ops={}, avg_time={}ns, accuracy={:.2}, charge={:.2}",
            metrics.total_operations,
            metrics.average_processing_time_ns,
            metrics.accuracy_rate,
            charge_level
        );
    }

    /// Test 12: Thread safety and concurrent operations
    #[test]
    fn test_thread_safety() {
        let shocker = Arc::new(ElectricEelShocker::new().unwrap());
        let test_data = create_test_market_data(50000.0, 0.1, 0.8);

        let handles: Vec<_> = (0..5)
            .map(|i| {
                let shocker_clone = Arc::clone(&shocker);
                let data_clone = test_data.clone();

                thread::spawn(move || {
                    let intensity = 0.2 + (i as f64) * 0.1; // Different intensities
                    if shocker_clone.get_charge_level() > intensity * 0.5 {
                        shocker_clone.generate_bioelectric_shock(&data_clone, intensity)
                    } else {
                        Err(Error::invalid_data("Insufficient charge"))
                    }
                })
            })
            .collect();

        let mut successful_operations = 0;
        let mut failed_operations = 0;

        for handle in handles {
            match handle.join().unwrap() {
                Ok(_) => successful_operations += 1,
                Err(_) => failed_operations += 1,
            }
        }

        println!(
            "Thread safety test: {} successful, {} failed operations",
            successful_operations, failed_operations
        );

        // At least some operations should succeed
        assert!(
            successful_operations > 0,
            "Some operations should succeed in concurrent access"
        );
    }

    /// Test 13: Pool type classification accuracy
    #[test]
    fn test_pool_type_classification() {
        let detector = HiddenLiquidityDetector::new().unwrap();
        let market_data = create_test_market_data(50000.0, 0.1, 0.8);

        // Create different disruption patterns to test classification
        let test_cases = vec![
            // Close price, high disruption -> IcebergOrder
            (market_data.price * 1.0001, 0.9, 0.6),
            // Far price, high info -> DarkPool
            (market_data.price * 1.03, 0.7, 0.8),
            // Medium distance, high disruption -> AlgorithmicBuffer
            (market_data.price * 1.01, 0.8, 0.5),
            // Far price, medium disruption -> InstitutionalReserve
            (market_data.price * 1.02, 0.6, 0.3),
        ];

        for (price, disruption_intensity, info_content) in test_cases {
            let level = parasitic::organisms::electric_eel::PriceLevel {
                price,
                disruption_intensity,
                information_content: info_content,
            };

            let pool_type = detector.classify_pool_type(&level, &market_data);

            // Should not classify as Unknown for well-defined test cases
            if disruption_intensity > 0.5 && info_content > 0.4 {
                assert_ne!(
                    pool_type,
                    PoolType::Unknown,
                    "Well-defined disruption pattern should be classified"
                );
            }
        }
    }

    /// Test 14: Error handling and edge cases
    #[test]
    fn test_error_handling() {
        let shocker = ElectricEelShocker::new().unwrap();

        // Test invalid parameters
        let invalid_market_data = MarketData {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            price: -100.0, // Invalid negative price
            volume: 1000.0,
            volatility: 0.1,
            bid: 50.0,
            ask: 51.0,
            spread_percent: 0.002,
            market_cap: Some(1000000000.0),
            liquidity_score: 0.8,
        };

        let result = shocker.generate_bioelectric_shock(&invalid_market_data, 0.5);
        assert!(result.is_err(), "Negative price should cause error");

        // Test invalid shock intensity
        let valid_market_data = create_test_market_data(50000.0, 0.1, 0.8);
        let result = shocker.generate_bioelectric_shock(&valid_market_data, 1.5);
        assert!(result.is_err(), "Shock intensity > 1.0 should cause error");

        // Test empty sequence
        let empty_results = shocker.generate_shock_sequence(&[], 0.5).unwrap();
        assert!(
            empty_results.is_empty(),
            "Empty sequence should return empty results"
        );
    }

    /// Test 15: Performance monitoring compliance
    #[test]
    fn test_performance_monitoring_compliance() {
        let shocker = ElectricEelShocker::new().unwrap();
        let test_data = create_test_market_data(50000.0, 0.1, 0.8);

        // Perform some operations
        for _ in 0..5 {
            if shocker.get_charge_level() > 0.1 {
                let _ = shocker.generate_bioelectric_shock(&test_data, 0.4);
            }
        }

        // Check performance requirements
        let meets_requirements = shocker.meets_requirements();
        println!("Performance requirements met: {}", meets_requirements);

        let stats = shocker.get_stats().unwrap();
        println!(
            "Performance stats: avg={}ns, max={}ns, throughput={:.2}ops/s",
            stats.avg_processing_time_ns,
            stats.max_processing_time_ns,
            stats.throughput_ops_per_sec
        );

        // Performance stats should be reasonable
        assert!(stats.avg_processing_time_ns > 0);
        assert!(stats.max_processing_time_ns >= stats.avg_processing_time_ns);
        assert!(stats.accuracy_rate >= 0.0 && stats.accuracy_rate <= 1.0);
    }
}

/// Integration tests for Electric Eel Shocker
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Integration test: Full workflow with multiple organisms
    #[test]
    fn test_full_workflow_integration() {
        // This would test integration with other organisms like Komodo Dragon
        // For now, we test the Electric Eel in isolation but with realistic scenarios

        let shocker = ElectricEelShocker::new().unwrap();
        let market_sequence = create_market_sequence(50000.0, 0.2, 10);

        // Simulate a full trading session
        let mut total_information_gained = 0.0;
        let mut total_pools_discovered = 0;

        for (i, market_data) in market_sequence.iter().enumerate() {
            if shocker.get_charge_level() > 0.2 {
                let intensity = 0.3 + (market_data.volatility * 0.5); // Adaptive intensity

                match shocker.generate_bioelectric_shock(market_data, intensity) {
                    Ok(result) => {
                        total_information_gained += result.information_gain;
                        total_pools_discovered += result.hidden_liquidity_pools.len();

                        println!(
                            "Step {}: intensity={:.2}, charge={:.2}, pools={}, info={:.3}",
                            i,
                            intensity,
                            result.bioelectric_charge_remaining,
                            result.hidden_liquidity_pools.len(),
                            result.information_gain
                        );
                    }
                    Err(e) => {
                        println!("Step {}: Operation failed: {}", i, e);
                        break;
                    }
                }

                // Simulate natural recharge between operations
                let _ = shocker.recharge_bioelectric_system(0.05);
            } else {
                // Force recharge when charge is too low
                let _ = shocker.recharge_bioelectric_system(0.3);
                println!("Step {}: Recharging bioelectric system", i);
            }
        }

        println!("Integration test results:");
        println!(
            "  Total information gained: {:.3}",
            total_information_gained
        );
        println!("  Total pools discovered: {}", total_pools_discovered);

        assert!(
            total_information_gained > 0.0,
            "Should gain some information"
        );

        // Final metrics
        let final_metrics = shocker.get_metrics().unwrap();
        println!("  Final operations: {}", final_metrics.total_operations);
        println!("  Final accuracy: {:.2}", final_metrics.accuracy_rate);
    }
}
