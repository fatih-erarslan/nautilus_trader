//! Integration tests for whale defense system
//! 
//! Comprehensive test suite validating sub-microsecond performance
//! and correctness under realistic conditions.

use whale_defense_core::{
    WhaleDefenseEngine, QuantumGameTheoryEngine, SteganographicOrderManager,
    MarketOrder, WhaleActivity, WhaleType, ThreatLevel, DefenseConfig,
    timing::Timestamp, config::*,
};
use std::time::Duration;
use std::thread;

/// Test complete whale defense workflow
#[tokio::test]
async fn test_complete_whale_defense_workflow() {
    unsafe {
        // Initialize system
        whale_defense_core::init().unwrap();
        
        let config = DefenseConfig::default();
        let mut engine = WhaleDefenseEngine::new(config).unwrap();
        engine.start().unwrap();
        
        // Create whale-sized order
        let whale_order = MarketOrder::new(100.0, 10000000.0, 1, 1, 0);
        
        // Process order and measure performance
        let start_time = Timestamp::now();
        let result = engine.process_market_order(whale_order).unwrap();
        let elapsed_time = start_time.elapsed_nanos();
        
        // Validate result
        assert!(result.is_some(), "Whale order should be detected");
        let defense_result = result.unwrap();
        assert!(defense_result.success, "Defense should succeed");
        
        // Validate performance
        assert!(
            elapsed_time < TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS,
            "Total defense time {} ns exceeds target {} ns",
            elapsed_time,
            TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS
        );
        
        // Validate sub-component performance
        assert!(
            defense_result.metrics.detection_latency_ns < TARGET_DETECTION_LATENCY_NS,
            "Detection latency {} ns exceeds target {} ns",
            defense_result.metrics.detection_latency_ns,
            TARGET_DETECTION_LATENCY_NS
        );
        
        assert!(
            defense_result.metrics.total_time_ns < TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS,
            "Total execution time {} ns exceeds target {} ns",
            defense_result.metrics.total_time_ns,
            TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS
        );
        
        engine.shutdown().unwrap();
        whale_defense_core::shutdown();
    }
}

/// Test quantum game theory engine performance
#[test]
fn test_quantum_game_theory_performance() {
    unsafe {
        let engine = QuantumGameTheoryEngine::new().unwrap();
        
        let whale_strategy = [0.7, 0.2, 0.05, 0.05]; // Aggressive whale
        let whale_size = 50000000.0; // $50M position
        let threat_level = ThreatLevel::Critical;
        
        // Measure strategy calculation time
        let start_time = Timestamp::now();
        let counter_strategy = engine.calculate_optimal_strategy(
            whale_strategy,
            whale_size,
            threat_level,
        ).unwrap();
        let elapsed_time = start_time.elapsed_nanos();
        
        // Validate performance
        assert!(
            elapsed_time < TARGET_DEFENSE_EXECUTION_NS / 2, // Half of defense budget
            "Strategy calculation time {} ns exceeds target {} ns",
            elapsed_time,
            TARGET_DEFENSE_EXECUTION_NS / 2
        );
        
        // Validate strategy correctness
        let sum: f64 = counter_strategy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Strategy must sum to 1.0");
        assert!(counter_strategy.iter().all(|&x| x >= 0.0), "All allocations must be non-negative");
        
        // Validate strategy makes sense for critical threat
        // Should prefer conservative/stealth approaches
        assert!(
            counter_strategy[2] + counter_strategy[3] > 0.5,
            "Critical threat should favor conservative/stealth strategies"
        );
        
        // Test multiple iterations for consistency
        for _ in 0..1000 {
            let start = Timestamp::now();
            let strategy = engine.calculate_optimal_strategy(
                whale_strategy,
                whale_size,
                threat_level,
            ).unwrap();
            let time = start.elapsed_nanos();
            
            assert!(time < TARGET_DEFENSE_EXECUTION_NS / 2);
            let sum: f64 = strategy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}

/// Test steganographic order manager performance
#[test]
fn test_steganographic_order_performance() {
    use whale_defense_core::core::{DefenseStrategy, WhaleSignal};
    
    unsafe {
        let manager = SteganographicOrderManager::new().unwrap();
        
        let strategy = DefenseStrategy {
            signal: WhaleSignal::HideAndWait,
            urgency: 0.8,
            allocation: [0.1, 0.2, 0.3, 0.4], // Favor stealth
            effectiveness: 0.9,
            timestamp: Timestamp::now(),
        };
        
        let activity = WhaleActivity {
            timestamp: Timestamp::now(),
            whale_type: WhaleType::Accumulation,
            volume: 10000000.0,
            price_impact: 0.5,
            momentum: 0.3,
            confidence: 0.9,
            threat_level: ThreatLevel::High,
        };
        
        // Test all stealth levels
        for stealth_level in 0..=3 {
            let start_time = Timestamp::now();
            let orders = manager.generate_defense_orders(&strategy, &activity, stealth_level).unwrap();
            let elapsed_time = start_time.elapsed_nanos();
            
            // Validate performance
            assert!(
                elapsed_time < TARGET_DEFENSE_EXECUTION_NS / 2,
                "Order generation time {} ns exceeds target {} ns for stealth level {}",
                elapsed_time,
                TARGET_DEFENSE_EXECUTION_NS / 2,
                stealth_level
            );
            
            // Validate order characteristics
            assert!(!orders.is_empty(), "Should generate at least one order");
            assert!(orders.len() <= 16, "Should not generate excessive orders");
            
            // Validate stealth characteristics
            if stealth_level > 0 {
                assert!(
                    orders.iter().any(|o| o.hidden_quantity > 0.0),
                    "Stealth level {} should have hidden quantities",
                    stealth_level
                );
            }
            
            // Validate total volume allocation
            let total_visible: f64 = orders.iter().map(|o| o.quantity).sum();
            let total_hidden: f64 = orders.iter().map(|o| o.hidden_quantity).sum();
            let total_volume = total_visible + total_hidden;
            
            assert!(
                total_volume > 0.0,
                "Should allocate some volume for defense"
            );
        }
    }
}

/// Test concurrent whale detection under load
#[test]
fn test_concurrent_whale_detection() {
    unsafe {
        whale_defense_core::init().unwrap();
        
        let config = DefenseConfig::default();
        let engine = std::sync::Arc::new(WhaleDefenseEngine::new(config).unwrap());
        engine.start().unwrap();
        
        let engine_clone = engine.clone();
        
        // Spawn multiple threads processing orders concurrently
        let handles: Vec<_> = (0..4).map(|thread_id| {
            let engine = engine_clone.clone();
            thread::spawn(move || {
                let mut performance_violations = 0;
                let mut total_orders = 0;
                
                for i in 0..1000 {
                    let order = MarketOrder::new(
                        100.0 + (i as f64 * 0.01),
                        1000.0 + (i as f64 * (thread_id + 1) as f64 * 1000.0),
                        1,
                        (i % 10) as u16,
                        0,
                    );
                    
                    let start_time = Timestamp::now();
                    let result = engine.process_market_order(order);
                    let elapsed_time = start_time.elapsed_nanos();
                    
                    if elapsed_time > TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS {
                        performance_violations += 1;
                    }
                    
                    total_orders += 1;
                    
                    // Ensure result is valid
                    assert!(result.is_ok(), "Order processing should not fail");
                }
                
                (performance_violations, total_orders)
            })
        }).collect();
        
        // Collect results
        let mut total_violations = 0;
        let mut total_orders = 0;
        
        for handle in handles {
            let (violations, orders) = handle.join().unwrap();
            total_violations += violations;
            total_orders += orders;
        }
        
        // Validate performance under concurrent load
        let violation_rate = total_violations as f64 / total_orders as f64;
        assert!(
            violation_rate < 0.01, // Less than 1% violations acceptable
            "Performance violation rate {:.2}% exceeds 1% threshold ({} violations out of {} orders)",
            violation_rate * 100.0,
            total_violations,
            total_orders
        );
        
        // Get final performance statistics
        let (detections, defenses, avg_latency) = engine.get_performance_stats();
        assert!(detections > 0, "Should have processed some orders");
        assert!(
            avg_latency < (TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS) as f64,
            "Average latency {:.0} ns exceeds target {} ns",
            avg_latency,
            TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS
        );
        
        whale_defense_core::shutdown();
    }
}

/// Test lock-free data structures under stress
#[test]
fn test_lockfree_structures_stress() {
    use whale_defense_core::lockfree::*;
    
    unsafe {
        // Test ring buffer
        let mut buffer = LockFreeRingBuffer::<u64>::new(1024).unwrap();
        
        // Fill and drain buffer multiple times
        for round in 0..100 {
            // Fill buffer
            for i in 0..512 {
                let value = (round * 1000 + i) as u64;
                assert!(buffer.try_write(value).is_ok(), "Write should succeed");
            }
            
            // Drain buffer
            for i in 0..512 {
                let expected = (round * 1000 + i) as u64;
                let actual = buffer.try_read().unwrap();
                assert_eq!(actual, expected, "Values should match");
            }
        }
        
        buffer.destroy();
        
        // Test queue
        let queue = LockFreeQueue::<u64>::new();
        
        for round in 0..100 {
            // Enqueue items
            for i in 0..100 {
                let value = (round * 100 + i) as u64;
                assert!(queue.enqueue(value).is_ok(), "Enqueue should succeed");
            }
            
            // Dequeue items
            for i in 0..100 {
                let expected = (round * 100 + i) as u64;
                let actual = queue.dequeue().unwrap();
                assert_eq!(actual, expected, "FIFO order should be maintained");
            }
        }
        
        // Test stack
        let stack = LockFreeStack::<u64>::new();
        
        for round in 0..100 {
            // Push items
            for i in 0..100 {
                let value = (round * 100 + i) as u64;
                assert!(stack.push(value).is_ok(), "Push should succeed");
            }
            
            // Pop items (LIFO order)
            for i in (0..100).rev() {
                let expected = (round * 100 + i) as u64;
                let actual = stack.pop().unwrap();
                assert_eq!(actual, expected, "LIFO order should be maintained");
            }
        }
    }
}

/// Test SIMD operations correctness and performance
#[cfg(feature = "simd")]
#[test]
fn test_simd_operations() {
    use whale_defense_core::simd::*;
    
    let features = SimdFeatures::detect();
    println!("SIMD Features: AVX-512F: {}, AVX2: {}, SSE4.1: {}", 
             features.has_avx512f, features.has_avx2, features.has_sse41);
    
    let dispatcher = SimdDispatcher::new();
    
    // Test whale pattern matching
    let prices = vec![100.0, 101.0, 99.0, 150.0, 98.0, 200.0, 97.0, 95.0];
    let volumes = vec![1000.0, 2000.0, 1500.0, 10000.0, 1200.0, 50000.0, 1800.0, 900.0];
    let thresholds = [5000.0, 120.0, 500000.0, 0.05];
    
    let start_time = Timestamp::now();
    let results = dispatcher.dispatch_whale_pattern_match(&prices, &volumes, &thresholds).unwrap();
    let elapsed_time = start_time.elapsed_nanos();
    
    assert_eq!(results.len(), prices.len());
    
    // Validate performance (should be very fast)
    assert!(elapsed_time < 1000, "SIMD whale pattern matching should complete in <1μs");
    
    // Validate results - should detect whales at indices 3 and 5
    assert!(results[3], "Should detect whale at index 3 (high volume)");
    assert!(results[5], "Should detect whale at index 5 (very high volume)");
    
    // Test moving average
    if features.has_avx2 {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        unsafe {
            let start_time = Timestamp::now();
            let averages = simd_moving_average(&data, 3).unwrap();
            let elapsed_time = start_time.elapsed_nanos();
            
            assert!(elapsed_time < 500, "SIMD moving average should complete in <500ns");
            assert_eq!(averages.len(), data.len() - 3 + 1);
            
            // Validate first average: (1+2+3)/3 = 2.0
            assert!((averages[0] - 2.0).abs() < 1e-10);
        }
    }
}

/// Test memory and cache performance
#[test]
fn test_memory_cache_performance() {
    use whale_defense_core::cache::*;
    
    // Test cache warm-up
    let start_time = Timestamp::now();
    warm_up_caches();
    let warmup_time = start_time.elapsed_nanos();
    
    // Warm-up should complete quickly
    assert!(warmup_time < 10_000_000, "Cache warm-up should complete in <10ms"); // 10ms
    
    // Test cache-optimized memcpy
    let src = vec![0xAAu8; 4096];
    let mut dst = vec![0u8; 4096];
    
    unsafe {
        let start_time = Timestamp::now();
        cache_optimized_memcpy(dst.as_mut_ptr(), src.as_ptr(), 4096);
        let copy_time = start_time.elapsed_nanos();
        
        // Should be very fast for 4KB
        assert!(copy_time < 10000, "4KB cache-optimized copy should complete in <10μs");
        
        // Validate correctness
        assert_eq!(src, dst, "Copied data should match source");
    }
    
    // Test prefetch operations (should not crash)
    let data = vec![0u64; 1024];
    for locality in [Locality::None, Locality::Low, Locality::Medium, Locality::High] {
        prefetch_data(data.as_ptr(), locality);
    }
}

/// Test timing precision and accuracy
#[test]
fn test_timing_precision() {
    use whale_defense_core::timing::*;
    
    // Calibrate TSC if not already done
    calibrate_tsc().unwrap();
    
    // Test timestamp precision
    let ts1 = Timestamp::now();
    std::thread::sleep(Duration::from_nanos(100)); // Small sleep
    let ts2 = Timestamp::now();
    
    let elapsed_cycles = ts2.elapsed_cycles();
    let elapsed_nanos = ts1.elapsed_nanos();
    
    assert!(elapsed_cycles > 0, "Should measure some elapsed cycles");
    assert!(elapsed_nanos >= 100, "Should measure at least 100ns elapsed time");
    
    // Test performance timer
    let timer = PerfTimer::start("test");
    
    // Small computation
    let mut sum = 0;
    for i in 0..1000 {
        sum += i;
    }
    std::hint::black_box(sum);
    
    let elapsed = timer.stop();
    assert!(elapsed > 0, "Should measure some computation time");
    assert!(elapsed < 1_000_000, "Simple computation should complete in <1ms");
}

/// Test error handling and recovery
#[test]
fn test_error_handling() {
    use whale_defense_core::error::*;
    
    // Test error code consistency
    assert_eq!(WhaleDefenseError::NotInitialized.code(), 0);
    assert_eq!(WhaleDefenseError::OutOfMemory.code(), 2);
    assert_eq!(WhaleDefenseError::Unknown.code(), 255);
    
    // Test error recovery flags
    assert!(!WhaleDefenseError::NotInitialized.is_recoverable());
    assert!(WhaleDefenseError::BufferOverflow.is_recoverable());
    
    // Test performance-critical error classification
    assert!(WhaleDefenseError::PerformanceThresholdExceeded.is_performance_critical());
    assert!(!WhaleDefenseError::InvalidParameter.is_performance_critical());
    
    // Test atomic error handler
    let handler = AtomicErrorHandler::new();
    handler.handle_fast_error(WhaleDefenseError::DetectionError);
    handler.handle_fast_error(WhaleDefenseError::DefenseError);
    
    let (count, last_error) = handler.get_stats();
    assert_eq!(count, 2);
    assert_eq!(last_error, WhaleDefenseError::DefenseError);
}

/// Comprehensive system integration test
#[test]
fn test_comprehensive_system_integration() {
    unsafe {
        // Initialize complete system
        whale_defense_core::init().unwrap();
        
        let config = DefenseConfig {
            detection_sensitivity: 0.7,
            volume_threshold: 3.0,
            price_impact_threshold: 0.02,
            momentum_threshold: 0.08,
            confidence_threshold: 0.8,
            max_concurrent_defenses: 8,
            performance_monitoring: true,
        };
        
        let mut engine = WhaleDefenseEngine::new(config).unwrap();
        engine.start().unwrap();
        
        // Test various market scenarios
        let test_scenarios = vec![
            ("normal_order", MarketOrder::new(100.0, 1000.0, 1, 1, 0), false),
            ("large_order", MarketOrder::new(100.0, 100000.0, 1, 1, 0), true),
            ("whale_order", MarketOrder::new(100.0, 10000000.0, 1, 1, 0), true),
            ("mega_whale", MarketOrder::new(100.0, 100000000.0, 1, 1, 0), true),
        ];
        
        let mut total_processing_time = 0u64;
        let mut whale_detections = 0;
        
        for (scenario_name, order, should_detect) in test_scenarios {
            let start_time = Timestamp::now();
            let result = engine.process_market_order(order);
            let processing_time = start_time.elapsed_nanos();
            
            total_processing_time += processing_time;
            
            // Validate result
            assert!(result.is_ok(), "Processing should succeed for {}", scenario_name);
            
            let defense_result = result.unwrap();
            if should_detect {
                assert!(defense_result.is_some(), "Should detect whale for {}", scenario_name);
                whale_detections += 1;
                
                let defense = defense_result.unwrap();
                assert!(defense.success, "Defense should succeed for {}", scenario_name);
                
                // Validate performance for whale scenarios
                assert!(
                    processing_time < TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS,
                    "Processing time {} ns exceeds target for {}",
                    processing_time,
                    scenario_name
                );
            } else {
                // Normal orders should still be fast but might not trigger defense
                assert!(
                    processing_time < TARGET_DETECTION_LATENCY_NS,
                    "Normal order processing time {} ns exceeds detection target for {}",
                    processing_time,
                    scenario_name
                );
            }
        }
        
        // Validate overall system performance
        assert!(whale_detections >= 3, "Should detect at least 3 whale scenarios");
        
        let avg_processing_time = total_processing_time / test_scenarios.len() as u64;
        assert!(
            avg_processing_time < TARGET_DETECTION_LATENCY_NS + TARGET_DEFENSE_EXECUTION_NS,
            "Average processing time {} ns exceeds targets",
            avg_processing_time
        );
        
        // Get final performance statistics
        let (detections, defenses, avg_latency) = engine.get_performance_stats();
        assert_eq!(detections as usize, test_scenarios.len());
        assert_eq!(defenses, whale_detections);
        
        engine.shutdown().unwrap();
        whale_defense_core::shutdown();
    }
}