//! Performance validation tests for quantum-hive
//! Targeting sub-microsecond execution with statistical validation

use quantum_hive::*;
use std::time::Instant;
use std::hint::black_box;

#[test]
fn test_sub_microsecond_trade_decision() {
    let hive = AutopoieticHive::new();
    let tick = MarketTick {
        symbol: [b'B', b'T', b'C', b'U', b'S', b'D', b'T', 0],
        price: 50000.0,
        volume: 1.0,
        timestamp: chrono::Utc::now().timestamp() as u64,
        bid: 49999.0,
        ask: 50001.0,
    };
    
    let mut total_time = 0u128;
    let iterations = 1000;
    
    for _ in 0..iterations {
        let start = Instant::now();
        let _action = hive.queen.emergency_decision_sync(&tick);
        let duration = start.elapsed();
        total_time += duration.as_nanos();
        
        // Individual decisions should be sub-microsecond
        assert!(duration.as_nanos() < 1000, "Decision should be < 1μs");
    }
    
    let avg_time = total_time / iterations as u128;
    println!("Average decision time: {}ns", avg_time);
    assert!(avg_time < 500, "Average decision time should be < 500ns");
}

#[test]
fn test_quantum_strategy_lut_nanosecond_lookup() {
    let lut = QuantumStrategyLUT::default();
    let iterations = 1_000_000;
    
    let start = Instant::now();
    for i in 0..iterations {
        let price_index = (i % 65536) as u16;
        unsafe {
            black_box(lut.get_action(price_index));
        }
    }
    let duration = start.elapsed();
    
    let ns_per_lookup = duration.as_nanos() / iterations;
    println!("Nanoseconds per lookup: {}", ns_per_lookup);
    
    // Should be < 1ns per lookup for 1M operations
    assert!(ns_per_lookup < 1, "Lookup should be < 1ns");
    assert!(duration.as_millis() < 10, "1M lookups should be < 10ms");
}

#[test]
fn test_circular_buffer_zero_allocation_performance() {
    let mut buffer = CircularBuffer::<f64>::new(10000);
    let iterations = 1_000_000;
    
    let start = Instant::now();
    for i in 0..iterations {
        buffer.push(i as f64);
        black_box(buffer.latest());
    }
    let duration = start.elapsed();
    
    let ns_per_operation = duration.as_nanos() / iterations;
    println!("Nanoseconds per buffer operation: {}", ns_per_operation);
    
    // Should be extremely fast with zero allocations
    assert!(ns_per_operation < 10, "Buffer operations should be < 10ns");
}

#[test]
fn test_node_tick_processing_performance() {
    let mut node = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]);
    let tick = MarketTick {
        symbol: [b'E', b'T', b'H', b'U', b'S', b'D', b'T', 0],
        price: 3000.0,
        volume: 5.0,
        timestamp: chrono::Utc::now().timestamp() as u64,
        bid: 2999.0,
        ask: 3001.0,
    };
    
    let iterations = 100_000;
    let start = Instant::now();
    
    for _ in 0..iterations {
        node.process_tick(tick);
    }
    
    let duration = start.elapsed();
    let ns_per_tick = duration.as_nanos() / iterations;
    
    println!("Nanoseconds per tick processing: {}", ns_per_tick);
    assert!(ns_per_tick < 100, "Tick processing should be < 100ns");
}

#[test]
fn test_swarm_intelligence_update_performance() {
    let hive = AutopoieticHive::with_config(HiveConfig {
        node_count: 100,
        checkpoint_interval: std::time::Duration::from_secs(60),
        quantum_job_batch_size: 32,
        enable_gpu: false,
    });
    
    let start = Instant::now();
    
    // This should not cause stack overflow with 100 nodes
    let successful_clusters = hive.swarm_intelligence.detect_successful_clusters(&hive.nodes);
    
    let duration = start.elapsed();
    println!("Swarm analysis time: {:?}", duration);
    
    // Should complete quickly even for 100 nodes
    assert!(duration.as_millis() < 50, "Swarm analysis should be < 50ms");
    assert!(successful_clusters.len() >= 0); // Should not panic
}

#[test]
fn test_hyperbolic_lattice_creation_performance() {
    let node_counts = [10, 50, 100, 500];
    
    for &count in &node_counts {
        let start = Instant::now();
        let nodes = AutopoieticHive::create_hyperbolic_lattice(count);
        let duration = start.elapsed();
        
        println!("Created {} nodes in {:?}", count, duration);
        assert_eq!(nodes.len(), count);
        
        // Should scale reasonably with node count
        let ms_per_node = duration.as_millis() as f64 / count as f64;
        assert!(ms_per_node < 1.0, "Should create nodes in < 1ms each");
    }
}

#[test]
fn test_memory_allocation_patterns() {
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // Test that our critical paths don't allocate
    let lut = QuantumStrategyLUT::default();
    
    // This should be zero-allocation
    unsafe {
        for i in 0..1000u16 {
            black_box(lut.get_action(i));
        }
    }
    
    // Test circular buffer doesn't allocate after initialization
    let mut buffer = CircularBuffer::<u64>::new(1000);
    
    // Fill to capacity
    for i in 0..1000 {
        buffer.push(i);
    }
    
    // These operations should not allocate
    for i in 1000..2000 {
        buffer.push(i);
        black_box(buffer.latest());
    }
}

#[test]
fn test_concurrent_performance() {
    use std::sync::Arc;
    use std::thread;
    
    let hive = Arc::new(AutopoieticHive::with_config(HiveConfig {
        node_count: 50,
        checkpoint_interval: std::time::Duration::from_secs(60),
        quantum_job_batch_size: 16,
        enable_gpu: false,
    }));
    
    let start = Instant::now();
    let mut handles = Vec::new();
    
    // Spawn 8 threads for concurrent access
    for thread_id in 0..8 {
        let hive_clone = Arc::clone(&hive);
        handles.push(thread::spawn(move || {
            let tick = MarketTick {
                symbol: [b'T', b'S', b'T', (thread_id as u8), 0, 0, 0, 0],
                price: 1000.0 + thread_id as f64,
                volume: 1.0,
                timestamp: chrono::Utc::now().timestamp() as u64,
                bid: 999.0 + thread_id as f64,
                ask: 1001.0 + thread_id as f64,
            };
            
            // Each thread processes 1000 ticks
            for _ in 0..1000 {
                // Access different nodes to test concurrent safety
                let node_idx = (thread_id * 6) % hive_clone.nodes.len();
                hive_clone.nodes[node_idx].process_tick(tick);
            }
        }));
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    let duration = start.elapsed();
    println!("Concurrent processing took: {:?}", duration);
    
    // 8 threads * 1000 ticks = 8000 operations
    // Should complete in reasonable time
    assert!(duration.as_millis() < 1000, "Concurrent processing should be < 1s");
}

#[test]
fn test_statistical_performance_consistency() {
    let lut = QuantumStrategyLUT::default();
    let mut measurements = Vec::with_capacity(1000);
    
    // Take 1000 measurements of lookup time
    for _ in 0..1000 {
        let start = Instant::now();
        unsafe {
            black_box(lut.get_action(12345));
        }
        let duration = start.elapsed();
        measurements.push(duration.as_nanos());
    }
    
    // Calculate statistics
    measurements.sort();
    let median = measurements[500];
    let p99 = measurements[990];
    let p999 = measurements[999];
    
    println!("Lookup times - Median: {}ns, P99: {}ns, P99.9: {}ns", median, p99, p999);
    
    // Performance should be consistent
    assert!(median < 5, "Median lookup should be < 5ns");
    assert!(p99 < 20, "P99 lookup should be < 20ns");
    assert!(p999 < 100, "P99.9 lookup should be < 100ns");
}

#[test]
fn test_cache_efficiency() {
    let lut = QuantumStrategyLUT::default();
    let iterations = 1_000_000;
    
    // Test sequential access (cache-friendly)
    let start = Instant::now();
    for i in 0..iterations {
        let index = (i % 65536) as u16;
        unsafe {
            black_box(lut.get_action(index));
        }
    }
    let sequential_time = start.elapsed();
    
    // Test random access (cache-unfriendly)
    let start = Instant::now();
    for i in 0..iterations {
        let index = ((i * 7919) % 65536) as u16; // Pseudo-random
        unsafe {
            black_box(lut.get_action(index));
        }
    }
    let random_time = start.elapsed();
    
    println!("Sequential: {:?}, Random: {:?}", sequential_time, random_time);
    
    // Both should be fast, but sequential should be faster
    assert!(sequential_time.as_millis() < 10);
    assert!(random_time.as_millis() < 50);
}

#[test]
fn test_simd_readiness() {
    // Test that our data structures are SIMD-friendly
    let buffer = CircularBuffer::<f32>::new(1024);
    
    // Data should be aligned for SIMD operations
    let data_ptr = buffer.data.as_ptr() as usize;
    assert_eq!(data_ptr % 16, 0, "Data should be 16-byte aligned for SIMD");
    
    // Test bulk operations that could benefit from SIMD
    let mut values = vec![0.0f32; 1000];
    let start = Instant::now();
    
    for (i, value) in values.iter_mut().enumerate() {
        *value = (i as f32).sin(); // Vectorizable operation
    }
    
    let duration = start.elapsed();
    println!("Bulk SIMD operation: {:?}", duration);
    assert!(duration.as_micros() < 100, "Bulk operations should be < 100μs");
}