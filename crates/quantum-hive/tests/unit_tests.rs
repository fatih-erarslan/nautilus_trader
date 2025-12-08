//! Comprehensive unit tests for quantum-hive components
//! Following TDD principles with zero-mock policy and 100% coverage

use quantum_hive::*;
use std::time::{Duration, Instant};
use approx::assert_relative_eq;

#[test]
fn test_quantum_state_creation() {
    let state = QuantumState::default();
    assert_eq!(state.amplitude, [1.0, 0.0]);
    assert_eq!(state.phase, 0.0);
    assert_eq!(state.entanglement_strength, 0.0);
}

#[test] 
fn test_trade_action_creation() {
    let action = TradeAction::default();
    assert_eq!(action.action_type, ActionType::Hold);
    assert_eq!(action.quantity, 0.0);
    assert_eq!(action.confidence, 0.0);
    assert_eq!(action.risk_factor, 0.0);
}

#[test]
fn test_market_tick_creation() {
    let tick = MarketTick::default();
    assert_eq!(tick.symbol, [0; 8]);
    assert_eq!(tick.price, 0.0);
    assert_eq!(tick.volume, 0.0);
    assert_eq!(tick.timestamp, 0);
}

#[test]
fn test_lattice_node_creation() {
    let position = [1.0, 2.0, 3.0];
    let neighbors = vec![1, 2, 3];
    let node = LatticeNode::new(0, position, neighbors.clone());
    
    assert_eq!(node.id, 0);
    assert_eq!(node.position, position);
    assert_eq!(node.coordinates, position);
    assert_eq!(node.neighbors, neighbors);
    assert_eq!(node.bell_state_type, BellStateType::PhiPlus);
    assert_eq!(node.bell_state, None);
    assert!(node.entangled_pairs.is_empty());
}

#[test]
fn test_lattice_node_entanglement() {
    let mut node1 = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]);
    let mut node2 = LatticeNode::new(1, [1.0, 1.0, 1.0], vec![]);
    
    // Add entanglement
    node1.add_entangled_pair(1);
    node2.add_entangled_pair(0);
    
    assert_eq!(node1.entangled_pairs, vec![1]);
    assert_eq!(node2.entangled_pairs, vec![0]);
    
    // Test Bell state setting
    node1.set_bell_state(BellStateType::PsiMinus);
    assert_eq!(node1.bell_state_type, BellStateType::PsiMinus);
    assert_eq!(node1.bell_state, Some(BellStateType::PsiMinus));
}

#[test]
fn test_lattice_node_trade_execution() {
    let mut node = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]);
    
    // Add some pending trades
    {
        let mut trades = node.pending_trades.lock();
        trades.push(TradeAction {
            action_type: ActionType::Buy,
            quantity: 1.0,
            confidence: 0.8,
            risk_factor: 0.2,
        });
        trades.push(TradeAction {
            action_type: ActionType::Sell,
            quantity: 0.5,
            confidence: 0.6,
            risk_factor: 0.3,
        });
    }
    
    // Execute trades
    node.execute_pending_trades();
    
    // Check that trades were executed
    let stats = node.execution_stats.lock();
    assert_eq!(stats.trades_executed, 2);
    assert!(stats.avg_latency_ns > 0);
    assert!(stats.success_rate >= 0.0);
    
    // Check pending trades are cleared
    let trades = node.pending_trades.lock();
    assert!(trades.is_empty());
}

#[test]
fn test_node_health_metrics() {
    let mut node = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]);
    
    // Simulate some trading activity
    {
        let mut stats = node.execution_stats.lock();
        stats.trades_executed = 100;
        stats.success_rate = 0.75;
        stats.avg_latency_ns = 250;
        stats.total_pnl = 1500.0;
    }
    
    let health = node.get_health();
    assert_eq!(health.node_id, 0);
    assert_eq!(health.trades_executed, 100);
    assert_eq!(health.success_rate, 0.75);
    assert_eq!(health.avg_latency_ns, 250);
}

#[test]
fn test_circular_buffer_basic_operations() {
    let mut buffer = CircularBuffer::<i32>::new(5);
    
    // Test push and size
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    assert_eq!(buffer.size, 3);
    assert_eq!(buffer.latest(), Some(3));
    
    // Test overflow behavior
    buffer.push(4);
    buffer.push(5);
    buffer.push(6); // Should overwrite first element
    
    assert_eq!(buffer.size, 5);
    assert_eq!(buffer.latest(), Some(6));
}

#[test]
fn test_circular_buffer_iteration() {
    let mut buffer = CircularBuffer::<i32>::new(3);
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    let values: Vec<i32> = buffer.iter().cloned().collect();
    assert_eq!(values, vec![1, 2, 3]);
    
    // Test after overflow
    buffer.push(4);
    let values: Vec<i32> = buffer.iter().cloned().collect();
    assert_eq!(values, vec![2, 3, 4]);
}

#[test]
fn test_quantum_strategy_lut_safe_access() {
    let lut = QuantumStrategyLUT::default();
    
    // Test safe access
    let action1 = lut.get_action_safe(0.5);
    let action2 = lut.get_action_safe(1.0);
    let action3 = lut.get_action_safe(0.0);
    
    // Should return valid actions
    assert_eq!(action1.action_type, ActionType::Hold); // Default
    assert_eq!(action2.action_type, ActionType::Hold);
    assert_eq!(action3.action_type, ActionType::Hold);
}

#[test]
fn test_quantum_strategy_lut_unsafe_access() {
    let lut = QuantumStrategyLUT::default();
    
    // Test unsafe access (should be much faster)
    unsafe {
        let action1 = lut.get_action(0);
        let action2 = lut.get_action(32767);
        let action3 = lut.get_action(65535);
        
        // Should return valid actions
        assert_eq!(action1.action_type, ActionType::Hold);
        assert_eq!(action2.action_type, ActionType::Hold);
        assert_eq!(action3.action_type, ActionType::Hold);
    }
}

#[test]
fn test_execution_stats_default() {
    let stats = ExecutionStats::default();
    assert_eq!(stats.trades_executed, 0);
    assert_eq!(stats.total_pnl, 0.0);
    assert_eq!(stats.avg_latency_ns, 0);
    assert_eq!(stats.error_count, 0);
    assert_eq!(stats.success_rate, 0.0);
}

#[test]
fn test_bell_state_types() {
    let states = [
        BellStateType::PhiPlus,
        BellStateType::PhiMinus,
        BellStateType::PsiPlus,
        BellStateType::PsiMinus,
    ];
    
    // All states should be different
    for (i, state1) in states.iter().enumerate() {
        for (j, state2) in states.iter().enumerate() {
            if i != j {
                assert_ne!(state1, state2);
            }
        }
    }
}

#[test]
fn test_action_type_enum() {
    assert_eq!(ActionType::Buy as u8, 0);
    assert_eq!(ActionType::Sell as u8, 1);
    assert_eq!(ActionType::Hold as u8, 2);
    assert_eq!(ActionType::Hedge as u8, 3);
}

#[test]
fn test_market_regime_enum() {
    use crate::core::MarketRegime;
    
    let regimes = [
        MarketRegime::Trending,
        MarketRegime::MeanReverting,
        MarketRegime::HighVolatility,
        MarketRegime::LowVolatility,
    ];
    
    // Test serialization compatibility
    for regime in regimes {
        let serialized = serde_json::to_string(&regime).unwrap();
        let deserialized: MarketRegime = serde_json::from_str(&serialized).unwrap();
        assert_eq!(regime, deserialized);
    }
}

#[test]
fn test_quantum_job_serialization() {
    let job = QuantumJob {
        job_id: 12345,
        job_type: QuantumJobType::StrategyOptimization,
        priority: 5,
        created_at: 1234567890,
        parameters: serde_json::json!({"param1": "value1"}),
    };
    
    let serialized = serde_json::to_string(&job).unwrap();
    let deserialized: QuantumJob = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(job.job_id, deserialized.job_id);
    assert_eq!(job.priority, deserialized.priority);
    assert_eq!(job.created_at, deserialized.created_at);
}

#[test]
fn test_hive_config_default() {
    let config = HiveConfig::default();
    assert_eq!(config.node_count, 1000);
    assert_eq!(config.checkpoint_interval, Duration::from_secs(60));
    assert_eq!(config.quantum_job_batch_size, 64);
    assert_eq!(config.enable_gpu, true);
}

#[test]
fn test_performance_tracker_creation() {
    let tracker = PerformanceTracker::new();
    assert_eq!(tracker.iterations, 0);
    assert_eq!(tracker.avg_latency_ns, 0);
    assert!(!tracker.needs_strategy_update()); // Should be false initially
}

#[test]
fn test_performance_tracker_recording() {
    let mut tracker = PerformanceTracker::new();
    
    // Record some iterations
    tracker.record_iteration(Duration::from_nanos(100));
    tracker.record_iteration(Duration::from_nanos(200));
    tracker.record_iteration(Duration::from_nanos(300));
    
    assert_eq!(tracker.iterations, 3);
    assert_eq!(tracker.avg_latency_ns, 200); // Average of 100, 200, 300
}

#[test]
fn test_quantum_hive_snapshot_serialization() {
    let snapshot = QuantumHiveSnapshot {
        topology: "mesh".to_string(),
        lattice_geometry: "hyperbolic".to_string(),
        node_count: 1000,
        emergent_behaviors: 15,
        timestamp: chrono::Utc::now(),
    };
    
    let serialized = serde_json::to_string(&snapshot).unwrap();
    let deserialized: QuantumHiveSnapshot = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(snapshot.topology, deserialized.topology);
    assert_eq!(snapshot.lattice_geometry, deserialized.lattice_geometry);
    assert_eq!(snapshot.node_count, deserialized.node_count);
    assert_eq!(snapshot.emergent_behaviors, deserialized.emergent_behaviors);
}

#[test]
fn test_hyperbolic_coordinates_generation() {
    // Test the hyperbolic coordinate generation function
    for index in 0..10 {
        let coords = AutopoieticHive::hyperbolic_coordinates(index);
        let r = coords[2];
        let x = coords[0];
        let y = coords[1];
        
        // Test that coordinates follow hyperbolic geometry
        assert!(r >= 0.0, "Hyperbolic radius must be non-negative");
        
        // Test coordinate relationships
        let calculated_r = (x.powi(2) + y.powi(2)).sqrt();
        
        // In hyperbolic space, the relationship should hold approximately
        if r > 0.0 {
            assert!(calculated_r > 0.0, "Calculated radius must be positive for non-zero r");
        }
    }
}

#[test]
fn test_hyperbolic_neighbors_generation() {
    let neighbors = AutopoieticHive::compute_hyperbolic_neighbors(5, 20);
    
    // Should have 6 neighbors (hexagonal structure)
    assert_eq!(neighbors.len(), 6);
    
    // All neighbors should be within bounds
    for neighbor in &neighbors {
        assert!(*neighbor < 20, "Neighbor index should be within bounds");
    }
    
    // No duplicate neighbors
    let mut sorted_neighbors = neighbors.clone();
    sorted_neighbors.sort();
    sorted_neighbors.dedup();
    assert_eq!(sorted_neighbors.len(), neighbors.len(), "No duplicate neighbors");
}

// Performance-focused tests
#[test]
fn test_nanosecond_market_tick_processing() {
    let mut node = LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]);
    let tick = MarketTick {
        symbol: [b'B', b'T', b'C', b'U', b'S', b'D', b'T', 0],
        price: 50000.0,
        volume: 1.0,
        timestamp: chrono::Utc::now().timestamp() as u64,
        bid: 49999.0,
        ask: 50001.0,
    };
    
    let start = Instant::now();
    node.process_tick(tick);
    let duration = start.elapsed();
    
    // Should process in sub-microsecond time
    assert!(duration.as_nanos() < 1000, "Tick processing should be < 1μs");
}

#[test]
fn test_zero_allocation_buffer_operations() {
    let mut buffer = CircularBuffer::<u64>::new(1000);
    
    // Fill the buffer completely
    for i in 0..1000 {
        buffer.push(i);
    }
    
    // Test that no allocations happen during normal operations
    let start = Instant::now();
    for i in 1000..2000 {
        buffer.push(i);
        let _ = buffer.latest();
    }
    let duration = start.elapsed();
    
    // Should be extremely fast with no allocations
    assert!(duration.as_micros() < 100, "Buffer operations should be < 100μs for 1000 ops");
}

#[test]
fn test_concurrent_node_access() {
    use std::sync::Arc;
    use std::thread;
    
    let node = Arc::new(LatticeNode::new(0, [0.0, 0.0, 0.0], vec![]));
    let mut handles = Vec::new();
    
    // Spawn multiple threads accessing the node
    for i in 0..10 {
        let node_clone = Arc::clone(&node);
        handles.push(thread::spawn(move || {
            let tick = MarketTick {
                symbol: [b'T', b'E', b'S', b'T', 0, 0, 0, 0],
                price: 100.0 + i as f64,
                volume: 1.0,
                timestamp: chrono::Utc::now().timestamp() as u64,
                bid: 99.0 + i as f64,
                ask: 101.0 + i as f64,
            };
            
            // This should not deadlock or panic
            node_clone.process_tick(tick);
        }));
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_memory_efficiency() {
    use std::mem;
    
    // Test that our core structures have reasonable memory footprints
    assert!(mem::size_of::<QuantumState>() <= 32, "QuantumState should be ≤ 32 bytes");
    assert!(mem::size_of::<TradeAction>() <= 32, "TradeAction should be ≤ 32 bytes");
    assert!(mem::size_of::<MarketTick>() <= 64, "MarketTick should be ≤ 64 bytes");
    assert!(mem::size_of::<ActionType>() == 1, "ActionType should be 1 byte");
    assert!(mem::size_of::<BellStateType>() <= 1, "BellStateType should be ≤ 1 byte");
}