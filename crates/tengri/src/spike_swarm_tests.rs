//! # Spike Swarm Test and Validation Suite
//! 
//! Comprehensive testing framework for the spike swarm neural network,
//! including performance benchmarks, critical dynamics validation,
//! and swarm coherence verification.

#[cfg(test)]
mod tests {
    use super::super::spike_swarm::*;
    use std::time::Instant;
    use std::collections::HashMap;
    
    /// Test basic spike swarm creation and initialization
    #[test]
    fn test_swarm_creation_basic() {
        let config = SpikeSwarmConfig {
            num_neurons: 1000,
            connectivity_prob: 0.01,
            ..Default::default()
        };
        
        let result = SpikeSwarm::new(config);
        assert!(result.is_ok(), "Swarm creation should succeed");
        
        let swarm = result.unwrap();
        assert_eq!(swarm.neurons.len(), 1000);
        assert!(!swarm.synapses.is_empty());
        assert_eq!(swarm.populations.len(), 10); // 3 rate + 3 temporal + 4 phase
    }
    
    /// Test million neuron swarm creation (memory and performance)
    #[test]
    #[ignore] // Ignore by default due to resource requirements
    fn test_million_neuron_swarm() {
        let config = SpikeSwarmConfig {
            num_neurons: SWARM_SIZE, // 1,000,000
            connectivity_prob: CONNECTIVITY_PROB, // 0.001
            parallel_processing: true,
            memory_optimization: 3,
            ..Default::default()
        };
        
        let start_time = Instant::now();
        let result = SpikeSwarm::new(config);
        let creation_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Million neuron swarm creation should succeed");
        assert!(creation_time.as_secs() < 60, "Should create within 60 seconds");
        
        let swarm = result.unwrap();
        assert_eq!(swarm.neurons.len(), SWARM_SIZE);
        
        // Memory should be under 1GB
        let status = swarm.get_status().unwrap();
        assert!(status.performance_stats.memory_usage < 1000.0, 
                "Memory usage should be under 1GB, got {:.2}MB", 
                status.performance_stats.memory_usage);
    }
    
    /// Test parallel processing speedup
    #[test]
    fn test_parallel_speedup() {
        let config_serial = SpikeSwarmConfig {
            num_neurons: 10000,
            parallel_processing: false,
            ..Default::default()
        };
        
        let config_parallel = SpikeSwarmConfig {
            num_neurons: 10000,
            parallel_processing: true,
            ..Default::default()
        };
        
        // Serial execution
        let start = Instant::now();
        let mut swarm_serial = SpikeSwarm::new(config_serial).unwrap();
        for _ in 0..100 {
            let _ = swarm_serial.step().unwrap();
        }
        let serial_time = start.elapsed();
        
        // Parallel execution
        let start = Instant::now();
        let mut swarm_parallel = SpikeSwarm::new(config_parallel).unwrap();
        for _ in 0..100 {
            let _ = swarm_parallel.step().unwrap();
        }
        let parallel_time = start.elapsed();
        
        // Parallel should be faster (at least 1.5x speedup)
        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("Parallel speedup: {:.2}x", speedup);
        assert!(speedup > 1.5, "Expected >1.5x speedup, got {:.2}x", speedup);
    }
    
    /// Test power-law avalanche distribution
    #[test]
    fn test_avalanche_power_law() {
        let config = SpikeSwarmConfig {
            num_neurons: 50000,
            connectivity_prob: 0.002,
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Inject initial activity to trigger avalanches
        for i in 0..100 {
            if let Some(neuron) = swarm.neurons.get_mut(i) {
                neuron.input_current = 20.0; // Strong input
            }
        }
        
        // Run simulation to collect avalanches
        let results = swarm.run(5000.0).unwrap(); // 5 seconds
        
        assert!(results.avalanche_count > 10, 
                "Should detect multiple avalanches, got {}", results.avalanche_count);
        
        // Validate critical dynamics
        let critical_dynamics = &results.final_status.critical_dynamics;
        
        // Power-law exponent should be close to 1.5
        let exponent_diff = (critical_dynamics.power_law_exponent - POWER_LAW_EXPONENT).abs();
        assert!(exponent_diff < 0.5, 
                "Power-law exponent should be ~{}, got {:.2}", 
                POWER_LAW_EXPONENT, critical_dynamics.power_law_exponent);
        
        // Criticality index should be reasonable
        assert!(critical_dynamics.criticality_index >= 0.0 && 
                critical_dynamics.criticality_index <= 1.0,
                "Criticality index should be in [0,1], got {:.2}",
                critical_dynamics.criticality_index);
    }
    
    /// Test synchronization detection
    #[test]
    fn test_synchronization_metrics() {
        let config = SpikeSwarmConfig {
            num_neurons: 10000,
            connectivity_prob: 0.005,
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Create synchronized input to test detection
        for i in 0..1000 {
            if let Some(neuron) = swarm.neurons.get_mut(i) {
                neuron.input_current = 15.0; // Moderate synchronous input
            }
        }
        
        // Run simulation
        for _ in 0..100 {
            let spikes = swarm.step().unwrap();
            if !spikes.is_empty() {
                break; // Found activity
            }
        }
        
        let status = swarm.get_status().unwrap();
        let sync_metrics = &status.sync_metrics;
        
        // Should detect some level of synchrony
        assert!(sync_metrics.global_synchrony > 0.0,
                "Should detect global synchrony, got {:.3}", 
                sync_metrics.global_synchrony);
        
        // Local synchrony should be measured for populations
        assert!(!sync_metrics.local_synchrony.is_empty(),
                "Should measure local synchrony for populations");
        
        for (pop_name, &sync_value) in &sync_metrics.local_synchrony {
            assert!(sync_value >= 0.0 && sync_value <= 1.0,
                    "Sync value for {} should be in [0,1], got {:.3}", 
                    pop_name, sync_value);
        }
    }
    
    /// Test spike encoding and decoding
    #[test]
    fn test_spike_encoding_all_types() {
        let config = SpikeSwarmConfig {
            num_neurons: 10000,
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        let test_signal = crate::types::TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "BTCUSDT".to_string(),
            signal_type: crate::types::SignalType::Buy,
            strength: 0.8,
            confidence: 0.9,
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("volume".to_string(), 1000000.0);
                map.insert("momentum".to_string(), 0.7);
                map
            },
            expires_at: None,
        };
        
        // Test all encoding types
        let encoding_types = [
            SpikeEncoding::Rate,
            SpikeEncoding::Temporal,
            SpikeEncoding::Phase,
        ];
        
        for &encoding in &encoding_types {
            // Test encoding
            let encode_result = swarm.encode_signal(&test_signal, encoding);
            assert!(encode_result.is_ok(), "Encoding {:?} should succeed", encoding);
            
            // Run some steps to generate activity
            for _ in 0..50 {
                let _ = swarm.step().unwrap();
            }
            
            // Test decoding
            let population_id = match encoding {
                SpikeEncoding::Rate => "rate_pop_0",
                SpikeEncoding::Temporal => "temporal_pop_0", 
                SpikeEncoding::Phase => "phase_pop_0",
            };
            
            let decode_result = swarm.decode_population(population_id, encoding);
            assert!(decode_result.is_ok(), "Decoding {:?} should succeed", encoding);
            
            // Note: Decoded signal might be None if no activity, which is valid
        }
    }
    
    /// Test population coding functionality
    #[test]
    fn test_population_coding() {
        let config = SpikeSwarmConfig {
            num_neurons: 20000,
            ..Default::default()
        };
        
        let swarm = SpikeSwarm::new(config).unwrap();
        let status = swarm.get_status().unwrap();
        
        // Verify all populations are created
        assert_eq!(status.population_activities.len(), 10);
        
        // Check population naming
        let expected_populations = [
            "rate_pop_0", "rate_pop_1", "rate_pop_2",
            "temporal_pop_0", "temporal_pop_1", "temporal_pop_2", 
            "phase_pop_0", "phase_pop_1", "phase_pop_2", "phase_pop_3"
        ];
        
        for &pop_name in &expected_populations {
            assert!(status.population_activities.contains_key(pop_name),
                    "Population {} should exist", pop_name);
        }
        
        // All activities should initially be 0 (no spikes yet)
        for (&activity) in status.population_activities.values() {
            assert!(activity >= 0.0 && activity <= 1.0,
                    "Population activity should be in [0,1], got {:.3}", activity);
        }
    }
    
    /// Test memory efficiency under sustained load
    #[test]
    fn test_memory_efficiency() {
        let config = SpikeSwarmConfig {
            num_neurons: 50000,
            connectivity_prob: 0.002,
            memory_optimization: 3,
            recording: RecordingConfig {
                record_spikes: true,
                record_potentials: false,
                record_avalanches: true,
                max_duration: 10.0, // Short duration
                sampling_rate: 100.0, // Lower sampling rate
            },
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Get initial memory usage
        let initial_status = swarm.get_status().unwrap();
        let initial_memory = initial_status.performance_stats.memory_usage;
        
        // Run sustained simulation
        for _ in 0..1000 {
            let _ = swarm.step().unwrap();
        }
        
        // Check final memory usage
        let final_status = swarm.get_status().unwrap();
        let final_memory = final_status.performance_stats.memory_usage;
        
        // Memory growth should be limited (less than 50% increase)
        let memory_growth = (final_memory - initial_memory) / initial_memory;
        assert!(memory_growth < 0.5,
                "Memory growth should be <50%, got {:.1}%", memory_growth * 100.0);
        
        println!("Memory: {:.1}MB -> {:.1}MB ({:.1}% growth)",
                 initial_memory, final_memory, memory_growth * 100.0);
    }
    
    /// Test critical dynamics validation (branching parameter)
    #[test]
    fn test_critical_dynamics_validation() {
        let config = SpikeSwarmConfig {
            num_neurons: 30000,
            connectivity_prob: 0.001, // Critical connectivity
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Inject seed activity
        for i in (0..100).step_by(10) {
            if let Some(neuron) = swarm.neurons.get_mut(i) {
                neuron.input_current = 10.0;
            }
        }
        
        // Run to reach critical state
        for _ in 0..1000 {
            let _ = swarm.step().unwrap();
        }
        
        let status = swarm.get_status().unwrap();
        let dynamics = &status.critical_dynamics;
        
        // Branching parameter should be near 1.0 for criticality
        assert!(dynamics.branching_parameter > 0.5 && dynamics.branching_parameter < 2.0,
                "Branching parameter should be ~1.0, got {:.2}", 
                dynamics.branching_parameter);
        
        // Should have collected avalanche data
        assert!(!dynamics.avalanche_sizes.is_empty(),
                "Should have avalanche size data");
        
        // Activity variance should be positive
        assert!(dynamics.activity_variance >= 0.0,
                "Activity variance should be non-negative, got {:.3}",
                dynamics.activity_variance);
    }
    
    /// Test error handling and recovery
    #[test]
    fn test_error_handling() {
        // Test with invalid configuration
        let invalid_config = SpikeSwarmConfig {
            num_neurons: 0, // Invalid
            ..Default::default()
        };
        
        let result = SpikeSwarm::new(invalid_config);
        assert!(result.is_err(), "Should fail with invalid configuration");
        
        // Test with minimal configuration
        let minimal_config = SpikeSwarmConfig {
            num_neurons: 10,
            connectivity_prob: 0.0, // No connections
            ..Default::default()
        };
        
        let result = SpikeSwarm::new(minimal_config);
        assert!(result.is_ok(), "Should handle minimal configuration");
        
        let mut swarm = result.unwrap();
        
        // Should handle steps even without connections
        for _ in 0..10 {
            let result = swarm.step();
            assert!(result.is_ok(), "Steps should succeed even without connections");
        }
    }
    
    /// Performance benchmark test
    #[test]
    #[ignore] // Resource intensive
    fn test_performance_benchmark() {
        let sizes = [1000, 10000, 100000];
        
        for &size in &sizes {
            let config = SpikeSwarmConfig {
                num_neurons: size,
                parallel_processing: true,
                ..Default::default()
            };
            
            let start = Instant::now();
            let mut swarm = SpikeSwarm::new(config).unwrap();
            let creation_time = start.elapsed();
            
            // Inject activity
            for i in 0..(size / 100).max(1) {
                if let Some(neuron) = swarm.neurons.get_mut(i) {
                    neuron.input_current = 15.0;
                }
            }
            
            // Benchmark simulation steps
            let start = Instant::now();
            let mut total_spikes = 0;
            
            for _ in 0..100 {
                let spikes = swarm.step().unwrap();
                total_spikes += spikes.len();
            }
            
            let simulation_time = start.elapsed();
            let steps_per_sec = 100.0 / simulation_time.as_secs_f64();
            
            println!("Size: {} neurons", size);
            println!("  Creation: {:.2}ms", creation_time.as_secs_f64() * 1000.0);
            println!("  Simulation: {:.1} steps/sec", steps_per_sec);
            println!("  Total spikes: {}", total_spikes);
            
            // Performance requirements
            if size == 1000 {
                assert!(steps_per_sec > 1000.0, "1K neurons should run >1000 steps/sec");
            } else if size == 10000 {
                assert!(steps_per_sec > 100.0, "10K neurons should run >100 steps/sec");
            } else if size == 100000 {
                assert!(steps_per_sec > 10.0, "100K neurons should run >10 steps/sec");
            }
        }
    }
}

/// Integration tests with other TENGRI components
#[cfg(test)]
mod integration_tests {
    use super::super::spike_swarm::*;
    use crate::types::*;
    
    /// Test integration with TENGRI trading signals
    #[test]
    fn test_tengri_signal_integration() {
        let config = SpikeSwarmConfig {
            num_neurons: 10000,
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Create realistic trading signals
        let signals = vec![
            TradingSignal {
                id: uuid::Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                signal_type: SignalType::Buy,
                strength: 0.8,
                confidence: 0.9,
                timestamp: chrono::Utc::now(),
                source: "TENGRI".to_string(),
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("rsi".to_string(), 25.0); // Oversold
                    map.insert("macd".to_string(), 0.5); // Bullish
                    map.insert("volume".to_string(), 1.2); // Above average
                    map
                },
                expires_at: Some(chrono::Utc::now() + chrono::Duration::minutes(5)),
            },
            TradingSignal {
                id: uuid::Uuid::new_v4(),
                symbol: "ETHUSDT".to_string(),
                signal_type: SignalType::Sell,
                strength: 0.6,
                confidence: 0.7,
                timestamp: chrono::Utc::now(),
                source: "TENGRI".to_string(),
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("rsi".to_string(), 75.0); // Overbought
                    map.insert("macd".to_string(), -0.3); // Bearish
                    map.insert("volume".to_string(), 0.8); // Below average
                    map
                },
                expires_at: Some(chrono::Utc::now() + chrono::Duration::minutes(3)),
            },
        ];
        
        // Process signals through swarm
        for signal in &signals {
            // Test all encoding types
            for &encoding in &[SpikeEncoding::Rate, SpikeEncoding::Temporal, SpikeEncoding::Phase] {
                let result = swarm.encode_signal(signal, encoding);
                assert!(result.is_ok(), "Signal encoding should succeed");
            }
        }
        
        // Run simulation to process signals
        for _ in 0..200 {
            let spikes = swarm.step().unwrap();
            if !spikes.is_empty() {
                println!("Step produced {} spikes", spikes.len());
            }
        }
        
        // Verify swarm processed signals
        let status = swarm.get_status().unwrap();
        assert!(status.recent_spikes > 0, "Should have processed signals into spikes");
        
        // Test decoding
        for &encoding in &[SpikeEncoding::Rate, SpikeEncoding::Temporal, SpikeEncoding::Phase] {
            let pop_id = match encoding {
                SpikeEncoding::Rate => "rate_pop_0",
                SpikeEncoding::Temporal => "temporal_pop_0",
                SpikeEncoding::Phase => "phase_pop_0",
            };
            
            let decoded = swarm.decode_population(pop_id, encoding).unwrap();
            if let Some(decoded_signal) = decoded {
                println!("Decoded signal: {:?} strength={:.2} confidence={:.2}",
                         decoded_signal.signal_type, decoded_signal.strength, decoded_signal.confidence);
            }
        }
    }
    
    /// Test swarm coherence across populations
    #[test]
    fn test_swarm_coherence() {
        let config = SpikeSwarmConfig {
            num_neurons: 20000,
            connectivity_prob: 0.002,
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Create coherent input signal
        let coherent_signal = TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "MARKET_INDEX".to_string(),
            signal_type: SignalType::StrongBuy,
            strength: 0.95,
            confidence: 0.98,
            timestamp: chrono::Utc::now(),
            source: "MARKET_COHERENCE_TEST".to_string(),
            metadata: std::collections::HashMap::new(),
            expires_at: None,
        };
        
        // Encode signal in multiple populations
        for &encoding in &[SpikeEncoding::Rate, SpikeEncoding::Temporal, SpikeEncoding::Phase] {
            let _ = swarm.encode_signal(&coherent_signal, encoding);
        }
        
        // Run simulation to observe coherence
        for _ in 0..500 {
            let _ = swarm.step().unwrap();
        }
        
        let status = swarm.get_status().unwrap();
        
        // Check for coherent activity across populations
        let mut active_populations = 0;
        let mut total_activity = 0.0;
        
        for (&activity) in status.population_activities.values() {
            if activity > 0.01 { // 1% threshold
                active_populations += 1;
                total_activity += activity;
            }
        }
        
        assert!(active_populations >= 3, 
                "Should have activity in multiple populations, got {}", active_populations);
        
        // Global synchrony should be elevated
        assert!(status.sync_metrics.global_synchrony > 0.01,
                "Global synchrony should be elevated, got {:.4}", 
                status.sync_metrics.global_synchrony);
        
        println!("Coherence test: {} active populations, {:.1}% avg activity, {:.3} global sync",
                 active_populations, total_activity * 100.0 / active_populations as f64,
                 status.sync_metrics.global_synchrony);
    }
}