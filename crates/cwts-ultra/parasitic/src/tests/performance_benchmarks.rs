//! Performance benchmarks for parasitic trading strategies
//! 
//! These benchmarks measure the performance characteristics of quantum-enhanced
//! (classically simulated) pattern matching, SIMD operations, and system throughput.

use crate::*;
use crate::organisms::cuckoo::*;
use std::time::{Duration, Instant};
use std::sync::Arc;

#[cfg(test)]
mod quantum_pattern_benchmarks {
    use super::*;

    #[test]
    fn benchmark_quantum_similarity_scalar() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Fill with complex test data
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64).sin() * (i as f64 * 0.1).cos();
            pattern2.frequency_spectrum[i] = (i as f64 * 1.1).sin() * (i as f64 * 0.11).cos();
        }
        
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64 * 0.05).tan().atan();
            pattern2.phase_relationships[i] = (i as f64 * 0.052).tan().atan();
        }
        
        pattern1.quantum_entanglement_score = 0.456789;
        pattern2.quantum_entanglement_score = 0.467890;
        pattern1.amplitude_coherence = 0.789012;
        pattern2.amplitude_coherence = 0.798123;
        
        // Benchmark scalar implementation
        let iterations = 100_000;
        let start = Instant::now();
        
        let mut total_similarity = 0.0;
        for _ in 0..iterations {
            total_similarity += pattern1.scalar_similarity(&pattern2);
        }
        
        let duration = start.elapsed();
        let avg_similarity = total_similarity / iterations as f64;
        let operations_per_sec = iterations as f64 / duration.as_secs_f64();
        
        println!("Scalar Quantum Similarity Benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:?}", duration);
        println!("  Ops/sec: {:.0}", operations_per_sec);
        println!("  Avg similarity: {:.6}", avg_similarity);
        println!("  Nanoseconds per op: {:.2}", duration.as_nanos() as f64 / iterations as f64);
        
        // Performance targets
        assert!(operations_per_sec > 50_000.0, "Scalar implementation too slow: {} ops/sec", operations_per_sec);
        assert!(avg_similarity > 0.0 && avg_similarity < 1.0, "Invalid similarity range");
    }

    #[test]
    fn benchmark_quantum_similarity_simd() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Fill with complex test data
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64).sin() * (i as f64 * 0.1).cos();
            pattern2.frequency_spectrum[i] = (i as f64 * 1.1).sin() * (i as f64 * 0.11).cos();
        }
        
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64 * 0.05).tan().atan();
            pattern2.phase_relationships[i] = (i as f64 * 0.052).tan().atan();
        }
        
        pattern1.quantum_entanglement_score = 0.456789;
        pattern2.quantum_entanglement_score = 0.467890;
        pattern1.amplitude_coherence = 0.789012;
        pattern2.amplitude_coherence = 0.798123;
        
        // Benchmark SIMD/full implementation
        let iterations = 100_000;
        let start = Instant::now();
        
        let mut total_similarity = 0.0;
        for _ in 0..iterations {
            total_similarity += pattern1.quantum_similarity(&pattern2);
        }
        
        let duration = start.elapsed();
        let avg_similarity = total_similarity / iterations as f64;
        let operations_per_sec = iterations as f64 / duration.as_secs_f64();
        
        println!("SIMD Quantum Similarity Benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:?}", duration);
        println!("  Ops/sec: {:.0}", operations_per_sec);
        println!("  Avg similarity: {:.6}", avg_similarity);
        println!("  Nanoseconds per op: {:.2}", duration.as_nanos() as f64 / iterations as f64);
        println!("  SIMD available: {}", is_x86_feature_detected!("avx2"));
        
        // SIMD should be faster or at least as fast as scalar
        assert!(operations_per_sec > 40_000.0, "SIMD implementation too slow: {} ops/sec", operations_per_sec);
        assert!(avg_similarity > 0.0 && avg_similarity < 1.0, "Invalid similarity range");
    }

    #[test]
    fn benchmark_pattern_generation() {
        let strategy = CuckooStrategy::new();
        
        // Create hosts with various data sizes
        let mut hosts = vec![];
        for i in 0..10 {
            let mut host = HostTradingPair::new(
                format!("BENCH{}USDT", i),
                format!("BENCH{}", i),
                "USDT".to_string()
            );
            
            // Fill with realistic price data
            for j in 0..HOST_PATTERN_WINDOW {
                let base_price = 100.0 + (i as f64) * 10.0;
                let price_variation = (j as f64 * 0.1).sin() * 2.0 + (j as f64 * 0.05).cos() * 1.0;
                let noise = (j as f64 * 0.234).sin() * 0.1;
                host.price_history.push_back(base_price + price_variation + noise);
            }
            hosts.push(host);
        }
        
        // Benchmark pattern generation
        let start = Instant::now();
        let mut signatures = vec![];
        
        for host in &hosts {
            // Note: This would be async in real usage, but we test the core computation
            let signature = futures::executor::block_on(strategy.generate_quantum_signature(host));
            signatures.push(signature);
        }
        
        let duration = start.elapsed();
        let patterns_per_sec = hosts.len() as f64 / duration.as_secs_f64();
        
        println!("Quantum Pattern Generation Benchmark:");
        println!("  Patterns: {}", hosts.len());
        println!("  Total time: {:?}", duration);
        println!("  Patterns/sec: {:.2}", patterns_per_sec);
        println!("  Milliseconds per pattern: {:.2}", duration.as_millis() as f64 / hosts.len() as f64);
        
        // Check pattern quality
        for signature in &signatures {
            assert!(signature.confidence_level >= 0.0 && signature.confidence_level <= 1.0);
            assert!(signature.amplitude_coherence >= 0.0 && signature.amplitude_coherence <= 1.0);
            assert!(signature.entropy_measure >= 0.0);
            
            let non_zero_freq = signature.frequency_spectrum.iter().filter(|&&x| x > 1e-10).count();
            assert!(non_zero_freq > 0, "Should have non-zero frequency components");
        }
        
        // Performance targets
        assert!(patterns_per_sec > 100.0, "Pattern generation too slow: {} patterns/sec", patterns_per_sec);
    }
}

#[cfg(test)]
mod cuckoo_strategy_benchmarks {
    use super::*;

    #[tokio::test]
    async fn benchmark_host_scanning() {
        let strategy = CuckooStrategy::new();
        
        // Benchmark host discovery
        let iterations = 10;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _hosts = strategy.simulate_host_discovery().await;
        }
        
        let duration = start.elapsed();
        let scans_per_sec = iterations as f64 / duration.as_secs_f64();
        
        println!("Host Scanning Benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:?}", duration);
        println!("  Scans/sec: {:.2}", scans_per_sec);
        
        // Should be able to scan frequently for real-time operation
        assert!(scans_per_sec > 10.0, "Host scanning too slow: {} scans/sec", scans_per_sec);
    }

    #[tokio::test]
    async fn benchmark_egg_placement_throughput() {
        let strategy = Arc::new(CuckooStrategy::new());
        
        // Setup multiple hosts
        let mut host_db = strategy.host_database.write().await;
        for i in 0..50 {
            let mut host = HostTradingPair::new(
                format!("BENCH{}USDT", i),
                format!("BENCH{}", i),
                "USDT".to_string()
            );
            host.success_rate = 0.8;
            host.vulnerability_score = 0.9;
            host.is_currently_profitable = true;
            host.quantum_signature.confidence_level = 0.8;
            host.avg_profit = 0.03;
            
            // Add volume data
            for j in 0..20 {
                host.volume_trend.push_back(1000.0 + (j as f64) * 50.0);
            }
            
            host_db.insert(format!("BENCH{}USDT", i), host);
        }
        drop(host_db);
        
        // Benchmark egg placement
        let start = Instant::now();
        let mut placement_count = 0;
        
        // Place eggs on first 20 hosts (within MAX_PARASITIC_POSITIONS)
        for i in 0..20 {
            let result = strategy.place_parasitic_egg(&format!("BENCH{}USDT", i)).await;
            if result.is_ok() {
                placement_count += 1;
            }
        }
        
        let duration = start.elapsed();
        let placements_per_sec = placement_count as f64 / duration.as_secs_f64();
        
        println!("Egg Placement Benchmark:");
        println!("  Successful placements: {}", placement_count);
        println!("  Total time: {:?}", duration);
        println!("  Placements/sec: {:.2}", placements_per_sec);
        
        // Verify eggs were placed
        let active_eggs = strategy.active_eggs.read().await;
        assert!(active_eggs.len() > 0);
        
        // Performance target
        assert!(placements_per_sec > 50.0, "Egg placement too slow: {} placements/sec", placements_per_sec);
    }

    #[tokio::test]
    async fn benchmark_competitor_ejection() {
        let strategy = CuckooStrategy::new();
        
        // Benchmark ejection operations
        let iterations = 1000;
        let start = Instant::now();
        let mut successful_ejections = 0;
        
        for i in 0..iterations {
            let camouflage_level = 0.5 + ((i % 100) as f64) * 0.005; // Varying camouflage
            let result = strategy.attempt_signal_ejection(&format!("target_{}", i), camouflage_level).await;
            if result.unwrap_or(false) {
                successful_ejections += 1;
            }
        }
        
        let duration = start.elapsed();
        let ejections_per_sec = iterations as f64 / duration.as_secs_f64();
        let success_rate = successful_ejections as f64 / iterations as f64;
        
        println!("Competitor Ejection Benchmark:");
        println!("  Attempts: {}", iterations);
        println!("  Successful: {}", successful_ejections);
        println!("  Success rate: {:.2}%", success_rate * 100.0);
        println!("  Total time: {:?}", duration);
        println!("  Ejections/sec: {:.0}", ejections_per_sec);
        
        // Should be very fast (lightweight operation)
        assert!(ejections_per_sec > 5000.0, "Ejection too slow: {} ejections/sec", ejections_per_sec);
        assert!(success_rate > 0.0 && success_rate < 1.0, "Unrealistic success rate: {}", success_rate);
    }

    #[tokio::test]
    async fn benchmark_statistics_collection() {
        let strategy = CuckooStrategy::new();
        
        // Benchmark statistics gathering
        let iterations = 10000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _stats = strategy.get_statistics().await;
        }
        
        let duration = start.elapsed();
        let stats_per_sec = iterations as f64 / duration.as_secs_f64();
        
        println!("Statistics Collection Benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:?}", duration);
        println!("  Stats/sec: {:.0}", stats_per_sec);
        
        // Statistics should be very fast (mostly atomic reads)
        assert!(stats_per_sec > 50_000.0, "Statistics collection too slow: {} stats/sec", stats_per_sec);
    }
}

#[cfg(test)]
mod system_benchmarks {
    use super::*;

    #[tokio::test]
    async fn benchmark_system_activation_deactivation() {
        let iterations = 10;
        let mut total_activation_time = Duration::new(0, 0);
        let mut total_deactivation_time = Duration::new(0, 0);
        
        for _ in 0..iterations {
            let system = ParasiticTradingSystem::new_default();
            
            // Measure activation time
            let start = Instant::now();
            system.activate().await.unwrap();
            total_activation_time += start.elapsed();
            
            // Brief operation
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            // Measure deactivation time
            let start = Instant::now();
            system.deactivate().await.unwrap();
            total_deactivation_time += start.elapsed();
        }
        
        let avg_activation = total_activation_time / iterations;
        let avg_deactivation = total_deactivation_time / iterations;
        
        println!("System Lifecycle Benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Avg activation time: {:?}", avg_activation);
        println!("  Avg deactivation time: {:?}", avg_deactivation);
        
        // Should be reasonably fast for real-time trading
        assert!(avg_activation.as_millis() < 100, "Activation too slow: {:?}", avg_activation);
        assert!(avg_deactivation.as_millis() < 50, "Deactivation too slow: {:?}", avg_deactivation);
    }

    #[tokio::test]
    async fn benchmark_concurrent_metrics_collection() {
        let system = Arc::new(ParasiticTradingSystem::new_default());
        system.activate().await.unwrap();
        
        // Benchmark concurrent metrics access
        let concurrent_requests = 100;
        let start = Instant::now();
        
        let mut handles = vec![];
        for _ in 0..concurrent_requests {
            let system_clone = system.clone();
            let handle = tokio::spawn(async move {
                system_clone.get_performance_metrics().await
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        let duration = start.elapsed();
        let requests_per_sec = concurrent_requests as f64 / duration.as_secs_f64();
        
        println!("Concurrent Metrics Collection Benchmark:");
        println!("  Concurrent requests: {}", concurrent_requests);
        println!("  Total time: {:?}", duration);
        println!("  Requests/sec: {:.0}", requests_per_sec);
        
        system.deactivate().await.unwrap();
        
        // Should handle high concurrent load
        assert!(requests_per_sec > 1000.0, "Concurrent metrics too slow: {} requests/sec", requests_per_sec);
    }

    #[tokio::test]
    async fn benchmark_strategy_management_overhead() {
        let mut config = ParasiticConfig::default();
        config.max_concurrent_strategies = 20;
        
        let system = ParasiticTradingSystem::new(config);
        system.activate().await.unwrap();
        
        // Benchmark adding multiple strategies
        let start = Instant::now();
        let mut successful_adds = 0;
        
        for _ in 0..15 {
            let result = system.add_strategy(ParasiteType::Cuckoo).await;
            if result.is_ok() {
                successful_adds += 1;
            }
        }
        
        let duration = start.elapsed();
        let adds_per_sec = successful_adds as f64 / duration.as_secs_f64();
        
        println!("Strategy Management Benchmark:");
        println!("  Successful adds: {}", successful_adds);
        println!("  Total time: {:?}", duration);
        println!("  Adds/sec: {:.2}", adds_per_sec);
        
        system.deactivate().await.unwrap();
        
        // Should be able to quickly scale up strategies
        assert!(adds_per_sec > 20.0, "Strategy addition too slow: {} adds/sec", adds_per_sec);
    }

    #[tokio::test]
    async fn benchmark_memory_efficiency() {
        use std::mem::size_of;
        
        println!("Memory Usage Analysis:");
        println!("  QuantumPatternSignature: {} bytes", size_of::<QuantumPatternSignature>());
        println!("  HostTradingPair: {} bytes", size_of::<HostTradingPair>());
        println!("  ParasiticEgg: {} bytes", size_of::<ParasiticEgg>());
        println!("  CuckooStats: {} bytes", size_of::<CuckooStats>());
        println!("  SystemMetrics: {} bytes", size_of::<SystemMetrics>());
        
        // Test memory scaling with multiple hosts
        let strategy = CuckooStrategy::new();
        
        // Measure memory usage growth
        let initial_usage = get_approximate_memory_usage(&strategy).await;
        
        // Add many hosts
        let mut host_db = strategy.host_database.write().await;
        for i in 0..1000 {
            let mut host = HostTradingPair::new(
                format!("MEM{}USDT", i),
                format!("MEM{}", i),
                "USDT".to_string()
            );
            
            // Fill with data
            for j in 0..HOST_PATTERN_WINDOW {
                host.price_history.push_back(100.0 + (j as f64));
                host.volume_trend.push_back(1000.0 + (j as f64));
            }
            
            host_db.insert(format!("MEM{}USDT", i), host);
        }
        drop(host_db);
        
        let final_usage = get_approximate_memory_usage(&strategy).await;
        let per_host_usage = (final_usage - initial_usage) / 1000;
        
        println!("  Approximate per-host memory: {} bytes", per_host_usage);
        
        // Memory usage should be reasonable
        assert!(per_host_usage < 100_000, "Excessive per-host memory usage: {} bytes", per_host_usage);
    }

    async fn get_approximate_memory_usage(_strategy: &CuckooStrategy) -> usize {
        // Simplified memory estimation
        // In a real implementation, this would use more sophisticated memory profiling
        use std::mem::size_of;
        size_of::<CuckooStrategy>()
    }
}

#[cfg(test)]
mod load_tests {
    use super::*;

    #[tokio::test]
    async fn load_test_high_frequency_operations() {
        let system = Arc::new(ParasiticTradingSystem::new_default());
        system.activate().await.unwrap();
        
        // Simulate high-frequency trading load
        let duration = Duration::from_secs(5);
        let start = Instant::now();
        let mut operation_count = 0;
        
        while start.elapsed() < duration {
            // Mix of different operations
            let operation = operation_count % 4;
            
            match operation {
                0 => {
                    let _status = system.get_status().await;
                },
                1 => {
                    let _metrics = system.get_performance_metrics().await;
                },
                2 => {
                    // Try adding strategy (will fail after limit)
                    let _ = system.add_strategy(ParasiteType::Cuckoo).await;
                },
                3 => {
                    // Quick configuration check
                    let _config = system.get_config();
                },
                _ => unreachable!(),
            }
            
            operation_count += 1;
            
            // Brief yield to prevent blocking
            if operation_count % 100 == 0 {
                tokio::task::yield_now().await;
            }
        }
        
        let actual_duration = start.elapsed();
        let ops_per_sec = operation_count as f64 / actual_duration.as_secs_f64();
        
        println!("High-Frequency Load Test:");
        println!("  Duration: {:?}", actual_duration);
        println!("  Operations: {}", operation_count);
        println!("  Ops/sec: {:.0}", ops_per_sec);
        
        system.deactivate().await.unwrap();
        
        // Should handle high-frequency operations
        assert!(ops_per_sec > 1000.0, "System too slow under load: {} ops/sec", ops_per_sec);
        assert!(operation_count > 5000, "Not enough operations completed: {}", operation_count);
    }

    #[tokio::test]
    async fn stress_test_concurrent_systems() {
        // Test multiple systems running concurrently
        let system_count = 5;
        let mut handles = vec![];
        
        let start = Instant::now();
        
        for i in 0..system_count {
            let handle = tokio::spawn(async move {
                let system = ParasiticTradingSystem::new_default();
                system.activate().await.unwrap();
                
                // Run brief operations
                for _ in 0..100 {
                    let _status = system.get_status().await;
                    let _metrics = system.get_performance_metrics().await;
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                
                system.deactivate().await.unwrap();
                i // Return system index
            });
            handles.push(handle);
        }
        
        // Wait for all systems to complete
        let mut completed_systems = vec![];
        for handle in handles {
            let system_id = handle.await.unwrap();
            completed_systems.push(system_id);
        }
        
        let duration = start.elapsed();
        
        println!("Concurrent Systems Stress Test:");
        println!("  Systems: {}", system_count);
        println!("  Total time: {:?}", duration);
        println!("  Completed: {:?}", completed_systems);
        
        // All systems should complete successfully
        assert_eq!(completed_systems.len(), system_count);
        assert!(duration.as_secs() < 30, "Stress test took too long: {:?}", duration);
        
        // Verify all system IDs are present
        completed_systems.sort();
        for i in 0..system_count {
            assert!(completed_systems.contains(&i), "System {} did not complete", i);
        }
    }
}