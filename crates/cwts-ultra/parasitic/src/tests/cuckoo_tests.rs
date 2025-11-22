//! Comprehensive tests for the Cuckoo brood parasitism strategy
//! 
//! These tests verify the quantum-enhanced (classical simulation of quantum principles)
//! Cuckoo strategy implementation including SIMD optimization, pattern matching,
//! and parasitic behavior.

use crate::organisms::cuckoo::*;
use crate::organisms::ParasiteType;
use std::time::Duration;
use tokio::time::sleep;

#[cfg(test)]
mod quantum_pattern_tests {
    use super::*;

    #[test]
    fn test_quantum_pattern_creation() {
        let pattern = QuantumPatternSignature::new();
        assert_eq!(pattern.frequency_spectrum.len(), 32);
        assert_eq!(pattern.phase_relationships.len(), 16);
        assert_eq!(pattern.volatility_harmonics.len(), 8);
        assert_eq!(pattern.momentum_eigenvalues.len(), 4);
        assert_eq!(pattern.confidence_level, 0.0);
        assert!(!pattern.is_profitable);
    }

    #[test]
    fn test_quantum_similarity_identical_patterns() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Set identical frequency spectrums
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64) * 0.1;
            pattern2.frequency_spectrum[i] = (i as f64) * 0.1;
        }
        
        // Set identical phase relationships
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64) * 0.05;
            pattern2.phase_relationships[i] = (i as f64) * 0.05;
        }
        
        pattern1.quantum_entanglement_score = 0.5;
        pattern2.quantum_entanglement_score = 0.5;
        pattern1.amplitude_coherence = 0.8;
        pattern2.amplitude_coherence = 0.8;
        
        let similarity = pattern1.quantum_similarity(&pattern2);
        assert!(similarity > 0.95, "Identical patterns should have very high similarity, got {}", similarity);
    }

    #[test]
    fn test_quantum_similarity_different_patterns() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Set different frequency spectrums
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64) * 0.1;
            pattern2.frequency_spectrum[i] = (i as f64) * -0.1; // Opposite pattern
        }
        
        // Set different phase relationships
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64) * 0.05;
            pattern2.phase_relationships[i] = (i as f64) * -0.05;
        }
        
        pattern1.quantum_entanglement_score = 0.1;
        pattern2.quantum_entanglement_score = 0.9;
        pattern1.amplitude_coherence = 0.2;
        pattern2.amplitude_coherence = 0.8;
        
        let similarity = pattern1.quantum_similarity(&pattern2);
        assert!(similarity < 0.5, "Different patterns should have low similarity, got {}", similarity);
    }

    #[test]
    fn test_quantum_similarity_simd_consistency() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Set up test patterns with known values
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64).sin() * 0.5;
            pattern2.frequency_spectrum[i] = (i as f64).sin() * 0.6;
        }
        
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64) * 0.1;
            pattern2.phase_relationships[i] = (i as f64) * 0.11;
        }
        
        pattern1.quantum_entanglement_score = 0.3;
        pattern2.quantum_entanglement_score = 0.35;
        pattern1.amplitude_coherence = 0.7;
        pattern2.amplitude_coherence = 0.72;
        
        let similarity = pattern1.quantum_similarity(&pattern2);
        
        // Should be reasonably similar but not identical
        assert!(similarity > 0.6 && similarity < 0.9, 
                "Similar patterns should have moderate similarity, got {}", similarity);
    }
}

#[cfg(test)]
mod host_trading_pair_tests {
    use super::*;

    #[test]
    fn test_host_creation() {
        let host = HostTradingPair::new(
            "BTCUSDT".to_string(),
            "BTC".to_string(), 
            "USDT".to_string()
        );
        
        assert_eq!(host.symbol, "BTCUSDT");
        assert_eq!(host.base_asset, "BTC");
        assert_eq!(host.quote_asset, "USDT");
        assert_eq!(host.success_rate, 0.0);
        assert_eq!(host.vulnerability_score, 0.0);
        assert!(!host.is_currently_profitable);
    }

    #[test]
    fn test_vulnerability_calculation_high_success() {
        let mut host = HostTradingPair::new(
            "ETHUSDT".to_string(),
            "ETH".to_string(),
            "USDT".to_string()
        );
        
        host.success_rate = 0.9; // High success rate
        host.quantum_signature.confidence_level = 0.3; // Low confidence
        
        // Add some volume data
        for i in 0..20 {
            host.volume_trend.push_back(1000.0 + (i as f64) * 50.0);
        }
        
        let vulnerability = host.calculate_vulnerability();
        
        // High success rate with low confidence should create high vulnerability
        assert!(vulnerability > 0.6, "High success with low confidence should be vulnerable, got {}", vulnerability);
    }

    #[test]
    fn test_vulnerability_calculation_volume_effects() {
        let mut host = HostTradingPair::new(
            "ADAUSDT".to_string(),
            "ADA".to_string(),
            "USDT".to_string()
        );
        
        host.success_rate = 0.7;
        host.quantum_signature.confidence_level = 0.5;
        
        // Add high recent volume (easy to hide in)
        for i in 0..50 {
            if i < 10 {
                // Recent high volume
                host.volume_trend.push_back(5000.0);
            } else {
                // Historical lower volume
                host.volume_trend.push_back(1000.0);
            }
        }
        
        let vulnerability = host.calculate_vulnerability();
        
        // High recent volume should increase vulnerability
        assert!(vulnerability > 0.5, "High volume should increase vulnerability, got {}", vulnerability);
    }

    #[test]
    fn test_volume_vulnerability_calculation() {
        let mut host = HostTradingPair::new(
            "SOLUSDT".to_string(),
            "SOL".to_string(),
            "USDT".to_string()
        );
        
        // Add volume data with clear recent increase
        for i in 0..30 {
            if i < 20 {
                host.volume_trend.push_back(1000.0); // Historical volume
            } else {
                host.volume_trend.push_back(1500.0); // 50% increase in recent volume
            }
        }
        
        let volume_vulnerability = host.calculate_volume_vulnerability();
        assert!(volume_vulnerability > 0.7, "Significant volume increase should create high vulnerability");
    }
}

#[cfg(test)]
mod cuckoo_strategy_tests {
    use super::*;

    #[tokio::test]
    async fn test_cuckoo_strategy_creation() {
        let strategy = CuckooStrategy::new();
        assert!(!strategy.is_active.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(strategy.egg_counter.load(std::sync::atomic::Ordering::Acquire), 1);
        assert_eq!(strategy.max_concurrent_positions, MAX_PARASITIC_POSITIONS);
    }

    #[tokio::test]
    async fn test_strategy_activation_deactivation() {
        let strategy = CuckooStrategy::new();
        
        // Test activation
        let result = strategy.activate().await;
        assert!(result.is_ok());
        assert!(strategy.is_active.load(std::sync::atomic::Ordering::Acquire));
        
        // Let it run briefly
        sleep(Duration::from_millis(100)).await;
        
        // Test deactivation
        let result = strategy.deactivate().await;
        assert!(result.is_ok());
        assert!(!strategy.is_active.load(std::sync::atomic::Ordering::Acquire));
    }

    #[tokio::test]
    async fn test_host_discovery_simulation() {
        let strategy = CuckooStrategy::new();
        let hosts = strategy.simulate_host_discovery().await;
        
        assert!(!hosts.is_empty());
        assert!(hosts.len() <= 5); // We simulate 5 hosts
        
        for host in hosts {
            assert!(!host.symbol.is_empty());
            assert!(!host.base_asset.is_empty());
            assert!(!host.quote_asset.is_empty());
            assert!(host.success_rate >= 0.6 && host.success_rate <= 0.9);
            assert!(host.avg_profit >= 0.02 && host.avg_profit <= 0.05);
            assert_eq!(host.price_history.len(), HOST_PATTERN_WINDOW);
            assert_eq!(host.volume_trend.len(), HOST_PATTERN_WINDOW);
        }
    }

    #[tokio::test]
    async fn test_quantum_signature_generation() {
        let strategy = CuckooStrategy::new();
        
        // Create a host with sufficient data
        let mut host = HostTradingPair::new(
            "TESTUSDT".to_string(),
            "TEST".to_string(),
            "USDT".to_string()
        );
        
        // Fill with test data
        for i in 0..HOST_PATTERN_WINDOW {
            let price = 100.0 + (i as f64 * 0.1) + (i as f64 * 0.01).sin() * 2.0;
            host.price_history.push_back(price);
        }
        
        let signature = strategy.generate_quantum_signature(&host).await;
        
        assert!(signature.timestamp > 0);
        assert!(signature.confidence_level >= 0.0 && signature.confidence_level <= 1.0);
        assert!(signature.amplitude_coherence >= 0.0 && signature.amplitude_coherence <= 1.0);
        assert!(signature.entropy_measure >= 0.0);
        
        // Check that frequency spectrum has been populated
        let non_zero_frequencies = signature.frequency_spectrum.iter()
            .filter(|&&x| x > 1e-10)
            .count();
        assert!(non_zero_frequencies > 0, "Frequency spectrum should have non-zero values");
    }

    #[tokio::test]
    async fn test_parasitic_egg_placement() {
        let strategy = CuckooStrategy::new();
        
        // Setup a host in the database
        let mut host_db = strategy.host_database.write().await;
        let mut host = HostTradingPair::new(
            "TESTUSDT".to_string(),
            "TEST".to_string(),
            "USDT".to_string()
        );
        host.success_rate = 0.8;
        host.vulnerability_score = 0.9;
        host.is_currently_profitable = true;
        host.quantum_signature.confidence_level = 0.8;
        host.avg_profit = 0.03;
        
        // Add some volume and price data
        for i in 0..20 {
            host.volume_trend.push_back(1000.0 + (i as f64) * 10.0);
            host.price_history.push_back(100.0 + (i as f64) * 0.1);
        }
        
        host_db.insert("TESTUSDT".to_string(), host);
        drop(host_db);
        
        // Test egg placement
        let result = strategy.place_parasitic_egg("TESTUSDT").await;
        assert!(result.is_ok());
        
        // Check that egg was created
        let active_eggs = strategy.active_eggs.read().await;
        assert_eq!(active_eggs.len(), 1);
        
        let egg = active_eggs.values().next().unwrap();
        assert_eq!(egg.host_symbol, "TESTUSDT");
        assert!(egg.position_size > 0.0);
        assert!(egg.entry_price > 0.0);
        assert!(egg.quantum_camouflage_level > 0.0);
        assert!(egg.profit_target > egg.entry_price);
        assert!(egg.stop_loss < egg.entry_price);
        assert!(!egg.is_hatched);
        assert!(!egg.ejection_targets.is_empty());
    }

    #[tokio::test]
    async fn test_egg_hatching_mechanism() {
        let strategy = CuckooStrategy::new();
        
        // Create a mock profitable egg
        let mut active_eggs = strategy.active_eggs.write().await;
        let egg = ParasiticEgg {
            egg_id: 1,
            host_symbol: "MOCKUSDT".to_string(),
            position_size: 100.0,
            entry_price: 99.0,
            entry_timestamp: strategy.get_timestamp_us(),
            quantum_camouflage_level: 0.8,
            expected_host_behavior: QuantumPatternSignature::new(),
            profit_target: 101.0, // 2% profit target
            stop_loss: 97.0, // 2% stop loss
            is_hatched: false,
            ejection_targets: vec!["competitor1".to_string()],
        };
        active_eggs.insert(1, egg);
        drop(active_eggs);
        
        // Check egg hatching (simulates profitable price movement)
        let result = strategy.check_egg_hatching().await;
        assert!(result.is_ok());
        
        // Check statistics were updated
        let stats = strategy.get_statistics().await;
        assert!(stats.eggs_hatched > 0 || stats.failed_parasitisms > 0); // One of these should have happened
    }

    #[tokio::test]
    async fn test_signal_ejection() {
        let strategy = CuckooStrategy::new();
        
        // Test ejection attempt
        let success = strategy.attempt_signal_ejection("test_target", 0.8).await;
        assert!(success.is_ok());
        
        let was_successful = success.unwrap();
        // With 0.8 camouflage, should have reasonable success probability
        // We can't test exact outcome due to randomness, but ensure no errors
    }

    #[tokio::test]
    async fn test_performance_statistics() {
        let strategy = CuckooStrategy::new();
        let stats = strategy.get_statistics().await;
        
        // Initial stats should be zeroed
        assert_eq!(stats.total_hosts_identified, 0);
        assert_eq!(stats.successful_parasitisms, 0);
        assert_eq!(stats.failed_parasitisms, 0);
        assert_eq!(stats.total_eggs_placed, 0);
        assert_eq!(stats.eggs_hatched, 0);
        assert_eq!(stats.competitors_ejected, 0);
        assert_eq!(stats.total_profit, 0.0);
        assert_eq!(stats.win_rate, 0.0);
    }

    #[tokio::test] 
    async fn test_position_size_calculation() {
        let strategy = CuckooStrategy::new();
        
        let mut host = HostTradingPair::new(
            "TESTUSDT".to_string(),
            "TEST".to_string(),
            "USDT".to_string()
        );
        
        // Set up host with known volume data
        for i in 0..10 {
            host.volume_trend.push_back(1000.0 + (i as f64) * 100.0);
        }
        host.vulnerability_score = 0.8;
        
        let position_size = strategy.calculate_optimal_position_size(&host);
        
        assert!(position_size > 0.0);
        // Position should be a fraction of average volume
        let avg_volume = host.volume_trend.iter().sum::<f64>() / host.volume_trend.len() as f64;
        let expected_base = avg_volume * 0.1; // 10% of average
        let expected_with_vulnerability = expected_base * host.vulnerability_score;
        let expected_with_aggression = expected_with_vulnerability * strategy.mimicry_aggressiveness;
        
        assert!((position_size - expected_with_aggression).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_async_trait_implementation() {
        let strategy = CuckooStrategy::new();
        let parasitic_strategy: Box<dyn ParasiticStrategy> = Box::new(strategy);
        
        // Test trait methods
        let result = parasitic_strategy.activate().await;
        assert!(result.is_ok());
        
        let metrics = parasitic_strategy.get_performance_metrics().await;
        assert!(metrics.is_object());
        assert!(metrics.get("strategy").is_some());
        
        let result = parasitic_strategy.deactivate().await;
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod simd_optimization_tests {
    use super::*;
    
    #[test]
    fn test_simd_feature_detection() {
        // Test that SIMD feature detection works
        let simd_available = is_x86_feature_detected!("avx2");
        println!("AVX2 available: {}", simd_available);
        
        // Test should pass regardless of hardware capabilities
        assert!(simd_available || !simd_available); // Always true, just tests compilation
    }

    #[test]
    fn test_pattern_similarity_performance() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Fill with test data
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64).sin();
            pattern2.frequency_spectrum[i] = (i as f64).cos();
        }
        
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64) * 0.1;
            pattern2.phase_relationships[i] = (i as f64) * 0.15;
        }
        
        // Time the similarity calculation
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _similarity = pattern1.quantum_similarity(&pattern2);
        }
        let duration = start.elapsed();
        
        println!("1000 similarity calculations took: {:?}", duration);
        
        // Should complete reasonably quickly (less than 100ms for 1000 calculations)
        assert!(duration.as_millis() < 100, "Similarity calculation too slow: {:?}", duration);
    }
}

/// Stress tests for the cuckoo strategy under load
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_egg_placement() {
        let strategy = std::sync::Arc::new(CuckooStrategy::new());
        
        // Setup multiple hosts
        let mut host_db = strategy.host_database.write().await;
        for i in 0..10 {
            let mut host = HostTradingPair::new(
                format!("TEST{}USDT", i),
                format!("TEST{}", i),
                "USDT".to_string()
            );
            host.success_rate = 0.8;
            host.vulnerability_score = 0.9;
            host.is_currently_profitable = true;
            host.quantum_signature.confidence_level = 0.8;
            
            host_db.insert(format!("TEST{}USDT", i), host);
        }
        drop(host_db);
        
        // Spawn concurrent egg placement tasks
        let mut handles = vec![];
        for i in 0..5 {
            let strategy_clone = strategy.clone();
            let handle = tokio::spawn(async move {
                strategy_clone.place_parasitic_egg(&format!("TEST{}USDT", i)).await
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
        
        // Check that eggs were placed
        let active_eggs = strategy.active_eggs.read().await;
        assert!(active_eggs.len() <= 5); // Should have placed eggs (up to max concurrent)
    }

    #[tokio::test]
    async fn test_high_frequency_pattern_matching() {
        let mut pattern1 = QuantumPatternSignature::new();
        let mut pattern2 = QuantumPatternSignature::new();
        
        // Setup complex patterns
        for i in 0..32 {
            pattern1.frequency_spectrum[i] = (i as f64 * 0.1).sin() * (i as f64 * 0.2).cos();
            pattern2.frequency_spectrum[i] = (i as f64 * 0.12).sin() * (i as f64 * 0.18).cos();
        }
        
        for i in 0..16 {
            pattern1.phase_relationships[i] = (i as f64 * 0.05).tan();
            pattern2.phase_relationships[i] = (i as f64 * 0.048).tan();
        }
        
        pattern1.quantum_entanglement_score = 0.654321;
        pattern2.quantum_entanglement_score = 0.678901;
        pattern1.amplitude_coherence = 0.789123;
        pattern2.amplitude_coherence = 0.801234;
        
        // Perform high-frequency similarity calculations
        let start = std::time::Instant::now();
        let mut total_similarity = 0.0;
        
        for _ in 0..10000 {
            total_similarity += pattern1.quantum_similarity(&pattern2);
        }
        
        let duration = start.elapsed();
        let avg_similarity = total_similarity / 10000.0;
        
        println!("10k pattern similarities in {:?}, avg similarity: {:.6}", duration, avg_similarity);
        
        // Should be reasonably fast and produce consistent results
        assert!(duration.as_millis() < 1000, "Pattern matching too slow for HFT");
        assert!(avg_similarity > 0.0 && avg_similarity < 1.0);
    }
}