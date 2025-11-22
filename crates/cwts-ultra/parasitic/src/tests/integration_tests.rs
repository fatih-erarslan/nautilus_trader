//! Integration tests for the parasitic trading system
//! 
//! Tests the complete system integration including multiple strategies,
//! system lifecycle, and cross-component interactions.

use crate::*;
use super::TestFixture;
use std::time::Duration;
use tokio::time::sleep;

#[cfg(test)]
mod system_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_system_lifecycle() {
        let system = ParasiticTradingSystem::new_default();
        
        // Test initial state
        assert!(!system.is_active());
        let status = system.get_status().await;
        assert_eq!(status["status"], "inactive");
        assert_eq!(status["active_strategies"].as_u64().unwrap(), 0);
        
        // Test activation
        let result = system.activate().await;
        assert!(result.is_ok());
        assert!(system.is_active());
        
        let status = system.get_status().await;
        assert_eq!(status["status"], "active");
        assert!(status["active_strategies"].as_u64().unwrap() > 0);
        
        // Let system run briefly to generate some activity
        sleep(Duration::from_millis(200)).await;
        
        // Test metrics collection
        let metrics = system.get_performance_metrics().await;
        assert!(metrics.total_profit >= 0.0);
        assert!(metrics.overall_win_rate >= 0.0 && metrics.overall_win_rate <= 1.0);
        
        // Test deactivation
        let result = system.deactivate().await;
        assert!(result.is_ok());
        assert!(!system.is_active());
        
        let status = system.get_status().await;
        assert_eq!(status["status"], "inactive");
        assert_eq!(status["active_strategies"].as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_strategy_management() {
        let mut config = ParasiticConfig::default();
        config.max_concurrent_strategies = 3;
        
        let system = ParasiticTradingSystem::new(config);
        system.activate().await.unwrap();
        
        // Test adding strategies
        let result = system.add_strategy(ParasiteType::Cuckoo).await;
        assert!(result.is_ok());
        
        // Try to add unsupported strategy
        let result = system.add_strategy(ParasiteType::Wasp).await;
        assert!(result.is_err());
        
        // Test strategy limit
        for _ in 0..5 {
            let _ = system.add_strategy(ParasiteType::Cuckoo).await;
        }
        
        let status = system.get_status().await;
        let active_count = status["active_strategies"].as_u64().unwrap();
        assert!(active_count <= 3); // Should respect max limit
        
        system.deactivate().await.unwrap();
    }

    #[tokio::test]
    async fn test_metrics_aggregation() {
        let system = ParasiticTradingSystem::new_default();
        system.activate().await.unwrap();
        
        // Let system run to generate metrics
        sleep(Duration::from_millis(300)).await;
        
        let metrics = system.get_performance_metrics().await;
        
        // Test metrics structure
        assert!(metrics.total_profit >= 0.0);
        assert!(metrics.total_successful_parasitisms >= 0);
        assert!(metrics.total_failed_parasitisms >= 0);
        assert!(metrics.total_hosts_identified >= 0);
        assert!(metrics.total_positions_placed >= 0);
        assert!(metrics.total_competitors_eliminated >= 0);
        
        // Test derived metrics
        let total_attempts = metrics.total_successful_parasitisms + metrics.total_failed_parasitisms;
        if total_attempts > 0 {
            let expected_win_rate = metrics.total_successful_parasitisms as f64 / total_attempts as f64;
            assert!((metrics.overall_win_rate - expected_win_rate).abs() < 1e-6);
        }
        
        system.deactivate().await.unwrap();
    }

    #[tokio::test]
    async fn test_configuration_effects() {
        // Test conservative configuration
        let conservative_config = ParasiticConfig {
            risk_tolerance: 0.1,
            aggression_factor: 0.5,
            max_concurrent_strategies: 1,
            ..ParasiticConfig::default()
        };
        
        let conservative_system = ParasiticTradingSystem::new(conservative_config);
        conservative_system.activate().await.unwrap();
        sleep(Duration::from_millis(100)).await;
        let conservative_metrics = conservative_system.get_performance_metrics().await;
        conservative_system.deactivate().await.unwrap();
        
        // Test aggressive configuration  
        let aggressive_config = ParasiticConfig {
            risk_tolerance: 0.8,
            aggression_factor: 3.0,
            max_concurrent_strategies: 8,
            ..ParasiticConfig::default()
        };
        
        let aggressive_system = ParasiticTradingSystem::new(aggressive_config);
        aggressive_system.activate().await.unwrap();
        sleep(Duration::from_millis(100)).await;
        let aggressive_metrics = aggressive_system.get_performance_metrics().await;
        aggressive_system.deactivate().await.unwrap();
        
        // Aggressive configuration might generate different results
        // We can't guarantee specific outcomes due to randomness, but ensure both work
        assert!(conservative_metrics.total_profit >= 0.0);
        assert!(aggressive_metrics.total_profit >= 0.0);
    }

    #[tokio::test]
    async fn test_quick_setup_methods() {
        // Test quick cuckoo setup
        let cuckoo_result = ParasiticTradingSystem::quick_cuckoo_setup().await;
        assert!(cuckoo_result.is_ok());
        
        let cuckoo_system = cuckoo_result.unwrap();
        assert!(cuckoo_system.is_active());
        assert_eq!(cuckoo_system.get_config().enabled_parasites.len(), 1);
        assert_eq!(cuckoo_system.get_config().enabled_parasites[0], ParasiteType::Cuckoo);
        assert_eq!(cuckoo_system.get_config().aggression_factor, 2.0);
        
        cuckoo_system.deactivate().await.unwrap();
        
        // Test high frequency setup
        let hft_result = ParasiticTradingSystem::high_frequency_setup().await;
        assert!(hft_result.is_ok());
        
        let hft_system = hft_result.unwrap();
        assert!(hft_system.is_active());
        assert_eq!(hft_system.get_config().max_concurrent_strategies, 16);
        assert_eq!(hft_system.get_config().quantum_threads, 8);
        assert_eq!(hft_system.get_config().simd_level, 2);
        assert_eq!(hft_system.get_config().aggression_factor, 3.0);
        assert_eq!(hft_system.get_config().risk_tolerance, 0.5);
        
        hft_system.deactivate().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_system_operations() {
        let system = std::sync::Arc::new(ParasiticTradingSystem::new_default());
        system.activate().await.unwrap();
        
        // Spawn concurrent operations
        let mut handles = vec![];
        
        // Multiple status checks
        for _ in 0..5 {
            let system_clone = system.clone();
            let handle = tokio::spawn(async move {
                system_clone.get_status().await
            });
            handles.push(handle);
        }
        
        // Multiple metrics requests
        for _ in 0..5 {
            let system_clone = system.clone();
            let handle = tokio::spawn(async move {
                system_clone.get_performance_metrics().await
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        for handle in handles {
            let result = handle.await;
            assert!(result.is_ok());
        }
        
        system.deactivate().await.unwrap();
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_double_activation() {
        let system = ParasiticTradingSystem::new_default();
        
        // First activation should succeed
        let result1 = system.activate().await;
        assert!(result1.is_ok());
        
        // Second activation should still work (idempotent)
        let result2 = system.activate().await;
        assert!(result2.is_ok());
        
        system.deactivate().await.unwrap();
    }

    #[tokio::test]
    async fn test_double_deactivation() {
        let system = ParasiticTradingSystem::new_default();
        system.activate().await.unwrap();
        
        // First deactivation should succeed
        let result1 = system.deactivate().await;
        assert!(result1.is_ok());
        
        // Second deactivation should still work (idempotent)
        let result2 = system.deactivate().await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_operations_on_inactive_system() {
        let system = ParasiticTradingSystem::new_default();
        // System is not activated
        
        // These operations should work on inactive system
        let status = system.get_status().await;
        assert_eq!(status["status"], "inactive");
        
        let metrics = system.get_performance_metrics().await;
        assert_eq!(metrics.total_profit, 0.0);
        
        // Adding strategies to inactive system should work
        let result = system.add_strategy(ParasiteType::Cuckoo).await;
        assert!(result.is_ok()); // Should queue for when activated
    }

    #[tokio::test] 
    async fn test_strategy_limit_enforcement() {
        let config = ParasiticConfig {
            max_concurrent_strategies: 2,
            ..ParasiticConfig::default()
        };
        
        let system = ParasiticTradingSystem::new(config);
        system.activate().await.unwrap();
        
        // Add strategies up to limit
        let result1 = system.add_strategy(ParasiteType::Cuckoo).await;
        assert!(result1.is_ok());
        
        let result2 = system.add_strategy(ParasiteType::Cuckoo).await;
        assert!(result2.is_ok());
        
        // Adding beyond limit should fail
        let result3 = system.add_strategy(ParasiteType::Cuckoo).await;
        assert!(result3.is_err());
        
        let error_msg = result3.unwrap_err().to_string();
        assert!(error_msg.contains("Maximum number of concurrent strategies"));
        
        system.deactivate().await.unwrap();
    }

    #[tokio::test]
    async fn test_unsupported_strategy_creation() {
        let result = create_parasite_strategy(ParasiteType::Wasp).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
        
        let result = create_parasite_strategy(ParasiteType::Cordyceps).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
        
        let result = create_parasite_strategy(ParasiteType::Lamprey).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}

#[cfg(test)]
mod parasite_type_tests {
    use super::*;

    #[test]
    fn test_parasite_type_properties() {
        let cuckoo = ParasiteType::Cuckoo;
        assert!(cuckoo.efficiency_rating() > 0.9);
        assert!(cuckoo.risk_level() < 0.4);
        assert!(!cuckoo.description().is_empty());
        
        let wasp = ParasiteType::Wasp;
        assert!(wasp.efficiency_rating() > 0.8);
        assert!(wasp.risk_level() < 0.6);
        assert!(!wasp.description().is_empty());
        
        let cordyceps = ParasiteType::Cordyceps;
        assert!(cordyceps.efficiency_rating() > 0.9);
        assert!(cordyceps.risk_level() > 0.7);
        assert!(!cordyceps.description().is_empty());
        
        let lamprey = ParasiteType::Lamprey;
        assert!(lamprey.efficiency_rating() > 0.7);
        assert!(lamprey.risk_level() < 0.3);
        assert!(!lamprey.description().is_empty());
    }

    #[test]
    fn test_parasite_type_efficiency_ranking() {
        let parasites = [
            ParasiteType::Cuckoo,
            ParasiteType::Wasp, 
            ParasiteType::Cordyceps,
            ParasiteType::Lamprey,
        ];
        
        // Cordyceps should be highest efficiency
        let max_efficiency = parasites.iter()
            .map(|p| p.efficiency_rating())
            .fold(0.0f64, f64::max);
        
        assert_eq!(max_efficiency, ParasiteType::Cordyceps.efficiency_rating());
        
        // Lamprey should be lowest efficiency
        let min_efficiency = parasites.iter()
            .map(|p| p.efficiency_rating())
            .fold(1.0f64, f64::min);
        
        assert_eq!(min_efficiency, ParasiteType::Lamprey.efficiency_rating());
    }

    #[test]
    fn test_parasite_type_risk_ranking() {
        // Lamprey should be lowest risk
        assert!(ParasiteType::Lamprey.risk_level() < ParasiteType::Cuckoo.risk_level());
        assert!(ParasiteType::Cuckoo.risk_level() < ParasiteType::Wasp.risk_level());
        assert!(ParasiteType::Wasp.risk_level() < ParasiteType::Cordyceps.risk_level());
        
        // Cordyceps should be highest risk
        let max_risk = [
            ParasiteType::Cuckoo,
            ParasiteType::Wasp,
            ParasiteType::Cordyceps, 
            ParasiteType::Lamprey,
        ].iter()
        .map(|p| p.risk_level())
        .fold(0.0f64, f64::max);
        
        assert_eq!(max_risk, ParasiteType::Cordyceps.risk_level());
    }
}