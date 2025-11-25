//! Comprehensive Integration Tests for HyperPhysics HFT Ecosystem
//!
//! Tests the integration of physics engines, biomimetic algorithms,
//! and trading decision systems.

use hyperphysics_hft_ecosystem::core::*;

#[tokio::test]
async fn test_ecosystem_builder_defaults() {
    let ecosystem = HFTEcosystem::builder()
        .build()
        .await
        .expect("Failed to build ecosystem with defaults");

    // Verify default configuration
    assert!(ecosystem.config.formal_verification);
    assert_eq!(ecosystem.config.target_latency_us, 1000);
    assert!(ecosystem.config.simd_enabled);
}

#[tokio::test]
async fn test_ecosystem_builder_custom_config() {
    let ecosystem = HFTEcosystem::builder()
        .with_physics_engine(PhysicsEngine::Rapier)
        .with_biomimetic_tier(BiomimeticTier::Tier1)
        .with_formal_verification(false)
        .with_target_latency_us(500)
        .build()
        .await
        .expect("Failed to build custom ecosystem");

    assert!(!ecosystem.config.formal_verification);
    assert_eq!(ecosystem.config.target_latency_us, 500);
}

#[tokio::test]
async fn test_ecosystem_builder_all_physics_engines() {
    // Test that all physics engines can be configured
    let engines = vec![
        PhysicsEngine::Jolt,
        PhysicsEngine::Rapier,
        PhysicsEngine::Avian,
        PhysicsEngine::Warp,
        PhysicsEngine::Taichi,
        PhysicsEngine::MuJoCo,
        PhysicsEngine::Genesis,
    ];

    for engine in engines {
        let result = HFTEcosystem::builder()
            .with_physics_engine(engine)
            .build()
            .await;

        assert!(result.is_ok(), "Failed to build with engine: {:?}", engine);
    }
}

#[tokio::test]
async fn test_ecosystem_builder_all_biomimetic_tiers() {
    // Test all biomimetic tiers
    let tiers = vec![
        BiomimeticTier::Tier1,
        BiomimeticTier::Tier2,
        BiomimeticTier::Tier3,
        BiomimeticTier::All,
    ];

    for tier in tiers {
        let result = HFTEcosystem::builder()
            .with_biomimetic_tier(tier)
            .build()
            .await;

        assert!(result.is_ok(), "Failed to build with tier: {:?}", tier);
    }
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_physics_router_rapier() {
    let router = PhysicsEngineRouter::new(PhysicsEngine::Rapier, true);
    let market_tick = MarketTick::default();

    let result = router.route(&market_tick).await;
    assert!(result.is_ok(), "Rapier routing failed");

    let physics_result = result.unwrap();
    assert!(physics_result.latency_us > 0, "Latency should be non-zero");
    assert!(physics_result.confidence >= 0.0 && physics_result.confidence <= 1.0);
}

#[tokio::test]
#[cfg(not(feature = "physics-rapier"))]
async fn test_physics_router_rapier_disabled() {
    let router = PhysicsEngineRouter::new(PhysicsEngine::Rapier, true);
    let market_tick = MarketTick::default();

    let result = router.route(&market_tick).await;
    assert!(result.is_err(), "Should fail when Rapier feature disabled");
}

#[tokio::test]
async fn test_physics_router_unimplemented_engines() {
    let engines = vec![
        PhysicsEngine::Avian,
        PhysicsEngine::Warp,
        PhysicsEngine::Taichi,
        PhysicsEngine::MuJoCo,
        PhysicsEngine::Genesis,
    ];

    let market_tick = MarketTick::default();

    for engine in engines {
        let router = PhysicsEngineRouter::new(engine, false);
        let result = router.route(&market_tick).await;

        assert!(result.is_err(), "Engine {:?} should not be implemented yet", engine);

        if let Err(e) = result {
            assert!(
                e.to_string().contains("not yet implemented"),
                "Error should indicate not implemented: {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_physics_router_determinism_flag() {
    // Deterministic routing
    let router_det = PhysicsEngineRouter::new(PhysicsEngine::Jolt, true);
    assert_eq!(router_det.deterministic, true);

    // Non-deterministic routing
    let router_non_det = PhysicsEngineRouter::new(PhysicsEngine::Rapier, false);
    assert_eq!(router_non_det.deterministic, false);
}

#[tokio::test]
async fn test_biomimetic_coordinator_tier1() {
    let coord = BiomimeticCoordinator::new(BiomimeticTier::Tier1);
    assert!(coord.is_ok(), "Failed to create Tier1 coordinator");
}

#[tokio::test]
async fn test_biomimetic_coordinator_all_tiers() {
    let tiers = vec![
        BiomimeticTier::Tier1,
        BiomimeticTier::Tier2,
        BiomimeticTier::Tier3,
        BiomimeticTier::All,
    ];

    for tier in tiers {
        let result = BiomimeticCoordinator::new(tier);
        assert!(result.is_ok(), "Failed to create coordinator for tier: {:?}", tier);
    }
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_biomimetic_coordinator_swarm_execution() {
    let mut coord = BiomimeticCoordinator::new(BiomimeticTier::Tier1)
        .expect("Failed to create coordinator");

    // Create mock physics result
    let physics_result = PhysicsResult {
        latency_us: 500,
        confidence: 0.8,
        data: vec![],
    };

    let decision = coord.coordinate_swarms(&physics_result).await;
    assert!(decision.is_ok(), "Swarm coordination failed");

    let decision = decision.unwrap();
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert_eq!(decision.tier_used, BiomimeticTier::Tier1);
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_full_ecosystem_execution_cycle() {
    let ecosystem = HFTEcosystem::builder()
        .with_physics_engine(PhysicsEngine::Rapier)
        .with_biomimetic_tier(BiomimeticTier::Tier1)
        .build()
        .await
        .expect("Failed to build ecosystem");

    let market_tick = MarketTick::default();
    let decision = ecosystem.execute_cycle(&market_tick).await;

    assert!(decision.is_ok(), "Execution cycle failed");

    let decision = decision.unwrap();
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    assert!(decision.size > 0.0);
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_ecosystem_latency_target() {
    let ecosystem = HFTEcosystem::builder()
        .with_physics_engine(PhysicsEngine::Rapier)
        .with_target_latency_us(1000) // 1ms target
        .build()
        .await
        .expect("Failed to build ecosystem");

    let start = std::time::Instant::now();
    let market_tick = MarketTick::default();
    let result = ecosystem.execute_cycle(&market_tick).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Execution failed");

    // Verify latency is reasonable (not strict enforcement, just sanity check)
    assert!(
        elapsed.as_micros() < 100_000, // 100ms max (very generous)
        "Latency too high: {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_market_tick_default() {
    let tick = MarketTick::default();
    assert!(tick.orderbook.is_empty());
    assert!(tick.trades.is_empty());
}

#[tokio::test]
async fn test_trading_decision_conversion() {
    let biomimetic_decision = BiomimeticDecision {
        consensus: Action::Buy,
        confidence: 0.85,
        tier_used: BiomimeticTier::Tier1,
        latency_us: 750,
    };

    let trading_decision: TradingDecision = biomimetic_decision.into();
    assert_eq!(trading_decision.action, Action::Buy);
    assert_eq!(trading_decision.confidence, 0.85);
    assert_eq!(trading_decision.size, 1.0); // Default
}

#[tokio::test]
async fn test_action_enum_values() {
    let actions = vec![Action::Buy, Action::Sell, Action::Hold];

    // Verify all actions are distinct
    assert_ne!(actions[0], actions[1]);
    assert_ne!(actions[1], actions[2]);
    assert_ne!(actions[0], actions[2]);
}

#[tokio::test]
async fn test_ecosystem_config_default_values() {
    let config = EcosystemConfig::default();

    assert!(config.formal_verification);
    assert_eq!(config.target_latency_us, 1000);
    assert!(config.simd_enabled);
    assert!(config.worker_threads > 0);
}

#[tokio::test]
async fn test_ecosystem_config_worker_threads() {
    let config = EcosystemConfig::default();
    let cpu_count = num_cpus::get();

    assert_eq!(
        config.worker_threads, cpu_count,
        "Worker threads should match CPU count"
    );
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_multiple_execution_cycles() {
    let ecosystem = HFTEcosystem::builder()
        .with_physics_engine(PhysicsEngine::Rapier)
        .build()
        .await
        .expect("Failed to build ecosystem");

    // Run multiple cycles to test state consistency
    for _ in 0..5 {
        let market_tick = MarketTick::default();
        let result = ecosystem.execute_cycle(&market_tick).await;
        assert!(result.is_ok(), "Execution cycle failed");
    }
}

#[tokio::test]
#[cfg(feature = "physics-rapier")]
async fn test_concurrent_execution_cycles() {
    use tokio::task::JoinSet;

    let ecosystem = std::sync::Arc::new(
        HFTEcosystem::builder()
            .with_physics_engine(PhysicsEngine::Rapier)
            .build()
            .await
            .expect("Failed to build ecosystem")
    );

    let mut tasks = JoinSet::new();

    // Spawn 10 concurrent execution cycles
    for _ in 0..10 {
        let eco_clone = ecosystem.clone();
        tasks.spawn(async move {
            let market_tick = MarketTick::default();
            eco_clone.execute_cycle(&market_tick).await
        });
    }

    // Wait for all tasks to complete
    let mut success_count = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(_)) => success_count += 1,
            _ => {}
        }
    }

    assert_eq!(success_count, 10, "All concurrent executions should succeed");
}

#[tokio::test]
async fn test_physics_engine_enum_coverage() {
    // Verify all physics engines are distinct
    use std::collections::HashSet;

    let engines = vec![
        PhysicsEngine::Jolt,
        PhysicsEngine::Rapier,
        PhysicsEngine::Avian,
        PhysicsEngine::Warp,
        PhysicsEngine::Taichi,
        PhysicsEngine::MuJoCo,
        PhysicsEngine::Genesis,
    ];

    let unique: HashSet<_> = engines.iter().collect();
    assert_eq!(unique.len(), engines.len(), "All engines should be unique");
}

#[tokio::test]
async fn test_biomimetic_tier_enum_coverage() {
    use std::collections::HashSet;

    let tiers = vec![
        BiomimeticTier::Tier1,
        BiomimeticTier::Tier2,
        BiomimeticTier::Tier3,
        BiomimeticTier::All,
    ];

    let unique: HashSet<_> = tiers.iter().collect();
    assert_eq!(unique.len(), tiers.len(), "All tiers should be unique");
}
