//! Test helper functions

use q_star_core::*;
use q_star_orchestrator::*;
use std::sync::Arc;
use std::time::Duration;

/// Create a test orchestrator with default settings
pub async fn create_default_orchestrator() -> Arc<QStarOrchestrator> {
    Arc::new(QStarOrchestrator::new(OrchestratorConfig::default()).await.unwrap())
}

/// Create a high-performance orchestrator for benchmarks
pub async fn create_benchmark_orchestrator() -> Arc<QStarOrchestrator> {
    let config = OrchestratorConfig {
        topology: SwarmTopology::Mesh,
        max_agents: 100,
        min_agents: 20,
        spawn_strategy: SpawnStrategy::Aggressive,
        coordination_strategy: CoordinationStrategy::Parallel,
        consensus_mechanism: ConsensusMechanism::Optimistic,
        health_check_interval: Duration::from_secs(60),
        auto_scale: true,
        fault_tolerance: false,
        performance_targets: PerformanceTargets {
            max_latency_us: 10,
            min_throughput: 1_000_000,
            max_memory_mb: 500,
            target_accuracy: 0.95,
        },
    };
    
    Arc::new(QStarOrchestrator::new(config).await.unwrap())
}

/// Create test market states
pub fn create_market_states() -> Vec<MarketState> {
    vec![
        // Trending market
        MarketState {
            timestamp: chrono::Utc::now(),
            prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
            volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            technical_indicators: vec![0.7, 0.75, 0.8, 0.85, 0.9],
            market_regime: MarketRegime::Trending,
            volatility: 0.015,
            liquidity: 0.9,
        },
        // Ranging market
        MarketState {
            timestamp: chrono::Utc::now(),
            prices: vec![100.0, 100.5, 99.8, 100.2, 100.0],
            volumes: vec![1000.0, 900.0, 950.0, 1000.0, 980.0],
            technical_indicators: vec![0.5, 0.48, 0.52, 0.49, 0.51],
            market_regime: MarketRegime::Ranging,
            volatility: 0.01,
            liquidity: 0.7,
        },
        // Volatile market
        MarketState {
            timestamp: chrono::Utc::now(),
            prices: vec![100.0, 105.0, 98.0, 103.0, 95.0],
            volumes: vec![2000.0, 2500.0, 3000.0, 2200.0, 2800.0],
            technical_indicators: vec![0.3, 0.8, 0.2, 0.7, 0.4],
            market_regime: MarketRegime::Volatile,
            volatility: 0.05,
            liquidity: 0.6,
        },
        // Crisis scenario
        MarketState {
            timestamp: chrono::Utc::now(),
            prices: vec![100.0, 95.0, 90.0, 85.0, 80.0],
            volumes: vec![5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
            technical_indicators: vec![0.2, 0.15, 0.1, 0.05, 0.0],
            market_regime: MarketRegime::Crisis,
            volatility: 0.1,
            liquidity: 0.3,
        },
    ]
}

/// Calculate percentile from sorted data
pub fn percentile(data: &[u128], p: f64) -> u128 {
    let idx = ((data.len() as f64 - 1.0) * p / 100.0) as usize;
    data[idx]
}

/// Measure average and P99 latency
pub async fn measure_latency<F, Fut>(iterations: usize, mut f: F) -> (u128, u128)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future,
{
    let mut latencies = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        f().await;
        latencies.push(start.elapsed().as_micros());
    }
    
    latencies.sort_unstable();
    let avg = latencies.iter().sum::<u128>() / latencies.len() as u128;
    let p99 = percentile(&latencies, 99.0);
    
    (avg, p99)
}