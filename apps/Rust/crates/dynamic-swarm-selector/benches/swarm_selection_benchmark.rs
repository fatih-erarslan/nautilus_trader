// Swarm Selection Benchmark for Dynamic Swarm Selector
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use dynamic_swarm_selector::{
    SwarmSelector, SelectionCriteria, AlgorithmType, MarketData, 
    PerformanceMetrics, AdaptiveSelectionEngine, SwarmCoordinator
};
use std::time::Instant;

fn create_sample_market_data(size: usize) -> Vec<MarketData> {
    (0..size).map(|i| MarketData {
        timestamp: chrono::Utc::now(),
        symbol: "BTCUSDT".to_string(),
        price: 50000.0 + (i as f64 * 0.1),
        volume: 1000.0 + (i as f64 * 10.0),
        volatility: 0.25,
        correlation: 0.8,
        regime: "trending".to_string(),
    }).collect()
}

fn create_selection_criteria() -> SelectionCriteria {
    SelectionCriteria {
        optimization_type: "portfolio_optimization".to_string(),
        market_regime: "volatile".to_string(),
        time_horizon: chrono::Duration::hours(24),
        risk_tolerance: 0.5,
        performance_weight: 0.7,
        diversification_weight: 0.3,
    }
}

fn benchmark_swarm_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("swarm_selection");
    
    for size in [10, 50, 100, 500].iter() {
        let market_data = create_sample_market_data(*size);
        let criteria = create_selection_criteria();
        let selector = SwarmSelector::new();
        
        group.bench_with_input(BenchmarkId::new("select_optimal_swarm", size), size, |b, _| {
            b.iter(|| {
                selector.select_optimal_swarm(&market_data, &criteria)
            })
        });
    }
    group.finish();
}

fn benchmark_algorithm_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_evaluation");
    
    let market_data = create_sample_market_data(1000);
    let selector = SwarmSelector::new();
    
    for algorithm in [
        AlgorithmType::ParticleSwarm,
        AlgorithmType::AntColony,
        AlgorithmType::GeneticAlgorithm,
        AlgorithmType::DifferentialEvolution,
        AlgorithmType::GreyWolf,
    ].iter() {
        group.bench_with_input(BenchmarkId::new("evaluate_algorithm", format!("{:?}", algorithm)), algorithm, |b, algo| {
            b.iter(|| {
                selector.evaluate_algorithm(algo.clone(), &market_data)
            })
        });
    }
    group.finish();
}

fn benchmark_performance_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_tracking");
    
    let market_data = create_sample_market_data(1000);
    let selector = SwarmSelector::new();
    
    group.bench_function("track_performance", |b| {
        b.iter(|| {
            selector.track_performance(&market_data)
        })
    });
    
    group.bench_function("update_metrics", |b| {
        b.iter(|| {
            let metrics = PerformanceMetrics {
                convergence_rate: 0.95,
                solution_quality: 0.88,
                computational_efficiency: 0.92,
                robustness: 0.85,
                diversification: 0.78,
            };
            selector.update_metrics(metrics)
        })
    });
    
    group.finish();
}

fn benchmark_adaptive_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_selection");
    
    let market_data = create_sample_market_data(1000);
    let engine = AdaptiveSelectionEngine::new();
    
    group.bench_function("adapt_selection", |b| {
        b.iter(|| {
            engine.adapt_selection(&market_data)
        })
    });
    
    group.bench_function("learn_from_feedback", |b| {
        b.iter(|| {
            engine.learn_from_feedback(&market_data, 0.85)
        })
    });
    
    group.finish();
}

fn benchmark_swarm_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("swarm_coordination");
    
    let market_data = create_sample_market_data(1000);
    let coordinator = SwarmCoordinator::new();
    
    group.bench_function("coordinate_swarms", |b| {
        b.iter(|| {
            coordinator.coordinate_swarms(&market_data)
        })
    });
    
    group.bench_function("balance_workload", |b| {
        b.iter(|| {
            coordinator.balance_workload(&market_data)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_swarm_selection,
    benchmark_algorithm_evaluation,
    benchmark_performance_tracking,
    benchmark_adaptive_selection,
    benchmark_swarm_coordination
);
criterion_main!(benches);