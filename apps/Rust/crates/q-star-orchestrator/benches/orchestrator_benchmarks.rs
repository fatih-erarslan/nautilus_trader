//! Benchmarks for Q* Orchestrator
//!
//! This benchmark suite measures the performance of the Q* orchestrator
//! across different scenarios and configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_star_orchestrator::{
    QStarOrchestrator, OrchestratorConfig, SwarmTopology, SpawnStrategy,
    SchedulingStrategy, ConsensusMechanism, factory,
};
use q_star_core::{MarketState, MarketRegime, QStarConfig, Experience, QStarAction};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create a test market state for benchmarking
fn create_test_market_state() -> MarketState {
    MarketState::new(
        50000.0,     // price
        1000000.0,   // volume
        0.02,        // volatility
        0.5,         // momentum
        0.001,       // spread
        MarketRegime::Trending,
        vec![0.1, 0.2, 0.3, 0.4, 0.5], // features
    )
}

/// Create a test experience for training benchmarks
fn create_test_experience() -> Experience {
    Experience {
        state: create_test_market_state(),
        action: QStarAction::Buy { amount: 1000.0 },
        reward: 0.1,
        next_state: create_test_market_state(),
        done: false,
    }
}

/// Benchmark orchestrator creation with different configurations
fn bench_orchestrator_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("orchestrator_creation");
    
    // Test different topologies
    let topologies = vec![
        ("mesh", SwarmTopology::Mesh),
        ("hierarchical", SwarmTopology::Hierarchical { levels: 3 }),
        ("ring", SwarmTopology::Ring),
        ("star", SwarmTopology::Star),
        ("dynamic", SwarmTopology::Dynamic),
    ];
    
    for (name, topology) in topologies {
        group.bench_with_input(
            BenchmarkId::new("topology", name),
            &topology,
            |b, topology| {
                b.to_async(&rt).iter(|| async {
                    let config = OrchestratorConfig {
                        topology: topology.clone(),
                        max_agents: 20,
                        min_agents: 5,
                        spawn_strategy: SpawnStrategy::Fixed {
                            explorers: 2,
                            exploiters: 2,
                            coordinators: 1,
                            quantum: 2,
                        },
                        scheduling_strategy: SchedulingStrategy::LoadBalanced,
                        consensus_mechanism: ConsensusMechanism::WeightedVote,
                        health_check_interval_ms: 1000,
                        monitoring_interval_ms: 100,
                        enable_autoscaling: false,
                        enable_fault_tolerance: false,
                        max_coordination_latency_us: 1000,
                    };
                    
                    let q_star_config = QStarConfig::default();
                    
                    let orchestrator = QStarOrchestrator::new(
                        config,
                        q_star_config,
                        10000.0,
                    ).await;
                    
                    black_box(orchestrator)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark decision making performance
fn bench_decision_making(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("decision_making");
    
    // Test different agent counts
    let agent_counts = vec![5, 10, 20, 50];
    
    for count in agent_counts {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("agents", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        rt.block_on(async {
                            let config = OrchestratorConfig {
                                topology: SwarmTopology::Hierarchical { levels: 3 },
                                max_agents: count,
                                min_agents: count,
                                spawn_strategy: SpawnStrategy::Fixed {
                                    explorers: count / 4,
                                    exploiters: count / 4,
                                    coordinators: count / 4,
                                    quantum: count / 4,
                                },
                                scheduling_strategy: SchedulingStrategy::LoadBalanced,
                                consensus_mechanism: ConsensusMechanism::WeightedVote,
                                health_check_interval_ms: 10000,
                                monitoring_interval_ms: 1000,
                                enable_autoscaling: false,
                                enable_fault_tolerance: false,
                                max_coordination_latency_us: 10000,
                            };
                            
                            let q_star_config = QStarConfig::default();
                            
                            let orchestrator = QStarOrchestrator::new(
                                config,
                                q_star_config,
                                10000.0,
                            ).await.unwrap();
                            
                            // Allow time for initialization
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            
                            (orchestrator, create_test_market_state())
                        })
                    },
                    |(orchestrator, state)| async move {
                        let result = orchestrator.decide(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus mechanisms
fn bench_consensus_mechanisms(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("consensus_mechanisms");
    
    let mechanisms = vec![
        ("majority_vote", ConsensusMechanism::MajorityVote),
        ("weighted_vote", ConsensusMechanism::WeightedVote),
        ("byzantine", ConsensusMechanism::Byzantine),
        ("quantum", ConsensusMechanism::Quantum),
    ];
    
    for (name, mechanism) in mechanisms {
        group.bench_with_input(
            BenchmarkId::new("mechanism", name),
            &mechanism,
            |b, mechanism| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        rt.block_on(async {
                            let config = OrchestratorConfig {
                                topology: SwarmTopology::Mesh,
                                max_agents: 10,
                                min_agents: 10,
                                spawn_strategy: SpawnStrategy::Fixed {
                                    explorers: 3,
                                    exploiters: 3,
                                    coordinators: 2,
                                    quantum: 2,
                                },
                                scheduling_strategy: SchedulingStrategy::LoadBalanced,
                                consensus_mechanism: mechanism.clone(),
                                health_check_interval_ms: 10000,
                                monitoring_interval_ms: 1000,
                                enable_autoscaling: false,
                                enable_fault_tolerance: false,
                                max_coordination_latency_us: 10000,
                            };
                            
                            let q_star_config = QStarConfig::default();
                            
                            let orchestrator = QStarOrchestrator::new(
                                config,
                                q_star_config,
                                10000.0,
                            ).await.unwrap();
                            
                            // Allow time for initialization
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            
                            (orchestrator, create_test_market_state())
                        })
                    },
                    |(orchestrator, state)| async move {
                        let result = orchestrator.decide(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark training performance
fn bench_training_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("training_performance");
    
    group.bench_function("single_experience", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                rt.block_on(async {
                    let orchestrator = factory::create_production_orchestrator(10000.0)
                        .await
                        .unwrap();
                    
                    // Allow time for initialization
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    
                    (orchestrator, create_test_experience())
                })
            },
            |(orchestrator, experience)| async move {
                let result = orchestrator.train(&experience).await;
                black_box(result)
            },
        );
    });
    
    group.finish();
}

/// Benchmark agent spawning and removal
fn bench_agent_lifecycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("agent_lifecycle");
    
    let agent_types = vec!["explorer", "exploiter", "coordinator", "quantum"];
    
    for agent_type in agent_types {
        group.bench_with_input(
            BenchmarkId::new("spawn", agent_type),
            &agent_type,
            |b, &agent_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        rt.block_on(async {
                            let orchestrator = factory::create_production_orchestrator(10000.0)
                                .await
                                .unwrap();
                            
                            // Allow time for initialization
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            
                            orchestrator
                        })
                    },
                    |orchestrator| async move {
                        let result = orchestrator.spawn_agent(agent_type).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark high-frequency trading scenario
fn bench_hft_scenario(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hft_scenario");
    
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("ultra_low_latency", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                rt.block_on(async {
                    let orchestrator = factory::create_hft_orchestrator(10000.0)
                        .await
                        .unwrap();
                    
                    // Allow time for initialization
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    
                    (orchestrator, create_test_market_state())
                })
            },
            |(orchestrator, state)| async move {
                let result = orchestrator.decide(&state).await;
                black_box(result)
            },
        );
    });
    
    group.finish();
}

/// Benchmark swarm status retrieval
fn bench_swarm_status(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("swarm_status");
    
    group.bench_function("get_status", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                rt.block_on(async {
                    let orchestrator = factory::create_production_orchestrator(10000.0)
                        .await
                        .unwrap();
                    
                    // Allow time for initialization
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    
                    orchestrator
                })
            },
            |orchestrator| async move {
                let status = orchestrator.get_swarm_status().await;
                black_box(status)
            },
        );
    });
    
    group.finish();
}

/// Benchmark memory usage under load
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_usage");
    
    group.bench_function("sustained_decisions", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                rt.block_on(async {
                    let orchestrator = factory::create_production_orchestrator(10000.0)
                        .await
                        .unwrap();
                    
                    // Allow time for initialization
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    
                    orchestrator
                })
            },
            |orchestrator| async move {
                let state = create_test_market_state();
                
                // Make multiple decisions to test memory usage
                for _ in 0..10 {
                    let result = orchestrator.decide(&state).await;
                    black_box(result);
                }
            },
        );
    });
    
    group.finish();
}

/// Benchmark coordination latency
fn bench_coordination_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("coordination_latency");
    
    // Test different coordination latency limits
    let latency_limits = vec![100, 500, 1000, 5000]; // microseconds
    
    for limit in latency_limits {
        group.bench_with_input(
            BenchmarkId::new("latency_limit_us", limit),
            &limit,
            |b, &limit| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        rt.block_on(async {
                            let config = OrchestratorConfig {
                                topology: SwarmTopology::Star,
                                max_agents: 10,
                                min_agents: 10,
                                spawn_strategy: SpawnStrategy::Fixed {
                                    explorers: 3,
                                    exploiters: 3,
                                    coordinators: 2,
                                    quantum: 2,
                                },
                                scheduling_strategy: SchedulingStrategy::LoadBalanced,
                                consensus_mechanism: ConsensusMechanism::WeightedVote,
                                health_check_interval_ms: 10000,
                                monitoring_interval_ms: 1000,
                                enable_autoscaling: false,
                                enable_fault_tolerance: false,
                                max_coordination_latency_us: limit,
                            };
                            
                            let q_star_config = QStarConfig::default();
                            
                            let orchestrator = QStarOrchestrator::new(
                                config,
                                q_star_config,
                                10000.0,
                            ).await.unwrap();
                            
                            // Allow time for initialization
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            
                            (orchestrator, create_test_market_state())
                        })
                    },
                    |(orchestrator, state)| async move {
                        let result = orchestrator.decide(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_orchestrator_creation,
    bench_decision_making,
    bench_consensus_mechanisms,
    bench_training_performance,
    bench_agent_lifecycle,
    bench_hft_scenario,
    bench_swarm_status,
    bench_memory_usage,
    bench_coordination_latency
);

criterion_main!(benches);