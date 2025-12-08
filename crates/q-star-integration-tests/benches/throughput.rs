//! Q* Throughput Benchmarks
//! 
//! Measures system throughput and scalability

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_star_core::*;
use q_star_orchestrator::*;
use std::time::Duration;
use futures::stream::{FuturesUnordered, StreamExt};

fn bench_single_agent_throughput(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("single_agent_throughput");
    group.measurement_time(Duration::from_secs(20));
    
    for agent_type in [AgentType::Explorer, AgentType::Exploiter, AgentType::Coordinator] {
        let agent = runtime.block_on(async {
            match agent_type {
                AgentType::Explorer => Box::new(ExplorerAgent::default()) as Box<dyn QStarAgent>,
                AgentType::Exploiter => Box::new(ExploiterAgent::default()) as Box<dyn QStarAgent>,
                AgentType::Coordinator => Box::new(CoordinatorAgent::default()) as Box<dyn QStarAgent>,
                _ => panic!("Unsupported agent type"),
            }
        });
        
        let states: Vec<MarketState> = (0..1000).map(|_| create_test_state()).collect();
        
        group.throughput(Throughput::Elements(states.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", agent_type)),
            &states,
            |b, states| {
                b.to_async(&runtime).iter(|| async {
                    for state in states {
                        black_box(agent.decide(state).await.unwrap());
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_orchestrator_throughput(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("orchestrator_throughput");
    group.measurement_time(Duration::from_secs(30));
    
    for num_agents in [10, 50, 100, 200] {
        let orchestrator = runtime.block_on(async {
            let config = OrchestratorConfig {
                topology: SwarmTopology::Mesh,
                max_agents: num_agents * 2,
                min_agents: num_agents / 2,
                spawn_strategy: SpawnStrategy::Aggressive,
                coordination_strategy: CoordinationStrategy::Parallel,
                consensus_mechanism: ConsensusMechanism::Optimistic,
                health_check_interval: Duration::from_secs(60),
                auto_scale: false,
                fault_tolerance: false,
                performance_targets: PerformanceTargets {
                    max_latency_us: 100,
                    min_throughput: 1_000_000,
                    max_memory_mb: 1000,
                    target_accuracy: 0.9,
                },
            };
            
            let orch = QStarOrchestrator::new(config).await.unwrap();
            
            // Pre-spawn agents
            for _ in 0..num_agents {
                orch.spawn_agent(AgentType::Explorer).await.unwrap();
            }
            
            orch
        });
        
        let tasks: Vec<QStarTask> = (0..1000).map(|i| QStarTask {
            id: format!("throughput_test_{}", i),
            state: create_test_state(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::Medium,
        }).collect();
        
        group.throughput(Throughput::Elements(tasks.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}agents", num_agents)),
            &tasks,
            |b, tasks| {
                b.to_async(&runtime).iter(|| async {
                    let mut futures = FuturesUnordered::new();
                    
                    for task in tasks {
                        let task_id = orchestrator.submit_task(task.clone()).await.unwrap();
                        futures.push(orchestrator.await_result(&task_id));
                    }
                    
                    while let Some(result) = futures.next().await {
                        black_box(result.unwrap());
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_processing(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .build()
        .unwrap();
    
    let mut group = c.benchmark_group("parallel_processing");
    group.measurement_time(Duration::from_secs(20));
    
    let orchestrator = runtime.block_on(async {
        let config = OrchestratorConfig {
            topology: SwarmTopology::Hierarchical,
            max_agents: 100,
            min_agents: 20,
            spawn_strategy: SpawnStrategy::Adaptive,
            coordination_strategy: CoordinationStrategy::Parallel,
            consensus_mechanism: ConsensusMechanism::MajorityVote,
            health_check_interval: Duration::from_secs(60),
            auto_scale: true,
            fault_tolerance: false,
            performance_targets: PerformanceTargets::default(),
        };
        
        QStarOrchestrator::new(config).await.unwrap()
    });
    
    for parallelism in [1, 2, 4, 8, 16] {
        runtime.block_on(async {
            orchestrator.clear_agents().await.unwrap();
            for _ in 0..(parallelism * 5) {
                orchestrator.spawn_agent(AgentType::Explorer).await.unwrap();
            }
        });
        
        let tasks: Vec<QStarTask> = (0..1000).map(|i| QStarTask {
            id: format!("parallel_test_{}_{}", parallelism, i),
            state: create_test_state(),
            constraints: TaskConstraints::default(),
            priority: TaskPriority::Medium,
        }).collect();
        
        group.throughput(Throughput::Elements(tasks.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x", parallelism)),
            &tasks,
            |b, tasks| {
                b.to_async(&runtime).iter(|| async {
                    let chunks: Vec<_> = tasks.chunks(tasks.len() / parallelism).collect();
                    let mut handles = vec![];
                    
                    for chunk in chunks {
                        let orchestrator = orchestrator.clone();
                        let chunk = chunk.to_vec();
                        
                        let handle = tokio::spawn(async move {
                            let mut results = vec![];
                            for task in chunk {
                                let id = orchestrator.submit_task(task).await.unwrap();
                                results.push(orchestrator.await_result(&id).await.unwrap());
                            }
                            results
                        });
                        
                    handles.push(handle);
                    }
                    
                    for handle in handles {
                        black_box(handle.await.unwrap());
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_burst_handling(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("burst_handling");
    group.measurement_time(Duration::from_secs(15));
    
    let orchestrator = runtime.block_on(async {
        let config = OrchestratorConfig {
            topology: SwarmTopology::Star,
            max_agents: 50,
            min_agents: 10,
            spawn_strategy: SpawnStrategy::Aggressive,
            coordination_strategy: CoordinationStrategy::LoadBalanced,
            consensus_mechanism: ConsensusMechanism::FirstValid,
            health_check_interval: Duration::from_secs(60),
            auto_scale: true,
            fault_tolerance: true,
            performance_targets: PerformanceTargets::default(),
        };
        
        let orch = QStarOrchestrator::new(config).await.unwrap();
        
        // Pre-spawn minimum agents
        for _ in 0..10 {
            orch.spawn_agent(AgentType::Explorer).await.unwrap();
        }
        
        orch
    });
    
    for burst_size in [100, 500, 1000, 5000] {
        let tasks: Vec<QStarTask> = (0..burst_size).map(|i| QStarTask {
            id: format!("burst_test_{}", i),
            state: create_test_state(),
            constraints: TaskConstraints {
                max_latency_us: 100,
                required_confidence: 0.8,
                risk_limit: 0.02,
            },
            priority: TaskPriority::High,
        }).collect();
        
        group.throughput(Throughput::Elements(tasks.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}burst", burst_size)),
            &tasks,
            |b, tasks| {
                b.to_async(&runtime).iter(|| async {
                    // Submit all tasks as fast as possible
                    let mut task_ids = vec![];
                    for task in tasks {
                        task_ids.push(orchestrator.submit_task(task.clone()).await.unwrap());
                    }
                    
                    // Wait for all results
                    for id in task_ids {
                        black_box(orchestrator.await_result(&id).await.unwrap());
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_sustained_throughput(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("sustained_throughput");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);
    
    let orchestrator = runtime.block_on(async {
        let config = OrchestratorConfig {
            topology: SwarmTopology::Mesh,
            max_agents: 100,
            min_agents: 50,
            spawn_strategy: SpawnStrategy::Balanced,
            coordination_strategy: CoordinationStrategy::Adaptive,
            consensus_mechanism: ConsensusMechanism::WeightedVote,
            health_check_interval: Duration::from_secs(5),
            auto_scale: true,
            fault_tolerance: true,
            performance_targets: PerformanceTargets {
                max_latency_us: 50,
                min_throughput: 1_000_000,
                max_memory_mb: 500,
                target_accuracy: 0.95,
            },
        };
        
        let orch = QStarOrchestrator::new(config).await.unwrap();
        
        // Pre-spawn optimal number of agents
        for _ in 0..75 {
            orch.spawn_agent(AgentType::Explorer).await.unwrap();
        }
        for _ in 0..20 {
            orch.spawn_agent(AgentType::Exploiter).await.unwrap();
        }
        for _ in 0..5 {
            orch.spawn_agent(AgentType::Coordinator).await.unwrap();
        }
        
        orch
    });
    
    // Generate diverse workload
    let tasks: Vec<QStarTask> = (0..10000).map(|i| {
        let state = match i % 4 {
            0 => create_trending_state(),
            1 => create_ranging_state(),
            2 => create_volatile_state(),
            _ => create_crisis_state(),
        };
        
        QStarTask {
            id: format!("sustained_test_{}", i),
            state,
            constraints: TaskConstraints::default(),
            priority: match i % 3 {
                0 => TaskPriority::High,
                1 => TaskPriority::Medium,
                _ => TaskPriority::Low,
            },
        }
    }).collect();
    
    group.throughput(Throughput::Elements(tasks.len() as u64));
    group.bench_function("sustained_1M_decisions", |b| {
        b.to_async(&runtime).iter(|| async {
            let start = std::time::Instant::now();
            let mut completed = 0u64;
            
            // Process tasks in batches
            for chunk in tasks.chunks(100) {
                let mut futures = FuturesUnordered::new();
                
                for task in chunk {
                    let id = orchestrator.submit_task(task.clone()).await.unwrap();
                    futures.push(orchestrator.await_result(&id));
                }
                
                while let Some(result) = futures.next().await {
                    if result.is_ok() {
                        completed += 1;
                    }
                }
            }
            
            let elapsed = start.elapsed();
            let throughput = completed as f64 / elapsed.as_secs_f64();
            
            assert!(throughput > 1_000_000.0, "Throughput {} < 1M decisions/sec", throughput);
            black_box(completed);
        });
    });
    
    group.finish();
}

// Helper functions
fn create_test_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 101.0, 99.5, 100.5, 101.5],
        volumes: vec![1000.0, 1100.0, 900.0, 1050.0, 1200.0],
        technical_indicators: vec![0.5, 0.6, 0.55, 0.58, 0.62],
        market_regime: MarketRegime::Trending,
        volatility: 0.02,
        liquidity: 0.85,
    }
}

fn create_trending_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 101.0, 102.0, 103.0, 104.0],
        volumes: vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        technical_indicators: vec![0.7, 0.75, 0.8, 0.85, 0.9],
        market_regime: MarketRegime::Trending,
        volatility: 0.015,
        liquidity: 0.9,
    }
}

fn create_ranging_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 100.5, 99.8, 100.2, 100.0],
        volumes: vec![1000.0, 900.0, 950.0, 1000.0, 980.0],
        technical_indicators: vec![0.5, 0.48, 0.52, 0.49, 0.51],
        market_regime: MarketRegime::Ranging,
        volatility: 0.01,
        liquidity: 0.7,
    }
}

fn create_volatile_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 105.0, 98.0, 103.0, 95.0],
        volumes: vec![2000.0, 2500.0, 3000.0, 2200.0, 2800.0],
        technical_indicators: vec![0.3, 0.8, 0.2, 0.7, 0.4],
        market_regime: MarketRegime::Volatile,
        volatility: 0.05,
        liquidity: 0.6,
    }
}

fn create_crisis_state() -> MarketState {
    MarketState {
        timestamp: chrono::Utc::now(),
        prices: vec![100.0, 95.0, 90.0, 85.0, 80.0],
        volumes: vec![5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
        technical_indicators: vec![0.2, 0.15, 0.1, 0.05, 0.0],
        market_regime: MarketRegime::Crisis,
        volatility: 0.1,
        liquidity: 0.3,
    }
}

criterion_group!(
    benches,
    bench_single_agent_throughput,
    bench_orchestrator_throughput,
    bench_parallel_processing,
    bench_burst_handling,
    bench_sustained_throughput
);

criterion_main!(benches);