//! Benchmarks for MCP orchestration system performance.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use mcp_orchestration::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark orchestrator startup time
fn bench_orchestrator_startup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("orchestrator_startup", |b| {
        b.to_async(&rt).iter(|| async {
            let orchestrator = black_box(
                OrchestratorBuilder::new()
                    .build()
                    .await
                    .expect("Failed to build orchestrator")
            );
            
            orchestrator.start().await.expect("Failed to start orchestrator");
            orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
        });
    });
}

/// Benchmark agent registration performance
fn bench_agent_registration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("agent_registration");
    
    for agent_count in [1, 10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*agent_count));
        group.bench_with_input(
            BenchmarkId::new("register_agents", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        let orchestrator = rt.block_on(async {
                            let orchestrator = OrchestratorBuilder::new()
                                .build()
                                .await
                                .expect("Failed to build orchestrator");
                            orchestrator.start().await.expect("Failed to start orchestrator");
                            orchestrator
                        });
                        
                        let agents: Vec<_> = (0..agent_count)
                            .map(|i| {
                                AgentInfo::new(
                                    AgentId::new(),
                                    match i % 3 {
                                        0 => AgentType::Risk,
                                        1 => AgentType::Neural,
                                        _ => AgentType::Quantum,
                                    },
                                    format!("Benchmark Agent {}", i),
                                    "1.0.0".to_string(),
                                )
                            })
                            .collect();
                        
                        (orchestrator, agents)
                    },
                    |(orchestrator, agents)| async move {
                        for agent in agents {
                            orchestrator
                                .register_agent(black_box(agent))
                                .await
                                .expect("Failed to register agent");
                        }
                        orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark task submission performance
fn bench_task_submission(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("task_submission");
    
    for task_count in [1, 10, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*task_count));
        group.bench_with_input(
            BenchmarkId::new("submit_tasks", task_count),
            task_count,
            |b, &task_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        let orchestrator = rt.block_on(async {
                            let orchestrator = OrchestratorBuilder::new()
                                .build()
                                .await
                                .expect("Failed to build orchestrator");
                            orchestrator.start().await.expect("Failed to start orchestrator");
                            
                            // Register some agents
                            for i in 0..5 {
                                let agent_info = AgentInfo::new(
                                    AgentId::new(),
                                    match i % 3 {
                                        0 => AgentType::Risk,
                                        1 => AgentType::Neural,
                                        _ => AgentType::Quantum,
                                    },
                                    format!("Agent {}", i),
                                    "1.0.0".to_string(),
                                );
                                orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
                            }
                            
                            orchestrator
                        });
                        
                        let tasks: Vec<_> = (0..task_count)
                            .map(|i| {
                                Task::new(
                                    format!("benchmark_task_{}", i),
                                    match i % 4 {
                                        0 => TaskPriority::Critical,
                                        1 => TaskPriority::High,
                                        2 => TaskPriority::Medium,
                                        _ => TaskPriority::Low,
                                    },
                                    format!("payload_{}", i).into_bytes(),
                                ).with_agent_type(match i % 3 {
                                    0 => AgentType::Risk,
                                    1 => AgentType::Neural,
                                    _ => AgentType::Quantum,
                                })
                            })
                            .collect();
                        
                        (orchestrator, tasks)
                    },
                    |(orchestrator, tasks)| async move {
                        for task in tasks {
                            let _ = orchestrator
                                .submit_task(black_box(task))
                                .await
                                .expect("Failed to submit task");
                        }
                        orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark message passing performance
fn bench_message_passing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("message_passing");
    
    for message_count in [10, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*message_count));
        group.bench_with_input(
            BenchmarkId::new("send_messages", message_count),
            message_count,
            |b, &message_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        rt.block_on(async {
                            let communication = Arc::new(communication::MessageRouter::new());
                            communication.start().await.expect("Failed to start communication");
                            
                            let agent1 = AgentId::new();
                            let agent2 = AgentId::new();
                            
                            communication.register_agent(agent1).await.expect("Failed to register agent1");
                            communication.register_agent(agent2).await.expect("Failed to register agent2");
                            
                            let messages: Vec<_> = (0..message_count)
                                .map(|i| {
                                    communication::Message::request(
                                        agent1,
                                        agent2,
                                        format!("benchmark_message_{}", i).into_bytes().into(),
                                    )
                                })
                                .collect();
                            
                            (communication, messages)
                        })
                    },
                    |(communication, messages)| async move {
                        for message in messages {
                            let _ = communication
                                .send_message(black_box(message))
                                .await
                                .expect("Failed to send message");
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_operations");
    
    for operation_count in [10, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*operation_count));
        group.bench_with_input(
            BenchmarkId::new("memory_ops", operation_count),
            operation_count,
            |b, &operation_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        rt.block_on(async {
                            let memory = Arc::new(memory::MemoryCoordinator::new(1000));
                            memory.start().await.expect("Failed to start memory coordinator");
                            
                            let agent_id = AgentId::new();
                            
                            (memory, agent_id, operation_count)
                        })
                    },
                    |(memory, agent_id, operation_count)| async move {
                        // Create operations
                        for i in 0..operation_count {
                            let region_id = memory
                                .create_region(
                                    format!("benchmark_region_{}", i),
                                    format!("Benchmark region {}", i),
                                    agent_id,
                                    format!("data_{}", i).into_bytes(),
                                )
                                .await
                                .expect("Failed to create memory region");
                            
                            // Read the region back
                            let _ = memory
                                .get_region(black_box(region_id), agent_id)
                                .await
                                .expect("Failed to get memory region");
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark load balancer performance
fn bench_load_balancer(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("load_balancer");
    
    for agent_count in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("select_agent", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        rt.block_on(async {
                            let communication = Arc::new(communication::MessageRouter::new());
                            let agent_registry = Arc::new(AgentRegistry::new(communication));
                            let load_balancer = Arc::new(load_balancer::AdaptiveLoadBalancer::new(agent_registry.clone()));
                            
                            load_balancer.start().await.expect("Failed to start load balancer");
                            
                            // Register agents
                            for i in 0..*agent_count {
                                let agent_info = AgentInfo::new(
                                    AgentId::new(),
                                    AgentType::Neural,
                                    format!("Load Test Agent {}", i),
                                    "1.0.0".to_string(),
                                );
                                agent_registry.register_agent(agent_info).await.expect("Failed to register agent");
                            }
                            
                            load_balancer
                        })
                    },
                    |load_balancer| async move {
                        // Perform load balancing selections
                        for _ in 0..100 {
                            let _ = load_balancer
                                .select_agent(black_box(Some(AgentType::Neural)))
                                .await
                                .expect("Failed to select agent");
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark coordination engine performance
fn bench_coordination_engine(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("coordination_session_lifecycle", |b| {
        b.to_async(&rt).iter_batched(
            || {
                rt.block_on(async {
                    let orchestrator = OrchestratorBuilder::new()
                        .build()
                        .await
                        .expect("Failed to build orchestrator");
                    orchestrator.start().await.expect("Failed to start orchestrator");
                    
                    // Register some agents
                    let mut agent_ids = Vec::new();
                    for i in 0..5 {
                        let agent_info = AgentInfo::new(
                            AgentId::new(),
                            AgentType::Risk,
                            format!("Coordination Agent {}", i),
                            "1.0.0".to_string(),
                        );
                        agent_ids.push(agent_info.id);
                        orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
                    }
                    
                    (orchestrator, agent_ids)
                })
            },
            |(orchestrator, agent_ids)| async move {
                // Create and manage coordination sessions
                for _ in 0..10 {
                    let task = Task::new(
                        "coordination_test",
                        TaskPriority::High,
                        b"coordination_payload".to_vec(),
                    ).with_agent_type(AgentType::Risk);
                    
                    let _ = orchestrator
                        .submit_task(black_box(task))
                        .await
                        .expect("Failed to submit coordination task");
                }
                
                orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
            },
            BatchSize::SmallInput,
        );
    });
}

/// Benchmark metrics collection performance
fn bench_metrics_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("metrics_collection");
    
    for metric_count in [100, 500, 1000, 2000].iter() {
        group.throughput(Throughput::Elements(*metric_count));
        group.bench_with_input(
            BenchmarkId::new("collect_metrics", metric_count),
            metric_count,
            |b, &metric_count| {
                b.to_async(&rt).iter_batched(
                    || {
                        let collector = Arc::new(metrics::InMemoryMetricsCollector::new());
                        collector
                    },
                    |collector| async move {
                        for i in 0..metric_count {
                            collector.record_metric(black_box(
                                metrics::Metric::counter(
                                    format!("benchmark_metric_{}", i),
                                    "Benchmark metric",
                                    i as u64,
                                ).with_label("type", "benchmark")
                                 .with_label("iteration", &i.to_string())
                            ));
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark full system under concurrent load
fn bench_concurrent_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_load");
    group.sample_size(10); // Reduce sample size for heavy benchmarks
    
    for concurrent_tasks in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*concurrent_tasks));
        group.bench_with_input(
            BenchmarkId::new("concurrent_operations", concurrent_tasks),
            concurrent_tasks,
            |b, &concurrent_tasks| {
                b.to_async(&rt).iter_batched(
                    || {
                        rt.block_on(async {
                            let orchestrator = OrchestratorBuilder::new()
                                .build()
                                .await
                                .expect("Failed to build orchestrator");
                            orchestrator.start().await.expect("Failed to start orchestrator");
                            
                            // Register agents for all types
                            for agent_type in [AgentType::Risk, AgentType::Neural, AgentType::Quantum] {
                                for i in 0..3 {
                                    let agent_info = AgentInfo::new(
                                        AgentId::new(),
                                        agent_type,
                                        format!("{:?} Agent {}", agent_type, i),
                                        "1.0.0".to_string(),
                                    );
                                    orchestrator.register_agent(agent_info).await.expect("Failed to register agent");
                                }
                            }
                            
                            // Wait for system to stabilize
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            
                            orchestrator
                        })
                    },
                    |orchestrator| async move {
                        // Submit tasks concurrently
                        let task_futures: Vec<_> = (0..concurrent_tasks)
                            .map(|i| {
                                let task = Task::new(
                                    format!("concurrent_task_{}", i),
                                    TaskPriority::Medium,
                                    format!("concurrent_payload_{}", i).into_bytes(),
                                ).with_agent_type(match i % 3 {
                                    0 => AgentType::Risk,
                                    1 => AgentType::Neural,
                                    _ => AgentType::Quantum,
                                });
                                
                                orchestrator.submit_task(black_box(task))
                            })
                            .collect();
                        
                        // Wait for all tasks to be submitted
                        let results = futures::future::join_all(task_futures).await;
                        
                        // Verify all succeeded
                        for result in results {
                            result.expect("Failed to submit concurrent task");
                        }
                        
                        orchestrator.shutdown().await.expect("Failed to shutdown orchestrator");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_orchestrator_startup,
    bench_agent_registration,
    bench_task_submission,
    bench_message_passing,
    bench_memory_operations,
    bench_load_balancer,
    bench_coordination_engine,
    bench_metrics_collection,
    bench_concurrent_load
);

criterion_main!(benches);