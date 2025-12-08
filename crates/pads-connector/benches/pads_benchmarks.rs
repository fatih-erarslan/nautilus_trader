//! PADS connector benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pads_connector::*;
use tokio::runtime::Runtime;

fn create_test_decision() -> PanarchyDecision {
    PanarchyDecision {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        context: DecisionContext {
            market_state: MarketContext {
                volatility: 0.5,
                trend_strength: 0.3,
                liquidity: 0.8,
                regime: "normal".to_string(),
            },
            system_state: SystemContext {
                resource_utilization: 0.6,
                active_scales: vec![ScaleLevel::Micro],
                current_phase: AdaptiveCyclePhase::Growth,
            },
            historical_performance: PerformanceContext {
                recent_success_rate: 0.75,
                adaptive_capacity_used: 0.4,
                resilience_score: 0.8,
            },
        },
        objectives: vec![
            Objective {
                name: "maximize_return".to_string(),
                weight: 0.6,
                target_value: 0.02,
                optimization_direction: OptimizationDirection::Maximize,
            },
            Objective {
                name: "minimize_risk".to_string(),
                weight: 0.4,
                target_value: 0.1,
                optimization_direction: OptimizationDirection::Minimize,
            },
        ],
        constraints: vec![
            Constraint {
                name: "max_drawdown".to_string(),
                constraint_type: ConstraintType::LessThan,
                value: 0.15,
            },
        ],
        urgency: 0.5,
        impact: 0.5,
        uncertainty: 0.3,
    }
}

fn benchmark_decision_processing(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let pads = runtime.block_on(async {
        let connector = PadsConnector::new(config).await.unwrap();
        connector.initialize().await.unwrap();
        connector
    });
    
    c.bench_function("pads_process_decision", |b| {
        b.to_async(&runtime).iter(|| async {
            let decision = create_test_decision();
            black_box(pads.process_decision(decision).await.unwrap())
        });
    });
}

fn benchmark_scale_determination(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let monitor = runtime.block_on(PadsMonitor::new(config.clone().into())).unwrap();
    let scale_manager = runtime.block_on(
        ScaleManager::new(config.into(), monitor.into())
    ).unwrap();
    
    let mut group = c.benchmark_group("scale_determination");
    
    for urgency in [0.1, 0.5, 0.9].iter() {
        group.bench_with_input(
            BenchmarkId::new("urgency", urgency),
            urgency,
            |b, &urgency| {
                b.to_async(&runtime).iter(|| async {
                    let mut decision = create_test_decision();
                    decision.urgency = urgency;
                    black_box(scale_manager.determine_scale(&decision).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_decision_routing(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let monitor = runtime.block_on(PadsMonitor::new(config.clone().into())).unwrap();
    let router = runtime.block_on(
        DecisionRouter::new(config.into(), monitor.into())
    ).unwrap();
    
    runtime.block_on(router.setup_routes()).unwrap();
    
    let mut group = c.benchmark_group("decision_routing");
    
    for scale in [ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro] {
        group.bench_with_input(
            BenchmarkId::new("scale", format!("{:?}", scale)),
            &scale,
            |b, &scale| {
                b.to_async(&runtime).iter(|| async {
                    let decision = create_test_decision();
                    let scale_info = PanarchyScale {
                        level: scale,
                        time_horizon: std::time::Duration::from_millis(100),
                        spatial_extent: 0.5,
                        connectivity: 0.5,
                        resilience: 0.5,
                        potential: 0.5,
                    };
                    black_box(router.route_decision(decision, scale_info).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_cross_scale_communication(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let monitor = runtime.block_on(PadsMonitor::new(config.clone().into())).unwrap();
    let communicator = runtime.block_on(
        CrossScaleCommunicator::new(config.into(), monitor.into())
    ).unwrap();
    
    runtime.block_on(communicator.setup_channels()).unwrap();
    
    c.bench_function("cross_scale_broadcast", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(
                communicator.broadcast_scale_transition(ScaleLevel::Micro).await.unwrap()
            )
        });
    });
}

fn benchmark_resilience_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let monitor = runtime.block_on(PadsMonitor::new(config.clone().into())).unwrap();
    let resilience = runtime.block_on(
        ResilienceEngine::new(config.into(), monitor.into())
    ).unwrap();
    
    runtime.block_on(resilience.configure()).unwrap();
    
    c.bench_function("resilience_capacity_check", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(resilience.get_capacity().await.unwrap())
        });
    });
    
    c.bench_function("resilience_status", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(resilience.get_status().await.unwrap())
        });
    });
}

fn benchmark_monitoring(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let monitor = runtime.block_on(PadsMonitor::new(config.into())).unwrap();
    
    runtime.block_on(monitor.start()).unwrap();
    
    c.bench_function("record_decision_metrics", |b| {
        b.iter(|| {
            let decision = create_test_decision();
            monitor.record_decision_start(&decision);
            
            let result = DecisionResult {
                decision_id: decision.id,
                timestamp: chrono::Utc::now(),
                scale_level: ScaleLevel::Micro,
                success: true,
                actions: vec![],
                metrics: DecisionMetrics {
                    processing_time_ms: 10,
                    confidence_score: 0.85,
                    resource_usage: 0.3,
                    adaptation_rate: 0.1,
                },
                cross_scale_effects: CrossScaleEffects {
                    upward_effects: vec![],
                    downward_effects: vec![],
                    lateral_effects: vec![],
                },
                errors: vec![],
            };
            
            monitor.record_decision_complete(&result);
        });
    });
}

fn benchmark_concurrent_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let config = PadsConfig::default();
    let pads = runtime.block_on(async {
        let connector = PadsConnector::new(config).await.unwrap();
        connector.initialize().await.unwrap();
        std::sync::Arc::new(connector)
    });
    
    let mut group = c.benchmark_group("concurrent_operations");
    
    for num_tasks in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_decisions", num_tasks),
            &num_tasks,
            |b, &num_tasks| {
                b.to_async(&runtime).iter(|| async {
                    let futures: Vec<_> = (0..num_tasks)
                        .map(|_| {
                            let pads = pads.clone();
                            let decision = create_test_decision();
                            tokio::spawn(async move {
                                pads.process_decision(decision).await.unwrap()
                            })
                        })
                        .collect();
                    
                    let results = futures::future::join_all(futures).await;
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_decision_processing,
    benchmark_scale_determination,
    benchmark_decision_routing,
    benchmark_cross_scale_communication,
    benchmark_resilience_operations,
    benchmark_monitoring,
    benchmark_concurrent_operations
);
criterion_main!(benches);