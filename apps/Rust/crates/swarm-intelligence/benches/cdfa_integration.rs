use criterion::{black_box, criterion_group, criterion_main, Criterion};
use swarm_intelligence::cdfa::CdfaIntegration;
use swarm_intelligence::optimizer::SwarmOptimizer;
use swarm_intelligence::config::SwarmConfig;

fn cdfa_integration_benchmark(c: &mut Criterion) {
    let config = SwarmConfig::default();
    let integration = CdfaIntegration::new(config);
    let time_series = vec![0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.35, 0.4];
    
    c.bench_function("cdfa_integration", |b| {
        b.iter(|| {
            integration.analyze_with_cdfa(black_box(&time_series))
        })
    });
}

fn swarm_optimization_benchmark(c: &mut Criterion) {
    let config = SwarmConfig::default();
    let optimizer = SwarmOptimizer::new(config);
    let parameters = vec![0.5, 0.3, 0.7, 0.2, 0.8];
    
    c.bench_function("swarm_optimization", |b| {
        b.iter(|| {
            optimizer.optimize_parameters(black_box(&parameters))
        })
    });
}

criterion_group!(benches, cdfa_integration_benchmark, swarm_optimization_benchmark);
criterion_main!(benches);