use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pads_assembly::performance::PadsPerformanceAnalyzer;
use pads_assembly::metrics::PadsMetrics;
use pads_assembly::config::PadsConfig;

fn pads_performance_benchmark(c: &mut Criterion) {
    let config = PadsConfig::default();
    let analyzer = PadsPerformanceAnalyzer::new(config);
    let performance_data = vec![
        (1.0, 0.5),
        (2.0, 0.7),
        (3.0, 0.6),
        (4.0, 0.8),
        (5.0, 0.9),
    ];
    
    c.bench_function("pads_performance", |b| {
        b.iter(|| {
            analyzer.analyze_performance(black_box(&performance_data))
        })
    });
}

fn pads_metrics_benchmark(c: &mut Criterion) {
    let metrics = PadsMetrics::new();
    let system_stats = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    c.bench_function("pads_metrics", |b| {
        b.iter(|| {
            metrics.calculate_metrics(black_box(&system_stats))
        })
    });
}

criterion_group!(benches, pads_performance_benchmark, pads_metrics_benchmark);
criterion_main!(benches);