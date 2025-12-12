use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_early_warning_system::immune::ImmuneSystem;
use quantum_early_warning_system::response::ResponseGenerator;
use quantum_early_warning_system::config::QuantumEarlyWarningConfig;

fn immune_response_benchmark(c: &mut Criterion) {
    let config = QuantumEarlyWarningConfig::default();
    let immune_system = ImmuneSystem::new(config);
    let threat_signature = vec![0.8, 0.3, 0.9, 0.1, 0.7];
    
    c.bench_function("immune_response", |b| {
        b.iter(|| {
            immune_system.generate_response(black_box(&threat_signature))
        })
    });
}

fn response_optimization_benchmark(c: &mut Criterion) {
    let generator = ResponseGenerator::new();
    let threat_level = 0.75;
    let context = vec![0.2, 0.5, 0.8, 0.3];
    
    c.bench_function("response_optimization", |b| {
        b.iter(|| {
            generator.optimize_response(black_box(threat_level), black_box(&context))
        })
    });
}

criterion_group!(benches, immune_response_benchmark, response_optimization_benchmark);
criterion_main!(benches);