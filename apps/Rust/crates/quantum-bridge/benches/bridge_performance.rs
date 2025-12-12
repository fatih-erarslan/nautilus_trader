use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_bridge::bridge::QuantumBridge;
use quantum_bridge::connector::QuantumConnector;
use quantum_bridge::config::QuantumBridgeConfig;

fn bridge_performance_benchmark(c: &mut Criterion) {
    let config = QuantumBridgeConfig::default();
    let bridge = QuantumBridge::new(config);
    let quantum_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    
    c.bench_function("bridge_performance", |b| {
        b.iter(|| {
            bridge.process_quantum_data(black_box(&quantum_data))
        })
    });
}

fn quantum_connector_benchmark(c: &mut Criterion) {
    let connector = QuantumConnector::new();
    let connection_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    
    c.bench_function("quantum_connector", |b| {
        b.iter(|| {
            connector.establish_connection(black_box(&connection_data))
        })
    });
}

criterion_group!(benches, bridge_performance_benchmark, quantum_connector_benchmark);
criterion_main!(benches);