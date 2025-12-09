//! Benchmark for Event-Driven SGNN

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_event_sgnn::{EventDrivenSGNNLayer, MultiScaleSGNN, Spike, encode_to_spikes, MarketEvent, EventType};

fn bench_spike_processing(c: &mut Criterion) {
    let mut layer = EventDrivenSGNNLayer::new(100, 0.1);
    let spikes = vec![
        Spike { neuron_id: 0, timestamp: 1000, intensity: 100 },
        Spike { neuron_id: 1, timestamp: 1001, intensity: 150 },
    ];

    c.bench_function("spike_processing", |b| {
        b.iter(|| {
            black_box(layer.process_spikes(&spikes))
        })
    });
}

fn bench_spike_encoding(c: &mut Criterion) {
    let event = MarketEvent {
        timestamp: 1000000,
        asset_id: 5,
        event_type: EventType::Trade,
        price: 100.0,
        volume: 1000.0,
    };

    c.bench_function("spike_encoding", |b| {
        b.iter(|| {
            black_box(encode_to_spikes(&event))
        })
    });
}

fn bench_multiscale_processing(c: &mut Criterion) {
    let mut sgnn = MultiScaleSGNN::new(64);
    let event = MarketEvent {
        timestamp: 1000000,
        asset_id: 0,
        event_type: EventType::Trade,
        price: 100.0,
        volume: 100.0,
    };

    c.bench_function("multiscale_processing", |b| {
        b.iter(|| {
            black_box(sgnn.process_event(event.clone()))
        })
    });
}

criterion_group!(benches, bench_spike_processing, bench_spike_encoding, bench_multiscale_processing);
criterion_main!(benches);
