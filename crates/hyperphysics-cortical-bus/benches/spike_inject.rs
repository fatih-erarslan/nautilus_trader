//! Spike injection latency benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyperphysics_cortical_bus::{BackendConfig, create_backend, Spike, CorticalBus};

fn bench_spike_inject(c: &mut Criterion) {
    let config = BackendConfig::default();
    let bus = create_backend(&config).unwrap();

    c.bench_function("spike_inject_single", |b| {
        b.iter(|| {
            let spike = Spike::new(black_box(12345), black_box(100), 50, 0xAB);
            bus.inject_spike(spike).unwrap();
        })
    });
}

fn bench_spike_batch(c: &mut Criterion) {
    let config = BackendConfig::default();
    let bus = create_backend(&config).unwrap();

    let mut group = c.benchmark_group("spike_inject_batch");

    for size in [64, 256, 1024, 4096].iter() {
        let spikes: Vec<Spike> = (0..*size)
            .map(|i| Spike::new(i, i as u16, 50, (i % 256) as u8))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                bus.inject_batch(black_box(&spikes)).unwrap();
                // Drain to prevent queue full
                let mut buffer = vec![Spike::default(); *size as usize];
                bus.poll_spikes(&mut buffer).unwrap();
            })
        });
    }

    group.finish();
}

fn bench_spike_poll(c: &mut Criterion) {
    let config = BackendConfig::default();
    let bus = create_backend(&config).unwrap();

    // Pre-fill queue
    for i in 0..1000 {
        let spike = Spike::new(i, i as u16, 50, (i % 256) as u8);
        let _ = bus.inject_spike(spike);
    }

    c.bench_function("spike_poll_batch", |b| {
        let mut buffer = [Spike::default(); 100];
        b.iter(|| {
            let count = bus.poll_spikes(black_box(&mut buffer)).unwrap();
            black_box(count);
        })
    });
}

criterion_group!(benches, bench_spike_inject, bench_spike_batch, bench_spike_poll);
criterion_main!(benches);
