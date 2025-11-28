//! Benchmark: EVT streaming algorithms (SPOT/DSPOT).

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyper_risk_engine::evt::{StreamingEVT, EVTConfig, SPOT, DSPOT, GPDParameters};

fn bench_spot_update(c: &mut Criterion) {
    let mut spot = SPOT::new(200, 0.01);

    // Initialize with data
    for i in 0..200 {
        spot.fit(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("spot_update", |b| {
        b.iter(|| spot.fit(black_box(1.05)))
    });
}

fn bench_spot_check(c: &mut Criterion) {
    let mut spot = SPOT::new(200, 0.01);

    // Initialize with data
    for i in 0..200 {
        spot.fit(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("spot_check", |b| {
        b.iter(|| spot.is_anomaly(black_box(1.5)))
    });
}

fn bench_dspot_update(c: &mut Criterion) {
    let mut dspot = DSPOT::new(200, 0.01, 1000);

    // Initialize with data
    for i in 0..200 {
        dspot.fit(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("dspot_update", |b| {
        b.iter(|| dspot.fit(black_box(1.05)))
    });
}

fn bench_streaming_evt(c: &mut Criterion) {
    let config = EVTConfig::default();
    let mut evt = StreamingEVT::new(config);

    // Initialize
    for i in 0..500 {
        evt.update(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("streaming_evt_update", |b| {
        b.iter(|| evt.update(black_box(1.05)))
    });
}

fn bench_gpd_quantile(c: &mut Criterion) {
    let gpd = GPDParameters {
        xi: 0.1,
        sigma: 0.05,
        threshold: 2.0,
        n_exceedances: 50,
        n_total: 1000,
    };

    c.bench_function("gpd_var", |b| {
        b.iter(|| gpd.var(black_box(0.99)))
    });

    c.bench_function("gpd_es", |b| {
        b.iter(|| gpd.expected_shortfall(black_box(0.99)))
    });
}

fn bench_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("spot_window_size");

    for window_size in [100, 200, 500, 1000] {
        let mut spot = SPOT::new(window_size, 0.01);

        // Initialize
        for i in 0..window_size {
            spot.fit(1.0 + (i as f64) * 0.001);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            &spot,
            |b, spot| {
                let mut spot = spot.clone();
                b.iter(|| spot.fit(black_box(1.05)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_spot_update,
    bench_spot_check,
    bench_dspot_update,
    bench_streaming_evt,
    bench_gpd_quantile,
    bench_window_sizes,
);

criterion_main!(benches);
