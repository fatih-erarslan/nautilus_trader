//! Benchmark: EVT streaming algorithms (SPOT/DSPOT).

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyper_risk_engine::evt::{StreamingEVT, SpotConfig, DspotConfig, GPDParams};
use hyper_risk_engine::evt::spot::SpotDetector;
use hyper_risk_engine::evt::dspot::DspotDetector;

fn bench_spot_process(c: &mut Criterion) {
    let config = SpotConfig {
        initial_batch_size: 200,
        threshold_quantile: 0.99,
        ..Default::default()
    };
    let mut spot = SpotDetector::new(config);

    // Initialize with data (calibration phase)
    for i in 0..200 {
        spot.process(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("spot_process", |b| {
        b.iter(|| spot.process(black_box(1.05)))
    });
}

fn bench_spot_calibration(c: &mut Criterion) {
    c.bench_function("spot_calibration", |b| {
        b.iter(|| {
            let config = SpotConfig {
                initial_batch_size: 200,
                threshold_quantile: 0.99,
                ..Default::default()
            };
            let mut spot = SpotDetector::new(config);

            // Calibration phase
            for i in 0..200 {
                spot.process(black_box(1.0 + (i as f64) * 0.001));
            }
            spot
        })
    });
}

fn bench_dspot_process(c: &mut Criterion) {
    let config = DspotConfig {
        spot_config: SpotConfig {
            initial_batch_size: 200,
            threshold_quantile: 0.99,
            ..Default::default()
        },
        drift_window: 500,
        ..Default::default()
    };
    let mut dspot = DspotDetector::new(config);

    // Initialize with data
    for i in 0..200 {
        dspot.process(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("dspot_process", |b| {
        b.iter(|| dspot.process(black_box(1.05)))
    });
}

fn bench_streaming_evt(c: &mut Criterion) {
    let config = SpotConfig::default();
    let mut evt = StreamingEVT::new(config);

    // Initialize
    for i in 0..500 {
        evt.process(1.0 + (i as f64) * 0.001);
    }

    c.bench_function("streaming_evt_process", |b| {
        b.iter(|| evt.process(black_box(1.05)))
    });
}

fn bench_gpd_quantile(c: &mut Criterion) {
    let gpd = GPDParams::new(0.1, 0.05, 2.0, 50);

    c.bench_function("gpd_var", |b| {
        b.iter(|| gpd.var(black_box(0.99), black_box(1000)))
    });

    c.bench_function("gpd_es", |b| {
        b.iter(|| gpd.es(black_box(0.99), black_box(1000)))
    });

    c.bench_function("gpd_survival_probability", |b| {
        b.iter(|| gpd.survival_probability(black_box(2.5)))
    });

    c.bench_function("gpd_quantile", |b| {
        b.iter(|| gpd.quantile(black_box(0.05)))
    });
}

fn bench_window_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("spot_window_size");

    for window_size in [100, 200, 500, 1000] {
        let config = SpotConfig {
            initial_batch_size: window_size,
            threshold_quantile: 0.99,
            ..Default::default()
        };
        let mut spot = SpotDetector::new(config);

        // Initialize
        for i in 0..window_size {
            spot.process(1.0 + (i as f64) * 0.001);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            &window_size,
            |b, _| {
                b.iter(|| spot.process(black_box(1.05)))
            },
        );
    }
    group.finish();
}

fn bench_tail_risk_estimation(c: &mut Criterion) {
    let config = SpotConfig {
        initial_batch_size: 500,
        threshold_quantile: 0.95,
        min_exceedances: 30,
        ..Default::default()
    };
    let mut spot = SpotDetector::new(config);

    // Initialize with varied data to get valid GPD estimation
    for i in 0..600 {
        let value = 1.0 + (i as f64) * 0.01 + (i % 7) as f64 * 0.1;
        spot.process(value);
    }

    c.bench_function("spot_tail_risk_estimate", |b| {
        b.iter(|| spot.tail_risk_estimate(black_box(0.99)))
    });
}

criterion_group!(
    benches,
    bench_spot_process,
    bench_spot_calibration,
    bench_dspot_process,
    bench_streaming_evt,
    bench_gpd_quantile,
    bench_window_sizes,
    bench_tail_risk_estimation,
);

criterion_main!(benches);
