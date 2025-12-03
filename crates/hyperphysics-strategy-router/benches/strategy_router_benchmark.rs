//! Performance benchmarks for hyperphysics-strategy-router
//!
//! Validates sub-100μs latency targets for routing decisions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperphysics_strategy_router::{
    StrategyRouter, RouterConfig,
    GatingNetwork, GatingConfig,
    PBitNoiseGenerator, NoiseConfig,
    RegimeDetector, RegimeConfig,
    LoadBalancer, LoadBalancerConfig,
    ExpertConfig,
};
use std::time::Duration;

fn bench_gating_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("gating_network");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    for num_experts in [4, 8, 16, 32].iter() {
        let config = GatingConfig {
            input_dim: 64,
            num_experts: *num_experts,
            top_k: 2.min(*num_experts),
            noisy_gating: false,
            ..Default::default()
        };
        let gate = GatingNetwork::new(config).unwrap();

        let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("route", num_experts),
            num_experts,
            |b, _| {
                b.iter(|| {
                    gate.route(black_box(&input), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("all_probs", num_experts),
            num_experts,
            |b, _| {
                b.iter(|| {
                    gate.all_probabilities(black_box(&input)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_pbit_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("pbit_noise");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    for num_pbits in [8, 16, 32, 64].iter() {
        let config = NoiseConfig {
            num_pbits: *num_pbits,
            temperature: 1.0,
            ..Default::default()
        };
        let mut gen = PBitNoiseGenerator::new(config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("generate", num_pbits),
            num_pbits,
            |b, _| {
                b.iter(|| {
                    gen.generate()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gaussian_100", num_pbits),
            num_pbits,
            |b, _| {
                b.iter(|| {
                    gen.generate_gaussian(100)
                })
            },
        );
    }

    group.finish();
}

fn bench_regime_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("regime_detector");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let mut detector = RegimeDetector::new(RegimeConfig {
        volatility_window: 20,
        trend_window: 50,
        ..Default::default()
    });

    // Prime with some data
    for i in 0..100 {
        detector.update(0.01 * (i as f64 % 3.0 - 1.0));
    }

    group.bench_function("update", |b| {
        b.iter(|| {
            detector.update(black_box(0.01))
        })
    });

    group.bench_function("current_regime", |b| {
        b.iter(|| {
            detector.current_regime()
        })
    });

    group.bench_function("expert_bias", |b| {
        b.iter(|| {
            detector.expert_bias()
        })
    });

    group.finish();
}

fn bench_load_balancer(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_balancer");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let mut lb = LoadBalancer::new(LoadBalancerConfig {
        num_experts: 8,
        ..Default::default()
    });

    group.bench_function("update_loads", |b| {
        b.iter(|| {
            lb.update_loads(black_box(&[(0, 0.6), (3, 0.4)]))
        })
    });

    // Prime with some data
    for _ in 0..100 {
        lb.update_loads(&[(0, 0.5), (1, 0.5)]);
    }

    let probs = vec![vec![0.125; 8]; 10];

    group.bench_function("balance_loss", |b| {
        b.iter(|| {
            lb.compute_balance_loss(black_box(&probs))
        })
    });

    let logits = vec![vec![1.0; 8]; 10];

    group.bench_function("z_loss", |b| {
        b.iter(|| {
            lb.compute_z_loss(black_box(&logits))
        })
    });

    group.bench_function("load_imbalance", |b| {
        b.iter(|| {
            lb.load_imbalance()
        })
    });

    group.finish();
}

fn bench_full_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_router");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(500);

    for input_dim in [32, 64, 128].iter() {
        let config = RouterConfig {
            input_dim: *input_dim,
            num_experts: 8,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: *input_dim,
                ..Default::default()
            },
            use_pbit_noise: true,
            use_regime_detection: true,
            ..Default::default()
        };
        let mut router = StrategyRouter::new(config).unwrap();

        let input: Vec<f64> = (0..*input_dim).map(|i| (i as f64) * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("route", input_dim),
            input_dim,
            |b, _| {
                b.iter(|| {
                    router.route(black_box(&input)).unwrap()
                })
            },
        );
    }

    // Without pBit noise
    let config = RouterConfig {
        input_dim: 64,
        num_experts: 8,
        top_k: 2,
        expert_config: ExpertConfig {
            dim: 64,
            ..Default::default()
        },
        use_pbit_noise: false,
        use_regime_detection: false,
        ..Default::default()
    };
    let mut router_simple = StrategyRouter::new(config).unwrap();
    let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();

    group.bench_function("route_simple", |b| {
        b.iter(|| {
            router_simple.route(black_box(&input)).unwrap()
        })
    });

    group.finish();
}

fn bench_latency_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_validation");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(5000);

    let config = RouterConfig {
        input_dim: 64,
        num_experts: 8,
        top_k: 2,
        expert_config: ExpertConfig {
            dim: 64,
            ..Default::default()
        },
        use_pbit_noise: false,
        use_regime_detection: false,
        ..Default::default()
    };
    let mut router = StrategyRouter::new(config).unwrap();
    let input: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01).collect();

    // Target: <50μs for routing decision
    group.bench_function("routing_50us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = router.route(black_box(&input)).unwrap();
            }
            start.elapsed()
        })
    });

    let gate_config = GatingConfig {
        input_dim: 64,
        num_experts: 8,
        top_k: 2,
        ..Default::default()
    };
    let gate = GatingNetwork::new(gate_config).unwrap();

    // Target: <10μs for gating
    group.bench_function("gating_10us_target", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = gate.route(black_box(&input), None).unwrap();
            }
            start.elapsed()
        })
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(200);

    let config = RouterConfig {
        input_dim: 64,
        num_experts: 8,
        top_k: 2,
        expert_config: ExpertConfig {
            dim: 64,
            ..Default::default()
        },
        use_pbit_noise: false,
        use_regime_detection: false,
        ..Default::default()
    };
    let mut router = StrategyRouter::new(config).unwrap();

    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let inputs: Vec<Vec<f64>> = (0..*batch_size)
            .map(|j| (0..64).map(|i| ((i + j) as f64) * 0.01).collect())
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_routing", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    for input in &inputs {
                        let _ = router.route(black_box(input)).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gating_network,
    bench_pbit_noise,
    bench_regime_detector,
    bench_load_balancer,
    bench_full_router,
    bench_latency_validation,
    bench_throughput,
);

criterion_main!(benches);
