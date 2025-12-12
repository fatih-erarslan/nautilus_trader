use cdfa_panarchy_analyzer::{PanarchyAnalyzer, MarketData, calculate_pcr, PCRComponents};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;

fn generate_market_data(size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(size);
    let mut volumes = Vec::with_capacity(size);
    
    let mut price = 100.0;
    for _ in 0..size {
        // Random walk for prices
        price *= 1.0 + rng.gen_range(-0.02..0.02);
        prices.push(price);
        
        // Random volumes
        volumes.push(rng.gen_range(900.0..1100.0));
    }
    
    (prices, volumes)
}

fn bench_full_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_analysis");
    
    for size in [50, 100, 200, 500, 1000].iter() {
        let (prices, volumes) = generate_market_data(*size);
        let mut analyzer = PanarchyAnalyzer::new();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    analyzer.analyze(
                        black_box(&prices),
                        black_box(&volumes),
                    )
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pcr_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcr_calculation");
    
    let (prices, _) = generate_market_data(1000);
    
    for period in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(period),
            period,
            |b, &period| {
                b.iter(|| {
                    calculate_pcr(
                        black_box(&prices),
                        black_box(period),
                    )
                });
            },
        );
    }
    
    group.finish();
}

fn bench_phase_identification(c: &mut Criterion) {
    let mut analyzer = PanarchyAnalyzer::new();
    
    // Generate PCR components
    let pcr_components: Vec<PCRComponents> = (0..1000)
        .map(|i| {
            let phase = i as f64 / 1000.0;
            PCRComponents::new(
                phase.sin().abs(),
                phase.cos().abs(),
                (phase * 2.0).sin().abs(),
            )
        })
        .collect();
    
    c.bench_function("phase_identification", |b| {
        b.iter(|| {
            analyzer.identify_regime(black_box(&pcr_components))
        });
    });
}

fn bench_simd_operations(c: &mut Criterion) {
    use cdfa_panarchy_analyzer::simd::{simd_mean, simd_std_dev, simd_autocorrelation};
    
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
    
    c.bench_function("simd_mean", |b| {
        b.iter(|| simd_mean(black_box(&data)));
    });
    
    c.bench_function("simd_std_dev", |b| {
        let mean = simd_mean(&data);
        b.iter(|| simd_std_dev(black_box(&data), black_box(mean)));
    });
    
    c.bench_function("simd_autocorrelation", |b| {
        b.iter(|| simd_autocorrelation(black_box(&data), black_box(1)));
    });
}

fn bench_regime_score(c: &mut Criterion) {
    use cdfa_panarchy_analyzer::{MarketPhase, PhaseScores};
    use cdfa_panarchy_analyzer::phase::calculate_regime_score;
    
    let phase_scores = PhaseScores {
        growth: 0.3,
        conservation: 0.4,
        release: 0.2,
        reorganization: 0.1,
    };
    
    c.bench_function("regime_score_calculation", |b| {
        b.iter(|| {
            calculate_regime_score(
                black_box(MarketPhase::Conservation),
                black_box(&phase_scores),
                black_box("normal"),
                black_box(0.5),
                black_box(0.5),
                black_box(25.0),
            )
        });
    });
}

fn bench_batch_analysis(c: &mut Criterion) {
    use cdfa_panarchy_analyzer::BatchPanarchyAnalyzer;
    
    let mut group = c.benchmark_group("batch_analysis");
    
    for batch_size in [10, 50, 100].iter() {
        let mut batch_analyzer = BatchPanarchyAnalyzer::new(*batch_size);
        
        let price_series: Vec<Vec<f64>> = (0..*batch_size)
            .map(|_| generate_market_data(100).0)
            .collect();
        
        let volume_series: Vec<Vec<f64>> = price_series.iter()
            .map(|prices| vec![1000.0; prices.len()])
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    batch_analyzer.analyze_batch(
                        black_box(&price_series),
                        black_box(&volume_series),
                    )
                });
            },
        );
    }
    
    group.finish();
}

// Sub-microsecond benchmarks for critical operations
fn bench_microsecond_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("microsecond_operations");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(1000);
    
    // Single PCR calculation
    let prices = vec![100.0; 20];
    let returns = vec![0.01; 20];
    let volatilities = vec![0.2; 20];
    
    group.bench_function("single_pcr_update", |b| {
        use cdfa_panarchy_analyzer::pcr::FastPCRCalculator;
        let mut calc = FastPCRCalculator::new(10, 1);
        
        b.iter(|| {
            calc.update(
                black_box(101.0),
                black_box(0.01),
                black_box(0.2),
            )
        });
    });
    
    // Phase score calculation
    let pcr = PCRComponents::new(0.7, 0.3, 0.6);
    
    group.bench_function("phase_score_calculation", |b| {
        b.iter(|| {
            let mut scores = pcr.phase_scores();
            scores.normalize();
            black_box(scores)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_full_analysis,
    bench_pcr_calculation,
    bench_phase_identification,
    bench_simd_operations,
    bench_regime_score,
    bench_batch_analysis,
    bench_microsecond_operations,
);

criterion_main!(benches);