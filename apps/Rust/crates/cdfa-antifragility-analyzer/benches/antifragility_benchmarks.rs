use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cdfa_antifragility_analyzer::{AntifragilityAnalyzer, AntifragilityParameters};

fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    
    let mut price = 100.0;
    for i in 0..n {
        let return_rate = 0.001 * ((i as f64) * 0.1).sin();
        price *= 1.0 + return_rate;
        prices.push(price);
        volumes.push(1000.0 + 100.0 * ((i as f64) * 0.05).cos());
    }
    
    (prices, volumes)
}

fn benchmark_antifragility_analysis(c: &mut Criterion) {
    let analyzer = AntifragilityAnalyzer::new();
    let (prices, volumes) = generate_test_data(1000);
    
    c.bench_function("antifragility_analysis_1000", |b| {
        b.iter(|| {
            let result = analyzer.analyze_prices(black_box(&prices), black_box(&volumes));
            black_box(result)
        })
    });
}

fn benchmark_different_sizes(c: &mut Criterion) {
    let analyzer = AntifragilityAnalyzer::new();
    
    for &size in &[100, 500, 1000, 2000, 5000] {
        let (prices, volumes) = generate_test_data(size);
        
        c.bench_function(&format!("antifragility_{}", size), |b| {
            b.iter(|| {
                let result = analyzer.analyze_prices(black_box(&prices), black_box(&volumes));
                black_box(result)
            })
        });
    }
}

fn benchmark_with_simd(c: &mut Criterion) {
    let mut params = AntifragilityParameters::default();
    params.enable_simd = true;
    let analyzer = AntifragilityAnalyzer::with_params(params);
    
    let (prices, volumes) = generate_test_data(1000);
    
    c.bench_function("antifragility_simd", |b| {
        b.iter(|| {
            let result = analyzer.analyze_prices(black_box(&prices), black_box(&volumes));
            black_box(result)
        })
    });
}

fn benchmark_without_simd(c: &mut Criterion) {
    let mut params = AntifragilityParameters::default();
    params.enable_simd = false;
    let analyzer = AntifragilityAnalyzer::with_params(params);
    
    let (prices, volumes) = generate_test_data(1000);
    
    c.bench_function("antifragility_no_simd", |b| {
        b.iter(|| {
            let result = analyzer.analyze_prices(black_box(&prices), black_box(&volumes));
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    benchmark_antifragility_analysis,
    benchmark_different_sizes,
    benchmark_with_simd,
    benchmark_without_simd
);
criterion_main!(benches);