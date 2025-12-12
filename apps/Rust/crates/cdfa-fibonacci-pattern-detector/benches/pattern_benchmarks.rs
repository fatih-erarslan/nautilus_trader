use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cdfa_fibonacci_pattern_detector::{FibonacciPatternDetector, PatternParameters};
use ndarray::Array1;
use std::time::Duration;

fn generate_market_data(size: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);
    
    let mut price = 1.0;
    for _ in 0..size {
        let change = rng.gen_range(-0.05..0.05);
        price += change;
        
        let spread = rng.gen_range(0.01..0.03);
        high.push(price + spread);
        low.push(price - spread);
        close.push(price + rng.gen_range(-spread..spread));
    }
    
    (Array1::from_vec(high), Array1::from_vec(low), Array1::from_vec(close))
}

fn benchmark_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = [50, 100, 200, 500];
    
    for &size in &sizes {
        let (high, low, close) = generate_market_data(size);
        let detector = FibonacciPatternDetector::new();
        
        group.bench_with_input(
            BenchmarkId::new("full_detection", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = detector.detect_patterns(
                        black_box(&high),
                        black_box(&low), 
                        black_box(&close)
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_swing_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("swing_detection");
    group.measurement_time(Duration::from_secs(5));
    
    let sizes = [100, 200, 500, 1000];
    
    for &size in &sizes {
        let (high, low, _) = generate_market_data(size);
        
        group.bench_with_input(
            BenchmarkId::new("swing_points", size),
            &size,
            |b, _| {
                b.iter(|| {
                    // This would test swing point detection if it was public
                    // For now, just benchmark full detection
                    let detector = FibonacciPatternDetector::new();
                    let close = Array1::from_vec(vec![1.0; size]);
                    let result = detector.detect_patterns(
                        black_box(&high),
                        black_box(&low),
                        black_box(&close)
                    );
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn benchmark_performance_targets(c: &mut Criterion) {
    use cdfa_fibonacci_pattern_detector::perf::*;
    
    let mut group = c.benchmark_group("performance_targets");
    group.measurement_time(Duration::from_secs(5));
    
    // Test with data size that should meet sub-microsecond targets
    let (high, low, close) = generate_market_data(100);
    let detector = FibonacciPatternDetector::new();
    
    group.bench_function("sub_microsecond_target", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = detector.detect_patterns(
                black_box(&high),
                black_box(&low),
                black_box(&close)
            );
            let elapsed = start.elapsed().as_nanos() as u64;
            
            // Verify we meet our performance target
            if elapsed > FULL_DETECTION_TARGET_NS {
                eprintln!("Performance target missed: {}ns > {}ns", elapsed, FULL_DETECTION_TARGET_NS);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_pattern_detection,
    benchmark_swing_detection,
    benchmark_performance_targets
);

criterion_main!(benches);