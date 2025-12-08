use criterion::{
    black_box, criterion_group, criterion_main, 
    BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale
};
use cdfa_unified::{
    core::{
        diversity::{PearsonDiversityMeasure, KendallDiversityMeasure},
        fusion::{ScoreFusion, RankFusion}
    },
    algorithms::{
        volatility::VolatilityEstimator,
        entropy::EntropyCalculator,
        statistics::StatisticsCalculator,
        p2_quantile::P2Quantile,
        alignment::AlignmentCalculator,
        calibration::CalibrationCalculator,
        wavelet::WaveletTransform
    },
    types::{CdfaArray, CdfaMatrix, CdfaFloat},
    CDFABuilder, CDFAUnified,
    error::CdfaResult
};
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};

// Performance targets for validation
const CORE_DIVERSITY_TARGET_MICROS: u64 = 10;
const SIGNAL_FUSION_TARGET_MICROS: u64 = 20;
const PATTERN_DETECTION_TARGET_MICROS: u64 = 50;
const FULL_WORKFLOW_TARGET_MICROS: u64 = 100;

// Test data sizes
const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 1000;
const LARGE_SIZE: usize = 10000;
const XLARGE_SIZE: usize = 100000;

fn generate_test_data(size: usize) -> (CdfaArray, CdfaMatrix) {
    let array = Array1::linspace(0.0, 1.0, size);
    let matrix = Array2::from_shape_fn((size / 10, 10), |(i, j)| {
        (i as CdfaFloat * 0.1) + (j as CdfaFloat * 0.01)
    });
    (array, matrix)
}

fn generate_correlation_matrix(size: usize) -> CdfaMatrix {
    let mut matrix = Array2::eye(size);
    for i in 0..size {
        for j in (i + 1)..size {
            let corr = 0.5 * ((i + j) as CdfaFloat / size as CdfaFloat);
            matrix[[i, j]] = corr;
            matrix[[j, i]] = corr;
        }
    }
    matrix
}

fn bench_pearson_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("diversity/pearson");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let correlation_matrix = generate_correlation_matrix(*size);
        let measure = PearsonDiversityMeasure::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_diversity", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = measure.calculate_diversity(black_box(&correlation_matrix));
                    let duration = start.elapsed();
                    
                    // Validate performance target
                    if *size == SMALL_SIZE {
                        assert!(
                            duration.as_micros() <= CORE_DIVERSITY_TARGET_MICROS as u128,
                            "Pearson diversity calculation took {}μs, target: {}μs",
                            duration.as_micros(),
                            CORE_DIVERSITY_TARGET_MICROS
                        );
                    }
                    
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_kendall_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("diversity/kendall");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let correlation_matrix = generate_correlation_matrix(*size);
        let measure = KendallDiversityMeasure::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_diversity", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = measure.calculate_diversity(black_box(&correlation_matrix));
                    let duration = start.elapsed();
                    
                    // Validate performance target
                    if *size == SMALL_SIZE {
                        assert!(
                            duration.as_micros() <= CORE_DIVERSITY_TARGET_MICROS as u128,
                            "Kendall diversity calculation took {}μs, target: {}μs",
                            duration.as_micros(),
                            CORE_DIVERSITY_TARGET_MICROS
                        );
                    }
                    
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_score_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/score");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (scores, _) = generate_test_data(*size);
        let weights = Array1::from_elem(*size, 1.0 / *size as CdfaFloat);
        let fusion = ScoreFusion::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("fuse_scores", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = fusion.fuse_scores(
                        black_box(&scores),
                        black_box(&weights)
                    );
                    let duration = start.elapsed();
                    
                    // Validate performance target
                    if *size == SMALL_SIZE {
                        assert!(
                            duration.as_micros() <= SIGNAL_FUSION_TARGET_MICROS as u128,
                            "Score fusion took {}μs, target: {}μs",
                            duration.as_micros(),
                            SIGNAL_FUSION_TARGET_MICROS
                        );
                    }
                    
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_rank_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/rank");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let ranks: Vec<usize> = (0..*size).collect();
        let weights = Array1::from_elem(*size, 1.0 / *size as CdfaFloat);
        let fusion = RankFusion::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("fuse_ranks", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    let result = fusion.fuse_ranks(
                        black_box(&ranks),
                        black_box(&weights)
                    );
                    let duration = start.elapsed();
                    
                    // Validate performance target
                    if *size == SMALL_SIZE {
                        assert!(
                            duration.as_micros() <= SIGNAL_FUSION_TARGET_MICROS as u128,
                            "Rank fusion took {}μs, target: {}μs",
                            duration.as_micros(),
                            SIGNAL_FUSION_TARGET_MICROS
                        );
                    }
                    
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_volatility_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/volatility");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (data, _) = generate_test_data(*size);
        let estimator = VolatilityEstimator::new(20); // 20-period window
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_volatility", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(estimator.calculate_volatility(black_box(&data)))
                })
            },
        );
    }
    group.finish();
}

fn bench_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/entropy");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (data, _) = generate_test_data(*size);
        let calculator = EntropyCalculator::new(10); // 10 bins
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_entropy", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(calculator.calculate_entropy(black_box(&data)))
                })
            },
        );
    }
    group.finish();
}

fn bench_statistics_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/statistics");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (data, _) = generate_test_data(*size);
        let calculator = StatisticsCalculator::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("calculate_all_statistics", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(calculator.calculate_all_statistics(black_box(&data)))
                })
            },
        );
    }
    group.finish();
}

fn bench_p2_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/p2_quantile");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (data, _) = generate_test_data(*size);
        let mut quantile = P2Quantile::new(0.5).unwrap(); // Median
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("update_quantile", size),
            size,
            |b, _| {
                b.iter(|| {
                    for &value in data.iter() {
                        quantile.update(black_box(value));
                    }
                    black_box(quantile.quantile())
                })
            },
        );
    }
    group.finish();
}

fn bench_full_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("workflow/complete");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE].iter() {
        let (data, matrix) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("complete_cdfa_workflow", size),
            size,
            |b, _| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    // Build CDFA instance
                    let cdfa = CDFABuilder::new()
                        .with_diversity_measure("pearson")
                        .unwrap()
                        .with_fusion_method("score")
                        .unwrap()
                        .with_simd_enabled(true)
                        .with_parallel_enabled(true)
                        .build()
                        .unwrap();
                    
                    // Execute complete workflow
                    let _diversity = cdfa.calculate_diversity(black_box(&matrix));
                    let _volatility = cdfa.calculate_volatility(black_box(&data));
                    let _entropy = cdfa.calculate_entropy(black_box(&data));
                    let _statistics = cdfa.calculate_statistics(black_box(&data));
                    
                    let duration = start.elapsed();
                    
                    // Validate performance target
                    if *size == SMALL_SIZE {
                        assert!(
                            duration.as_micros() <= FULL_WORKFLOW_TARGET_MICROS as u128,
                            "Full CDFA workflow took {}μs, target: {}μs",
                            duration.as_micros(),
                            FULL_WORKFLOW_TARGET_MICROS
                        );
                    }
                    
                    black_box(())
                })
            },
        );
    }
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation");
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        group.bench_with_input(
            BenchmarkId::new("matrix_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let matrix = Array2::<CdfaFloat>::zeros((size, size));
                    black_box(matrix)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("array_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let array = Array1::<CdfaFloat>::zeros(size);
                    black_box(array)
                })
            },
        );
    }
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE].iter() {
        let (data1, _) = generate_test_data(*size);
        let (data2, _) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectorized_addition", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = &data1 + &data2;
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("vectorized_multiplication", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = &data1 * &data2;
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = data1.dot(&data2);
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for size in [MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE].iter() {
        let (_, matrix) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_matrix_operations", size),
            size,
            |b, _| {
                b.iter(|| {
                    use rayon::prelude::*;
                    let result: CdfaFloat = matrix
                        .axis_iter(ndarray::Axis(0))
                        .into_par_iter()
                        .map(|row| row.sum())
                        .sum();
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

// Performance regression tests
fn bench_regression_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression/performance");
    
    // Test against known performance baselines
    let baseline_data = generate_test_data(1000);
    
    group.bench_function("baseline_pearson_1000", |b| {
        let correlation_matrix = generate_correlation_matrix(1000);
        let measure = PearsonDiversityMeasure::new();
        
        b.iter(|| {
            let start = Instant::now();
            let result = measure.calculate_diversity(black_box(&correlation_matrix));
            let duration = start.elapsed();
            
            // Ensure no regression from baseline (should be under 50μs for 1000x1000)
            assert!(
                duration.as_micros() <= 50,
                "Performance regression detected: {}μs > 50μs",
                duration.as_micros()
            );
            
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_pearson_diversity,
    bench_kendall_diversity,
    bench_score_fusion,
    bench_rank_fusion,
    bench_volatility_estimation,
    bench_entropy_calculation,
    bench_statistics_calculation,
    bench_p2_quantile,
    bench_full_workflow,
    bench_memory_usage,
    bench_simd_operations,
    bench_parallel_operations,
    bench_regression_tests
);

criterion_main!(benches);