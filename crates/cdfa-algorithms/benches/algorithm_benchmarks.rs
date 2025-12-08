//! Performance benchmarks for cdfa-algorithms
//!
//! Measures performance of wavelet transforms, entropy calculations, and volatility analysis

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use cdfa_algorithms::{
    wavelet::*,
    entropy::*,
    volatility::*,
    utils,
};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

fn generate_signal(size: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let t: Array1<f64> = Array1::range(0.0, size as f64, 1.0) / size as f64;
    
    // Multi-frequency signal with noise
    t.mapv(|x| {
        (2.0 * PI * 10.0 * x).sin() +
        0.5 * (2.0 * PI * 25.0 * x).sin() +
        0.2 * (2.0 * PI * 50.0 * x).sin() +
        0.1 * (rng.gen::<f64>() - 0.5)
    })
}

fn generate_returns(size: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array1::from_shape_fn(size, |_| 0.01 * (rng.gen::<f64>() - 0.5) * 2.0)
}

fn bench_wavelet_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet_transforms");
    
    for size in [64, 256, 1024, 4096].iter() {
        let signal = generate_signal(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Haar wavelet
        if size % 2 == 0 {
            group.bench_with_input(BenchmarkId::new("dwt_haar", size), size, |b, _| {
                b.iter(|| {
                    WaveletTransform::dwt_haar(black_box(&signal.view())).unwrap()
                });
            });
        }
        
        // Daubechies-4 wavelet
        if size % 2 == 0 && *size >= 8 {
            group.bench_with_input(BenchmarkId::new("dwt_db4", size), size, |b, _| {
                b.iter(|| {
                    WaveletTransform::dwt_db4(black_box(&signal.view())).unwrap()
                });
            });
        }
        
        // Continuous wavelet transform (smaller sizes only)
        if *size <= 512 {
            let scales = Array1::range(1.0, 33.0, 4.0);
            group.bench_with_input(BenchmarkId::new("cwt_morlet", size), size, |b, _| {
                b.iter(|| {
                    WaveletTransform::cwt_morlet(black_box(&signal.view()), black_box(&scales.view())).unwrap()
                });
            });
        }
    }
    
    group.finish();
}

fn bench_wavelet_packet(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet_packet");
    
    for &(size, level) in [(64, 2), (256, 3), (1024, 4)].iter() {
        let signal = generate_signal(size, 43);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("decompose", format!("{}x{}", size, level)), 
            &(size, level), 
            |b, _| {
                b.iter(|| {
                    WaveletPacket::decompose(black_box(&signal.view()), level).unwrap()
                });
            }
        );
        
        // Benchmark reconstruction
        let packet = WaveletPacket::decompose(&signal.view(), level).unwrap();
        group.bench_with_input(
            BenchmarkId::new("reconstruct", format!("{}x{}", size, level)), 
            &(size, level), 
            |b, _| {
                b.iter(|| {
                    packet.reconstruct().unwrap()
                });
            }
        );
    }
    
    group.finish();
}

fn bench_entropy_measures(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_measures");
    
    for size in [50, 100, 500, 1000, 5000].iter() {
        let signal = generate_signal(*size, 44);
        
        // Normalize for probability distribution
        let prob_dist = {
            let positive = signal.mapv(|x| x.abs() + 1e-10);
            &positive / positive.sum()
        };
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Shannon entropy
        group.bench_with_input(BenchmarkId::new("shannon_entropy", size), size, |b, _| {
            b.iter(|| {
                ShannonEntropy::calculate(black_box(&prob_dist.view())).unwrap()
            });
        });
        
        // Sample entropy (computationally intensive)
        if *size <= 1000 {
            group.bench_with_input(BenchmarkId::new("sample_entropy", size), size, |b, _| {
                b.iter(|| {
                    SampleEntropy::calculate(black_box(&signal.view()), 2, 0.2).unwrap()
                });
            });
        }
        
        // Approximate entropy
        if *size <= 1000 {
            group.bench_with_input(BenchmarkId::new("approximate_entropy", size), size, |b, _| {
                b.iter(|| {
                    ApproximateEntropy::calculate(black_box(&signal.view()), 2, 0.2).unwrap()
                });
            });
        }
        
        // Permutation entropy
        group.bench_with_input(BenchmarkId::new("permutation_entropy", size), size, |b, _| {
            b.iter(|| {
                PermutationEntropy::calculate(black_box(&signal.view()), 3, 1).unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_volatility_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("volatility_models");
    
    for size in [100, 500, 1000, 5000].iter() {
        let returns = generate_returns(*size, 45);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // EWMA volatility
        group.bench_with_input(BenchmarkId::new("ewma_volatility", size), size, |b, _| {
            b.iter(|| {
                VolatilityClustering::ewma_volatility(black_box(&returns.view()), 0.94).unwrap()
            });
        });
        
        // GARCH volatility
        let params = GarchParams {
            omega: 0.000001,
            alpha: 0.1,
            beta: 0.85,
        };
        
        group.bench_with_input(BenchmarkId::new("garch_volatility", size), size, |b, _| {
            b.iter(|| {
                VolatilityClustering::garch_volatility(black_box(&returns.view()), &params).unwrap()
            });
        });
        
        // Realized volatility
        group.bench_with_input(BenchmarkId::new("realized_volatility", size), size, |b, _| {
            b.iter(|| {
                VolatilityClustering::realized_volatility(black_box(&returns.view())).unwrap()
            });
        });
        
        // Volatility regime detection (computationally intensive)
        if *size <= 1000 {
            group.bench_with_input(BenchmarkId::new("regime_detection", size), size, |b, _| {
                b.iter(|| {
                    VolatilityRegime::detect(black_box(&returns.view()), 50).unwrap()
                });
            });
        }
    }
    
    group.finish();
}

fn bench_signal_processing_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_utils");
    
    for size in [100, 1000, 10000].iter() {
        let signal = generate_signal(*size, 46);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Normalization
        group.bench_with_input(BenchmarkId::new("normalize", size), size, |b, _| {
            b.iter(|| {
                utils::normalize(black_box(&signal.view()))
            });
        });
        
        // Detrending
        group.bench_with_input(BenchmarkId::new("detrend", size), size, |b, _| {
            b.iter(|| {
                utils::detrend(black_box(&signal.view()))
            });
        });
        
        // Savitzky-Golay filter
        if *size >= 21 {
            group.bench_with_input(BenchmarkId::new("savgol_filter", size), size, |b, _| {
                b.iter(|| {
                    utils::savgol_filter(black_box(&signal.view()), 21, 3).unwrap()
                });
            });
        }
        
        // SNR calculation
        let noise = Array1::from_shape_fn(*size, |_| 0.1 * (rand::random::<f64>() - 0.5));
        group.bench_with_input(BenchmarkId::new("snr", size), size, |b, _| {
            b.iter(|| {
                utils::snr(black_box(&signal.view()), black_box(&noise.view())).unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_complete_analysis_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_pipeline");
    
    for size in [256, 1024, 4096].iter() {
        let signal = generate_signal(*size, 47);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("wavelet_entropy_analysis", size), size, |b, _| {
            b.iter(|| {
                // Normalize signal
                let normalized = utils::normalize(&signal.view());
                
                // Wavelet decomposition
                let (approx, detail) = WaveletTransform::dwt_db4(&normalized.view()).unwrap();
                
                // Calculate entropy of wavelet coefficients
                let detail_prob = {
                    let energy = detail.mapv(|x| x * x);
                    let total = energy.sum();
                    energy / total
                };
                let wavelet_entropy = ShannonEntropy::calculate(&detail_prob.view()).unwrap();
                
                // Return processing result
                (approx, detail, wavelet_entropy)
            });
        });
        
        // Returns analysis pipeline
        let returns = generate_returns(*size, 48);
        
        group.bench_with_input(BenchmarkId::new("volatility_entropy_analysis", size), size, |b, _| {
            b.iter(|| {
                // Calculate volatility
                let vol = VolatilityClustering::ewma_volatility(&returns.view(), 0.94).unwrap();
                
                // Calculate entropy of returns
                let entropy = if returns.len() >= 10 {
                    SampleEntropy::calculate(&returns.view(), 2, 0.2).unwrap_or(0.0)
                } else {
                    0.0
                };
                
                // Detect volatility regime
                let regime = if returns.len() >= 100 {
                    Some(VolatilityRegime::detect(&returns.view(), 50).ok())
                } else {
                    None
                };
                
                (vol, entropy, regime)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_wavelet_transforms,
    bench_wavelet_packet,
    bench_entropy_measures,
    bench_volatility_models,
    bench_signal_processing_utils,
    bench_complete_analysis_pipeline
);
criterion_main!(benches);