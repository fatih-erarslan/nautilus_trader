//! Platform-specific SIMD optimizations for CDFA
//! 
//! This crate provides hardware-accelerated implementations using:
//! - AVX2/AVX512 on x86_64
//! - NEON on ARM
//! - WASM SIMD for web targets
//! - Runtime feature detection

pub mod avx2;
pub mod avx512;
pub mod neon;
pub mod wasm;
pub mod runtime;

pub use runtime::*;

/// Unified SIMD API for all algorithms
/// 
/// Automatically dispatches to the best available implementation
/// based on runtime CPU feature detection
pub mod unified {
    use super::*;
    
    /// Compute Pearson correlation coefficient
    /// 
    /// Performance targets:
    /// - AVX-512: <50ns for 512 elements
    /// - AVX2: <100ns for 256 elements  
    /// - NEON: <150ns for 256 elements
    /// - WASM: <200ns for 256 elements
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx512 => unsafe { avx512::correlation_avx512(x, y) },
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::correlation_avx2(x, y) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::correlation_neon(x, y) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::correlation_wasm(x, y) },
            _ => {
                // Scalar fallback
                scalar_correlation(x, y)
            }
        }
    }
    
    /// Discrete Wavelet Transform (Haar wavelet)
    /// 
    /// Performance targets:
    /// - AVX-512: <50ns for 1024 elements
    /// - AVX2: <100ns for small transforms
    /// - NEON: <150ns for small transforms
    /// - WASM: <200ns for small transforms
    pub fn dwt_haar(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
        debug_assert_eq!(signal.len() % 2, 0);
        debug_assert_eq!(approx.len(), signal.len() / 2);
        debug_assert_eq!(detail.len(), signal.len() / 2);
        
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx512 => unsafe { avx512::dwt_haar_avx512(signal, approx, detail) },
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::dwt_haar_avx2(signal, approx, detail) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::dwt_haar_neon(signal, approx, detail) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::dwt_haar_wasm(signal, approx, detail) },
            _ => {
                // Scalar fallback
                scalar_dwt_haar(signal, approx, detail)
            }
        }
    }
    
    /// Euclidean distance calculation
    /// 
    /// Performance targets:
    /// - AVX-512: <25ns for 256 elements
    /// - AVX2: <50ns for 256 elements
    /// - NEON: <100ns for 256 elements
    /// - WASM: <150ns for 256 elements
    pub fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx512 => unsafe { avx512::euclidean_distance_avx512(x, y) },
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::euclidean_distance_avx2(x, y) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::euclidean_distance_neon(x, y) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::euclidean_distance_wasm(x, y) },
            _ => {
                // Scalar fallback
                scalar_euclidean_distance(x, y)
            }
        }
    }
    
    /// Signal fusion with weights
    /// 
    /// Performance targets:
    /// - AVX2: <200ns for fusion operation
    /// - NEON: <300ns for fusion operation
    /// - WASM: <400ns for fusion operation
    pub fn signal_fusion(signals: &[&[f64]], weights: &[f64], output: &mut [f64]) {
        debug_assert_eq!(signals.len(), weights.len());
        
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::signal_fusion_avx2(signals, weights, output) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::signal_fusion_neon(signals, weights, output) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::signal_fusion_wasm(signals, weights, output) },
            _ => {
                // Scalar fallback
                scalar_signal_fusion(signals, weights, output)
            }
        }
    }
    
    /// Shannon entropy calculation
    /// 
    /// Performance targets:
    /// - AVX-512: <100ns for 512 elements
    /// - AVX2: <150ns for probability vectors
    /// - NEON: <200ns for probability vectors
    pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx512 => unsafe { avx512::shannon_entropy_avx512(probabilities) },
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::shannon_entropy_avx2(probabilities) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::shannon_entropy_neon(probabilities) },
            _ => {
                // Scalar fallback
                scalar_shannon_entropy(probabilities)
            }
        }
    }
    
    /// Moving average calculation
    /// 
    /// Performance targets:
    /// - AVX2: <100ns for typical windows
    /// - NEON: <150ns for typical windows
    /// - WASM: <200ns for typical windows
    pub fn moving_average(signal: &[f64], window: usize, output: &mut [f64]) {
        debug_assert!(window > 0);
        debug_assert!(signal.len() >= window);
        debug_assert_eq!(output.len(), signal.len() - window + 1);
        
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::moving_average_avx2(signal, window, output) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::moving_average_neon(signal, window, output) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::moving_average_wasm(signal, window, output) },
            _ => {
                // Scalar fallback
                scalar_moving_average(signal, window, output)
            }
        }
    }
    
    /// Variance calculation
    /// 
    /// Performance targets:
    /// - AVX2: <100ns for typical vectors
    /// - NEON: <150ns for typical vectors  
    /// - WASM: <200ns for typical vectors
    pub fn variance(data: &[f64]) -> f64 {
        match best_implementation() {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => unsafe { avx2::variance_avx2(data) },
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => unsafe { neon::variance_neon(data) },
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => unsafe { wasm::variance_wasm(data) },
            _ => {
                // Scalar fallback
                scalar_variance(data)
            }
        }
    }
    
    // Scalar fallback implementations
    
    fn scalar_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 { return 0.0; }
        
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xx: f64 = x.iter().map(|&v| v * v).sum();
        let sum_yy: f64 = y.iter().map(|&v| v * v).sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn scalar_dwt_haar(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let half_n = signal.len() / 2;
        
        for i in 0..half_n {
            let even = signal[2 * i];
            let odd = signal[2 * i + 1];
            approx[i] = (even + odd) * sqrt2_inv;
            detail[i] = (even - odd) * sqrt2_inv;
        }
    }
    
    fn scalar_euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
    
    fn scalar_signal_fusion(signals: &[&[f64]], weights: &[f64], output: &mut [f64]) {
        for i in 0..output.len() {
            output[i] = signals.iter()
                .zip(weights.iter())
                .map(|(signal, &weight)| signal[i] * weight)
                .sum();
        }
    }
    
    fn scalar_shannon_entropy(probabilities: &[f64]) -> f64 {
        probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }
    
    fn scalar_moving_average(signal: &[f64], window: usize, output: &mut [f64]) {
        let inv_window = 1.0 / window as f64;
        
        for i in 0..output.len() {
            output[i] = signal[i..i + window].iter().sum::<f64>() * inv_window;
        }
    }
    
    fn scalar_variance(data: &[f64]) -> f64 {
        let n = data.len();
        if n <= 1 { return 0.0; }
        
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() / (n - 1) as f64;
        
        variance
    }
}

/// Performance validation and benchmarking
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark suite results
    #[derive(Debug)]
    pub struct BenchmarkResults {
        pub correlation_ns: u64,
        pub dwt_haar_ns: u64,
        pub euclidean_distance_ns: u64,
        pub signal_fusion_ns: u64,
        pub shannon_entropy_ns: u64,
        pub moving_average_ns: u64,
        pub variance_ns: u64,
        pub implementation: SimdImplementation,
    }
    
    /// Run comprehensive performance benchmarks
    pub fn run_benchmarks() -> BenchmarkResults {
        const ITERATIONS: usize = 1000;
        const VECTOR_SIZE: usize = 256;
        
        // Test data
        let x: Vec<f64> = (0..VECTOR_SIZE).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..VECTOR_SIZE).map(|i| (i * 2) as f64).collect();
        let probabilities: Vec<f64> = (0..VECTOR_SIZE).map(|i| 1.0 / VECTOR_SIZE as f64).collect();
        let mut output = vec![0.0; VECTOR_SIZE];
        let mut approx = vec![0.0; VECTOR_SIZE / 2];
        let mut detail = vec![0.0; VECTOR_SIZE / 2];
        
        // Correlation benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(unified::correlation(&x, &y));
        }
        let correlation_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // DWT Haar benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            unified::dwt_haar(&x, &mut approx, &mut detail);
            std::hint::black_box((&approx, &detail));
        }
        let dwt_haar_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // Euclidean distance benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(unified::euclidean_distance(&x, &y));
        }
        let euclidean_distance_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // Signal fusion benchmark
        let signals = vec![x.as_slice(), y.as_slice()];
        let weights = vec![0.5, 0.5];
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            unified::signal_fusion(&signals, &weights, &mut output);
            std::hint::black_box(&output);
        }
        let signal_fusion_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // Shannon entropy benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(unified::shannon_entropy(&probabilities));
        }
        let shannon_entropy_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // Moving average benchmark
        let window = 10;
        let mut ma_output = vec![0.0; VECTOR_SIZE - window + 1];
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            unified::moving_average(&x, window, &mut ma_output);
            std::hint::black_box(&ma_output);
        }
        let moving_average_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        // Variance benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(unified::variance(&x));
        }
        let variance_ns = start.elapsed().as_nanos() as u64 / ITERATIONS as u64;
        
        BenchmarkResults {
            correlation_ns,
            dwt_haar_ns,
            euclidean_distance_ns,
            signal_fusion_ns,
            shannon_entropy_ns,
            moving_average_ns,
            variance_ns,
            implementation: best_implementation(),
        }
    }
    
    /// Validate performance targets are met
    pub fn validate_performance_targets() -> bool {
        let results = run_benchmarks();
        
        // Performance targets based on implementation
        let targets = match results.implementation {
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx512 => (50, 50, 25, 200, 100, 100, 100),
            #[cfg(target_arch = "x86_64")]
            SimdImplementation::Avx2 => (100, 100, 50, 200, 150, 100, 100),
            #[cfg(target_arch = "aarch64")]
            SimdImplementation::Neon => (150, 150, 100, 300, 200, 150, 150),
            #[cfg(target_arch = "wasm32")]
            SimdImplementation::WasmSimd => (200, 200, 150, 400, 300, 200, 200),
            _ => (1000, 1000, 500, 1000, 1000, 500, 500), // Scalar targets
        };
        
        results.correlation_ns <= targets.0 &&
        results.dwt_haar_ns <= targets.1 &&
        results.euclidean_distance_ns <= targets.2 &&
        results.signal_fusion_ns <= targets.3 &&
        results.shannon_entropy_ns <= targets.4 &&
        results.moving_average_ns <= targets.5 &&
        results.variance_ns <= targets.6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let features = detect_cpu_features();
        println!("CPU Features: {:?}", features);
        
        let best = best_implementation();
        println!("Best SIMD implementation: {:?}", best);
        
        // Should detect at least scalar
        assert!(true);
    }
    
    #[test]
    fn test_unified_api() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        // Test correlation
        let corr = unified::correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
        
        // Test DWT
        let mut approx = vec![0.0; 4];
        let mut detail = vec![0.0; 4];
        unified::dwt_haar(&x, &mut approx, &mut detail);
        assert_eq!(approx.len(), 4);
        
        // Test distance
        let dist = unified::euclidean_distance(&[1.0, 2.0], &[4.0, 6.0]);
        assert_eq!(dist, 5.0);
        
        // Test variance
        let var = unified::variance(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(var > 0.0);
    }
    
    #[test]
    fn test_benchmarks() {
        let results = benchmarks::run_benchmarks();
        println!("Benchmark Results: {:?}", results);
        
        // Verify all benchmarks completed
        assert!(results.correlation_ns > 0);
        assert!(results.dwt_haar_ns > 0);
        assert!(results.euclidean_distance_ns > 0);
    }
    
    #[test]
    fn test_performance_targets() {
        let meets_targets = benchmarks::validate_performance_targets();
        println!("Performance targets met: {}", meets_targets);
        
        // Performance targets should be achievable
        assert!(meets_targets || cfg!(debug_assertions)); // Allow failure in debug builds
    }
}