//! Parallel algorithms for CDFA computations
//!
//! Implements highly optimized parallel algorithms for diversity calculation,
//! fusion processing, and wavelet transforms using Rayon and custom techniques.

use rayon::prelude::*;
use rayon::{ThreadPoolBuilder, ThreadPool};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::Float;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cdfa_core::error::Result;
use cdfa_core::types::{AnalysisResult, DiversityMatrix, Signal};
use cdfa_algorithms::wavelet::WaveletTransform;
use cdfa_algorithms::statistics::Statistics;

use crate::lock_free::{WaitFreeCorrelationMatrix, LockFreeResultAggregator};

/// Parallel diversity calculator
///
/// Computes pairwise diversity metrics across multiple signals/results
/// using parallel algorithms optimized for cache locality.
pub struct ParallelDiversityCalculator {
    /// Thread pool for computations
    thread_pool: Arc<ThreadPool>,
    
    /// Chunk size for parallel processing
    chunk_size: usize,
    
    /// Enable SIMD optimizations
    enable_simd: bool,
}

impl ParallelDiversityCalculator {
    /// Creates a new parallel diversity calculator
    pub fn new(num_threads: Option<usize>, chunk_size: usize) -> Result<Self> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(num_cpus::get))
            .thread_name(|i| format!("cdfa-diversity-{}", i))
            .build()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
        
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            chunk_size,
            enable_simd: cfg!(feature = "simd"),
        })
    }
    
    /// Computes Kendall's tau correlation in parallel (ultra-optimized)
    pub fn kendall_tau_parallel(&self, x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        
        if n < 2 {
            return 0.0;
        }
        
        // Optimize chunk size for cache locality (64-byte cache lines)
        let optimal_chunk_size = (64 / std::mem::size_of::<f64>()).max(self.chunk_size);
        
        // Count concordant and discordant pairs in parallel with cache-optimized chunks
        let (concordant, discordant) = self.thread_pool.install(|| {
            (0..n).into_par_iter()
                .chunks(optimal_chunk_size)
                .map(|chunk| {
                    let mut local_concordant = 0i64;
                    let mut local_discordant = 0i64;
                    
                    // Manual loop unrolling for better performance
                    let chunk_slice = chunk.collect::<Vec<_>>();
                    let chunk_len = chunk_slice.len();
                    
                    for idx in 0..chunk_len {
                        let i = chunk_slice[idx];
                        
                        // Vectorizable inner loop with stride access pattern
                        let mut j = i + 1;
                        while j < n {
                            // Process multiple pairs at once for better branch prediction
                            let pairs_to_process = (n - j).min(8);
                            
                            for k in 0..pairs_to_process {
                                let j_actual = j + k;
                                if j_actual >= n { break; }
                                
                                let x_diff = x[i] - x[j_actual];
                                let y_diff = y[i] - y[j_actual];
                                let product = x_diff * y_diff;
                                
                                // Branchless counting using arithmetic
                                local_concordant += (product > 0.0) as i64;
                                local_discordant += (product < 0.0) as i64;
                            }
                            
                            j += pairs_to_process;
                        }
                    }
                    
                    (local_concordant, local_discordant)
                })
                .reduce(
                    || (0, 0),
                    |(c1, d1), (c2, d2)| (c1 + c2, d1 + d2)
                )
        });
        
        let total_pairs = (n * (n - 1)) / 2;
        2.0 * (concordant - discordant) as f64 / total_pairs as f64
    }
    
    /// Computes diversity matrix for analysis results in parallel
    pub fn compute_diversity_matrix(&self, results: &[AnalysisResult]) -> DiversityMatrix {
        let n = results.len();
        let matrix = Arc::new(WaitFreeCorrelationMatrix::new(n));
        
        // Compute upper triangle in parallel
        self.thread_pool.install(|| {
            (0..n).into_par_iter().for_each(|i| {
                for j in (i + 1)..n {
                    let diversity = self.compute_result_diversity(&results[i], &results[j]);
                    matrix.update(i, j, diversity);
                    matrix.update(j, i, diversity); // Symmetric
                }
            });
        });
        
        matrix.to_diversity_matrix()
    }
    
    /// Computes diversity between two analysis results
    fn compute_result_diversity(&self, a: &AnalysisResult, b: &AnalysisResult) -> f64 {
        // Prediction diversity
        let pred_div = (a.prediction - b.prediction).abs() / 2.0;
        
        // Confidence diversity
        let conf_div = (a.confidence - b.confidence).abs();
        
        // Feature diversity (if available)
        let feat_div = if !a.features.is_empty() && !b.features.is_empty() {
            self.compute_feature_diversity(a, b)
        } else {
            0.0
        };
        
        // Weighted combination
        0.4 * pred_div + 0.3 * conf_div + 0.3 * feat_div
    }
    
    /// Computes feature-based diversity
    fn compute_feature_diversity(&self, a: &AnalysisResult, b: &AnalysisResult) -> f64 {
        let a_features: Vec<f64> = a.features.iter().map(|f| f.value).collect();
        let b_features: Vec<f64> = b.features.iter().map(|f| f.value).collect();
        
        if a_features.len() != b_features.len() {
            return 1.0; // Maximum diversity for different feature sets
        }
        
        // Compute cosine distance
        let dot_product: f64 = a_features.iter()
            .zip(&b_features)
            .map(|(a, b)| a * b)
            .sum();
        
        let a_norm: f64 = a_features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm: f64 = b_features.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if a_norm == 0.0 || b_norm == 0.0 {
            return 1.0;
        }
        
        1.0 - (dot_product / (a_norm * b_norm)).abs()
    }
}

/// Concurrent fusion processor
///
/// Performs parallel fusion of multiple signal streams with
/// optimized memory access patterns.
pub struct ConcurrentFusionProcessor {
    /// Thread pool
    thread_pool: Arc<ThreadPool>,
    
    /// Result aggregator
    aggregator: Arc<LockFreeResultAggregator>,
    
    /// Fusion weights cache
    weights_cache: dashmap::DashMap<u64, Vec<f64>>,
}

impl ConcurrentFusionProcessor {
    /// Creates a new concurrent fusion processor
    pub fn new(num_threads: Option<usize>) -> Result<Self> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(num_cpus::get))
            .thread_name(|i| format!("cdfa-fusion-{}", i))
            .build()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
        
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            aggregator: Arc::new(LockFreeResultAggregator::new()),
            weights_cache: dashmap::DashMap::new(),
        })
    }
    
    /// Fuses multiple signals using weighted average in parallel (zero-overhead)
    pub fn fuse_signals_weighted(&self, signals: &[Signal], weights: &[f64]) -> Result<Signal> {
        assert_eq!(signals.len(), weights.len());
        
        if signals.is_empty() {
            return Err(cdfa_core::error::Error::InvalidInput("No signals to fuse".into()));
        }
        
        let signal_len = signals[0].len();
        let timestamp_ns = signals[0].timestamp_ns;
        let num_signals = signals.len();
        
        // Pre-compute normalized weights to avoid division in hot loop
        let weight_sum: f64 = weights.iter().sum();
        let inv_weight_sum = 1.0 / weight_sum;
        let normalized_weights: Vec<f64> = weights.iter()
            .map(|w| w * inv_weight_sum)
            .collect();
        
        // Ultra-optimized parallel fusion with manual vectorization hints
        let fused_values = self.thread_pool.install(|| {
            // Use SIMD-friendly chunk size (multiple of 8 for AVX)
            let chunk_size = 8;
            
            (0..signal_len).into_par_iter()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    let chunk_vec: Vec<usize> = chunk.collect();
                    let mut results = Vec::with_capacity(chunk_vec.len());
                    
                    // Process chunks with manual loop unrolling
                    for &i in &chunk_vec {
                        let mut fused_value = 0.0;
                        
                        // Manual unroll for up to 8 signals (common case)
                        match num_signals {
                            1 => {
                                fused_value = signals[0].values[i] * normalized_weights[0];
                            }
                            2 => {
                                fused_value = signals[0].values[i] * normalized_weights[0]
                                            + signals[1].values[i] * normalized_weights[1];
                            }
                            3 => {
                                fused_value = signals[0].values[i] * normalized_weights[0]
                                            + signals[1].values[i] * normalized_weights[1]
                                            + signals[2].values[i] * normalized_weights[2];
                            }
                            4 => {
                                fused_value = signals[0].values[i] * normalized_weights[0]
                                            + signals[1].values[i] * normalized_weights[1]
                                            + signals[2].values[i] * normalized_weights[2]
                                            + signals[3].values[i] * normalized_weights[3];
                            }
                            _ => {
                                // General case with FMA-friendly accumulation
                                for (signal, &weight) in signals.iter().zip(&normalized_weights) {
                                    fused_value += signal.values[i] * weight;
                                }
                            }
                        }
                        
                        results.push(fused_value);
                    }
                    
                    results
                })
                .collect()
        });
        
        Ok(Signal::new(
            cdfa_core::types::SignalId(timestamp_ns),
            timestamp_ns,
            fused_values,
        ))
    }
    
    /// Performs adaptive fusion based on signal quality
    pub fn adaptive_fusion(&self, signals: &[Signal], quality_scores: &[f64]) -> Result<Signal> {
        // Compute adaptive weights based on quality
        let weights = self.compute_adaptive_weights(quality_scores);
        self.fuse_signals_weighted(signals, &weights)
    }
    
    /// Computes adaptive weights from quality scores
    fn compute_adaptive_weights(&self, quality_scores: &[f64]) -> Vec<f64> {
        let max_score = quality_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_score = quality_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_score - min_score;
        
        if range == 0.0 {
            // Equal weights if all qualities are the same
            vec![1.0 / quality_scores.len() as f64; quality_scores.len()]
        } else {
            // Exponential weighting based on normalized quality
            let weights: Vec<f64> = quality_scores.iter()
                .map(|&q| {
                    let normalized = (q - min_score) / range;
                    (2.0 * normalized).exp()
                })
                .collect();
            
            // Normalize
            let sum: f64 = weights.iter().sum();
            weights.iter().map(|w| w / sum).collect()
        }
    }
}

/// Multi-threaded wavelet transformer
///
/// Performs wavelet transforms on multiple signals concurrently
/// with optimized memory usage.
pub struct MultiThreadedWaveletTransformer {
    /// Thread pool
    thread_pool: Arc<ThreadPool>,
    
    /// Wavelet type
    wavelet_type: String,
    
    /// Decomposition level
    level: usize,
    
    /// Transform cache
    transform_cache: dashmap::DashMap<u64, Arc<Vec<f64>>>,
}

impl MultiThreadedWaveletTransformer {
    /// Creates a new multi-threaded wavelet transformer
    pub fn new(
        num_threads: Option<usize>,
        wavelet_type: String,
        level: usize,
    ) -> Result<Self> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(num_cpus::get))
            .thread_name(|i| format!("cdfa-wavelet-{}", i))
            .build()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
        
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            wavelet_type,
            level,
            transform_cache: dashmap::DashMap::new(),
        })
    }
    
    /// Transforms multiple signals in parallel
    pub fn transform_batch(&self, signals: &[Signal]) -> Result<Vec<Vec<f64>>> {
        self.thread_pool.install(|| {
            signals.par_iter()
                .map(|signal| self.transform_single(signal))
                .collect()
        })
    }
    
    /// Transforms a single signal with caching
    fn transform_single(&self, signal: &Signal) -> Result<Vec<f64>> {
        // Check cache
        let cache_key = self.compute_cache_key(signal);
        if let Some(cached) = self.transform_cache.get(&cache_key) {
            return Ok((**cached).clone());
        }
        
        // Perform transform
        let transform = WaveletTransform::new(&self.wavelet_type)?;
        let coefficients = transform.decompose(&signal.values, self.level)?;
        
        // Cache result
        let result = Arc::new(coefficients.clone());
        self.transform_cache.insert(cache_key, result);
        
        Ok(coefficients)
    }
    
    /// Computes cache key for a signal
    fn compute_cache_key(&self, signal: &Signal) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        signal.id.hash(&mut hasher);
        signal.timestamp_ns.hash(&mut hasher);
        signal.len().hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clears the transform cache
    pub fn clear_cache(&self) {
        self.transform_cache.clear();
    }
}

/// Parallel statistics calculator
///
/// Computes various statistical measures in parallel for large datasets
pub struct ParallelStatisticsCalculator {
    /// Thread pool
    thread_pool: Arc<ThreadPool>,
    
    /// Chunk size for parallel processing
    chunk_size: usize,
}

impl ParallelStatisticsCalculator {
    /// Creates a new parallel statistics calculator
    pub fn new(num_threads: Option<usize>, chunk_size: usize) -> Result<Self> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(num_cpus::get))
            .thread_name(|i| format!("cdfa-stats-{}", i))
            .build()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
        
        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            chunk_size,
        })
    }
    
    /// Computes rolling statistics in parallel
    pub fn rolling_statistics(&self, data: &[f64], window_size: usize) -> RollingStats {
        let n = data.len();
        let num_windows = n.saturating_sub(window_size - 1);
        
        let (means, vars, skews, kurts) = self.thread_pool.install(|| {
            (0..num_windows).into_par_iter()
                .chunks(self.chunk_size)
                .map(|chunk| {
                    let mut local_means = Vec::with_capacity(chunk.len());
                    let mut local_vars = Vec::with_capacity(chunk.len());
                    let mut local_skews = Vec::with_capacity(chunk.len());
                    let mut local_kurts = Vec::with_capacity(chunk.len());
                    
                    for i in chunk {
                        let window = &data[i..i + window_size];
                        let stats = Statistics::compute_all(window);
                        
                        local_means.push(stats.mean);
                        local_vars.push(stats.variance);
                        local_skews.push(stats.skewness);
                        local_kurts.push(stats.kurtosis);
                    }
                    
                    (local_means, local_vars, local_skews, local_kurts)
                })
                .reduce(
                    || (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                    |(mut m1, mut v1, mut s1, mut k1), (m2, v2, s2, k2)| {
                        m1.extend(m2);
                        v1.extend(v2);
                        s1.extend(s2);
                        k1.extend(k2);
                        (m1, v1, s1, k1)
                    }
                )
        });
        
        RollingStats {
            means,
            variances: vars,
            skewness: skews,
            kurtosis: kurts,
            window_size,
        }
    }
    
    /// Computes correlation matrix in parallel
    pub fn correlation_matrix(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_features = data.ncols();
        let mut corr_matrix = Array2::zeros((n_features, n_features));
        
        // Compute correlations in parallel
        let correlations: Vec<((usize, usize), f64)> = self.thread_pool.install(|| {
            (0..n_features).into_par_iter()
                .flat_map(|i| {
                    (i..n_features).into_par_iter()
                        .map(move |j| {
                            let col_i = data.column(i);
                            let col_j = data.column(j);
                            let corr = self.pearson_correlation(col_i.as_slice().unwrap(), 
                                                              col_j.as_slice().unwrap());
                            ((i, j), corr)
                        })
                })
                .collect()
        });
        
        // Fill the matrix
        for ((i, j), corr) in correlations {
            corr_matrix[[i, j]] = corr;
            if i != j {
                corr_matrix[[j, i]] = corr; // Symmetric
            }
        }
        
        corr_matrix
    }
    
    /// Computes Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
        let sum_yy: f64 = y.iter().map(|yi| yi * yi).sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Rolling statistics result
#[derive(Debug, Clone)]
pub struct RollingStats {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
    pub window_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdfa_core::types::SignalId;
    
    #[test]
    fn test_parallel_diversity_calculator() {
        let calc = ParallelDiversityCalculator::new(None, 64).unwrap();
        
        // Test Kendall's tau
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 1.0, 3.0, 5.0];
        let tau = calc.kendall_tau_parallel(&x, &y);
        assert!(tau > 0.0 && tau < 1.0);
        
        // Test diversity matrix
        let results = vec![
            AnalysisResult::new("a1".to_string(), 0.7, 0.8),
            AnalysisResult::new("a2".to_string(), 0.6, 0.9),
            AnalysisResult::new("a3".to_string(), 0.8, 0.7),
        ];
        
        let matrix = calc.compute_diversity_matrix(&results);
        assert_eq!(matrix.dimension(), 3);
        assert!(matrix.get(0, 1) > 0.0);
    }
    
    #[test]
    fn test_concurrent_fusion_processor() {
        let processor = ConcurrentFusionProcessor::new(None).unwrap();
        
        let signals = vec![
            Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]),
            Signal::new(SignalId(2), 1000, vec![4.0, 5.0, 6.0]),
            Signal::new(SignalId(3), 1000, vec![7.0, 8.0, 9.0]),
        ];
        
        let weights = vec![0.5, 0.3, 0.2];
        let fused = processor.fuse_signals_weighted(&signals, &weights).unwrap();
        
        assert_eq!(fused.len(), 3);
        // Check first value: 1.0*0.5 + 4.0*0.3 + 7.0*0.2 = 3.1
        assert!((fused.values[0] - 3.1).abs() < 0.001);
    }
    
    #[test]
    fn test_parallel_statistics_calculator() {
        let calc = ParallelStatisticsCalculator::new(None, 64).unwrap();
        
        // Test rolling statistics
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = calc.rolling_statistics(&data, 3);
        
        assert_eq!(stats.means.len(), 8);
        assert_eq!(stats.window_size, 3);
        assert!((stats.means[0] - 2.0).abs() < 0.001); // Mean of [1,2,3]
        
        // Test correlation matrix
        let data_2d = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
            5.0, 10.0,
        ]).unwrap();
        
        let corr = calc.correlation_matrix(&data_2d);
        assert_eq!(corr.dim(), (2, 2));
        assert!((corr[[0, 1]] - 1.0).abs() < 0.001); // Perfect correlation
    }
}