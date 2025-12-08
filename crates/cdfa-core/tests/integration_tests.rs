//! Integration tests for cdfa-core
//!
//! Tests complete CDFA workflows and cross-component integration

use cdfa_core::prelude::*;
use ndarray::{array, Array1, Array2};
use approx::assert_relative_eq;
use std::time::Instant;

#[test]
fn test_complete_cdfa_pipeline() {
    // Simulate multiple signal sources with diverse predictions
    let signal_sources = array![
        [0.8, 0.6, 0.9, 0.3, 0.7, 0.5, 0.85, 0.4],  // Source 1: Technical indicators
        [0.7, 0.8, 0.6, 0.4, 0.9, 0.6, 0.75, 0.5],  // Source 2: ML model
        [0.9, 0.5, 0.8, 0.5, 0.6, 0.7, 0.80, 0.3],  // Source 3: Sentiment analysis
        [0.6, 0.7, 0.7, 0.6, 0.8, 0.8, 0.70, 0.6],  // Source 4: Volume analysis
    ];
    
    let start = Instant::now();
    
    // Step 1: Calculate pairwise diversity metrics
    let n_sources = signal_sources.nrows();
    let mut diversity_matrix = Array2::<f64>::zeros((n_sources, n_sources));
    
    for i in 0..n_sources {
        for j in i+1..n_sources {
            let source_i = signal_sources.row(i);
            let source_j = signal_sources.row(j);
            
            // Calculate multiple diversity metrics
            let kendall = kendall_tau(&source_i, &source_j).unwrap();
            let spearman = spearman_correlation(&source_i, &source_j).unwrap();
            let pearson = pearson_correlation(&source_i, &source_j).unwrap();
            
            // Combine diversity metrics (lower correlation = higher diversity)
            let combined_diversity = 1.0 - ((kendall.abs() + spearman.abs() + pearson.abs()) / 3.0);
            
            diversity_matrix[[i, j]] = combined_diversity;
            diversity_matrix[[j, i]] = combined_diversity;
        }
        diversity_matrix[[i, i]] = 0.0; // Self-diversity is 0
    }
    
    // Step 2: Calculate diversity-based weights
    let avg_diversity: Array1<f64> = diversity_matrix.mean_axis(ndarray::Axis(1)).unwrap();
    let total_diversity = avg_diversity.sum();
    let weights = if total_diversity > 0.0 {
        &avg_diversity / total_diversity
    } else {
        Array1::ones(n_sources) / n_sources as f64
    };
    
    // Step 3: Perform fusion with different methods
    let params = FusionParams {
        weights: Some(weights.clone()),
        diversity_threshold: 0.5,
        score_weight: 0.7,
    };
    
    // Test different fusion methods
    let fused_weighted = CdfaFusion::fuse(&signal_sources.view(), FusionMethod::WeightedAverage, Some(params.clone())).unwrap();
    let fused_borda = CdfaFusion::fuse(&signal_sources.view(), FusionMethod::BordaCount, None).unwrap();
    let fused_adaptive = CdfaFusion::fuse(&signal_sources.view(), FusionMethod::Adaptive, Some(params)).unwrap();
    
    let elapsed = start.elapsed();
    
    // Verify results
    assert_eq!(fused_weighted.len(), signal_sources.ncols());
    assert_eq!(fused_borda.len(), signal_sources.ncols());
    assert_eq!(fused_adaptive.len(), signal_sources.ncols());
    
    // Performance check - entire pipeline should be fast
    assert!(elapsed.as_micros() < 1000, "Pipeline too slow: {:?}", elapsed);
    
    // Verify fusion produces reasonable results
    for i in 0..fused_weighted.len() {
        assert!(fused_weighted[i] >= 0.0 && fused_weighted[i] <= 1.0);
        assert!(fused_borda[i] >= 0.0);
        assert!(fused_adaptive[i] >= 0.0 && fused_adaptive[i] <= 1.0);
    }
    
    println!("CDFA Pipeline Performance: {:?}", elapsed);
    println!("Diversity matrix:\n{:?}", diversity_matrix);
    println!("Weights: {:?}", weights);
    println!("Fused (weighted): {:?}", fused_weighted);
    println!("Fused (Borda): {:?}", fused_borda);
    println!("Fused (adaptive): {:?}", fused_adaptive);
}

#[test]
fn test_time_series_cdfa() {
    // Simulate time series data from different prediction models
    let time_steps = 100;
    let t: Array1<f64> = Array1::range(0.0, time_steps as f64, 1.0);
    
    // Create diverse signal patterns
    let signal1 = t.mapv(|x| (x * 0.1).sin() * 0.5 + 0.5);  // Sinusoidal
    let signal2 = t.mapv(|x| ((x * 0.05).sin() + (x * 0.02).cos()) * 0.3 + 0.5);  // Multi-frequency
    let signal3 = t.mapv(|x| if x as i32 % 20 < 10 { 0.7 } else { 0.3 });  // Square wave
    let signal4 = t.mapv(|x| 0.5 + (x / time_steps as f64) * 0.3);  // Linear trend
    
    // Stack signals
    let signals = ndarray::stack![ndarray::Axis(0), signal1, signal2, signal3, signal4];
    
    // Calculate DTW-based diversity
    let mut dtw_diversity = Array2::<f64>::zeros((4, 4));
    for i in 0..4 {
        for j in i+1..4 {
            let dtw_sim = dtw_similarity(&signals.row(i), &signals.row(j)).unwrap();
            dtw_diversity[[i, j]] = 1.0 - dtw_sim;  // Convert similarity to diversity
            dtw_diversity[[j, i]] = dtw_diversity[[i, j]];
        }
    }
    
    // Perform adaptive fusion based on DTW diversity
    let dtw_weights = dtw_diversity.mean_axis(ndarray::Axis(1)).unwrap();
    let normalized_weights = &dtw_weights / dtw_weights.sum();
    
    let params = FusionParams {
        weights: Some(normalized_weights),
        diversity_threshold: 0.3,
        score_weight: 0.8,
    };
    
    let fused = CdfaFusion::fuse(&signals.view(), FusionMethod::Adaptive, Some(params)).unwrap();
    
    // Verify temporal consistency
    assert_eq!(fused.len(), time_steps);
    
    // Check that fusion preserves signal bounds
    let min_val = fused.fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = fused.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(min_val >= 0.0);
    assert!(max_val <= 1.0);
    
    println!("DTW Diversity Matrix:\n{:?}", dtw_diversity);
    println!("DTW-based weights: {:?}", normalized_weights);
}

#[test]
fn test_streaming_cdfa() {
    // Simulate streaming scenario with windowed fusion
    let window_size = 20;
    let n_windows = 50;
    let n_sources = 3;
    
    let mut all_fused_results = Vec::new();
    let mut total_time = std::time::Duration::new(0, 0);
    
    for window_idx in 0..n_windows {
        let start = Instant::now();
        
        // Generate window of signals (simulating real-time data)
        let mut window_signals = Array2::<f64>::zeros((n_sources, window_size));
        for i in 0..n_sources {
            for j in 0..window_size {
                let t = (window_idx * window_size + j) as f64;
                window_signals[[i, j]] = match i {
                    0 => (t * 0.02).sin() * 0.3 + 0.5 + (t * 0.001).cos() * 0.1,
                    1 => 0.5 + (t * 0.005).sin() * 0.4,
                    2 => if (t as i32) % 30 < 15 { 0.6 } else { 0.4 },
                    _ => 0.5,
                };
            }
        }
        
        // Quick diversity assessment for adaptive fusion
        let quick_diversity = {
            let mut div = 0.0;
            for i in 0..n_sources {
                for j in i+1..n_sources {
                    let corr = pearson_correlation_fast(&window_signals.row(i), &window_signals.row(j)).unwrap_or(0.0);
                    div += 1.0 - corr.abs();
                }
            }
            div / ((n_sources * (n_sources - 1)) as f64 / 2.0)
        };
        
        // Adaptive fusion based on current diversity
        let method = if quick_diversity > 0.5 {
            FusionMethod::Average  // High diversity: simple average
        } else {
            FusionMethod::BordaCount  // Low diversity: rank-based
        };
        
        let fused = CdfaFusion::fuse(&window_signals.view(), method, None).unwrap();
        all_fused_results.extend_from_slice(fused.as_slice().unwrap());
        
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        // Each window should process very quickly
        assert!(elapsed.as_micros() < 100, "Window {} too slow: {:?}", window_idx, elapsed);
    }
    
    // Verify streaming results
    assert_eq!(all_fused_results.len(), n_windows * window_size);
    
    let avg_window_time = total_time / n_windows as u32;
    println!("Streaming CDFA - Average window processing time: {:?}", avg_window_time);
    println!("Total processed: {} samples in {:?}", all_fused_results.len(), total_time);
}

#[test]
fn test_high_dimensional_cdfa() {
    // Test with many sources and high-dimensional signals
    let n_sources = 20;
    let signal_dim = 50;
    
    // Generate random but structured signals
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let mut signals = Array2::<f64>::zeros((n_sources, signal_dim));
    for i in 0..n_sources {
        // Each source has a different bias and variance
        let bias = 0.3 + (i as f64) * 0.03;
        let variance = 0.1 + (i as f64) * 0.01;
        
        for j in 0..signal_dim {
            signals[[i, j]] = bias + rng.gen::<f64>() * variance;
        }
    }
    
    let start = Instant::now();
    
    // Calculate correlation matrix efficiently
    let corr_matrix = pearson_correlation_matrix(&signals.view()).unwrap();
    
    // Extract diversity from correlation matrix
    let mut diversity_scores = Array1::<f64>::zeros(n_sources);
    for i in 0..n_sources {
        let mut div_sum = 0.0;
        for j in 0..n_sources {
            if i != j {
                div_sum += 1.0 - corr_matrix[[i, j]].abs();
            }
        }
        diversity_scores[i] = div_sum / (n_sources - 1) as f64;
    }
    
    // Normalize to get weights
    let weights = &diversity_scores / diversity_scores.sum();
    
    let params = FusionParams {
        weights: Some(weights),
        ..Default::default()
    };
    
    let fused = CdfaFusion::fuse(&signals.view(), FusionMethod::WeightedAverage, Some(params)).unwrap();
    
    let elapsed = start.elapsed();
    
    assert_eq!(fused.len(), signal_dim);
    assert!(elapsed.as_millis() < 10, "High-dimensional fusion too slow: {:?}", elapsed);
    
    println!("High-dimensional CDFA ({} sources, {} dims) completed in {:?}", 
             n_sources, signal_dim, elapsed);
}

#[test]
fn test_extreme_diversity_scenarios() {
    // Test 1: All sources identical (zero diversity)
    let identical_signals = array![
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ];
    
    let fused_identical = CdfaFusion::fuse(&identical_signals.view(), FusionMethod::Average, None).unwrap();
    for &val in &fused_identical {
        assert_relative_eq!(val, 0.5, epsilon = 1e-10);
    }
    
    // Test 2: Maximally diverse sources
    let diverse_signals = array![
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ];
    
    let fused_diverse = CdfaFusion::fuse(&diverse_signals.view(), FusionMethod::Average, None).unwrap();
    for &val in &fused_diverse {
        assert_relative_eq!(val, 0.5, epsilon = 1e-10);
    }
    
    // Test 3: Partially correlated sources
    let partial_signals = array![
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],  // Shifted version
        [0.5, 0.4, 0.3, 0.2, 0.1],  // Reversed
    ];
    
    let params = FusionParams {
        diversity_threshold: 0.3,
        score_weight: 0.6,
        ..Default::default()
    };
    
    let fused_adaptive = CdfaFusion::fuse(&partial_signals.view(), FusionMethod::Adaptive, Some(params)).unwrap();
    assert_eq!(fused_adaptive.len(), 5);
    
    // Adaptive fusion should produce intermediate results
    for &val in &fused_adaptive {
        assert!(val >= 0.0 && val <= 1.0);
    }
}