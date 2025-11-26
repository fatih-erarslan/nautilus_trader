//! Integration tests for parallel processing module

use neuro_divergent::optimizations::parallel::*;
use neuro_divergent::optimizations::parallel::benchmark::*;
use ndarray::{Array1, Array2, arr2, Array};

/// Generate test data
fn generate_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    Array::from_shape_fn((n_samples, n_features), |(i, j)| {
        (i as f64 + j as f64) * 0.1
    })
}

#[test]
fn test_parallel_config_default() {
    let config = ParallelConfig::default();
    assert!(config.effective_thread_count() > 0);
    assert!(config.work_stealing);
}

#[test]
fn test_parallel_config_custom_threads() {
    let config = ParallelConfig::default().with_threads(4);
    assert_eq!(config.num_threads, Some(4));
    assert_eq!(config.effective_thread_count(), 4);
}

#[test]
fn test_parallel_batch_inference_basic() {
    let batches = vec![
        arr2(&[[1.0, 2.0], [3.0, 4.0]]),
        arr2(&[[5.0, 6.0], [7.0, 8.0]]),
        arr2(&[[9.0, 10.0], [11.0, 12.0]]),
    ];

    let results = parallel_batch_inference(&batches, |batch| {
        Ok(vec![batch.sum(); batch.nrows()])
    }).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].len(), 2);
    assert_eq!(results[1].len(), 2);
    assert_eq!(results[2].len(), 2);
}

#[test]
fn test_parallel_batch_inference_correctness() {
    let batches: Vec<Array2<f64>> = (0..10)
        .map(|i| generate_test_data(5, 3) * (i as f64 + 1.0))
        .collect();

    // Sequential computation
    let sequential: Vec<Vec<f64>> = batches
        .iter()
        .map(|batch| {
            batch.outer_iter()
                .map(|row| row.sum())
                .collect()
        })
        .collect();

    // Parallel computation
    let parallel = parallel_batch_inference(&batches, |batch| {
        Ok(batch.outer_iter()
            .map(|row| row.sum())
            .collect())
    }).unwrap();

    // Verify results match
    assert_eq!(sequential.len(), parallel.len());
    for (seq, par) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert!((s - p).abs() < 1e-10);
        }
    }
}

#[test]
fn test_parallel_batch_inference_with_uncertainty() {
    let batches = vec![
        generate_test_data(10, 5),
        generate_test_data(10, 5),
    ];

    let num_samples = 10;

    let results = parallel_batch_inference_with_uncertainty(
        &batches,
        |batch| Ok(vec![batch.sum(); batch.nrows()]),
        num_samples,
    ).unwrap();

    assert_eq!(results.len(), 2);

    for (means, stds) in &results {
        assert_eq!(means.len(), 10);
        assert_eq!(stds.len(), 10);

        // All stds should be 0 since we have deterministic function
        for &std in stds {
            assert!(std < 1e-10);
        }
    }
}

#[test]
fn test_parallel_preprocess() {
    let chunks = vec![
        arr2(&[[1.0, 2.0], [3.0, 4.0]]),
        arr2(&[[5.0, 6.0], [7.0, 8.0]]),
    ];

    let results = parallel_preprocess(&chunks, |chunk| {
        // Simple normalization
        Ok(chunk * 2.0)
    }).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][[0, 0]], 2.0);
    assert_eq!(results[1][[1, 1]], 16.0);
}

#[test]
fn test_parallel_gradient_computation() {
    let batches = vec![
        (arr2(&[[1.0, 2.0]]), Array1::from_vec(vec![1.0])),
        (arr2(&[[3.0, 4.0]]), Array1::from_vec(vec![2.0])),
    ];

    let results = parallel_gradient_computation(&batches, |x, y| {
        // Mock gradient: just return scaled input
        let grad = x * y[0];
        Ok(vec![grad])
    }).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 1);
    assert_eq!(results[0][0][[0, 0]], 1.0);
    assert_eq!(results[1][0][[0, 0]], 6.0);
}

#[test]
fn test_aggregate_gradients() {
    let grad1 = vec![arr2(&[[1.0, 2.0], [3.0, 4.0]])];
    let grad2 = vec![arr2(&[[2.0, 3.0], [4.0, 5.0]])];
    let grad3 = vec![arr2(&[[3.0, 4.0], [5.0, 6.0]])];

    let gradients = vec![grad1, grad2, grad3];

    let result = aggregate_gradients(&gradients).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape(), &[2, 2]);

    // Check averaging
    assert!((result[0][[0, 0]] - 2.0).abs() < 1e-10);
    assert!((result[0][[0, 1]] - 3.0).abs() < 1e-10);
    assert!((result[0][[1, 0]] - 4.0).abs() < 1e-10);
    assert!((result[0][[1, 1]] - 5.0).abs() < 1e-10);
}

#[test]
fn test_aggregate_gradients_empty() {
    let gradients: Vec<Vec<Array2<f64>>> = vec![];
    let result = aggregate_gradients(&gradients);
    assert!(result.is_err());
}

#[test]
fn test_parallel_cross_validation() {
    let data = generate_test_data(100, 10);
    let labels = Array1::from_vec((0..100).map(|i| i as f64).collect());

    let scores = parallel_cross_validation(
        &data,
        &labels,
        5,
        |_train_x, _train_y, val_x, val_y| {
            // Mock evaluation: just return mean of validation labels
            Ok(val_y.mean().unwrap())
        },
    ).unwrap();

    assert_eq!(scores.len(), 5);

    // All scores should be reasonable (around 0-100 range)
    for score in &scores {
        assert!(*score >= 0.0);
        assert!(*score <= 100.0);
    }
}

#[test]
fn test_parallel_grid_search() {
    let train_data = generate_test_data(50, 10);
    let train_labels = Array1::from_vec((0..50).map(|i| i as f64).collect());
    let val_data = generate_test_data(20, 10);
    let val_labels = Array1::from_vec((0..20).map(|i| i as f64).collect());

    #[derive(Clone)]
    struct TestParams {
        alpha: f64,
    }

    let param_grid = vec![
        TestParams { alpha: 0.1 },
        TestParams { alpha: 0.5 },
        TestParams { alpha: 1.0 },
    ];

    let results = parallel_grid_search(
        &param_grid,
        &train_data,
        &train_labels,
        &val_data,
        &val_labels,
        |params, _train_x, _train_y, _val_x, val_y| {
            // Mock training: return param-dependent score
            Ok(params.alpha * val_y.mean().unwrap())
        },
    ).unwrap();

    assert_eq!(results.len(), 3);

    // Scores should increase with alpha
    assert!(results[0].1 < results[1].1);
    assert!(results[1].1 < results[2].1);
}

#[test]
fn test_parallel_ensemble_predict_mean() {
    let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    let models = vec![
        |_: &Array2<f64>| Ok(vec![1.0, 2.0]),
        |_: &Array2<f64>| Ok(vec![3.0, 4.0]),
        |_: &Array2<f64>| Ok(vec![5.0, 6.0]),
    ];

    let result = parallel_ensemble_predict(
        &input,
        &models,
        EnsembleAggregation::Mean,
    ).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 3.0);  // (1 + 3 + 5) / 3
    assert_eq!(result[1], 4.0);  // (2 + 4 + 6) / 3
}

#[test]
fn test_parallel_ensemble_predict_median() {
    let input = arr2(&[[1.0, 2.0]]);

    let models = vec![
        |_: &Array2<f64>| Ok(vec![1.0]),
        |_: &Array2<f64>| Ok(vec![5.0]),
        |_: &Array2<f64>| Ok(vec![3.0]),
    ];

    let result = parallel_ensemble_predict(
        &input,
        &models,
        EnsembleAggregation::Median,
    ).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 3.0);  // median of [1, 3, 5]
}

#[test]
fn test_parallel_ensemble_predict_weighted() {
    let input = arr2(&[[1.0, 2.0]]);

    let models = vec![
        |_: &Array2<f64>| Ok(vec![10.0]),
        |_: &Array2<f64>| Ok(vec![20.0]),
    ];

    let weights = vec![0.8, 0.2];

    let result = parallel_ensemble_predict(
        &input,
        &models,
        EnsembleAggregation::WeightedMean(weights),
    ).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 12.0);  // (10 * 0.8 + 20 * 0.2) / 1.0
}

#[test]
fn test_parallel_ensemble_predict_weighted_mismatch() {
    let input = arr2(&[[1.0, 2.0]]);

    let models = vec![
        |_: &Array2<f64>| Ok(vec![10.0]),
        |_: &Array2<f64>| Ok(vec![20.0]),
    ];

    let weights = vec![0.8];  // Wrong length

    let result = parallel_ensemble_predict(
        &input,
        &models,
        EnsembleAggregation::WeightedMean(weights),
    );

    assert!(result.is_err());
}

#[test]
fn test_parallel_data_augmentation() {
    let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let num_augmentations = 5;

    let results = parallel_data_augmentation(
        &data,
        |d| Ok(d + 1.0),  // Simple augmentation: add 1
        num_augmentations,
    ).unwrap();

    assert_eq!(results.len(), 5);
    for result in &results {
        assert_eq!(result[[0, 0]], 2.0);
        assert_eq!(result[[1, 1]], 5.0);
    }
}

#[test]
fn test_parallel_feature_extraction() {
    let samples = vec![
        Array1::from_vec(vec![1.0, 2.0, 3.0]),
        Array1::from_vec(vec![4.0, 5.0, 6.0]),
        Array1::from_vec(vec![7.0, 8.0, 9.0]),
    ];

    let results = parallel_feature_extraction(&samples, |sample| {
        // Extract mean as feature
        Ok(Array1::from_vec(vec![sample.mean().unwrap()]))
    }).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0][0], 2.0);
    assert_eq!(results[1][0], 5.0);
    assert_eq!(results[2][0], 8.0);
}

#[test]
fn test_scalability_benchmark() {
    let thread_counts = vec![1, 2, 4];

    // Simple workload
    let workload = || -> neuro_divergent::Result<()> {
        let batches = vec![generate_test_data(10, 10); 20];
        parallel_batch_inference(&batches, |batch| {
            Ok(vec![batch.sum()])
        })?;
        Ok(())
    };

    let results = scalability_benchmark(&thread_counts, &workload).unwrap();

    assert_eq!(results.len(), 3);

    // Verify speedup properties
    assert_eq!(results[0].num_threads, 1);
    assert_eq!(results[0].speedup, 1.0);
    assert_eq!(results[0].efficiency, 1.0);

    // Higher thread counts should have speedup > 1
    for result in &results[1..] {
        assert!(result.speedup >= 1.0);
        assert!(result.efficiency > 0.0);
        assert!(result.efficiency <= 1.0);
    }
}

#[test]
fn test_amdahl_analysis() {
    // Create mock benchmark results
    let results = vec![
        BenchmarkResult {
            num_threads: 1,
            duration_ms: 100.0,
            speedup: 1.0,
            efficiency: 1.0,
        },
        BenchmarkResult {
            num_threads: 2,
            duration_ms: 55.0,
            speedup: 1.82,
            efficiency: 0.91,
        },
        BenchmarkResult {
            num_threads: 4,
            duration_ms: 30.0,
            speedup: 3.33,
            efficiency: 0.83,
        },
    ];

    let parallel_fraction = amdahl_analysis(&results);

    // Should be high (>0.8) for well-parallelized workload
    assert!(parallel_fraction > 0.7);
    assert!(parallel_fraction <= 1.0);
}

#[test]
fn test_matrix_ops_parallel_matvec_batch() {
    use neuro_divergent::optimizations::parallel::matrix_ops::*;

    let matrix = arr2(&[
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    let vectors = vec![
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![2.0, 2.0]),
    ];

    let results = parallel_matvec_batch(&matrix, &vectors);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][0], 3.0);  // 1*1 + 2*1
    assert_eq!(results[0][1], 7.0);  // 3*1 + 4*1
    assert_eq!(results[1][0], 6.0);  // 1*2 + 2*2
    assert_eq!(results[1][1], 14.0); // 3*2 + 4*2
}

#[test]
fn test_matrix_ops_parallel_elementwise() {
    use neuro_divergent::optimizations::parallel::matrix_ops::*;

    let arrays = vec![
        arr2(&[[1.0, 2.0]]),
        arr2(&[[3.0, 4.0]]),
    ];

    let results = parallel_elementwise_op(&arrays, |arr| arr * 2.0);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][[0, 0]], 2.0);
    assert_eq!(results[1][[0, 1]], 8.0);
}
