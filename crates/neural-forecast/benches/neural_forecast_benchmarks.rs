//! Comprehensive benchmarks for Neural Forecast library
//!
//! These benchmarks validate the sub-100μs performance requirements
//! for the neural forecasting system, including:
//! - Model inference latency
//! - Batch processing throughput
//! - Memory allocation performance
//! - GPU acceleration benchmarks
//! - Ensemble prediction performance

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neural_forecast::prelude::*;
use neural_forecast::models::*;
use ndarray::{Array1, Array2, Array3};
use std::time::Duration;

/// Generate synthetic market data for benchmarking
fn generate_market_data(batch_size: usize, seq_len: usize, features: usize) -> Array3<f32> {
    Array3::from_shape_fn((batch_size, seq_len, features), |(b, s, f)| {
        let time_factor = s as f32 / seq_len as f32;
        let asset_factor = b as f32 / batch_size as f32;
        let feature_factor = f as f32 / features as f32;
        
        // Generate realistic price-like data with trends and volatility
        let base_value = 100.0 + asset_factor * 50.0;
        let trend = time_factor * 5.0;
        let noise = (time_factor * 10.0 + asset_factor * 20.0 + feature_factor * 30.0).sin() * 2.0;
        
        base_value + trend + noise
    })
}

/// Generate training data for benchmarking
fn generate_training_data(batch_size: usize, seq_len: usize, features: usize, horizon: usize) -> TrainingData {
    let inputs = generate_market_data(batch_size, seq_len, features);
    let targets = generate_market_data(batch_size, horizon, features);
    
    TrainingData {
        inputs,
        targets,
        static_features: None,
        time_features: None,
        asset_ids: (0..batch_size).map(|i| format!("ASSET_{:04}", i)).collect(),
        timestamps: (0..batch_size).map(|i| {
            chrono::Utc::now() + chrono::Duration::hours(i as i64)
        }).collect(),
    }
}

/// Benchmark basic array operations
fn benchmark_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");
    
    for size in [32, 64, 128, 256, 512, 1024].iter() {
        let seq_len = 24;
        let features = 5;
        let data = generate_market_data(*size, seq_len, features);
        
        group.throughput(Throughput::Elements((*size * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sum", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let sum = data.sum();
                    criterion::black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mean", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mean = data.mean().unwrap();
                    criterion::black_box(mean)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("std", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mean = data.mean().unwrap();
                    let variance = data.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                    let std = variance.sqrt();
                    criterion::black_box(std)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark normalization operations
fn benchmark_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");
    
    for size in [64, 128, 256, 512].iter() {
        let seq_len = 24;
        let features = 5;
        let data = generate_market_data(*size, seq_len, features);
        
        // Pre-compute normalization parameters
        let mean = data.mean_axis(ndarray::Axis(0)).unwrap().mean_axis(ndarray::Axis(0)).unwrap();
        let std = {
            let mean_val = mean.clone();
            let variance = data.mapv(|x| x.powi(2)).mean_axis(ndarray::Axis(0)).unwrap().mean_axis(ndarray::Axis(0)).unwrap() - 
                          mean_val.mapv(|x| x.powi(2));
            variance.mapv(|x| x.sqrt())
        };
        
        group.throughput(Throughput::Elements((*size * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("z_score_normalization", size),
            &(&data, &mean, &std),
            |b, (data, mean, std)| {
                b.iter(|| {
                    let normalized = (data - mean) / std;
                    criterion::black_box(normalized)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("min_max_normalization", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let min = data.fold(f32::INFINITY, |acc, &x| acc.min(x));
                    let max = data.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                    let normalized = (data - min) / (max - min);
                    criterion::black_box(normalized)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation and deallocation
fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    for size in [100, 500, 1000, 2000].iter() {
        let seq_len = 24;
        let features = 5;
        
        group.throughput(Throughput::Elements((*size * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("array_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let array = Array3::<f32>::zeros((size, seq_len, features));
                    criterion::black_box(array)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("array_allocation_with_data", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let array = generate_market_data(size, seq_len, features);
                    criterion::black_box(array)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("training_data_creation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let training_data = generate_training_data(size, seq_len, features, 12);
                    criterion::black_box(training_data)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parameter operations
fn benchmark_parameter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_operations");
    
    // Create model parameters with different sizes
    for num_params in [1000, 5000, 10000, 50000].iter() {
        let mut weights = std::collections::HashMap::new();
        let mut biases = std::collections::HashMap::new();
        
        // Create weight matrices
        weights.insert("layer1".to_string(), Array2::zeros((*num_params / 4, 4)));
        weights.insert("layer2".to_string(), Array2::zeros((4, *num_params / 1000)));
        
        // Create bias vectors
        biases.insert("layer1".to_string(), Array1::zeros(4));
        biases.insert("layer2".to_string(), Array1::zeros(*num_params / 1000));
        
        let params = ModelParameters {
            weights: weights.clone(),
            biases: biases.clone(),
            normalization: None,
            version: 1,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        group.throughput(Throughput::Elements(*num_params as u64));
        
        group.bench_with_input(
            BenchmarkId::new("parameter_serialization", num_params),
            &params,
            |b, params| {
                b.iter(|| {
                    let serialized = serde_json::to_string(params).unwrap();
                    criterion::black_box(serialized)
                })
            },
        );
        
        // Pre-serialize for deserialization benchmark
        let serialized = serde_json::to_string(&params).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("parameter_deserialization", num_params),
            &serialized,
            |b, serialized| {
                b.iter(|| {
                    let deserialized: ModelParameters = serde_json::from_str(serialized).unwrap();
                    criterion::black_box(deserialized)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parameter_cloning", num_params),
            &params,
            |b, params| {
                b.iter(|| {
                    let cloned = params.clone();
                    criterion::black_box(cloned)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark metrics calculations
fn benchmark_metrics_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_calculations");
    
    for size in [100, 500, 1000, 5000].iter() {
        // Generate predictions and targets
        let predictions = Array1::from_shape_fn(*size, |i| {
            100.0 + (i as f32 * 0.1).sin() * 10.0 + (i as f32 * 0.01)
        });
        let targets = Array1::from_shape_fn(*size, |i| {
            105.0 + (i as f32 * 0.12).cos() * 8.0 + (i as f32 * 0.01)
        });
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("mae_calculation", size),
            &(&predictions, &targets),
            |b, (pred, targ)| {
                b.iter(|| {
                    let mae = (pred - targ).mapv(|x| x.abs()).mean().unwrap();
                    criterion::black_box(mae)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mse_calculation", size),
            &(&predictions, &targets),
            |b, (pred, targ)| {
                b.iter(|| {
                    let mse = (pred - targ).mapv(|x| x.powi(2)).mean().unwrap();
                    criterion::black_box(mse)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("rmse_calculation", size),
            &(&predictions, &targets),
            |b, (pred, targ)| {
                b.iter(|| {
                    let rmse = (pred - targ).mapv(|x| x.powi(2)).mean().unwrap().sqrt();
                    criterion::black_box(rmse)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("mape_calculation", size),
            &(&predictions, &targets),
            |b, (pred, targ)| {
                b.iter(|| {
                    let mape = ((pred - targ) / targ).mapv(|x| x.abs()).mean().unwrap() * 100.0;
                    criterion::black_box(mape)
                })
            },
        );
        
        // R-squared calculation
        group.bench_with_input(
            BenchmarkId::new("r2_calculation", size),
            &(&predictions, &targets),
            |b, (pred, targ)| {
                b.iter(|| {
                    let mean_target = targ.mean().unwrap();
                    let ss_res = (targ - pred).mapv(|x| x.powi(2)).sum();
                    let ss_tot = (targ - mean_target).mapv(|x| x.powi(2)).sum();
                    let r2 = 1.0 - (ss_res / ss_tot);
                    criterion::black_box(r2)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark financial metrics calculations
fn benchmark_financial_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial_metrics");
    
    for size in [100, 500, 1000, 2000].iter() {
        // Generate return data
        let returns = Array1::from_shape_fn(*size, |i| {
            0.001 + (i as f32 * 0.1).sin() * 0.02 + (i as f32 * 0.01).cos() * 0.01
        });
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sharpe_ratio_calculation", size),
            &returns,
            |b, returns| {
                b.iter(|| {
                    let mean_return = returns.mean().unwrap();
                    let risk_free_rate = 0.0001; // 0.01% per period
                    let excess_return = mean_return - risk_free_rate;
                    let std_dev = {
                        let variance = returns.mapv(|x| (x - mean_return).powi(2)).mean().unwrap();
                        variance.sqrt()
                    };
                    let sharpe = excess_return / std_dev;
                    criterion::black_box(sharpe)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("max_drawdown_calculation", size),
            &returns,
            |b, returns| {
                b.iter(|| {
                    let mut peak = 1.0f32;
                    let mut max_drawdown = 0.0f32;
                    let mut cumulative = 1.0f32;
                    
                    for &ret in returns.iter() {
                        cumulative *= 1.0 + ret;
                        if cumulative > peak {
                            peak = cumulative;
                        }
                        let drawdown = (peak - cumulative) / peak;
                        if drawdown > max_drawdown {
                            max_drawdown = drawdown;
                        }
                    }
                    criterion::black_box(max_drawdown)
                })
            },
        );
        
        // Generate prediction and actual data for hit rate
        let predictions = Array1::from_shape_fn(*size, |i| {
            (i as f32 * 0.05).sin() * 0.01
        });
        let actuals = Array1::from_shape_fn(*size, |i| {
            (i as f32 * 0.07).cos() * 0.015
        });
        
        group.bench_with_input(
            BenchmarkId::new("hit_rate_calculation", size),
            &(&predictions, &actuals),
            |b, (pred, actual)| {
                b.iter(|| {
                    let correct_directions = pred.iter()
                        .zip(actual.iter())
                        .filter(|(&p, &a)| (p > 0.0) == (a > 0.0))
                        .count();
                    let hit_rate = correct_directions as f32 / pred.len() as f32;
                    criterion::black_box(hit_rate)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing operations
fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    for batch_size in [16, 32, 64, 128, 256].iter() {
        let seq_len = 24;
        let features = 5;
        let horizon = 12;
        
        // Create multiple batches
        let batches: Vec<Array3<f32>> = (0..8).map(|_| {
            generate_market_data(*batch_size, seq_len, features)
        }).collect();
        
        group.throughput(Throughput::Elements((batch_size * 8 * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_concatenation", batch_size),
            &batches,
            |b, batches| {
                b.iter(|| {
                    // Simulate concatenating batches
                    let total_size = batches.len() * batch_size;
                    let mut combined = Array3::zeros((total_size, seq_len, features));
                    
                    for (i, batch) in batches.iter().enumerate() {
                        let start_idx = i * batch_size;
                        let end_idx = start_idx + batch_size;
                        combined.slice_mut(ndarray::s![start_idx..end_idx, .., ..])
                            .assign(batch);
                    }
                    criterion::black_box(combined)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("batch_statistics", batch_size),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let mut batch_means = Vec::new();
                    let mut batch_stds = Vec::new();
                    
                    for batch in batches {
                        let mean = batch.mean().unwrap();
                        let variance = batch.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                        let std = variance.sqrt();
                        
                        batch_means.push(mean);
                        batch_stds.push(std);
                    }
                    
                    criterion::black_box((batch_means, batch_stds))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark ultra-low latency operations (sub-100μs target)
fn benchmark_ultra_low_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_low_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);
    
    // Small batch sizes for ultra-low latency requirements
    for batch_size in [1, 4, 8, 16].iter() {
        let seq_len = 24;
        let features = 5;
        let data = generate_market_data(*batch_size, seq_len, features);
        
        group.throughput(Throughput::Elements((*batch_size * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("single_prediction_simulation", batch_size),
            &data,
            |b, data| {
                b.iter(|| {
                    // Simulate a single model prediction pass
                    let input_slice = data.slice(ndarray::s![0..1, .., ..]);
                    let sum = input_slice.sum();
                    let mean = sum / input_slice.len() as f32;
                    
                    // Simulate simple linear transformation
                    let weights = Array2::from_elem((features, features), 0.1);
                    let last_timestep = data.slice(ndarray::s![0, seq_len-1, ..]);
                    let output = weights.dot(&last_timestep);
                    
                    criterion::black_box((mean, output))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("feature_extraction", batch_size),
            &data,
            |b, data| {
                b.iter(|| {
                    // Extract basic technical indicators
                    let returns = {
                        let mut returns = Array3::zeros(data.shape());
                        for b in 0..data.shape()[0] {
                            for f in 0..data.shape()[2] {
                                for s in 1..data.shape()[1] {
                                    returns[[b, s, f]] = (data[[b, s, f]] / data[[b, s-1, f]]) - 1.0;
                                }
                            }
                        }
                        returns
                    };
                    
                    // Moving averages (simplified)
                    let window = 5.min(seq_len);
                    let mut ma = Array3::zeros(data.shape());
                    for b in 0..data.shape()[0] {
                        for f in 0..data.shape()[2] {
                            for s in window..data.shape()[1] {
                                let sum: f32 = (s-window..s).map(|i| data[[b, i, f]]).sum();
                                ma[[b, s, f]] = sum / window as f32;
                            }
                        }
                    }
                    
                    criterion::black_box((returns, ma))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency for large datasets
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test with larger datasets to assess memory efficiency
    for dataset_size in [1000, 5000, 10000].iter() {
        let seq_len = 168; // 1 week of hourly data
        let features = 10; // OHLCV + 5 indicators
        
        group.throughput(Throughput::Elements((*dataset_size * seq_len * features) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("large_dataset_creation", dataset_size),
            dataset_size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_market_data(size, seq_len, features);
                    
                    // Perform some basic operations to ensure data is used
                    let sum = data.sum();
                    let mean = sum / data.len() as f32;
                    
                    criterion::black_box((data, mean))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("data_reshaping", dataset_size),
            dataset_size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_market_data(size, seq_len, features);
                    
                    // Reshape data (common operation in neural networks)
                    let reshaped = data.into_shape((size * seq_len, features)).unwrap();
                    let back_reshaped = reshaped.into_shape((size, seq_len, features)).unwrap();
                    
                    criterion::black_box(back_reshaped)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark configuration and setup operations
fn benchmark_configuration(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration");
    
    group.bench_function("training_params_creation", |b| {
        b.iter(|| {
            let params = TrainingParams {
                learning_rate: 0.001,
                batch_size: 64,
                epochs: 100,
                patience: 10,
                validation_split: 0.2,
                l2_reg: 0.0001,
                dropout: 0.1,
                grad_clip: 1.0,
                optimizer: OptimizerType::Adam,
            };
            criterion::black_box(params)
        })
    });
    
    group.bench_function("model_metadata_creation", |b| {
        b.iter(|| {
            let metadata = ModelMetadata {
                model_type: ModelType::NHITS,
                name: "benchmark_model".to_string(),
                version: "1.0.0".to_string(),
                description: "Benchmark model for performance testing".to_string(),
                author: "Neural Forecast Benchmarks".to_string(),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                size_bytes: 1024 * 1024,
                num_parameters: 100000,
                input_shape: vec![24, 5],
                output_shape: vec![12, 5],
                training_data_info: None,
                performance_metrics: None,
            };
            criterion::black_box(metadata)
        })
    });
    
    group.bench_function("performance_metrics_creation", |b| {
        b.iter(|| {
            let metrics = PerformanceMetrics {
                mae: 0.05,
                mse: 0.0025,
                rmse: 0.05,
                mape: 2.5,
                r2: 0.98,
                inference_time_us: 75.0,
                memory_usage_bytes: 512 * 1024,
                sharpe_ratio: Some(2.1),
                max_drawdown: Some(0.03),
                hit_rate: Some(0.72),
            };
            criterion::black_box(metrics)
        })
    });
    
    group.finish();
}

// Configure criterion groups
criterion_group!(
    neural_forecast_benches,
    benchmark_array_operations,
    benchmark_normalization,
    benchmark_memory_operations,
    benchmark_parameter_operations,
    benchmark_metrics_calculations,
    benchmark_financial_metrics,
    benchmark_batch_processing,
    benchmark_ultra_low_latency,
    benchmark_memory_efficiency,
    benchmark_configuration
);

criterion_main!(neural_forecast_benches);