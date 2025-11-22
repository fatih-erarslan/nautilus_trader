/// Production Validation Suite
/// 
/// Comprehensive test suite to validate all production-grade fixes:
/// - Async-safe git2 operations
/// - Async-safe RocksDB operations  
/// - Thread safety validation
/// - Performance benchmarks
/// - Integration testing

#[cfg(test)]
mod production_validation {
    use std::sync::Arc;
    use tokio::task::JoinSet;
    use tempfile::TempDir;
    
    #[cfg(feature = "async-wrappers")]
    use cwts_ultra::async_wrappers::{AsyncGitRepository, AsyncRocksDB, AsyncRocksDBConfig};

    /// Test async git operations under concurrent load
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_concurrent_git_operations() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("test_repo");
        
        // Initialize git repository
        std::process::Command::new("git")
            .args(&["init", repo_path.to_str().unwrap()])
            .output()
            .expect("Failed to init git repository");
            
        // Configure git user
        std::process::Command::new("git")
            .args(&["-C", repo_path.to_str().unwrap(), "config", "user.name", "Test User"])
            .output()
            .expect("Failed to configure git user name");
            
        std::process::Command::new("git")
            .args(&["-C", repo_path.to_str().unwrap(), "config", "user.email", "test@example.com"])
            .output()
            .expect("Failed to configure git user email");

        let repo = Arc::new(AsyncGitRepository::open(&repo_path).await.unwrap());
        let mut join_set = JoinSet::new();

        // Spawn 50 concurrent tasks performing git operations
        for i in 0..50 {
            let repo_clone = Arc::clone(&repo);
            join_set.spawn(async move {
                // Test concurrent branch operations
                let branch_result = repo_clone.current_branch().await;
                assert!(branch_result.is_ok());

                // Test concurrent status operations
                let status_result = repo_clone.status().await;
                assert!(status_result.is_ok());

                // Create test file for add operation
                let file_path = format!("test_file_{}.txt", i);
                let full_path = repo_path.join(&file_path);
                tokio::fs::write(&full_path, format!("Test content {}", i))
                    .await
                    .unwrap();

                // Test concurrent add operations
                let add_result = repo_clone.add_files(&[&file_path]).await;
                assert!(add_result.is_ok());

                i
            });
        }

        // Wait for all tasks to complete
        let mut completed = 0;
        while let Some(result) = join_set.join_next().await {
            assert!(result.is_ok());
            completed += 1;
        }

        assert_eq!(completed, 50);

        // Verify performance metrics
        let metrics = repo.get_metrics().await;
        assert!(metrics.operation_count >= 100); // At least 2 operations per task
        assert!(metrics.total_duration.as_secs() < 30); // Should complete within 30 seconds
    }

    /// Test async RocksDB operations under concurrent load
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_concurrent_rocksdb_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");
        
        let config = AsyncRocksDBConfig {
            iterator_buffer_size: 1000,
            cache_ttl_secs: 60,
            enable_wal: true,
            block_cache_size: 32 * 1024 * 1024, // 32MB
            enable_compression: true,
        };

        let db = Arc::new(AsyncRocksDB::open(&db_path, config).await.unwrap());
        let mut join_set = JoinSet::new();

        // Spawn 100 concurrent tasks performing database operations
        for i in 0..100 {
            let db_clone = Arc::clone(&db);
            join_set.spawn(async move {
                let key = format!("key_{:04}", i);
                let value = format!("value_{}_with_some_longer_content_to_test_performance", i);

                // Test concurrent writes
                let put_result = db_clone.put(&key, &value).await;
                assert!(put_result.is_ok());

                // Test concurrent reads
                let get_result = db_clone.get(&key).await;
                assert!(get_result.is_ok());
                assert_eq!(get_result.unwrap().unwrap(), value.as_bytes());

                // Test some deletes
                if i % 10 == 0 {
                    let delete_result = db_clone.delete(&key).await;
                    assert!(delete_result.is_ok());
                }

                i
            });
        }

        // Wait for all tasks to complete
        let mut completed = 0;
        while let Some(result) = join_set.join_next().await {
            assert!(result.is_ok());
            completed += 1;
        }

        assert_eq!(completed, 100);

        // Flush to ensure all writes are persisted
        db.flush().await.unwrap();

        // Verify performance metrics
        let metrics = db.get_metrics().await;
        assert!(metrics.read_operations >= 100);
        assert!(metrics.write_operations >= 100);
        assert!(metrics.cache_hits > 0);

        // Test iterator functionality
        let iter = db.iter().await.unwrap();
        let all_items = iter.collect().await.unwrap();
        
        // Should have 90 items (100 - 10 deleted)
        assert_eq!(all_items.len(), 90);
    }

    /// Test Send + Sync trait bounds compilation
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_send_sync_trait_bounds() {
        use std::sync::Arc;
        
        // This test ensures our wrappers are properly Send + Sync
        async fn git_operations(repo: Arc<AsyncGitRepository>) {
            let _ = repo.current_branch().await;
            let _ = repo.status().await;
        }

        async fn db_operations(db: Arc<AsyncRocksDB>) {
            let _ = db.get("test_key").await;
            let _ = db.put("test_key", "test_value").await;
        }

        let temp_dir = TempDir::new().unwrap();
        
        // Test git wrapper
        let git_path = temp_dir.path().join("git_repo");
        std::process::Command::new("git")
            .args(&["init", git_path.to_str().unwrap()])
            .output()
            .expect("Failed to init git repository");
            
        let repo = Arc::new(AsyncGitRepository::open(&git_path).await.unwrap());
        
        // Test db wrapper
        let db_path = temp_dir.path().join("rocks_db");
        let config = AsyncRocksDBConfig::default();
        let db = Arc::new(AsyncRocksDB::open(&db_path, config).await.unwrap());

        // These operations should compile and work across task boundaries
        let git_task = tokio::spawn(git_operations(Arc::clone(&repo)));
        let db_task = tokio::spawn(db_operations(Arc::clone(&db)));

        let (git_result, db_result) = tokio::join!(git_task, db_task);
        assert!(git_result.is_ok());
        assert!(db_result.is_ok());
    }

    /// Performance benchmark for async wrappers
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn benchmark_async_wrapper_performance() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("bench_db");
        
        let config = AsyncRocksDBConfig {
            iterator_buffer_size: 10000,
            cache_ttl_secs: 300,
            enable_wal: true,
            block_cache_size: 64 * 1024 * 1024, // 64MB
            enable_compression: true,
        };

        let db = AsyncRocksDB::open(&db_path, config).await.unwrap();
        
        let start_time = std::time::Instant::now();
        let num_operations = 1000;

        // Benchmark write performance
        for i in 0..num_operations {
            let key = format!("bench_key_{:06}", i);
            let value = format!("benchmark_value_{}_with_substantial_content_for_realistic_testing_scenarios", i);
            db.put(&key, &value).await.unwrap();
        }

        // Flush writes
        db.flush().await.unwrap();

        let write_duration = start_time.elapsed();
        println!("Write performance: {} ops/sec", 
                num_operations as f64 / write_duration.as_secs_f64());

        // Benchmark read performance
        let read_start = std::time::Instant::now();
        
        for i in 0..num_operations {
            let key = format!("bench_key_{:06}", i);
            let result = db.get(&key).await.unwrap();
            assert!(result.is_some());
        }

        let read_duration = read_start.elapsed();
        println!("Read performance: {} ops/sec", 
                num_operations as f64 / read_duration.as_secs_f64());

        // Performance assertions
        assert!(write_duration.as_millis() < 5000); // Should complete within 5 seconds
        assert!(read_duration.as_millis() < 2000);  // Reads should be faster

        let metrics = db.get_metrics().await;
        assert_eq!(metrics.read_operations, num_operations);
        assert_eq!(metrics.write_operations, num_operations);
    }

    /// Test error handling and recovery
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_error_handling_and_recovery() {
        // Test git error handling
        let invalid_path = "/invalid/path/that/does/not/exist";
        let git_result = AsyncGitRepository::open(invalid_path).await;
        assert!(git_result.is_err());

        // Test RocksDB error handling
        let invalid_db_path = "/invalid/path/for/database";
        let config = AsyncRocksDBConfig::default();
        let db_result = AsyncRocksDB::open(invalid_db_path, config).await;
        assert!(db_result.is_err());

        // Test recovery after errors
        let temp_dir = TempDir::new().unwrap();
        let valid_db_path = temp_dir.path().join("recovery_db");
        let db = AsyncRocksDB::open(&valid_db_path, AsyncRocksDBConfig::default()).await.unwrap();
        
        // Test graceful handling of non-existent keys
        let missing_key_result = db.get("non_existent_key").await.unwrap();
        assert!(missing_key_result.is_none());
    }

    /// Test memory usage and cleanup
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_memory_management() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("memory_test_db");
        
        let config = AsyncRocksDBConfig {
            cache_ttl_secs: 1, // Short TTL for testing cache cleanup
            ..Default::default()
        };

        let db = AsyncRocksDB::open(&db_path, config).await.unwrap();

        // Fill cache
        for i in 0..100 {
            let key = format!("cache_key_{}", i);
            let value = format!("cache_value_{}", i);
            db.put(&key, &value).await.unwrap();
            db.get(&key).await.unwrap(); // Populate cache
        }

        db.flush().await.unwrap();

        // Wait for cache TTL expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Clear cache manually
        db.clear_cache().await;

        // Test that database still works after cache clear
        let result = db.get("cache_key_0").await.unwrap();
        assert!(result.is_some());
    }

    /// Integration test with real workload patterns
    #[cfg(feature = "async-wrappers")]
    #[tokio::test]
    async fn test_production_workload_simulation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("workload_db");
        
        let config = AsyncRocksDBConfig {
            iterator_buffer_size: 5000,
            cache_ttl_secs: 300,
            enable_wal: true,
            block_cache_size: 128 * 1024 * 1024, // 128MB
            enable_compression: true,
        };

        let db = Arc::new(AsyncRocksDB::open(&db_path, config).await.unwrap());
        let mut join_set = JoinSet::new();

        // Simulate writer threads
        for worker_id in 0..5 {
            let db_clone = Arc::clone(&db);
            join_set.spawn(async move {
                for i in 0..200 {
                    let key = format!("worker_{}_item_{:04}", worker_id, i);
                    let value = serde_json::json!({
                        "worker_id": worker_id,
                        "item_id": i,
                        "timestamp": chrono::Utc::now().timestamp(),
                        "data": format!("payload_data_for_item_{}", i)
                    }).to_string();
                    
                    db_clone.put(&key, &value).await.unwrap();
                    
                    // Simulate some processing time
                    if i % 50 == 0 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    }
                }
                worker_id
            });
        }

        // Simulate reader threads
        for reader_id in 0..3 {
            let db_clone = Arc::clone(&db);
            join_set.spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Let some writes happen first
                
                for worker_id in 0..5 {
                    for i in 0..50 {
                        let key = format!("worker_{}_item_{:04}", worker_id, i);
                        let result = db_clone.get(&key).await.unwrap();
                        if let Some(value) = result {
                            let parsed: serde_json::Value = serde_json::from_slice(&value).unwrap();
                            assert_eq!(parsed["worker_id"], worker_id);
                            assert_eq!(parsed["item_id"], i);
                        }
                    }
                }
                1000 + reader_id // Different return value to distinguish readers
            });
        }

        // Wait for all tasks
        let mut writers_completed = 0;
        let mut readers_completed = 0;
        
        while let Some(result) = join_set.join_next().await {
            let worker_result = result.unwrap();
            if worker_result < 1000 {
                writers_completed += 1;
            } else {
                readers_completed += 1;
            }
        }

        assert_eq!(writers_completed, 5);
        assert_eq!(readers_completed, 3);

        // Flush all pending writes
        db.flush().await.unwrap();

        // Verify final state
        let metrics = db.get_metrics().await;
        assert!(metrics.write_operations >= 1000); // 5 workers * 200 items
        assert!(metrics.read_operations >= 750);   // 3 readers * 5 workers * 50 items
        assert!(metrics.cache_hits > 0);

        // Test iterator over all data
        let iter = db.iter().await.unwrap();
        let all_items = iter.collect().await.unwrap();
        assert_eq!(all_items.len(), 1000); // All items should be present

        // Test prefix iteration
        let prefix_iter = db.iter_prefix("worker_0_").await.unwrap();
        let worker_0_items = prefix_iter.collect().await.unwrap();
        assert_eq!(worker_0_items.len(), 200); // Worker 0's items
    }
}

/// Module for testing alternative neural implementation
#[cfg(test)]
mod neural_validation {
    use cwts_ultra::neural::alternative_impl::*;

    #[test]
    fn test_alternative_neural_implementation() {
        // Test activation functions
        let input = ndarray::Array1::from(vec![-1.0, 0.0, 1.0, 2.0]);
        
        let relu = ActivationFunction::ReLU.apply(&input);
        assert_eq!(relu, ndarray::Array1::from(vec![0.0, 0.0, 1.0, 2.0]));

        let sigmoid = ActivationFunction::Sigmoid.apply(&input);
        assert!(sigmoid[0] > 0.0 && sigmoid[0] < 0.5);
        assert!((sigmoid[1] - 0.5).abs() < 1e-6);
        assert!(sigmoid[2] > 0.5 && sigmoid[2] < 1.0);
        assert!(sigmoid[3] > sigmoid[2]);

        // Test MLP creation and forward pass
        let layer_sizes = [4, 8, 3];
        let activations = [ActivationFunction::ReLU, ActivationFunction::Sigmoid];
        let mut mlp = MLP::new(&layer_sizes, &activations, 0.01).unwrap();

        let test_input = ndarray::Array1::from(vec![1.0, -0.5, 0.8, -0.2]);
        let output = mlp.forward(&test_input).unwrap();
        
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0)); // Sigmoid bounds

        // Test LSTM cell
        let lstm = LSTMCell::new(4, 6);
        let input = ndarray::Array1::from(vec![0.5, -0.3, 0.8, 0.1]);
        let hidden = ndarray::Array1::zeros(6);
        let cell = ndarray::Array1::zeros(6);

        let (new_hidden, new_cell) = lstm.forward(&input, &hidden, &cell).unwrap();
        assert_eq!(new_hidden.len(), 6);
        assert_eq!(new_cell.len(), 6);
        
        // Verify outputs are in reasonable ranges
        assert!(new_hidden.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_neural_training() {
        // Create a simple network for XOR problem
        let layer_sizes = [2, 4, 1];
        let activations = [ActivationFunction::ReLU, ActivationFunction::Sigmoid];
        let mut mlp = MLP::new(&layer_sizes, &activations, 0.1).unwrap();

        // XOR training data
        let training_data = vec![
            (ndarray::Array1::from(vec![0.0, 0.0]), ndarray::Array1::from(vec![0.0])),
            (ndarray::Array1::from(vec![0.0, 1.0]), ndarray::Array1::from(vec![1.0])),
            (ndarray::Array1::from(vec![1.0, 0.0]), ndarray::Array1::from(vec![1.0])),
            (ndarray::Array1::from(vec![1.0, 1.0]), ndarray::Array1::from(vec![0.0])),
        ];

        // Test training steps
        let initial_loss = mlp.train_step(&training_data[0].0, &training_data[0].1).unwrap();
        assert!(initial_loss >= 0.0);

        // Train for a few iterations
        for _ in 0..10 {
            for (input, target) in &training_data {
                let loss = mlp.train_step(input, target).unwrap();
                assert!(loss >= 0.0);
            }
        }

        // Test batch prediction
        let inputs = ndarray::Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        ).unwrap();
        
        let outputs = mlp.predict_batch(&inputs).unwrap();
        assert_eq!(outputs.shape(), &[4, 1]);
    }

    #[test]
    fn test_neural_serialization() {
        let layer_sizes = [3, 5, 2];
        let activations = [ActivationFunction::Tanh, ActivationFunction::Linear];
        let original_mlp = MLP::new(&layer_sizes, &activations, 0.05).unwrap();

        // Test serialization to JSON
        let serialized = serde_json::to_string(&original_mlp).unwrap();
        let deserialized_mlp: MLP = serde_json::from_str(&serialized).unwrap();

        // Verify structure is preserved
        assert_eq!(original_mlp.layers.len(), deserialized_mlp.layers.len());
        assert!((original_mlp.learning_rate - deserialized_mlp.learning_rate).abs() < 1e-6);

        // Test that forward pass produces same results
        let test_input = ndarray::Array1::from(vec![0.5, -0.3, 0.8]);
        
        let mut original_copy = original_mlp.clone();
        let mut deserialized_copy = deserialized_mlp;
        
        let original_output = original_copy.forward(&test_input).unwrap();
        let deserialized_output = deserialized_copy.forward(&test_input).unwrap();

        // Outputs should be identical (within floating point precision)
        for (orig, deser) in original_output.iter().zip(deserialized_output.iter()) {
            assert!((orig - deser).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neural_trainer() {
        let layer_sizes = [2, 4, 1];
        let activations = [ActivationFunction::ReLU, ActivationFunction::Sigmoid];
        let mlp = MLP::new(&layer_sizes, &activations, 0.1).unwrap();
        
        let optimizer = OptimizerType::SGD { learning_rate: 0.1 };
        let mut trainer = NeuralTrainer::new(mlp, optimizer, 2);

        // Simple training data
        let training_data = vec![
            (ndarray::Array1::from(vec![0.0, 1.0]), ndarray::Array1::from(vec![1.0])),
            (ndarray::Array1::from(vec![1.0, 0.0]), ndarray::Array1::from(vec![1.0])),
            (ndarray::Array1::from(vec![0.0, 0.0]), ndarray::Array1::from(vec![0.0])),
            (ndarray::Array1::from(vec![1.0, 1.0]), ndarray::Array1::from(vec![0.0])),
        ];

        let metrics = trainer.train_epoch(&training_data).unwrap();
        
        assert_eq!(metrics.epoch, 1);
        assert!(metrics.total_loss >= 0.0);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert!(metrics.learning_rate > 0.0);
        assert!(metrics.training_time.as_millis() > 0);
    }
}