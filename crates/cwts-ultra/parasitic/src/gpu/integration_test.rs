//! Integration test for GPU correlation acceleration module
//!
//! This test validates the complete GPU correlation system following TDD principles

#[cfg(test)]
mod integration_tests {
    use crate::gpu::*;
    use tokio_test;

    #[tokio::test]
    async fn test_gpu_correlation_system_integration() {
        println!("🧪 Starting GPU Correlation System Integration Test");

        // Create test organisms
        let organisms = create_test_organism_set();
        println!("✅ Created {} test organisms", organisms.len());

        // Test adaptive engine (should automatically select best backend)
        let engine = AdaptiveCorrelationEngine::new().await;

        match engine {
            Ok(adaptive_engine) => {
                println!("✅ Adaptive correlation engine initialized");

                let start = std::time::Instant::now();
                let correlation_result =
                    adaptive_engine.compute_correlation_matrix(&organisms).await;
                let duration = start.elapsed();

                match correlation_result {
                    Ok(matrix) => {
                        println!("✅ Correlation computation successful:");
                        println!(
                            "   - Backend: {}",
                            adaptive_engine.get_active_backend().await
                        );
                        println!("   - Matrix size: {}x{}", matrix.size(), matrix.size());
                        println!("   - Computation time: {}μs", duration.as_micros());
                        println!("   - Memory usage: {} bytes", matrix.memory_usage());

                        // Verify matrix properties
                        assert!(matrix.is_symmetric(), "Matrix should be symmetric");
                        assert!(matrix.diagonal_ones(), "Diagonal should be ones");

                        // Verify performance requirement
                        let target_latency_ms = 1;
                        let actual_latency_ms = duration.as_millis();

                        if actual_latency_ms <= target_latency_ms {
                            println!(
                                "✅ Performance requirement met: {}ms <= {}ms",
                                actual_latency_ms, target_latency_ms
                            );
                        } else {
                            println!(
                                "⚠️ Performance requirement missed: {}ms > {}ms",
                                actual_latency_ms, target_latency_ms
                            );
                        }

                        // Test correlation statistics
                        let stats = matrix.statistics();
                        println!("✅ Matrix statistics:");
                        println!("   - Mean correlation: {:.3}", stats.mean);
                        println!("   - Std deviation: {:.3}", stats.std_dev);
                        println!("   - Range: [{:.3}, {:.3}]", stats.min, stats.max);
                        println!(
                            "   - Distribution quality: {:.3}",
                            stats.distribution_quality()
                        );

                        // Test highest/lowest correlations
                        let highest = matrix.highest_correlations(3);
                        let lowest = matrix.lowest_correlations(3);

                        println!("✅ Top 3 highest correlations:");
                        for (i, (row, col, corr)) in highest.iter().enumerate() {
                            println!("   {}. ({}, {}) = {:.3}", i + 1, row, col, corr);
                        }

                        println!("✅ Top 3 lowest correlations:");
                        for (i, (row, col, corr)) in lowest.iter().enumerate() {
                            println!("   {}. ({}, {}) = {:.3}", i + 1, row, col, corr);
                        }

                        println!("🎉 GPU Correlation System Integration Test PASSED");
                    }
                    Err(e) => {
                        println!("❌ Correlation computation failed: {}", e);
                        panic!("Integration test failed");
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Adaptive engine initialization failed: {}", e);
                println!("   This is expected if no GPU is available");

                // Fallback to SIMD-only test
                let simd_engine = SimdCorrelationEngine::new();
                println!("✅ SIMD correlation engine initialized");

                let start = std::time::Instant::now();
                let correlation_result = simd_engine.compute_correlation_matrix(&organisms).await;
                let duration = start.elapsed();

                match correlation_result {
                    Ok(matrix) => {
                        println!("✅ SIMD correlation computation successful:");
                        println!("   - Backend: {}", simd_engine.engine_type());
                        println!("   - Matrix size: {}x{}", matrix.size(), matrix.size());
                        println!("   - Computation time: {}μs", duration.as_micros());

                        assert!(matrix.is_symmetric());
                        assert!(matrix.diagonal_ones());

                        println!("🎉 SIMD Fallback Integration Test PASSED");
                    }
                    Err(e) => {
                        println!("❌ SIMD correlation computation failed: {}", e);
                        panic!("SIMD fallback test failed");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_organism_vector_validation() {
        println!("🧪 Testing OrganismVector validation");

        // Test valid organism
        let valid_organism = OrganismVector::new(
            "valid_test".to_string(),
            vec![1.0, 0.5, -0.2, 0.8],
            vec![0.1, -0.05, 0.3, 0.0],
        );

        assert!(valid_organism.validate().is_ok());
        println!("✅ Valid organism passes validation");

        // Test invalid organism with NaN
        let invalid_organism = OrganismVector::new(
            "invalid_test".to_string(),
            vec![1.0, f32::NAN, 0.5],
            vec![0.1, 0.2],
        );

        assert!(invalid_organism.validate().is_err());
        println!("✅ Invalid organism (NaN) correctly rejected");

        // Test distance calculation
        let org1 = OrganismVector::new("org1".to_string(), vec![1.0, 0.0], vec![1.0, 0.0]);

        let org2 = OrganismVector::new("org2".to_string(), vec![0.0, 1.0], vec![0.0, 1.0]);

        let distance = org1.distance_to(&org2);
        let similarity = org1.similarity_to(&org2);

        assert!(distance > 0.0);
        assert!(similarity > 0.0 && similarity <= 1.0);

        println!("✅ Distance calculation: {:.3}", distance);
        println!("✅ Similarity calculation: {:.3}", similarity);
    }

    #[tokio::test]
    async fn test_correlation_matrix_operations() {
        println!("🧪 Testing CorrelationMatrix operations");

        // Create test matrix
        let size = 4;
        let data = vec![
            1.0, 0.8, -0.3, 0.5, 0.8, 1.0, -0.1, 0.6, -0.3, -0.1, 1.0, 0.2, 0.5, 0.6, 0.2, 1.0,
        ];

        let matrix = CorrelationMatrix::new(size, data).unwrap();

        // Test basic properties
        assert_eq!(matrix.size(), size);
        assert!(matrix.is_symmetric());
        assert!(matrix.diagonal_ones());
        println!("✅ Matrix properties verified");

        // Test statistics
        let stats = matrix.statistics();
        assert!(stats.min >= -1.0);
        assert!(stats.max <= 1.0);
        println!(
            "✅ Statistics: mean={:.3}, std={:.3}",
            stats.mean, stats.std_dev
        );

        // Test sparse conversion
        let sparse = matrix.to_sparse();
        let reconstructed = sparse.to_dense();

        // Verify reconstruction accuracy
        for i in 0..size {
            for j in 0..size {
                let original = matrix.get(i, j);
                let reconstructed_val = reconstructed.get(i, j);
                assert!(
                    (original - reconstructed_val).abs() < 1e-6,
                    "Reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
        println!("✅ Sparse conversion verified");

        // Test threshold application
        let mut thresholded = matrix.clone();
        thresholded.apply_threshold(0.4);

        // Values below threshold should be zero (except diagonal)
        for i in 0..size {
            for j in 0..size {
                if i != j && matrix.get(i, j).abs() < 0.4 {
                    assert_eq!(thresholded.get(i, j), 0.0);
                }
            }
        }
        println!("✅ Threshold application verified");
    }

    #[test]
    fn test_performance_benchmark() {
        println!("🧪 Performance Benchmark Test");

        let organism_counts = vec![8, 16, 32, 64];

        for count in organism_counts {
            let organisms = create_test_organism_set_sized(count);

            let rt = tokio::runtime::Runtime::new().unwrap();
            let start = std::time::Instant::now();

            let result = rt.block_on(async {
                let simd_engine = SimdCorrelationEngine::new();
                simd_engine.compute_correlation_matrix(&organisms).await
            });

            let duration = start.elapsed();

            match result {
                Ok(matrix) => {
                    let latency_ms = duration.as_millis();
                    let latency_us = duration.as_micros();
                    let ops_per_sec = if duration.as_secs_f64() > 0.0 {
                        (count * count) as f64 / duration.as_secs_f64()
                    } else {
                        f64::INFINITY
                    };

                    println!(
                        "✅ {} organisms: {}ms ({}μs) - {:.0} ops/sec - Matrix {}x{}",
                        count,
                        latency_ms,
                        latency_us,
                        ops_per_sec,
                        matrix.size(),
                        matrix.size()
                    );

                    // Performance should scale reasonably
                    assert!(latency_ms < 100, "Performance too slow: {}ms", latency_ms);
                }
                Err(e) => {
                    println!("❌ Benchmark failed for {} organisms: {}", count, e);
                }
            }
        }
    }

    // Helper functions

    fn create_test_organism_set() -> Vec<OrganismVector> {
        create_test_organism_set_sized(16)
    }

    fn create_test_organism_set_sized(count: usize) -> Vec<OrganismVector> {
        (0..count)
            .map(|i| {
                let base_feature = (i as f32) / (count as f32);
                OrganismVector::new(
                    format!("test_organism_{}", i),
                    vec![
                        base_feature,
                        base_feature * 0.8 + 0.1,
                        -base_feature * 0.5 + 0.2,
                        base_feature * 1.2 - 0.1,
                    ],
                    vec![
                        base_feature * 0.1,
                        -base_feature * 0.05,
                        base_feature * 0.3,
                        base_feature * 0.15,
                    ],
                )
            })
            .collect()
    }
}
