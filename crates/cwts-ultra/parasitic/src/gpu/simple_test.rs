//! Simple GPU module validation test
//!
//! Basic functionality test to verify the GPU correlation module works

use crate::gpu::*;

#[tokio::test]
async fn test_organism_vector_basic() {
    let organism = OrganismVector::new(
        "test_organism".to_string(),
        vec![1.0, 0.5, -0.2, 0.8],
        vec![0.1, -0.05, 0.3, 0.0],
    );

    // Test basic properties
    assert_eq!(organism.id(), "test_organism");
    assert_eq!(organism.features().len(), 4);
    assert_eq!(organism.performance_history().len(), 4);

    // Test validation
    assert!(organism.validate().is_ok());

    println!("✅ OrganismVector basic functionality verified");
}

#[tokio::test]
async fn test_correlation_matrix_basic() {
    let size = 3;
    let data = vec![1.0, 0.5, -0.2, 0.5, 1.0, 0.8, -0.2, 0.8, 1.0];

    let matrix = CorrelationMatrix::new(size, data).unwrap();

    // Test basic properties
    assert_eq!(matrix.size(), size);
    assert!(matrix.is_symmetric());
    assert!(matrix.diagonal_ones());

    // Test element access
    assert_eq!(matrix.get(0, 1), 0.5);
    assert_eq!(matrix.get(1, 0), 0.5);
    assert_eq!(matrix.get(0, 2), -0.2);

    println!("✅ CorrelationMatrix basic functionality verified");
}

#[tokio::test]
async fn test_simd_correlation_engine() {
    let engine = SimdCorrelationEngine::new();

    // Test engine properties
    assert!(engine.is_available());
    assert!(engine.engine_type().contains("SIMD"));

    let perf_info = engine.get_performance_info();
    assert!(perf_info.max_organisms > 0);
    assert!(perf_info.estimated_latency_micros > 0);

    println!(
        "✅ SimdCorrelationEngine: {} - max_organisms: {}, latency: {}μs",
        engine.engine_type(),
        perf_info.max_organisms,
        perf_info.estimated_latency_micros
    );
}

#[tokio::test]
async fn test_small_correlation_computation() {
    let organisms = vec![
        OrganismVector::new("org1".to_string(), vec![1.0, 0.5], vec![0.1, 0.2]),
        OrganismVector::new("org2".to_string(), vec![0.8, 0.6], vec![0.2, 0.1]),
        OrganismVector::new("org3".to_string(), vec![0.2, 0.9], vec![-0.1, 0.3]),
    ];

    let engine = SimdCorrelationEngine::new();

    let start = std::time::Instant::now();
    let result = engine.compute_correlation_matrix(&organisms).await;
    let duration = start.elapsed();

    match result {
        Ok(matrix) => {
            println!("✅ Correlation computation successful:");
            println!("   - Matrix size: {}x{}", matrix.size(), matrix.size());
            println!("   - Computation time: {}μs", duration.as_micros());

            // Verify matrix properties
            assert_eq!(matrix.size(), organisms.len());
            assert!(matrix.is_symmetric());
            assert!(matrix.diagonal_ones());

            // Test some correlations
            for i in 0..matrix.size() {
                assert_eq!(matrix.get(i, i), 1.0, "Diagonal element should be 1.0");
                for j in 0..matrix.size() {
                    let corr = matrix.get(i, j);
                    assert!(
                        corr >= -1.0 && corr <= 1.0,
                        "Correlation out of range: {}",
                        corr
                    );
                    assert!(!corr.is_nan(), "Correlation should not be NaN");
                }
            }

            println!("✅ Small correlation computation verified");
        }
        Err(e) => {
            panic!("Correlation computation failed: {}", e);
        }
    }
}

#[test]
fn test_gpu_feature_detection() {
    let simd_features = detect_simd_features();

    println!("🔍 SIMD Feature Detection:");
    println!("   - SSE4.2: {}", simd_features.has_sse42);
    println!("   - AVX2: {}", simd_features.has_avx2);
    println!("   - AVX-512F: {}", simd_features.has_avx512f);
    println!("   - FMA: {}", simd_features.has_fma);

    // At minimum should have SSE4.2 on modern x86_64
    #[cfg(target_arch = "x86_64")]
    assert!(
        simd_features.has_sse42,
        "SSE4.2 should be available on x86_64"
    );

    println!("✅ Feature detection completed");
}

#[test]
fn test_gpu_constants() {
    assert!(MAX_ORGANISMS > 0);
    assert!(TARGET_LATENCY_MICROS > 0);
    assert_eq!(SIMD_ALIGNMENT, 64);

    println!("✅ GPU constants verified:");
    println!("   - MAX_ORGANISMS: {}", MAX_ORGANISMS);
    println!("   - TARGET_LATENCY_MICROS: {}", TARGET_LATENCY_MICROS);
    println!("   - SIMD_ALIGNMENT: {}", SIMD_ALIGNMENT);
}
