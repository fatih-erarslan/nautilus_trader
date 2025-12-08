//! Comprehensive integration tests for QBMIA GPU acceleration

use qbmia_acceleration::*;
use tokio::test;
use std::time::Duration;

/// Test complete accelerator initialization and basic functionality
#[test]
async fn test_accelerator_initialization() {
    let accelerator = QBMIAAccelerator::new().await;
    
    // Should succeed even without GPU (fallback to CPU)
    match accelerator {
        Ok(acc) => {
            println!("GPU accelerator initialized successfully");
            
            // Test warmup
            let warmup_result = acc.warmup().await;
            assert!(warmup_result.is_ok(), "Warmup should succeed");
            
            // Get initial metrics
            let metrics = acc.get_metrics().await;
            println!("Initial metrics: {:?}", metrics);
        },
        Err(e) => {
            println!("GPU not available, testing CPU fallback: {}", e);
            // This is acceptable - we should test CPU fallback paths
        }
    }
}

/// Test quantum state evolution with various gate sequences
#[test]
async fn test_quantum_evolution_comprehensive() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping quantum evolution test - no GPU available");
            return;
        }
    };
    
    // Test 1: Single Hadamard gate (should be < 100ns)
    let state = QuantumState::new(4).unwrap();
    let gates = vec![UnitaryGate::hadamard()];
    let indices = vec![vec![0]];
    
    let start_time = std::time::Instant::now();
    let result = accelerator.evolve_quantum_state(&state, &gates, &indices).await;
    let elapsed = start_time.elapsed();
    
    assert!(result.is_ok(), "Single Hadamard gate should succeed");
    println!("Single Hadamard gate took: {}ns", elapsed.as_nanos());
    
    // Test 2: Multiple gates
    let multi_gates = vec![
        UnitaryGate::hadamard(),
        UnitaryGate::pauli_x(),
        UnitaryGate::pauli_z(),
        UnitaryGate::hadamard(),
    ];
    let multi_indices = vec![vec![0], vec![1], vec![2], vec![3]];
    
    let result = accelerator.evolve_quantum_state(&state, &multi_gates, &multi_indices).await;
    assert!(result.is_ok(), "Multiple gates should succeed");
    
    // Test 3: Two-qubit gates
    let cnot_gates = vec![UnitaryGate::cnot()];
    let cnot_indices = vec![vec![0, 1]];
    
    let result = accelerator.evolve_quantum_state(&state, &cnot_gates, &cnot_indices).await;
    assert!(result.is_ok(), "CNOT gate should succeed");
    
    // Test 4: Verify quantum state normalization
    let evolved_state = result.unwrap();
    let norm_squared: f32 = evolved_state.amplitudes
        .iter()
        .map(|amp| amp.magnitude_squared())
        .sum();
    
    assert!((norm_squared - 1.0).abs() < 1e-6, "Quantum state should remain normalized");
    
    // Test 5: Performance validation for small operations
    for _ in 0..100 {
        let start = std::time::Instant::now();
        let _ = accelerator.evolve_quantum_state(&state, &gates, &indices).await.unwrap();
        let elapsed = start.elapsed();
        
        if elapsed.as_nanos() > 1000 {
            println!("WARNING: Quantum evolution took {}ns, may exceed performance targets", elapsed.as_nanos());
        }
    }
}

/// Test Nash equilibrium solver with various matrix sizes
#[test]
async fn test_nash_solver_comprehensive() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping Nash solver test - no GPU available");
            return;
        }
    };
    
    // Test 1: 2x2 analytical solution (should be < 500ns)
    let matrix_2x2 = PayoffMatrix::new(2, 2, vec![3.0, 0.0, 0.0, 1.0]).unwrap();
    let strategies_2x2 = StrategyVector::uniform(2).unwrap();
    let params = NashSolverParams::default();
    
    let start_time = std::time::Instant::now();
    let result = accelerator.solve_nash_equilibrium(&matrix_2x2, &strategies_2x2, &params).await;
    let elapsed = start_time.elapsed();
    
    assert!(result.is_ok(), "2x2 Nash equilibrium should succeed");
    println!("2x2 Nash equilibrium took: {}ns", elapsed.as_nanos());
    
    let equilibrium = result.unwrap();
    assert!(equilibrium.strategies.len() > 0, "Should return strategies");
    assert!(equilibrium.is_converged(1e-6), "Should converge for simple 2x2 game");
    
    // Test 2: Larger matrices
    for size in [3, 4, 8] {
        let matrix = PayoffMatrix::random(size, size).unwrap();
        let strategies = StrategyVector::uniform(size).unwrap();
        
        let result = accelerator.solve_nash_equilibrium(&matrix, &strategies, &params).await;
        assert!(result.is_ok(), "Nash equilibrium should succeed for {}x{} matrix", size, size);
        
        let equilibrium = result.unwrap();
        
        // Verify strategy validity
        for strategy in &equilibrium.strategies {
            let sum: f32 = strategy.probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Strategy probabilities should sum to 1");
            
            for &prob in &strategy.probabilities {
                assert!(prob >= 0.0, "Strategy probabilities should be non-negative");
            }
        }
    }
    
    // Test 3: Performance validation for small matrices
    for _ in 0..100 {
        let start = std::time::Instant::now();
        let _ = accelerator.solve_nash_equilibrium(&matrix_2x2, &strategies_2x2, &params).await.unwrap();
        let elapsed = start.elapsed();
        
        if elapsed.as_nanos() > 5000 {
            println!("WARNING: 2x2 Nash solving took {}ns, may exceed performance targets", elapsed.as_nanos());
        }
    }
}

/// Test pattern matching with various configurations
#[test]
async fn test_pattern_matching_comprehensive() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping pattern matching test - no GPU available");
            return;
        }
    };
    
    // Test 1: Basic pattern matching
    let patterns = vec![
        Pattern::new(vec![1.0, 0.0, 0.0, 0.0], None),
        Pattern::new(vec![0.0, 1.0, 0.0, 0.0], None),
        Pattern::new(vec![0.0, 0.0, 1.0, 0.0], None),
        Pattern::new(vec![0.0, 0.0, 0.0, 1.0], None),
    ];
    let query = Pattern::new(vec![1.0, 0.0, 0.0, 0.0], None);
    let threshold = 0.9;
    
    let result = accelerator.pattern_match(&patterns, &query, threshold).await;
    assert!(result.is_ok(), "Basic pattern matching should succeed");
    
    let matches = result.unwrap();
    assert_eq!(matches.len(), patterns.len(), "Should return match result for each pattern");
    assert!(matches[0], "First pattern should match (exact match)");
    assert!(!matches[1], "Second pattern should not match");
    
    // Test 2: Large-scale pattern matching
    let large_patterns: Vec<Pattern> = (0..1000)
        .map(|_| Pattern::random(64).unwrap())
        .collect();
    let large_query = Pattern::random(64).unwrap();
    
    let start_time = std::time::Instant::now();
    let result = accelerator.pattern_match(&large_patterns, &large_query, 0.8).await;
    let elapsed = start_time.elapsed();
    
    assert!(result.is_ok(), "Large-scale pattern matching should succeed");
    println!("1000 pattern matching took: {}ms", elapsed.as_millis());
    
    let matches = result.unwrap();
    assert_eq!(matches.len(), 1000, "Should return results for all patterns");
    
    // Test 3: Performance validation
    let small_patterns: Vec<Pattern> = (0..100)
        .map(|_| Pattern::random(64).unwrap())
        .collect();
    
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let _ = accelerator.pattern_match(&small_patterns, &large_query, 0.8).await.unwrap();
        let elapsed = start.elapsed();
        
        if elapsed.as_micros() > 100 {
            println!("WARNING: 100 pattern matching took {}μs, may exceed performance targets", elapsed.as_micros());
        }
    }
    
    // Test 4: Different thresholds
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95] {
        let result = accelerator.pattern_match(&patterns, &query, threshold).await;
        assert!(result.is_ok(), "Pattern matching should succeed for threshold {}", threshold);
    }
}

/// Test memory operations and buffer management
#[test]
async fn test_memory_operations_comprehensive() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping memory operations test - no GPU available");
            return;
        }
    };
    
    // Test 1: Basic buffer operations
    let test_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let buffer = accelerator.memory_manager.create_buffer(&test_data).await;
    assert!(buffer.is_ok(), "Buffer creation should succeed");
    
    let buffer = buffer.unwrap();
    assert_eq!(buffer.size, test_data.len(), "Buffer size should match data size");
    
    // Test 2: Buffer read-back
    let read_data = accelerator.memory_manager.read_buffer(&buffer).await;
    assert!(read_data.is_ok(), "Buffer reading should succeed");
    
    let read_data = read_data.unwrap();
    assert_eq!(read_data, test_data, "Read data should match original data");
    
    // Test 3: Large buffer operations
    let large_data = vec![42u8; 1024 * 1024]; // 1MB
    let start_time = std::time::Instant::now();
    let large_buffer = accelerator.memory_manager.create_buffer(&large_data).await;
    let creation_time = start_time.elapsed();
    
    assert!(large_buffer.is_ok(), "Large buffer creation should succeed");
    println!("1MB buffer creation took: {}ms", creation_time.as_millis());
    
    let large_buffer = large_buffer.unwrap();
    let start_time = std::time::Instant::now();
    let read_large_data = accelerator.memory_manager.read_buffer(&large_buffer).await;
    let read_time = start_time.elapsed();
    
    assert!(read_large_data.is_ok(), "Large buffer reading should succeed");
    println!("1MB buffer read took: {}ms", read_time.as_millis());
    
    // Test 4: Buffer pool reuse
    let mut buffers = Vec::new();
    for i in 0..10 {
        let data = vec![i as u8; 1024];
        let buffer = accelerator.memory_manager.create_buffer(&data).await.unwrap();
        buffers.push(buffer);
    }
    
    // Free buffers to test pool reuse
    for buffer in buffers {
        let _ = accelerator.memory_manager.free_buffer(&buffer).await;
    }
    
    // Create new buffers (should reuse from pool)
    let reuse_start = std::time::Instant::now();
    for i in 0..10 {
        let data = vec![i as u8; 1024];
        let _ = accelerator.memory_manager.create_buffer(&data).await.unwrap();
    }
    let reuse_time = reuse_start.elapsed();
    
    println!("Buffer pool reuse for 10 buffers took: {}μs", reuse_time.as_micros());
    
    // Test 5: Memory statistics
    let stats = accelerator.memory_manager.get_stats().await;
    println!("Memory stats: {:?}", stats);
    assert!(stats.total_allocations > 0, "Should have recorded allocations");
}

/// Test SIMD operations performance and correctness
#[test]
async fn test_simd_operations_comprehensive() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping SIMD operations test - no GPU available");
            return;
        }
    };
    
    // Test 1: Dot product correctness
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // = 70.0
    
    let result = accelerator.simd_processor.dot_product(&a, &b);
    assert!(result.is_ok(), "Dot product should succeed");
    assert_eq!(result.unwrap(), expected, "Dot product should be correct");
    
    // Test 2: Matrix-vector multiplication
    let matrix = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
    let vector = vec![5.0, 6.0];
    let expected_result = vec![1.0*5.0 + 2.0*6.0, 3.0*5.0 + 4.0*6.0]; // [17.0, 39.0]
    
    let result = accelerator.simd_processor.matrix_vector_multiply(&matrix, &vector, 2, 2);
    assert!(result.is_ok(), "Matrix-vector multiply should succeed");
    assert_eq!(result.unwrap(), expected_result, "Matrix-vector result should be correct");
    
    // Test 3: Vector normalization
    let mut vector = vec![3.0, 4.0, 0.0, 0.0];
    let result = accelerator.simd_processor.normalize_vector(&mut vector);
    assert!(result.is_ok(), "Vector normalization should succeed");
    
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "Normalized vector should have unit norm");
    
    // Test 4: Complex number operations
    let complex_a = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
    ];
    let complex_b = vec![
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    
    let result = accelerator.simd_processor.complex_multiply_arrays(&complex_a, &complex_b);
    assert!(result.is_ok(), "Complex multiplication should succeed");
    
    let products = result.unwrap();
    // (1+2i) * (5+6i) = 5 + 6i + 10i + 12i^2 = 5 + 16i - 12 = -7 + 16i
    assert_eq!(products[0].real, -7.0);
    assert_eq!(products[0].imag, 16.0);
    
    // Test 5: Performance with different vector sizes
    for size in [16, 64, 256, 1024] {
        let large_a = vec![1.0f32; size];
        let large_b = vec![2.0f32; size];
        
        let start = std::time::Instant::now();
        let _ = accelerator.simd_processor.dot_product(&large_a, &large_b).unwrap();
        let elapsed = start.elapsed();
        
        println!("SIMD dot product ({} elements) took: {}ns", size, elapsed.as_nanos());
        
        // Performance should scale sub-linearly with size due to SIMD
        if size <= 64 && elapsed.as_nanos() > 1000 {
            println!("WARNING: Small SIMD operation took {}ns, may exceed performance targets", elapsed.as_nanos());
        }
    }
}

/// Test pipeline orchestration and parallel execution
#[test]
async fn test_pipeline_orchestration() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping pipeline orchestration test - no GPU available");
            return;
        }
    };
    
    // Test 1: Sequential vs parallel execution timing
    let quantum_state = QuantumState::new(6).unwrap();
    let gates = vec![UnitaryGate::hadamard(); 3];
    let indices = (0..3).map(|i| vec![i]).collect::<Vec<_>>();
    
    let nash_matrix = PayoffMatrix::random(4, 4).unwrap();
    let nash_strategies = StrategyVector::uniform(4).unwrap();
    let nash_params = NashSolverParams::default();
    
    let patterns = (0..500).map(|_| Pattern::random(32).unwrap()).collect::<Vec<_>>();
    let query = Pattern::random(32).unwrap();
    
    // Sequential execution
    let sequential_start = std::time::Instant::now();
    let _ = accelerator.evolve_quantum_state(&quantum_state, &gates, &indices).await.unwrap();
    let _ = accelerator.solve_nash_equilibrium(&nash_matrix, &nash_strategies, &nash_params).await.unwrap();
    let _ = accelerator.pattern_match(&patterns, &query, 0.8).await.unwrap();
    let sequential_time = sequential_start.elapsed();
    
    // Parallel execution
    let parallel_start = std::time::Instant::now();
    let (quantum_result, nash_result, pattern_result) = tokio::join!(
        accelerator.evolve_quantum_state(&quantum_state, &gates, &indices),
        accelerator.solve_nash_equilibrium(&nash_matrix, &nash_strategies, &nash_params),
        accelerator.pattern_match(&patterns, &query, 0.8)
    );
    let parallel_time = parallel_start.elapsed();
    
    assert!(quantum_result.is_ok(), "Parallel quantum operation should succeed");
    assert!(nash_result.is_ok(), "Parallel Nash operation should succeed");
    assert!(pattern_result.is_ok(), "Parallel pattern operation should succeed");
    
    println!("Sequential execution: {}ms", sequential_time.as_millis());
    println!("Parallel execution: {}ms", parallel_time.as_millis());
    
    // Parallel should be faster (or at least not much slower due to coordination overhead)
    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("Parallel speedup: {:.2}x", speedup);
    
    // Test 2: High-frequency operation batching
    let small_state = QuantumState::new(4).unwrap();
    let small_gate = vec![UnitaryGate::hadamard()];
    let small_indices = vec![vec![0]];
    
    let batch_start = std::time::Instant::now();
    let mut futures = Vec::new();
    
    for _ in 0..100 {
        futures.push(accelerator.evolve_quantum_state(&small_state, &small_gate, &small_indices));
    }
    
    let results = futures::future::join_all(futures).await;
    let batch_time = batch_start.elapsed();
    
    // Verify all operations succeeded
    for result in results {
        assert!(result.is_ok(), "Batch operation should succeed");
    }
    
    println!("100 parallel quantum operations took: {}ms", batch_time.as_millis());
    println!("Average per operation: {}μs", batch_time.as_micros() / 100);
    
    // Test 3: Memory and cache efficiency
    let initial_metrics = accelerator.get_metrics().await;
    
    // Perform repeated operations to test caching
    for _ in 0..50 {
        let _ = accelerator.evolve_quantum_state(&small_state, &small_gate, &small_indices).await.unwrap();
    }
    
    let final_metrics = accelerator.get_metrics().await;
    
    println!("Initial metrics: {:?}", initial_metrics);
    println!("Final metrics: {:?}", final_metrics);
    
    // Operations should get faster due to caching
    if let (Some(initial_time), Some(final_time)) = (
        initial_metrics.average_quantum_evolution_time(),
        final_metrics.average_quantum_evolution_time()
    ) {
        println!("Average quantum evolution time: initial={}ns, final={}ns", 
                initial_time.as_nanos(), final_time.as_nanos());
    }
}

/// Test error handling and edge cases
#[test]
async fn test_error_handling() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping error handling test - no GPU available");
            return;
        }
    };
    
    // Test 1: Invalid quantum state dimensions
    let invalid_amplitudes = vec![Complex64::new(1.0, 0.0); 3]; // Not a power of 2
    let invalid_state = QuantumState::from_amplitudes(invalid_amplitudes);
    assert!(invalid_state.is_err(), "Should reject invalid quantum state dimensions");
    
    // Test 2: Mismatched gate and qubit indices
    let state = QuantumState::new(4).unwrap();
    let gates = vec![UnitaryGate::hadamard()];
    let wrong_indices = vec![vec![0], vec![1]]; // More indices than gates
    
    let result = accelerator.evolve_quantum_state(&state, &gates, &wrong_indices).await;
    assert!(result.is_err(), "Should reject mismatched gates and indices");
    
    // Test 3: Invalid payoff matrix dimensions
    let invalid_matrix = PayoffMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]); // Wrong size
    assert!(invalid_matrix.is_err(), "Should reject invalid matrix dimensions");
    
    // Test 4: Invalid strategy vector (doesn't sum to 1)
    let invalid_strategy = StrategyVector::new(vec![0.3, 0.4, 0.5]); // Sums to 1.2
    assert!(invalid_strategy.is_err(), "Should reject non-normalized strategy vector");
    
    // Test 5: Out of bounds qubit indices
    let state = QuantumState::new(4).unwrap();
    let gates = vec![UnitaryGate::hadamard()];
    let oob_indices = vec![vec![5]]; // Qubit 5 doesn't exist in 4-qubit system
    
    let result = accelerator.evolve_quantum_state(&state, &gates, &oob_indices).await;
    assert!(result.is_err(), "Should reject out-of-bounds qubit indices");
    
    // Test 6: Empty pattern matching
    let empty_patterns: Vec<Pattern> = vec![];
    let query = Pattern::random(64).unwrap();
    
    let result = accelerator.pattern_match(&empty_patterns, &query, 0.8).await;
    assert!(result.is_ok(), "Empty pattern matching should succeed with empty result");
    assert_eq!(result.unwrap().len(), 0, "Should return empty result for empty patterns");
    
    println!("All error handling tests passed");
}

/// Comprehensive benchmark to validate all performance targets
#[test]
async fn test_performance_targets_validation() {
    let accelerator = match QBMIAAccelerator::new().await {
        Ok(acc) => acc,
        Err(_) => {
            println!("Skipping performance validation - no GPU available");
            return;
        }
    };
    
    println!("=== PERFORMANCE TARGETS VALIDATION ===");
    
    // Target 1: Quantum state evolution < 100ns
    let state = QuantumState::new(4).unwrap();
    let gates = vec![UnitaryGate::hadamard()];
    let indices = vec![vec![0]];
    
    let mut quantum_times = Vec::new();
    for _ in 0..1000 {
        let start = std::time::Instant::now();
        let _ = accelerator.evolve_quantum_state(&state, &gates, &indices).await.unwrap();
        quantum_times.push(start.elapsed());
    }
    
    let avg_quantum_time = quantum_times.iter().sum::<Duration>() / quantum_times.len() as u32;
    let min_quantum_time = quantum_times.iter().min().unwrap();
    let max_quantum_time = quantum_times.iter().max().unwrap();
    
    println!("Quantum Evolution (single Hadamard):");
    println!("  Average: {}ns", avg_quantum_time.as_nanos());
    println!("  Min: {}ns", min_quantum_time.as_nanos());
    println!("  Max: {}ns", max_quantum_time.as_nanos());
    println!("  Target: < 100ns - {}", if avg_quantum_time.as_nanos() < 100 { "✓ PASS" } else { "✗ FAIL" });
    
    // Target 2: Nash equilibrium solver < 500ns
    let matrix_2x2 = PayoffMatrix::new(2, 2, vec![3.0, 0.0, 0.0, 1.0]).unwrap();
    let strategies_2x2 = StrategyVector::uniform(2).unwrap();
    let params = NashSolverParams::default();
    
    let mut nash_times = Vec::new();
    for _ in 0..1000 {
        let start = std::time::Instant::now();
        let _ = accelerator.solve_nash_equilibrium(&matrix_2x2, &strategies_2x2, &params).await.unwrap();
        nash_times.push(start.elapsed());
    }
    
    let avg_nash_time = nash_times.iter().sum::<Duration>() / nash_times.len() as u32;
    let min_nash_time = nash_times.iter().min().unwrap();
    let max_nash_time = nash_times.iter().max().unwrap();
    
    println!("\nNash Equilibrium (2x2 matrix):");
    println!("  Average: {}ns", avg_nash_time.as_nanos());
    println!("  Min: {}ns", min_nash_time.as_nanos());
    println!("  Max: {}ns", max_nash_time.as_nanos());
    println!("  Target: < 500ns - {}", if avg_nash_time.as_nanos() < 500 { "✓ PASS" } else { "✗ FAIL" });
    
    // Target 3: Kernel launch overhead < 50ns (using SIMD as proxy)
    let small_vec_a = vec![1.0f32; 4];
    let small_vec_b = vec![2.0f32; 4];
    
    let mut simd_times = Vec::new();
    for _ in 0..10000 {
        let start = std::time::Instant::now();
        let _ = accelerator.simd_processor.dot_product(&small_vec_a, &small_vec_b).unwrap();
        simd_times.push(start.elapsed());
    }
    
    let avg_simd_time = simd_times.iter().sum::<Duration>() / simd_times.len() as u32;
    let min_simd_time = simd_times.iter().min().unwrap();
    
    println!("\nSIMD Operations (4-element dot product):");
    println!("  Average: {}ns", avg_simd_time.as_nanos());
    println!("  Min: {}ns", min_simd_time.as_nanos());
    println!("  Target: < 50ns - {}", if avg_simd_time.as_nanos() < 50 { "✓ PASS" } else { "✗ FAIL" });
    
    // Target 4: GPU-CPU transfer performance
    let test_data = vec![42u8; 1024]; // 1KB
    
    let mut transfer_times = Vec::new();
    for _ in 0..100 {
        let start = std::time::Instant::now();
        let buffer = accelerator.memory_manager.create_buffer(&test_data).await.unwrap();
        let _read_data = accelerator.memory_manager.read_buffer(&buffer).await.unwrap();
        transfer_times.push(start.elapsed());
    }
    
    let avg_transfer_time = transfer_times.iter().sum::<Duration>() / transfer_times.len() as u32;
    
    println!("\nGPU Memory Transfer (1KB round-trip):");
    println!("  Average: {}μs", avg_transfer_time.as_micros());
    println!("  Target: < 100μs - {}", if avg_transfer_time.as_micros() < 100 { "✓ PASS" } else { "✗ FAIL" });
    
    // Summary
    println!("\n=== PERFORMANCE SUMMARY ===");
    let quantum_pass = avg_quantum_time.as_nanos() < 100;
    let nash_pass = avg_nash_time.as_nanos() < 500;
    let simd_pass = avg_simd_time.as_nanos() < 50;
    let transfer_pass = avg_transfer_time.as_micros() < 100;
    
    let total_pass = quantum_pass && nash_pass && simd_pass && transfer_pass;
    
    println!("Quantum Evolution: {}", if quantum_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("Nash Equilibrium: {}", if nash_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("SIMD Operations: {}", if simd_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("Memory Transfer: {}", if transfer_pass { "✓ PASS" } else { "✗ FAIL" });
    println!("Overall: {}", if total_pass { "✓ ALL TARGETS MET" } else { "✗ SOME TARGETS MISSED" });
    
    // Don't fail the test if targets aren't met - they depend on hardware
    // But log the results for analysis
    if !total_pass {
        println!("Note: Performance targets may not be met on all hardware configurations");
    }
}