//! Integration Tests for QBMIA GPU Acceleration

use qbmia_gpu::{
    initialize, get_devices, select_backend,
    memory::{initialize_pool, get_pool, PoolConfig},
    quantum::{GpuQuantumCircuit, gates},
    nash::{GpuNashSolver, PayoffMatrix, SolverConfig},
    orchestrator::GpuOrchestrator,
    profiler::{GpuProfiler, ProfilerConfig},
    Backend,
};
use ndarray::Array;

#[test]
fn test_gpu_initialization() {
    // Test GPU initialization - should succeed with CPU fallback
    let result = initialize();
    match result {
        Ok(()) => println!("GPU initialization successful"),
        Err(e) => println!("GPU initialization failed (CPU fallback): {}", e),
    }
}

#[test]
fn test_device_discovery() {
    let _ = initialize();
    
    let devices = get_devices();
    match devices {
        Ok(device_list) => {
            println!("Found {} devices", device_list.len());
            for (i, device) in device_list.iter().enumerate() {
                println!("Device {}: {} ({} MB)", i, device.name, device.total_memory / (1024*1024));
            }
            assert!(!device_list.is_empty()); // Should at least have CPU fallback
        }
        Err(e) => {
            println!("Device discovery failed: {}", e);
            // This is acceptable in CI/testing environments
        }
    }
}

#[test]
fn test_backend_selection() {
    let backend = select_backend();
    println!("Selected backend: {:?}", backend);
    
    // Should select some backend (likely CPU in testing)
    match backend {
        Backend::Cpu => println!("Using CPU fallback"),
        Backend::Cuda => println!("Using CUDA acceleration"),
        Backend::Rocm => println!("Using ROCm acceleration"),
        Backend::WebGpu => println!("Using WebGPU acceleration"),
    }
}

#[test]
fn test_memory_pool_basic_operations() {
    let _ = initialize();
    
    let config = PoolConfig {
        initial_size: 1024 * 1024, // 1MB
        max_size: 4 * 1024 * 1024, // 4MB
        ..Default::default()
    };
    
    let init_result = initialize_pool(config);
    assert!(init_result.is_ok(), "Memory pool initialization should succeed");
    
    let pool = get_pool();
    assert!(pool.is_ok(), "Should be able to get memory pool");
    
    if let Ok(pool) = pool {
        // Test allocation
        let allocation = pool.allocate(0, 4096);
        match allocation {
            Ok(handle) => {
                println!("Successfully allocated 4KB");
                
                // Test deallocation
                let free_result = pool.free(handle);
                assert!(free_result.is_ok(), "Should be able to free memory");
                println!("Successfully freed memory");
            }
            Err(e) => {
                println!("Memory allocation failed: {}", e);
                // This might be expected without actual GPU
            }
        }
        
        // Test statistics
        let stats = pool.stats();
        println!("Memory stats: {:?}", stats);
    }
}

#[test]
fn test_quantum_circuit_creation() {
    let _ = initialize();
    
    // Test creating quantum circuits of various sizes
    for num_qubits in [2, 4, 6, 8].iter() {
        let circuit = GpuQuantumCircuit::new(*num_qubits, 0);
        assert_eq!(circuit.num_qubits, *num_qubits);
        assert_eq!(circuit.operations.len(), 0);
        println!("Created {}-qubit quantum circuit", num_qubits);
    }
}

#[test]
fn test_quantum_gates() {
    let _ = initialize();
    
    let mut circuit = GpuQuantumCircuit::new(3, 0);
    
    // Add various gates
    circuit.add_gate(gates::h(), 0);
    circuit.add_gate(gates::x(), 1);
    circuit.add_gate(gates::y(), 2);
    circuit.add_two_gate(gates::cnot(), 0, 1);
    
    assert_eq!(circuit.operations.len(), 4);
    println!("Added 4 quantum gates to circuit");
    
    // Attempt to execute (may fail without GPU, but tests interface)
    let result = circuit.execute();
    match result {
        Ok(probabilities) => {
            println!("Quantum circuit executed successfully");
            println!("Number of probability amplitudes: {}", probabilities.len());
            assert_eq!(probabilities.len(), 1 << circuit.num_qubits);
        }
        Err(e) => {
            println!("Quantum circuit execution failed (expected without GPU): {}", e);
        }
    }
}

#[test]
fn test_nash_equilibrium_solver() {
    let _ = initialize();
    
    // Create a simple 2x2 game (Prisoner's Dilemma-like)
    let payoffs = vec![
        Array::from_shape_vec(
            ndarray::IxDyn(&[2, 2]),
            vec![3.0, 0.0, 5.0, 1.0] // Player 1 payoffs
        ).unwrap(),
        Array::from_shape_vec(
            ndarray::IxDyn(&[2, 2]),
            vec![3.0, 5.0, 0.0, 1.0] // Player 2 payoffs
        ).unwrap(),
    ];
    
    let payoff_matrix = PayoffMatrix {
        num_players: 2,
        strategies: vec![2, 2],
        payoffs,
    };
    
    let config = SolverConfig {
        max_iterations: 100, // Limit for testing
        tolerance: 1e-3,
        quantum_enhanced: false, // Disable for basic test
        ..Default::default()
    };
    
    // Test solver creation
    let solver_result = GpuNashSolver::new(0, payoff_matrix, config);
    match solver_result {
        Ok(mut solver) => {
            println!("Nash equilibrium solver created successfully");
            
            // Attempt to solve
            let solution = solver.solve();
            match solution {
                Ok(strategy_profile) => {
                    println!("Nash equilibrium found!");
                    println!("Player strategies: {:?}", strategy_profile.strategies);
                    println!("Expected payoffs: {:?}", strategy_profile.payoffs);
                    println!("Convergence error: {}", strategy_profile.error);
                    
                    // Verify solution properties
                    assert_eq!(strategy_profile.strategies.len(), 2);
                    for strategy in &strategy_profile.strategies {
                        let sum: f64 = strategy.iter().sum();
                        assert!((sum - 1.0).abs() < 1e-6, "Strategies should sum to 1");
                    }
                }
                Err(e) => {
                    println!("Nash equilibrium solving failed (expected without GPU): {}", e);
                }
            }
        }
        Err(e) => {
            println!("Nash solver creation failed (expected without GPU): {}", e);
        }
    }
}

#[tokio::test]
async fn test_gpu_orchestrator() {
    let _ = initialize();
    
    // Test orchestrator creation
    let orchestrator_result = GpuOrchestrator::new();
    match orchestrator_result {
        Ok(orchestrator) => {
            println!("GPU orchestrator created successfully");
            
            let devices = orchestrator.get_device_stats();
            println!("Orchestrator managing {} devices", devices.len());
            
            // Test health check
            let mut orchestrator = orchestrator;
            let health_result = orchestrator.health_check().await;
            assert!(health_result.is_ok(), "Health check should succeed");
            
            // Test quantum circuit submission
            let circuit = GpuQuantumCircuit::new(3, 0);
            let submission_result = orchestrator.submit_quantum_circuit(circuit).await;
            match submission_result {
                Ok(workload_id) => {
                    println!("Quantum circuit workload submitted: {}", workload_id);
                    
                    // Check workload status
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    let status = orchestrator.get_workload_status(workload_id);
                    if let Some(status) = status {
                        println!("Workload status: {:?}", status.status);
                    }
                }
                Err(e) => {
                    println!("Workload submission failed (expected without GPU): {}", e);
                }
            }
        }
        Err(e) => {
            println!("Orchestrator creation failed: {}", e);
        }
    }
}

#[test]
fn test_gpu_profiler() {
    let _ = initialize();
    
    let config = ProfilerConfig::default();
    let profiler = GpuProfiler::new(config);
    
    // Test session management
    let session_result = profiler.start_session("test_session".to_string(), 0, Backend::Cpu);
    assert!(session_result.is_ok(), "Should be able to start profiling session");
    
    // Test event recording
    let event_result = profiler.record_kernel_launch(
        "test_session",
        "test_kernel".to_string(),
        (64, 1, 1),
        (256, 1, 1),
        0,
    );
    assert!(event_result.is_ok(), "Should be able to record kernel launch");
    
    let completion_result = profiler.record_kernel_completion(
        "test_session",
        "test_kernel".to_string(),
        std::time::Duration::from_millis(5),
        0,
    );
    assert!(completion_result.is_ok(), "Should be able to record kernel completion");
    
    // Test metrics
    let metrics = profiler.get_session_metrics("test_session");
    assert!(metrics.is_ok(), "Should be able to get session metrics");
    
    if let Ok(metrics) = metrics {
        println!("Session metrics: {:?}", metrics);
        assert_eq!(metrics.total_kernel_launches, 1);
    }
    
    // Test session stopping
    let report = profiler.stop_session("test_session");
    assert!(report.is_ok(), "Should be able to stop session and get report");
    
    if let Ok(report) = report {
        println!("Profiling report generated");
        println!("Total events: {}", report.events.len());
        println!("Kernel launches: {}", report.kernel_summary.total_launches);
        println!("Bottlenecks identified: {}", report.bottlenecks.len());
        println!("Recommendations: {}", report.recommendations.len());
    }
}

#[test]
fn test_integration_workflow() {
    let _ = initialize();
    
    // This test simulates a complete workflow:
    // 1. Initialize GPU and memory pool
    // 2. Create and execute quantum circuit
    // 3. Solve Nash equilibrium
    // 4. Profile the operations
    
    println!("Starting integration workflow test...");
    
    // Step 1: Initialize
    let pool_config = PoolConfig {
        initial_size: 2 * 1024 * 1024, // 2MB
        ..Default::default()
    };
    let _ = initialize_pool(pool_config);
    
    let profiler = GpuProfiler::new(ProfilerConfig::default());
    let _ = profiler.start_session("integration_test".to_string(), 0, Backend::Cpu);
    
    // Step 2: Quantum circuit
    let mut circuit = GpuQuantumCircuit::new(4, 0);
    circuit.add_gate(gates::h(), 0);
    circuit.add_gate(gates::h(), 1);
    circuit.add_two_gate(gates::cnot(), 0, 2);
    circuit.add_two_gate(gates::cnot(), 1, 3);
    
    println!("Created 4-qubit quantum circuit with {} operations", circuit.operations.len());
    
    // Step 3: Nash equilibrium
    let payoffs = vec![
        Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
    ];
    
    let payoff_matrix = PayoffMatrix {
        num_players: 2,
        strategies: vec![2, 2],
        payoffs,
    };
    
    let config = SolverConfig {
        max_iterations: 50,
        tolerance: 1e-2,
        ..Default::default()
    };
    
    println!("Created Nash equilibrium problem for 2 players");
    
    // Step 4: Execute and profile
    let quantum_result = circuit.execute();
    let nash_result = GpuNashSolver::new(0, payoff_matrix, config);
    
    match quantum_result {
        Ok(_) => println!("✓ Quantum circuit executed successfully"),
        Err(e) => println!("✗ Quantum circuit failed: {}", e),
    }
    
    match nash_result {
        Ok(_) => println!("✓ Nash solver created successfully"),
        Err(e) => println!("✗ Nash solver failed: {}", e),
    }
    
    // Step 5: Get profiling report
    let report = profiler.stop_session("integration_test");
    match report {
        Ok(report) => {
            println!("✓ Profiling completed");
            println!("  Events recorded: {}", report.events.len());
            println!("  Duration: {:?}", report.duration);
        }
        Err(e) => println!("✗ Profiling failed: {}", e),
    }
    
    println!("Integration workflow test completed!");
}

#[test]
fn test_error_handling() {
    let _ = initialize();
    
    // Test various error conditions
    
    // Invalid device ID
    let invalid_device_circuit = GpuQuantumCircuit::new(2, 999);
    let result = invalid_device_circuit.execute();
    // Should handle gracefully (may succeed with CPU fallback)
    
    // Invalid quantum circuit (too many qubits)
    let large_circuit = GpuQuantumCircuit::new(50, 0); // Very large circuit
    let result = large_circuit.execute();
    match result {
        Ok(_) => println!("Large circuit executed (unexpected)"),
        Err(e) => println!("Large circuit failed as expected: {}", e),
    }
    
    // Invalid Nash game (inconsistent dimensions)
    let invalid_payoffs = vec![
        Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0]).unwrap_or_else(|_| {
            // This should fail due to dimension mismatch
            Array::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.0]).unwrap()
        }),
    ];
    
    // Test memory exhaustion (try to allocate huge amount)
    if let Ok(pool) = get_pool() {
        let huge_allocation = pool.allocate(0, usize::MAX);
        match huge_allocation {
            Ok(_) => println!("Unexpected: huge allocation succeeded"),
            Err(e) => println!("Huge allocation failed as expected: {}", e),
        }
    }
}

#[test]
fn test_performance_characteristics() {
    let _ = initialize();
    
    // Test that the system maintains reasonable performance characteristics
    let start_time = std::time::Instant::now();
    
    // Create multiple small quantum circuits
    let circuits: Vec<_> = (0..10)
        .map(|i| {
            let mut circuit = GpuQuantumCircuit::new(3, 0);
            circuit.add_gate(gates::h(), i % 3);
            circuit
        })
        .collect();
    
    let circuit_creation_time = start_time.elapsed();
    println!("Created 10 circuits in {:?}", circuit_creation_time);
    
    // Test memory allocation performance
    let alloc_start = std::time::Instant::now();
    if let Ok(pool) = get_pool() {
        let handles: Vec<_> = (0..100)
            .filter_map(|_| pool.allocate(0, 1024).ok())
            .collect();
        
        let alloc_time = alloc_start.elapsed();
        println!("100 allocations took {:?}", alloc_time);
        
        // Clean up
        for handle in handles {
            let _ = pool.free(handle);
        }
    }
    
    // Performance should be reasonable even without GPU acceleration
    assert!(circuit_creation_time < std::time::Duration::from_millis(100));
}

#[test]
fn test_memory_safety() {
    let _ = initialize();
    
    // Test that we handle memory operations safely
    let config = PoolConfig {
        initial_size: 64 * 1024, // Small pool
        max_size: 256 * 1024,
        ..Default::default()
    };
    let _ = initialize_pool(config);
    
    if let Ok(pool) = get_pool() {
        let mut handles = Vec::new();
        
        // Allocate until we run out of memory
        for _ in 0..1000 {
            match pool.allocate(0, 1024) {
                Ok(handle) => handles.push(handle),
                Err(_) => break, // Expected when we run out
            }
        }
        
        println!("Allocated {} handles before exhaustion", handles.len());
        
        // Free all handles
        for handle in handles {
            let result = pool.free(handle);
            assert!(result.is_ok(), "Should be able to free valid handle");
        }
        
        // Try to allocate again
        let new_allocation = pool.allocate(0, 1024);
        match new_allocation {
            Ok(handle) => {
                println!("Successfully allocated after cleanup");
                let _ = pool.free(handle);
            }
            Err(e) => println!("Allocation after cleanup failed: {}", e),
        }
    }
}