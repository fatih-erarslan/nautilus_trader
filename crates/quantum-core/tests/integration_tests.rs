//! Integration tests for quantum-core crate

use quantum_core::*;
use quantum_core::quantum_circuits::*;
use quantum_core::quantum_device::*;
use quantum_core::hardware::*;
use quantum_core::memory::*;
use quantum_core::metrics::*;
use quantum_core::utils::*;
use tokio_test;
use std::time::Duration;

#[tokio::test]
async fn test_complete_quantum_workflow() {
    // Test the complete quantum workflow from circuit creation to execution
    
    // 1. Initialize quantum core
    assert!(initialize().is_ok());
    
    // 2. Create quantum circuit
    let mut circuit = CircuitBuilder::new("test_circuit".to_string(), 3)
        .hadamard(0).unwrap()
        .cnot(0, 1).unwrap()
        .cnot(1, 2).unwrap()
        .rotation_x(0, std::f64::consts::PI / 4.0).unwrap()
        .build();
    
    // 3. Validate circuit
    assert!(circuit.validate().is_ok());
    
    // 4. Optimize circuit
    assert!(circuit.optimize(OptimizationLevel::Basic).is_ok());
    
    // 5. Calculate circuit statistics
    let stats = circuit.calculate_stats().unwrap();
    assert!(stats.total_gates > 0);
    assert!(stats.depth > 0);
    
    // 6. Execute circuit
    let config = ExecutionConfig::default();
    let executor = CircuitExecution::new(config);
    let result = executor.execute(&mut circuit).await;
    
    assert!(result.is_ok());
    let execution_result = result.unwrap();
    assert!(execution_result.success);
    assert!(execution_result.measurement_results.is_some());
}

#[tokio::test]
async fn test_quantum_device_management() {
    // Test quantum device management
    
    // 1. Create device manager
    let manager = DeviceManager::new();
    
    // 2. Create quantum device
    let device = QuantumDevice::new(
        "Test Quantum Device".to_string(),
        DeviceType::Simulator,
        DeviceBackend::CustomSimulator,
        DeviceConfig::default(),
    );
    
    // 3. Register device
    assert!(manager.register_device(device).await.is_ok());
    
    // 4. List devices
    let devices = manager.list_devices().await;
    assert_eq!(devices.len(), 1);
    
    // 5. Create and execute circuit on device
    let mut circuit = CircuitBuilder::new("device_test".to_string(), 2)
        .hadamard(0).unwrap()
        .cnot(0, 1).unwrap()
        .build();
    
    let result = manager.execute_circuit(&mut circuit).await;
    assert!(result.is_ok());
    
    // 6. Get device health report
    let health_report = manager.get_health_report().await;
    assert!(!health_report.is_empty());
}

#[tokio::test]
async fn test_hardware_management() {
    // Test hardware management
    
    // 1. Create hardware manager
    let manager = HardwareManager::new().unwrap();
    
    // 2. Discover devices
    assert!(manager.discover_devices().await.is_ok());
    
    // 3. List devices
    let devices = manager.list_devices();
    assert!(!devices.is_empty());
    
    // 4. Find best device for operation
    let best_device = manager.find_best_device("quantum_simulation");
    assert!(best_device.is_some());
    
    // 5. Get system capabilities
    let capabilities = manager.get_system_capabilities();
    assert!(capabilities.total_devices > 0);
    
    // 6. Get health report
    let health_report = manager.get_health_report();
    assert!(!health_report.is_empty());
}

#[tokio::test]
async fn test_memory_management() {
    // Test quantum memory management
    
    // 1. Create memory manager
    let manager = QuantumMemoryManager::new().unwrap();
    
    // 2. Allocate quantum state
    let state_id = manager.allocate_quantum_state(3).await.unwrap();
    
    // 3. Create and store quantum state
    let state = QuantumState::new(3).unwrap();
    assert!(manager.store_quantum_state("test_state".to_string(), state).await.is_ok());
    
    // 4. Retrieve quantum state
    let retrieved_state = manager.retrieve_quantum_state("test_state").await;
    assert!(retrieved_state.is_some());
    
    // 5. Get memory statistics
    let stats = manager.get_global_stats().await;
    assert!(!stats.is_empty());
    
    // 6. Run garbage collection
    assert!(manager.global_garbage_collect().await.is_ok());
    
    // 7. Deallocate memory
    assert!(manager.deallocate(&state_id).await.is_ok());
}

#[tokio::test]
async fn test_metrics_collection() {
    // Test metrics collection and monitoring
    
    // 1. Create metrics collector
    let collector = QuantumMetricsCollector::new(Duration::seconds(1));
    
    // 2. Record quantum metrics
    assert!(collector.record_quantum_fidelity(0.99).is_ok());
    assert!(collector.record_quantum_error_rate(0.01).is_ok());
    assert!(collector.record_execution_time(150.0).is_ok());
    assert!(collector.record_memory_usage(512.0).is_ok());
    assert!(collector.record_throughput(200.0).is_ok());
    
    // 3. Get current metrics
    let quantum_metrics = collector.get_quantum_metrics();
    assert_eq!(quantum_metrics.fidelity, 0.99);
    assert_eq!(quantum_metrics.error_rate, 0.01);
    
    let performance_metrics = collector.get_performance_metrics();
    assert_eq!(performance_metrics.execution_time_ms, 150.0);
    assert_eq!(performance_metrics.memory_usage_mb, 512.0);
    assert_eq!(performance_metrics.throughput_ops_per_second, 200.0);
    
    // 4. Get metric aggregations
    let aggregation = collector.get_aggregation(
        "quantum_fidelity",
        AggregationMethod::Average,
        TimeWindow::Hour,
    );
    assert!(aggregation.is_some());
    
    // 5. Generate metrics report
    let report = collector.generate_report(TimeWindow::Hour);
    assert!(report.is_ok());
    
    // 6. Test alerting (trigger alert with low fidelity)
    assert!(collector.record_quantum_fidelity(0.85).is_ok());
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let alerts = collector.get_active_alerts().await;
    assert!(!alerts.is_empty());
}

#[test]
fn test_quantum_utilities() {
    // Test quantum utility functions
    
    // 1. Test quantum state generation
    let bell_state = quantum_utils::bell_state(2).unwrap();
    assert_eq!(bell_state.num_qubits(), 2);
    assert!(quantum_utils::is_normalized(&bell_state).unwrap());
    
    let ghz_state = quantum_utils::ghz_state(3).unwrap();
    assert_eq!(ghz_state.num_qubits(), 3);
    assert!(quantum_utils::is_normalized(&ghz_state).unwrap());
    
    let w_state = quantum_utils::w_state(3).unwrap();
    assert_eq!(w_state.num_qubits(), 3);
    assert!(quantum_utils::is_normalized(&w_state).unwrap());
    
    // 2. Test fidelity calculation
    let state1 = quantum_utils::bell_state(2).unwrap();
    let state2 = quantum_utils::bell_state(2).unwrap();
    let fidelity = quantum_utils::calculate_fidelity(&state1, &state2).unwrap();
    assert!((fidelity - 1.0).abs() < 1e-10);
    
    // 3. Test entropy calculation
    let entropy = quantum_utils::calculate_entropy(&bell_state).unwrap();
    assert!(entropy > 0.0);
    
    // 4. Test density matrix
    let density_matrix = quantum_utils::to_density_matrix(&bell_state).unwrap();
    assert_eq!(density_matrix.len(), 4);
    
    let trace = quantum_utils::trace_density_matrix(&density_matrix);
    assert!((trace.re - 1.0).abs() < 1e-10);
}

#[test]
fn test_mathematical_utilities() {
    // Test mathematical utility functions
    
    // 1. Test basic math functions
    assert_eq!(math_utils::factorial(5), 120);
    assert_eq!(math_utils::binomial_coefficient(5, 2), 10);
    assert_eq!(math_utils::fibonacci(10), 55);
    assert_eq!(math_utils::gcd(48, 18), 6);
    assert_eq!(math_utils::lcm(12, 18), 36);
    
    // 2. Test prime functions
    assert!(math_utils::is_prime(17));
    assert!(!math_utils::is_prime(15));
    assert_eq!(math_utils::next_prime(16), 17);
    
    // 3. Test modular arithmetic
    assert_eq!(math_utils::mod_exp(2, 10, 1000), 24);
    
    // 4. Test matrix operations
    let matrix = [[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(math_utils::det_2x2(&matrix), -2.0);
    
    let inv = math_utils::inv_2x2(&matrix);
    assert!(inv.is_some());
    
    // 5. Test utility functions
    assert_eq!(math_utils::lerp(0.0, 10.0, 0.5), 5.0);
    assert_eq!(math_utils::clamp(15.0, 0.0, 10.0), 10.0);
    
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let distance = math_utils::euclidean_distance(&a, &b);
    assert!((distance - 27.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_conversion_utilities() {
    // Test conversion utility functions
    
    // 1. Test angle conversions
    let degrees = 90.0;
    let radians = conversion_utils::deg_to_rad(degrees);
    assert!((radians - std::f64::consts::PI / 2.0).abs() < 1e-10);
    
    let back_to_degrees = conversion_utils::rad_to_deg(radians);
    assert!((back_to_degrees - degrees).abs() < 1e-10);
    
    // 2. Test complex number conversions
    let complex = num_complex::Complex64::new(1.0, 1.0);
    let (magnitude, phase) = conversion_utils::complex_to_polar(complex);
    let back_to_complex = conversion_utils::polar_to_complex(magnitude, phase);
    assert!((complex - back_to_complex).norm() < 1e-10);
    
    // 3. Test binary conversions
    let binary = "1010";
    let decimal = conversion_utils::binary_to_decimal(binary).unwrap();
    assert_eq!(decimal, 10);
    
    let back_to_binary = conversion_utils::decimal_to_binary(decimal, 4);
    assert_eq!(back_to_binary, binary);
    
    // 4. Test state to probabilities
    let state = testing_utils::create_test_bell_state();
    let probabilities = conversion_utils::state_to_probabilities(&state).unwrap();
    assert_eq!(probabilities.len(), 4);
    assert!((probabilities[0] - 0.5).abs() < 1e-10);
    assert!((probabilities[3] - 0.5).abs() < 1e-10);
}

#[test]
fn test_validation_utilities() {
    // Test validation utility functions
    
    // 1. Test quantum state validation
    let state = testing_utils::create_test_state(2);
    assert!(validation_utils::validate_quantum_state(&state).is_ok());
    
    // 2. Test qubit index validation
    assert!(validation_utils::validate_qubit_index(0, 2).is_ok());
    assert!(validation_utils::validate_qubit_index(2, 2).is_err());
    
    // 3. Test probability validation
    assert!(validation_utils::validate_probability(0.5).is_ok());
    assert!(validation_utils::validate_probability(1.5).is_err());
    assert!(validation_utils::validate_probability(-0.1).is_err());
    
    // 4. Test angle validation
    assert!(validation_utils::validate_angle(std::f64::consts::PI).is_ok());
    assert!(validation_utils::validate_angle(f64::NAN).is_err());
    assert!(validation_utils::validate_angle(f64::INFINITY).is_err());
    
    // 5. Test matrix validation
    let matrix = vec![
        vec![num_complex::Complex64::new(1.0, 0.0), num_complex::Complex64::new(0.0, 1.0)],
        vec![num_complex::Complex64::new(0.0, -1.0), num_complex::Complex64::new(1.0, 0.0)],
    ];
    assert!(validation_utils::validate_matrix_dimensions(&matrix).is_ok());
    
    let invalid_matrix = vec![
        vec![num_complex::Complex64::new(1.0, 0.0)],
        vec![num_complex::Complex64::new(0.0, 1.0), num_complex::Complex64::new(1.0, 0.0)],
    ];
    assert!(validation_utils::validate_matrix_dimensions(&invalid_matrix).is_err());
}

#[test]
fn test_performance_utilities() {
    // Test performance utility functions
    
    // 1. Test timer
    let timer = performance_utils::Timer::start();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let duration = timer.elapsed();
    assert!(duration.as_millis() >= 10);
    
    // 2. Test memoization cache
    let mut cache = performance_utils::MemoCache::new(100);
    
    let result1 = cache.get_or_compute("key1", || 42);
    assert_eq!(result1, 42);
    
    let result2 = cache.get_or_compute("key1", || 99);
    assert_eq!(result2, 42); // Should return cached value
    
    // 3. Test parallel operations
    let mut amplitudes = vec![num_complex::Complex64::new(1.0, 0.0); 100];
    
    performance_utils::parallel_amplitude_operation(&mut amplitudes, |amp| {
        *amp *= 2.0;
    }).unwrap();
    
    for amp in &amplitudes {
        assert_eq!(amp.re, 2.0);
        assert_eq!(amp.im, 0.0);
    }
    
    // 4. Test parallel dot product
    let a = vec![num_complex::Complex64::new(1.0, 0.0); 100];
    let b = vec![num_complex::Complex64::new(2.0, 0.0); 100];
    let dot_product = performance_utils::parallel_dot_product(&a, &b);
    assert_eq!(dot_product.re, 200.0);
    assert_eq!(dot_product.im, 0.0);
}

#[test]
fn test_quantum_config() {
    // Test quantum configuration
    
    // 1. Test default configuration
    let config = QuantumConfig::default();
    assert_eq!(config.max_qubits, 32);
    assert_eq!(config.measurement_shots, 1024);
    assert!(config.enable_parallel);
    
    // 2. Test configuration builder
    let custom_config = QuantumConfig::new()
        .with_precision(1e-12)
        .with_max_qubits(16)
        .with_parallel(false)
        .with_cache_size(500)
        .with_optimization_level(1)
        .with_measurement_shots(2048)
        .with_random_seed(12345);
    
    assert_eq!(custom_config.precision, 1e-12);
    assert_eq!(custom_config.max_qubits, 16);
    assert!(!custom_config.enable_parallel);
    assert_eq!(custom_config.cache_size, 500);
    assert_eq!(custom_config.optimization_level, 1);
    assert_eq!(custom_config.measurement_shots, 2048);
    assert_eq!(custom_config.random_seed, Some(12345));
    
    // 3. Test configuration validation
    assert!(custom_config.validate().is_ok());
    
    let invalid_config = QuantumConfig::new()
        .with_precision(-1.0);
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_testing_utilities() {
    // Test testing utility functions
    
    // 1. Test test state creation
    let state = testing_utils::create_test_state(2);
    assert_eq!(state.num_qubits(), 2);
    
    let bell_state = testing_utils::create_test_bell_state();
    assert_eq!(bell_state.num_qubits(), 2);
    
    let random_state = testing_utils::create_test_random_state(3);
    assert_eq!(random_state.num_qubits(), 3);
    
    // 2. Test state comparison
    let state1 = testing_utils::create_test_state(2);
    let state2 = testing_utils::create_test_state(2);
    assert!(testing_utils::assert_states_approx_equal(&state1, &state2, 1e-10).is_ok());
    
    // 3. Test probability comparison
    assert!(testing_utils::assert_prob_approx_equal(0.5, 0.5, 1e-10).is_ok());
    assert!(testing_utils::assert_prob_approx_equal(0.5, 0.6, 1e-10).is_err());
    
    // 4. Test angle generation
    let angles = testing_utils::generate_test_angles(4);
    assert_eq!(angles.len(), 4);
    assert_eq!(angles[0], 0.0);
    assert!((angles[1] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    
    // 5. Test benchmarking
    let (result, duration) = testing_utils::benchmark_function(|| {
        math_utils::factorial(10)
    }, 1000);
    
    assert_eq!(result, 3628800);
    assert!(duration.as_nanos() > 0);
}

#[tokio::test]
async fn test_complete_integration() {
    // Test complete integration of all components
    
    // 1. Initialize quantum core
    assert!(initialize().is_ok());
    
    // 2. Create hardware manager and discover devices
    let hardware_manager = HardwareManager::new().unwrap();
    assert!(hardware_manager.discover_devices().await.is_ok());
    
    // 3. Create memory manager
    let memory_manager = QuantumMemoryManager::new().unwrap();
    
    // 4. Create metrics collector
    let metrics_collector = QuantumMetricsCollector::new(Duration::seconds(1));
    
    // 5. Create quantum device manager
    let device_manager = DeviceManager::new();
    
    // 6. Create and register quantum device
    let device = QuantumDevice::new(
        "Integration Test Device".to_string(),
        DeviceType::Simulator,
        DeviceBackend::CustomSimulator,
        DeviceConfig::default(),
    );
    assert!(device_manager.register_device(device).await.is_ok());
    
    // 7. Create complex quantum circuit
    let mut circuit = CircuitBuilder::new("integration_test".to_string(), 4)
        .hadamard(0).unwrap()
        .hadamard(1).unwrap()
        .cnot(0, 1).unwrap()
        .cnot(1, 2).unwrap()
        .cnot(2, 3).unwrap()
        .rotation_x(0, std::f64::consts::PI / 4.0).unwrap()
        .rotation_y(1, std::f64::consts::PI / 3.0).unwrap()
        .rotation_z(2, std::f64::consts::PI / 6.0).unwrap()
        .phase(3, std::f64::consts::PI / 8.0).unwrap()
        .build();
    
    // 8. Optimize circuit
    assert!(circuit.optimize(OptimizationLevel::Aggressive).is_ok());
    
    // 9. Allocate memory for quantum state
    let state_id = memory_manager.allocate_quantum_state(4).await.unwrap();
    
    // 10. Execute circuit on device
    let execution_result = device_manager.execute_circuit(&mut circuit).await;
    assert!(execution_result.is_ok());
    
    let result = execution_result.unwrap();
    assert!(result.success);
    
    // 11. Record metrics
    assert!(metrics_collector.record_quantum_fidelity(result.fidelity).is_ok());
    assert!(metrics_collector.record_execution_time(result.execution_time_ns as f64 / 1_000_000.0).is_ok());
    assert!(metrics_collector.record_memory_usage(result.memory_usage_bytes as f64 / (1024.0 * 1024.0)).is_ok());
    
    // 12. Generate comprehensive report
    let report = metrics_collector.generate_report(TimeWindow::Hour);
    assert!(report.is_ok());
    
    // 13. Clean up
    assert!(memory_manager.deallocate(&state_id).await.is_ok());
    assert!(memory_manager.global_garbage_collect().await.is_ok());
    
    // 14. Verify system health
    let health_report = device_manager.get_health_report().await;
    assert!(!health_report.is_empty());
    
    let hardware_health = hardware_manager.get_health_report();
    assert!(!hardware_health.is_empty());
    
    println!("Integration test completed successfully!");
    println!("Circuit executed with {} gates", circuit.instructions.len());
    println!("Execution time: {:.2} ms", result.execution_time_ns as f64 / 1_000_000.0);
    println!("Memory usage: {:.2} MB", result.memory_usage_bytes as f64 / (1024.0 * 1024.0));
    println!("Fidelity: {:.6}", result.fidelity);
}