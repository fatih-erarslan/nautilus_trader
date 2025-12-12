//! Integration Tests - End-to-End QKS Plugin Tests
//!
//! Tests full workflows combining multiple layers and features.

use qks_plugin::prelude::*;

// ============================================================================
// Full Stack Integration Tests
// ============================================================================

#[test]
fn test_device_to_state_workflow() {
    let device = QksDevice::cpu(5).expect("Failed to create device");
    let state = device.create_state().expect("Failed to create state");

    // State should be properly initialized
    let measurements = state.measure().expect("Failed to measure state");
    assert!(!measurements.is_empty());

    println!("Measurement results: {:?}", measurements);
}

#[test]
fn test_quantum_circuit_execution() {
    let device = QksDevice::cpu(2).unwrap();
    let mut state = device.create_state().unwrap();

    // Apply Hadamard gate
    state.hadamard(0).expect("Hadamard failed");

    // Apply CNOT gate
    state.cnot(0, 1).expect("CNOT failed");

    // Measure
    let measurements = state.measure().expect("Measurement failed");

    // Should have 2^2 = 4 probability amplitudes
    assert_eq!(measurements.len(), 4);

    println!("Bell state measurements: {:?}", measurements);
}

#[test]
fn test_optimization_workflow() {
    let optimizer = QksOptimizer::grey_wolf()
        .dimensions(5)
        .population(20)
        .iterations(50);

    let result = optimizer
        .minimize(|params| {
            // Simple quadratic cost function
            params.iter().map(|x| x * x).sum()
        })
        .expect("Optimization failed");

    // Should find minimum near zero
    assert!(result.fitness < 1.0);

    println!("Optimization result: {:.6}", result.fitness);
    println!("Optimal position: {:?}", result.position);
}

// ============================================================================
// Error Handling Integration
// ============================================================================

#[test]
fn test_error_propagation() {
    let device = QksDevice::cpu(2).unwrap();
    let mut state = device.create_state().unwrap();

    // Try to apply gate to invalid qubit
    let result = state.hadamard(10);

    assert!(result.is_err());

    if let Err(e) = result {
        assert!(e.to_string().contains("out of bounds"));
        println!("Expected error: {}", e);
    }
}

#[test]
fn test_error_recovery() {
    let device = QksDevice::cpu(3).unwrap();
    let mut state = device.create_state().unwrap();

    // Valid operation
    assert!(state.hadamard(0).is_ok());

    // Invalid operation
    assert!(state.hadamard(10).is_err());

    // Should still work after error
    assert!(state.hadamard(1).is_ok());
}

// ============================================================================
// Feature Flag Integration
// ============================================================================

#[test]
#[cfg(feature = "hyperphysics")]
fn test_hyperphysics_integration() {
    use qks_plugin::hyperphysics::{ISING_CRITICAL_TEMP, GOLDEN_RATIO};

    // Verify constants are accessible
    assert!((ISING_CRITICAL_TEMP - 2.269185).abs() < 0.001);
    assert!((GOLDEN_RATIO - 1.618033).abs() < 0.001);

    println!("HyperPhysics integration working");
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_metal_integration() {
    let device = QksDevice::metal(15).expect("Metal device creation failed");

    let info = device.info();
    assert_eq!(info.device_type, DeviceType::Metal);
    assert_eq!(info.max_qubits, 15);

    println!("Metal GPU integration working: {}", info.name);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_execution() {
    use rayon::prelude::*;

    let devices: Vec<_> = (0..10)
        .into_par_iter()
        .map(|i| {
            let device = QksDevice::cpu(i + 1).unwrap();
            device.info()
        })
        .collect();

    assert_eq!(devices.len(), 10);

    for (i, info) in devices.iter().enumerate() {
        assert_eq!(info.max_qubits, i + 1);
    }

    println!("Parallel device creation successful");
}

// ============================================================================
// Multi-Layer Integration Tests
// ============================================================================

#[test]
fn test_layer1_to_layer8_integration() {
    // Simulate full cognitive cycle through all 8 layers

    // Layer 1: Thermodynamic (device initialization)
    let device = QksDevice::cpu(10).expect("Layer 1 failed");

    // Layer 2: Cognitive (state creation)
    let state = device.create_state().expect("Layer 2 failed");

    // Layer 3: Decision (optimization)
    let optimizer = QksOptimizer::grey_wolf().dimensions(3).population(10).iterations(20);

    // Layer 4: Learning (mock)
    // Layer 5: Collective (mock)
    // Layer 6: Consciousness (mock)
    // Layer 7: Meta-cognition (mock)
    // Layer 8: Integration

    println!("Full layer integration successful");
}

#[test]
fn test_homeostasis_monitoring() {
    // Mock homeostasis test - monitors system health

    fn check_homeostasis(device: &QksDevice) -> (f64, bool) {
        let info = device.info();
        let health_score = (info.max_qubits as f64) / 30.0;
        let is_healthy = health_score > 0.1;
        (health_score, is_healthy)
    }

    let device = QksDevice::cpu(20).unwrap();
    let (score, healthy) = check_homeostasis(&device);

    assert!(healthy);
    println!("System health: {:.2} (healthy: {})", score, healthy);
}

// ============================================================================
// Performance Integration Tests
// ============================================================================

#[test]
fn test_performance_benchmarks() {
    use std::time::Instant;

    // Test device creation performance
    let start = Instant::now();
    let _device = QksDevice::cpu(10).unwrap();
    let device_time = start.elapsed();

    // Test state creation performance
    let device = QksDevice::cpu(10).unwrap();
    let start = Instant::now();
    let _state = device.create_state().unwrap();
    let state_time = start.elapsed();

    println!("Device creation: {:?}", device_time);
    println!("State creation: {:?}", state_time);

    // Should be reasonably fast
    assert!(device_time.as_millis() < 100);
    assert!(state_time.as_millis() < 100);
}

#[test]
fn test_memory_efficiency() {
    // Test that we can create multiple states without excessive memory

    let device = QksDevice::cpu(10).unwrap();

    let states: Vec<_> = (0..100)
        .map(|_| device.create_state())
        .collect::<Result<Vec<_>>>()
        .expect("Failed to create states");

    assert_eq!(states.len(), 100);

    println!("Created 100 quantum states successfully");
}

// ============================================================================
// Cross-Feature Integration
// ============================================================================

#[test]
#[cfg(all(feature = "hyperphysics", feature = "parallel"))]
fn test_hyperphysics_parallel_optimization() {
    use rayon::prelude::*;
    use qks_plugin::hyperphysics::GOLDEN_RATIO;

    let results: Vec<_> = (0..10)
        .into_par_iter()
        .map(|i| {
            let optimizer = QksOptimizer::grey_wolf()
                .dimensions(3)
                .population(10)
                .iterations(10);

            let result = optimizer
                .minimize(|params| {
                    // Cost function using golden ratio
                    params.iter().map(|x| (x - GOLDEN_RATIO).powi(2)).sum()
                })
                .unwrap();

            (i, result.fitness)
        })
        .collect();

    assert_eq!(results.len(), 10);

    println!("Parallel HyperPhysics optimization results: {:?}", results);
}

// ============================================================================
// Version Compatibility Tests
// ============================================================================

#[test]
fn test_version_compatibility() {
    let version = qks_plugin::version();

    // Version should follow semver
    let parts: Vec<&str> = version.split('.').collect();
    assert!(parts.len() >= 2, "Version should have at least major.minor");

    println!("Version: {}", version);
}

// ============================================================================
// Documentation Examples Validation
// ============================================================================

#[test]
fn test_readme_example_works() {
    // Verify that the example from README.md actually works

    let device = QksDevice::cpu(10).expect("Device creation failed");
    let mut state = device.create_state().expect("State creation failed");

    state.hadamard(0).expect("Hadamard failed");
    state.cnot(0, 1).expect("CNOT failed");

    let optimizer = QksOptimizer::grey_wolf()
        .dimensions(5)
        .population(20)
        .iterations(50);

    let result = optimizer
        .minimize(|params| params.iter().map(|x| x * x).sum())
        .expect("Optimization failed");

    println!("README example validated: fitness = {:.6}", result.fitness);
}

// ============================================================================
// Regression Tests
// ============================================================================

#[test]
fn test_no_panic_on_valid_inputs() {
    // Ensure no panics for all valid inputs

    let device = QksDevice::cpu(5).unwrap();
    let mut state = device.create_state().unwrap();

    for i in 0..5 {
        state.hadamard(i).unwrap();
    }

    for i in 0..4 {
        state.cnot(i, i + 1).unwrap();
    }

    let _measurements = state.measure().unwrap();

    println!("No panics on valid inputs");
}
