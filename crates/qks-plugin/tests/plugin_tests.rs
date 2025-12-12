//! Core Plugin Lifecycle Tests
//!
//! Tests for plugin initialization, versioning, feature detection, and cleanup.

use qks_plugin::prelude::*;

#[test]
fn test_plugin_version() {
    let version = qks_plugin::version();
    assert!(!version.is_empty());
    assert!(version.contains('.'));
    println!("QKS Plugin version: {}", version);
}

#[test]
fn test_plugin_features() {
    let features = qks_plugin::features();

    // Core features should always be present
    assert!(features.contains(&"core"));
    assert!(features.contains(&"simulator"));

    // Feature list should be non-empty
    assert!(!features.is_empty());

    println!("Available features: {:?}", features);
}

#[test]
fn test_available_devices() {
    let devices = qks_plugin::available_devices();

    // At minimum, CPU should always be available
    assert!(devices.contains(&DeviceType::Cpu));

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // On macOS with metal feature, Metal should be available
        assert!(devices.contains(&DeviceType::Metal));
    }

    println!("Available devices: {:?}", devices);
}

#[test]
fn test_device_cpu_initialization() {
    let device = QksDevice::cpu(10);
    assert!(device.is_ok());

    let device = device.unwrap();
    let info = device.info();

    assert_eq!(info.device_type, DeviceType::Cpu);
    assert_eq!(info.max_qubits, 10);
    assert!(info.name.contains("Cpu"));
}

#[test]
#[cfg(feature = "metal")]
fn test_device_metal_initialization() {
    let device = QksDevice::metal(20);
    assert!(device.is_ok());

    let device = device.unwrap();
    let info = device.info();

    assert_eq!(info.device_type, DeviceType::Metal);
    assert_eq!(info.max_qubits, 20);
    assert!(info.name.contains("Metal"));
}

#[test]
#[cfg(feature = "metal")]
fn test_device_metal_qubit_limit() {
    // Should succeed for 30 qubits
    let device_30 = QksDevice::metal(30);
    assert!(device_30.is_ok());

    // Should fail for 31 qubits
    let device_31 = QksDevice::metal(31);
    assert!(device_31.is_err());

    if let Err(e) = device_31 {
        assert!(e.to_string().contains("maximum 30 qubits"));
    }
}

#[test]
fn test_state_creation() {
    let device = QksDevice::cpu(5).unwrap();
    let state = device.create_state();

    assert!(state.is_ok());
}

#[test]
fn test_multiple_device_instances() {
    let device1 = QksDevice::cpu(10).unwrap();
    let device2 = QksDevice::cpu(15).unwrap();

    let info1 = device1.info();
    let info2 = device2.info();

    assert_eq!(info1.max_qubits, 10);
    assert_eq!(info2.max_qubits, 15);
}

#[test]
#[cfg(feature = "hyperphysics")]
fn test_hyperphysics_feature() {
    use qks_plugin::hyperphysics::ISING_CRITICAL_TEMP;

    // Verify Onsager's exact solution for 2D Ising model
    assert!((ISING_CRITICAL_TEMP - 2.269185).abs() < 0.001);
}

#[test]
#[cfg(feature = "hyperphysics")]
fn test_golden_ratio_constant() {
    use qks_plugin::hyperphysics::GOLDEN_RATIO;

    // Verify φ = (1 + √5) / 2
    let expected_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert!((GOLDEN_RATIO - expected_phi).abs() < 1e-10);
}

#[test]
fn test_error_handling_cpu_device() {
    // Create device with reasonable number of qubits
    let device = QksDevice::cpu(100);
    assert!(device.is_ok());
}

#[test]
fn test_device_info_completeness() {
    let device = QksDevice::cpu(12).unwrap();
    let info = device.info();

    // Verify all fields are properly initialized
    assert!(!info.name.is_empty());
    assert!(info.max_qubits > 0);
}

#[cfg(all(test, feature = "serde"))]
mod serialization_tests {
    use super::*;

    #[test]
    fn test_device_type_serialization() {
        // These tests would verify serde support when enabled
        // Implementation depends on serde derive additions
    }
}
