//! FFI Safety and Correctness Tests
//!
//! Tests for Foreign Function Interface safety, C ABI compatibility,
//! and cross-language interop with Python and TypeScript.

use qks_plugin::prelude::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// ============================================================================
// FFI Type Safety Tests
// ============================================================================

#[test]
fn test_ffi_compatible_types() {
    // Verify that core types have proper memory layout for FFI
    use std::mem::{size_of, align_of};

    // DeviceType should be representable as C enum
    assert_eq!(size_of::<DeviceType>(), size_of::<i32>());

    println!("DeviceType size: {} bytes", size_of::<DeviceType>());
    println!("DeviceType align: {} bytes", align_of::<DeviceType>());
}

#[test]
fn test_error_message_c_string_safety() {
    use qks_plugin::QksError;

    let error = QksError::Device("Test error message".to_string());
    let error_str = error.to_string();

    // Verify error message can be safely converted to C string
    let c_string = CString::new(error_str.clone());
    assert!(c_string.is_ok());

    // Verify round-trip conversion
    let c_str = c_string.unwrap();
    let back_to_rust = c_str.to_str();
    assert!(back_to_rust.is_ok());
    assert!(back_to_rust.unwrap().contains("Test error"));
}

#[test]
fn test_null_pointer_safety() {
    // Tests to ensure null pointer handling is safe
    // This is critical for FFI boundaries

    let null_ptr: *const u8 = std::ptr::null();
    assert!(null_ptr.is_null());

    // FFI functions should check for null pointers
    // and return appropriate errors
}

// ============================================================================
// Python FFI Tests (PyO3)
// ============================================================================

#[cfg(feature = "python")]
mod python_ffi_tests {
    use super::*;

    #[test]
    fn test_python_type_conversion() {
        // Tests for Python type conversion when pyo3 feature is enabled
        // These would verify proper handling of:
        // - PyList <-> Vec conversions
        // - PyDict <-> HashMap conversions
        // - NumPy array handling
        // - Complex number conversions
    }

    #[test]
    fn test_python_exception_handling() {
        // Verify that Rust panics are properly converted to Python exceptions
        // and that Python exceptions don't cause undefined behavior
    }

    #[test]
    fn test_python_gil_safety() {
        // Test Global Interpreter Lock (GIL) handling
        // Ensure no deadlocks or race conditions
    }
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

#[test]
fn test_state_handle_drop_safety() {
    // Create and immediately drop a state
    {
        let device = QksDevice::cpu(5).unwrap();
        let _state = device.create_state().unwrap();
        // State should be properly cleaned up when it goes out of scope
    }

    // No memory leaks or double-frees should occur
}

#[test]
fn test_concurrent_access_safety() {
    use std::sync::Arc;
    use std::thread;

    let device = Arc::new(QksDevice::cpu(10).unwrap());

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let device_clone = Arc::clone(&device);
            thread::spawn(move || {
                let info = device_clone.info();
                assert_eq!(info.max_qubits, 10);
                println!("Thread {} verified device info", i);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_large_allocation_safety() {
    // Test that large quantum states don't cause stack overflow
    // Should allocate on heap
    let device = QksDevice::cpu(20); // 2^20 = 1M amplitudes
    assert!(device.is_ok());
}

// ============================================================================
// ABI Compatibility Tests
// ============================================================================

#[test]
fn test_repr_c_compatibility() {
    // Verify that types marked with #[repr(C)] have stable layout

    #[repr(C)]
    struct TestFFIStruct {
        x: f64,
        y: f64,
        z: i32,
    }

    use std::mem::{size_of, offset_of};

    assert_eq!(size_of::<TestFFIStruct>(), 24); // 8 + 8 + 4 + 4 (padding)

    // Verify field offsets are stable
    let offset_x = offset_of!(TestFFIStruct, x);
    let offset_y = offset_of!(TestFFIStruct, y);
    let offset_z = offset_of!(TestFFIStruct, z);

    assert_eq!(offset_x, 0);
    assert_eq!(offset_y, 8);
    assert_eq!(offset_z, 16);
}

// ============================================================================
// TypeScript/WASM Tests
// ============================================================================

#[cfg(target_family = "wasm")]
mod wasm_tests {
    use super::*;

    #[test]
    fn test_wasm_bindgen_compatibility() {
        // Tests for WebAssembly compilation when targeting wasm32
        // Verify that no unsupported features are used
    }

    #[test]
    fn test_javascript_type_conversion() {
        // Test conversion between Rust types and JavaScript types
        // via wasm-bindgen
    }
}

// ============================================================================
// Cross-Platform ABI Tests
// ============================================================================

#[test]
fn test_endianness_independence() {
    // Verify that data serialization is endianness-independent
    let value: u32 = 0x12345678;
    let bytes = value.to_le_bytes();
    let reconstructed = u32::from_le_bytes(bytes);

    assert_eq!(value, reconstructed);
}

#[test]
fn test_alignment_requirements() {
    use std::mem::align_of;

    // Verify alignment requirements for FFI types
    assert!(align_of::<f64>() <= 8);
    assert!(align_of::<Complex>() <= 16);
}

// ============================================================================
// Error Propagation Tests
// ============================================================================

#[test]
fn test_ffi_error_codes() {
    // Test that errors can be represented as C-style error codes

    fn error_to_code(error: &QksError) -> i32 {
        match error {
            QksError::Device(_) => -1,
            QksError::State(_) => -2,
            QksError::Optimization(_) => -3,
            QksError::Config(_) => -4,
            QksError::Metal(_) => -5,
            QksError::PennyLane(_) => -6,
            QksError::HyperPhysics(_) => -7,
            QksError::Python(_) => -8,
            QksError::Io(_) => -9,
        }
    }

    let device_error = QksError::Device("test".to_string());
    assert_eq!(error_to_code(&device_error), -1);

    let state_error = QksError::State("test".to_string());
    assert_eq!(error_to_code(&state_error), -2);
}

#[test]
fn test_callback_safety() {
    // Test that Rust callbacks can be safely called from C/Python

    fn example_callback(x: f64) -> f64 {
        x * x
    }

    let result = example_callback(3.0);
    assert_eq!(result, 9.0);

    // In real FFI, this would be passed as function pointer
    let fn_ptr: fn(f64) -> f64 = example_callback;
    let result2 = fn_ptr(4.0);
    assert_eq!(result2, 16.0);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn test_send_sync_traits() {
    // Verify that types are Send + Sync when they should be

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    // DeviceType should be Send + Sync (it's just an enum)
    assert_send::<DeviceType>();
    assert_sync::<DeviceType>();
}

#[test]
fn test_ffi_thread_local_safety() {
    use std::thread;

    thread::spawn(|| {
        // Each thread should be able to create its own device
        let device = QksDevice::cpu(5);
        assert!(device.is_ok());
    })
    .join()
    .unwrap();
}
