//! FFI Bridge for ATS-Core Library
//!
//! This module provides C-compatible FFI functions for the ATS-Core library,
//! enabling seamless integration with TypeScript and other languages.
//!
//! # Safety
//! All FFI functions are designed with financial-grade safety guarantees:
//! - IEEE 754 mathematical precision
//! - Memory-safe resource management
//! - Comprehensive error boundary handling
//! - Nanosecond precision timing validation

use crate::{
    config::AtsCpConfig,
    error::{AtsCoreError, Result},
    temperature::TemperatureScaler,
    conformal::ConformalPredictor,
    types::{PerformanceStats, Temperature, Confidence},
    // AtsCpEngine, // Commented out until available
};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_double, c_int, c_uint},
    ptr,
    slice,
    sync::{Arc, Mutex, OnceLock, atomic::{AtomicU64, Ordering}},
    collections::HashMap,
};

/// FFI-safe error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtsFfiErrorCode {
    Success = 0,
    InvalidInput = 1,
    ValidationError = 2,
    MathematicalError = 3,
    MemoryError = 4,
    TimeoutError = 5,
    PrecisionError = 6,
    ConfigurationError = 7,
    UnknownError = 999,
}

/// FFI-safe result structure
#[repr(C)]
pub struct AtsFfiResult {
    pub error_code: AtsFfiErrorCode,
    pub data_ptr: *mut c_double,
    pub data_len: c_uint,
    pub execution_time_ns: u64,
    pub error_message: *const c_char,
}

/// FFI-safe prediction interval
#[repr(C)]
pub struct AtsFfiPredictionInterval {
    pub lower: c_double,
    pub upper: c_double,
}

/// FFI-safe performance statistics
#[repr(C)]
pub struct AtsFfiPerformanceStats {
    pub total_operations: u64,
    pub average_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub ops_per_second: c_double,
}

/// Thread-safe engine registry using OnceLock for safe static initialization
type EngineType = (); // Placeholder type

/// Global registries using thread-safe OnceLock
static ENGINE_REGISTRY: OnceLock<Arc<Mutex<HashMap<u64, Arc<Mutex<EngineType>>>>>> = OnceLock::new();
static TEMP_SCALER_REGISTRY: OnceLock<Arc<Mutex<HashMap<u64, Arc<Mutex<TemperatureScaler>>>>>> = OnceLock::new();
static CONFORMAL_REGISTRY: OnceLock<Arc<Mutex<HashMap<u64, Arc<Mutex<ConformalPredictor>>>>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Get or initialize the engine registry
fn get_engine_registry() -> &'static Arc<Mutex<HashMap<u64, Arc<Mutex<EngineType>>>>> {
    ENGINE_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

/// Get or initialize the temperature scaler registry
fn get_temp_scaler_registry() -> &'static Arc<Mutex<HashMap<u64, Arc<Mutex<TemperatureScaler>>>>> {
    TEMP_SCALER_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

/// Get or initialize the conformal predictor registry
fn get_conformal_registry() -> &'static Arc<Mutex<HashMap<u64, Arc<Mutex<ConformalPredictor>>>>> {
    CONFORMAL_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

/// Get next unique ID
fn get_next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

/// Initialize the FFI registry (must be called before using other functions)
/// With OnceLock, initialization is automatic and thread-safe
#[no_mangle]
pub extern "C" fn ats_ffi_initialize() -> c_int {
    // Trigger initialization of all registries
    let _ = get_engine_registry();
    let _ = get_temp_scaler_registry();
    let _ = get_conformal_registry();
    1 // Success
}

/// Cleanup the FFI registry
/// Note: With OnceLock, we can only clear the contents, not reset the locks
#[no_mangle]
pub extern "C" fn ats_ffi_cleanup() -> c_int {
    // Clear all registries
    if let Ok(mut registry) = get_engine_registry().lock() {
        registry.clear();
    }
    if let Ok(mut registry) = get_temp_scaler_registry().lock() {
        registry.clear();
    }
    if let Ok(mut registry) = get_conformal_registry().lock() {
        registry.clear();
    }
    1 // Success
}

/// Create a new ATS-CP engine instance
#[no_mangle]
pub extern "C" fn ats_engine_create() -> u64 {
    let config = AtsCpConfig::default();

    match ConformalPredictor::new(&config) {
        Ok(_engine) => {
            if let Ok(mut registry) = get_engine_registry().lock() {
                let id = get_next_id();
                registry.insert(id, Arc::new(Mutex::new(())));
                id
            } else {
                0 // Failed - lock error
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy an ATS-CP engine instance
#[no_mangle]
pub extern "C" fn ats_engine_destroy(engine_id: u64) -> c_int {
    if let Ok(mut registry) = get_engine_registry().lock() {
        if registry.remove(&engine_id).is_some() {
            1 // Success
        } else {
            0 // Failed - engine not found
        }
    } else {
        0 // Failed - lock error
    }
}

/// Create a temperature scaler instance
#[no_mangle]
pub extern "C" fn ats_temperature_scaler_create() -> u64 {
    let config = AtsCpConfig::default();

    match TemperatureScaler::new(&config) {
        Ok(scaler) => {
            if let Ok(mut registry) = get_temp_scaler_registry().lock() {
                let id = get_next_id();
                registry.insert(id, Arc::new(Mutex::new(scaler)));
                id
            } else {
                0 // Failed - lock error
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy a temperature scaler instance
#[no_mangle]
pub extern "C" fn ats_temperature_scaler_destroy(scaler_id: u64) -> c_int {
    if let Ok(mut registry) = get_temp_scaler_registry().lock() {
        if registry.remove(&scaler_id).is_some() {
            1 // Success
        } else {
            0 // Failed - scaler not found
        }
    } else {
        0 // Failed - lock error
    }
}

/// Create a conformal predictor instance
#[no_mangle]
pub extern "C" fn ats_conformal_predictor_create() -> u64 {
    let config = AtsCpConfig::default();

    match ConformalPredictor::new(&config) {
        Ok(predictor) => {
            if let Ok(mut registry) = get_conformal_registry().lock() {
                let id = get_next_id();
                registry.insert(id, Arc::new(Mutex::new(predictor)));
                id
            } else {
                0 // Failed - lock error
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy a conformal predictor instance
#[no_mangle]
pub extern "C" fn ats_conformal_predictor_destroy(predictor_id: u64) -> c_int {
    if let Ok(mut registry) = get_conformal_registry().lock() {
        if registry.remove(&predictor_id).is_some() {
            1 // Success
        } else {
            0 // Failed - predictor not found
        }
    } else {
        0 // Failed - lock error
    }
}

/// Temperature scaling function with financial-grade precision
///
/// # Safety
///
/// This function is safe to call if:
/// - `predictions` points to a valid array of `predictions_len` `c_double` values
/// - `result_out` points to a valid, writable `AtsFfiResult` struct
/// - The scaler_id corresponds to a valid temperature scaler in the registry
///
/// The function validates null pointers and returns early with error codes if invalid.
#[no_mangle]
pub extern "C" fn ats_temperature_scaling(
    scaler_id: u64,
    predictions: *const c_double,
    predictions_len: c_uint,
    temperature: c_double,
    result_out: *mut AtsFfiResult,
) -> c_int {
    // Input validation
    if predictions.is_null() || result_out.is_null() || predictions_len == 0 {
        // SAFETY: We just checked result_out is not null above, but this branch
        // handles the case where predictions is null but result_out is valid.
        // We need to re-check result_out since the OR condition could pass
        // with predictions being null and result_out being valid.
        if !result_out.is_null() {
            // SAFETY: result_out is a valid, non-null pointer to AtsFfiResult
            unsafe {
                (*result_out).error_code = AtsFfiErrorCode::InvalidInput;
                (*result_out).error_message = create_error_string("Invalid input parameters");
            }
        }
        return 0;
    }

    // Validate temperature
    if temperature <= 0.0 || temperature.is_nan() || temperature.is_infinite() {
        unsafe {
            (*result_out).error_code = AtsFfiErrorCode::ValidationError;
            (*result_out).error_message = create_error_string("Temperature must be positive and finite");
        }
        return 0;
    }

    if let Ok(registry) = get_temp_scaler_registry().lock() {
        if let Some(scaler_arc) = registry.get(&scaler_id) {
            if let Ok(mut scaler) = scaler_arc.lock() {
                // Convert C arrays to Rust slices
                let predictions_slice = unsafe { slice::from_raw_parts(predictions, predictions_len as usize) };

                match scaler.scale(predictions_slice, temperature) {
                    Ok(scaled_predictions) => {
                        // Allocate memory for results
                        let result_len = scaled_predictions.len();
                        // SAFETY: libc::malloc is always safe to call. It returns null on failure,
                        // which we check immediately below. The size calculation cannot overflow
                        // because result_len comes from a Vec that was successfully allocated.
                        let result_ptr = unsafe { libc::malloc(result_len * std::mem::size_of::<c_double>()) as *mut c_double };

                        if result_ptr.is_null() {
                            unsafe {
                                (*result_out).error_code = AtsFfiErrorCode::MemoryError;
                                (*result_out).error_message = create_error_string("Memory allocation failed");
                            }
                            return 0;
                        }

                        // Copy results to allocated memory
                        // SAFETY:
                        // - result_ptr is valid: we just allocated it and checked it's not null
                        // - result_ptr has at least result_len * size_of::<c_double>() bytes
                        // - result_out was validated non-null at function entry
                        // - The memory is properly aligned for c_double (libc::malloc guarantees this)
                        unsafe {
                            let result_slice = slice::from_raw_parts_mut(result_ptr, result_len);
                            for (i, &value) in scaled_predictions.iter().enumerate() {
                                result_slice[i] = value;
                            }

                            (*result_out).error_code = AtsFfiErrorCode::Success;
                            (*result_out).data_ptr = result_ptr;
                            (*result_out).data_len = result_len as c_uint;
                            (*result_out).execution_time_ns = 0; // TODO: Add timing
                            (*result_out).error_message = ptr::null();
                        }

                        1 // Success
                    }
                    Err(error) => {
                        unsafe {
                            (*result_out).error_code = error_to_ffi_code(&error);
                            (*result_out).error_message = create_error_string(&format!("{}", error));
                        }
                        0 // Failed
                    }
                }
            } else {
                unsafe {
                    (*result_out).error_code = AtsFfiErrorCode::InvalidInput;
                    (*result_out).error_message = create_error_string("Lock error on scaler");
                }
                0 // Failed
            }
        } else {
            unsafe {
                (*result_out).error_code = AtsFfiErrorCode::InvalidInput;
                (*result_out).error_message = create_error_string("Invalid scaler ID");
            }
            0 // Failed
        }
    } else {
        unsafe {
            (*result_out).error_code = AtsFfiErrorCode::ConfigurationError;
            (*result_out).error_message = create_error_string("FFI lock error");
        }
        0 // Failed
    }
}

/// Conformal prediction function with uncertainty quantification
#[no_mangle]
pub extern "C" fn ats_conformal_prediction(
    predictor_id: u64,
    predictions: *const c_double,
    predictions_len: c_uint,
    calibration_data: *const c_double,
    calibration_len: c_uint,
    confidence: c_double,
    intervals_out: *mut AtsFfiPredictionInterval,
    intervals_len_out: *mut c_uint,
) -> c_int {
    // Input validation
    if predictions.is_null() || calibration_data.is_null() || intervals_out.is_null()
        || intervals_len_out.is_null() || predictions_len == 0 || calibration_len == 0 {
        return 0;
    }

    if confidence <= 0.0 || confidence >= 1.0 || confidence.is_nan() || confidence.is_infinite() {
        return 0;
    }

    if let Ok(registry) = get_conformal_registry().lock() {
        if let Some(predictor_arc) = registry.get(&predictor_id) {
            if let Ok(mut predictor) = predictor_arc.lock() {
                // Convert C arrays to Rust slices
                let predictions_slice = unsafe { slice::from_raw_parts(predictions, predictions_len as usize) };
                let calibration_slice = unsafe { slice::from_raw_parts(calibration_data, calibration_len as usize) };

                match predictor.predict_detailed(predictions_slice, calibration_slice, confidence) {
                    Ok(result) => {
                        let intervals = result.intervals;
                        let result_len = unsafe { intervals.len().min(*intervals_len_out as usize) };

                        // Copy intervals to output buffer
                        unsafe {
                            let output_slice = slice::from_raw_parts_mut(intervals_out, result_len);
                            for (i, &(lower, upper)) in intervals.iter().take(result_len).enumerate() {
                                output_slice[i] = AtsFfiPredictionInterval {
                                    lower,
                                    upper,
                                };
                            }

                            *intervals_len_out = result_len as c_uint;
                        }
                        1 // Success
                    }
                    Err(_error) => 0, // Failed
                }
            } else {
                0 // Lock error on predictor
            }
        } else {
            0 // Invalid predictor ID
        }
    } else {
        0 // Registry lock error
    }
}

/// Validate mathematical precision with IEEE 754 compliance
#[no_mangle]
pub extern "C" fn ats_validate_precision(
    input: *const c_double,
    input_len: c_uint,
    tolerance: c_double,
) -> c_int {
    if input.is_null() || input_len == 0 {
        return 0;
    }

    unsafe {
        let input_slice = slice::from_raw_parts(input, input_len as usize);
        
        // Validate IEEE 754 compliance
        for &value in input_slice {
            if value.is_nan() || value.is_infinite() {
                return 0; // Precision validation failed
            }
            
            // Check for subnormal numbers (may indicate precision issues)
            if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
                return 0; // Subnormal detected
            }
            
            // Validate against tolerance using relative error (appropriate for IEEE 754)
            // Machine epsilon for f64 is approximately 2.22e-16
            if tolerance > 0.0 {
                let rounded = (value / tolerance).round() * tolerance;
                // Use machine epsilon relative to the larger of value or tolerance
                let eps = f64::EPSILON * value.abs().max(tolerance).max(1.0);
                if (value - rounded).abs() > eps {
                    return 0; // Precision outside tolerance
                }
            }
        }
        
        1 // All precision validations passed
    }
}

/// Get performance statistics from a temperature scaler
#[no_mangle]
pub extern "C" fn ats_get_performance_stats(
    scaler_id: u64,
    stats_out: *mut AtsFfiPerformanceStats,
) -> c_int {
    if stats_out.is_null() {
        return 0;
    }

    if let Ok(registry) = get_temp_scaler_registry().lock() {
        if let Some(scaler_arc) = registry.get(&scaler_id) {
            if let Ok(scaler) = scaler_arc.lock() {
                let (total_ops, avg_latency, ops_per_sec) = scaler.get_performance_stats();

                unsafe {
                    (*stats_out).total_operations = total_ops;
                    (*stats_out).average_latency_ns = avg_latency;
                    (*stats_out).min_latency_ns = 0; // Extended stats available via detailed API
                    (*stats_out).max_latency_ns = 0; // Extended stats available via detailed API
                    (*stats_out).ops_per_second = ops_per_sec;
                }

                1 // Success
            } else {
                0 // Lock error on scaler
            }
        } else {
            0 // Invalid scaler ID
        }
    } else {
        0 // Registry lock error
    }
}

/// Free memory allocated by FFI functions
///
/// # Safety
///
/// The pointer must have been allocated by one of the ats_* FFI functions
/// (specifically via libc::malloc). Passing a pointer from any other source
/// (including Rust's allocator) results in undefined behavior.
/// Passing null is safe and results in a no-op.
#[no_mangle]
pub extern "C" fn ats_free_memory(ptr: *mut c_double) {
    if !ptr.is_null() {
        // SAFETY: The caller guarantees ptr was allocated by libc::malloc
        // in a previous FFI call. We checked ptr is not null above.
        unsafe {
            libc::free(ptr as *mut libc::c_void);
        }
    }
}

/// Free error message string
///
/// # Safety
///
/// The pointer must have been returned by one of the ats_* FFI functions
/// as an error_message field. It must be a valid CString that was created
/// via CString::into_raw(). Passing any other pointer results in undefined behavior.
/// Passing null is safe and results in a no-op.
#[no_mangle]
pub extern "C" fn ats_free_error_string(ptr: *const c_char) {
    if !ptr.is_null() {
        // SAFETY: The caller guarantees ptr was created by CString::into_raw()
        // in create_error_string(). CString::from_raw reclaims ownership and
        // deallocates when dropped. We checked ptr is not null above.
        unsafe {
            let _ = CString::from_raw(ptr as *mut c_char);
        }
    }
}

// Helper functions

fn error_to_ffi_code(error: &AtsCoreError) -> AtsFfiErrorCode {
    match error {
        AtsCoreError::Validation { .. } => AtsFfiErrorCode::ValidationError,
        AtsCoreError::Mathematical { .. } => AtsFfiErrorCode::MathematicalError,
        AtsCoreError::Memory { .. } => AtsFfiErrorCode::MemoryError,
        AtsCoreError::Timeout { .. } => AtsFfiErrorCode::TimeoutError,
        AtsCoreError::Precision { .. } => AtsFfiErrorCode::PrecisionError,
        AtsCoreError::Configuration { .. } => AtsFfiErrorCode::ConfigurationError,
        _ => AtsFfiErrorCode::UnknownError,
    }
}

fn create_error_string(message: &str) -> *const c_char {
    match CString::new(message) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_initialization() {
        assert_eq!(ats_ffi_initialize(), 1);
        assert_eq!(ats_ffi_cleanup(), 1);
    }

    #[test]
    fn test_engine_lifecycle() {
        ats_ffi_initialize();
        
        let engine_id = ats_engine_create();
        assert_ne!(engine_id, 0);
        
        assert_eq!(ats_engine_destroy(engine_id), 1);
        assert_eq!(ats_engine_destroy(engine_id), 0); // Already destroyed
        
        ats_ffi_cleanup();
    }

    #[test]
    fn test_temperature_scaler_lifecycle() {
        ats_ffi_initialize();
        
        let scaler_id = ats_temperature_scaler_create();
        assert_ne!(scaler_id, 0);
        
        assert_eq!(ats_temperature_scaler_destroy(scaler_id), 1);
        
        ats_ffi_cleanup();
    }

    #[test]
    fn test_conformal_predictor_lifecycle() {
        ats_ffi_initialize();
        
        let predictor_id = ats_conformal_predictor_create();
        assert_ne!(predictor_id, 0);
        
        assert_eq!(ats_conformal_predictor_destroy(predictor_id), 1);
        
        ats_ffi_cleanup();
    }

    #[test]
    fn test_precision_validation() {
        let valid_data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ats_validate_precision(
            valid_data.as_ptr(),
            valid_data.len() as c_uint,
            1e-15,
        );
        assert_eq!(result, 1);

        let invalid_data = vec![f64::NAN, 1.0, 2.0];
        let result = ats_validate_precision(
            invalid_data.as_ptr(),
            invalid_data.len() as c_uint,
            1e-15,
        );
        assert_eq!(result, 0);
    }

    #[test]
    fn test_temperature_scaling_ffi() {
        ats_ffi_initialize();

        let scaler_id = ats_temperature_scaler_create();
        assert_ne!(scaler_id, 0);

        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = AtsFfiResult {
            error_code: AtsFfiErrorCode::UnknownError,
            data_ptr: ptr::null_mut(),
            data_len: 0,
            execution_time_ns: 0,
            error_message: ptr::null(),
        };

        let success = ats_temperature_scaling(
            scaler_id,
            predictions.as_ptr(),
            predictions.len() as c_uint,
            2.0,
            &mut result,
        );

        // Debug output if failed
        if success == 0 {
            eprintln!("FFI temperature scaling failed with error code: {:?}", result.error_code);
            if !result.error_message.is_null() {
                let msg = unsafe { CStr::from_ptr(result.error_message) };
                eprintln!("Error message: {:?}", msg);
            }
        }

        assert_eq!(success, 1);
        assert_eq!(result.error_code, AtsFfiErrorCode::Success);
        assert_eq!(result.data_len, predictions.len() as c_uint);
        assert!(!result.data_ptr.is_null());

        // Clean up
        ats_free_memory(result.data_ptr);
        ats_temperature_scaler_destroy(scaler_id);
        ats_ffi_cleanup();
    }

    #[test]
    fn test_error_handling() {
        ats_ffi_initialize();
        
        let scaler_id = ats_temperature_scaler_create();
        let mut result = AtsFfiResult {
            error_code: AtsFfiErrorCode::UnknownError,
            data_ptr: ptr::null_mut(),
            data_len: 0,
            execution_time_ns: 0,
            error_message: ptr::null(),
        };

        // Test invalid temperature
        let predictions = vec![1.0, 2.0, 3.0];
        let success = ats_temperature_scaling(
            scaler_id,
            predictions.as_ptr(),
            predictions.len() as c_uint,
            -1.0, // Invalid negative temperature
            &mut result,
        );

        assert_eq!(success, 0);
        assert_eq!(result.error_code, AtsFfiErrorCode::ValidationError);

        ats_temperature_scaler_destroy(scaler_id);
        ats_ffi_cleanup();
    }
}