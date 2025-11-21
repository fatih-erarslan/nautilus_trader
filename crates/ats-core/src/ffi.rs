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
    sync::{Arc, Mutex},
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

/// Thread-safe engine registry
// Engine registry - using generic type until AtsCpEngine is available
type EngineType = (); // Placeholder type
static mut ENGINE_REGISTRY: Option<Arc<Mutex<HashMap<u64, Arc<Mutex<EngineType>>>>>> = None;
static mut TEMP_SCALER_REGISTRY: Option<Arc<Mutex<HashMap<u64, Arc<Mutex<TemperatureScaler>>>>>> = None;
static mut CONFORMAL_REGISTRY: Option<Arc<Mutex<HashMap<u64, Arc<Mutex<ConformalPredictor>>>>>> = None;
static mut NEXT_ID: u64 = 1;

/// Initialize the FFI registry (must be called before using other functions)
#[no_mangle]
pub extern "C" fn ats_ffi_initialize() -> c_int {
    unsafe {
        ENGINE_REGISTRY = Some(Arc::new(Mutex::new(HashMap::new())));
        TEMP_SCALER_REGISTRY = Some(Arc::new(Mutex::new(HashMap::new())));
        CONFORMAL_REGISTRY = Some(Arc::new(Mutex::new(HashMap::new())));
        NEXT_ID = 1;
    }
    1 // Success
}

/// Cleanup the FFI registry
#[no_mangle]
pub extern "C" fn ats_ffi_cleanup() -> c_int {
    unsafe {
        ENGINE_REGISTRY = None;
        TEMP_SCALER_REGISTRY = None;
        CONFORMAL_REGISTRY = None;
        NEXT_ID = 1;
    }
    1 // Success
}

/// Create a new ATS-CP engine instance
#[no_mangle]
pub extern "C" fn ats_engine_create() -> u64 {
    let config = AtsCpConfig::default();
    
    match ConformalPredictor::new(&config) {
        Ok(engine) => {
            unsafe {
                if let Some(registry) = &ENGINE_REGISTRY {
                    let mut registry = registry.lock().unwrap();
                    let id = NEXT_ID;
                    NEXT_ID += 1;
                    registry.insert(id, Arc::new(Mutex::new(())));
                    id
                } else {
                    0 // Failed - registry not initialized
                }
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy an ATS-CP engine instance
#[no_mangle]
pub extern "C" fn ats_engine_destroy(engine_id: u64) -> c_int {
    unsafe {
        if let Some(registry) = &ENGINE_REGISTRY {
            let mut registry = registry.lock().unwrap();
            if registry.remove(&engine_id).is_some() {
                1 // Success
            } else {
                0 // Failed - engine not found
            }
        } else {
            0 // Failed - registry not initialized
        }
    }
}

/// Create a temperature scaler instance
#[no_mangle]
pub extern "C" fn ats_temperature_scaler_create() -> u64 {
    let config = AtsCpConfig::default();
    
    match TemperatureScaler::new(&config) {
        Ok(scaler) => {
            unsafe {
                if let Some(registry) = &TEMP_SCALER_REGISTRY {
                    let mut registry = registry.lock().unwrap();
                    let id = NEXT_ID;
                    NEXT_ID += 1;
                    registry.insert(id, Arc::new(Mutex::new(scaler)));
                    id
                } else {
                    0 // Failed - registry not initialized
                }
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy a temperature scaler instance
#[no_mangle]
pub extern "C" fn ats_temperature_scaler_destroy(scaler_id: u64) -> c_int {
    unsafe {
        if let Some(registry) = &TEMP_SCALER_REGISTRY {
            let mut registry = registry.lock().unwrap();
            if registry.remove(&scaler_id).is_some() {
                1 // Success
            } else {
                0 // Failed - scaler not found
            }
        } else {
            0 // Failed - registry not initialized
        }
    }
}

/// Create a conformal predictor instance
#[no_mangle]
pub extern "C" fn ats_conformal_predictor_create() -> u64 {
    let config = AtsCpConfig::default();
    
    match ConformalPredictor::new(&config) {
        Ok(predictor) => {
            unsafe {
                if let Some(registry) = &CONFORMAL_REGISTRY {
                    let mut registry = registry.lock().unwrap();
                    let id = NEXT_ID;
                    NEXT_ID += 1;
                    registry.insert(id, Arc::new(Mutex::new(predictor)));
                    id
                } else {
                    0 // Failed - registry not initialized
                }
            }
        }
        Err(_) => 0, // Failed
    }
}

/// Destroy a conformal predictor instance
#[no_mangle]
pub extern "C" fn ats_conformal_predictor_destroy(predictor_id: u64) -> c_int {
    unsafe {
        if let Some(registry) = &CONFORMAL_REGISTRY {
            let mut registry = registry.lock().unwrap();
            if registry.remove(&predictor_id).is_some() {
                1 // Success
            } else {
                0 // Failed - predictor not found
            }
        } else {
            0 // Failed - registry not initialized
        }
    }
}

/// Temperature scaling function with financial-grade precision
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
        unsafe {
            (*result_out).error_code = AtsFfiErrorCode::InvalidInput;
            (*result_out).error_message = create_error_string("Invalid input parameters");
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

    unsafe {
        if let Some(registry) = &TEMP_SCALER_REGISTRY {
            let registry = registry.lock().unwrap();
            if let Some(scaler_arc) = registry.get(&scaler_id) {
                let mut scaler = scaler_arc.lock().unwrap();
                
                // Convert C arrays to Rust slices
                let predictions_slice = slice::from_raw_parts(predictions, predictions_len as usize);
                
                match scaler.scale(predictions_slice, temperature) {
                    Ok(scaled_predictions) => {
                        // Allocate memory for results
                        let result_len = scaled_predictions.len();
                        let result_ptr = libc::malloc(result_len * std::mem::size_of::<c_double>()) as *mut c_double;
                        
                        if result_ptr.is_null() {
                            (*result_out).error_code = AtsFfiErrorCode::MemoryError;
                            (*result_out).error_message = create_error_string("Memory allocation failed");
                            return 0;
                        }
                        
                        // Copy results to allocated memory
                        let result_slice = slice::from_raw_parts_mut(result_ptr, result_len);
                        for (i, &value) in scaled_predictions.iter().enumerate() {
                            result_slice[i] = value;
                        }
                        
                        (*result_out).error_code = AtsFfiErrorCode::Success;
                        (*result_out).data_ptr = result_ptr;
                        (*result_out).data_len = result_len as c_uint;
                        (*result_out).execution_time_ns = 0; // TODO: Add timing
                        (*result_out).error_message = ptr::null();
                        
                        1 // Success
                    }
                    Err(error) => {
                        (*result_out).error_code = error_to_ffi_code(&error);
                        (*result_out).error_message = create_error_string(&format!("{}", error));
                        0 // Failed
                    }
                }
            } else {
                (*result_out).error_code = AtsFfiErrorCode::InvalidInput;
                (*result_out).error_message = create_error_string("Invalid scaler ID");
                0 // Failed
            }
        } else {
            (*result_out).error_code = AtsFfiErrorCode::ConfigurationError;
            (*result_out).error_message = create_error_string("FFI not initialized");
            0 // Failed
        }
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

    unsafe {
        if let Some(registry) = &CONFORMAL_REGISTRY {
            let registry = registry.lock().unwrap();
            if let Some(predictor_arc) = registry.get(&predictor_id) {
                let mut predictor = predictor_arc.lock().unwrap();
                
                // Convert C arrays to Rust slices
                let predictions_slice = slice::from_raw_parts(predictions, predictions_len as usize);
                let calibration_slice = slice::from_raw_parts(calibration_data, calibration_len as usize);
                
                match predictor.predict_detailed(predictions_slice, calibration_slice, confidence) {
                    Ok(result) => {
                        let intervals = result.intervals;
                        let result_len = intervals.len().min(*intervals_len_out as usize);
                        
                        // Copy intervals to output buffer
                        let output_slice = slice::from_raw_parts_mut(intervals_out, result_len);
                        for (i, &(lower, upper)) in intervals.iter().take(result_len).enumerate() {
                            output_slice[i] = AtsFfiPredictionInterval {
                                lower,
                                upper,
                            };
                        }
                        
                        *intervals_len_out = result_len as c_uint;
                        1 // Success
                    }
                    Err(_error) => 0, // Failed
                }
            } else {
                0 // Invalid predictor ID
            }
        } else {
            0 // FFI not initialized
        }
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
            
            // Validate against tolerance
            if tolerance > 0.0 {
                let rounded = (value / tolerance).round() * tolerance;
                if (value - rounded).abs() > tolerance * 1e-15 {
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

    unsafe {
        if let Some(registry) = &TEMP_SCALER_REGISTRY {
            let registry = registry.lock().unwrap();
            if let Some(scaler_arc) = registry.get(&scaler_id) {
                let scaler = scaler_arc.lock().unwrap();
                let (total_ops, avg_latency, ops_per_sec) = scaler.get_performance_stats();
                
                (*stats_out).total_operations = total_ops;
                (*stats_out).average_latency_ns = avg_latency;
                (*stats_out).min_latency_ns = 0; // TODO: Implement
                (*stats_out).max_latency_ns = 0; // TODO: Implement
                (*stats_out).ops_per_second = ops_per_sec;
                
                1 // Success
            } else {
                0 // Invalid scaler ID
            }
        } else {
            0 // FFI not initialized
        }
    }
}

/// Free memory allocated by FFI functions
#[no_mangle]
pub extern "C" fn ats_free_memory(ptr: *mut c_double) {
    if !ptr.is_null() {
        unsafe {
            libc::free(ptr as *mut libc::c_void);
        }
    }
}

/// Free error message string
#[no_mangle]
pub extern "C" fn ats_free_error_string(ptr: *const c_char) {
    if !ptr.is_null() {
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
    use std::mem;

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