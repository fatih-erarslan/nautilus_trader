//! Foreign Function Interface (FFI) for CDFA Unified Library
//!
//! This module provides safe and efficient FFI interfaces for multiple languages:
//! - C API with proper error handling and memory management
//! - Python bindings with PyO3 and NumPy integration
//! - Zero-copy operations where possible
//! - Financial system safety with validation and error boundaries
//!
//! ## Safety Guarantees
//!
//! - All C API functions perform input validation
//! - Memory is properly allocated and deallocated
//! - Error codes are returned for all operations
//! - Thread safety is maintained for concurrent access
//! - Financial data integrity is preserved

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::unified::UnifiedCdfa;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_uint};
use std::ptr;
use std::sync::Arc;
use parking_lot::RwLock;

#[cfg(feature = "python")]
pub mod python;

// Global error handling
static mut LAST_ERROR: Option<String> = None;
static mut ERROR_MUTEX: parking_lot::Mutex<()> = parking_lot::const_mutex(());

/// Error codes for C API
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CdfaErrorCode {
    Success = 0,
    InvalidInput = 1,
    DimensionMismatch = 2,
    MathError = 3,
    NumericalError = 4,
    SimdError = 5,
    ParallelError = 6,
    GpuError = 7,
    MlError = 8,
    DetectionError = 9,
    ConfigError = 10,
    ResourceError = 11,
    TimeoutError = 12,
    UnsupportedOperation = 13,
    FeatureNotEnabled = 14,
    ExternalError = 15,
    UnknownError = 99,
}

impl From<&CdfaError> for CdfaErrorCode {
    fn from(error: &CdfaError) -> Self {
        match error {
            CdfaError::InvalidInput { .. } => CdfaErrorCode::InvalidInput,
            CdfaError::DimensionMismatch { .. } => CdfaErrorCode::DimensionMismatch,
            CdfaError::MathError { .. } => CdfaErrorCode::MathError,
            CdfaError::NumericalError { .. } => CdfaErrorCode::NumericalError,
            CdfaError::SimdError { .. } => CdfaErrorCode::SimdError,
            CdfaError::ParallelError { .. } => CdfaErrorCode::ParallelError,
            #[cfg(feature = "gpu")]
            CdfaError::GpuError { .. } => CdfaErrorCode::GpuError,
            #[cfg(feature = "ml")]
            CdfaError::MlError { .. } => CdfaErrorCode::MlError,
            #[cfg(feature = "detectors")]
            CdfaError::DetectionError { .. } => CdfaErrorCode::DetectionError,
            CdfaError::ConfigError { .. } => CdfaErrorCode::ConfigError,
            CdfaError::ResourceError { .. } => CdfaErrorCode::ResourceError,
            CdfaError::TimeoutError { .. } => CdfaErrorCode::TimeoutError,
            CdfaError::UnsupportedOperation { .. } => CdfaErrorCode::UnsupportedOperation,
            CdfaError::FeatureNotEnabled { .. } => CdfaErrorCode::FeatureNotEnabled,
            CdfaError::ExternalError { .. } => CdfaErrorCode::ExternalError,
            _ => CdfaErrorCode::UnknownError,
        }
    }
}

/// Opaque handle for UnifiedCdfa instances
#[repr(C)]
pub struct CdfaHandle {
    cdfa: Arc<RwLock<UnifiedCdfa>>,
}

/// C-compatible array structure
#[repr(C)]
pub struct CArray2D {
    data: *mut c_double,
    rows: c_uint,
    cols: c_uint,
    owns_data: bool,
}

/// C-compatible array structure for 1D data
#[repr(C)]
pub struct CArray1D {
    data: *mut c_double,
    len: c_uint,
    owns_data: bool,
}

/// C-compatible configuration structure
#[repr(C)]
pub struct CCdfaConfig {
    num_threads: c_uint,
    enable_simd: bool,
    enable_gpu: bool,
    tolerance: c_double,
    max_iterations: c_uint,
    convergence_threshold: c_double,
    cache_size_mb: c_uint,
    enable_distributed: bool,
}

impl From<CdfaConfig> for CCdfaConfig {
    fn from(config: CdfaConfig) -> Self {
        Self {
            num_threads: config.num_threads as c_uint,
            enable_simd: config.enable_simd,
            enable_gpu: config.enable_gpu,
            tolerance: config.tolerance,
            max_iterations: config.max_iterations as c_uint,
            convergence_threshold: config.convergence_threshold,
            cache_size_mb: config.cache_size_mb as c_uint,
            enable_distributed: config.enable_distributed,
        }
    }
}

impl From<CCdfaConfig> for CdfaConfig {
    fn from(config: CCdfaConfig) -> Self {
        Self {
            num_threads: config.num_threads as usize,
            enable_simd: config.enable_simd,
            enable_gpu: config.enable_gpu,
            tolerance: config.tolerance,
            max_iterations: config.max_iterations as usize,
            convergence_threshold: config.convergence_threshold,
            cache_size_mb: config.cache_size_mb as usize,
            enable_distributed: config.enable_distributed,
            diversity_method: None,
            fusion_method: None,
            enabled_detectors: None,
            enabled_analyzers: None,
            enabled_algorithms: None,
        }
    }
}

/// C-compatible analysis result
#[repr(C)]
pub struct CAnalysisResult {
    data: CArray1D,
    metrics_count: c_uint,
    metric_names: *mut *mut c_char,
    metric_values: *mut c_double,
    patterns_count: c_uint,
    execution_time_us: u64,
    error_code: CdfaErrorCode,
}

// Helper functions for error handling
fn set_last_error(error: &CdfaError) {
    let _guard = unsafe { ERROR_MUTEX.lock() };
    unsafe {
        LAST_ERROR = Some(error.to_string());
    }
}

fn handle_result<T>(result: Result<T>) -> (T, CdfaErrorCode) 
where 
    T: Default,
{
    match result {
        Ok(value) => (value, CdfaErrorCode::Success),
        Err(error) => {
            set_last_error(&error);
            (T::default(), CdfaErrorCode::from(&error))
        }
    }
}

// Memory management utilities
impl CArray2D {
    fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols].into_boxed_slice();
        Self {
            data: Box::into_raw(data) as *mut c_double,
            rows: rows as c_uint,
            cols: cols as c_uint,
            owns_data: true,
        }
    }

    fn from_ptr(data: *mut c_double, rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows: rows as c_uint,
            cols: cols as c_uint,
            owns_data: false,
        }
    }

    fn to_ndarray(&self) -> Result<FloatArray2> {
        if self.data.is_null() {
            return Err(CdfaError::invalid_input("Null data pointer"));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(self.data, (self.rows * self.cols) as usize)
        };

        FloatArray2::from_shape_vec(
            (self.rows as usize, self.cols as usize),
            slice.to_vec(),
        ).map_err(|e| CdfaError::invalid_input(format!("Array shape error: {}", e)))
    }

    fn from_ndarray(array: &FloatArray2) -> Self {
        let (rows, cols) = array.dim();
        let mut c_array = Self::new(rows, cols);
        
        let slice = unsafe {
            std::slice::from_raw_parts_mut(c_array.data, rows * cols)
        };
        
        for (i, &value) in array.iter().enumerate() {
            slice[i] = value;
        }
        
        c_array
    }
}

impl CArray1D {
    fn new(len: usize) -> Self {
        let data = vec![0.0; len].into_boxed_slice();
        Self {
            data: Box::into_raw(data) as *mut c_double,
            len: len as c_uint,
            owns_data: true,
        }
    }

    fn from_ptr(data: *mut c_double, len: usize) -> Self {
        Self {
            data,
            len: len as c_uint,
            owns_data: false,
        }
    }

    fn to_ndarray(&self) -> Result<FloatArray1> {
        if self.data.is_null() {
            return Err(CdfaError::invalid_input("Null data pointer"));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(self.data, self.len as usize)
        };

        Ok(FloatArray1::from_vec(slice.to_vec()))
    }

    fn from_ndarray(array: &FloatArray1) -> Self {
        let len = array.len();
        let mut c_array = Self::new(len);
        
        let slice = unsafe {
            std::slice::from_raw_parts_mut(c_array.data, len)
        };
        
        for (i, &value) in array.iter().enumerate() {
            slice[i] = value;
        }
        
        c_array
    }
}

impl Drop for CArray2D {
    fn drop(&mut self) {
        if self.owns_data && !self.data.is_null() {
            unsafe {
                let _boxed = Box::from_raw(std::slice::from_raw_parts_mut(
                    self.data,
                    (self.rows * self.cols) as usize,
                ));
            }
        }
    }
}

impl Drop for CArray1D {
    fn drop(&mut self) {
        if self.owns_data && !self.data.is_null() {
            unsafe {
                let _boxed = Box::from_raw(std::slice::from_raw_parts_mut(
                    self.data,
                    self.len as usize,
                ));
            }
        }
    }
}

// C API Exports
extern "C" {
    // Core lifecycle functions
    
    /// Create a new CDFA instance with default configuration
    /// 
    /// # Safety
    /// 
    /// The returned handle must be freed using cdfa_destroy()
    /// 
    /// # Returns
    /// 
    /// Valid handle on success, null pointer on failure
    pub fn cdfa_create() -> *mut CdfaHandle;

    /// Create a new CDFA instance with custom configuration
    /// 
    /// # Arguments
    /// 
    /// * `config` - Configuration structure
    /// 
    /// # Safety
    /// 
    /// The returned handle must be freed using cdfa_destroy()
    /// 
    /// # Returns
    /// 
    /// Valid handle on success, null pointer on failure
    pub fn cdfa_create_with_config(config: *const CCdfaConfig) -> *mut CdfaHandle;

    /// Destroy a CDFA instance and free associated memory
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Handle to destroy
    /// 
    /// # Safety
    /// 
    /// Handle must be valid and not used after this call
    pub fn cdfa_destroy(handle: *mut CdfaHandle);

    /// Get the last error message
    /// 
    /// # Returns
    /// 
    /// Null-terminated string with error message, or null if no error
    /// The returned string is valid until the next error occurs
    pub fn cdfa_get_last_error() -> *const c_char;

    /// Clear the last error
    pub fn cdfa_clear_error();

    // Analysis functions
    
    /// Perform comprehensive CDFA analysis
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Valid CDFA handle
    /// * `data` - Input data matrix (rows = observations, cols = features)
    /// * `result` - Output result structure
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0)
    /// 
    /// # Safety
    /// 
    /// - `handle` must be valid
    /// - `data` must point to valid array
    /// - `result` will be allocated and must be freed with cdfa_free_result()
    pub fn cdfa_analyze(
        handle: *mut CdfaHandle,
        data: *const CArray2D,
        result: *mut *mut CAnalysisResult,
    ) -> CdfaErrorCode;

    /// Calculate diversity metrics only
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Valid CDFA handle
    /// * `data` - Input data matrix
    /// * `result` - Output diversity scores
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0)
    pub fn cdfa_calculate_diversity(
        handle: *mut CdfaHandle,
        data: *const CArray2D,
        result: *mut CArray1D,
    ) -> CdfaErrorCode;

    /// Apply fusion algorithms
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Valid CDFA handle
    /// * `scores` - Input diversity scores
    /// * `data` - Original data matrix
    /// * `result` - Output fused scores
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0)
    pub fn cdfa_apply_fusion(
        handle: *mut CdfaHandle,
        scores: *const CArray1D,
        data: *const CArray2D,
        result: *mut CArray1D,
    ) -> CdfaErrorCode;

    // Configuration functions
    
    /// Get current configuration
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Valid CDFA handle
    /// * `config` - Output configuration structure
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0)
    pub fn cdfa_get_config(
        handle: *mut CdfaHandle,
        config: *mut CCdfaConfig,
    ) -> CdfaErrorCode;

    /// Update configuration
    /// 
    /// # Arguments
    /// 
    /// * `handle` - Valid CDFA handle
    /// * `config` - New configuration
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0)
    pub fn cdfa_set_config(
        handle: *mut CdfaHandle,
        config: *const CCdfaConfig,
    ) -> CdfaErrorCode;

    // Utility functions
    
    /// Validate input data for financial system safety
    /// 
    /// # Arguments
    /// 
    /// * `data` - Data to validate
    /// 
    /// # Returns
    /// 
    /// Error code (Success = 0 if valid)
    pub fn cdfa_validate_data(data: *const CArray2D) -> CdfaErrorCode;

    /// Get library version information
    /// 
    /// # Returns
    /// 
    /// Null-terminated version string
    pub fn cdfa_get_version() -> *const c_char;

    /// Get build information
    /// 
    /// # Returns
    /// 
    /// Null-terminated build info string
    pub fn cdfa_get_build_info() -> *const c_char;

    // Memory management
    
    /// Free analysis result
    /// 
    /// # Arguments
    /// 
    /// * `result` - Result to free
    /// 
    /// # Safety
    /// 
    /// Result must be valid and not used after this call
    pub fn cdfa_free_result(result: *mut CAnalysisResult);

    /// Allocate 2D array
    /// 
    /// # Arguments
    /// 
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// 
    /// # Returns
    /// 
    /// Allocated array or null on failure
    pub fn cdfa_alloc_array2d(rows: c_uint, cols: c_uint) -> *mut CArray2D;

    /// Allocate 1D array
    /// 
    /// # Arguments
    /// 
    /// * `len` - Array length
    /// 
    /// # Returns
    /// 
    /// Allocated array or null on failure
    pub fn cdfa_alloc_array1d(len: c_uint) -> *mut CArray1D;

    /// Free 2D array
    /// 
    /// # Arguments
    /// 
    /// * `array` - Array to free
    pub fn cdfa_free_array2d(array: *mut CArray2D);

    /// Free 1D array
    /// 
    /// # Arguments
    /// 
    /// * `array` - Array to free
    pub fn cdfa_free_array1d(array: *mut CArray1D);
}

// Actual C API implementations
#[no_mangle]
pub extern "C" fn cdfa_create() -> *mut CdfaHandle {
    match UnifiedCdfa::new() {
        Ok(cdfa) => Box::into_raw(Box::new(CdfaHandle {
            cdfa: Arc::new(RwLock::new(cdfa)),
        })),
        Err(error) => {
            set_last_error(&error);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_create_with_config(config: *const CCdfaConfig) -> *mut CdfaHandle {
    if config.is_null() {
        set_last_error(&CdfaError::invalid_input("Config cannot be null"));
        return ptr::null_mut();
    }

    let config = unsafe { &*config };
    let rust_config = CdfaConfig::from(*config);

    match UnifiedCdfa::builder().with_config(rust_config).build() {
        Ok(cdfa) => Box::into_raw(Box::new(CdfaHandle {
            cdfa: Arc::new(RwLock::new(cdfa)),
        })),
        Err(error) => {
            set_last_error(&error);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_destroy(handle: *mut CdfaHandle) {
    if !handle.is_null() {
        unsafe {
            let _boxed = Box::from_raw(handle);
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_get_last_error() -> *const c_char {
    let _guard = unsafe { ERROR_MUTEX.lock() };
    unsafe {
        match &LAST_ERROR {
            Some(error) => {
                match CString::new(error.clone()) {
                    Ok(c_string) => c_string.into_raw(),
                    Err(_) => ptr::null(),
                }
            }
            None => ptr::null(),
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_clear_error() {
    let _guard = unsafe { ERROR_MUTEX.lock() };
    unsafe {
        LAST_ERROR = None;
    }
}

#[no_mangle]
pub extern "C" fn cdfa_analyze(
    handle: *mut CdfaHandle,
    data: *const CArray2D,
    result: *mut *mut CAnalysisResult,
) -> CdfaErrorCode {
    if handle.is_null() || data.is_null() || result.is_null() {
        set_last_error(&CdfaError::invalid_input("Null pointer provided"));
        return CdfaErrorCode::InvalidInput;
    }

    let handle = unsafe { &*handle };
    let data_array = unsafe { &*data };

    let (analysis_result, error_code) = handle_result(|| -> Result<AnalysisResult> {
        let ndarray = data_array.to_ndarray()?;
        let cdfa = handle.cdfa.read();
        cdfa.analyze(&ndarray.view())
    }());

    if error_code != CdfaErrorCode::Success {
        return error_code;
    }

    // Convert result to C structure
    let c_result = Box::new(CAnalysisResult {
        data: CArray1D::from_ndarray(&analysis_result.data),
        metrics_count: analysis_result.metrics.len() as c_uint,
        metric_names: ptr::null_mut(), // TODO: Implement metric names export
        metric_values: ptr::null_mut(), // TODO: Implement metric values export
        patterns_count: analysis_result.patterns.len() as c_uint,
        execution_time_us: analysis_result.performance.execution_time_us,
        error_code: CdfaErrorCode::Success,
    });

    unsafe {
        *result = Box::into_raw(c_result);
    }

    CdfaErrorCode::Success
}

#[no_mangle]
pub extern "C" fn cdfa_validate_data(data: *const CArray2D) -> CdfaErrorCode {
    if data.is_null() {
        return CdfaErrorCode::InvalidInput;
    }

    let data_array = unsafe { &*data };
    
    // Basic validation
    if data_array.rows < 2 || data_array.cols < 2 {
        return CdfaErrorCode::InvalidInput;
    }

    if data_array.data.is_null() {
        return CdfaErrorCode::InvalidInput;
    }

    // Check for NaN or infinite values in financial data
    let slice = unsafe {
        std::slice::from_raw_parts(data_array.data, (data_array.rows * data_array.cols) as usize)
    };

    for &value in slice {
        if !value.is_finite() {
            return CdfaErrorCode::InvalidInput;
        }
        
        // Additional financial validation
        if value.abs() > 1e15 {  // Unreasonably large financial values
            return CdfaErrorCode::InvalidInput;
        }
    }

    CdfaErrorCode::Success
}

#[no_mangle]
pub extern "C" fn cdfa_get_version() -> *const c_char {
    static VERSION_CSTRING: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION_CSTRING.get_or_init(|| {
        CString::new(crate::VERSION).unwrap_or_else(|_| CString::new("unknown").unwrap())
    }).as_ptr()
}

// Additional utility functions for memory management
#[no_mangle]
pub extern "C" fn cdfa_alloc_array2d(rows: c_uint, cols: c_uint) -> *mut CArray2D {
    if rows == 0 || cols == 0 {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(CArray2D::new(rows as usize, cols as usize)))
}

#[no_mangle]
pub extern "C" fn cdfa_alloc_array1d(len: c_uint) -> *mut CArray1D {
    if len == 0 {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(CArray1D::new(len as usize)))
}

#[no_mangle]
pub extern "C" fn cdfa_free_array2d(array: *mut CArray2D) {
    if !array.is_null() {
        unsafe {
            let _boxed = Box::from_raw(array);
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_free_array1d(array: *mut CArray1D) {
    if !array.is_null() {
        unsafe {
            let _boxed = Box::from_raw(array);
        }
    }
}

#[no_mangle]
pub extern "C" fn cdfa_free_result(result: *mut CAnalysisResult) {
    if !result.is_null() {
        unsafe {
            let _boxed = Box::from_raw(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_c_api_lifecycle() {
        // Test handle creation and destruction
        let handle = cdfa_create();
        assert!(!handle.is_null());
        
        cdfa_destroy(handle);
    }

    #[test]
    fn test_array_conversion() {
        let rust_array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let c_array = CArray2D::from_ndarray(&rust_array);
        let converted_back = c_array.to_ndarray().unwrap();
        
        assert_eq!(rust_array, converted_back);
    }

    #[test]
    fn test_data_validation() {
        let valid_data = CArray2D::new(3, 3);
        assert_eq!(cdfa_validate_data(&valid_data), CdfaErrorCode::Success);
        
        // Test invalid data
        let invalid_data = CArray2D::from_ptr(ptr::null_mut(), 1, 1);
        assert_eq!(cdfa_validate_data(&invalid_data), CdfaErrorCode::InvalidInput);
    }

    #[test]
    fn test_financial_validation() {
        // Test with financial constraints
        let mut data = CArray2D::new(3, 3);
        let slice = unsafe {
            std::slice::from_raw_parts_mut(data.data, 9)
        };
        
        // Fill with reasonable financial data
        for (i, val) in slice.iter_mut().enumerate() {
            *val = (i + 1) as f64 * 100.0;  // Stock prices
        }
        
        assert_eq!(cdfa_validate_data(&data), CdfaErrorCode::Success);
        
        // Test with unreasonable values
        slice[0] = 1e16;  // Unreasonably large
        assert_eq!(cdfa_validate_data(&data), CdfaErrorCode::InvalidInput);
    }
}