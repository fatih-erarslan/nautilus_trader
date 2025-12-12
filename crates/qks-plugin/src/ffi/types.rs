//! FFI-safe type definitions
//!
//! All types in this module are repr(C) for C ABI compatibility

use std::os::raw::{c_char, c_int, c_void};

/// Opaque handle to QKS plugin instance
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QksHandle {
    /// Internal pointer (opaque to C)
    pub(crate) inner: *mut c_void,
    /// Version tag for ABA problem prevention
    pub(crate) version: u32,
}

/// Opaque handle to QKS configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QksConfigHandle {
    pub(crate) inner: *mut c_void,
    pub(crate) version: u32,
}

/// Opaque handle to QKS state
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QksStateHandle {
    pub(crate) inner: *mut c_void,
    pub(crate) version: u32,
}

/// FFI result type (error code)
pub type QksResult = c_int;

/// Success code
pub const QKS_SUCCESS: QksResult = 0;

/// Error codes (negative values)
pub const QKS_ERROR_GENERIC: QksResult = -1;
pub const QKS_ERROR_INVALID_HANDLE: QksResult = -2;
pub const QKS_ERROR_NULL_POINTER: QksResult = -3;
pub const QKS_ERROR_INVALID_CONFIG: QksResult = -4;
pub const QKS_ERROR_LAYER_NOT_INITIALIZED: QksResult = -5;
pub const QKS_ERROR_OUT_OF_MEMORY: QksResult = -7;
pub const QKS_ERROR_INTERNAL: QksResult = -8;

/// Plugin state snapshot (C-compatible)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QksState {
    /// Current Phi (consciousness) value
    pub phi: f64,

    /// Current energy level (0.0 to 1.0)
    pub energy: f64,

    /// Homeostasis stability (0.0 to 1.0)
    pub stability: f64,

    /// Number of active layers
    pub active_layers: u8,

    /// Total iterations executed
    pub total_iterations: u64,

    /// Average iteration time (microseconds)
    pub avg_iteration_us: f64,

    /// Plugin state (0=uninitialized, 1=ready, 2=active, 3=paused, 4=shutdown)
    pub plugin_state: u8,

    /// Layer status bitmap (bit N = layer N initialized)
    pub layer_status: u8,
}

/// Configuration parameters (C-compatible)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QksConfigParams {
    /// Consciousness threshold
    pub phi_threshold: f64,

    /// Energy setpoint
    pub energy_setpoint: f64,

    /// Enable meta-learning (0=false, 1=true)
    pub enable_meta_learning: u8,

    /// Enable collective intelligence
    pub enable_collective: u8,

    /// Enable GPU acceleration
    pub enable_gpu: u8,

    /// Max iterations per cycle
    pub max_iterations: u32,

    /// Energy budget per cycle
    pub energy_budget: f64,

    /// Thread pool size (0=auto)
    pub thread_pool_size: u32,

    /// PID gains
    pub pid_kp: f64,
    pub pid_ki: f64,
    pub pid_kd: f64,
}

impl Default for QksConfigParams {
    fn default() -> Self {
        Self {
            phi_threshold: 1.0,
            energy_setpoint: 0.7,
            enable_meta_learning: 1,
            enable_collective: 1,
            enable_gpu: if cfg!(target_os = "macos") { 1 } else { 0 },
            max_iterations: 100,
            energy_budget: 100.0,
            thread_pool_size: 0,
            pid_kp: 1.0,
            pid_ki: 0.1,
            pid_kd: 0.05,
        }
    }
}

/// Null handle constants
pub const QKS_NULL_HANDLE: QksHandle = QksHandle {
    inner: std::ptr::null_mut(),
    version: 0,
};

pub const QKS_NULL_CONFIG: QksConfigHandle = QksConfigHandle {
    inner: std::ptr::null_mut(),
    version: 0,
};

pub const QKS_NULL_STATE: QksStateHandle = QksStateHandle {
    inner: std::ptr::null_mut(),
    version: 0,
};

/// Check if handle is null
#[inline]
pub fn is_null_handle(handle: QksHandle) -> bool {
    handle.inner.is_null()
}

#[inline]
pub fn is_null_config(handle: QksConfigHandle) -> bool {
    handle.inner.is_null()
}

#[inline]
pub fn is_null_state(handle: QksStateHandle) -> bool {
    handle.inner.is_null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_handle_check() {
        assert!(is_null_handle(QKS_NULL_HANDLE));

        let non_null = QksHandle {
            inner: 0x1234 as *mut c_void,
            version: 1,
        };
        assert!(!is_null_handle(non_null));
    }

    #[test]
    fn test_default_config_params() {
        let params = QksConfigParams::default();
        assert_eq!(params.phi_threshold, 1.0);
        assert_eq!(params.enable_meta_learning, 1);
    }

    #[test]
    fn test_size_alignment() {
        // Ensure structures are properly aligned for C
        assert_eq!(std::mem::align_of::<QksHandle>(), std::mem::align_of::<*mut c_void>());
        assert_eq!(std::mem::align_of::<QksState>(), std::mem::align_of::<f64>());
    }
}
