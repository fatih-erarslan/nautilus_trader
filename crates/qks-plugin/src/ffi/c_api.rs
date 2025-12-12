//! C-compatible FFI functions
//!
//! All functions are marked with #[no_mangle] and extern "C" for C ABI compatibility
//!
//! ## Safety
//!
//! All functions perform null pointer checks and handle panics gracefully.
//! Errors are returned via error codes and stored in thread-local storage.

use super::types::*;
use crate::config::{QksConfig, QksConfigBuilder};
use crate::plugin::QksPlugin;
use crate::error::{QksError, set_last_error, clear_last_error};
use crate::handle::{GLOBAL_REGISTRIES, HandleType};
use std::os::raw::{c_char, c_void};
use std::ffi::{CStr, CString};
use std::panic;

/// Create a new QKS plugin instance with default configuration
///
/// # Safety
/// Returns a valid handle on success, NULL_HANDLE on failure
#[no_mangle]
pub extern "C" fn qks_create() -> QksHandle {
    qks_create_with_config(std::ptr::null())
}

/// Create a new QKS plugin instance with custom configuration
///
/// # Safety
/// - `config` may be null (will use defaults)
/// - Returns a valid handle on success, NULL_HANDLE on failure
#[no_mangle]
pub extern "C" fn qks_create_with_config(config: *const QksConfigParams) -> QksHandle {
    clear_last_error();

    let result = panic::catch_unwind(|| {
        let qks_config = if config.is_null() {
            QksConfig::default()
        } else {
            let params = unsafe { &*config };
            convert_ffi_config(params)
        };

        let plugin = QksPlugin::new(qks_config);
        let opaque_handle = GLOBAL_REGISTRIES.plugins.insert(plugin);

        QksHandle {
            inner: opaque_handle.id as *mut c_void,
            version: opaque_handle.version,
        }
    });

    match result {
        Ok(handle) => handle,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_create".to_string()));
            QKS_NULL_HANDLE
        }
    }
}

/// Destroy a QKS plugin instance
///
/// # Safety
/// - `handle` must be a valid handle from qks_create
/// - After this call, the handle is invalid and must not be used
#[no_mangle]
pub extern "C" fn qks_destroy(handle: QksHandle) -> QksResult {
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.remove(opaque)
            .map(|_| QKS_SUCCESS)
            .unwrap_or_else(|e| {
                set_last_error(&e);
                QKS_ERROR_INVALID_HANDLE
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_destroy".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

/// Initialize the QKS plugin
///
/// # Safety
/// - `handle` must be a valid handle
#[no_mangle]
pub extern "C" fn qks_initialize(handle: QksHandle) -> QksResult {
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.get(opaque)
            .and_then(|plugin_arc| {
                // Get mutable access (requires interior mutability or Arc<RwLock<>>)
                // For now, we'll use unsafe to get mutable access
                // In production, plugin should use Arc<RwLock<QksPlugin>>
                let plugin_ptr = Arc::into_raw(plugin_arc) as *mut QksPlugin;
                let plugin = unsafe { &mut *plugin_ptr };
                let result = plugin.initialize();

                // Convert back to Arc
                unsafe { Arc::from_raw(plugin_ptr); }

                result
            })
            .map(|_| QKS_SUCCESS)
            .unwrap_or_else(|e| {
                set_last_error(&e);
                match e {
                    QksError::InvalidHandle => QKS_ERROR_INVALID_HANDLE,
                    QksError::NullPointer => QKS_ERROR_NULL_POINTER,
                    _ => QKS_ERROR_GENERIC,
                }
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_initialize".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

/// Start the QKS plugin
///
/// # Safety
/// - `handle` must be a valid, initialized handle
#[no_mangle]
pub extern "C" fn qks_start(handle: QksHandle) -> QksResult {
    modify_plugin(handle, |plugin| plugin.start())
}

/// Stop the QKS plugin
///
/// # Safety
/// - `handle` must be a valid handle
#[no_mangle]
pub extern "C" fn qks_stop(handle: QksHandle) -> QksResult {
    modify_plugin(handle, |plugin| plugin.shutdown())
}

/// Process input data through the cognitive system
///
/// # Safety
/// - `handle` must be a valid, active handle
/// - `input` must be a valid pointer to `input_len` bytes
/// - `output` must be a valid pointer to buffer with `output_capacity` bytes
/// - `output_len` will be set to actual output size
#[no_mangle]
pub extern "C" fn qks_process(
    handle: QksHandle,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
    output_len: *mut usize,
) -> QksResult {
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    if input.is_null() || output.is_null() || output_len.is_null() {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let input_slice = unsafe { std::slice::from_raw_parts(input, input_len) };

        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.get(opaque)
            .and_then(|plugin_arc| {
                let plugin_ptr = Arc::into_raw(plugin_arc) as *mut QksPlugin;
                let plugin = unsafe { &mut *plugin_ptr };

                let result = plugin.process(input_slice)
                    .and_then(|output_data| {
                        if output_data.len() > output_capacity {
                            Err(QksError::BufferTooSmall {
                                required: output_data.len(),
                                provided: output_capacity,
                            })
                        } else {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    output_data.as_ptr(),
                                    output,
                                    output_data.len()
                                );
                                *output_len = output_data.len();
                            }
                            Ok(())
                        }
                    });

                unsafe { Arc::from_raw(plugin_ptr); }
                result
            })
            .map(|_| QKS_SUCCESS)
            .unwrap_or_else(|e| {
                set_last_error(&e);
                match e {
                    QksError::BufferTooSmall { .. } => QKS_ERROR_GENERIC,
                    _ => QKS_ERROR_GENERIC,
                }
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_process".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

/// Get current Phi (consciousness) value
///
/// # Safety
/// - `handle` must be a valid handle
/// - `phi` must be a valid pointer
#[no_mangle]
pub extern "C" fn qks_get_phi(handle: QksHandle, phi: *mut f64) -> QksResult {
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    if phi.is_null() {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.get(opaque)
            .and_then(|plugin_arc| {
                plugin_arc.get_phi()
                    .map(|phi_value| {
                        unsafe { *phi = phi_value; }
                        QKS_SUCCESS
                    })
            })
            .unwrap_or_else(|e| {
                set_last_error(&e);
                QKS_ERROR_GENERIC
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_get_phi".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

/// Get current plugin state
///
/// # Safety
/// - `handle` must be a valid handle
/// - `state` must be a valid pointer
#[no_mangle]
pub extern "C" fn qks_get_state(handle: QksHandle, state: *mut QksState) -> QksResult {
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    if state.is_null() {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.get(opaque)
            .map(|plugin_arc| {
                let metrics = plugin_arc.get_metrics();
                let plugin_state = plugin_arc.get_plugin_state();
                let layer_status = plugin_arc.get_layer_status();

                let qks_state = QksState {
                    phi: metrics.current_phi,
                    energy: metrics.current_energy,
                    stability: metrics.homeostasis_stability,
                    active_layers: metrics.active_layers as u8,
                    total_iterations: metrics.total_iterations,
                    avg_iteration_us: metrics.avg_iteration_us,
                    plugin_state: plugin_state as u8,
                    layer_status: layer_status.iter().enumerate()
                        .fold(0u8, |acc, (i, &initialized)| {
                            if initialized { acc | (1 << i) } else { acc }
                        }),
                };

                unsafe { *state = qks_state; }
                QKS_SUCCESS
            })
            .unwrap_or_else(|e| {
                set_last_error(&e);
                QKS_ERROR_INVALID_HANDLE
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in qks_get_state".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

/// Get last error message
///
/// # Safety
/// - Returns pointer to static string (do not free)
/// - String is valid until next QKS call
#[no_mangle]
pub extern "C" fn qks_get_error_message() -> *const c_char {
    use crate::error::get_last_error;

    thread_local! {
        static LAST_ERROR_MSG: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
    }

    LAST_ERROR_MSG.with(|cell| {
        let msg = get_last_error().unwrap_or_else(|| "No error".to_string());
        let c_string = CString::new(msg).unwrap_or_else(|_| CString::new("Invalid error message").unwrap());
        let ptr = c_string.as_ptr();
        *cell.borrow_mut() = Some(c_string);
        ptr
    })
}

// Helper functions

fn convert_ffi_config(params: &QksConfigParams) -> QksConfig {
    QksConfigBuilder::new()
        .phi_threshold(params.phi_threshold)
        .energy_setpoint(params.energy_setpoint)
        .meta_learning(params.enable_meta_learning != 0)
        .collective(params.enable_collective != 0)
        .gpu(params.enable_gpu != 0)
        .max_iterations(params.max_iterations as usize)
        .energy_budget(params.energy_budget)
        .threads(params.thread_pool_size as usize)
        .pid_gains(params.pid_kp, params.pid_ki, params.pid_kd)
        .build()
}

fn modify_plugin<F>(handle: QksHandle, f: F) -> QksResult
where
    F: FnOnce(&mut QksPlugin) -> Result<(), QksError> + panic::UnwindSafe,
{
    clear_last_error();

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let result = panic::catch_unwind(|| {
        let opaque = crate::handle::OpaqueHandle {
            id: handle.inner as u64,
            version: handle.version,
            type_tag: HandleType::Plugin.as_u32(),
        };

        GLOBAL_REGISTRIES.plugins.get(opaque)
            .and_then(|plugin_arc| {
                let plugin_ptr = Arc::into_raw(plugin_arc) as *mut QksPlugin;
                let plugin = unsafe { &mut *plugin_ptr };
                let result = f(plugin);
                unsafe { Arc::from_raw(plugin_ptr); }
                result
            })
            .map(|_| QKS_SUCCESS)
            .unwrap_or_else(|e| {
                set_last_error(&e);
                QKS_ERROR_GENERIC
            })
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error(&QksError::Internal("Panic in modify_plugin".to_string()));
            QKS_ERROR_INTERNAL
        }
    }
}

use std::sync::Arc;
