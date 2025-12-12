//! Callback mechanism for async operations
//!
//! Allows C/C++ code to register callbacks for:
//! - Cognitive loop iterations
//! - State changes
//! - Error notifications
//! - Emergence events

use super::types::*;
use std::os::raw::c_void;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Callback function type
///
/// # Arguments
/// - `context`: User-provided context pointer
/// - `event_type`: Type of event (0=iteration, 1=state_change, 2=error, 3=emergence)
/// - `data`: Event-specific data pointer
/// - `data_len`: Size of data in bytes
pub type QksCallback = extern "C" fn(
    context: *mut c_void,
    event_type: u32,
    data: *const u8,
    data_len: usize,
);

/// Event types for callbacks
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QksEventType {
    /// Cognitive loop iteration completed
    Iteration = 0,

    /// Plugin state changed
    StateChange = 1,

    /// Error occurred
    Error = 2,

    /// Emergence event detected
    Emergence = 3,
}

/// Callback context (stored with each registered callback)
#[derive(Debug)]
pub struct QksCallbackContext {
    /// User callback function
    callback: QksCallback,

    /// User-provided context pointer
    context: *mut c_void,
}

unsafe impl Send for QksCallbackContext {}
unsafe impl Sync for QksCallbackContext {}

/// Global callback registry
pub struct CallbackRegistry {
    /// Map from handle ID to callbacks
    callbacks: RwLock<HashMap<u64, Vec<QksCallbackContext>>>,
}

impl CallbackRegistry {
    pub fn new() -> Self {
        Self {
            callbacks: RwLock::new(HashMap::new()),
        }
    }

    /// Register a callback for a handle
    pub fn register(&self, handle_id: u64, callback: QksCallback, context: *mut c_void) {
        let mut callbacks = self.callbacks.write();
        let ctx = QksCallbackContext { callback, context };

        callbacks
            .entry(handle_id)
            .or_insert_with(Vec::new)
            .push(ctx);
    }

    /// Unregister all callbacks for a handle
    pub fn unregister(&self, handle_id: u64) {
        let mut callbacks = self.callbacks.write();
        callbacks.remove(&handle_id);
    }

    /// Invoke all callbacks for a handle
    pub fn invoke(&self, handle_id: u64, event_type: QksEventType, data: &[u8]) {
        let callbacks = self.callbacks.read();
        if let Some(contexts) = callbacks.get(&handle_id) {
            for ctx in contexts {
                (ctx.callback)(
                    ctx.context,
                    event_type as u32,
                    data.as_ptr(),
                    data.len(),
                );
            }
        }
    }
}

impl Default for CallbackRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global callback registry instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_CALLBACKS: CallbackRegistry = CallbackRegistry::new();
}

/// Set a callback for a QKS plugin instance
///
/// # Safety
/// - `handle` must be a valid handle
/// - `callback` must be a valid function pointer
/// - `context` will be passed to callback (may be null)
#[no_mangle]
pub extern "C" fn qks_set_callback(
    handle: QksHandle,
    callback: QksCallback,
    context: *mut c_void,
) -> QksResult {
    use super::types::is_null_handle;
    use crate::error::{set_last_error, QksError};

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let handle_id = handle.inner as u64;
    GLOBAL_CALLBACKS.register(handle_id, callback, context);

    QKS_SUCCESS
}

/// Remove all callbacks for a handle
///
/// # Safety
/// - `handle` must be a valid handle
#[no_mangle]
pub extern "C" fn qks_clear_callbacks(handle: QksHandle) -> QksResult {
    use super::types::is_null_handle;
    use crate::error::{set_last_error, QksError};

    if is_null_handle(handle) {
        set_last_error(&QksError::NullPointer);
        return QKS_ERROR_NULL_POINTER;
    }

    let handle_id = handle.inner as u64;
    GLOBAL_CALLBACKS.unregister(handle_id);

    QKS_SUCCESS
}

/// Helper: Invoke callbacks from plugin code
pub(crate) fn invoke_callbacks(handle_id: u64, event_type: QksEventType, data: &[u8]) {
    GLOBAL_CALLBACKS.invoke(handle_id, event_type, data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_registry() {
        let registry = CallbackRegistry::new();

        extern "C" fn test_callback(_ctx: *mut c_void, _evt: u32, _data: *const u8, _len: usize) {
            // Test callback
        }

        let handle_id = 42;
        registry.register(handle_id, test_callback, std::ptr::null_mut());

        // Invoke should not crash
        registry.invoke(handle_id, QksEventType::Iteration, &[1, 2, 3]);

        registry.unregister(handle_id);
    }
}
