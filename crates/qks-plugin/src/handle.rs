//! Opaque handle management for safe FFI
//!
//! This module provides type-safe, memory-safe handle management:
//! - Opaque pointers for FFI boundary
//! - Handle validation and lifecycle management
//! - Thread-safe handle registry
//! - Automatic cleanup on drop

use crate::error::{QksError, QksResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque handle for FFI (void* equivalent)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpaqueHandle {
    /// Unique handle ID
    pub id: u64,

    /// Version number for ABA problem prevention
    pub version: u32,

    /// Handle type tag for runtime validation
    pub type_tag: u32,
}

/// Handle type tags
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleType {
    Plugin = 1,
    Runtime = 2,
    State = 3,
    Config = 4,
    Layer = 5,
}

impl HandleType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn from_u32(tag: u32) -> Option<Self> {
        match tag {
            1 => Some(Self::Plugin),
            2 => Some(Self::Runtime),
            3 => Some(Self::State),
            4 => Some(Self::Config),
            5 => Some(Self::Layer),
            _ => None,
        }
    }
}

impl OpaqueHandle {
    /// Create a new opaque handle
    pub fn new(id: u64, handle_type: HandleType) -> Self {
        Self {
            id,
            version: 1,
            type_tag: handle_type.as_u32(),
        }
    }

    /// Check if handle is null (id = 0)
    pub fn is_null(&self) -> bool {
        self.id == 0
    }

    /// Get handle type
    pub fn handle_type(&self) -> Option<HandleType> {
        HandleType::from_u32(self.type_tag)
    }

    /// Validate handle type
    pub fn validate_type(&self, expected: HandleType) -> QksResult<()> {
        if self.type_tag == expected.as_u32() {
            Ok(())
        } else {
            Err(QksError::InvalidHandle)
        }
    }
}

/// Null handle constant
pub const NULL_HANDLE: OpaqueHandle = OpaqueHandle {
    id: 0,
    version: 0,
    type_tag: 0,
};

/// Handle registry for managing opaque handles
pub struct HandleRegistry<T> {
    /// Map from handle ID to stored value
    handles: RwLock<HashMap<u64, Arc<T>>>,

    /// Next available handle ID
    next_id: AtomicU64,

    /// Handle type for this registry
    handle_type: HandleType,
}

impl<T> HandleRegistry<T> {
    /// Create a new handle registry
    pub fn new(handle_type: HandleType) -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1), // Start from 1 (0 is reserved for NULL)
            handle_type,
        }
    }

    /// Insert a value and return its handle
    pub fn insert(&self, value: T) -> OpaqueHandle {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let handle = OpaqueHandle::new(id, self.handle_type);

        let mut handles = self.handles.write();
        handles.insert(id, Arc::new(value));

        handle
    }

    /// Get a value by handle (immutable reference)
    pub fn get(&self, handle: OpaqueHandle) -> QksResult<Arc<T>> {
        // Validate handle type
        handle.validate_type(self.handle_type)?;

        // Check if handle is null
        if handle.is_null() {
            return Err(QksError::NullPointer);
        }

        // Get value from registry
        let handles = self.handles.read();
        handles
            .get(&handle.id)
            .cloned()
            .ok_or(QksError::InvalidHandle)
    }

    /// Remove a value by handle
    pub fn remove(&self, handle: OpaqueHandle) -> QksResult<Arc<T>> {
        // Validate handle type
        handle.validate_type(self.handle_type)?;

        // Check if handle is null
        if handle.is_null() {
            return Err(QksError::NullPointer);
        }

        // Remove from registry
        let mut handles = self.handles.write();
        handles
            .remove(&handle.id)
            .ok_or(QksError::InvalidHandle)
    }

    /// Check if handle exists
    pub fn contains(&self, handle: OpaqueHandle) -> bool {
        if handle.is_null() {
            return false;
        }

        let handles = self.handles.read();
        handles.contains_key(&handle.id)
    }

    /// Get count of active handles
    pub fn len(&self) -> usize {
        self.handles.read().len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.handles.read().is_empty()
    }

    /// Clear all handles
    pub fn clear(&self) {
        let mut handles = self.handles.write();
        handles.clear();
    }
}

/// Global handle registries (one per handle type)
pub struct GlobalHandleRegistries {
    pub plugins: HandleRegistry<crate::plugin::QksPlugin>,
    pub runtimes: HandleRegistry<crate::runtime::CognitiveRuntime>,
    pub states: HandleRegistry<crate::plugin::PluginState>,
    pub configs: HandleRegistry<crate::config::QksConfig>,
}

impl GlobalHandleRegistries {
    pub fn new() -> Self {
        Self {
            plugins: HandleRegistry::new(HandleType::Plugin),
            runtimes: HandleRegistry::new(HandleType::Runtime),
            states: HandleRegistry::new(HandleType::State),
            configs: HandleRegistry::new(HandleType::Config),
        }
    }
}

impl Default for GlobalHandleRegistries {
    fn default() -> Self {
        Self::new()
    }
}

/// Global handle registries instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_REGISTRIES: GlobalHandleRegistries = GlobalHandleRegistries::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opaque_handle_creation() {
        let handle = OpaqueHandle::new(42, HandleType::Plugin);
        assert_eq!(handle.id, 42);
        assert_eq!(handle.version, 1);
        assert_eq!(handle.type_tag, HandleType::Plugin.as_u32());
        assert!(!handle.is_null());
    }

    #[test]
    fn test_null_handle() {
        assert_eq!(NULL_HANDLE.id, 0);
        assert!(NULL_HANDLE.is_null());
    }

    #[test]
    fn test_handle_type_validation() {
        let handle = OpaqueHandle::new(1, HandleType::Plugin);
        assert!(handle.validate_type(HandleType::Plugin).is_ok());
        assert!(handle.validate_type(HandleType::Runtime).is_err());
    }

    #[test]
    fn test_handle_registry() {
        let registry: HandleRegistry<i32> = HandleRegistry::new(HandleType::Plugin);

        // Insert value
        let handle = registry.insert(42);
        assert!(!handle.is_null());
        assert_eq!(registry.len(), 1);

        // Get value
        let value = registry.get(handle).unwrap();
        assert_eq!(*value, 42);

        // Remove value
        let removed = registry.remove(handle).unwrap();
        assert_eq!(*removed, 42);
        assert_eq!(registry.len(), 0);

        // Try to get removed handle
        assert!(registry.get(handle).is_err());
    }

    #[test]
    fn test_handle_registry_invalid_type() {
        let registry: HandleRegistry<i32> = HandleRegistry::new(HandleType::Plugin);

        // Create handle with wrong type
        let wrong_handle = OpaqueHandle::new(1, HandleType::Runtime);

        // Should fail type validation
        assert!(registry.get(wrong_handle).is_err());
    }

    #[test]
    fn test_handle_registry_null() {
        let registry: HandleRegistry<i32> = HandleRegistry::new(HandleType::Plugin);

        // Try to get null handle
        assert!(registry.get(NULL_HANDLE).is_err());
    }
}
