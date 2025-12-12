//! Swift FFI bindings for HyperPhysics Cognition System
//!
//! This crate provides a C-compatible API that can be called from Swift.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │           SWIFT APPLICATION                         │
//! │  (Sentry macOS app)                                 │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │      SWIFT WRAPPER CLASSES                          │
//! │  (CognitionSystem.swift)                            │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │         C FFI LAYER (THIS CRATE)                    │
//! │  hyperphysics_cognition_*() functions               │
//! └─────────────────┬───────────────────────────────────┘
//!                   │
//!                   ▼
//! ┌─────────────────────────────────────────────────────┐
//! │    RUST COGNITION SYSTEM                            │
//! │  hyperphysics-cognition crate                       │
//! └─────────────────────────────────────────────────────┘
//! ```

use hyperphysics_cognition::prelude::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::sync::Once;
use tracing::{error, info};

static INIT: Once = Once::new();

/// Initialize tracing (call once at startup)
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_init_tracing() {
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter("hyperphysics_cognition=debug")
            .init();
        info!("🧠 HyperPhysics Cognition FFI initialized");
    });
}

// ============================================================================
// Opaque Pointers (Swift sees these as OpaquePointer)
// ============================================================================

/// Opaque pointer to CognitionSystem
pub type CognitionSystemHandle = *mut c_void;

// ============================================================================
// Configuration
// ============================================================================

/// C-compatible cognition configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CCognitionConfig {
    pub enable_attention: bool,
    pub enable_loops: bool,
    pub enable_dream: bool,
    pub enable_learning: bool,
    pub enable_integration: bool,
    pub default_curvature: c_double,
    pub loop_frequency: c_double,
    pub dream_threshold: c_double,
}

impl From<CCognitionConfig> for CognitionConfig {
    fn from(c: CCognitionConfig) -> Self {
        Self {
            enable_attention: c.enable_attention,
            enable_loops: c.enable_loops,
            enable_dream: c.enable_dream,
            enable_learning: c.enable_learning,
            enable_integration: c.enable_integration,
            default_curvature: c.default_curvature,
            loop_frequency: c.loop_frequency,
            dream_threshold: c.dream_threshold,
        }
    }
}

/// Create default configuration
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_config_default() -> CCognitionConfig {
    let config = CognitionConfig::default();
    CCognitionConfig {
        enable_attention: config.enable_attention,
        enable_loops: config.enable_loops,
        enable_dream: config.enable_dream,
        enable_learning: config.enable_learning,
        enable_integration: config.enable_integration,
        default_curvature: config.default_curvature,
        loop_frequency: config.loop_frequency,
        dream_threshold: config.dream_threshold,
    }
}

// ============================================================================
// Cognition System
// ============================================================================

/// Create new cognition system
///
/// # Returns
/// Handle to cognition system, or NULL on error
///
/// # Safety
/// The returned handle must be freed with `hyperphysics_cognition_destroy()`
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_create(
    config: CCognitionConfig,
) -> CognitionSystemHandle {
    let rust_config: CognitionConfig = config.into();

    match CognitionSystem::new(rust_config) {
        Ok(system) => {
            let boxed = Box::new(system);
            Box::into_raw(boxed) as CognitionSystemHandle
        }
        Err(e) => {
            error!("Failed to create cognition system: {}", e);
            ptr::null_mut()
        }
    }
}

/// Destroy cognition system
///
/// # Safety
/// - Handle must be valid (returned from `hyperphysics_cognition_create`)
/// - Handle must not be used after this call
/// - Handle must not be freed twice
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_destroy(handle: CognitionSystemHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut CognitionSystem);
    }
}

/// Get current arousal level
///
/// # Safety
/// Handle must be valid
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_get_arousal(
    handle: CognitionSystemHandle,
) -> c_double {
    if handle.is_null() {
        return 0.0;
    }

    let system = &*(handle as *const CognitionSystem);
    system.arousal().value()
}

/// Set arousal level
///
/// # Safety
/// Handle must be valid
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_set_arousal(
    handle: CognitionSystemHandle,
    level: c_double,
) {
    if handle.is_null() {
        return;
    }

    let system = &*(handle as *const CognitionSystem);
    system.set_arousal(ArousalLevel::new(level));
}

/// Get current cognitive load
///
/// # Safety
/// Handle must be valid
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_get_load(
    handle: CognitionSystemHandle,
) -> c_double {
    if handle.is_null() {
        return 0.0;
    }

    let system = &*(handle as *const CognitionSystem);
    system.cognitive_load().value()
}

/// Set cognitive load
///
/// # Safety
/// Handle must be valid
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_set_load(
    handle: CognitionSystemHandle,
    load: c_double,
) {
    if handle.is_null() {
        return;
    }

    let system = &*(handle as *const CognitionSystem);
    system.set_cognitive_load(CognitiveLoad::new(load));
}

/// Check if system is healthy
///
/// # Safety
/// Handle must be valid
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_is_healthy(
    handle: CognitionSystemHandle,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let system = &*(handle as *const CognitionSystem);
    system.is_healthy()
}

// ============================================================================
// Cognition Phases
// ============================================================================

/// Cognition phase enum (must match Rust enum)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCognitionPhase {
    Perceiving = 0,
    Cognizing = 1,
    Deliberating = 2,
    Intending = 3,
    Integrating = 4,
    Acting = 5,
}

impl From<CognitionPhase> for CCognitionPhase {
    fn from(phase: CognitionPhase) -> Self {
        match phase {
            CognitionPhase::Perceiving => CCognitionPhase::Perceiving,
            CognitionPhase::Cognizing => CCognitionPhase::Cognizing,
            CognitionPhase::Deliberating => CCognitionPhase::Deliberating,
            CognitionPhase::Intending => CCognitionPhase::Intending,
            CognitionPhase::Integrating => CCognitionPhase::Integrating,
            CognitionPhase::Acting => CCognitionPhase::Acting,
        }
    }
}

impl From<CCognitionPhase> for CognitionPhase {
    fn from(phase: CCognitionPhase) -> Self {
        match phase {
            CCognitionPhase::Perceiving => CognitionPhase::Perceiving,
            CCognitionPhase::Cognizing => CognitionPhase::Cognizing,
            CCognitionPhase::Deliberating => CognitionPhase::Deliberating,
            CCognitionPhase::Intending => CognitionPhase::Intending,
            CCognitionPhase::Integrating => CognitionPhase::Integrating,
            CCognitionPhase::Acting => CognitionPhase::Acting,
        }
    }
}

/// Get next phase in loop
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_phase_next(phase: CCognitionPhase) -> CCognitionPhase {
    let rust_phase: CognitionPhase = phase.into();
    rust_phase.next().into()
}

/// Get phase name (caller must free the returned string)
///
/// # Safety
/// The returned string must be freed with `hyperphysics_cognition_free_string()`
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_phase_name(phase: CCognitionPhase) -> *mut c_char {
    let rust_phase: CognitionPhase = phase.into();
    let name = rust_phase.name();

    match CString::new(name) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by the FFI
///
/// # Safety
/// - ptr must have been returned by an FFI function that allocates strings
/// - ptr must not be used after this call
/// - ptr must not be freed twice
#[no_mangle]
pub unsafe extern "C" fn hyperphysics_cognition_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

// ============================================================================
// Version Info
// ============================================================================

/// Get version string (caller must free)
///
/// # Safety
/// The returned string must be freed with `hyperphysics_cognition_free_string()`
#[no_mangle]
pub extern "C" fn hyperphysics_cognition_version() -> *mut c_char {
    let version = env!("CARGO_PKG_VERSION");
    match CString::new(version) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            let config = hyperphysics_cognition_config_default();
            let handle = hyperphysics_cognition_create(config);
            assert!(!handle.is_null());

            assert!(hyperphysics_cognition_is_healthy(handle));

            hyperphysics_cognition_set_arousal(handle, 0.8);
            let arousal = hyperphysics_cognition_get_arousal(handle);
            assert!((arousal - 0.8).abs() < 1e-6);

            hyperphysics_cognition_destroy(handle);
        }
    }

    #[test]
    fn test_phase_cycle() {
        let mut phase = CCognitionPhase::Perceiving;
        phase = hyperphysics_cognition_phase_next(phase);
        assert_eq!(phase, CCognitionPhase::Cognizing);
        phase = hyperphysics_cognition_phase_next(phase);
        assert_eq!(phase, CCognitionPhase::Deliberating);
    }

    #[test]
    fn test_phase_name() {
        unsafe {
            let name_ptr = hyperphysics_cognition_phase_name(CCognitionPhase::Perceiving);
            assert!(!name_ptr.is_null());

            let c_str = CStr::from_ptr(name_ptr);
            let name = c_str.to_str().unwrap();
            assert_eq!(name, "Perception");

            hyperphysics_cognition_free_string(name_ptr);
        }
    }
}
