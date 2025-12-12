//! # WebAssembly Bindings
//!
//! WASM bindings for browser/edge deployment.
//! Placeholder for future implementation.

use wasm_bindgen::prelude::*;

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get version (WASM)
#[wasm_bindgen]
pub fn wasm_version() -> String {
    crate::version().to_string()
}

/// Get sentinel count (WASM)
#[wasm_bindgen]
pub fn wasm_sentinel_count() -> usize {
    crate::sentinel_count()
}
