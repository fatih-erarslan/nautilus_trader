//! # HyperPhysics-Wassette Integration
//!
//! Integration bridge between HyperPhysics physics/neural framework
//! and Wassette/WASM runtime for sandboxed execution.
//!
//! ## Features
//!
//! - **WASM Execution**: Run WebAssembly modules in sandboxed environment
//! - **Neural WASM**: Execute neural networks compiled to WASM
//! - **Hyperbolic WASM**: WASM modules with hyperbolic geometry support
//! - **JIT Optimization**: JIT compilation optimized with HyperPhysics
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                  HyperPhysics Core                          │
//! │   (Neural Networks, Optimization, Hyperbolic Geometry)      │
//! └────────────────────┬───────────────────────────────────────┘
//!                      │
//!                      ▼
//! ┌────────────────────────────────────────────────────────────┐
//! │            Wassette/WASM Integration Layer                  │
//! │  - Module loading and validation                            │
//! │  - Host function bindings (physics/neural)                  │
//! │  - JIT optimization                                         │
//! │  - Hyperbolic geometry exports                              │
//! └────────────────────┬───────────────────────────────────────┘
//!                      │
//!                      ▼
//! ┌────────────────────────────────────────────────────────────┐
//! │              Wassette WASM Runtime                          │
//! │  (WebAssembly Execution Engine)                             │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use hyperphysics_wassette::{WasmRuntime, WasmModule};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize WASM runtime
//!     let runtime = WasmRuntime::new()?;
//!
//!     // Load WASM module
//!     let module = runtime.load_module("neural_network.wasm").await?;
//!
//!     // Execute function
//!     let result = module.call("forward", &[1.0, 2.0, 3.0]).await?;
//!
//!     println!("Result: {:?}", result);
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

pub mod runtime;
pub mod module;
pub mod host_functions;
pub mod optimization;
pub mod neural;

pub use runtime::WasmRuntime;
pub use module::WasmModule;
pub use host_functions::HostFunctions;
pub use optimization::WasmOptimizer;

/// WASM execution backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmBackend {
    /// Wasmi interpreter (default, always available)
    Wasmi,
    /// Wasmtime JIT compiler (optional, faster)
    Wasmtime,
    /// Wassette runtime (optional, experimental)
    Wassette,
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::Wasmi
    }
}

/// Check if WASM backend is available
pub fn backend_available(backend: WasmBackend) -> bool {
    match backend {
        WasmBackend::Wasmi => true, // Always available
        WasmBackend::Wasmtime => cfg!(feature = "wasmtime-runtime"),
        WasmBackend::Wassette => cfg!(feature = "wassette-runtime"),
    }
}

/// Get available WASM backends
pub fn available_backends() -> Vec<WasmBackend> {
    let mut backends = vec![WasmBackend::Wasmi];

    if backend_available(WasmBackend::Wasmtime) {
        backends.push(WasmBackend::Wasmtime);
    }

    if backend_available(WasmBackend::Wassette) {
        backends.push(WasmBackend::Wassette);
    }

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backends() {
        let backends = available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&WasmBackend::Wasmi));
    }

    #[test]
    fn test_wasmi_always_available() {
        assert!(backend_available(WasmBackend::Wasmi));
    }
}
