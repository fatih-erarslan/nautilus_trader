//! Host functions exposed to WASM modules

use anyhow::Result;
use wasmi::{Caller, Func, Linker, Store};

/// Host functions for WASM modules
pub struct HostFunctions {
    // Future: HyperPhysics-specific host functions
}

impl HostFunctions {
    /// Create new host functions
    pub fn new() -> Self {
        Self {}
    }

    /// Register host functions with Wasmi linker
    pub fn register_with_linker(
        &self,
        linker: &mut Linker<()>,
        store: &mut Store<()>,
    ) -> Result<()> {
        // Register hyperbolic distance function
        linker.define(
            "hyperphysics",
            "hyperbolic_distance",
            Func::wrap(store, |_caller: Caller<()>, x1: f64, y1: f64, x2: f64, y2: f64| -> f64 {
                // Simple Poincaré disk distance (placeholder)
                let dx = x2 - x1;
                let dy = y2 - y1;
                (dx * dx + dy * dy).sqrt()
            }),
        )?;

        // Register logging function
        linker.define(
            "hyperphysics",
            "log",
            Func::wrap(store, |_caller: Caller<()>, value: f64| {
                tracing::info!("WASM log: {}", value);
            }),
        )?;

        Ok(())
    }
}

impl Default for HostFunctions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_functions_creation() {
        let host_funcs = HostFunctions::new();
        // Basic creation test
        assert!(true);
    }
}
