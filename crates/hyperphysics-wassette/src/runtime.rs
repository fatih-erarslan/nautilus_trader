//! WASM runtime implementation using wasmi/wasmtime/wassette

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};
use wasmi::{Engine, Linker, Module, Store};

use crate::module::WasmModule;
use crate::host_functions::HostFunctions;
use crate::WasmBackend;

/// WASM runtime for executing WebAssembly modules
pub struct WasmRuntime {
    /// WASM backend
    backend: WasmBackend,

    /// Wasmi engine (if using Wasmi)
    wasmi_engine: Option<Arc<Engine>>,

    /// Host functions
    host_functions: Arc<HostFunctions>,
}

impl WasmRuntime {
    /// Create new WASM runtime with default backend (Wasmi)
    pub fn new() -> Result<Self> {
        Self::with_backend(WasmBackend::default())
    }

    /// Create runtime with specific backend
    pub fn with_backend(backend: WasmBackend) -> Result<Self> {
        if !crate::backend_available(backend) {
            anyhow::bail!("WASM backend {:?} not available", backend);
        }

        let wasmi_engine = match backend {
            WasmBackend::Wasmi => {
                let config = wasmi::Config::default();
                Some(Arc::new(Engine::new(&config)))
            }
            _ => None,
        };

        let host_functions = Arc::new(HostFunctions::new());

        Ok(Self {
            backend,
            wasmi_engine,
            host_functions,
        })
    }

    /// Load WASM module from file
    pub async fn load_module(&self, path: impl AsRef<Path>) -> Result<WasmModule> {
        let path = path.as_ref();
        info!("Loading WASM module: {}", path.display());

        let wasm_bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("Failed to read WASM file: {}", path.display()))?;

        self.load_module_from_bytes(&wasm_bytes).await
    }

    /// Load WASM module from bytes
    pub async fn load_module_from_bytes(&self, wasm_bytes: &[u8]) -> Result<WasmModule> {
        debug!("Loading WASM module from {} bytes", wasm_bytes.len());

        match self.backend {
            WasmBackend::Wasmi => self.load_wasmi_module(wasm_bytes).await,
            WasmBackend::Wasmtime => {
                anyhow::bail!("Wasmtime backend not yet implemented")
            }
            WasmBackend::Wassette => {
                anyhow::bail!("Wassette backend not yet implemented")
            }
        }
    }

    /// Load module using Wasmi backend
    async fn load_wasmi_module(&self, wasm_bytes: &[u8]) -> Result<WasmModule> {
        let engine = self.wasmi_engine.as_ref()
            .context("Wasmi engine not initialized")?;

        // Parse and validate module
        let module = Module::new(engine, wasm_bytes)
            .context("Failed to parse WASM module")?;

        // Create store and linker
        let mut store = Store::new(engine, ());
        let mut linker = Linker::new(engine);

        // Register host functions
        self.host_functions.register_with_linker(&mut linker, &mut store)?;

        Ok(WasmModule::new_wasmi(module, store, linker))
    }

    /// Get backend type
    pub fn backend(&self) -> WasmBackend {
        self.backend
    }

    /// Get host functions
    pub fn host_functions(&self) -> &HostFunctions {
        &self.host_functions
    }
}

impl Default for WasmRuntime {
    fn default() -> Self {
        Self::new().expect("Failed to create WasmRuntime")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = WasmRuntime::new();
        assert!(runtime.is_ok());
    }

    #[test]
    fn test_runtime_backend() {
        let runtime = WasmRuntime::new().unwrap();
        assert_eq!(runtime.backend(), WasmBackend::Wasmi);
    }

    #[tokio::test]
    async fn test_load_invalid_module() {
        let runtime = WasmRuntime::new().unwrap();
        let result = runtime.load_module_from_bytes(&[0, 1, 2, 3]).await;
        assert!(result.is_err());
    }
}
