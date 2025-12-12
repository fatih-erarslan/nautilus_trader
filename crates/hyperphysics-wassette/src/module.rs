//! WASM module wrapper for execution

use anyhow::{Context, Result};
use wasmi::{Linker, Module, Store};

/// WASM module ready for execution
pub struct WasmModule {
    /// Wasmi module (if using Wasmi backend)
    wasmi_module: Option<Module>,

    /// Wasmi store
    wasmi_store: Option<Store<()>>,

    /// Wasmi linker
    wasmi_linker: Option<Linker<()>>,
}

impl WasmModule {
    /// Create new Wasmi-backed module
    pub(crate) fn new_wasmi(
        module: Module,
        store: Store<()>,
        linker: Linker<()>,
    ) -> Self {
        Self {
            wasmi_module: Some(module),
            wasmi_store: Some(store),
            wasmi_linker: Some(linker),
        }
    }

    /// Call exported function with f64 arguments
    pub async fn call(&mut self, func_name: &str, args: &[f64]) -> Result<Vec<f64>> {
        if let (Some(module), Some(store), Some(linker)) = (
            &self.wasmi_module,
            &mut self.wasmi_store,
            &self.wasmi_linker,
        ) {
            // Instantiate module
            let instance = linker
                .instantiate(store, module)
                .context("Failed to instantiate WASM module")?
                .start(store)
                .context("Failed to start WASM instance")?;

            // Get exported function
            let func = instance
                .get_export(store, func_name)
                .context("Function not found in module")?
                .into_func()
                .context("Export is not a function")?;

            // Convert arguments
            let wasm_args: Vec<wasmi::Value> = args
                .iter()
                .map(|&x| wasmi::Value::F64(x.into()))
                .collect();

            // Call function
            let mut results = vec![wasmi::Value::F64(0.0.into())];
            func.call(store, &wasm_args, &mut results)
                .context("Failed to execute WASM function")?;

            // Convert results
            let output: Vec<f64> = results
                .iter()
                .filter_map(|v| match v {
                    wasmi::Value::F64(f) => Some(f64::from(*f)),
                    _ => None,
                })
                .collect();

            Ok(output)
        } else {
            anyhow::bail!("Module not initialized")
        }
    }

    /// Get list of exported functions
    pub fn exports(&self) -> Vec<String> {
        if let Some(module) = &self.wasmi_module {
            module
                .exports()
                .filter_map(|export| {
                    if matches!(export.ty(), wasmi::ExternType::Func(_)) {
                        Some(export.name().to_string())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Module structure tests will require valid WASM binary
        // Placeholder test
        assert!(true);
    }
}
