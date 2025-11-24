//! Kernel registry and management for GPU compute shaders

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::ComputePipeline;

/// Kernel category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelCategory {
    /// Physics simulation kernels
    Physics,
    /// Financial computation kernels
    Finance,
    /// Neural network kernels
    Neural,
    /// Utility kernels (reduction, scan, etc.)
    Utility,
}

/// Kernel metadata
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel name
    pub name: String,
    /// Category
    pub category: KernelCategory,
    /// Entry point in shader
    pub entry_point: String,
    /// Workgroup size (x, y, z)
    pub workgroup_size: [u32; 3],
    /// Description
    pub description: String,
    /// WGSL source code
    pub source: String,
}

/// Registered kernel with compiled pipeline
pub struct RegisteredKernel {
    /// Kernel metadata
    pub info: KernelInfo,
    /// Compiled pipeline (lazy)
    pub pipeline: Option<Arc<ComputePipeline>>,
}

/// Registry for managing compute kernels
pub struct KernelRegistry {
    /// Registered kernels by name
    kernels: HashMap<String, RegisteredKernel>,
    /// Built-in kernel sources
    builtins_loaded: bool,
}

impl KernelRegistry {
    /// Create new kernel registry
    pub fn new() -> Self {
        let mut registry = Self {
            kernels: HashMap::new(),
            builtins_loaded: false,
        };
        registry.load_builtins();
        registry
    }

    /// Load built-in kernels
    fn load_builtins(&mut self) {
        if self.builtins_loaded {
            return;
        }

        // PCG Random number generator (corrected)
        self.register(KernelInfo {
            name: "pcg_rand".to_string(),
            category: KernelCategory::Utility,
            entry_point: "main".to_string(),
            workgroup_size: [256, 1, 1],
            description: "PCG-based random number generation with proper [0,1) range".to_string(),
            source: include_str!("shaders/pcg_rand.wgsl").to_string(),
        });

        // Parallel reduction
        self.register(KernelInfo {
            name: "parallel_reduce_sum".to_string(),
            category: KernelCategory::Utility,
            entry_point: "main".to_string(),
            workgroup_size: [256, 1, 1],
            description: "Parallel sum reduction with workgroup optimization".to_string(),
            source: include_str!("shaders/reduce_sum.wgsl").to_string(),
        });

        // pBit sampling kernel
        self.register(KernelInfo {
            name: "pbit_sample".to_string(),
            category: KernelCategory::Physics,
            entry_point: "main".to_string(),
            workgroup_size: [256, 1, 1],
            description: "Probabilistic bit sampling with Boltzmann distribution".to_string(),
            source: include_str!("shaders/pbit_sample.wgsl").to_string(),
        });

        // Monte Carlo path simulation
        self.register(KernelInfo {
            name: "monte_carlo_gbm".to_string(),
            category: KernelCategory::Finance,
            entry_point: "main".to_string(),
            workgroup_size: [128, 1, 1],
            description: "Geometric Brownian Motion Monte Carlo simulation".to_string(),
            source: include_str!("shaders/monte_carlo_gbm.wgsl").to_string(),
        });

        // Matrix multiplication (tiled)
        self.register(KernelInfo {
            name: "matmul_tiled".to_string(),
            category: KernelCategory::Neural,
            entry_point: "main".to_string(),
            workgroup_size: [16, 16, 1],
            description: "Tiled matrix multiplication optimized for RDNA2".to_string(),
            source: include_str!("shaders/matmul_tiled.wgsl").to_string(),
        });

        self.builtins_loaded = true;
    }

    /// Register a kernel
    pub fn register(&mut self, info: KernelInfo) {
        self.kernels.insert(
            info.name.clone(),
            RegisteredKernel {
                info,
                pipeline: None,
            },
        );
    }

    /// Get kernel by name
    pub fn get(&self, name: &str) -> Option<&RegisteredKernel> {
        self.kernels.get(name)
    }

    /// Get mutable kernel by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut RegisteredKernel> {
        self.kernels.get_mut(name)
    }

    /// List all kernels in a category
    pub fn list_category(&self, category: KernelCategory) -> Vec<&KernelInfo> {
        self.kernels
            .values()
            .filter(|k| k.info.category == category)
            .map(|k| &k.info)
            .collect()
    }

    /// List all kernel names
    pub fn list_names(&self) -> Vec<&str> {
        self.kernels.keys().map(|s| s.as_str()).collect()
    }

    /// Get total kernel count
    pub fn count(&self) -> usize {
        self.kernels.len()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builtins() {
        let registry = KernelRegistry::new();
        assert!(registry.count() >= 5);
        assert!(registry.get("pcg_rand").is_some());
        assert!(registry.get("monte_carlo_gbm").is_some());
    }

    #[test]
    fn test_list_category() {
        let registry = KernelRegistry::new();
        let finance_kernels = registry.list_category(KernelCategory::Finance);
        assert!(!finance_kernels.is_empty());
    }
}
