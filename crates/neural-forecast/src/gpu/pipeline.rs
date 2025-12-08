//! GPU compute pipeline management and caching

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{ComputePipeline, Device, ShaderModule, BindGroupLayout, PipelineLayout};
use crate::{Result, NeuralForecastError};
use super::GPUOperation;

/// Cache for compute pipelines to avoid recompilation
#[derive(Debug, Default)]
pub struct PipelineCache {
    pipelines: HashMap<GPUOperation, Arc<ComputePipeline>>,
    shader_modules: HashMap<String, Arc<ShaderModule>>,
    bind_group_layouts: HashMap<String, Arc<BindGroupLayout>>,
    pipeline_layouts: HashMap<String, Arc<PipelineLayout>>,
}

/// Pipeline builder for complex operations
#[derive(Debug)]
pub struct PipelineBuilder {
    device: Arc<Device>,
    operation: GPUOperation,
    shader_source: Option<String>,
    bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    push_constant_ranges: Vec<wgpu::PushConstantRange>,
    label: Option<String>,
}

/// Pipeline configuration for different operations
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub workgroup_size: [u32; 3],
    pub shared_memory_size: u32,
    pub max_threads: u32,
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for GPU kernels
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Aggressive,
}

impl PipelineCache {
    /// Create new pipeline cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Get pipeline from cache
    pub fn get(&self, operation: &GPUOperation) -> Option<Arc<ComputePipeline>> {
        self.pipelines.get(operation).cloned()
    }

    /// Insert pipeline into cache
    pub fn insert(&mut self, operation: GPUOperation, pipeline: ComputePipeline) {
        self.pipelines.insert(operation, Arc::new(pipeline));
    }

    /// Get shader module from cache
    pub fn get_shader_module(&self, name: &str) -> Option<Arc<ShaderModule>> {
        self.shader_modules.get(name).cloned()
    }

    /// Insert shader module into cache
    pub fn insert_shader_module(&mut self, name: String, module: ShaderModule) {
        self.shader_modules.insert(name, Arc::new(module));
    }

    /// Get bind group layout from cache
    pub fn get_bind_group_layout(&self, name: &str) -> Option<Arc<BindGroupLayout>> {
        self.bind_group_layouts.get(name).cloned()
    }

    /// Insert bind group layout into cache
    pub fn insert_bind_group_layout(&mut self, name: String, layout: BindGroupLayout) {
        self.bind_group_layouts.insert(name, Arc::new(layout));
    }

    /// Get pipeline layout from cache
    pub fn get_pipeline_layout(&self, name: &str) -> Option<Arc<PipelineLayout>> {
        self.pipeline_layouts.get(name).cloned()
    }

    /// Insert pipeline layout into cache
    pub fn insert_pipeline_layout(&mut self, name: String, layout: PipelineLayout) {
        self.pipeline_layouts.insert(name, Arc::new(layout));
    }

    /// Clear all cached items
    pub fn clear(&mut self) {
        self.pipelines.clear();
        self.shader_modules.clear();
        self.bind_group_layouts.clear();
        self.pipeline_layouts.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            pipelines: self.pipelines.len(),
            shader_modules: self.shader_modules.len(),
            bind_group_layouts: self.bind_group_layouts.len(),
            pipeline_layouts: self.pipeline_layouts.len(),
        }
    }

    /// Pre-compile common pipelines
    pub fn pre_compile_common_pipelines(&mut self, device: &Device) -> Result<()> {
        let common_operations = vec![
            GPUOperation::MatMul,
            GPUOperation::Add,
            GPUOperation::Mul,
            GPUOperation::Relu,
            GPUOperation::Sigmoid,
            GPUOperation::Tanh,
            GPUOperation::Softmax,
            GPUOperation::LayerNorm,
            GPUOperation::BatchNorm,
        ];

        for operation in common_operations {
            let builder = PipelineBuilder::new(Arc::new(device.clone()), operation.clone());
            let pipeline = builder.build()?;
            self.insert(operation, pipeline);
        }

        Ok(())
    }
}

impl PipelineBuilder {
    /// Create new pipeline builder
    pub fn new(device: Arc<Device>, operation: GPUOperation) -> Self {
        Self {
            device,
            operation,
            shader_source: None,
            bind_group_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
            label: None,
        }
    }

    /// Set shader source
    pub fn with_shader_source(mut self, source: String) -> Self {
        self.shader_source = Some(source);
        self
    }

    /// Add bind group layout
    pub fn add_bind_group_layout(mut self, layout: Arc<BindGroupLayout>) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    /// Add push constant range
    pub fn add_push_constant_range(mut self, range: wgpu::PushConstantRange) -> Self {
        self.push_constant_ranges.push(range);
        self
    }

    /// Set label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Build the compute pipeline
    pub fn build(self) -> Result<ComputePipeline> {
        let shader_source = self.shader_source.ok_or_else(|| {
            NeuralForecastError::GpuError("No shader source provided".to_string())
        })?;

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", self.operation.name())),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout_refs: Vec<&BindGroupLayout> = self.bind_group_layouts
            .iter()
            .map(|layout| layout.as_ref())
            .collect();

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", self.operation.name())),
            bind_group_layouts: &bind_group_layout_refs,
            push_constant_ranges: &self.push_constant_ranges,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self.label.as_deref().or(Some(&format!("{} Pipeline", self.operation.name()))),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(pipeline)
    }
}

impl PipelineConfig {
    /// Create configuration for matrix multiplication
    pub fn matmul(m: u32, k: u32, n: u32) -> Self {
        let workgroup_size = Self::optimal_workgroup_size_for_matmul(m, k, n);
        
        Self {
            workgroup_size,
            shared_memory_size: 16384, // 16KB shared memory
            max_threads: 1024,
            optimization_level: OptimizationLevel::Release,
        }
    }

    /// Create configuration for element-wise operations
    pub fn elementwise(size: u32) -> Self {
        let workgroup_size = Self::optimal_workgroup_size_for_elementwise(size);
        
        Self {
            workgroup_size,
            shared_memory_size: 0,
            max_threads: 1024,
            optimization_level: OptimizationLevel::Release,
        }
    }

    /// Create configuration for reduction operations
    pub fn reduction(size: u32) -> Self {
        let workgroup_size = Self::optimal_workgroup_size_for_reduction(size);
        
        Self {
            workgroup_size,
            shared_memory_size: 4096, // 4KB shared memory for reduction
            max_threads: 512,
            optimization_level: OptimizationLevel::Release,
        }
    }

    /// Create configuration for attention operations
    pub fn attention(seq_len: u32, d_model: u32, n_heads: u32) -> Self {
        let workgroup_size = Self::optimal_workgroup_size_for_attention(seq_len, d_model, n_heads);
        
        Self {
            workgroup_size,
            shared_memory_size: 32768, // 32KB shared memory for attention
            max_threads: 1024,
            optimization_level: OptimizationLevel::Aggressive,
        }
    }

    /// Calculate optimal workgroup size for matrix multiplication
    fn optimal_workgroup_size_for_matmul(m: u32, k: u32, n: u32) -> [u32; 3] {
        // Optimize for common matrix sizes in neural networks
        let tile_size = if m <= 64 && k <= 64 && n <= 64 {
            8
        } else if m <= 256 && k <= 256 && n <= 256 {
            16
        } else {
            32
        };
        
        [tile_size, tile_size, 1]
    }

    /// Calculate optimal workgroup size for element-wise operations
    fn optimal_workgroup_size_for_elementwise(size: u32) -> [u32; 3] {
        let workgroup_size = if size <= 1024 {
            64
        } else if size <= 65536 {
            128
        } else {
            256
        };
        
        [workgroup_size, 1, 1]
    }

    /// Calculate optimal workgroup size for reduction operations
    fn optimal_workgroup_size_for_reduction(size: u32) -> [u32; 3] {
        let workgroup_size = if size <= 1024 {
            64
        } else if size <= 16384 {
            128
        } else {
            256
        };
        
        [workgroup_size, 1, 1]
    }

    /// Calculate optimal workgroup size for attention operations
    fn optimal_workgroup_size_for_attention(seq_len: u32, d_model: u32, n_heads: u32) -> [u32; 3] {
        let head_dim = d_model / n_heads;
        
        let workgroup_size = if seq_len <= 64 && head_dim <= 64 {
            [8, 8, 1]
        } else if seq_len <= 256 && head_dim <= 128 {
            [16, 8, 1]
        } else {
            [32, 8, 1]
        };
        
        workgroup_size
    }

    /// Get dispatch size for the given problem size
    pub fn dispatch_size(&self, problem_size: [u32; 3]) -> [u32; 3] {
        [
            (problem_size[0] + self.workgroup_size[0] - 1) / self.workgroup_size[0],
            (problem_size[1] + self.workgroup_size[1] - 1) / self.workgroup_size[1],
            (problem_size[2] + self.workgroup_size[2] - 1) / self.workgroup_size[2],
        ]
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub pipelines: usize,
    pub shader_modules: usize,
    pub bind_group_layouts: usize,
    pub pipeline_layouts: usize,
}

/// Pipeline performance metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub operation: GPUOperation,
    pub compile_time_ms: f64,
    pub execution_time_us: f64,
    pub memory_usage_bytes: u64,
    pub workgroup_efficiency: f32,
}

/// Pipeline optimizer for performance tuning
pub struct PipelineOptimizer {
    device: Arc<Device>,
    metrics: HashMap<GPUOperation, PipelineMetrics>,
}

impl PipelineOptimizer {
    /// Create new pipeline optimizer
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            metrics: HashMap::new(),
        }
    }

    /// Optimize pipeline for specific operation
    pub fn optimize_pipeline(&mut self, operation: GPUOperation, problem_size: [u32; 3]) -> Result<PipelineConfig> {
        match operation {
            GPUOperation::MatMul => {
                Ok(PipelineConfig::matmul(problem_size[0], problem_size[1], problem_size[2]))
            }
            GPUOperation::Add | GPUOperation::Mul | GPUOperation::Relu | 
            GPUOperation::Sigmoid | GPUOperation::Tanh => {
                Ok(PipelineConfig::elementwise(problem_size[0] * problem_size[1] * problem_size[2]))
            }
            GPUOperation::Softmax | GPUOperation::LayerNorm | GPUOperation::BatchNorm => {
                Ok(PipelineConfig::reduction(problem_size[0] * problem_size[1] * problem_size[2]))
            }
            GPUOperation::Attention => {
                Ok(PipelineConfig::attention(problem_size[0], problem_size[1], problem_size[2]))
            }
            _ => {
                // Default configuration for unknown operations
                Ok(PipelineConfig::elementwise(problem_size[0] * problem_size[1] * problem_size[2]))
            }
        }
    }

    /// Record pipeline performance metrics
    pub fn record_metrics(&mut self, metrics: PipelineMetrics) {
        self.metrics.insert(metrics.operation.clone(), metrics);
    }

    /// Get performance metrics for operation
    pub fn get_metrics(&self, operation: &GPUOperation) -> Option<&PipelineMetrics> {
        self.metrics.get(operation)
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self, operation: &GPUOperation) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(metrics) = self.get_metrics(operation) {
            if metrics.workgroup_efficiency < 0.8 {
                recommendations.push("Consider adjusting workgroup size for better occupancy".to_string());
            }
            
            if metrics.execution_time_us > 1000.0 {
                recommendations.push("Consider using mixed precision or quantization".to_string());
            }
            
            if metrics.memory_usage_bytes > 100_000_000 {
                recommendations.push("Consider memory optimization techniques".to_string());
            }
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_cache_creation() {
        let cache = PipelineCache::new();
        let stats = cache.stats();
        assert_eq!(stats.pipelines, 0);
        assert_eq!(stats.shader_modules, 0);
    }

    #[test]
    fn test_pipeline_config_matmul() {
        let config = PipelineConfig::matmul(128, 128, 128);
        assert_eq!(config.workgroup_size, [16, 16, 1]);
        assert_eq!(config.shared_memory_size, 16384);
    }

    #[test]
    fn test_pipeline_config_elementwise() {
        let config = PipelineConfig::elementwise(1024);
        assert_eq!(config.workgroup_size, [64, 1, 1]);
        assert_eq!(config.shared_memory_size, 0);
    }

    #[test]
    fn test_dispatch_size_calculation() {
        let config = PipelineConfig::matmul(128, 128, 128);
        let dispatch = config.dispatch_size([128, 128, 1]);
        assert_eq!(dispatch, [8, 8, 1]);
    }

    #[test]
    fn test_optimization_level() {
        let config = PipelineConfig::matmul(128, 128, 128);
        matches!(config.optimization_level, OptimizationLevel::Release);
    }
}