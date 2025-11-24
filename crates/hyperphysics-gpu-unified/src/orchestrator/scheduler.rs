//! Pipeline scheduling and caching

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use wgpu::{ComputePipeline, Device, ShaderModule};

/// Pipeline scheduler with caching
pub struct PipelineScheduler {
    /// Cached pipelines by shader hash
    pipelines: RwLock<HashMap<u64, Arc<ComputePipeline>>>,
    /// Cached shader modules
    shaders: RwLock<HashMap<u64, Arc<ShaderModule>>>,
    /// Enable caching
    caching_enabled: bool,
}

impl PipelineScheduler {
    /// Create new pipeline scheduler
    pub fn new(caching_enabled: bool) -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
            shaders: RwLock::new(HashMap::new()),
            caching_enabled,
        }
    }

    /// Get or create a compute pipeline
    pub fn get_or_create_pipeline(
        &self,
        device: &Device,
        shader_source: &str,
        entry_point: &str,
        label: Option<&str>,
    ) -> Arc<ComputePipeline> {
        let hash = Self::hash_shader(shader_source, entry_point);

        // Check cache first
        if self.caching_enabled {
            if let Some(pipeline) = self.pipelines.read().get(&hash) {
                return pipeline.clone();
            }
        }

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout: None, // Auto-layout
            module: &shader,
            entry_point,
            compilation_options: Default::default(),
        });

        let pipeline = Arc::new(pipeline);

        // Cache if enabled
        if self.caching_enabled {
            self.pipelines.write().insert(hash, pipeline.clone());
        }

        pipeline
    }

    /// Hash shader source and entry point for caching
    fn hash_shader(source: &str, entry_point: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        entry_point.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear all cached pipelines
    pub fn clear_cache(&self) {
        self.pipelines.write().clear();
        self.shaders.write().clear();
    }

    /// Get number of cached pipelines
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_hash() {
        let hash1 = PipelineScheduler::hash_shader("fn main() {}", "main");
        let hash2 = PipelineScheduler::hash_shader("fn main() {}", "main");
        let hash3 = PipelineScheduler::hash_shader("fn main() { return; }", "main");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
