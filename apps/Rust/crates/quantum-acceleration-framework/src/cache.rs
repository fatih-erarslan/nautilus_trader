//! Kernel cache for pre-compiled GPU shaders and patterns

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use crate::{
    QBMIAError, QBMIAResult, Pattern, gpu::GpuPipeline,
    GpuBufferUsage, KernelParams
};

/// High-performance kernel cache for GPU operations
pub struct KernelCache {
    /// Compiled compute pipelines cache
    compute_pipelines: Arc<RwLock<HashMap<String, CachedPipeline>>>,
    
    /// Pattern matching kernels cache
    pattern_kernels: Arc<RwLock<HashMap<PatternMatchingKey, CachedPatternKernel>>>,
    
    /// Shader source cache
    shader_cache: Arc<RwLock<HashMap<String, String>>>,
    
    /// Cache statistics
    stats: Arc<tokio::sync::Mutex<CacheStats>>,
}

impl KernelCache {
    /// Create new kernel cache
    pub async fn new() -> QBMIAResult<Self> {
        tracing::info!("Initializing kernel cache");
        
        let compute_pipelines = Arc::new(RwLock::new(HashMap::new()));
        let pattern_kernels = Arc::new(RwLock::new(HashMap::new()));
        let shader_cache = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(tokio::sync::Mutex::new(CacheStats::new()));
        
        let cache = Self {
            compute_pipelines,
            pattern_kernels,
            shader_cache,
            stats,
        };
        
        // Pre-populate cache with common kernels
        cache.precompile_common_kernels().await?;
        
        tracing::info!("Kernel cache initialized");
        Ok(cache)
    }
    
    /// Execute pattern matching with cached kernels
    pub async fn execute_pattern_matching(
        &self,
        patterns: &[Pattern],
        query: &Pattern,
        threshold: f32,
    ) -> QBMIAResult<Vec<bool>> {
        let start_time = std::time::Instant::now();
        
        // Create cache key based on pattern dimensions and count
        let cache_key = PatternMatchingKey {
            pattern_count: patterns.len(),
            pattern_dimension: query.dimension(),
            threshold_bucket: self.threshold_to_bucket(threshold),
        };
        
        // Try to get cached kernel
        let cached_kernel = {
            let pattern_kernels = self.pattern_kernels.read().await;
            pattern_kernels.get(&cache_key).cloned()
        };
        
        let kernel = if let Some(cached) = cached_kernel {
            // Use cached kernel
            let mut stats = self.stats.lock().await;
            stats.record_cache_hit();
            cached
        } else {
            // Create new kernel and cache it
            let new_kernel = self.create_pattern_matching_kernel(&cache_key).await?;
            
            let mut pattern_kernels = self.pattern_kernels.write().await;
            pattern_kernels.insert(cache_key.clone(), new_kernel.clone());
            
            let mut stats = self.stats.lock().await;
            stats.record_cache_miss();
            
            new_kernel
        };
        
        // Execute kernel
        let matches = self.execute_pattern_kernel(&kernel, patterns, query, threshold).await?;
        
        let execution_time = start_time.elapsed();
        
        // Update performance stats
        let mut stats = self.stats.lock().await;
        stats.record_pattern_matching(execution_time, patterns.len());
        
        tracing::debug!(
            "Pattern matching ({} patterns) executed in {:.3}ns",
            patterns.len(),
            execution_time.as_nanos()
        );
        
        Ok(matches)
    }
    
    /// Get or create compute pipeline with caching
    pub async fn get_compute_pipeline(
        &self,
        gpu_pipeline: &GpuPipeline,
        shader_source: &str,
        entry_point: &str,
    ) -> QBMIAResult<CachedPipeline> {
        let cache_key = format!("{}:{}", shader_source, entry_point);
        
        // Try to get from cache
        {
            let pipelines = self.compute_pipelines.read().await;
            if let Some(cached) = pipelines.get(&cache_key) {
                let mut stats = self.stats.lock().await;
                stats.record_cache_hit();
                return Ok(cached.clone());
            }
        }
        
        // Create new pipeline
        let start_time = std::time::Instant::now();
        let pipeline = gpu_pipeline.get_compute_pipeline(shader_source, entry_point).await?;
        let compilation_time = start_time.elapsed();
        
        let cached_pipeline = CachedPipeline {
            pipeline,
            shader_hash: self.hash_shader(shader_source),
            entry_point: entry_point.to_string(),
            compilation_time,
            created_at: std::time::Instant::now(),
            usage_count: 0,
            last_used: std::time::Instant::now(),
        };
        
        // Cache the pipeline
        {
            let mut pipelines = self.compute_pipelines.write().await;
            pipelines.insert(cache_key, cached_pipeline.clone());
        }
        
        let mut stats = self.stats.lock().await;
        stats.record_cache_miss();
        stats.record_compilation(compilation_time);
        
        Ok(cached_pipeline)
    }
    
    /// Pre-compile common kernels for performance
    async fn precompile_common_kernels(&self) -> QBMIAResult<()> {
        tracing::info!("Pre-compiling common kernels");
        
        // Pre-compile pattern matching kernels for common sizes
        let common_sizes = [
            (100, 64),   // 100 patterns, 64-dim
            (1000, 64),  // 1000 patterns, 64-dim
            (100, 128),  // 100 patterns, 128-dim
            (1000, 128), // 1000 patterns, 128-dim
            (10000, 64), // 10000 patterns, 64-dim
        ];
        
        for &(pattern_count, pattern_dim) in &common_sizes {
            for &threshold in &[0.7, 0.8, 0.9] {
                let cache_key = PatternMatchingKey {
                    pattern_count,
                    pattern_dimension: pattern_dim,
                    threshold_bucket: self.threshold_to_bucket(threshold),
                };
                
                let kernel = self.create_pattern_matching_kernel(&cache_key).await?;
                
                let mut pattern_kernels = self.pattern_kernels.write().await;
                pattern_kernels.insert(cache_key, kernel);
            }
        }
        
        // Pre-compile shader sources for common operations
        let common_shaders = self.get_common_shader_sources();
        for (name, source) in common_shaders {
            let mut shader_cache = self.shader_cache.write().await;
            shader_cache.insert(name, source);
        }
        
        tracing::info!("Common kernel pre-compilation completed");
        Ok(())
    }
    
    /// Create pattern matching kernel for given parameters
    async fn create_pattern_matching_kernel(&self, key: &PatternMatchingKey) -> QBMIAResult<CachedPatternKernel> {
        let shader_source = self.generate_pattern_matching_shader(key);
        let workgroup_size = self.calculate_optimal_workgroup_size(key.pattern_count);
        
        let kernel = CachedPatternKernel {
            shader_source,
            workgroup_size,
            pattern_count: key.pattern_count,
            pattern_dimension: key.pattern_dimension,
            threshold_bucket: key.threshold_bucket,
            created_at: std::time::Instant::now(),
            usage_count: 0,
            last_used: std::time::Instant::now(),
        };
        
        Ok(kernel)
    }
    
    /// Execute pattern matching kernel
    async fn execute_pattern_kernel(
        &self,
        kernel: &CachedPatternKernel,
        patterns: &[Pattern],
        query: &Pattern,
        threshold: f32,
    ) -> QBMIAResult<Vec<bool>> {
        // This would need a GPU pipeline reference to actually execute
        // For now, simulate the execution with CPU fallback
        let mut matches = Vec::with_capacity(patterns.len());
        
        for pattern in patterns {
            let similarity = pattern.cosine_similarity(query);
            matches.push(similarity > threshold);
        }
        
        Ok(matches)
    }
    
    /// Generate optimized pattern matching shader
    fn generate_pattern_matching_shader(&self, key: &PatternMatchingKey) -> String {
        let workgroup_size = self.calculate_optimal_workgroup_size(key.pattern_count);
        
        format!(r#"
        @group(0) @binding(0) var<storage, read> patterns: array<f32>;
        @group(0) @binding(1) var<storage, read> query: array<f32>;
        @group(0) @binding(2) var<uniform> params: PatternParams;
        @group(0) @binding(3) var<storage, read_write> results: array<u32>;
        
        struct PatternParams {{
            pattern_count: u32,
            pattern_dimension: u32,
            threshold: f32,
            _padding: f32,
        }}
        
        var<workgroup> shared_patterns: array<f32, {}>;
        var<workgroup> shared_query: array<f32, {}>;
        
        @compute @workgroup_size({})
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>) {{
            
            let pattern_index = global_id.x;
            let local_index = local_id.x;
            let dimension = params.pattern_dimension;
            
            if (pattern_index >= params.pattern_count) {{
                return;
            }}
            
            // Load query into shared memory
            if (local_index < dimension) {{
                shared_query[local_index] = query[local_index];
            }}
            workgroupBarrier();
            
            // Calculate cosine similarity for this pattern
            var dot_product = 0.0;
            var pattern_norm_sq = 0.0;
            var query_norm_sq = 0.0;
            
            let pattern_offset = pattern_index * dimension;
            
            // Use vectorized operations for better performance
            for (var i = 0u; i < dimension; i += 4u) {{
                let end_i = min(i + 4u, dimension);
                
                for (var j = i; j < end_i; j++) {{
                    let pattern_val = patterns[pattern_offset + j];
                    let query_val = shared_query[j];
                    
                    dot_product += pattern_val * query_val;
                    pattern_norm_sq += pattern_val * pattern_val;
                    query_norm_sq += query_val * query_val;
                }}
            }}
            
            // Calculate cosine similarity
            let pattern_norm = sqrt(pattern_norm_sq);
            let query_norm = sqrt(query_norm_sq);
            
            var similarity = 0.0;
            if (pattern_norm > 1e-10 && query_norm > 1e-10) {{
                similarity = dot_product / (pattern_norm * query_norm);
            }}
            
            // Check threshold and store result
            if (similarity > params.threshold) {{
                results[pattern_index] = 1u;
            }} else {{
                results[pattern_index] = 0u;
            }}
        }}
        "#, 
        key.pattern_dimension, // shared_patterns size
        key.pattern_dimension, // shared_query size
        workgroup_size
        )
    }
    
    /// Calculate optimal workgroup size for pattern count
    fn calculate_optimal_workgroup_size(&self, pattern_count: usize) -> u32 {
        // Choose workgroup size based on pattern count for optimal occupancy
        if pattern_count <= 256 {
            64  // Small workgroup for small pattern sets
        } else if pattern_count <= 1024 {
            128 // Medium workgroup
        } else {
            256 // Large workgroup for large pattern sets
        }
    }
    
    /// Convert threshold to bucket for caching
    fn threshold_to_bucket(&self, threshold: f32) -> u32 {
        // Bucket thresholds to reduce cache fragmentation
        (threshold * 100.0).round() as u32
    }
    
    /// Get common shader sources for pre-compilation
    fn get_common_shader_sources(&self) -> Vec<(String, String)> {
        vec![
            (
                "vector_add".to_string(),
                r#"
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&result)) {
                        return;
                    }
                    result[index] = a[index] + b[index];
                }
                "#.to_string(),
            ),
            (
                "vector_multiply".to_string(),
                r#"
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&result)) {
                        return;
                    }
                    result[index] = a[index] * b[index];
                }
                "#.to_string(),
            ),
            (
                "reduce_sum".to_string(),
                r#"
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                
                var<workgroup> shared_data: array<f32, 256>;
                
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
                    let index = global_id.x;
                    let local_index = local_id.x;
                    
                    // Load data into shared memory
                    if (index < arrayLength(&input)) {
                        shared_data[local_index] = input[index];
                    } else {
                        shared_data[local_index] = 0.0;
                    }
                    
                    workgroupBarrier();
                    
                    // Parallel reduction
                    for (var stride = 128u; stride > 0u; stride >>= 1u) {
                        if (local_index < stride) {
                            shared_data[local_index] += shared_data[local_index + stride];
                        }
                        workgroupBarrier();
                    }
                    
                    // Write result
                    if (local_index == 0u) {
                        output[global_id.x / 256u] = shared_data[0];
                    }
                }
                "#.to_string(),
            ),
        ]
    }
    
    /// Hash shader source for caching
    fn hash_shader(&self, shader_source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        shader_source.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Garbage collect old cache entries
    pub async fn garbage_collect(&self) -> QBMIAResult<usize> {
        let start_time = std::time::Instant::now();
        let mut freed_count = 0;
        
        let retention_period = std::time::Duration::from_secs(300); // 5 minutes
        let now = std::time::Instant::now();
        
        // Clean up compute pipelines
        {
            let mut pipelines = self.compute_pipelines.write().await;
            let initial_count = pipelines.len();
            
            pipelines.retain(|_, cached| {
                now.duration_since(cached.last_used) < retention_period || cached.usage_count > 10
            });
            
            freed_count += initial_count - pipelines.len();
        }
        
        // Clean up pattern kernels
        {
            let mut pattern_kernels = self.pattern_kernels.write().await;
            let initial_count = pattern_kernels.len();
            
            pattern_kernels.retain(|_, cached| {
                now.duration_since(cached.last_used) < retention_period || cached.usage_count > 5
            });
            
            freed_count += initial_count - pattern_kernels.len();
        }
        
        let gc_time = start_time.elapsed();
        let mut stats = self.stats.lock().await;
        stats.record_garbage_collection(gc_time, freed_count);
        
        tracing::debug!("Cache garbage collection freed {} entries in {:.3}ms",
                       freed_count, gc_time.as_secs_f64() * 1000.0);
        
        Ok(freed_count)
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }
    
    /// Clear all caches
    pub async fn clear(&self) -> QBMIAResult<()> {
        {
            let mut pipelines = self.compute_pipelines.write().await;
            pipelines.clear();
        }
        
        {
            let mut pattern_kernels = self.pattern_kernels.write().await;
            pattern_kernels.clear();
        }
        
        {
            let mut shader_cache = self.shader_cache.write().await;
            shader_cache.clear();
        }
        
        let mut stats = self.stats.lock().await;
        *stats = CacheStats::new();
        
        tracing::info!("All caches cleared");
        Ok(())
    }
}

/// Cached compute pipeline
#[derive(Debug, Clone)]
pub struct CachedPipeline {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub shader_hash: u64,
    pub entry_point: String,
    pub compilation_time: std::time::Duration,
    pub created_at: std::time::Instant,
    pub usage_count: u64,
    pub last_used: std::time::Instant,
}

/// Cached pattern matching kernel
#[derive(Debug, Clone)]
pub struct CachedPatternKernel {
    pub shader_source: String,
    pub workgroup_size: u32,
    pub pattern_count: usize,
    pub pattern_dimension: usize,
    pub threshold_bucket: u32,
    pub created_at: std::time::Instant,
    pub usage_count: u64,
    pub last_used: std::time::Instant,
}

/// Pattern matching cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternMatchingKey {
    pattern_count: usize,
    pattern_dimension: usize,
    threshold_bucket: u32, // Bucketed threshold for caching efficiency
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub compilations: u64,
    pub garbage_collections: u64,
    pub entries_freed: u64,
    
    pub total_compilation_time: std::time::Duration,
    pub total_gc_time: std::time::Duration,
    pub pattern_matching_executions: u64,
    pub total_pattern_matching_time: std::time::Duration,
}

impl CacheStats {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_cache_hit(&mut self) {
        self.total_requests += 1;
        self.cache_hits += 1;
    }
    
    fn record_cache_miss(&mut self) {
        self.total_requests += 1;
        self.cache_misses += 1;
    }
    
    fn record_compilation(&mut self, duration: std::time::Duration) {
        self.compilations += 1;
        self.total_compilation_time += duration;
    }
    
    fn record_pattern_matching(&mut self, duration: std::time::Duration, pattern_count: usize) {
        self.pattern_matching_executions += 1;
        self.total_pattern_matching_time += duration;
    }
    
    fn record_garbage_collection(&mut self, duration: std::time::Duration, freed_count: usize) {
        self.garbage_collections += 1;
        self.entries_freed += freed_count as u64;
        self.total_gc_time += duration;
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.cache_hits as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
    
    pub fn average_compilation_time(&self) -> Option<std::time::Duration> {
        if self.compilations > 0 {
            Some(self.total_compilation_time / self.compilations as u32)
        } else {
            None
        }
    }
    
    pub fn average_pattern_matching_time(&self) -> Option<std::time::Duration> {
        if self.pattern_matching_executions > 0 {
            Some(self.total_pattern_matching_time / self.pattern_matching_executions as u32)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_kernel_cache_creation() {
        let cache = KernelCache::new().await;
        assert!(cache.is_ok());
    }
    
    #[tokio::test]
    async fn test_pattern_matching_cache_key() {
        let key1 = PatternMatchingKey {
            pattern_count: 100,
            pattern_dimension: 64,
            threshold_bucket: 80,
        };
        
        let key2 = PatternMatchingKey {
            pattern_count: 100,
            pattern_dimension: 64,
            threshold_bucket: 80,
        };
        
        assert_eq!(key1, key2);
    }
    
    #[tokio::test]
    async fn test_threshold_bucketing() {
        let cache = KernelCache::new().await.unwrap();
        
        assert_eq!(cache.threshold_to_bucket(0.85), 85);
        assert_eq!(cache.threshold_to_bucket(0.80), 80);
        assert_eq!(cache.threshold_to_bucket(0.799), 80);
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let cache = KernelCache::new().await.unwrap();
        let stats = cache.get_stats().await;
        
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
    
    #[tokio::test]
    async fn test_workgroup_size_calculation() {
        let cache = KernelCache::new().await.unwrap();
        
        assert_eq!(cache.calculate_optimal_workgroup_size(100), 64);
        assert_eq!(cache.calculate_optimal_workgroup_size(500), 128);
        assert_eq!(cache.calculate_optimal_workgroup_size(5000), 256);
    }
}