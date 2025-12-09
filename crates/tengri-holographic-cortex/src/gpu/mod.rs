//! # GPU Compute Module
//!
//! Provides GPU-accelerated kernels for hyperbolic message passing using:
//! - **WGSL** for cross-platform via wgpu
//! - **Metal** for optimized AMD GPU performance on macOS
//!
//! ## Hardware Target
//! - Intel i9-13900K (AVX2/AVX-512)
//! - AMD Radeon 6800XT (primary GPU, 16GB VRAM)
//! - AMD Radeon 5500XT (secondary GPU, 8GB VRAM)
//!
//! ## Kernel Overview
//!
//! | Kernel | Purpose | Complexity |
//! |--------|---------|------------|
//! | `compute_edge_messages` | Hyperbolic distance + message computation | O(E) |
//! | `aggregate_and_update` | Message aggregation + pBit sampling | O(N×E) |
//! | `aggregate_csr` | CSR-optimized aggregation | O(E) |
//! | `mobius_aggregate` | Möbius-weighted hyperbolic blend | O(N×E) |
//! | `compute_stdp` | STDP weight updates | O(E) |
//! | `pbit_sample_batch` | Batch pBit sampling with RNG | O(N) |
//! | `compute_fields_simd` | SIMD field computation | O(N²/4) |

pub mod runtime;

pub use runtime::GpuRuntime;

use std::borrow::Cow;

/// WGSL shader source for cross-platform GPU compute
pub const WGSL_SOURCE: &str = include_str!("hyperbolic_mp.wgsl");

/// Metal shader source for optimized AMD performance
pub const METAL_SOURCE: &str = include_str!("hyperbolic_mp.metal");

/// GPU configuration for the Tengri Holographic Cortex
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of nodes in the graph
    pub num_nodes: u32,
    /// Number of edges in the graph
    pub num_edges: u32,
    /// Base temperature for pBit sampling
    pub base_temperature: f32,
    /// Hyperbolic curvature (typically -1.0)
    pub curvature: f32,
    /// Workgroup size for compute dispatches
    pub workgroup_size: u32,
    /// Use CSR format for sparse operations
    pub use_csr: bool,
    /// Enable STDP weight updates
    pub enable_stdp: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            num_nodes: 65536,       // 64K nodes
            num_edges: 655360,      // ~10 edges per node
            base_temperature: 2.27, // Ising T_c
            curvature: -1.0,
            workgroup_size: 256,
            use_csr: true,
            enable_stdp: true,
        }
    }
}

/// Node data for GPU buffers (matches shader struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNodeData {
    /// Lorentz coordinates [x0, x1, ..., x11]
    pub coords: [f32; 12],
    /// Bias value
    pub bias: f32,
    /// Temperature
    pub temperature: f32,
    /// State (0 or 1)
    pub state: u32,
    /// Padding for alignment
    pub _padding: u32,
}

impl GpuNodeData {
    /// Create a new GPU node at the hyperbolic origin
    pub fn origin(temperature: f32) -> Self {
        let mut coords = [0.0f32; 12];
        coords[0] = 1.0; // x0 = 1 for origin
        Self {
            coords,
            bias: 0.0,
            temperature,
            state: 0,
            _padding: 0,
        }
    }
    
    /// Create from Euclidean coordinates (lifts to hyperboloid)
    pub fn from_euclidean(z: &[f32], bias: f32, temperature: f32, state: u32) -> Self {
        let mut coords = [0.0f32; 12];
        let mut spatial_norm_sq = 0.0f32;
        
        for (i, &val) in z.iter().take(11).enumerate() {
            coords[i + 1] = val;
            spatial_norm_sq += val * val;
        }
        coords[0] = (1.0 + spatial_norm_sq).sqrt();
        
        Self {
            coords,
            bias,
            temperature,
            state,
            _padding: 0,
        }
    }
}

/// Edge data for GPU buffers
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEdgeData {
    /// Source node index
    pub src: u32,
    /// Destination node index
    pub dst: u32,
    /// Edge weight
    pub weight: f32,
    /// Padding
    pub _padding: u32,
}

/// Spike timing data for STDP
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpikeTime {
    /// Node ID
    pub node_id: u32,
    /// Tick when spike occurred
    pub tick: u32,
}

/// GPU dispatch parameters
#[derive(Debug, Clone)]
pub struct DispatchParams {
    /// Number of workgroups for node operations
    pub node_dispatch: u32,
    /// Number of workgroups for edge operations
    pub edge_dispatch: u32,
}

impl DispatchParams {
    /// Calculate dispatch parameters from config
    pub fn from_config(config: &GpuConfig) -> Self {
        let wg = config.workgroup_size;
        Self {
            node_dispatch: (config.num_nodes + wg - 1) / wg,
            edge_dispatch: (config.num_edges + wg - 1) / wg,
        }
    }
}

/// CSR sparse matrix format for efficient GPU operations
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Row offsets (length: num_nodes + 1)
    pub offsets: Vec<u32>,
    /// Column indices
    pub indices: Vec<u32>,
    /// Weights
    pub weights: Vec<f32>,
}

impl CsrMatrix {
    /// Create empty CSR matrix
    pub fn empty(num_nodes: usize) -> Self {
        Self {
            offsets: vec![0; num_nodes + 1],
            indices: Vec::new(),
            weights: Vec::new(),
        }
    }
    
    /// Create from edge list
    pub fn from_edges(edges: &[GpuEdgeData], num_nodes: usize) -> Self {
        // Count edges per node
        let mut counts = vec![0u32; num_nodes];
        for edge in edges {
            counts[edge.dst as usize] += 1;
        }
        
        // Compute offsets
        let mut offsets = vec![0u32; num_nodes + 1];
        for i in 0..num_nodes {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        
        let num_edges = offsets[num_nodes] as usize;
        let mut indices = vec![0u32; num_edges];
        let mut weights = vec![0.0f32; num_edges];
        let mut current = offsets.clone();
        
        // Fill indices and weights
        for edge in edges {
            let dst = edge.dst as usize;
            let pos = current[dst] as usize;
            indices[pos] = edge.src;
            weights[pos] = edge.weight;
            current[dst] += 1;
        }
        
        Self { offsets, indices, weights }
    }
    
    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Memory estimates for GPU buffers
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Node buffer size in bytes
    pub nodes_bytes: usize,
    /// Edge buffer size in bytes
    pub edges_bytes: usize,
    /// Message buffer size in bytes
    pub messages_bytes: usize,
    /// CSR buffer size in bytes (if used)
    pub csr_bytes: usize,
    /// Total estimated VRAM usage
    pub total_bytes: usize,
}

impl MemoryEstimate {
    /// Calculate memory estimates from config
    pub fn from_config(config: &GpuConfig) -> Self {
        let node_size = std::mem::size_of::<GpuNodeData>();
        let edge_size = std::mem::size_of::<GpuEdgeData>();
        
        let nodes_bytes = config.num_nodes as usize * node_size * 2; // Input + output
        let edges_bytes = config.num_edges as usize * edge_size;
        let messages_bytes = config.num_edges as usize * 4; // f32 per edge
        
        // CSR: offsets + indices + weights
        let csr_bytes = if config.use_csr {
            (config.num_nodes as usize + 1) * 4 + config.num_edges as usize * 8
        } else {
            0
        };
        
        let total_bytes = nodes_bytes + edges_bytes + messages_bytes + csr_bytes;
        
        Self {
            nodes_bytes,
            edges_bytes,
            messages_bytes,
            csr_bytes,
            total_bytes,
        }
    }
    
    /// Format as human-readable string
    pub fn format(&self) -> String {
        fn fmt_bytes(b: usize) -> String {
            if b >= 1 << 30 {
                format!("{:.2} GB", b as f64 / (1 << 30) as f64)
            } else if b >= 1 << 20 {
                format!("{:.2} MB", b as f64 / (1 << 20) as f64)
            } else if b >= 1 << 10 {
                format!("{:.2} KB", b as f64 / (1 << 10) as f64)
            } else {
                format!("{} B", b)
            }
        }
        
        format!(
            "Nodes: {}, Edges: {}, Messages: {}, CSR: {}, Total: {}",
            fmt_bytes(self.nodes_bytes),
            fmt_bytes(self.edges_bytes),
            fmt_bytes(self.messages_bytes),
            fmt_bytes(self.csr_bytes),
            fmt_bytes(self.total_bytes),
        )
    }
}

/// Performance estimates for the hardware target
#[derive(Debug, Clone)]
pub struct PerfEstimate {
    /// Theoretical peak TFLOPS
    pub peak_tflops: f32,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbs: f32,
    /// Estimated time per message pass (ms)
    pub message_pass_ms: f32,
    /// Estimated time per Möbius aggregate (ms)
    pub mobius_aggregate_ms: f32,
    /// Estimated ticks per second
    pub ticks_per_second: f32,
}

impl PerfEstimate {
    /// Estimate for AMD Radeon 6800XT
    pub fn radeon_6800xt(config: &GpuConfig) -> Self {
        // 6800XT specs: ~20.74 TFLOPS, ~512 GB/s
        let peak_tflops = 20.74;
        let memory_bandwidth_gbs = 512.0;
        
        let mem = MemoryEstimate::from_config(config);
        
        // Message pass: read 2 nodes per edge, write 1 message
        // ~24 bytes read + 4 bytes write per edge = 28 bytes
        let mp_bytes = config.num_edges as f32 * 28.0;
        let message_pass_ms = mp_bytes / (memory_bandwidth_gbs * 1e6);
        
        // Möbius aggregate: more compute intensive
        // ~200 FLOPs per edge for Möbius ops
        let mobius_flops = config.num_edges as f32 * 200.0;
        let mobius_aggregate_ms = (mobius_flops / (peak_tflops * 1e9)).max(message_pass_ms * 2.0);
        
        // Total time per tick ≈ message_pass + mobius + overhead
        let total_ms = message_pass_ms + mobius_aggregate_ms + 0.1;
        let ticks_per_second = 1000.0 / total_ms;
        
        Self {
            peak_tflops,
            memory_bandwidth_gbs,
            message_pass_ms,
            mobius_aggregate_ms,
            ticks_per_second,
        }
    }
    
    /// Estimate for AMD Radeon 5500XT (secondary GPU)
    pub fn radeon_5500xt(config: &GpuConfig) -> Self {
        // 5500XT specs: ~5.2 TFLOPS, ~224 GB/s
        let peak_tflops = 5.2;
        let memory_bandwidth_gbs = 224.0;
        
        let mem = MemoryEstimate::from_config(config);
        let mp_bytes = config.num_edges as f32 * 28.0;
        let message_pass_ms = mp_bytes / (memory_bandwidth_gbs * 1e6);
        
        let mobius_flops = config.num_edges as f32 * 200.0;
        let mobius_aggregate_ms = (mobius_flops / (peak_tflops * 1e9)).max(message_pass_ms * 2.0);
        
        let total_ms = message_pass_ms + mobius_aggregate_ms + 0.1;
        let ticks_per_second = 1000.0 / total_ms;
        
        Self {
            peak_tflops,
            memory_bandwidth_gbs,
            message_pass_ms,
            mobius_aggregate_ms,
            ticks_per_second,
        }
    }
}

// Bytemuck support for GPU buffer types
mod bytemuck_impl {
    // Re-export bytemuck traits if available
    #[cfg(feature = "gpu")]
    pub use bytemuck::{Pod, Zeroable};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_node_origin() {
        let node = GpuNodeData::origin(2.27);
        
        // Should be on hyperboloid: ⟨x,x⟩_L = -1
        let constraint = -node.coords[0].powi(2)
            + node.coords[1..].iter().map(|x| x.powi(2)).sum::<f32>();
        
        assert!((constraint + 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_gpu_node_from_euclidean() {
        let z = vec![0.1f32; 11];
        let node = GpuNodeData::from_euclidean(&z, 0.0, 1.0, 0);
        
        // Check lift: x0 = sqrt(1 + ||z||^2)
        let z_norm_sq: f32 = z.iter().map(|x| x * x).sum();
        let expected_x0 = (1.0 + z_norm_sq).sqrt();
        
        assert!((node.coords[0] - expected_x0).abs() < 1e-6);
    }
    
    #[test]
    fn test_csr_from_edges() {
        let edges = vec![
            GpuEdgeData { src: 0, dst: 1, weight: 0.5, _padding: 0 },
            GpuEdgeData { src: 1, dst: 0, weight: 0.5, _padding: 0 },
            GpuEdgeData { src: 0, dst: 2, weight: 0.3, _padding: 0 },
        ];
        
        let csr = CsrMatrix::from_edges(&edges, 3);
        
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.offsets.len(), 4);
    }
    
    #[test]
    fn test_memory_estimate() {
        let config = GpuConfig::default();
        let mem = MemoryEstimate::from_config(&config);
        
        // With 64K nodes, 655K edges, should be < 100 MB
        assert!(mem.total_bytes < 100 * 1024 * 1024);
        
        println!("Memory estimate: {}", mem.format());
    }
    
    #[test]
    fn test_perf_estimate() {
        let config = GpuConfig::default();
        let perf = PerfEstimate::radeon_6800xt(&config);
        
        // Should achieve > 1000 ticks/sec with default config
        assert!(perf.ticks_per_second > 1000.0);
        
        println!("6800XT: {:.0} ticks/sec", perf.ticks_per_second);
    }
    
    #[test]
    fn test_dispatch_params() {
        let config = GpuConfig {
            num_nodes: 10000,
            num_edges: 100000,
            workgroup_size: 256,
            ..Default::default()
        };
        
        let dispatch = DispatchParams::from_config(&config);
        
        assert_eq!(dispatch.node_dispatch, 40); // ceil(10000/256)
        assert_eq!(dispatch.edge_dispatch, 391); // ceil(100000/256)
    }
}
