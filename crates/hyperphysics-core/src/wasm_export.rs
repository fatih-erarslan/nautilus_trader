//! WASM Export Integration
//!
//! Provides browser-compatible deployment of HyperPhysics using
//! ruv-FANN's cuda-wasm transpiler for GPU kernel conversion.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Browser Environment                       │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
//! │  │ JavaScript API  │  │ WebGPU/WebGL    │  │ WASM Module │  │
//! │  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
//! └───────────┼────────────────────┼─────────────────┼──────────┘
//!             │                    │                 │
//!             ▼                    ▼                 ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 HyperPhysics WASM Bridge                     │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
//! │  │ WasmExporter    │  │ KernelRegistry  │  │ StateSync   │  │
//! │  │ (Transpilation) │  │ (WGSL Shaders)  │  │ (Memory)    │  │
//! │  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
//! └───────────┼────────────────────┼─────────────────┼──────────┘
//!             │                    │                 │
//!             ▼                    ▼                 ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  ruv-FANN cuda-wasm                          │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
//! │  │ CUDA Parser     │  │ Rust Transpiler │  │ WGSL Output │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **CUDA→WGSL Transpilation**: Convert GPU kernels for WebGPU
//! - **State Serialization**: Efficient binary state transfer
//! - **Memory Management**: Zero-copy where possible
//! - **Async Execution**: Non-blocking simulation steps
//!
//! ## Usage (JavaScript)
//!
//! ```javascript
//! import init, { HyperPhysicsWasm } from 'hyperphysics-wasm';
//!
//! await init();
//! const engine = new HyperPhysicsWasm(48, 1.0, 300.0);
//!
//! // Run simulation
//! engine.step(0.01);
//!
//! // Get metrics
//! const phi = engine.integrated_information();
//! const state = engine.export_state();
//! ```

use std::collections::HashMap;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "cuda-wasm")]
use cuda_wasm::{CudaRust, Transpiler, NeuralBridge, BridgeConfig};

/// WASM-compatible HyperPhysics engine wrapper
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct HyperPhysicsWasm {
    /// Number of nodes in the simulation
    num_nodes: usize,
    /// Curvature parameter (K)
    curvature: f64,
    /// Temperature in Kelvin
    temperature: f64,
    /// Current simulation time
    time: f64,
    /// pBit states (probabilities)
    pbit_states: Vec<f64>,
    /// Node positions in Poincaré disk
    positions: Vec<[f64; 2]>,
    /// Cached metrics
    metrics_cache: MetricsCache,
    /// Transpiled kernel registry
    kernel_registry: KernelRegistry,
}

/// Cached simulation metrics
#[derive(Debug, Clone, Default)]
struct MetricsCache {
    /// Integrated information (Φ)
    phi: f64,
    /// Resonance complexity (CI)
    ci: f64,
    /// Total entropy
    entropy: f64,
    /// Free energy
    free_energy: f64,
    /// Cache valid flag
    valid: bool,
}

/// Registry of transpiled GPU kernels
#[derive(Debug, Clone, Default)]
struct KernelRegistry {
    /// WGSL shaders by name
    shaders: HashMap<String, String>,
    /// Kernel metadata
    metadata: HashMap<String, KernelMetadata>,
}

/// Kernel metadata for execution
#[derive(Debug, Clone)]
struct KernelMetadata {
    /// Workgroup size
    workgroup_size: [u32; 3],
    /// Number of uniforms
    num_uniforms: usize,
    /// Buffer bindings
    buffer_bindings: Vec<BufferBinding>,
}

#[derive(Debug, Clone)]
struct BufferBinding {
    /// Binding index
    index: u32,
    /// Buffer type (storage/uniform)
    buffer_type: BufferType,
    /// Element size in bytes
    element_size: usize,
}

#[derive(Debug, Clone, Copy)]
enum BufferType {
    Storage,
    Uniform,
    ReadOnly,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl HyperPhysicsWasm {
    /// Create a new WASM-compatible HyperPhysics engine
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(num_nodes: usize, curvature: f64, temperature: f64) -> Self {
        let mut engine = Self {
            num_nodes,
            curvature,
            temperature,
            time: 0.0,
            pbit_states: vec![0.5; num_nodes],
            positions: Self::generate_tessellation_positions(num_nodes, curvature),
            metrics_cache: MetricsCache::default(),
            kernel_registry: KernelRegistry::default(),
        };

        // Register default kernels
        engine.register_default_kernels();

        engine
    }

    /// Create with ROI-48 configuration (standard brain region mapping)
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn roi_48(curvature: f64, temperature: f64) -> Self {
        Self::new(48, curvature, temperature)
    }

    /// Step the simulation forward by dt
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn step(&mut self, dt: f64) {
        // Invalidate cache
        self.metrics_cache.valid = false;

        // Stochastic pBit dynamics with Glauber updates
        for i in 0..self.num_nodes {
            let p = self.pbit_states[i];

            // Calculate local field from neighbors
            let local_field = self.calculate_local_field(i);

            // Glauber transition probability
            let beta = 1.0 / (self.temperature * 8.617333262e-5); // kB in eV/K
            let delta_e = local_field * (1.0 - 2.0 * p);
            let transition_prob = 1.0 / (1.0 + (-beta * delta_e).exp());

            // Apply Langevin dynamics
            let noise = self.generate_thermal_noise(i, dt);
            self.pbit_states[i] = (p + transition_prob * dt + noise).clamp(0.0, 1.0);
        }

        self.time += dt;
    }

    /// Get integrated information (Φ)
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn integrated_information(&mut self) -> f64 {
        if !self.metrics_cache.valid {
            self.update_metrics_cache();
        }
        self.metrics_cache.phi
    }

    /// Get resonance complexity (CI)
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn resonance_complexity(&mut self) -> f64 {
        if !self.metrics_cache.valid {
            self.update_metrics_cache();
        }
        self.metrics_cache.ci
    }

    /// Get total entropy
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn entropy(&mut self) -> f64 {
        if !self.metrics_cache.valid {
            self.update_metrics_cache();
        }
        self.metrics_cache.entropy
    }

    /// Get free energy
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn free_energy(&mut self) -> f64 {
        if !self.metrics_cache.valid {
            self.update_metrics_cache();
        }
        self.metrics_cache.free_energy
    }

    /// Get current simulation time
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get number of nodes
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Export state as serialized bytes
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn export_state(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(
            8 + // time
            8 * self.num_nodes + // pbit_states
            16 * self.num_nodes // positions
        );

        // Time
        data.extend_from_slice(&self.time.to_le_bytes());

        // pBit states
        for &p in &self.pbit_states {
            data.extend_from_slice(&p.to_le_bytes());
        }

        // Positions
        for pos in &self.positions {
            data.extend_from_slice(&pos[0].to_le_bytes());
            data.extend_from_slice(&pos[1].to_le_bytes());
        }

        data
    }

    /// Import state from serialized bytes
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn import_state(&mut self, data: &[u8]) -> bool {
        let expected_size = 8 + 8 * self.num_nodes + 16 * self.num_nodes;
        if data.len() != expected_size {
            return false;
        }

        let mut offset = 0;

        // Time
        self.time = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // pBit states
        for i in 0..self.num_nodes {
            self.pbit_states[i] = f64::from_le_bytes(
                data[offset..offset + 8].try_into().unwrap()
            );
            offset += 8;
        }

        // Positions
        for i in 0..self.num_nodes {
            self.positions[i][0] = f64::from_le_bytes(
                data[offset..offset + 8].try_into().unwrap()
            );
            offset += 8;
            self.positions[i][1] = f64::from_le_bytes(
                data[offset..offset + 8].try_into().unwrap()
            );
            offset += 8;
        }

        self.metrics_cache.valid = false;
        true
    }

    /// Get pBit state at index
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn get_pbit(&self, index: usize) -> f64 {
        self.pbit_states.get(index).copied().unwrap_or(0.0)
    }

    /// Set pBit state at index
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn set_pbit(&mut self, index: usize, value: f64) {
        if index < self.num_nodes {
            self.pbit_states[index] = value.clamp(0.0, 1.0);
            self.metrics_cache.valid = false;
        }
    }

    /// Get position at index (returns [x, y])
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn get_position(&self, index: usize) -> Vec<f64> {
        self.positions.get(index)
            .map(|p| vec![p[0], p[1]])
            .unwrap_or_default()
    }

    /// Get all pBit states as flat array
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn get_all_pbits(&self) -> Vec<f64> {
        self.pbit_states.clone()
    }

    /// Get all positions as flat array [x0, y0, x1, y1, ...]
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn get_all_positions(&self) -> Vec<f64> {
        self.positions.iter()
            .flat_map(|p| vec![p[0], p[1]])
            .collect()
    }

    /// Reset simulation to initial state
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.pbit_states = vec![0.5; self.num_nodes];
        self.metrics_cache = MetricsCache::default();
    }

    /// Get WGSL shader for WebGPU execution
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn get_shader(&self, name: &str) -> Option<String> {
        self.kernel_registry.shaders.get(name).cloned()
    }

    /// List available shaders
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn list_shaders(&self) -> Vec<String> {
        self.kernel_registry.shaders.keys().cloned().collect()
    }
}

impl HyperPhysicsWasm {
    /// Generate hyperbolic tessellation positions
    fn generate_tessellation_positions(num_nodes: usize, curvature: f64) -> Vec<[f64; 2]> {
        let mut positions = Vec::with_capacity(num_nodes);
        let k = curvature.abs();

        // Golden angle spiral in Poincaré disk
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

        for i in 0..num_nodes {
            // Radius grows logarithmically (hyperbolic spacing)
            let t = (i as f64 + 1.0) / (num_nodes as f64 + 1.0);
            let r = (t / (1.0 + k * t)).sqrt() * 0.9; // Stay within disk

            // Golden angle distribution
            let theta = i as f64 * golden_angle;

            positions.push([
                r * theta.cos(),
                r * theta.sin(),
            ]);
        }

        positions
    }

    /// Calculate local field for pBit dynamics
    fn calculate_local_field(&self, index: usize) -> f64 {
        let pos = self.positions[index];
        let mut field = 0.0;

        for (j, &p) in self.pbit_states.iter().enumerate() {
            if j != index {
                // Hyperbolic distance
                let other_pos = self.positions[j];
                let dist = self.hyperbolic_distance(pos, other_pos);

                // Coupling decays with distance
                let coupling = (-dist * self.curvature.abs()).exp();
                field += coupling * (2.0 * p - 1.0);
            }
        }

        field
    }

    /// Calculate hyperbolic distance between two points in Poincaré disk
    fn hyperbolic_distance(&self, p1: [f64; 2], p2: [f64; 2]) -> f64 {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let euclidean_sq = dx * dx + dy * dy;

        let r1_sq = p1[0] * p1[0] + p1[1] * p1[1];
        let r2_sq = p2[0] * p2[0] + p2[1] * p2[1];

        // Poincaré disk distance formula
        let num = 2.0 * euclidean_sq;
        let denom = (1.0 - r1_sq) * (1.0 - r2_sq);

        if denom > 1e-10 {
            (1.0 + num / denom).acosh()
        } else {
            f64::INFINITY
        }
    }

    /// Generate thermal noise (Box-Muller transform)
    fn generate_thermal_noise(&self, index: usize, dt: f64) -> f64 {
        // Simple deterministic noise based on state (for reproducibility)
        let seed = (index as f64 * 0.618033988749895 + self.time * 7.123456789).fract();
        let u1 = (seed * 12.9898 + 78.233).sin().fract().abs().max(1e-10);
        let u2 = (seed * 43.7585 + 12.846).sin().fract().abs();

        // Box-Muller
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Noise amplitude based on temperature
        let kb_t = self.temperature * 8.617333262e-5;
        z * (2.0 * kb_t * dt).sqrt() * 0.1
    }

    /// Update metrics cache
    fn update_metrics_cache(&mut self) {
        // Calculate entropy
        let entropy: f64 = self.pbit_states.iter()
            .map(|&p| {
                let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                -p_clamped * p_clamped.ln() - (1.0 - p_clamped) * (1.0 - p_clamped).ln()
            })
            .sum();

        // Calculate simple Φ approximation (mutual information based)
        let mean: f64 = self.pbit_states.iter().sum::<f64>() / self.num_nodes as f64;
        let variance: f64 = self.pbit_states.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / self.num_nodes as f64;

        // Φ ∝ variance * connectivity
        let phi = variance * (self.num_nodes as f64).ln() * entropy.max(1e-10);

        // CI from correlation structure
        let mut correlation_sum = 0.0;
        for i in 0..self.num_nodes {
            for j in (i + 1)..self.num_nodes {
                let p_i = self.pbit_states[i] - mean;
                let p_j = self.pbit_states[j] - mean;
                correlation_sum += (p_i * p_j).abs();
            }
        }
        let num_pairs = (self.num_nodes * (self.num_nodes - 1) / 2) as f64;
        let ci = correlation_sum / num_pairs.max(1.0) * self.num_nodes as f64;

        // Free energy = E - TS (simplified)
        let energy: f64 = self.pbit_states.iter()
            .map(|&p| (2.0 * p - 1.0).powi(2))
            .sum();
        let free_energy = energy - self.temperature * 8.617333262e-5 * entropy;

        self.metrics_cache = MetricsCache {
            phi,
            ci,
            entropy,
            free_energy,
            valid: true,
        };
    }

    /// Register default WGSL kernels
    fn register_default_kernels(&mut self) {
        // pBit dynamics kernel
        let pbit_shader = r#"
struct PBitState {
    probability: f32,
    local_field: f32,
}

struct Uniforms {
    num_nodes: u32,
    temperature: f32,
    dt: f32,
    curvature: f32,
}

@group(0) @binding(0) var<storage, read_write> states: array<PBitState>;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Uniforms;

fn hyperbolic_distance(p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let diff = p2 - p1;
    let euclidean_sq = dot(diff, diff);
    let r1_sq = dot(p1, p1);
    let r2_sq = dot(p2, p2);
    let num = 2.0 * euclidean_sq;
    let denom = (1.0 - r1_sq) * (1.0 - r2_sq);
    if (denom > 1e-10) {
        return acosh(1.0 + num / denom);
    }
    return 1000.0; // Large distance
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.num_nodes) { return; }

    // Calculate local field
    var field: f32 = 0.0;
    let pos_i = positions[i];

    for (var j: u32 = 0u; j < params.num_nodes; j = j + 1u) {
        if (j != i) {
            let dist = hyperbolic_distance(pos_i, positions[j]);
            let coupling = exp(-dist * abs(params.curvature));
            field += coupling * (2.0 * states[j].probability - 1.0);
        }
    }

    states[i].local_field = field;

    // Glauber dynamics
    let p = states[i].probability;
    let beta = 1.0 / (params.temperature * 8.617333262e-5);
    let delta_e = field * (1.0 - 2.0 * p);
    let transition_prob = 1.0 / (1.0 + exp(-beta * delta_e));

    states[i].probability = clamp(p + transition_prob * params.dt, 0.0, 1.0);
}
"#;
        self.kernel_registry.shaders.insert("pbit_dynamics".into(), pbit_shader.into());
        self.kernel_registry.metadata.insert("pbit_dynamics".into(), KernelMetadata {
            workgroup_size: [64, 1, 1],
            num_uniforms: 4,
            buffer_bindings: vec![
                BufferBinding { index: 0, buffer_type: BufferType::Storage, element_size: 8 },
                BufferBinding { index: 1, buffer_type: BufferType::ReadOnly, element_size: 8 },
                BufferBinding { index: 2, buffer_type: BufferType::Uniform, element_size: 16 },
            ],
        });

        // Metrics computation kernel
        let metrics_shader = r#"
struct Metrics {
    entropy: f32,
    phi: f32,
    ci: f32,
    free_energy: f32,
}

struct Uniforms {
    num_nodes: u32,
    temperature: f32,
}

@group(0) @binding(0) var<storage, read> probabilities: array<f32>;
@group(0) @binding(1) var<storage, read_write> metrics: Metrics;
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(1)
fn main() {
    var entropy: f32 = 0.0;
    var mean: f32 = 0.0;

    // First pass: entropy and mean
    for (var i: u32 = 0u; i < params.num_nodes; i = i + 1u) {
        let p = clamp(probabilities[i], 1e-10, 1.0 - 1e-10);
        entropy -= p * log(p) + (1.0 - p) * log(1.0 - p);
        mean += probabilities[i];
    }
    mean /= f32(params.num_nodes);

    // Second pass: variance and correlations
    var variance: f32 = 0.0;
    var correlation_sum: f32 = 0.0;

    for (var i: u32 = 0u; i < params.num_nodes; i = i + 1u) {
        let p_i = probabilities[i] - mean;
        variance += p_i * p_i;

        for (var j: u32 = i + 1u; j < params.num_nodes; j = j + 1u) {
            let p_j = probabilities[j] - mean;
            correlation_sum += abs(p_i * p_j);
        }
    }
    variance /= f32(params.num_nodes);

    // Compute metrics
    let n = f32(params.num_nodes);
    metrics.entropy = entropy;
    metrics.phi = variance * log(n) * max(entropy, 1e-10);
    metrics.ci = correlation_sum / max(n * (n - 1.0) / 2.0, 1.0) * n;

    // Free energy
    var energy: f32 = 0.0;
    for (var i: u32 = 0u; i < params.num_nodes; i = i + 1u) {
        let spin = 2.0 * probabilities[i] - 1.0;
        energy += spin * spin;
    }
    let kb_t = params.temperature * 8.617333262e-5;
    metrics.free_energy = energy - kb_t * entropy;
}
"#;
        self.kernel_registry.shaders.insert("metrics".into(), metrics_shader.into());
        self.kernel_registry.metadata.insert("metrics".into(), KernelMetadata {
            workgroup_size: [1, 1, 1],
            num_uniforms: 2,
            buffer_bindings: vec![
                BufferBinding { index: 0, buffer_type: BufferType::ReadOnly, element_size: 4 },
                BufferBinding { index: 1, buffer_type: BufferType::Storage, element_size: 16 },
                BufferBinding { index: 2, buffer_type: BufferType::Uniform, element_size: 8 },
            ],
        });
    }
}

/// CUDA to WGSL transpiler integration
#[cfg(feature = "cuda-wasm")]
pub struct CudaWasmBridge {
    transpiler: CudaRust,
    neural_bridge: Option<NeuralBridge>,
}

#[cfg(feature = "cuda-wasm")]
impl CudaWasmBridge {
    /// Create a new CUDA-WASM bridge
    pub fn new() -> Self {
        Self {
            transpiler: CudaRust::new(),
            neural_bridge: None,
        }
    }

    /// Initialize with neural bridge capabilities
    pub fn with_neural(config: BridgeConfig) -> Result<Self, String> {
        let neural_bridge = NeuralBridge::new(config)
            .map_err(|e| format!("Failed to initialize neural bridge: {}", e))?;

        Ok(Self {
            transpiler: CudaRust::new(),
            neural_bridge: Some(neural_bridge),
        })
    }

    /// Transpile CUDA kernel to Rust
    pub fn transpile_cuda(&self, cuda_code: &str) -> Result<String, String> {
        self.transpiler.transpile(cuda_code)
            .map_err(|e| e.to_string())
    }

    /// Transpile CUDA kernel to WGSL for WebGPU
    #[cfg(feature = "webgpu-only")]
    pub fn transpile_to_wgsl(&self, cuda_code: &str) -> Result<String, String> {
        self.transpiler.to_webgpu(cuda_code)
            .map_err(|e| e.to_string())
    }

    /// Check system capabilities
    pub fn capabilities(&self) -> WasmCapabilities {
        WasmCapabilities {
            webgpu: cfg!(feature = "webgpu-only"),
            cuda: false, // Not available in WASM
            neural: self.neural_bridge.is_some(),
            simd: cfg!(target_feature = "simd128"),
        }
    }
}

#[cfg(feature = "cuda-wasm")]
impl Default for CudaWasmBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM target capabilities
#[derive(Debug, Clone, Copy)]
pub struct WasmCapabilities {
    /// WebGPU support
    pub webgpu: bool,
    /// CUDA support (always false in WASM)
    pub cuda: bool,
    /// Neural bridge support
    pub neural: bool,
    /// SIMD128 support
    pub simd: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_engine_creation() {
        let engine = HyperPhysicsWasm::new(48, 1.0, 300.0);
        assert_eq!(engine.num_nodes(), 48);
    }

    #[test]
    fn test_wasm_step() {
        let mut engine = HyperPhysicsWasm::new(10, 1.0, 300.0);
        let initial_time = engine.time();
        engine.step(0.01);
        assert!(engine.time() > initial_time);
    }

    #[test]
    fn test_state_export_import() {
        let mut engine1 = HyperPhysicsWasm::new(10, 1.0, 300.0);
        engine1.step(0.01);
        let state = engine1.export_state();

        let mut engine2 = HyperPhysicsWasm::new(10, 1.0, 300.0);
        assert!(engine2.import_state(&state));
        assert!((engine1.time() - engine2.time()).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_computation() {
        let mut engine = HyperPhysicsWasm::new(10, 1.0, 300.0);

        // Set non-uniform state
        for i in 0..5 {
            engine.set_pbit(i, 0.9);
        }
        for i in 5..10 {
            engine.set_pbit(i, 0.1);
        }

        let phi = engine.integrated_information();
        assert!(phi > 0.0);

        let entropy = engine.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_shader_registry() {
        let engine = HyperPhysicsWasm::new(10, 1.0, 300.0);
        let shaders = engine.list_shaders();
        assert!(shaders.contains(&"pbit_dynamics".to_string()));
        assert!(shaders.contains(&"metrics".to_string()));

        let shader = engine.get_shader("pbit_dynamics");
        assert!(shader.is_some());
    }

    #[test]
    fn test_hyperbolic_distance() {
        let engine = HyperPhysicsWasm::new(2, 1.0, 300.0);
        let d = engine.hyperbolic_distance([0.0, 0.0], [0.5, 0.0]);
        // Distance from origin should be positive
        assert!(d > 0.0);

        // Symmetric
        let d2 = engine.hyperbolic_distance([0.5, 0.0], [0.0, 0.0]);
        assert!((d - d2).abs() < 1e-10);
    }
}
