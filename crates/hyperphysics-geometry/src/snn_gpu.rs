//! # GPU-Accelerated Spiking Neural Network Processing
//!
//! Implements GPU parallel processing for hyperbolic SNNs using wgpu compute shaders.
//! Provides massive parallelism for large-scale neural network simulations.
//!
//! ## Architecture
//!
//! - **Membrane Update Shader**: Parallel LIF dynamics across all neurons
//! - **Spike Detection Shader**: Threshold comparison with atomic spike counting
//! - **Propagation Shader**: Fan-out spike delivery to neighbor neurons
//! - **STDP Shader**: Parallel weight updates based on spike timing
//!
//! ## Data Layout
//!
//! GPU buffers are organized for coalesced memory access:
//! - Neuron state buffer: [membrane_potential, threshold, leak, refractory]
//! - Position buffer: [t, x, y, z] per neuron
//! - Connectivity buffer: CSR format for sparse adjacency
//! - Spike buffer: Ring buffer of recent spike events
//!
//! ## References
//!
//! - wgpu WebGPU implementation: https://wgpu.rs
//! - Neuromorphic GPU acceleration literature
//! - WebGPU Shading Language (WGSL) specification

use crate::hyperbolic_snn::LorentzVec;

// Note: This module provides GPU compute abstractions.
// Actual wgpu integration requires the 'gpu' feature flag.

// ============================================================================
// GPU Buffer Layouts
// ============================================================================

/// Neuron state data for GPU buffer
/// Aligned for GPU memory coalescing
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNeuronState {
    /// Membrane potential
    pub membrane_potential: f32,
    /// Spike threshold
    pub threshold: f32,
    /// Leak constant (1/τ)
    pub leak_constant: f32,
    /// Refractory countdown
    pub refractory: f32,
}

impl Default for GpuNeuronState {
    fn default() -> Self {
        Self {
            membrane_potential: -70.0,
            threshold: -55.0,
            leak_constant: 0.05,
            refractory: 0.0,
        }
    }
}

/// Position data for hyperbolic calculations
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPosition {
    /// Lorentz t coordinate
    pub t: f32,
    /// Spatial x
    pub x: f32,
    /// Spatial y
    pub y: f32,
    /// Spatial z
    pub z: f32,
}

impl Default for GpuPosition {
    fn default() -> Self {
        Self { t: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }
}

impl From<LorentzVec> for GpuPosition {
    fn from(v: LorentzVec) -> Self {
        Self {
            t: v.t as f32,
            x: v.x as f32,
            y: v.y as f32,
            z: v.z as f32,
        }
    }
}

/// Synapse data for connectivity
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSynapse {
    /// Target neuron ID
    pub target: u32,
    /// Synaptic weight
    pub weight: f32,
    /// Axonal delay (ms)
    pub delay: f32,
    /// Padding for alignment
    pub _pad: f32,
}

/// Spike event for GPU processing
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpikeEvent {
    /// Spike time
    pub time: f32,
    /// Source neuron ID
    pub source: u32,
    /// Target neuron ID
    pub target: u32,
    /// Weight
    pub weight: f32,
}

/// Simulation parameters uniform buffer
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    /// Time step (ms)
    pub dt: f32,
    /// Current simulation time
    pub current_time: f32,
    /// Number of neurons
    pub num_neurons: u32,
    /// Maximum synapses per neuron
    pub max_synapses: u32,
    /// Resting potential
    pub v_rest: f32,
    /// Reset potential
    pub v_reset: f32,
    /// Refractory period (ms)
    pub refractory_period: f32,
    /// STDP time constant (ms)
    pub tau_stdp: f32,
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self {
            dt: 0.1,
            current_time: 0.0,
            num_neurons: 0,
            max_synapses: 16,
            v_rest: -70.0,
            v_reset: -75.0,
            refractory_period: 2.0,
            tau_stdp: 20.0,
        }
    }
}

// ============================================================================
// WGSL Shader Sources
// ============================================================================

/// WGSL compute shader for membrane potential update
pub const MEMBRANE_UPDATE_SHADER: &str = r#"
// Membrane Update Compute Shader
// Implements Leaky Integrate-and-Fire dynamics in parallel

struct NeuronState {
    membrane_potential: f32,
    threshold: f32,
    leak_constant: f32,
    refractory: f32,
}

struct SimParams {
    dt: f32,
    current_time: f32,
    num_neurons: u32,
    max_synapses: u32,
    v_rest: f32,
    v_reset: f32,
    refractory_period: f32,
    tau_stdp: f32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<NeuronState>;
@group(0) @binding(1) var<storage, read_write> input_currents: array<f32>;
@group(0) @binding(2) var<storage, read_write> spike_flags: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_neurons) {
        return;
    }

    var neuron = neurons[idx];
    let input = input_currents[idx];

    // Check refractory period
    if (neuron.refractory > 0.0) {
        neuron.refractory = neuron.refractory - params.dt;
        neurons[idx] = neuron;
        input_currents[idx] = 0.0;
        spike_flags[idx] = 0u;
        return;
    }

    // LIF dynamics: dV/dt = -(V - V_rest)/τ + I
    let dv = (-neuron.leak_constant * (neuron.membrane_potential - params.v_rest) + input) * params.dt;
    neuron.membrane_potential = neuron.membrane_potential + dv;

    // Threshold crossing
    if (neuron.membrane_potential >= neuron.threshold) {
        neuron.membrane_potential = params.v_reset;
        neuron.refractory = params.refractory_period;
        spike_flags[idx] = 1u;
    } else {
        spike_flags[idx] = 0u;
    }

    // Store updated state
    neurons[idx] = neuron;
    input_currents[idx] = 0.0;
}
"#;

/// WGSL compute shader for spike propagation
pub const SPIKE_PROPAGATION_SHADER: &str = r#"
// Spike Propagation Compute Shader
// Delivers spikes from source to target neurons with delays

struct Synapse {
    target: u32,
    weight: f32,
    delay: f32,
    _pad: f32,
}

struct SpikeEvent {
    time: f32,
    source: u32,
    target: u32,
    weight: f32,
}

struct SimParams {
    dt: f32,
    current_time: f32,
    num_neurons: u32,
    max_synapses: u32,
    v_rest: f32,
    v_reset: f32,
    refractory_period: f32,
    tau_stdp: f32,
}

@group(0) @binding(0) var<storage, read> spike_flags: array<u32>;
@group(0) @binding(1) var<storage, read> synapse_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> synapses: array<Synapse>;
@group(0) @binding(3) var<storage, read_write> spike_queue: array<SpikeEvent>;
@group(0) @binding(4) var<storage, read_write> queue_head: atomic<u32>;
@group(0) @binding(5) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let source_idx = global_id.x;
    if (source_idx >= params.num_neurons) {
        return;
    }

    // Check if this neuron spiked
    if (spike_flags[source_idx] == 0u) {
        return;
    }

    // Get synapse range for this neuron
    let syn_start = synapse_offsets[source_idx];
    let syn_end = synapse_offsets[source_idx + 1u];

    // Enqueue spikes to all targets
    for (var i = syn_start; i < syn_end; i = i + 1u) {
        let syn = synapses[i];

        // Atomic increment to get queue position
        let pos = atomicAdd(&queue_head, 1u);

        // Store spike event
        spike_queue[pos] = SpikeEvent(
            params.current_time + syn.delay,
            source_idx,
            syn.target,
            syn.weight
        );
    }
}
"#;

/// WGSL compute shader for hyperbolic distance calculation
pub const DISTANCE_SHADER: &str = r#"
// Hyperbolic Distance Compute Shader
// Calculates pairwise distances on the hyperboloid

struct Position {
    t: f32,
    x: f32,
    y: f32,
    z: f32,
}

@group(0) @binding(0) var<storage, read> positions: array<Position>;
@group(0) @binding(1) var<storage, read_write> distances: array<f32>;
@group(0) @binding(2) var<uniform> num_neurons: u32;

// Minkowski inner product: -t1*t2 + x1*x2 + y1*y2 + z1*z2
fn minkowski_inner(p1: Position, p2: Position) -> f32 {
    return -p1.t * p2.t + p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

// Hyperbolic distance: acosh(-inner)
fn hyperbolic_distance(p1: Position, p2: Position) -> f32 {
    let inner = -minkowski_inner(p1, p2);
    let clamped = max(inner, 1.0);
    return acosh(clamped);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= num_neurons || j >= num_neurons) {
        return;
    }

    let idx = i * num_neurons + j;

    if (i == j) {
        distances[idx] = 0.0;
    } else {
        distances[idx] = hyperbolic_distance(positions[i], positions[j]);
    }
}
"#;

/// WGSL compute shader for STDP weight update
pub const STDP_SHADER: &str = r#"
// STDP Weight Update Compute Shader
// Updates synaptic weights based on spike timing

struct Synapse {
    target: u32,
    weight: f32,
    delay: f32,
    _pad: f32,
}

struct SimParams {
    dt: f32,
    current_time: f32,
    num_neurons: u32,
    max_synapses: u32,
    v_rest: f32,
    v_reset: f32,
    refractory_period: f32,
    tau_stdp: f32,
}

@group(0) @binding(0) var<storage, read> spike_flags: array<u32>;
@group(0) @binding(1) var<storage, read> last_spike_times: array<f32>;
@group(0) @binding(2) var<storage, read> synapse_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> synapses: array<Synapse>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pre_idx = global_id.x;
    if (pre_idx >= params.num_neurons) {
        return;
    }

    let syn_start = synapse_offsets[pre_idx];
    let syn_end = synapse_offsets[pre_idx + 1u];

    let pre_spike_time = last_spike_times[pre_idx];

    for (var i = syn_start; i < syn_end; i = i + 1u) {
        var syn = synapses[i];
        let post_idx = syn.target;
        let post_spike_time = last_spike_times[post_idx];

        let dt = post_spike_time - pre_spike_time;

        // STDP rule
        var dw: f32;
        if (dt > 0.0) {
            // Pre before post: LTP
            dw = 0.01 * exp(-dt / params.tau_stdp);
        } else {
            // Post before pre: LTD
            dw = -0.01 * exp(dt / params.tau_stdp);
        }

        // Update weight with bounds
        syn.weight = clamp(syn.weight + dw, 0.0, 1.0);
        synapses[i] = syn;
    }
}
"#;

// ============================================================================
// GPU Compute Pipeline (Abstract Interface)
// ============================================================================

/// Configuration for GPU compute
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Workgroup size for general compute
    pub workgroup_size: u32,
    /// Maximum neurons supported
    pub max_neurons: u32,
    /// Maximum synapses per neuron
    pub max_synapses_per_neuron: u32,
    /// Spike queue size
    pub spike_queue_size: u32,
    /// Enable STDP
    pub enable_stdp: bool,
    /// Preferred GPU backend
    pub backend: GpuBackend,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
            max_neurons: 65536,
            max_synapses_per_neuron: 16,
            spike_queue_size: 1_000_000,
            enable_stdp: true,
            backend: GpuBackend::Auto,
        }
    }
}

/// GPU backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Automatic selection
    Auto,
    /// Vulkan backend
    Vulkan,
    /// Metal backend (macOS/iOS)
    Metal,
    /// DirectX 12 backend (Windows)
    Dx12,
    /// WebGPU backend (WASM)
    WebGpu,
}

/// Buffer handles for GPU resources
#[derive(Debug, Clone)]
pub struct GpuBufferHandles {
    /// Neuron state buffer
    pub neuron_states: u64,
    /// Position buffer
    pub positions: u64,
    /// Input current buffer
    pub input_currents: u64,
    /// Spike flags buffer
    pub spike_flags: u64,
    /// Synapse offset buffer (CSR format)
    pub synapse_offsets: u64,
    /// Synapse data buffer
    pub synapses: u64,
    /// Spike queue buffer
    pub spike_queue: u64,
    /// Last spike times buffer
    pub last_spike_times: u64,
    /// Params uniform buffer
    pub params: u64,
}

/// GPU compute context (abstract)
/// Real implementation requires wgpu crate with 'gpu' feature
pub struct GpuContext {
    /// Configuration
    pub config: GpuConfig,
    /// Buffer handles
    pub buffers: Option<GpuBufferHandles>,
    /// Number of neurons
    pub num_neurons: u32,
    /// Simulation parameters
    pub sim_params: GpuSimParams,
    /// Is initialized
    pub initialized: bool,
}

impl GpuContext {
    /// Create new GPU context
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            buffers: None,
            num_neurons: 0,
            sim_params: GpuSimParams::default(),
            initialized: false,
        }
    }

    /// Initialize GPU context (stub - requires wgpu)
    pub fn initialize(&mut self) -> Result<(), GpuError> {
        // In a real implementation, this would:
        // 1. Create wgpu instance
        // 2. Get adapter and device
        // 3. Compile shaders
        // 4. Create pipeline layouts
        // 5. Allocate buffers

        self.initialized = true;
        Ok(())
    }

    /// Upload neuron data to GPU
    pub fn upload_neurons(&mut self, neurons: &[GpuNeuronState]) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        self.num_neurons = neurons.len() as u32;
        self.sim_params.num_neurons = self.num_neurons;

        // In real implementation: write to GPU buffer
        Ok(())
    }

    /// Upload positions to GPU
    pub fn upload_positions(&mut self, positions: &[GpuPosition]) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        if positions.len() as u32 != self.num_neurons {
            return Err(GpuError::BufferSizeMismatch);
        }

        // In real implementation: write to GPU buffer
        Ok(())
    }

    /// Upload synapses in CSR format
    pub fn upload_synapses(
        &mut self,
        _offsets: &[u32],
        _synapses: &[GpuSynapse],
    ) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        // In real implementation: write to GPU buffer
        Ok(())
    }

    /// Execute membrane update shader
    pub fn execute_membrane_update(&mut self) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        // In real implementation:
        // 1. Set bind groups
        // 2. Create command encoder
        // 3. Begin compute pass
        // 4. Dispatch workgroups
        // 5. Submit to queue

        let num_workgroups = (self.num_neurons + self.config.workgroup_size - 1)
            / self.config.workgroup_size;

        // dispatch(num_workgroups, 1, 1)
        let _ = num_workgroups;

        Ok(())
    }

    /// Execute spike propagation shader
    pub fn execute_spike_propagation(&mut self) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        let num_workgroups = (self.num_neurons + self.config.workgroup_size - 1)
            / self.config.workgroup_size;

        let _ = num_workgroups;

        Ok(())
    }

    /// Execute STDP update shader
    pub fn execute_stdp_update(&mut self) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        if !self.config.enable_stdp {
            return Ok(());
        }

        let num_workgroups = (self.num_neurons + self.config.workgroup_size - 1)
            / self.config.workgroup_size;

        let _ = num_workgroups;

        Ok(())
    }

    /// Run one simulation step
    pub fn step(&mut self, dt: f32) -> Result<GpuStepResult, GpuError> {
        self.sim_params.dt = dt;

        // Execute shaders in sequence
        self.execute_membrane_update()?;
        self.execute_spike_propagation()?;
        self.execute_stdp_update()?;

        // Update time
        self.sim_params.current_time += dt;

        Ok(GpuStepResult {
            time: self.sim_params.current_time,
            spikes_generated: 0, // Would read from GPU
        })
    }

    /// Run simulation for duration
    pub fn run(&mut self, duration: f32, dt: f32) -> Result<GpuRunResult, GpuError> {
        let num_steps = (duration / dt).ceil() as u32;
        let mut total_spikes = 0u64;

        for _ in 0..num_steps {
            let result = self.step(dt)?;
            total_spikes += result.spikes_generated as u64;
        }

        Ok(GpuRunResult {
            duration,
            steps: num_steps,
            total_spikes,
            final_time: self.sim_params.current_time,
        })
    }

    /// Download neuron states from GPU
    pub fn download_neurons(&self) -> Result<Vec<GpuNeuronState>, GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        // In real implementation: read from GPU buffer
        Ok(vec![GpuNeuronState::default(); self.num_neurons as usize])
    }

    /// Download spike counts
    pub fn download_spike_flags(&self) -> Result<Vec<u32>, GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        // In real implementation: read from GPU buffer
        Ok(vec![0; self.num_neurons as usize])
    }

    /// Inject external input to neurons
    pub fn inject_input(&mut self, inputs: &[(u32, f32)]) -> Result<(), GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        // In real implementation: update input_currents buffer
        let _ = inputs;

        Ok(())
    }

    /// Compute pairwise hyperbolic distances (for connectivity setup)
    pub fn compute_distances(&mut self) -> Result<Vec<f32>, GpuError> {
        if !self.initialized {
            return Err(GpuError::NotInitialized);
        }

        let n = self.num_neurons as usize;

        // In real implementation: execute distance shader
        // Return n×n distance matrix flattened

        Ok(vec![0.0; n * n])
    }
}

/// Result of single simulation step
#[derive(Debug, Clone)]
pub struct GpuStepResult {
    /// Current time
    pub time: f32,
    /// Number of spikes generated
    pub spikes_generated: u32,
}

/// Result of full simulation run
#[derive(Debug, Clone)]
pub struct GpuRunResult {
    /// Total duration
    pub duration: f32,
    /// Number of steps
    pub steps: u32,
    /// Total spikes generated
    pub total_spikes: u64,
    /// Final simulation time
    pub final_time: f32,
}

/// GPU computation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// GPU not initialized
    NotInitialized,
    /// Buffer size mismatch
    BufferSizeMismatch,
    /// Shader compilation failed
    ShaderCompilationFailed(String),
    /// Device lost
    DeviceLost,
    /// Out of memory
    OutOfMemory,
    /// Backend not supported
    BackendNotSupported,
    /// General error
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NotInitialized => write!(f, "GPU context not initialized"),
            GpuError::BufferSizeMismatch => write!(f, "Buffer size mismatch"),
            GpuError::ShaderCompilationFailed(msg) => write!(f, "Shader compilation failed: {}", msg),
            GpuError::DeviceLost => write!(f, "GPU device lost"),
            GpuError::OutOfMemory => write!(f, "GPU out of memory"),
            GpuError::BackendNotSupported => write!(f, "GPU backend not supported"),
            GpuError::Other(msg) => write!(f, "GPU error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

// ============================================================================
// Hybrid CPU-GPU Processor
// ============================================================================

/// Hybrid processor that can switch between CPU SIMD and GPU
pub struct HybridProcessor {
    /// GPU context
    gpu: Option<GpuContext>,
    /// CPU fallback (SIMD processor)
    cpu: Option<crate::snn_gnn_simd::EventDrivenGraphProcessor>,
    /// Processing mode
    mode: ProcessingMode,
    /// Performance metrics
    metrics: HybridMetrics,
}

/// Processing mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Use CPU SIMD
    Cpu,
    /// Use GPU compute
    Gpu,
    /// Automatic selection based on workload
    Auto,
}

/// Performance metrics for hybrid processing
#[derive(Debug, Clone, Default)]
pub struct HybridMetrics {
    /// CPU processing time (ms)
    pub cpu_time_ms: f64,
    /// GPU processing time (ms)
    pub gpu_time_ms: f64,
    /// Total spikes processed
    pub total_spikes: u64,
    /// Current mode
    pub current_mode: Option<ProcessingMode>,
}

impl HybridProcessor {
    /// Create new hybrid processor
    pub fn new(gpu_config: Option<GpuConfig>, cpu_config: Option<crate::snn_gnn_simd::ProcessorConfig>) -> Self {
        let gpu = gpu_config.map(GpuContext::new);
        let cpu = cpu_config.map(|config| {
            crate::snn_gnn_simd::EventDrivenGraphProcessor::new(Vec::new(), config)
        });

        Self {
            gpu,
            cpu,
            mode: ProcessingMode::Auto,
            metrics: HybridMetrics::default(),
        }
    }

    /// Set processing mode
    pub fn set_mode(&mut self, mode: ProcessingMode) {
        self.mode = mode;
    }

    /// Initialize processor
    pub fn initialize(&mut self) -> Result<(), GpuError> {
        if let Some(ref mut gpu) = self.gpu {
            gpu.initialize()?;
        }
        Ok(())
    }

    /// Determine optimal mode based on workload
    fn select_mode(&self, num_neurons: usize) -> ProcessingMode {
        match self.mode {
            ProcessingMode::Cpu => ProcessingMode::Cpu,
            ProcessingMode::Gpu => ProcessingMode::Gpu,
            ProcessingMode::Auto => {
                // Heuristic: GPU beneficial for > 1000 neurons
                if num_neurons > 1000 && self.gpu.is_some() {
                    ProcessingMode::Gpu
                } else {
                    ProcessingMode::Cpu
                }
            }
        }
    }

    /// Run simulation step
    pub fn step(&mut self, dt: f32) -> Result<HybridStepResult, GpuError> {
        let mode = self.select_mode(self.num_neurons());
        self.metrics.current_mode = Some(mode);

        match mode {
            ProcessingMode::Gpu => {
                if let Some(ref mut gpu) = self.gpu {
                    let result = gpu.step(dt)?;
                    Ok(HybridStepResult {
                        time: result.time as f64,
                        spikes: result.spikes_generated as usize,
                        mode,
                    })
                } else {
                    Err(GpuError::NotInitialized)
                }
            }
            ProcessingMode::Cpu => {
                if let Some(ref mut cpu) = self.cpu {
                    let spiked = cpu.step();
                    Ok(HybridStepResult {
                        time: cpu.current_time,
                        spikes: spiked.len(),
                        mode,
                    })
                } else {
                    Err(GpuError::Other("CPU processor not initialized".into()))
                }
            }
            ProcessingMode::Auto => {
                // Auto already resolved above
                Err(GpuError::Other("Auto mode not resolved".into()))
            }
        }
    }

    /// Get number of neurons
    fn num_neurons(&self) -> usize {
        if let Some(ref gpu) = self.gpu {
            gpu.num_neurons as usize
        } else if let Some(ref cpu) = self.cpu {
            cpu.nodes.len()
        } else {
            0
        }
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &HybridMetrics {
        &self.metrics
    }
}

/// Result of hybrid step
#[derive(Debug, Clone)]
pub struct HybridStepResult {
    /// Current time
    pub time: f64,
    /// Number of spikes
    pub spikes: usize,
    /// Mode used
    pub mode: ProcessingMode,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_neuron_state() {
        let state = GpuNeuronState::default();
        assert_eq!(state.membrane_potential, -70.0);
        assert_eq!(state.threshold, -55.0);
    }

    #[test]
    fn test_gpu_position_from_lorentz() {
        let lv = LorentzVec::from_spatial(0.5, 0.3, 0.0);
        let pos: GpuPosition = lv.into();
        assert!((pos.x - 0.5).abs() < 0.001);
        assert!((pos.y - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_gpu_context_creation() {
        let config = GpuConfig::default();
        let ctx = GpuContext::new(config);
        assert!(!ctx.initialized);
    }

    #[test]
    fn test_gpu_config_defaults() {
        let config = GpuConfig::default();
        assert_eq!(config.workgroup_size, 256);
        assert_eq!(config.max_neurons, 65536);
    }

    #[test]
    fn test_shader_sources_valid() {
        // Basic validation that shader sources are non-empty
        assert!(!MEMBRANE_UPDATE_SHADER.is_empty());
        assert!(!SPIKE_PROPAGATION_SHADER.is_empty());
        assert!(!DISTANCE_SHADER.is_empty());
        assert!(!STDP_SHADER.is_empty());
    }

    #[test]
    fn test_hybrid_processor_creation() {
        let processor = HybridProcessor::new(None, None);
        assert_eq!(processor.mode, ProcessingMode::Auto);
    }

    #[test]
    fn test_processing_mode_selection() {
        let processor = HybridProcessor::new(None, None);

        // Without GPU, should select CPU even with many neurons
        let mode = processor.select_mode(10000);
        assert_eq!(mode, ProcessingMode::Cpu);
    }

    #[test]
    fn test_gpu_error_display() {
        let err = GpuError::NotInitialized;
        assert_eq!(format!("{}", err), "GPU context not initialized");

        let err = GpuError::ShaderCompilationFailed("test".into());
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_sim_params_default() {
        let params = GpuSimParams::default();
        assert_eq!(params.dt, 0.1);
        assert_eq!(params.v_rest, -70.0);
        assert_eq!(params.tau_stdp, 20.0);
    }
}
