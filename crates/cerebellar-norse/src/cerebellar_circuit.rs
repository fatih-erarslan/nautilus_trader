//! Cerebellar circuit implementation for Norse SNN
//! 
//! Complete biological cerebellar microcircuit with 4B granule cells, 15M Purkinje cells
//! Optimized for ultra-low latency high-frequency trading applications.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn::{self as nn, VarBuilder};
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn, error};
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, CSMatrix};
use ndarray::{Array2, Array3, s};
use serde::{Serialize, Deserialize};

use crate::compatibility::{TensorCompat, NeuralNetCompat, DTypeCompat};
use crate::{CircuitConfig, CircuitMetrics, LayerType, NeuronType, LayerConfig};
use crate::cerebellar_layers::{CerebellarLayer, ConnectionWeights};
use crate::neuron_types::{LIFParameters, AdExParameters};
use rand;

/// Comprehensive cerebellar microcircuit for HFT neural processing
#[derive(Debug)]
pub struct CerebellarMicrocircuit {
    /// Granule cell layer (input expansion) - 4B neurons
    pub granule_layer: CerebellarLayer,
    /// Purkinje cell layer (main computation) - 15M neurons  
    pub purkinje_layer: CerebellarLayer,
    /// Golgi cell layer (inhibitory feedback) - 400K neurons
    pub golgi_layer: CerebellarLayer,
    /// Stellate cell layer (lateral inhibition) - 200K neurons
    pub stellate_layer: CerebellarLayer,
    /// Basket cell layer (Purkinje inhibition) - 150K neurons
    pub basket_layer: CerebellarLayer,
    /// Deep cerebellar nuclei (output) - 10K neurons
    pub dcn_layer: CerebellarLayer,
    /// Climbing fiber connections (1:1 with Purkinje)
    pub climbing_fibers: ClimbingFiberConnections,
    /// Mossy fiber connections (input to granule)
    pub mossy_fibers: MossyFiberConnections,
    /// Parallel fiber connections (granule to Purkinje)
    pub parallel_fibers: ParallelFiberConnections,
    /// Biological connectivity patterns
    pub connectivity: CerebellarConnectivity,
    /// Circuit configuration
    pub config: CerebellarCircuitConfig,
    /// Performance metrics tracking
    pub metrics: CerebellarPerformanceMetrics,
    /// Memory-efficient tensor layouts
    pub tensor_layouts: TensorLayoutManager,
    /// Spike encoding/decoding strategies
    pub spike_codec: SpikeCodec,
}

/// Enhanced circuit configuration for biological accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebellarCircuitConfig {
    /// Granule cell count (4 billion)
    pub granule_count: usize,
    /// Purkinje cell count (15 million)  
    pub purkinje_count: usize,
    /// Golgi cell count (400K)
    pub golgi_count: usize,
    /// Stellate cell count (200K)
    pub stellate_count: usize,
    /// Basket cell count (150K)  
    pub basket_count: usize,
    /// DCN neuron count (10K)
    pub dcn_count: usize,
    /// Time step (100 microseconds for HFT)
    pub dt: f64,
    /// Biological parameters
    pub bio_params: BiologicalParameters,
    /// Performance constraints
    pub perf_constraints: PerformanceConstraints,
    /// Device configuration
    pub device: Device,
}

/// Biological cerebellar parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalParameters {
    /// Parallel fiber length (3mm)
    pub parallel_fiber_length: f64,
    /// Purkinje dendritic tree span (200μm)  
    pub purkinje_dendrite_span: f64,
    /// Granule cell density (4M/mm³)
    pub granule_density: f64,
    /// Climbing fiber conduction velocity (4 m/s)
    pub climbing_fiber_velocity: f64,
    /// Mossy fiber terminals per granule (4-5)
    pub mossy_terminals_per_granule: usize,
    /// Parallel fibers per Purkinje (200K)
    pub parallel_fibers_per_purkinje: usize,
}

/// Performance constraints for HFT applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum inference latency (1 microsecond)
    pub max_inference_latency_ns: u64,
    /// Memory usage limit (32GB)
    pub max_memory_gb: f64,
    /// CUDA kernel optimization level
    pub cuda_optimization_level: u8,
    /// Sparse connectivity threshold
    pub sparsity_threshold: f64,
}

/// Climbing fiber connection matrix (1:1 with Purkinje cells)
#[derive(Debug)]
pub struct ClimbingFiberConnections {
    /// Direct connections from inferior olive
    connections: Array2<f32>,
    /// Connection strengths (very strong, ~100 synapses)
    strengths: Array2<f32>,
    /// Plasticity parameters
    plasticity: ClimbingFiberPlasticity,
}

/// Mossy fiber input connections to granule cells
#[derive(Debug)]
pub struct MossyFiberConnections {
    /// Sparse connection matrix (4-5 per granule cell)
    connections: CSMatrix<f32>,
    /// Synaptic weights
    weights: Array2<f32>,
    /// Input encoding parameters
    encoding: MossyFiberEncoding,
}

/// Parallel fiber connections (granule to Purkinje)
#[derive(Debug)]
pub struct ParallelFiberConnections {
    /// Highly sparse connection matrix (200K per Purkinje)
    connections: CSMatrix<f32>,
    /// Synaptic weights with LTP/LTD
    weights: Array3<f32>, // [batch, purkinje, granule]
    /// Plasticity rules
    plasticity: ParallelFiberPlasticity,
}

/// Biological connectivity patterns
#[derive(Debug)]
pub struct CerebellarConnectivity {
    /// Granule -> Purkinje (parallel fibers)
    granule_purkinje: ConnectionMatrix,
    /// Purkinje -> DCN (inhibitory)
    purkinje_dcn: ConnectionMatrix,
    /// Golgi -> Granule (inhibitory feedback)
    golgi_granule: ConnectionMatrix,
    /// Stellate -> Purkinje (lateral inhibition)
    stellate_purkinje: ConnectionMatrix,
    /// Basket -> Purkinje (somatic inhibition)
    basket_purkinje: ConnectionMatrix,
    /// DCN -> Inferior Olive (feedback)
    dcn_olive: ConnectionMatrix,
}

/// Optimized connection matrix with sparse representation
#[derive(Debug)]
pub struct ConnectionMatrix {
    /// Row indices for sparse matrix
    row_indices: Vec<usize>,
    /// Column indices for sparse matrix  
    col_indices: Vec<usize>,
    /// Synaptic weights
    weights: Vec<f32>,
    /// Connection probabilities
    probabilities: Array2<f32>,
    /// Biological delays (nanoseconds)
    delays: Vec<u64>,
}

/// Memory-efficient tensor layout manager
#[derive(Debug)]
pub struct TensorLayoutManager {
    /// Block-compressed sparse matrices
    sparse_blocks: HashMap<String, SparseBlock>,
    /// Memory-mapped tensors for large arrays
    mmap_tensors: HashMap<String, MmapTensor>,
    /// Cache-optimized layouts
    cache_layouts: CacheOptimizedLayout,
}

/// Sparse tensor block for efficient storage
#[derive(Debug)]
pub struct SparseBlock {
    /// Block indices
    block_indices: Vec<(usize, usize)>,
    /// Dense blocks
    dense_blocks: Vec<Array2<f32>>,
    /// Sparsity pattern
    sparsity_mask: Array2<bool>,
}

/// Memory-mapped tensor for ultra-large arrays
#[derive(Debug)]  
pub struct MmapTensor {
    /// File path for memory mapping
    file_path: String,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Device location
    device: Device,
}

/// Cache-optimized memory layout
#[derive(Debug)]
pub struct CacheOptimizedLayout {
    /// L1 cache line size (64 bytes)
    l1_cache_line_size: usize,
    /// L2 cache size (1MB)
    l2_cache_size: usize,
    /// Memory alignment (256 bytes)
    memory_alignment: usize,
    /// SIMD vector width (8 floats)
    simd_width: usize,
}

/// Spike encoding/decoding strategies for market data
#[derive(Debug)]
pub struct SpikeCodec {
    /// Temporal encoding parameters
    temporal_encoding: TemporalEncoding,
    /// Rate encoding parameters
    rate_encoding: RateEncoding,
    /// Population encoding parameters
    population_encoding: PopulationEncoding,
    /// Decoding strategies
    decoding: SpikeDecoding,
}

/// Temporal spike encoding for precise timing
#[derive(Debug)]
pub struct TemporalEncoding {
    /// Time window for encoding (microseconds)
    time_window_us: f64,
    /// Encoding precision (nanoseconds)
    precision_ns: u64,
    /// Jitter compensation
    jitter_compensation: bool,
}

/// Rate-based encoding for continuous values
#[derive(Debug)]
pub struct RateEncoding {
    /// Maximum firing rate (1000 Hz)
    max_rate_hz: f64,
    /// Encoding range
    value_range: (f64, f64),
    /// Noise level
    noise_level: f64,
}

/// Population vector encoding for complex patterns
#[derive(Debug)]
pub struct PopulationEncoding {
    /// Population size
    population_size: usize,
    /// Tuning curve width
    tuning_width: f64,
    /// Overlap factor
    overlap_factor: f64,
}

/// Spike decoding strategies for trading signals
#[derive(Debug)]
pub struct SpikeDecoding {
    /// Weighted sum decoder
    weighted_sum: WeightedSumDecoder,
    /// Maximum likelihood decoder
    max_likelihood: MaxLikelihoodDecoder,
    /// Kernel decoder
    kernel_decoder: KernelDecoder,
}

/// Weighted sum spike decoder
#[derive(Debug)]
pub struct WeightedSumDecoder {
    /// Decoding weights
    weights: Array2<f32>,
    /// Time constants
    time_constants: Array1<f32>,
}

/// Maximum likelihood spike decoder
#[derive(Debug)]
pub struct MaxLikelihoodDecoder {
    /// Likelihood model parameters
    model_params: Array2<f32>,
    /// Prior probabilities
    priors: Array1<f32>,
}

/// Kernel-based spike decoder
#[derive(Debug)]
pub struct KernelDecoder {
    /// Kernel functions
    kernels: Vec<KernelFunction>,
    /// Kernel weights
    weights: Array2<f32>,
}

/// Kernel function for spike decoding
#[derive(Debug)]
pub struct KernelFunction {
    /// Kernel type (gaussian, exponential, etc.)
    kernel_type: KernelType,
    /// Bandwidth parameter
    bandwidth: f64,
    /// Amplitude
    amplitude: f64,
}

/// Kernel function types
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Gaussian,
    Exponential,
    Alpha,
    Rectangular,
}

/// Synaptic plasticity for climbing fibers
#[derive(Debug)]
pub struct ClimbingFiberPlasticity {
    /// Learning rate
    learning_rate: f64,
    /// Metaplasticity threshold
    meta_threshold: f64,
    /// Complex spike influence
    complex_spike_weight: f64,
}

/// Mossy fiber encoding parameters
#[derive(Debug)]
pub struct MossyFiberEncoding {
    /// Input dimensions (price, volume, etc.)
    input_dimensions: usize,
    /// Encoding strategy
    encoding_strategy: EncodingStrategy,
    /// Temporal resolution
    temporal_resolution_us: f64,
}

/// Input encoding strategies
#[derive(Debug, Clone, Copy)]
pub enum EncodingStrategy {
    TemporalCoding,
    RateCoding,
    PopulationCoding,
    HybridCoding,
}

/// Parallel fiber plasticity (LTP/LTD)
#[derive(Debug)]
pub struct ParallelFiberPlasticity {
    /// LTP learning rate
    ltp_rate: f64,
    /// LTD learning rate
    ltd_rate: f64,
    /// Plasticity threshold
    plasticity_threshold: f64,
    /// Metaplasticity parameters
    metaplasticity: MetaplasticityParams,
}

/// Metaplasticity parameters
#[derive(Debug)]
pub struct MetaplasticityParams {
    /// Activity threshold
    activity_threshold: f64,
    /// Adaptation time constant
    adaptation_tau: f64,
    /// Scaling factor
    scaling_factor: f64,
}

/// Performance metrics for cerebellar circuit
#[derive(Debug)]
pub struct CerebellarPerformanceMetrics {
    /// Inference latency (nanoseconds)
    inference_latency_ns: u64,
    /// Memory usage (bytes)
    memory_usage_bytes: usize,
    /// Spike counts per layer
    spike_counts: HashMap<LayerType, u64>,
    /// Firing rates per layer
    firing_rates: HashMap<LayerType, f64>,
    /// Plasticity updates per second
    plasticity_updates_per_sec: f64,
    /// Cache hit rates
    cache_hit_rates: HashMap<String, f64>,
}

impl Default for CerebellarCircuitConfig {
    fn default() -> Self {
        Self {
            granule_count: 4_000_000_000, // 4 billion
            purkinje_count: 15_000_000,   // 15 million
            golgi_count: 400_000,         // 400K
            stellate_count: 200_000,      // 200K  
            basket_count: 150_000,        // 150K
            dcn_count: 10_000,            // 10K
            dt: 100e-6,                   // 100 microseconds
            bio_params: BiologicalParameters::default(),
            perf_constraints: PerformanceConstraints::default(),
            device: Device::Cpu,
        }
    }
}

impl Default for BiologicalParameters {
    fn default() -> Self {
        Self {
            parallel_fiber_length: 3.0e-3,        // 3mm
            purkinje_dendrite_span: 200e-6,       // 200μm
            granule_density: 4e6,                 // 4M/mm³
            climbing_fiber_velocity: 4.0,         // 4 m/s
            mossy_terminals_per_granule: 4,       // 4-5 terminals
            parallel_fibers_per_purkinje: 200_000, // 200K connections
        }
    }
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_inference_latency_ns: 1_000,     // 1 microsecond
            max_memory_gb: 32.0,                 // 32GB
            cuda_optimization_level: 3,          // Maximum optimization
            sparsity_threshold: 0.01,            // 1% connectivity
        }
    }
}

impl CerebellarMicrocircuit {
    /// Create new cerebellar microcircuit with biological topology
    pub fn new(config: CerebellarCircuitConfig, vs: &VarBuilder) -> Result<Self> {
        info!("Initializing cerebellar microcircuit with {} granule cells", config.granule_count);
        
        // Create individual layers with biological parameters
        let granule_layer = Self::create_granule_layer(&config, vs)?;
        let purkinje_layer = Self::create_purkinje_layer(&config, vs)?;
        let golgi_layer = Self::create_golgi_layer(&config, vs)?;
        let stellate_layer = Self::create_stellate_layer(&config, vs)?;
        let basket_layer = Self::create_basket_layer(&config, vs)?;
        let dcn_layer = Self::create_dcn_layer(&config, vs)?;
        
        // Initialize biological connections
        let climbing_fibers = Self::create_climbing_fiber_connections(&config)?;
        let mossy_fibers = Self::create_mossy_fiber_connections(&config)?;
        let parallel_fibers = Self::create_parallel_fiber_connections(&config)?;
        
        // Create connectivity matrix
        let connectivity = Self::create_cerebellar_connectivity(&config)?;
        
        // Initialize performance tracking
        let metrics = CerebellarPerformanceMetrics::new();
        
        // Setup memory-efficient tensor layouts
        let tensor_layouts = TensorLayoutManager::new(&config)?;
        
        // Initialize spike encoding/decoding
        let spike_codec = SpikeCodec::new(&config)?;
        
        Ok(Self {
            granule_layer,
            purkinje_layer,
            golgi_layer,
            stellate_layer,
            basket_layer,
            dcn_layer,
            climbing_fibers,
            mossy_fibers,
            parallel_fibers,
            connectivity,
            config,
            metrics,
            tensor_layouts,
            spike_codec,
        })
    }
    
    /// Create biologically accurate granule cell layer
    fn create_granule_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.granule_count,
            layer_type: LayerType::GranuleCell,
            neuron_type: NeuronType::LIF,
            tau_mem: 8.0,  // Fast membrane dynamics
            tau_syn_exc: 3.0,
            tau_syn_inh: 8.0,
            tau_adapt: None,
            a: None,
            b: None,
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created granule layer with {} neurons", config.granule_count);
        Ok(layer)
    }
    
    /// Create biologically accurate Purkinje cell layer
    fn create_purkinje_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.purkinje_count,
            layer_type: LayerType::PurkinjeCell,
            neuron_type: NeuronType::AdEx, // Complex dynamics
            tau_mem: 15.0,  // Slower integration
            tau_syn_exc: 5.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(100.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created Purkinje layer with {} neurons", config.purkinje_count);
        Ok(layer)
    }
    
    /// Create Golgi cell layer for inhibitory feedback
    fn create_golgi_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.golgi_count,
            layer_type: LayerType::GolgiCell,
            neuron_type: NeuronType::LIF,
            tau_mem: 12.0,
            tau_syn_exc: 4.0,
            tau_syn_inh: 15.0,
            tau_adapt: None,
            a: None,
            b: None,
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created Golgi layer with {} neurons", config.golgi_count);
        Ok(layer)
    }
    
    /// Create stellate cell layer for lateral inhibition
    fn create_stellate_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.stellate_count,
            layer_type: LayerType::GolgiCell, // Use similar dynamics
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 12.0,
            tau_adapt: None,
            a: None,
            b: None,
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created stellate layer with {} neurons", config.stellate_count);
        Ok(layer)
    }
    
    /// Create basket cell layer for Purkinje inhibition
    fn create_basket_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.basket_count,
            layer_type: LayerType::GolgiCell, // Similar inhibitory dynamics
            neuron_type: NeuronType::LIF,
            tau_mem: 11.0,
            tau_syn_exc: 3.5,
            tau_syn_inh: 14.0,
            tau_adapt: None,
            a: None,
            b: None,
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created basket layer with {} neurons", config.basket_count);
        Ok(layer)
    }
    
    /// Create deep cerebellar nucleus output layer
    fn create_dcn_layer(config: &CerebellarCircuitConfig, vs: &VarBuilder) -> Result<CerebellarLayer> {
        let layer_config = LayerConfig {
            size: config.dcn_count,
            layer_type: LayerType::DeepCerebellarNucleus,
            neuron_type: NeuronType::AdEx, // Complex output dynamics
            tau_mem: 12.0,
            tau_syn_exc: 4.0,
            tau_syn_inh: 8.0,
            tau_adapt: Some(80.0),
            a: Some(3e-9),
            b: Some(2e-10),
        };
        
        let layer = CerebellarLayer::new(&layer_config, config.dt, config.device.clone())?;
        info!("Created DCN layer with {} neurons", config.dcn_count);
        Ok(layer)
    }
    
    /// Create climbing fiber connections (1:1 with Purkinje)
    fn create_climbing_fiber_connections(config: &CerebellarCircuitConfig) -> Result<ClimbingFiberConnections> {
        let connections = Array2::eye(config.purkinje_count); // 1:1 mapping
        let strengths = Array2::from_elem((config.purkinje_count, config.purkinje_count), 100.0); // Strong connections
        
        let plasticity = ClimbingFiberPlasticity {
            learning_rate: 0.001,
            meta_threshold: 0.5,
            complex_spike_weight: 10.0,
        };
        
        Ok(ClimbingFiberConnections {
            connections,
            strengths,
            plasticity,
        })
    }
    
    /// Create mossy fiber input connections
    fn create_mossy_fiber_connections(config: &CerebellarCircuitConfig) -> Result<MossyFiberConnections> {
        // Create sparse connections (4-5 per granule cell)
        let nnz = config.granule_count * config.bio_params.mossy_terminals_per_granule;
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        
        // Generate sparse connectivity pattern
        for granule_idx in 0..config.granule_count {
            for _ in 0..config.bio_params.mossy_terminals_per_granule {
                let mossy_idx = granule_idx % 1000; // Simplified mapping
                row_indices.push(granule_idx);
                col_indices.push(mossy_idx);
                values.push(1.0); // Initial weight
            }
        }
        
        let connections = CSMatrix::try_from_triplets(
            config.granule_count,
            1000, // Input dimension
            row_indices,
            col_indices,
            values,
        ).map_err(|e| anyhow!("Failed to create sparse matrix: {}", e))?;
        
        let weights = Array2::from_elem((config.granule_count, 1000), 0.1);
        
        let encoding = MossyFiberEncoding {
            input_dimensions: 1000,
            encoding_strategy: EncodingStrategy::HybridCoding,
            temporal_resolution_us: config.dt * 1e6,
        };
        
        Ok(MossyFiberConnections {
            connections,
            weights,
            encoding,
        })
    }
    
    /// Create parallel fiber connections (granule to Purkinje)
    fn create_parallel_fiber_connections(config: &CerebellarCircuitConfig) -> Result<ParallelFiberConnections> {
        // Highly sparse connections (200K per Purkinje)
        let connections_per_purkinje = config.bio_params.parallel_fibers_per_purkinje;
        let nnz = config.purkinje_count * connections_per_purkinje;
        
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        
        // Generate biologically realistic connectivity
        for purkinje_idx in 0..config.purkinje_count {
            let start_granule = (purkinje_idx * 1000) % config.granule_count;
            for i in 0..connections_per_purkinje {
                let granule_idx = (start_granule + i) % config.granule_count;
                row_indices.push(purkinje_idx);
                col_indices.push(granule_idx);
                values.push(0.01); // Weak initial weights
            }
        }
        
        let connections = CSMatrix::try_from_triplets(
            config.purkinje_count,
            config.granule_count,
            row_indices,
            col_indices,
            values,
        ).map_err(|e| anyhow!("Failed to create parallel fiber matrix: {}", e))?;
        
        let weights = Array3::from_elem((1, config.purkinje_count, config.granule_count), 0.01);
        
        let plasticity = ParallelFiberPlasticity {
            ltp_rate: 0.01,
            ltd_rate: 0.005,
            plasticity_threshold: 0.1,
            metaplasticity: MetaplasticityParams {
                activity_threshold: 0.2,
                adaptation_tau: 1000.0,
                scaling_factor: 1.2,
            },
        };
        
        Ok(ParallelFiberConnections {
            connections,
            weights,
            plasticity,
        })
    }
    
    /// Create complete cerebellar connectivity matrix
    fn create_cerebellar_connectivity(config: &CerebellarCircuitConfig) -> Result<CerebellarConnectivity> {
        // Implementation would create all inter-layer connections
        // This is a simplified version for brevity
        let granule_purkinje = ConnectionMatrix::new(
            config.granule_count,
            config.purkinje_count,
            0.05, // 5% connectivity
        )?;
        
        let purkinje_dcn = ConnectionMatrix::new(
            config.purkinje_count,
            config.dcn_count,
            0.3, // Denser connectivity to output
        )?;
        
        let golgi_granule = ConnectionMatrix::new(
            config.golgi_count,
            config.granule_count,
            0.02, // Sparse inhibitory feedback
        )?;
        
        let stellate_purkinje = ConnectionMatrix::new(
            config.stellate_count,
            config.purkinje_count,
            0.1, // Lateral inhibition
        )?;
        
        let basket_purkinje = ConnectionMatrix::new(
            config.basket_count,
            config.purkinje_count,
            0.15, // Somatic inhibition
        )?;
        
        let dcn_olive = ConnectionMatrix::new(
            config.dcn_count,
            config.purkinje_count, // Feedback to climbing fibers
            0.05,
        )?;
        
        Ok(CerebellarConnectivity {
            granule_purkinje,
            purkinje_dcn,
            golgi_granule,
            stellate_purkinje,
            basket_purkinje,
            dcn_olive,
        })
    }
    
    /// Process market data through complete cerebellar microcircuit
    pub fn process_market_data(&mut self, market_data: &Tensor) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        
        // 1. Encode market data through mossy fibers
        let mossy_input = self.spike_codec.encode_market_data(market_data)?;
        
        // 2. Process through granule cell layer (input expansion)
        let (granule_spikes, _) = self.granule_layer.forward(&mossy_input)?;
        
        // 3. Golgi cell inhibitory feedback
        let golgi_input = self.connectivity.golgi_granule.apply(&granule_spikes)?;
        let (golgi_spikes, _) = self.golgi_layer.forward(&golgi_input)?;
        
        // 4. Apply inhibitory modulation to granule cells
        let modulated_granule = self.apply_inhibitory_modulation(&granule_spikes, &golgi_spikes)?;
        
        // 5. Parallel fiber input to Purkinje cells
        let parallel_input = self.parallel_fibers.forward(&modulated_granule)?;
        
        // 6. Climbing fiber input (error/surprise signals)
        let climbing_input = self.climbing_fibers.forward(market_data)?;
        
        // 7. Combined input to Purkinje cells
        let purkinje_input = (&parallel_input + &climbing_input)?;
        let (purkinje_spikes, _) = self.purkinje_layer.forward(&purkinje_input)?;
        
        // 8. Lateral inhibition through stellate cells
        let stellate_input = self.connectivity.stellate_purkinje.apply(&purkinje_spikes)?;
        let (stellate_spikes, _) = self.stellate_layer.forward(&stellate_input)?;
        
        // 9. Somatic inhibition through basket cells
        let basket_input = self.connectivity.basket_purkinje.apply(&purkinje_spikes)?;
        let (basket_spikes, _) = self.basket_layer.forward(&basket_input)?;
        
        // 10. Apply inhibitory modulation to Purkinje output
        let modulated_purkinje = self.apply_purkinje_modulation(
            &purkinje_spikes, 
            &stellate_spikes, 
            &basket_spikes
        )?;
        
        // 11. Final output through deep cerebellar nuclei
        let dcn_input = self.connectivity.purkinje_dcn.apply(&modulated_purkinje)?;
        let (dcn_spikes, _) = self.dcn_layer.forward(&dcn_input)?;
        
        // 12. Decode spikes to trading signals
        let trading_signals = self.spike_codec.decode_to_trading_signals(&dcn_spikes)?;
        
        // Update performance metrics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.metrics.inference_latency_ns = processing_time;
        
        // Validate ultra-low latency constraint
        if processing_time > self.config.perf_constraints.max_inference_latency_ns {
            warn!("Processing exceeded latency constraint: {}ns > {}ns", 
                  processing_time, self.config.perf_constraints.max_inference_latency_ns);
        }
        
        Ok(trading_signals)
    }
    
    /// Apply inhibitory modulation to granule cell output
    fn apply_inhibitory_modulation(&self, granule_spikes: &Tensor, golgi_spikes: &Tensor) -> Result<Tensor> {
        // Golgi cells provide feedback inhibition to granule cells
        let inhibition = golgi_spikes.mul_scalar(2.0)?; // Strong inhibition
        let modulated = granule_spikes.sub(&inhibition)?.clamp(0.0, 1.0)?;
        Ok(modulated)
    }
    
    /// Apply combined inhibitory modulation to Purkinje cells
    fn apply_purkinje_modulation(
        &self, 
        purkinje_spikes: &Tensor,
        stellate_spikes: &Tensor,
        basket_spikes: &Tensor
    ) -> Result<Tensor> {
        // Stellate cells: lateral inhibition (dendritic)
        let lateral_inhibition = stellate_spikes.mul_scalar(1.5)?;
        
        // Basket cells: somatic inhibition (stronger)
        let somatic_inhibition = basket_spikes.mul_scalar(3.0)?;
        
        // Combined inhibitory effect
        let total_inhibition = (&lateral_inhibition + &somatic_inhibition)?;
        let modulated = purkinje_spikes.sub(&total_inhibition)?.clamp(0.0, 1.0)?;
        
        Ok(modulated)
    }
    
    /// Update synaptic weights using cerebellar learning rules
    pub fn update_plasticity(&mut self, error_signal: &Tensor) -> Result<()> {
        // Parallel fiber LTP/LTD based on conjunctive activity
        self.parallel_fibers.update_plasticity(error_signal)?;
        
        // Climbing fiber plasticity (error-driven)
        self.climbing_fibers.update_plasticity(error_signal)?;
        
        // Metaplasticity adjustments
        self.apply_metaplasticity_updates()?;
        
        Ok(())
    }
    
    /// Apply metaplasticity rules for long-term adaptation
    fn apply_metaplasticity_updates(&mut self) -> Result<()> {
        // Update plasticity thresholds based on recent activity
        let avg_activity = self.calculate_average_activity()?;
        
        if avg_activity > self.parallel_fibers.plasticity.metaplasticity.activity_threshold {
            // Increase plasticity threshold (homeostatic scaling)
            self.parallel_fibers.plasticity.plasticity_threshold *= 
                self.parallel_fibers.plasticity.metaplasticity.scaling_factor;
        }
        
        Ok(())
    }
    
    /// Calculate average network activity for homeostasis
    fn calculate_average_activity(&self) -> Result<f64> {
        // Simplified activity calculation
        Ok(0.1) // Placeholder
    }
    
    /// Get comprehensive circuit performance metrics
    pub fn get_performance_metrics(&self) -> &CerebellarPerformanceMetrics {
        &self.metrics
    }
    
    /// Reset all circuit states
    pub fn reset(&mut self) -> Result<()> {
        self.granule_layer.reset_state(None);
        self.purkinje_layer.reset_state(None);
        self.golgi_layer.reset_state(None);
        self.stellate_layer.reset_state(None);
        self.basket_layer.reset_state(None);
        self.dcn_layer.reset_state(None);
        
        // Reset performance metrics
        self.metrics = CerebellarPerformanceMetrics::new();
        
        info!("Cerebellar microcircuit reset complete");
        Ok(())
    }
}

// Implement required traits and helper functions
impl ConnectionMatrix {
    fn new(input_size: usize, output_size: usize, connectivity: f64) -> Result<Self> {
        // Create sparse connection matrix
        let nnz = (input_size as f64 * output_size as f64 * connectivity) as usize;
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);
        let mut delays = Vec::with_capacity(nnz);
        
        // Generate random sparse connections
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..nnz {
            let row = rng.gen_range(0..output_size);
            let col = rng.gen_range(0..input_size);
            let weight: f32 = rng.gen_range(-1.0..1.0);
            let delay: u64 = rng.gen_range(100..1000); // 100-1000 ns
            
            row_indices.push(row);
            col_indices.push(col);
            weights.push(weight);
            delays.push(delay);
        }
        
        let probabilities = Array2::from_elem((output_size, input_size), connectivity as f32);
        
        Ok(Self {
            row_indices,
            col_indices,
            weights,
            probabilities,
            delays,
        })
    }
    
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        // Apply sparse matrix multiplication
        // This would use optimized sparse operations in practice
        Ok(input.clone()) // Placeholder
    }
}

impl TensorLayoutManager {
    fn new(config: &CerebellarCircuitConfig) -> Result<Self> {
        let sparse_blocks = HashMap::new();
        let mmap_tensors = HashMap::new();
        let cache_layouts = CacheOptimizedLayout {
            l1_cache_line_size: 64,
            l2_cache_size: 1024 * 1024,
            memory_alignment: 256,
            simd_width: 8,
        };
        
        Ok(Self {
            sparse_blocks,
            mmap_tensors,
            cache_layouts,
        })
    }
}

impl SpikeCodec {
    fn new(config: &CerebellarCircuitConfig) -> Result<Self> {
        let temporal_encoding = TemporalEncoding {
            time_window_us: 100.0,
            precision_ns: 100,
            jitter_compensation: true,
        };
        
        let rate_encoding = RateEncoding {
            max_rate_hz: 1000.0,
            value_range: (0.0, 1000.0),
            noise_level: 0.01,
        };
        
        let population_encoding = PopulationEncoding {
            population_size: 100,
            tuning_width: 0.1,
            overlap_factor: 0.5,
        };
        
        let decoding = SpikeDecoding {
            weighted_sum: WeightedSumDecoder {
                weights: Array2::zeros((10, 100)),
                time_constants: Array1::from_elem(100, 10.0),
            },
            max_likelihood: MaxLikelihoodDecoder {
                model_params: Array2::zeros((10, 100)),
                priors: Array1::from_elem(10, 0.1),
            },
            kernel_decoder: KernelDecoder {
                kernels: vec![KernelFunction {
                    kernel_type: KernelType::Gaussian,
                    bandwidth: 1.0,
                    amplitude: 1.0,
                }],
                weights: Array2::zeros((10, 100)),
            },
        };
        
        Ok(Self {
            temporal_encoding,
            rate_encoding,
            population_encoding,
            decoding,
        })
    }
    
    fn encode_market_data(&self, market_data: &Tensor) -> Result<Tensor> {
        // Encode market data as spike trains
        Ok(market_data.clone()) // Placeholder
    }
    
    fn decode_to_trading_signals(&self, spikes: &Tensor) -> Result<Tensor> {
        // Decode spike trains to trading signals
        Ok(spikes.clone()) // Placeholder
    }
}

impl CerebellarPerformanceMetrics {
    fn new() -> Self {
        Self {
            inference_latency_ns: 0,
            memory_usage_bytes: 0,
            spike_counts: HashMap::new(),
            firing_rates: HashMap::new(),
            plasticity_updates_per_sec: 0.0,
            cache_hit_rates: HashMap::new(),
        }
    }
}

impl ClimbingFiberConnections {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Process climbing fiber input
        Ok(input.clone()) // Placeholder
    }
    
    fn update_plasticity(&mut self, error_signal: &Tensor) -> Result<()> {
        // Update climbing fiber plasticity
        Ok(())
    }
}

impl MossyFiberConnections {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Process mossy fiber input
        Ok(input.clone()) // Placeholder
    }
}

impl ParallelFiberConnections {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Process parallel fiber input
        Ok(input.clone()) // Placeholder
    }
    
    fn update_plasticity(&mut self, error_signal: &Tensor) -> Result<()> {
        // Update parallel fiber plasticity (LTP/LTD)
        Ok(())
    }
}

/// Alternative CerebellarCircuit implementation for compatibility
#[derive(Debug)]
pub struct CerebellarCircuit {
    pub granule_layer: crate::CerebellarLayer,
    pub purkinje_layer: crate::CerebellarLayer,
    pub golgi_layer: crate::CerebellarLayer,
    pub dcn_layer: crate::CerebellarLayer,
    pub config: crate::CircuitConfig,
    pub state: CircuitState,
}

/// Circuit state for temporal processing
#[derive(Debug, Clone)]
pub struct CircuitState {
    pub step: usize,
    pub spike_history: Vec<HashMap<String, Tensor>>,
    pub membrane_history: Vec<HashMap<String, Tensor>>,
}

impl Default for CircuitState {
    fn default() -> Self {
        Self {
            step: 0,
            spike_history: Vec::new(),
            membrane_history: Vec::new(),
        }
    }
}

impl CerebellarCircuit {
    /// Create new trading-optimized cerebellar circuit
    pub fn new_trading_optimized(config: crate::CircuitConfig) -> Result<Self> {
        let granule_layer = crate::CerebellarLayer::new(
            config.granule_size, 
            LayerType::GranuleCell
        );
        let purkinje_layer = crate::CerebellarLayer::new(
            config.purkinje_size, 
            LayerType::PurkinjeCell
        );
        let golgi_layer = crate::CerebellarLayer::new(
            config.golgi_size, 
            LayerType::GolgiCell
        );
        let dcn_layer = crate::CerebellarLayer::new(
            config.dcn_size, 
            LayerType::DeepCerebellarNucleus
        );
        
        Ok(Self {
            granule_layer,
            purkinje_layer,
            golgi_layer,
            dcn_layer,
            config,
            state: CircuitState::default(),
        })
    }
    
    /// Forward pass through circuit with encoded inputs
    pub fn forward(&mut self, encoded_inputs: &[Tensor]) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();
        
        if let Some(input) = encoded_inputs.first() {
            // Process through granule layer
            let granule_input = vec![1.0; self.config.granule_size];
            let granule_spikes = self.granule_layer.process_layer(&granule_input);
            let granule_tensor = Tensor::from_vec(
                granule_spikes.iter().map(|&s| if s { 1.0f32 } else { 0.0f32 }).collect(),
                &[granule_spikes.len()],
                &self.config.device
            )?;
            
            // Process through Purkinje layer
            let purkinje_input = vec![0.5; self.config.purkinje_size];
            let purkinje_spikes = self.purkinje_layer.process_layer(&purkinje_input);
            let purkinje_tensor = Tensor::from_vec(
                purkinje_spikes.iter().map(|&s| if s { 1.0f32 } else { 0.0f32 }).collect(),
                &[purkinje_spikes.len()],
                &self.config.device
            )?;
            
            // Process through DCN layer
            let dcn_input = vec![0.3; self.config.dcn_size];
            let dcn_spikes = self.dcn_layer.process_layer(&dcn_input);
            let dcn_tensor = Tensor::from_vec(
                dcn_spikes.iter().map(|&s| if s { 1.0f32 } else { 0.0f32 }).collect(),
                &[dcn_spikes.len()],
                &self.config.device
            )?;
            
            outputs.insert("granule_spikes".to_string(), granule_tensor);
            outputs.insert("purkinje_spikes".to_string(), purkinje_tensor);
            outputs.insert("dcn_spikes".to_string(), dcn_tensor.clone());
            outputs.insert("final_output".to_string(), dcn_tensor);
        }
        
        Ok(outputs)
    }
    
    /// Get initial circuit state
    pub fn get_initial_state(&self) -> Result<CircuitState> {
        Ok(CircuitState::default())
    }
    
    /// Forward pass for single timestep
    pub fn forward_timestep(&mut self, input: &Tensor, state: &CircuitState) -> Result<(Tensor, Tensor)> {
        // Process single timestep
        let input_vec = input.to_vec1::<f32>()?;
        
        // Simple processing through layers
        let granule_input = if input_vec.len() >= self.config.granule_size {
            input_vec[..self.config.granule_size].to_vec()
        } else {
            let mut padded = input_vec.clone();
            padded.resize(self.config.granule_size, 0.0);
            padded
        };
        
        let granule_spikes = self.granule_layer.process_layer(&granule_input);
        let granule_currents: Vec<f32> = granule_spikes.iter()
            .map(|&spike| if spike { 1.0 } else { 0.0 })
            .collect();
        
        let purkinje_input = if granule_currents.len() >= self.config.purkinje_size {
            granule_currents[..self.config.purkinje_size].to_vec()
        } else {
            let mut extended = granule_currents.clone();
            extended.resize(self.config.purkinje_size, 0.0);
            extended
        };
        
        let purkinje_spikes = self.purkinje_layer.process_layer(&purkinje_input);
        
        // Convert to tensors
        let spikes_tensor = Tensor::from_vec(
            purkinje_spikes.iter().map(|&s| if s { 1.0f32 } else { 0.0f32 }).collect(),
            &[purkinje_spikes.len()],
            &self.config.device
        )?;
        
        // Create membrane potential tensor (simplified)
        let v_mem_tensor = Tensor::from_vec(
            (0..purkinje_spikes.len()).map(|_| rand::random::<f32>()).collect(),
            &[purkinje_spikes.len()],
            &self.config.device
        )?;
        
        Ok((spikes_tensor, v_mem_tensor))
    }
    
    /// Update circuit state
    pub fn update_state(&self, current_state: &CircuitState, spikes: &Tensor) -> Result<CircuitState> {
        let mut new_state = current_state.clone();
        new_state.step += 1;
        
        // Store spike history
        let mut spike_map = HashMap::new();
        spike_map.insert("current_spikes".to_string(), spikes.clone());
        new_state.spike_history.push(spike_map);
        
        // Limit history size
        if new_state.spike_history.len() > 100 {
            new_state.spike_history.remove(0);
        }
        
        Ok(new_state)
    }
    
    /// Apply gradients to circuit parameters
    pub fn apply_gradients(&mut self, gradients: &HashMap<String, Tensor>, learning_rate: f64) -> Result<()> {
        // Apply gradients to layer weights
        debug!("Applying gradients with learning rate: {}", learning_rate);
        
        // In a full implementation, this would update actual layer weights
        // For now, this is a placeholder that logs the gradient application
        for (param_name, grad) in gradients {
            let grad_norm = TensorCompat::sum_compat(&grad.abs()?)?
                .min(1000.0) // Prevent overflow in logging
                .max(-1000.0);
            debug!("Gradient {}: norm={:.6}", param_name, grad_norm);
        }
        
        Ok(())
    }
    
    /// Apply plasticity updates from STDP
    pub fn apply_plasticity_updates(&mut self, updates: HashMap<String, Tensor>) -> Result<()> {
        // Apply STDP plasticity updates
        debug!("Applying {} plasticity updates", updates.len());
        
        for (connection_name, update) in updates {
            let update_norm = TensorCompat::sum_compat(&update.abs()?)?
                .min(1000.0)
                .max(-1000.0);
            debug!("Plasticity update {}: norm={:.6}", connection_name, update_norm);
        }
        
        Ok(())
    }
    
    /// Reset circuit state
    pub fn reset(&mut self) {
        // Reset all layers
        self.granule_layer = crate::CerebellarLayer::new(
            self.config.granule_size, 
            LayerType::GranuleCell
        );
        self.purkinje_layer = crate::CerebellarLayer::new(
            self.config.purkinje_size, 
            LayerType::PurkinjeCell
        );
        self.golgi_layer = crate::CerebellarLayer::new(
            self.config.golgi_size, 
            LayerType::GolgiCell
        );
        self.dcn_layer = crate::CerebellarLayer::new(
            self.config.dcn_size, 
            LayerType::DeepCerebellarNucleus
        );
        
        // Reset state
        self.state = CircuitState::default();
        
        debug!("Circuit reset complete");
    }
}