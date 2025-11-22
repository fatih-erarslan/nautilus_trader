//! Configuration for Quantum LSTM

use crate::types::{EncodingType, BiologicalEffect};
use serde::{Deserialize, Serialize};

/// Configuration for Quantum LSTM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLSTMConfig {
    // Model architecture
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of qubits per gate
    pub num_qubits: usize,
    /// Dropout rate
    pub dropout: f64,
    
    // Quantum configuration
    /// Encoding type for quantum states
    pub encoding_type: EncodingType,
    /// Number of shots for quantum measurements
    pub shots: Option<usize>,
    /// Quantum device backend
    pub quantum_backend: String,
    /// Enable quantum error mitigation
    pub error_mitigation: bool,
    
    // Biological effects
    /// Enable biological quantum effects
    pub use_biological_effects: bool,
    /// Enabled biological effects
    pub biological_effects: Vec<BiologicalEffect>,
    /// Decoherence rate for coherence simulation
    pub decoherence_rate: f64,
    /// Barrier height for tunneling simulation
    pub barrier_height: f64,
    
    // Attention configuration
    /// Enable quantum attention
    pub use_attention: bool,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Attention dropout
    pub attention_dropout: f64,
    
    // Memory configuration
    /// Enable quantum associative memory
    pub use_quantum_memory: bool,
    /// Number of memory slots
    pub memory_size: usize,
    /// Number of ancilla qubits for error correction
    pub num_ancilla_qubits: usize,
    
    // Performance configuration
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size
    pub cache_size: usize,
    /// Number of parallel workers
    pub num_workers: usize,
    
    // Hardware acceleration
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID
    pub gpu_device_id: usize,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    
    // Training configuration
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Gradient clipping value
    pub gradient_clip: Option<f64>,
    
    // Monitoring
    /// Enable metrics collection
    pub collect_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: usize,
    /// Enable profiling
    pub enable_profiling: bool,
}

impl Default for QuantumLSTMConfig {
    fn default() -> Self {
        Self {
            // Model architecture
            input_size: 10,
            hidden_size: 64,
            num_layers: 2,
            num_qubits: 8,
            dropout: 0.1,
            
            // Quantum configuration
            encoding_type: EncodingType::Amplitude,
            shots: None,
            quantum_backend: "default.qubit".to_string(),
            error_mitigation: true,
            
            // Biological effects
            use_biological_effects: true,
            biological_effects: vec![
                BiologicalEffect::Tunneling,
                BiologicalEffect::Coherence,
                BiologicalEffect::Criticality,
            ],
            decoherence_rate: 0.05,
            barrier_height: 0.5,
            
            // Attention configuration
            use_attention: true,
            num_attention_heads: 4,
            attention_dropout: 0.1,
            
            // Memory configuration
            use_quantum_memory: true,
            memory_size: 100,
            num_ancilla_qubits: 4,
            
            // Performance configuration
            batch_size: 32,
            max_seq_length: 100,
            enable_cache: true,
            cache_size: 10000,
            num_workers: 4,
            
            // Hardware acceleration
            use_gpu: false,
            gpu_device_id: 0,
            use_simd: true,
            
            // Training configuration
            learning_rate: 0.001,
            weight_decay: 0.0001,
            gradient_clip: Some(1.0),
            
            // Monitoring
            collect_metrics: true,
            metrics_interval: 100,
            enable_profiling: false,
        }
    }
}

impl QuantumLSTMConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Builder pattern: set number of qubits
    pub fn with_num_qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = num_qubits;
        self
    }
    
    /// Builder pattern: set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }
    
    /// Builder pattern: enable/disable biological effects
    pub fn with_biological_effects(mut self, enable: bool) -> Self {
        self.use_biological_effects = enable;
        self
    }
    
    /// Builder pattern: set quantum backend
    pub fn with_quantum_backend(mut self, backend: impl Into<String>) -> Self {
        self.quantum_backend = backend.into();
        self
    }
    
    /// Builder pattern: enable GPU acceleration
    pub fn with_gpu(mut self, device_id: usize) -> Self {
        self.use_gpu = true;
        self.gpu_device_id = device_id;
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.num_qubits == 0 {
            return Err("Number of qubits must be greater than 0".into());
        }
        
        if self.num_layers == 0 {
            return Err("Number of layers must be greater than 0".into());
        }
        
        if self.hidden_size == 0 {
            return Err("Hidden size must be greater than 0".into());
        }
        
        if self.input_size == 0 {
            return Err("Input size must be greater than 0".into());
        }
        
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("Dropout must be in range [0, 1)".into());
        }
        
        if self.use_attention && self.num_attention_heads == 0 {
            return Err("Number of attention heads must be greater than 0 when attention is enabled".into());
        }
        
        if self.use_attention && self.hidden_size % self.num_attention_heads != 0 {
            return Err("Hidden size must be divisible by number of attention heads".into());
        }
        
        Ok(())
    }
}