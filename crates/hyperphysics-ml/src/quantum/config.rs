//! Configuration for quantum-inspired neural networks
//!
//! Provides comprehensive configuration with builder pattern.

use super::types::{EncodingType, BiologicalEffect};

/// Configuration for Quantum-Inspired LSTM
#[derive(Debug, Clone)]
pub struct QuantumLSTMConfig {
    // Model architecture
    /// Input feature size
    pub input_size: usize,
    /// Hidden state size
    pub hidden_size: usize,
    /// Number of stacked LSTM layers
    pub num_layers: usize,
    /// Dropout probability between layers
    pub dropout: f32,
    /// Use bidirectional processing
    pub bidirectional: bool,

    // Quantum-inspired configuration
    /// State encoding type
    pub encoding_type: EncodingType,
    /// Use complex-valued operations
    pub use_complex: bool,
    /// Number of virtual "qubits" per hidden unit
    pub qubits_per_unit: usize,

    // Biological effects
    /// Enable biological quantum effects
    pub use_biological_effects: bool,
    /// Which biological effects to apply
    pub biological_effects: Vec<BiologicalEffect>,
    /// Decoherence rate (0 = no decoherence, 1 = full decoherence per step)
    pub decoherence_rate: f32,
    /// Tunneling probability (probability of barrier penetration)
    pub tunneling_probability: f32,
    /// Coherence time (steps before significant decoherence)
    pub coherence_time: f32,

    // Attention configuration
    /// Use quantum-inspired attention
    pub use_attention: bool,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Attention dropout
    pub attention_dropout: f32,

    // Performance configuration
    /// Maximum sequence length (for positional encoding)
    pub max_seq_length: usize,
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,

    // Training configuration
    /// Initial learning rate
    pub learning_rate: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f32>,
    /// Phase regularization strength
    pub phase_regularization: f32,
}

impl Default for QuantumLSTMConfig {
    fn default() -> Self {
        Self {
            // Model architecture
            input_size: 10,
            hidden_size: 64,
            num_layers: 2,
            dropout: 0.1,
            bidirectional: false,

            // Quantum-inspired configuration
            encoding_type: EncodingType::Amplitude,
            use_complex: true,
            qubits_per_unit: 4,

            // Biological effects
            use_biological_effects: true,
            biological_effects: vec![
                BiologicalEffect::Tunneling,
                BiologicalEffect::Coherence,
                BiologicalEffect::Criticality,
            ],
            decoherence_rate: 0.05,
            tunneling_probability: 0.1,
            coherence_time: 20.0,

            // Attention configuration
            use_attention: false,
            num_attention_heads: 4,
            attention_dropout: 0.1,

            // Performance configuration
            max_seq_length: 512,
            gradient_checkpointing: false,

            // Training configuration
            learning_rate: 0.001,
            weight_decay: 0.0001,
            gradient_clip: Some(1.0),
            phase_regularization: 0.01,
        }
    }
}

impl QuantumLSTMConfig {
    /// Create new configuration with defaults
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            ..Default::default()
        }
    }

    /// Builder: set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Builder: set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout.clamp(0.0, 0.99);
        self
    }

    /// Builder: enable/disable bidirectional
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Builder: set encoding type
    pub fn with_encoding(mut self, encoding: EncodingType) -> Self {
        self.encoding_type = encoding;
        self
    }

    /// Builder: enable/disable complex operations
    pub fn with_complex(mut self, use_complex: bool) -> Self {
        self.use_complex = use_complex;
        self
    }

    /// Builder: enable/disable biological effects
    pub fn with_biological_effects(mut self, enable: bool) -> Self {
        self.use_biological_effects = enable;
        self
    }

    /// Builder: set specific biological effects
    pub fn with_effects(mut self, effects: Vec<BiologicalEffect>) -> Self {
        self.biological_effects = effects;
        self.use_biological_effects = !self.biological_effects.is_empty();
        self
    }

    /// Builder: set decoherence rate
    pub fn with_decoherence_rate(mut self, rate: f32) -> Self {
        self.decoherence_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Builder: set tunneling probability
    pub fn with_tunneling(mut self, prob: f32) -> Self {
        self.tunneling_probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Builder: enable attention
    pub fn with_attention(mut self, num_heads: usize) -> Self {
        self.use_attention = true;
        self.num_attention_heads = num_heads;
        self
    }

    /// Builder: set maximum sequence length
    pub fn with_max_seq_length(mut self, max_len: usize) -> Self {
        self.max_seq_length = max_len;
        self
    }

    /// Builder: enable gradient checkpointing
    pub fn with_gradient_checkpointing(mut self, enable: bool) -> Self {
        self.gradient_checkpointing = enable;
        self
    }

    /// Builder: set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set gradient clipping
    pub fn with_gradient_clip(mut self, clip: f32) -> Self {
        self.gradient_clip = Some(clip);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.input_size == 0 {
            return Err("Input size must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("Hidden size must be > 0".to_string());
        }
        if self.num_layers == 0 {
            return Err("Number of layers must be > 0".to_string());
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("Dropout must be in [0, 1)".to_string());
        }
        if self.use_attention && self.hidden_size % self.num_attention_heads != 0 {
            return Err(format!(
                "Hidden size {} must be divisible by attention heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.decoherence_rate < 0.0 || self.decoherence_rate > 1.0 {
            return Err("Decoherence rate must be in [0, 1]".to_string());
        }
        if self.tunneling_probability < 0.0 || self.tunneling_probability > 1.0 {
            return Err("Tunneling probability must be in [0, 1]".to_string());
        }
        Ok(())
    }

    /// Get effective hidden size (doubled if bidirectional)
    pub fn effective_hidden_size(&self) -> usize {
        if self.bidirectional {
            self.hidden_size * 2
        } else {
            self.hidden_size
        }
    }
}

/// Configuration for BioCognitive LSTM (simplified biological model)
#[derive(Debug, Clone)]
pub struct BioCognitiveConfig {
    /// Input feature size
    pub input_size: usize,
    /// Hidden state size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,

    // Biological parameters
    /// Membrane time constant (ms-equivalent in steps)
    pub tau_membrane: f32,
    /// Synaptic time constant
    pub tau_synaptic: f32,
    /// Spike threshold
    pub spike_threshold: f32,
    /// Refractory period (steps)
    pub refractory_period: usize,
    /// Adaptation strength
    pub adaptation_strength: f32,

    // Oscillation parameters
    /// Enable gamma oscillations (30-100 Hz equivalent)
    pub enable_gamma: bool,
    /// Gamma frequency
    pub gamma_frequency: f32,
    /// Enable theta oscillations (4-8 Hz equivalent)
    pub enable_theta: bool,
    /// Theta frequency
    pub theta_frequency: f32,

    // Plasticity parameters
    /// Enable short-term plasticity
    pub enable_stp: bool,
    /// Facilitation time constant
    pub tau_facilitation: f32,
    /// Depression time constant
    pub tau_depression: f32,
}

impl Default for BioCognitiveConfig {
    fn default() -> Self {
        Self {
            input_size: 10,
            hidden_size: 64,
            num_layers: 1,

            // Biological parameters (normalized to step-based time)
            tau_membrane: 10.0,
            tau_synaptic: 5.0,
            spike_threshold: 1.0,
            refractory_period: 2,
            adaptation_strength: 0.1,

            // Oscillations
            enable_gamma: true,
            gamma_frequency: 0.1,  // Normalized frequency
            enable_theta: true,
            theta_frequency: 0.02, // Normalized frequency

            // Plasticity
            enable_stp: true,
            tau_facilitation: 50.0,
            tau_depression: 200.0,
        }
    }
}

impl BioCognitiveConfig {
    /// Create new config
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            ..Default::default()
        }
    }

    /// Builder: set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Builder: set membrane time constant
    pub fn with_tau_membrane(mut self, tau: f32) -> Self {
        self.tau_membrane = tau.max(0.1);
        self
    }

    /// Builder: enable/disable oscillations
    pub fn with_oscillations(mut self, gamma: bool, theta: bool) -> Self {
        self.enable_gamma = gamma;
        self.enable_theta = theta;
        self
    }

    /// Builder: enable/disable short-term plasticity
    pub fn with_stp(mut self, enable: bool) -> Self {
        self.enable_stp = enable;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.input_size == 0 {
            return Err("Input size must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("Hidden size must be > 0".to_string());
        }
        if self.tau_membrane <= 0.0 {
            return Err("Membrane time constant must be > 0".to_string());
        }
        if self.tau_synaptic <= 0.0 {
            return Err("Synaptic time constant must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_config_builder() {
        let config = QuantumLSTMConfig::new(10, 64)
            .with_num_layers(3)
            .with_dropout(0.2)
            .with_bidirectional(true)
            .with_biological_effects(true)
            .with_attention(8);

        assert_eq!(config.input_size, 10);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 3);
        assert!((config.dropout - 0.2).abs() < 1e-6);
        assert!(config.bidirectional);
        assert!(config.use_attention);
        assert_eq!(config.num_attention_heads, 8);
    }

    #[test]
    fn test_quantum_config_validation() {
        let valid = QuantumLSTMConfig::new(10, 64);
        assert!(valid.validate().is_ok());

        let invalid = QuantumLSTMConfig {
            input_size: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_bio_config() {
        let config = BioCognitiveConfig::new(10, 64)
            .with_num_layers(2)
            .with_oscillations(true, false)
            .with_stp(true);

        assert_eq!(config.num_layers, 2);
        assert!(config.enable_gamma);
        assert!(!config.enable_theta);
        assert!(config.enable_stp);
    }
}
