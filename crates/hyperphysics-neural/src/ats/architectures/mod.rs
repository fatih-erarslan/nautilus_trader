//! Neural Architecture Catalog Bridge
//!
//! Re-exports and adapts ats-core's 27+ neural architectures for hyperphysics-neural.
//!
//! ## Available Architectures
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Architecture Categories                       │
//! │                                                                  │
//! │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
//! │  │   Basic    │  │ Sequential │  │   Vision   │  │ Attention  │ │
//! │  │   (MLP)    │  │(LSTM,GRU)  │  │(CNN,ResNet)│  │(Transformer│ │
//! │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
//! │                                                                  │
//! │  1. MLP         7. LSTM       13. CNN-1D      21. Transformer   │
//! │  2. Wide-MLP    8. GRU        14. CNN-2D      22. BERT-style    │
//! │  3. Deep-MLP    9. BiLSTM     15. ResNet      23. GPT-style     │
//! │  4. Sparse-MLP  10. BiGRU     16. DenseNet    24. CLIP-style    │
//! │  5. Highway     11. Stacked   17. MobileNet                      │
//! │  6. Residual    12. Encoder   18. EfficientNet                   │
//! │                               19. VGG                            │
//! │                               20. Inception                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};

// Re-export ats-core architectures
pub use ats_core::ruv_fann_integration::NeuralArchitecture;

/// Architecture category for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchitectureCategory {
    /// Basic feedforward (MLP variants)
    Basic,
    /// Sequential/Recurrent (LSTM, GRU)
    Sequential,
    /// Convolutional (CNN, ResNet)
    Convolutional,
    /// Attention-based (Transformer)
    Attention,
    /// Specialized HFT optimized
    HftOptimized,
}

/// Architecture specification with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSpec {
    /// Architecture name
    pub name: String,
    /// Category
    pub category: ArchitectureCategory,
    /// Typical latency tier (microseconds)
    pub typical_latency_us: u64,
    /// Memory footprint estimate (KB)
    pub memory_kb: u64,
    /// Best for HFT?
    pub hft_optimized: bool,
    /// Description
    pub description: String,
}

/// Pre-defined architecture catalog
pub struct ArchitectureCatalog;

impl ArchitectureCatalog {
    /// Get all available architectures
    pub fn all() -> Vec<ArchitectureSpec> {
        vec![
            // Basic MLPs
            Self::mlp(),
            Self::wide_mlp(),
            Self::deep_mlp(),
            Self::sparse_mlp(),
            Self::highway_net(),
            Self::residual_mlp(),

            // Sequential
            Self::lstm(),
            Self::gru(),
            Self::bilstm(),
            Self::bigru(),
            Self::stacked_lstm(),
            Self::encoder_decoder(),

            // Convolutional
            Self::cnn_1d(),
            Self::cnn_2d(),
            Self::resnet(),
            Self::densenet(),
            Self::mobilenet(),
            Self::efficientnet(),
            Self::vgg(),
            Self::inception(),

            // Attention
            Self::transformer(),
            Self::bert_style(),
            Self::gpt_style(),
            Self::clip_style(),

            // HFT Specialized
            Self::hft_mlp(),
            Self::hft_ensemble(),
            Self::hft_adaptive(),
        ]
    }

    /// Filter by category
    pub fn by_category(category: ArchitectureCategory) -> Vec<ArchitectureSpec> {
        Self::all()
            .into_iter()
            .filter(|a| a.category == category)
            .collect()
    }

    /// Get HFT-optimized architectures
    pub fn hft_optimized() -> Vec<ArchitectureSpec> {
        Self::all()
            .into_iter()
            .filter(|a| a.hft_optimized)
            .collect()
    }

    /// Filter by latency constraint
    pub fn by_latency(max_latency_us: u64) -> Vec<ArchitectureSpec> {
        Self::all()
            .into_iter()
            .filter(|a| a.typical_latency_us <= max_latency_us)
            .collect()
    }

    // Basic MLPs

    fn mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "MLP".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 5,
            memory_kb: 64,
            hft_optimized: true,
            description: "Standard Multi-Layer Perceptron".into(),
        }
    }

    fn wide_mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Wide-MLP".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 10,
            memory_kb: 256,
            hft_optimized: true,
            description: "Wide MLP with larger hidden layers".into(),
        }
    }

    fn deep_mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Deep-MLP".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 15,
            memory_kb: 128,
            hft_optimized: false,
            description: "Deep MLP with many layers".into(),
        }
    }

    fn sparse_mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Sparse-MLP".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 3,
            memory_kb: 32,
            hft_optimized: true,
            description: "Sparse connections for ultra-fast inference".into(),
        }
    }

    fn highway_net() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Highway".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 12,
            memory_kb: 96,
            hft_optimized: false,
            description: "Highway network with gating".into(),
        }
    }

    fn residual_mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Residual-MLP".into(),
            category: ArchitectureCategory::Basic,
            typical_latency_us: 8,
            memory_kb: 80,
            hft_optimized: true,
            description: "MLP with residual connections".into(),
        }
    }

    // Sequential

    fn lstm() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "LSTM".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 50,
            memory_kb: 256,
            hft_optimized: false,
            description: "Long Short-Term Memory network".into(),
        }
    }

    fn gru() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "GRU".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 35,
            memory_kb: 192,
            hft_optimized: false,
            description: "Gated Recurrent Unit".into(),
        }
    }

    fn bilstm() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "BiLSTM".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 100,
            memory_kb: 512,
            hft_optimized: false,
            description: "Bidirectional LSTM".into(),
        }
    }

    fn bigru() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "BiGRU".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 70,
            memory_kb: 384,
            hft_optimized: false,
            description: "Bidirectional GRU".into(),
        }
    }

    fn stacked_lstm() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Stacked-LSTM".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 150,
            memory_kb: 768,
            hft_optimized: false,
            description: "Multi-layer stacked LSTM".into(),
        }
    }

    fn encoder_decoder() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Encoder-Decoder".into(),
            category: ArchitectureCategory::Sequential,
            typical_latency_us: 200,
            memory_kb: 1024,
            hft_optimized: false,
            description: "Sequence-to-sequence encoder-decoder".into(),
        }
    }

    // Convolutional

    fn cnn_1d() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "CNN-1D".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 20,
            memory_kb: 128,
            hft_optimized: true,
            description: "1D Convolutional network for time series".into(),
        }
    }

    fn cnn_2d() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "CNN-2D".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 100,
            memory_kb: 512,
            hft_optimized: false,
            description: "2D Convolutional network".into(),
        }
    }

    fn resnet() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "ResNet".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 200,
            memory_kb: 2048,
            hft_optimized: false,
            description: "Residual Network".into(),
        }
    }

    fn densenet() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "DenseNet".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 250,
            memory_kb: 4096,
            hft_optimized: false,
            description: "Densely Connected Network".into(),
        }
    }

    fn mobilenet() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "MobileNet".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 50,
            memory_kb: 512,
            hft_optimized: false,
            description: "Lightweight mobile-optimized CNN".into(),
        }
    }

    fn efficientnet() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "EfficientNet".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 75,
            memory_kb: 768,
            hft_optimized: false,
            description: "Efficient scaling CNN".into(),
        }
    }

    fn vgg() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "VGG".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 300,
            memory_kb: 8192,
            hft_optimized: false,
            description: "VGG-style deep CNN".into(),
        }
    }

    fn inception() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Inception".into(),
            category: ArchitectureCategory::Convolutional,
            typical_latency_us: 180,
            memory_kb: 3072,
            hft_optimized: false,
            description: "Inception/GoogLeNet architecture".into(),
        }
    }

    // Attention

    fn transformer() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "Transformer".into(),
            category: ArchitectureCategory::Attention,
            typical_latency_us: 500,
            memory_kb: 4096,
            hft_optimized: false,
            description: "Standard Transformer architecture".into(),
        }
    }

    fn bert_style() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "BERT-style".into(),
            category: ArchitectureCategory::Attention,
            typical_latency_us: 1000,
            memory_kb: 16384,
            hft_optimized: false,
            description: "BERT-style bidirectional encoder".into(),
        }
    }

    fn gpt_style() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "GPT-style".into(),
            category: ArchitectureCategory::Attention,
            typical_latency_us: 800,
            memory_kb: 8192,
            hft_optimized: false,
            description: "GPT-style autoregressive decoder".into(),
        }
    }

    fn clip_style() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "CLIP-style".into(),
            category: ArchitectureCategory::Attention,
            typical_latency_us: 1500,
            memory_kb: 32768,
            hft_optimized: false,
            description: "CLIP-style multimodal encoder".into(),
        }
    }

    // HFT Specialized

    fn hft_mlp() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "HFT-MLP".into(),
            category: ArchitectureCategory::HftOptimized,
            typical_latency_us: 2,
            memory_kb: 16,
            hft_optimized: true,
            description: "Ultra-low latency MLP for HFT".into(),
        }
    }

    fn hft_ensemble() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "HFT-Ensemble".into(),
            category: ArchitectureCategory::HftOptimized,
            typical_latency_us: 8,
            memory_kb: 48,
            hft_optimized: true,
            description: "Ensemble of small MLPs for HFT".into(),
        }
    }

    fn hft_adaptive() -> ArchitectureSpec {
        ArchitectureSpec {
            name: "HFT-Adaptive".into(),
            category: ArchitectureCategory::HftOptimized,
            typical_latency_us: 5,
            memory_kb: 32,
            hft_optimized: true,
            description: "Adaptive depth MLP for HFT".into(),
        }
    }
}

/// Create FANN network from architecture spec
pub fn create_fann_from_spec(
    spec: &ArchitectureSpec,
    input_dim: usize,
    output_dim: usize,
) -> Result<crate::fann::FannNetwork, crate::fann::FannError> {
    use crate::fann::{FannConfig, FannNetwork};

    // Choose hidden dimensions based on architecture
    let hidden_dims = match spec.category {
        ArchitectureCategory::Basic | ArchitectureCategory::HftOptimized => {
            if spec.name.contains("Wide") {
                vec![256, 128]
            } else if spec.name.contains("Deep") {
                vec![64, 64, 64, 32]
            } else if spec.name.contains("Sparse") {
                vec![32]
            } else {
                vec![64, 32]
            }
        }
        ArchitectureCategory::Sequential => vec![128, 64],
        ArchitectureCategory::Convolutional => vec![128, 64, 32],
        ArchitectureCategory::Attention => vec![256, 128, 64],
    };

    let config = if spec.hft_optimized {
        FannConfig::hft(input_dim, &hidden_dims, output_dim)
    } else {
        FannConfig::regression(input_dim, &hidden_dims, output_dim)
    };

    FannNetwork::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_all() {
        let all = ArchitectureCatalog::all();
        assert!(all.len() >= 27);
    }

    #[test]
    fn test_catalog_by_category() {
        let basic = ArchitectureCatalog::by_category(ArchitectureCategory::Basic);
        assert!(!basic.is_empty());
        assert!(basic.iter().all(|a| a.category == ArchitectureCategory::Basic));
    }

    #[test]
    fn test_catalog_hft_optimized() {
        let hft = ArchitectureCatalog::hft_optimized();
        assert!(!hft.is_empty());
        assert!(hft.iter().all(|a| a.hft_optimized));
    }

    #[test]
    fn test_catalog_by_latency() {
        let fast = ArchitectureCatalog::by_latency(20);
        assert!(!fast.is_empty());
        assert!(fast.iter().all(|a| a.typical_latency_us <= 20));
    }

    #[test]
    fn test_create_fann_from_spec() {
        let spec = ArchitectureCatalog::hft_optimized()[0].clone();
        let network = create_fann_from_spec(&spec, 10, 3).unwrap();

        assert_eq!(network.input_dim(), 10);
        assert_eq!(network.output_dim(), 3);
    }

    #[test]
    fn test_architecture_categories() {
        let all = ArchitectureCatalog::all();

        let mut has_basic = false;
        let mut has_sequential = false;
        let mut has_conv = false;
        let mut has_attention = false;
        let mut has_hft = false;

        for arch in all {
            match arch.category {
                ArchitectureCategory::Basic => has_basic = true,
                ArchitectureCategory::Sequential => has_sequential = true,
                ArchitectureCategory::Convolutional => has_conv = true,
                ArchitectureCategory::Attention => has_attention = true,
                ArchitectureCategory::HftOptimized => has_hft = true,
            }
        }

        assert!(has_basic);
        assert!(has_sequential);
        assert!(has_conv);
        assert!(has_attention);
        assert!(has_hft);
    }
}
