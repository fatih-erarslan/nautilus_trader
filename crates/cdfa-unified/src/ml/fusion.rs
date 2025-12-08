//! TorchScript Fusion Integration
//!
//! This module provides integration with the TorchScript fusion system for
//! hardware-accelerated signal processing and neural network inference.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use candle_core::{Device, Tensor};
use crate::ml::{MLError, MLResult, MLModel, MLFramework, MLTask, ModelMetadata};

/// TorchScript fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Device for computation
    pub device: String,
    /// Fusion type
    pub fusion_type: FusionType,
    /// Minimum weight threshold
    pub min_weight: f32,
    /// Confidence factor
    pub confidence_factor: f32,
    /// Diversity factor
    pub diversity_factor: f32,
}

/// Fusion types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionType {
    /// Score-based fusion
    Score,
    /// Rank-based fusion
    Rank,
    /// Hybrid fusion
    Hybrid,
    /// Weighted fusion
    Weighted,
    /// Layered fusion
    Layered,
    /// Adaptive fusion
    Adaptive,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            fusion_type: FusionType::Hybrid,
            min_weight: 0.01,
            confidence_factor: 0.7,
            diversity_factor: 0.3,
        }
    }
}

/// Fusion result
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Fused signal
    pub fused_signal: Array1<f32>,
    /// Confidence scores
    pub confidence: Array1<f32>,
    /// Weights used in fusion
    pub weights: Vec<Array1<f32>>,
    /// Fusion metadata
    pub metadata: std::collections::HashMap<String, f64>,
}

/// TorchScript fusion model
pub struct TorchScriptFusion {
    /// Configuration
    config: FusionConfig,
    /// Device for computation
    device: Device,
    /// Model metadata
    metadata: ModelMetadata,
    /// Training state
    is_trained: bool,
}

impl TorchScriptFusion {
    /// Create new TorchScript fusion model
    pub fn new(config: FusionConfig) -> MLResult<Self> {
        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).map_err(|e| MLError::HardwareError {
                message: format!("CUDA not available: {}", e),
            })?,
            "metal" => Device::new_metal(0).map_err(|e| MLError::HardwareError {
                message: format!("Metal not available: {}", e),
            })?,
            _ => Device::Cpu,
        };

        let metadata = ModelMetadata::new(
            format!("torchscript-fusion-{}", uuid::Uuid::new_v4()),
            "TorchScript Fusion".to_string(),
            MLFramework::TorchScript,
            MLTask::SignalFusion,
        );

        Ok(Self {
            config,
            device,
            metadata,
            is_trained: true, // Fusion doesn't require traditional training
        })
    }

    /// Fuse signals using the configured method
    pub fn fuse_signals(
        &self,
        signals: &Array2<f32>,
        confidences: &Array2<f32>,
        fusion_type: FusionType,
    ) -> MLResult<FusionResult> {
        if signals.shape() != confidences.shape() {
            return Err(MLError::DimensionMismatch {
                expected: format!("{:?}", signals.shape()),
                actual: format!("{:?}", confidences.shape()),
            });
        }

        if signals.is_empty() {
            return Err(MLError::InferenceError {
                message: "Empty input signals".to_string(),
            });
        }

        match fusion_type {
            FusionType::Score => self.score_fusion(signals, confidences),
            FusionType::Rank => self.rank_fusion(signals, confidences),
            FusionType::Hybrid => self.hybrid_fusion(signals, confidences),
            FusionType::Weighted => self.weighted_fusion(signals, confidences),
            FusionType::Layered => self.layered_fusion(signals, confidences),
            FusionType::Adaptive => self.adaptive_fusion(signals, confidences),
        }
    }

    /// Score-based fusion
    fn score_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let (n_signals, n_timepoints) = signals.dim();
        let mut fused_signal = Array1::zeros(n_timepoints);
        let mut confidence = Array1::zeros(n_timepoints);
        let mut weights = Vec::new();

        for t in 0..n_timepoints {
            let mut signal_weights = Array1::zeros(n_signals);
            let mut total_weight = 0.0;
            let mut weighted_sum = 0.0;
            let mut confidence_sum = 0.0;

            for i in 0..n_signals {
                let conf = confidences[[i, t]];
                let weight = conf.max(self.config.min_weight);
                signal_weights[i] = weight;
                total_weight += weight;
                weighted_sum += weight * signals[[i, t]];
                confidence_sum += conf;
            }

            if total_weight > 0.0 {
                signal_weights /= total_weight;
                fused_signal[t] = weighted_sum / total_weight;
                confidence[t] = confidence_sum / n_signals as f32;
            }

            weights.push(signal_weights);
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("fusion_type".to_string(), 0.0); // Score = 0
        metadata.insert("n_signals".to_string(), n_signals as f64);
        metadata.insert("n_timepoints".to_string(), n_timepoints as f64);

        Ok(FusionResult {
            fused_signal,
            confidence,
            weights,
            metadata,
        })
    }

    /// Rank-based fusion
    fn rank_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let (n_signals, n_timepoints) = signals.dim();
        let mut fused_signal = Array1::zeros(n_timepoints);
        let mut confidence = Array1::zeros(n_timepoints);
        let mut weights = Vec::new();

        for t in 0..n_timepoints {
            // Get signal values and their indices for ranking
            let mut signal_pairs: Vec<(f32, usize)> = signals.column(t)
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();

            // Sort by signal value
            signal_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut signal_weights = Array1::zeros(n_signals);
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (rank, &(signal_val, signal_idx)) in signal_pairs.iter().enumerate() {
                let rank_weight = (n_signals - rank) as f32 / n_signals as f32;
                let conf_weight = confidences[[signal_idx, t]];
                let combined_weight = rank_weight * conf_weight;

                signal_weights[signal_idx] = combined_weight;
                weighted_sum += combined_weight * signal_val;
                total_weight += combined_weight;
            }

            if total_weight > 0.0 {
                signal_weights /= total_weight;
                fused_signal[t] = weighted_sum / total_weight;
                confidence[t] = confidences.column(t).mean().unwrap_or(0.0);
            }

            weights.push(signal_weights);
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("fusion_type".to_string(), 1.0); // Rank = 1
        metadata.insert("n_signals".to_string(), n_signals as f64);
        metadata.insert("n_timepoints".to_string(), n_timepoints as f64);

        Ok(FusionResult {
            fused_signal,
            confidence,
            weights,
            metadata,
        })
    }

    /// Hybrid fusion (combination of score and rank)
    fn hybrid_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let score_result = self.score_fusion(signals, confidences)?;
        let rank_result = self.rank_fusion(signals, confidences)?;

        let alpha = 0.5; // Mixing parameter
        let fused_signal = &score_result.fused_signal * alpha + &rank_result.fused_signal * (1.0 - alpha);
        let confidence = &score_result.confidence * alpha + &rank_result.confidence * (1.0 - alpha);

        // Combine weights
        let mut weights = Vec::new();
        for (score_w, rank_w) in score_result.weights.iter().zip(rank_result.weights.iter()) {
            let combined_w = score_w * alpha + rank_w * (1.0 - alpha);
            weights.push(combined_w);
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("fusion_type".to_string(), 2.0); // Hybrid = 2
        metadata.insert("alpha".to_string(), alpha as f64);
        metadata.insert("n_signals".to_string(), signals.nrows() as f64);
        metadata.insert("n_timepoints".to_string(), signals.ncols() as f64);

        Ok(FusionResult {
            fused_signal,
            confidence,
            weights,
            metadata,
        })
    }

    /// Weighted fusion with diversity consideration
    fn weighted_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let (n_signals, n_timepoints) = signals.dim();
        let mut fused_signal = Array1::zeros(n_timepoints);
        let mut confidence = Array1::zeros(n_timepoints);
        let mut weights = Vec::new();

        // Compute diversity matrix
        let diversity_matrix = self.compute_diversity_matrix(signals);

        for t in 0..n_timepoints {
            let mut signal_weights = Array1::zeros(n_signals);
            let mut total_weight = 0.0;
            let mut weighted_sum = 0.0;

            for i in 0..n_signals {
                let conf_weight = confidences[[i, t]];
                let diversity_weight = diversity_matrix[[i, i]]; // Diagonal represents self-diversity
                
                let combined_weight = self.config.confidence_factor * conf_weight +
                                    self.config.diversity_factor * diversity_weight;
                
                signal_weights[i] = combined_weight.max(self.config.min_weight);
                total_weight += signal_weights[i];
                weighted_sum += signal_weights[i] * signals[[i, t]];
            }

            if total_weight > 0.0 {
                signal_weights /= total_weight;
                fused_signal[t] = weighted_sum / total_weight;
                confidence[t] = confidences.column(t).mean().unwrap_or(0.0);
            }

            weights.push(signal_weights);
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("fusion_type".to_string(), 3.0); // Weighted = 3
        metadata.insert("confidence_factor".to_string(), self.config.confidence_factor as f64);
        metadata.insert("diversity_factor".to_string(), self.config.diversity_factor as f64);
        metadata.insert("n_signals".to_string(), n_signals as f64);
        metadata.insert("n_timepoints".to_string(), n_timepoints as f64);

        Ok(FusionResult {
            fused_signal,
            confidence,
            weights,
            metadata,
        })
    }

    /// Layered fusion with hierarchical structure
    fn layered_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let (n_signals, n_timepoints) = signals.dim();
        
        // For layered fusion, group signals and fuse hierarchically
        let group_size = 2; // Pair signals
        let mut intermediate_signals = Vec::new();
        let mut intermediate_confidences = Vec::new();

        // First layer: pairwise fusion
        for i in (0..n_signals).step_by(group_size) {
            let end_idx = (i + group_size).min(n_signals);
            let group_signals = signals.slice(s![i..end_idx, ..]).to_owned();
            let group_confidences = confidences.slice(s![i..end_idx, ..]).to_owned();

            let group_result = self.score_fusion(&group_signals, &group_confidences)?;
            intermediate_signals.push(group_result.fused_signal);
            intermediate_confidences.push(group_result.confidence);
        }

        // Convert intermediate results to arrays
        let n_intermediate = intermediate_signals.len();
        let mut intermediate_signal_array = Array2::zeros((n_intermediate, n_timepoints));
        let mut intermediate_confidence_array = Array2::zeros((n_intermediate, n_timepoints));

        for (i, (sig, conf)) in intermediate_signals.iter().zip(intermediate_confidences.iter()).enumerate() {
            intermediate_signal_array.row_mut(i).assign(sig);
            intermediate_confidence_array.row_mut(i).assign(conf);
        }

        // Final layer: fuse intermediate results
        let final_result = self.score_fusion(&intermediate_signal_array, &intermediate_confidence_array)?;

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("fusion_type".to_string(), 4.0); // Layered = 4
        metadata.insert("group_size".to_string(), group_size as f64);
        metadata.insert("n_layers".to_string(), 2.0);
        metadata.insert("n_signals".to_string(), n_signals as f64);
        metadata.insert("n_timepoints".to_string(), n_timepoints as f64);

        Ok(FusionResult {
            fused_signal: final_result.fused_signal,
            confidence: final_result.confidence,
            weights: final_result.weights,
            metadata,
        })
    }

    /// Adaptive fusion that selects method based on signal properties
    fn adaptive_fusion(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> MLResult<FusionResult> {
        let (n_signals, n_timepoints) = signals.dim();
        
        // Analyze signal properties to choose fusion method
        let signal_variance = signals.var_axis(Axis(0), 0.0).mean().unwrap_or(0.0);
        let confidence_variance = confidences.var_axis(Axis(0), 0.0).mean().unwrap_or(0.0);
        let signal_correlation = self.compute_average_correlation(signals);

        // Decision logic for adaptive method selection
        let chosen_method = if signal_variance > 0.5 && confidence_variance > 0.1 {
            FusionType::Weighted // High variance - use diversity-aware weighting
        } else if signal_correlation > 0.8 {
            FusionType::Rank // High correlation - use rank-based fusion
        } else if n_signals > 5 {
            FusionType::Layered // Many signals - use hierarchical approach
        } else {
            FusionType::Hybrid // Default to hybrid
        };

        let mut result = match chosen_method {
            FusionType::Weighted => self.weighted_fusion(signals, confidences)?,
            FusionType::Rank => self.rank_fusion(signals, confidences)?,
            FusionType::Layered => self.layered_fusion(signals, confidences)?,
            _ => self.hybrid_fusion(signals, confidences)?,
        };

        // Update metadata
        result.metadata.insert("fusion_type".to_string(), 5.0); // Adaptive = 5
        result.metadata.insert("chosen_method".to_string(), chosen_method as u8 as f64);
        result.metadata.insert("signal_variance".to_string(), signal_variance as f64);
        result.metadata.insert("confidence_variance".to_string(), confidence_variance as f64);
        result.metadata.insert("signal_correlation".to_string(), signal_correlation as f64);

        Ok(result)
    }

    /// Compute diversity matrix for signals
    fn compute_diversity_matrix(&self, signals: &Array2<f32>) -> Array2<f32> {
        let n_signals = signals.nrows();
        let mut diversity = Array2::zeros((n_signals, n_signals));

        for i in 0..n_signals {
            for j in 0..n_signals {
                if i == j {
                    diversity[[i, j]] = 1.0; // Self-diversity
                } else {
                    // Compute correlation and convert to diversity (1 - correlation)
                    let corr = self.compute_correlation(&signals.row(i), &signals.row(j));
                    diversity[[i, j]] = (1.0 - corr.abs()).max(0.0);
                }
            }
        }

        diversity
    }

    /// Compute correlation between two signals
    fn compute_correlation(&self, signal1: &ndarray::ArrayView1<f32>, signal2: &ndarray::ArrayView1<f32>) -> f32 {
        let mean1 = signal1.mean().unwrap_or(0.0);
        let mean2 = signal2.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for (&x1, &x2) in signal1.iter().zip(signal2.iter()) {
            let diff1 = x1 - mean1;
            let diff2 = x2 - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute average correlation among all signals
    fn compute_average_correlation(&self, signals: &Array2<f32>) -> f32 {
        let n_signals = signals.nrows();
        if n_signals < 2 {
            return 0.0;
        }

        let mut total_correlation = 0.0;
        let mut count = 0;

        for i in 0..n_signals {
            for j in (i + 1)..n_signals {
                let corr = self.compute_correlation(&signals.row(i), &signals.row(j));
                total_correlation += corr.abs();
                count += 1;
            }
        }

        if count > 0 {
            total_correlation / count as f32
        } else {
            0.0
        }
    }
}

impl MLModel for TorchScriptFusion {
    type Input = Array2<f32>;
    type Output = Array2<f32>;
    type Config = FusionConfig;

    fn new(config: Self::Config) -> MLResult<Self> {
        Self::new(config)
    }

    fn fit(&mut self, _x: &Self::Input, _y: &Self::Output) -> MLResult<()> {
        // TorchScript fusion doesn't require traditional training
        self.is_trained = true;
        self.metadata.touch();
        Ok(())
    }

    fn predict(&self, x: &Self::Input) -> MLResult<Self::Output> {
        // For prediction, assume x contains both signals and confidences
        // Split x into signals and confidences (assuming equal split)
        let total_signals = x.nrows();
        if total_signals % 2 != 0 {
            return Err(MLError::DimensionMismatch {
                expected: "Even number of rows (signals + confidences)".to_string(),
                actual: format!("{} rows", total_signals),
            });
        }

        let n_signals = total_signals / 2;
        let signals = x.slice(s![0..n_signals, ..]).to_owned();
        let confidences = x.slice(s![n_signals.., ..]).to_owned();

        let result = self.fuse_signals(&signals, &confidences, self.config.fusion_type)?;
        
        // Return fused signal as a single row
        let fused_2d = result.fused_signal.insert_axis(Axis(0));
        Ok(fused_2d)
    }

    fn evaluate(&self, x: &Self::Input, y: &Self::Output) -> MLResult<f64> {
        let predictions = self.predict(x)?;
        
        // Compute correlation with target as quality metric
        if predictions.nrows() != y.nrows() || predictions.ncols() != y.ncols() {
            return Err(MLError::DimensionMismatch {
                expected: format!("{:?}", y.shape()),
                actual: format!("{:?}", predictions.shape()),
            });
        }

        let corr = self.compute_correlation(&predictions.row(0), &y.row(0));
        Ok(corr as f64)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }

    fn to_bytes(&self) -> MLResult<Vec<u8>> {
        let model_data = (&self.config, self.is_trained);
        Ok(bincode::serialize(&model_data)?)
    }

    fn from_bytes(bytes: &[u8]) -> MLResult<Self> {
        let (config, is_trained): (FusionConfig, bool) = bincode::deserialize(bytes)?;
        let mut model = Self::new(config)?;
        model.is_trained = is_trained;
        Ok(model)
    }

    fn framework(&self) -> MLFramework {
        MLFramework::TorchScript
    }

    fn task(&self) -> MLTask {
        MLTask::SignalFusion
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    fn parameter_count(&self) -> usize {
        // Fusion models have minimal parameters (just configuration)
        std::mem::size_of::<FusionConfig>()
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig {
            device: "cpu".to_string(),
            fusion_type: FusionType::Hybrid,
            min_weight: 0.01,
            confidence_factor: 0.7,
            diversity_factor: 0.3,
        };

        assert_eq!(config.device, "cpu");
        assert_eq!(config.fusion_type, FusionType::Hybrid);
        assert_eq!(config.min_weight, 0.01);
    }

    #[test]
    fn test_torchscript_fusion_creation() {
        let config = FusionConfig::default();
        let model = TorchScriptFusion::new(config);
        
        assert!(model.is_ok());
        let model = model.unwrap();
        assert!(model.is_trained());
        assert_eq!(model.framework(), MLFramework::TorchScript);
        assert_eq!(model.task(), MLTask::SignalFusion);
    }

    #[test]
    fn test_score_fusion() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        let signals = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, // Signal 1
                2.0, 3.0, 4.0, 5.0, 6.0, // Signal 2
                0.5, 1.5, 2.5, 3.5, 4.5, // Signal 3
            ],
        ).unwrap();

        let confidences = Array2::from_shape_vec(
            (3, 5),
            vec![
                0.9, 0.8, 0.7, 0.6, 0.5, // Confidence 1
                0.8, 0.9, 0.8, 0.7, 0.6, // Confidence 2
                0.6, 0.7, 0.8, 0.9, 1.0, // Confidence 3
            ],
        ).unwrap();

        let result = fusion.fuse_signals(&signals, &confidences, FusionType::Score).unwrap();

        assert_eq!(result.fused_signal.len(), 5);
        assert_eq!(result.confidence.len(), 5);
        assert_eq!(result.weights.len(), 5);

        // Check that weights sum to 1 at each time point
        for weights_t in &result.weights {
            let weight_sum: f32 = weights_t.sum();
            assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rank_fusion() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        let signals = Array2::from_shape_vec(
            (3, 4),
            vec![
                3.0, 1.0, 4.0, 2.0, // Signal 1
                1.0, 3.0, 2.0, 4.0, // Signal 2
                2.0, 4.0, 1.0, 3.0, // Signal 3
            ],
        ).unwrap();

        let confidences = Array2::from_shape_vec(
            (3, 4),
            vec![0.8; 12], // Equal confidence
        ).unwrap();

        let result = fusion.fuse_signals(&signals, &confidences, FusionType::Rank).unwrap();

        assert_eq!(result.fused_signal.len(), 4);
        assert_eq!(result.confidence.len(), 4);
        assert_eq!(result.weights.len(), 4);
        assert_eq!(result.metadata.get("fusion_type"), Some(&1.0));
    }

    #[test]
    fn test_hybrid_fusion() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        let signals = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 2.0, 3.0, // Signal 1
                2.0, 3.0, 1.0, // Signal 2
            ],
        ).unwrap();

        let confidences = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.9, 0.8, 0.7, // Confidence 1
                0.7, 0.8, 0.9, // Confidence 2
            ],
        ).unwrap();

        let result = fusion.fuse_signals(&signals, &confidences, FusionType::Hybrid).unwrap();

        assert_eq!(result.fused_signal.len(), 3);
        assert_eq!(result.metadata.get("fusion_type"), Some(&2.0));
        assert!(result.metadata.contains_key("alpha"));
    }

    #[test]
    fn test_adaptive_fusion() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        let signals = Array2::from_shape_vec(
            (4, 10),
            (0..40).map(|i| i as f32 / 10.0).collect(),
        ).unwrap();

        let confidences = Array2::from_shape_vec(
            (4, 10),
            vec![0.8; 40],
        ).unwrap();

        let result = fusion.fuse_signals(&signals, &confidences, FusionType::Adaptive).unwrap();

        assert_eq!(result.fused_signal.len(), 10);
        assert_eq!(result.metadata.get("fusion_type"), Some(&5.0));
        assert!(result.metadata.contains_key("chosen_method"));
        assert!(result.metadata.contains_key("signal_variance"));
        assert!(result.metadata.contains_key("signal_correlation"));
    }

    #[test]
    fn test_correlation_computation() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        let signal1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let signal2 = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let signal3 = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let corr12 = fusion.compute_correlation(&signal1.view(), &signal2.view());
        let corr13 = fusion.compute_correlation(&signal1.view(), &signal3.view());

        assert_abs_diff_eq!(corr12, 1.0, epsilon = 1e-6); // Perfect positive correlation
        assert_abs_diff_eq!(corr13, -1.0, epsilon = 1e-6); // Perfect negative correlation
    }

    #[test]
    fn test_error_handling() {
        let config = FusionConfig::default();
        let fusion = TorchScriptFusion::new(config).unwrap();

        // Test dimension mismatch
        let signals = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let confidences = Array2::from_shape_vec((3, 3), vec![0.8; 9]).unwrap();

        let result = fusion.fuse_signals(&signals, &confidences, FusionType::Score);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MLError::DimensionMismatch { .. }));

        // Test empty input
        let empty_signals = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let empty_confidences = Array2::from_shape_vec((0, 0), vec![]).unwrap();

        let result = fusion.fuse_signals(&empty_signals, &empty_confidences, FusionType::Score);
        assert!(result.is_err());
    }
}