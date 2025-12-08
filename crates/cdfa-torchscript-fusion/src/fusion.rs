//! Core TorchScript Fusion implementation
//!
//! This module provides the main TorchScriptFusion struct and implementations
//! for all six fusion types with hardware acceleration support.

use crate::{
    device::{DeviceManager, DeviceType},
    error::{FusionError, Result, ResultExt},
    types::{FusionType, FusionResult, FusionParams, FusionMetadata, PerformanceMetrics},
};
use candle_core::{Device, Tensor, DType};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::time::Instant;

/// Main TorchScript Fusion implementation
#[derive(Debug)]
pub struct TorchScriptFusion {
    /// Device manager for hardware acceleration
    device_manager: DeviceManager,
    /// Compiled models cache
    model_cache: HashMap<FusionType, CompiledModel>,
    /// Performance statistics
    performance_stats: PerformanceStats,
}

/// Compiled model for a specific fusion type
#[derive(Debug, Clone)]
struct CompiledModel {
    /// Fusion type
    fusion_type: FusionType,
    /// Compilation timestamp
    compiled_at: std::time::SystemTime,
    /// Model parameters
    params: FusionParams,
    /// Device used for compilation
    device: Device,
}

/// Performance statistics tracker
#[derive(Debug, Default)]
struct PerformanceStats {
    /// Total operations performed
    total_operations: u64,
    /// Total inference time in microseconds
    total_inference_time_us: u64,
    /// Total compilation time in microseconds
    total_compilation_time_us: u64,
    /// Memory usage peaks
    peak_memory_usage: u64,
}

impl TorchScriptFusion {
    /// Create a new TorchScript Fusion instance with automatic device detection
    pub async fn new() -> Result<Self> {
        let device_manager = DeviceManager::new()?;
        
        log::info!(
            "TorchScript Fusion initialized with device: {:?}",
            device_manager.primary_device()
        );

        Ok(Self {
            device_manager,
            model_cache: HashMap::new(),
            performance_stats: PerformanceStats::default(),
        })
    }

    /// Create a new instance with a specific device
    pub async fn with_device(device: Device) -> Result<Self> {
        let device_manager = DeviceManager::with_device(device)?;
        
        log::info!(
            "TorchScript Fusion initialized with specified device: {:?}",
            device_manager.primary_device()
        );

        Ok(Self {
            device_manager,
            model_cache: HashMap::new(),
            performance_stats: PerformanceStats::default(),
        })
    }

    /// Fuse multiple signals using the specified fusion type
    pub async fn fuse_signals(
        &mut self,
        signals: &Array2<f32>,
        confidences: &Array2<f32>,
        fusion_type: FusionType,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        let start_time = Instant::now();

        // Validate inputs
        self.validate_inputs(signals, confidences)?;

        // Ensure model is compiled
        self.ensure_model_compiled(fusion_type, params).await?;

        // Convert to tensors on device
        let signals_tensor = self.array_to_tensor(signals)?;
        let confidences_tensor = self.array_to_tensor(confidences)?;

        // Perform fusion based on type
        let result = match fusion_type {
            FusionType::Score => {
                self.score_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
            FusionType::Rank => {
                self.rank_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
            FusionType::Hybrid => {
                self.hybrid_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
            FusionType::Weighted => {
                self.weighted_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
            FusionType::Layered => {
                self.layered_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
            FusionType::Adaptive => {
                self.adaptive_fusion(&signals_tensor, &confidences_tensor, params).await?
            }
        };

        // Update performance statistics
        let inference_time = start_time.elapsed().as_micros() as u64;
        self.performance_stats.total_operations += 1;
        self.performance_stats.total_inference_time_us += inference_time;

        // Add metadata
        let mut metadata = FusionMetadata::default();
        metadata.set_timestamp();
        metadata.inference_time_us = Some(inference_time);
        metadata.device = Some(format!("{:?}", self.device_manager.primary_device()));
        metadata.num_signals = Some(signals.nrows());
        metadata.sequence_length = Some(signals.ncols());

        // Create final result
        let mut final_result = result;
        final_result.metadata = metadata;

        // Validate result
        final_result.validate()?;

        Ok(final_result)
    }

    /// Score-based fusion: Confidence-weighted linear combination
    async fn score_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        let device = self.device_manager.primary_device();

        // Normalize confidences to weights
        let conf_sum = confidences.sum_keepdim(0)?;
        let weights = confidences.broadcast_div(&conf_sum)?;

        // Apply minimum weight threshold
        let min_weight_tensor = Tensor::full(params.min_weight, weights.shape(), device)?;
        let weights = weights.maximum(&min_weight_tensor)?;

        // Re-normalize after minimum weight application
        let weight_sum = weights.sum_keepdim(0)?;
        let weights = weights.broadcast_div(&weight_sum)?;

        // Compute weighted sum
        let weighted_signals = signals.broadcast_mul(&weights)?;
        let fused_signal = weighted_signals.sum_keepdim(0)?;

        // Compute confidence as weighted average of input confidences
        let confidence = confidences.broadcast_mul(&weights)?.sum_keepdim(0)?;

        // Convert back to ndarray
        let fused_array = self.tensor_to_array1(&fused_signal.squeeze(0)?)?;
        let confidence_array = self.tensor_to_array1(&confidence.squeeze(0)?)?;
        let weights_arrays = self.tensor_to_weight_arrays(&weights)?;

        Ok(FusionResult::new(
            fused_array,
            confidence_array,
            weights_arrays,
            FusionType::Score,
        ))
    }

    /// Rank-based fusion: Ordering-based combination
    async fn rank_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        // Convert to CPU for ranking operations (GPU ranking is complex)
        let signals_cpu = signals.to_device(&Device::Cpu)?;
        let confidences_cpu = confidences.to_device(&Device::Cpu)?;

        let signals_array = self.tensor_to_array2(&signals_cpu)?;
        let confidences_array = self.tensor_to_array2(&confidences_cpu)?;

        let (n_signals, seq_len) = signals_array.dim();
        let mut ranks = Array2::zeros((n_signals, seq_len));

        // Compute ranks for each time step
        for t in 0..seq_len {
            let mut values: Vec<(f32, usize)> = signals_array
                .column(t)
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();

            // Sort by value (descending)
            values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Assign ranks
            for (rank, &(_, signal_idx)) in values.iter().enumerate() {
                ranks[(signal_idx, t)] = (n_signals - rank) as f32;
            }
        }

        // Normalize ranks to weights based on confidence
        let mut weights = Array2::zeros((n_signals, seq_len));
        for t in 0..seq_len {
            let conf_col = confidences_array.column(t);
            let rank_col = ranks.column(t);
            
            // Weight ranks by confidence
            let weighted_ranks: Array1<f32> = rank_col.iter()
                .zip(conf_col.iter())
                .map(|(&r, &c)| r * c)
                .collect();
            
            let sum = weighted_ranks.sum();
            if sum > 0.0 {
                for (i, &wr) in weighted_ranks.iter().enumerate() {
                    weights[(i, t)] = (wr / sum).max(params.min_weight);
                }
                
                // Re-normalize
                let row_sum: f32 = weights.column(t).sum();
                if row_sum > 0.0 {
                    for i in 0..n_signals {
                        weights[(i, t)] /= row_sum;
                    }
                }
            } else {
                // Equal weights fallback
                let equal_weight = 1.0 / n_signals as f32;
                for i in 0..n_signals {
                    weights[(i, t)] = equal_weight;
                }
            }
        }

        // Compute fused signal
        let mut fused_signal = Array1::zeros(seq_len);
        let mut confidence = Array1::zeros(seq_len);

        for t in 0..seq_len {
            for i in 0..n_signals {
                let weight = weights[(i, t)];
                fused_signal[t] += weight * signals_array[(i, t)];
                confidence[t] += weight * confidences_array[(i, t)];
            }
        }

        let weights_arrays: Vec<Array1<f32>> = (0..n_signals)
            .map(|i| weights.row(i).to_owned())
            .collect();

        Ok(FusionResult::new(
            fused_signal,
            confidence,
            weights_arrays,
            FusionType::Rank,
        ))
    }

    /// Hybrid fusion: Adaptive combination of score and rank methods
    async fn hybrid_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        // Get score-based result
        let score_result = self.score_fusion(signals, confidences, params).await?;
        
        // Get rank-based result
        let rank_result = self.rank_fusion(signals, confidences, params).await?;

        // Combine using score_alpha parameter
        let alpha = params.score_alpha;
        let beta = 1.0 - alpha;

        let fused_signal = &score_result.fused_signal * alpha + &rank_result.fused_signal * beta;
        let confidence = &score_result.confidence * alpha + &rank_result.confidence * beta;

        // Combine weights
        let n_signals = score_result.weights.len();
        let seq_len = score_result.fused_signal.len();
        let mut combined_weights = Vec::with_capacity(n_signals);

        for i in 0..n_signals {
            let combined_weight = &score_result.weights[i] * alpha + &rank_result.weights[i] * beta;
            combined_weights.push(combined_weight);
        }

        Ok(FusionResult::new(
            fused_signal,
            confidence,
            combined_weights,
            FusionType::Hybrid,
        ))
    }

    /// Weighted fusion: Diversity-aware weighted combination
    async fn weighted_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        // Convert to CPU for diversity calculations
        let signals_cpu = signals.to_device(&Device::Cpu)?;
        let confidences_cpu = confidences.to_device(&Device::Cpu)?;

        let signals_array = self.tensor_to_array2(&signals_cpu)?;
        let confidences_array = self.tensor_to_array2(&confidences_cpu)?;

        let (n_signals, seq_len) = signals_array.dim();

        // Calculate pairwise correlations for diversity assessment
        let mut diversity_matrix = Array2::zeros((n_signals, n_signals));
        for i in 0..n_signals {
            for j in i..n_signals {
                let corr = if i == j {
                    1.0
                } else {
                    self.calculate_correlation(
                        &signals_array.row(i).to_owned(),
                        &signals_array.row(j).to_owned(),
                    )?
                };
                diversity_matrix[(i, j)] = corr;
                diversity_matrix[(j, i)] = corr;
            }
        }

        // Calculate diversity scores (1 - average correlation with others)
        let mut diversity_scores = Array1::zeros(n_signals);
        for i in 0..n_signals {
            let avg_corr: f32 = diversity_matrix.row(i).iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &corr)| corr.abs())
                .sum::<f32>() / (n_signals - 1) as f32;
            diversity_scores[i] = 1.0 - avg_corr;
        }

        // Combine confidence and diversity
        let mut weights = Array2::zeros((n_signals, seq_len));
        for t in 0..seq_len {
            for i in 0..n_signals {
                let conf = confidences_array[(i, t)];
                let div = diversity_scores[i];
                weights[(i, t)] = params.confidence_factor * conf + params.diversity_factor * div;
            }

            // Normalize and apply minimum weight
            let col_sum: f32 = weights.column(t).sum();
            if col_sum > 0.0 {
                for i in 0..n_signals {
                    weights[(i, t)] = (weights[(i, t)] / col_sum).max(params.min_weight);
                }
                
                // Re-normalize after minimum weight
                let new_sum: f32 = weights.column(t).sum();
                for i in 0..n_signals {
                    weights[(i, t)] /= new_sum;
                }
            } else {
                let equal_weight = 1.0 / n_signals as f32;
                for i in 0..n_signals {
                    weights[(i, t)] = equal_weight;
                }
            }
        }

        // Apply nonlinear weighting if enabled
        if params.use_nonlinear_weighting {
            for t in 0..seq_len {
                for i in 0..n_signals {
                    weights[(i, t)] = weights[(i, t)].powf(params.nonlinear_exponent);
                }
                
                // Re-normalize
                let col_sum: f32 = weights.column(t).sum();
                if col_sum > 0.0 {
                    for i in 0..n_signals {
                        weights[(i, t)] /= col_sum;
                    }
                }
            }
        }

        // Compute fused signal
        let mut fused_signal = Array1::zeros(seq_len);
        let mut confidence = Array1::zeros(seq_len);

        for t in 0..seq_len {
            for i in 0..n_signals {
                let weight = weights[(i, t)];
                fused_signal[t] += weight * signals_array[(i, t)];
                confidence[t] += weight * confidences_array[(i, t)];
            }
        }

        let weights_arrays: Vec<Array1<f32>> = (0..n_signals)
            .map(|i| weights.row(i).to_owned())
            .collect();

        Ok(FusionResult::new(
            fused_signal,
            confidence,
            weights_arrays,
            FusionType::Weighted,
        ))
    }

    /// Layered fusion: Hierarchical fusion with sub-grouping
    async fn layered_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        let signals_array = self.tensor_to_array2(&signals.to_device(&Device::Cpu)?)?;
        let confidences_array = self.tensor_to_array2(&confidences.to_device(&Device::Cpu)?)?;

        let (n_signals, seq_len) = signals_array.dim();

        // Group signals into layers based on similarity
        let groups = self.create_signal_groups(&signals_array, 3)?; // Up to 3 groups

        let mut group_results = Vec::new();
        let mut group_weights = Vec::new();

        // Fuse within each group
        for group in &groups {
            if group.is_empty() {
                continue;
            }

            // Extract group signals and confidences
            let group_signals: Array2<f32> = Array2::from_shape_fn(
                (group.len(), seq_len),
                |(i, j)| signals_array[(group[i], j)]
            );
            let group_confidences: Array2<f32> = Array2::from_shape_fn(
                (group.len(), seq_len),
                |(i, j)| confidences_array[(group[i], j)]
            );

            // Convert back to tensors for group fusion
            let group_signals_tensor = self.array_to_tensor(&group_signals)?;
            let group_confidences_tensor = self.array_to_tensor(&group_confidences)?;

            // Fuse within group using score fusion
            let group_result = self.score_fusion(
                &group_signals_tensor,
                &group_confidences_tensor,
                params,
            ).await?;

            group_results.push(group_result.fused_signal);
            group_weights.push(group_result.average_confidence());
        }

        // Fuse group results
        if group_results.is_empty() {
            return Err(FusionError::invalid_input("No valid groups created"));
        }

        // Normalize group weights
        let total_weight: f32 = group_weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(FusionError::numerical("Invalid group weights"));
        }

        for weight in &mut group_weights {
            *weight /= total_weight;
        }

        // Combine group results
        let mut fused_signal = Array1::zeros(seq_len);
        let mut final_confidence = Array1::zeros(seq_len);

        for (i, group_signal) in group_results.iter().enumerate() {
            let weight = group_weights[i];
            for t in 0..seq_len {
                fused_signal[t] += weight * group_signal[t];
                final_confidence[t] += weight; // Accumulate group confidence
            }
        }

        // Create weights array (distributed across original signals)
        let mut weights_arrays = vec![Array1::zeros(seq_len); n_signals];
        for (group_idx, group) in groups.iter().enumerate() {
            let group_weight = group_weights[group_idx] / group.len() as f32;
            for &signal_idx in group {
                weights_arrays[signal_idx].fill(group_weight);
            }
        }

        Ok(FusionResult::new(
            fused_signal,
            final_confidence,
            weights_arrays,
            FusionType::Layered,
        ))
    }

    /// Adaptive fusion: Dynamic method selection based on signal properties
    async fn adaptive_fusion(
        &self,
        signals: &Tensor,
        confidences: &Tensor,
        params: &FusionParams,
    ) -> Result<FusionResult> {
        // Analyze signal properties to determine best fusion method
        let signals_array = self.tensor_to_array2(&signals.to_device(&Device::Cpu)?)?;
        let analysis = self.analyze_signal_properties(&signals_array)?;

        // Select fusion method based on analysis
        let selected_method = if analysis.high_correlation {
            // High correlation -> use rank-based to emphasize differences
            FusionType::Rank
        } else if analysis.high_variance {
            // High variance -> use weighted fusion for diversity
            FusionType::Weighted
        } else if analysis.low_snr {
            // Low SNR -> use layered fusion for robustness
            FusionType::Layered
        } else {
            // Default to hybrid fusion
            FusionType::Hybrid
        };

        log::debug!("Adaptive fusion selected method: {:?}", selected_method);

        // Perform selected fusion
        let mut result = match selected_method {
            FusionType::Score => self.score_fusion(signals, confidences, params).await?,
            FusionType::Rank => self.rank_fusion(signals, confidences, params).await?,
            FusionType::Hybrid => self.hybrid_fusion(signals, confidences, params).await?,
            FusionType::Weighted => self.weighted_fusion(signals, confidences, params).await?,
            FusionType::Layered => self.layered_fusion(signals, confidences, params).await?,
            FusionType::Adaptive => {
                // Prevent infinite recursion - fallback to hybrid
                self.hybrid_fusion(signals, confidences, params).await?
            }
        };

        // Update result to reflect adaptive nature
        result.fusion_type = FusionType::Adaptive;
        result.metadata.add_custom(
            "selected_method".to_string(),
            serde_json::Value::String(selected_method.to_string()),
        );
        result.metadata.add_custom(
            "signal_analysis".to_string(),
            serde_json::to_value(&analysis)?,
        );

        Ok(result)
    }

    // Helper methods...

    /// Validate input dimensions and values
    fn validate_inputs(&self, signals: &Array2<f32>, confidences: &Array2<f32>) -> Result<()> {
        if signals.is_empty() || confidences.is_empty() {
            return Err(FusionError::EmptyInput);
        }

        let signals_shape = signals.dim();
        let confidences_shape = confidences.dim();

        if signals_shape != confidences_shape {
            return Err(FusionError::dimension_mismatch(
                format!("{:?}", signals_shape),
                format!("{:?}", confidences_shape),
            ));
        }

        // Check for non-finite values
        for &val in signals.iter() {
            if !val.is_finite() {
                return Err(FusionError::numerical("Non-finite value in signals"));
            }
        }

        for &val in confidences.iter() {
            if !val.is_finite() || val < 0.0 || val > 1.0 {
                return Err(FusionError::numerical("Invalid confidence value"));
            }
        }

        Ok(())
    }

    /// Ensure model is compiled for the given fusion type and parameters
    async fn ensure_model_compiled(&mut self, fusion_type: FusionType, params: &FusionParams) -> Result<()> {
        if !self.model_cache.contains_key(&fusion_type) {
            let start_time = Instant::now();
            
            // Create compiled model entry
            let compiled_model = CompiledModel {
                fusion_type,
                compiled_at: std::time::SystemTime::now(),
                params: params.clone(),
                device: self.device_manager.primary_device().clone(),
            };

            self.model_cache.insert(fusion_type, compiled_model);

            let compilation_time = start_time.elapsed().as_micros() as u64;
            self.performance_stats.total_compilation_time_us += compilation_time;

            log::debug!("Compiled model for {:?} in {}μs", fusion_type, compilation_time);
        }

        Ok(())
    }

    /// Convert ndarray to tensor on device
    fn array_to_tensor(&self, array: &Array2<f32>) -> Result<Tensor> {
        let device = self.device_manager.primary_device();
        let shape = array.dim();
        let data: Vec<f32> = array.as_slice().unwrap().to_vec();
        
        Tensor::from_slice(&data, (shape.0, shape.1), device)
            .map_err(|e| FusionError::device(format!("Failed to create tensor: {}", e)))
    }

    /// Convert tensor to ndarray Array1
    fn tensor_to_array1(&self, tensor: &Tensor) -> Result<Array1<f32>> {
        let tensor_cpu = tensor.to_device(&Device::Cpu)?;
        let data = tensor_cpu.to_vec1::<f32>()?;
        Ok(Array1::from_vec(data))
    }

    /// Convert tensor to ndarray Array2
    fn tensor_to_array2(&self, tensor: &Tensor) -> Result<Array2<f32>> {
        let tensor_cpu = tensor.to_device(&Device::Cpu)?;
        let data = tensor_cpu.to_vec2::<f32>()?;
        let shape = tensor.shape();
        
        Array2::from_shape_vec((shape[0], shape[1]), data.into_iter().flatten().collect())
            .map_err(|e| FusionError::dimension_mismatch(
                format!("tensor shape: {:?}", shape),
                format!("array error: {}", e),
            ))
    }

    /// Convert weights tensor to array of Array1
    fn tensor_to_weight_arrays(&self, weights: &Tensor) -> Result<Vec<Array1<f32>>> {
        let weights_array = self.tensor_to_array2(weights)?;
        let n_signals = weights_array.nrows();
        
        Ok((0..n_signals)
            .map(|i| weights_array.row(i).to_owned())
            .collect())
    }

    /// Calculate correlation between two signals
    fn calculate_correlation(&self, x: &Array1<f32>, y: &Array1<f32>) -> Result<f32> {
        if x.len() != y.len() || x.is_empty() {
            return Err(FusionError::dimension_mismatch(
                format!("x length: {}", x.len()),
                format!("y length: {}", y.len()),
            ));
        }

        let n = x.len() as f32;
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom > f32::EPSILON {
            Ok(sum_xy / denom)
        } else {
            Ok(0.0)
        }
    }

    /// Create signal groups based on similarity
    fn create_signal_groups(&self, signals: &Array2<f32>, max_groups: usize) -> Result<Vec<Vec<usize>>> {
        let n_signals = signals.nrows();
        if n_signals <= max_groups {
            // Each signal gets its own group
            return Ok((0..n_signals).map(|i| vec![i]).collect());
        }

        let mut groups = vec![vec![0]]; // Start with first signal in first group
        
        // Assign remaining signals to groups based on correlation
        for i in 1..n_signals {
            let signal_i = signals.row(i).to_owned();
            let mut best_group = 0;
            let mut best_correlation = -1.0;

            // Find the group with highest average correlation
            for (group_idx, group) in groups.iter().enumerate() {
                let mut total_corr = 0.0;
                for &j in group {
                    let signal_j = signals.row(j).to_owned();
                    total_corr += self.calculate_correlation(&signal_i, &signal_j)?;
                }
                let avg_corr = total_corr / group.len() as f32;
                
                if avg_corr > best_correlation {
                    best_correlation = avg_corr;
                    best_group = group_idx;
                }
            }

            // Add to best group or create new group if correlation is low
            if best_correlation > 0.3 {
                groups[best_group].push(i);
            } else if groups.len() < max_groups {
                groups.push(vec![i]);
            } else {
                // Add to largest group
                let largest_group = groups
                    .iter_mut()
                    .max_by_key(|g| g.len())
                    .unwrap();
                largest_group.push(i);
            }
        }

        Ok(groups)
    }

    /// Analyze signal properties for adaptive fusion
    fn analyze_signal_properties(&self, signals: &Array2<f32>) -> Result<SignalAnalysis> {
        let (n_signals, seq_len) = signals.dim();
        
        // Calculate average correlation
        let mut total_corr = 0.0;
        let mut corr_count = 0;
        
        for i in 0..n_signals {
            for j in (i + 1)..n_signals {
                let corr = self.calculate_correlation(
                    &signals.row(i).to_owned(),
                    &signals.row(j).to_owned(),
                )?;
                total_corr += corr.abs();
                corr_count += 1;
            }
        }
        
        let avg_correlation = if corr_count > 0 {
            total_corr / corr_count as f32
        } else {
            0.0
        };

        // Calculate signal variance
        let mut variances = Vec::new();
        for i in 0..n_signals {
            let signal = signals.row(i);
            let mean = signal.mean().unwrap_or(0.0);
            let variance = signal.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / seq_len as f32;
            variances.push(variance);
        }
        
        let avg_variance = variances.iter().sum::<f32>() / n_signals as f32;
        let variance_std = {
            let var_mean = avg_variance;
            let var_variance = variances.iter()
                .map(|&v| (v - var_mean).powi(2))
                .sum::<f32>() / n_signals as f32;
            var_variance.sqrt()
        };

        // Estimate SNR (simplified)
        let mut snr_estimates = Vec::new();
        for i in 0..n_signals {
            let signal = signals.row(i);
            let signal_power = signal.iter().map(|&x| x.powi(2)).sum::<f32>() / seq_len as f32;
            
            // Estimate noise as high-frequency component (simple approximation)
            let mut noise_power = 0.0;
            for j in 1..seq_len {
                let diff = signal[j] - signal[j - 1];
                noise_power += diff.powi(2);
            }
            noise_power /= (seq_len - 1) as f32;
            
            let snr = if noise_power > f32::EPSILON {
                10.0 * (signal_power / noise_power).log10()
            } else {
                100.0 // Very high SNR
            };
            snr_estimates.push(snr);
        }
        
        let avg_snr = snr_estimates.iter().sum::<f32>() / n_signals as f32;

        Ok(SignalAnalysis {
            avg_correlation,
            avg_variance,
            variance_std,
            avg_snr,
            high_correlation: avg_correlation > 0.7,
            high_variance: variance_std > avg_variance * 0.5,
            low_snr: avg_snr < 10.0,
        })
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceStats {
        &self.performance_stats
    }

    /// Get device manager
    pub fn device_manager(&self) -> &DeviceManager {
        &self.device_manager
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.model_cache.clear();
        log::info!("Model cache cleared");
    }
}

/// Signal analysis results for adaptive fusion
#[derive(Debug, Clone, serde::Serialize)]
struct SignalAnalysis {
    avg_correlation: f32,
    avg_variance: f32,
    variance_std: f32,
    avg_snr: f32,
    high_correlation: bool,
    high_variance: bool,
    low_snr: bool,
}

impl PerformanceStats {
    /// Get average inference time in microseconds
    pub fn avg_inference_time_us(&self) -> f64 {
        if self.total_operations > 0 {
            self.total_inference_time_us as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    /// Get average compilation time in microseconds
    pub fn avg_compilation_time_us(&self) -> f64 {
        if self.total_operations > 0 {
            self.total_compilation_time_us as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    /// Check if sub-microsecond performance is achieved
    pub fn is_sub_microsecond(&self) -> bool {
        self.avg_inference_time_us() < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_torchscript_fusion_creation() {
        let fusion = TorchScriptFusion::new().await;
        assert!(fusion.is_ok());
    }

    #[tokio::test]
    async fn test_score_fusion() {
        let mut fusion = TorchScriptFusion::new().await.unwrap();
        
        let signals = Array2::from_shape_fn((3, 10), |(i, j)| {
            (i as f32 + 1.0) * (j as f32 + 1.0) * 0.1
        });
        let confidences = Array2::from_elem((3, 10), 0.8);
        
        let result = fusion.fuse_signals(
            &signals,
            &confidences,
            FusionType::Score,
            &FusionParams::default(),
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.fusion_type, FusionType::Score);
        assert_eq!(result.num_signals(), 3);
        assert_eq!(result.sequence_length(), 10);
        assert!(result.validate().is_ok());
    }

    #[tokio::test]
    async fn test_all_fusion_types() {
        let mut fusion = TorchScriptFusion::new().await.unwrap();
        
        let signals = Array2::from_shape_fn((2, 5), |(i, j)| {
            (i + j) as f32 * 0.1
        });
        let confidences = Array2::from_elem((2, 5), 0.9);
        
        for fusion_type in FusionType::all() {
            let result = fusion.fuse_signals(
                &signals,
                &confidences,
                fusion_type,
                &FusionParams::default(),
            ).await;
            
            assert!(result.is_ok(), "Failed for fusion type: {:?}", fusion_type);
            let result = result.unwrap();
            assert!(result.validate().is_ok());
        }
    }

    #[tokio::test]
    async fn test_input_validation() {
        let mut fusion = TorchScriptFusion::new().await.unwrap();
        
        // Empty input
        let empty_signals = Array2::zeros((0, 0));
        let empty_confidences = Array2::zeros((0, 0));
        
        let result = fusion.fuse_signals(
            &empty_signals,
            &empty_confidences,
            FusionType::Score,
            &FusionParams::default(),
        ).await;
        
        assert!(result.is_err());
        
        // Dimension mismatch
        let signals = Array2::zeros((2, 5));
        let confidences = Array2::zeros((3, 5));
        
        let result = fusion.fuse_signals(
            &signals,
            &confidences,
            FusionType::Score,
            &FusionParams::default(),
        ).await;
        
        assert!(result.is_err());
    }

    #[test]
    fn test_correlation_calculation() {
        let fusion = TorchScriptFusion {
            device_manager: DeviceManager::default(),
            model_cache: HashMap::new(),
            performance_stats: PerformanceStats::default(),
        };
        
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        
        let corr = fusion.calculate_correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-6);
        
        let z = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let corr = fusion.calculate_correlation(&x, &z).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-6);
    }
}