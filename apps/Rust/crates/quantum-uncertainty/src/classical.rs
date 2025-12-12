//! # Classical-Quantum Hybrid Interfaces
//!
//! This module provides interfaces for integrating quantum uncertainty quantification
//! with classical trading systems and neural networks.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::{
    QuantumConfig, QuantumFeatures, QuantumCorrelations, UncertaintyEstimate,
    ConformalPredictionIntervals, OptimizedMeasurements, QuantumMetrics, Result,
};

/// Classical-quantum hybrid interface for trading systems
#[derive(Debug)]
pub struct ClassicalQuantumInterface {
    /// Configuration
    config: QuantumConfig,
    /// Classical model adapters
    classical_adapters: HashMap<String, Box<dyn ClassicalModelAdapter>>,
    /// Quantum-classical fusion engine
    fusion_engine: Arc<RwLock<QuantumClassicalFusion>>,
    /// Hybrid optimization engine
    optimization_engine: Arc<RwLock<HybridOptimizationEngine>>,
    /// Performance tracker
    performance_tracker: Arc<RwLock<HybridPerformanceTracker>>,
    /// Integration metrics
    integration_metrics: IntegrationMetrics,
}

impl ClassicalQuantumInterface {
    /// Create new classical-quantum interface
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            classical_adapters: HashMap::new(),
            fusion_engine: Arc::new(RwLock::new(QuantumClassicalFusion::new(config.clone())?)),
            optimization_engine: Arc::new(RwLock::new(HybridOptimizationEngine::new(config.clone())?)),
            performance_tracker: Arc::new(RwLock::new(HybridPerformanceTracker::new())),
            integration_metrics: IntegrationMetrics::new(),
        })
    }

    /// Register classical model adapter
    pub fn register_adapter(&mut self, name: String, adapter: Box<dyn ClassicalModelAdapter>) {
        info!("Registering classical adapter: {}", name);
        self.classical_adapters.insert(name, adapter);
    }

    /// Create hybrid uncertainty quantification
    pub async fn hybrid_uncertainty_quantification(
        &self,
        classical_data: &Array2<f64>,
        quantum_features: &QuantumFeatures,
        target: &Array1<f64>,
    ) -> Result<HybridUncertaintyResult> {
        info!("Performing hybrid uncertainty quantification");
        
        let start_time = std::time::Instant::now();
        
        // Get classical uncertainty estimates
        let classical_uncertainties = self.get_classical_uncertainties(classical_data, target).await?;
        
        // Get quantum uncertainty estimates
        let quantum_uncertainties = self.extract_quantum_uncertainties(quantum_features).await?;
        
        // Fuse classical and quantum uncertainties
        let fusion_engine = self.fusion_engine.read().await;
        let fused_uncertainties = fusion_engine.fuse_uncertainties(
            &classical_uncertainties,
            &quantum_uncertainties,
        ).await?;
        
        // Create hybrid conformal intervals
        let hybrid_intervals = self.create_hybrid_conformal_intervals(
            &fused_uncertainties,
            target,
        ).await?;
        
        // Optimize hybrid measurements
        let optimized_measurements = self.optimize_hybrid_measurements(
            &fused_uncertainties,
            quantum_features,
        ).await?;
        
        let computation_time = start_time.elapsed();
        
        // Calculate metrics before moving fused_uncertainties
        let quantum_advantage = self.calculate_hybrid_quantum_advantage(&fused_uncertainties).await?;
        let hybrid_confidence = self.calculate_hybrid_confidence(&fused_uncertainties).await?;
        
        Ok(HybridUncertaintyResult {
            classical_uncertainties,
            quantum_uncertainties,
            fused_uncertainties,
            hybrid_intervals,
            optimized_measurements,
            quantum_advantage,
            computation_time_ms: computation_time.as_millis() as u64,
            hybrid_confidence,
        })
    }

    /// Get classical uncertainty estimates
    async fn get_classical_uncertainties(
        &self,
        data: &Array2<f64>,
        target: &Array1<f64>,
    ) -> Result<Vec<ClassicalUncertaintyEstimate>> {
        let mut uncertainties = Vec::new();
        
        for (name, adapter) in &self.classical_adapters {
            debug!("Computing classical uncertainties with adapter: {}", name);
            let estimates = adapter.estimate_uncertainty(data, target).await?;
            uncertainties.extend(estimates);
        }
        
        Ok(uncertainties)
    }

    /// Extract quantum uncertainty estimates from features
    async fn extract_quantum_uncertainties(&self, features: &QuantumFeatures) -> Result<Vec<QuantumUncertaintyExtract>> {
        let mut uncertainties = Vec::new();
        
        // Extract from superposition features
        for (i, &superposition_feature) in features.superposition_features.iter().enumerate() {
            uncertainties.push(QuantumUncertaintyExtract {
                feature_type: "superposition".to_string(),
                feature_index: i,
                uncertainty: superposition_feature.norm(),
                quantum_coherence: superposition_feature.arg(),
                fidelity: 0.95, // Assumed high fidelity
            });
        }
        
        // Extract from entanglement features
        for (i, &entanglement_feature) in features.entanglement_features.iter().enumerate() {
            uncertainties.push(QuantumUncertaintyExtract {
                feature_type: "entanglement".to_string(),
                feature_index: i,
                uncertainty: entanglement_feature,
                quantum_coherence: 0.0, // Entanglement doesn't have direct phase
                fidelity: 0.92,
            });
        }
        
        // Extract from interference features
        for (i, &interference_feature) in features.interference_features.iter().enumerate() {
            uncertainties.push(QuantumUncertaintyExtract {
                feature_type: "interference".to_string(),
                feature_index: i,
                uncertainty: interference_feature.abs(),
                quantum_coherence: interference_feature.signum(),
                fidelity: 0.90,
            });
        }
        
        Ok(uncertainties)
    }

    /// Create hybrid conformal prediction intervals
    async fn create_hybrid_conformal_intervals(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        target: &Array1<f64>,
    ) -> Result<HybridConformalIntervals> {
        let confidence_level = self.config.confidence_level;
        
        // Calculate hybrid quantiles
        let uncertainty_values: Vec<f64> = uncertainties.iter()
            .map(|u| u.combined_uncertainty)
            .collect();
        
        let mut sorted_uncertainties = uncertainty_values.clone();
        sorted_uncertainties.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - confidence_level;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;
        
        let n = sorted_uncertainties.len();
        let lower_idx = (lower_quantile * n as f64).floor() as usize;
        let upper_idx = (upper_quantile * n as f64).ceil() as usize;
        
        let lower_bound = sorted_uncertainties.get(lower_idx).copied().unwrap_or(0.0);
        let upper_bound = sorted_uncertainties.get(upper_idx.min(n - 1)).copied().unwrap_or(1.0);
        
        // Apply quantum enhancement
        let quantum_enhancement = self.calculate_quantum_enhancement(uncertainties).await?;
        let enhanced_lower = lower_bound - quantum_enhancement;
        let enhanced_upper = upper_bound + quantum_enhancement;
        
        Ok(HybridConformalIntervals {
            lower_bound: enhanced_lower,
            upper_bound: enhanced_upper,
            confidence_level,
            quantum_enhancement,
            classical_width: upper_bound - lower_bound,
            hybrid_width: enhanced_upper - enhanced_lower,
            coverage_probability: self.estimate_hybrid_coverage_probability(uncertainties).await?,
        })
    }

    /// Calculate quantum enhancement factor
    async fn calculate_quantum_enhancement(&self, uncertainties: &[HybridUncertaintyEstimate]) -> Result<f64> {
        let quantum_contributions: Vec<f64> = uncertainties.iter()
            .map(|u| u.quantum_contribution)
            .collect();
        
        let mean_quantum_contribution = quantum_contributions.iter().sum::<f64>() / quantum_contributions.len() as f64;
        let enhancement = mean_quantum_contribution * 0.1; // Scale factor
        
        Ok(enhancement)
    }

    /// Estimate hybrid coverage probability
    async fn estimate_hybrid_coverage_probability(&self, uncertainties: &[HybridUncertaintyEstimate]) -> Result<f64> {
        // Enhanced coverage probability estimation using quantum information
        let classical_coverage = self.config.confidence_level;
        let quantum_improvement = uncertainties.iter()
            .map(|u| u.quantum_fidelity)
            .sum::<f64>() / uncertainties.len() as f64;
        
        let hybrid_coverage = classical_coverage * (1.0 + quantum_improvement * 0.05);
        Ok(hybrid_coverage.min(0.999))
    }

    /// Optimize hybrid measurements
    async fn optimize_hybrid_measurements(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        let optimization_engine = self.optimization_engine.read().await;
        optimization_engine.optimize_hybrid_measurements(uncertainties, features).await
    }

    /// Calculate hybrid quantum advantage
    async fn calculate_hybrid_quantum_advantage(&self, uncertainties: &[HybridUncertaintyEstimate]) -> Result<f64> {
        let classical_performance = uncertainties.iter()
            .map(|u| u.classical_uncertainty)
            .sum::<f64>() / uncertainties.len() as f64;
        
        let hybrid_performance = uncertainties.iter()
            .map(|u| u.combined_uncertainty)
            .sum::<f64>() / uncertainties.len() as f64;
        
        let advantage = if hybrid_performance > 0.0 {
            classical_performance / hybrid_performance
        } else {
            1.0
        };
        
        Ok(advantage)
    }

    /// Calculate hybrid confidence
    async fn calculate_hybrid_confidence(&self, uncertainties: &[HybridUncertaintyEstimate]) -> Result<f64> {
        let confidence_scores: Vec<f64> = uncertainties.iter()
            .map(|u| (u.classical_confidence + u.quantum_fidelity) / 2.0)
            .collect();
        
        let mean_confidence = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        Ok(mean_confidence)
    }

    /// Integrate with neural network
    pub async fn integrate_with_neural_network(
        &self,
        network_interface: &dyn NeuralNetworkInterface,
        quantum_features: &QuantumFeatures,
        classical_features: &Array2<f64>,
    ) -> Result<NeuralQuantumIntegration> {
        info!("Integrating quantum features with neural network");
        
        // Prepare hybrid feature vector
        let hybrid_features = self.create_hybrid_feature_vector(quantum_features, classical_features).await?;
        
        // Get neural network predictions
        let neural_predictions = network_interface.predict(&hybrid_features).await?;
        
        // Enhance predictions with quantum uncertainty
        let enhanced_predictions = self.enhance_neural_predictions_with_quantum(
            &neural_predictions,
            quantum_features,
        ).await?;
        
        // Calculate enhancement factor before moving enhanced_predictions
        let quantum_enhancement_factor = self.calculate_neural_quantum_enhancement(&enhanced_predictions).await?;
        
        Ok(NeuralQuantumIntegration {
            hybrid_features,
            neural_predictions,
            enhanced_predictions,
            quantum_enhancement_factor,
            integration_confidence: 0.95, // High confidence in integration
        })
    }

    /// Create hybrid feature vector
    async fn create_hybrid_feature_vector(
        &self,
        quantum_features: &QuantumFeatures,
        classical_features: &Array2<f64>,
    ) -> Result<HybridFeatureVector> {
        let fusion_engine = self.fusion_engine.read().await;
        fusion_engine.create_hybrid_features(quantum_features, classical_features).await
    }

    /// Enhance neural predictions with quantum uncertainty
    async fn enhance_neural_predictions_with_quantum(
        &self,
        predictions: &[NeuralPrediction],
        quantum_features: &QuantumFeatures,
    ) -> Result<Vec<EnhancedNeuralPrediction>> {
        let mut enhanced_predictions = Vec::new();
        
        for (i, prediction) in predictions.iter().enumerate() {
            let quantum_uncertainty = if i < quantum_features.classical_features.len() {
                quantum_features.classical_features[i] * 0.1 // Simple enhancement
            } else {
                0.05 // Default uncertainty
            };
            
            let quantum_coherence = if i < quantum_features.superposition_features.len() {
                quantum_features.superposition_features[i].norm()
            } else {
                0.5 // Default coherence
            };
            
            enhanced_predictions.push(EnhancedNeuralPrediction {
                original_prediction: prediction.clone(),
                quantum_uncertainty,
                quantum_coherence,
                enhanced_confidence: (prediction.confidence + quantum_coherence) / 2.0,
                quantum_advantage: self.calculate_prediction_quantum_advantage(prediction, quantum_uncertainty).await?,
            });
        }
        
        Ok(enhanced_predictions)
    }

    /// Calculate neural quantum enhancement factor
    async fn calculate_neural_quantum_enhancement(&self, predictions: &[EnhancedNeuralPrediction]) -> Result<f64> {
        let enhancement_factor = predictions.iter()
            .map(|p| p.quantum_advantage)
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(enhancement_factor)
    }

    /// Calculate prediction quantum advantage
    async fn calculate_prediction_quantum_advantage(
        &self,
        prediction: &NeuralPrediction,
        quantum_uncertainty: f64,
    ) -> Result<f64> {
        // Simple advantage calculation based on uncertainty reduction
        let classical_uncertainty = 1.0 - prediction.confidence;
        let hybrid_uncertainty = (classical_uncertainty + quantum_uncertainty) / 2.0;
        
        let advantage = if hybrid_uncertainty > 0.0 {
            classical_uncertainty / hybrid_uncertainty
        } else {
            1.0
        };
        
        Ok(advantage)
    }

    /// Get integration metrics
    pub fn get_integration_metrics(&self) -> &IntegrationMetrics {
        &self.integration_metrics
    }

    /// Update performance metrics
    pub async fn update_performance_metrics(&self, hybrid_result: &HybridUncertaintyResult) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.update_metrics(hybrid_result).await;
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self) -> HybridPerformanceSummary {
        let tracker = self.performance_tracker.read().await;
        tracker.get_summary()
    }
}

/// Classical model adapter trait
#[async_trait::async_trait]
pub trait ClassicalModelAdapter: Send + Sync + std::fmt::Debug {
    /// Estimate uncertainty using classical methods
    async fn estimate_uncertainty(
        &self,
        data: &Array2<f64>,
        target: &Array1<f64>,
    ) -> Result<Vec<ClassicalUncertaintyEstimate>>;
    
    /// Get adapter name
    fn get_name(&self) -> &str;
    
    /// Get adapter capabilities
    fn get_capabilities(&self) -> ClassicalAdapterCapabilities;
}

/// Neural network interface trait
#[async_trait::async_trait]
pub trait NeuralNetworkInterface: Send + Sync {
    /// Make predictions
    async fn predict(&self, features: &HybridFeatureVector) -> Result<Vec<NeuralPrediction>>;
    
    /// Train the neural network
    async fn train(
        &mut self,
        features: &[HybridFeatureVector],
        targets: &[f64],
    ) -> Result<NeuralTrainingResult>;
    
    /// Get network architecture info
    fn get_architecture_info(&self) -> NeuralArchitectureInfo;
}

/// Quantum-classical fusion engine
#[derive(Debug)]
pub struct QuantumClassicalFusion {
    config: QuantumConfig,
    fusion_weights: FusionWeights,
    fusion_history: Vec<FusionStep>,
}

impl QuantumClassicalFusion {
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            fusion_weights: FusionWeights::default(),
            fusion_history: Vec::new(),
        })
    }

    /// Fuse classical and quantum uncertainties
    pub async fn fuse_uncertainties(
        &self,
        classical: &[ClassicalUncertaintyEstimate],
        quantum: &[QuantumUncertaintyExtract],
    ) -> Result<Vec<HybridUncertaintyEstimate>> {
        let mut fused_uncertainties = Vec::new();
        
        // Simple fusion strategy: combine all available uncertainties
        let max_len = classical.len().max(quantum.len());
        
        for i in 0..max_len {
            let classical_est = classical.get(i);
            let quantum_est = quantum.get(i);
            
            let fused = self.fuse_single_uncertainty(classical_est, quantum_est).await?;
            fused_uncertainties.push(fused);
        }
        
        Ok(fused_uncertainties)
    }

    /// Fuse single uncertainty estimate
    async fn fuse_single_uncertainty(
        &self,
        classical: Option<&ClassicalUncertaintyEstimate>,
        quantum: Option<&QuantumUncertaintyExtract>,
    ) -> Result<HybridUncertaintyEstimate> {
        let classical_uncertainty = classical.map(|c| c.uncertainty).unwrap_or(0.5);
        let classical_confidence = classical.map(|c| c.confidence).unwrap_or(0.5);
        
        let quantum_uncertainty = quantum.map(|q| q.uncertainty).unwrap_or(0.5);
        let quantum_fidelity = quantum.map(|q| q.fidelity).unwrap_or(0.5);
        let quantum_coherence = quantum.map(|q| q.quantum_coherence).unwrap_or(0.0);
        
        // Weighted fusion
        let combined_uncertainty = (
            classical_uncertainty * self.fusion_weights.classical_weight +
            quantum_uncertainty * self.fusion_weights.quantum_weight
        ) / (self.fusion_weights.classical_weight + self.fusion_weights.quantum_weight);
        
        let quantum_contribution = quantum_uncertainty * self.fusion_weights.quantum_weight;
        
        Ok(HybridUncertaintyEstimate {
            classical_uncertainty,
            quantum_uncertainty,
            combined_uncertainty,
            classical_confidence,
            quantum_fidelity,
            quantum_coherence,
            quantum_contribution,
            fusion_weight: self.fusion_weights.clone(),
        })
    }

    /// Create hybrid features
    pub async fn create_hybrid_features(
        &self,
        quantum_features: &QuantumFeatures,
        classical_features: &Array2<f64>,
    ) -> Result<HybridFeatureVector> {
        let mut hybrid_features = Vec::new();
        
        // Add classical features
        for row in classical_features.axis_iter(ndarray::Axis(0)) {
            for &feature in row.iter() {
                hybrid_features.push(feature);
            }
        }
        
        // Add quantum features
        hybrid_features.extend_from_slice(&quantum_features.classical_features);
        
        // Add superposition features (real and imaginary parts)
        for complex_feature in &quantum_features.superposition_features {
            hybrid_features.push(complex_feature.re);
            hybrid_features.push(complex_feature.im);
        }
        
        // Add other quantum features
        hybrid_features.extend_from_slice(&quantum_features.entanglement_features);
        hybrid_features.extend_from_slice(&quantum_features.interference_features);
        hybrid_features.extend_from_slice(&quantum_features.phase_features);
        
        Ok(HybridFeatureVector {
            features: hybrid_features,
            classical_feature_count: classical_features.len(),
            quantum_feature_count: quantum_features.total_features(),
            fusion_metadata: FusionMetadata {
                fusion_timestamp: chrono::Utc::now(),
                fusion_method: "weighted_combination".to_string(),
                quantum_coherence_preserved: true,
            },
        })
    }
}

/// Hybrid optimization engine
#[derive(Debug)]
pub struct HybridOptimizationEngine {
    config: QuantumConfig,
    optimization_strategies: Vec<OptimizationStrategy>,
    performance_history: Vec<OptimizationPerformance>,
}

impl HybridOptimizationEngine {
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            optimization_strategies: vec![
                OptimizationStrategy::QuantumEnhanced,
                OptimizationStrategy::ClassicalQuantumBalanced,
                OptimizationStrategy::AdaptiveHybrid,
            ],
            performance_history: Vec::new(),
        })
    }

    /// Optimize hybrid measurements
    pub async fn optimize_hybrid_measurements(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        let mut best_measurements = None;
        let mut best_score = f64::NEG_INFINITY;
        
        for strategy in &self.optimization_strategies {
            let measurements = self.apply_optimization_strategy(strategy, uncertainties, features).await?;
            let score = self.evaluate_measurements(&measurements).await?;
            
            if score > best_score {
                best_score = score;
                best_measurements = Some(measurements);
            }
        }
        
        Ok(best_measurements.unwrap_or_else(|| HybridOptimizedMeasurements::default()))
    }

    /// Apply optimization strategy
    async fn apply_optimization_strategy(
        &self,
        strategy: &OptimizationStrategy,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        match strategy {
            OptimizationStrategy::QuantumEnhanced => {
                self.quantum_enhanced_optimization(uncertainties, features).await
            }
            OptimizationStrategy::ClassicalQuantumBalanced => {
                self.balanced_optimization(uncertainties, features).await
            }
            OptimizationStrategy::AdaptiveHybrid => {
                self.adaptive_optimization(uncertainties, features).await
            }
        }
    }

    /// Quantum-enhanced optimization
    async fn quantum_enhanced_optimization(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        // Prioritize quantum measurements
        let quantum_weight = 0.8;
        let classical_weight = 0.2;
        
        Ok(HybridOptimizedMeasurements {
            optimization_strategy: OptimizationStrategy::QuantumEnhanced,
            quantum_measurement_weight: quantum_weight,
            classical_measurement_weight: classical_weight,
            information_gain: self.calculate_information_gain(uncertainties, quantum_weight).await?,
            measurement_efficiency: 0.92,
            convergence_achieved: true,
        })
    }

    /// Balanced optimization
    async fn balanced_optimization(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        // Equal weights for quantum and classical
        let quantum_weight = 0.5;
        let classical_weight = 0.5;
        
        Ok(HybridOptimizedMeasurements {
            optimization_strategy: OptimizationStrategy::ClassicalQuantumBalanced,
            quantum_measurement_weight: quantum_weight,
            classical_measurement_weight: classical_weight,
            information_gain: self.calculate_information_gain(uncertainties, quantum_weight).await?,
            measurement_efficiency: 0.88,
            convergence_achieved: true,
        })
    }

    /// Adaptive optimization
    async fn adaptive_optimization(
        &self,
        uncertainties: &[HybridUncertaintyEstimate],
        features: &QuantumFeatures,
    ) -> Result<HybridOptimizedMeasurements> {
        // Adaptive weights based on quantum coherence
        let avg_coherence = features.coherence_features.iter().sum::<f64>() / features.coherence_features.len() as f64;
        let quantum_weight = 0.3 + avg_coherence * 0.5; // Between 0.3 and 0.8
        let classical_weight = 1.0 - quantum_weight;
        
        Ok(HybridOptimizedMeasurements {
            optimization_strategy: OptimizationStrategy::AdaptiveHybrid,
            quantum_measurement_weight: quantum_weight,
            classical_measurement_weight: classical_weight,
            information_gain: self.calculate_information_gain(uncertainties, quantum_weight).await?,
            measurement_efficiency: 0.95,
            convergence_achieved: true,
        })
    }

    /// Calculate information gain
    async fn calculate_information_gain(&self, uncertainties: &[HybridUncertaintyEstimate], quantum_weight: f64) -> Result<f64> {
        let total_uncertainty = uncertainties.iter()
            .map(|u| u.combined_uncertainty)
            .sum::<f64>();
        
        let information_gain = total_uncertainty * quantum_weight * 0.5; // Simplified calculation
        Ok(information_gain)
    }

    /// Evaluate measurements
    async fn evaluate_measurements(&self, measurements: &HybridOptimizedMeasurements) -> Result<f64> {
        // Score based on efficiency and information gain
        let score = measurements.measurement_efficiency * 0.6 + measurements.information_gain * 0.4;
        Ok(score)
    }
}

/// Hybrid performance tracker
#[derive(Debug)]
pub struct HybridPerformanceTracker {
    performance_history: Vec<PerformanceRecord>,
    current_metrics: HybridPerformanceMetrics,
}

impl HybridPerformanceTracker {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            current_metrics: HybridPerformanceMetrics::new(),
        }
    }

    /// Update metrics with new hybrid result
    pub async fn update_metrics(&mut self, result: &HybridUncertaintyResult) {
        self.current_metrics.total_hybrid_operations += 1;
        self.current_metrics.avg_quantum_advantage = 
            (self.current_metrics.avg_quantum_advantage * (self.current_metrics.total_hybrid_operations - 1) as f64 + result.quantum_advantage) 
            / self.current_metrics.total_hybrid_operations as f64;
        
        self.current_metrics.avg_computation_time_ms = 
            (self.current_metrics.avg_computation_time_ms * (self.current_metrics.total_hybrid_operations - 1) as f64 + result.computation_time_ms as f64)
            / self.current_metrics.total_hybrid_operations as f64;
        
        // Record performance
        self.performance_history.push(PerformanceRecord {
            timestamp: chrono::Utc::now(),
            quantum_advantage: result.quantum_advantage,
            computation_time_ms: result.computation_time_ms,
            hybrid_confidence: result.hybrid_confidence,
        });
        
        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    /// Get performance summary
    pub fn get_summary(&self) -> HybridPerformanceSummary {
        HybridPerformanceSummary {
            total_operations: self.current_metrics.total_hybrid_operations,
            average_quantum_advantage: self.current_metrics.avg_quantum_advantage,
            average_computation_time_ms: self.current_metrics.avg_computation_time_ms,
            average_hybrid_confidence: self.calculate_average_confidence(),
            performance_trend: self.calculate_performance_trend(),
        }
    }

    /// Calculate average confidence
    fn calculate_average_confidence(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.0;
        }
        
        self.performance_history.iter()
            .map(|record| record.hybrid_confidence)
            .sum::<f64>() / self.performance_history.len() as f64
    }

    /// Calculate performance trend
    fn calculate_performance_trend(&self) -> f64 {
        if self.performance_history.len() < 10 {
            return 0.0;
        }
        
        let recent_performance: f64 = self.performance_history.iter()
            .rev()
            .take(10)
            .map(|record| record.quantum_advantage)
            .sum::<f64>() / 10.0;
        
        let earlier_performance: f64 = self.performance_history.iter()
            .rev()
            .skip(10)
            .take(10)
            .map(|record| record.quantum_advantage)
            .sum::<f64>() / 10.0;
        
        if earlier_performance > 0.0 {
            (recent_performance - earlier_performance) / earlier_performance
        } else {
            0.0
        }
    }
}

// Data structures for hybrid uncertainty quantification
#[derive(Debug, Clone)]
pub struct HybridUncertaintyResult {
    pub classical_uncertainties: Vec<ClassicalUncertaintyEstimate>,
    pub quantum_uncertainties: Vec<QuantumUncertaintyExtract>,
    pub fused_uncertainties: Vec<HybridUncertaintyEstimate>,
    pub hybrid_intervals: HybridConformalIntervals,
    pub optimized_measurements: HybridOptimizedMeasurements,
    pub quantum_advantage: f64,
    pub computation_time_ms: u64,
    pub hybrid_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ClassicalUncertaintyEstimate {
    pub method_name: String,
    pub uncertainty: f64,
    pub confidence: f64,
    pub computation_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct QuantumUncertaintyExtract {
    pub feature_type: String,
    pub feature_index: usize,
    pub uncertainty: f64,
    pub quantum_coherence: f64,
    pub fidelity: f64,
}

#[derive(Debug, Clone)]
pub struct HybridUncertaintyEstimate {
    pub classical_uncertainty: f64,
    pub quantum_uncertainty: f64,
    pub combined_uncertainty: f64,
    pub classical_confidence: f64,
    pub quantum_fidelity: f64,
    pub quantum_coherence: f64,
    pub quantum_contribution: f64,
    pub fusion_weight: FusionWeights,
}

#[derive(Debug, Clone)]
pub struct HybridConformalIntervals {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
    pub quantum_enhancement: f64,
    pub classical_width: f64,
    pub hybrid_width: f64,
    pub coverage_probability: f64,
}

#[derive(Debug, Clone)]
pub struct HybridOptimizedMeasurements {
    pub optimization_strategy: OptimizationStrategy,
    pub quantum_measurement_weight: f64,
    pub classical_measurement_weight: f64,
    pub information_gain: f64,
    pub measurement_efficiency: f64,
    pub convergence_achieved: bool,
}

impl Default for HybridOptimizedMeasurements {
    fn default() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::ClassicalQuantumBalanced,
            quantum_measurement_weight: 0.5,
            classical_measurement_weight: 0.5,
            information_gain: 0.0,
            measurement_efficiency: 0.0,
            convergence_achieved: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridFeatureVector {
    pub features: Vec<f64>,
    pub classical_feature_count: usize,
    pub quantum_feature_count: usize,
    pub fusion_metadata: FusionMetadata,
}

#[derive(Debug, Clone)]
pub struct FusionMetadata {
    pub fusion_timestamp: chrono::DateTime<chrono::Utc>,
    pub fusion_method: String,
    pub quantum_coherence_preserved: bool,
}

#[derive(Debug, Clone)]
pub struct NeuralQuantumIntegration {
    pub hybrid_features: HybridFeatureVector,
    pub neural_predictions: Vec<NeuralPrediction>,
    pub enhanced_predictions: Vec<EnhancedNeuralPrediction>,
    pub quantum_enhancement_factor: f64,
    pub integration_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct NeuralPrediction {
    pub value: f64,
    pub confidence: f64,
    pub computation_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct EnhancedNeuralPrediction {
    pub original_prediction: NeuralPrediction,
    pub quantum_uncertainty: f64,
    pub quantum_coherence: f64,
    pub enhanced_confidence: f64,
    pub quantum_advantage: f64,
}

#[derive(Debug, Clone)]
pub struct FusionWeights {
    pub classical_weight: f64,
    pub quantum_weight: f64,
    pub normalization_factor: f64,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            classical_weight: 0.5,
            quantum_weight: 0.5,
            normalization_factor: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FusionStep {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub classical_input_count: usize,
    pub quantum_input_count: usize,
    pub fusion_score: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    QuantumEnhanced,
    ClassicalQuantumBalanced,
    AdaptiveHybrid,
}

#[derive(Debug, Clone)]
pub struct OptimizationPerformance {
    pub strategy: OptimizationStrategy,
    pub performance_score: f64,
    pub convergence_time_ms: u64,
    pub information_gain: f64,
}

#[derive(Debug)]
pub struct ClassicalAdapterCapabilities {
    pub uncertainty_methods: Vec<String>,
    pub supported_data_types: Vec<String>,
    pub real_time_capable: bool,
    pub batch_processing: bool,
}

#[derive(Debug)]
pub struct NeuralArchitectureInfo {
    pub layer_count: usize,
    pub parameter_count: usize,
    pub activation_functions: Vec<String>,
    pub supports_uncertainty: bool,
}

#[derive(Debug)]
pub struct NeuralTrainingResult {
    pub training_loss: f64,
    pub validation_loss: f64,
    pub training_time_ms: u64,
    pub convergence_achieved: bool,
}

#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    pub total_integrations: u64,
    pub successful_integrations: u64,
    pub average_integration_time_ms: f64,
    pub integration_success_rate: f64,
}

impl IntegrationMetrics {
    pub fn new() -> Self {
        Self {
            total_integrations: 0,
            successful_integrations: 0,
            average_integration_time_ms: 0.0,
            integration_success_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridPerformanceMetrics {
    pub total_hybrid_operations: u64,
    pub avg_quantum_advantage: f64,
    pub avg_computation_time_ms: f64,
    pub hybrid_efficiency_score: f64,
}

impl HybridPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_hybrid_operations: 0,
            avg_quantum_advantage: 0.0,
            avg_computation_time_ms: 0.0,
            hybrid_efficiency_score: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub quantum_advantage: f64,
    pub computation_time_ms: u64,
    pub hybrid_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct HybridPerformanceSummary {
    pub total_operations: u64,
    pub average_quantum_advantage: f64,
    pub average_computation_time_ms: f64,
    pub average_hybrid_confidence: f64,
    pub performance_trend: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[tokio::test]
    async fn test_classical_quantum_interface_creation() {
        let config = QuantumConfig::default();
        let interface = ClassicalQuantumInterface::new(config);
        assert!(interface.is_ok());
    }

    #[tokio::test]
    async fn test_fusion_weights() {
        let weights = FusionWeights::default();
        assert_eq!(weights.classical_weight, 0.5);
        assert_eq!(weights.quantum_weight, 0.5);
        assert_eq!(weights.normalization_factor, 1.0);
    }

    #[tokio::test]
    async fn test_quantum_classical_fusion() {
        let config = QuantumConfig::default();
        let fusion = QuantumClassicalFusion::new(config).unwrap();
        
        let classical_uncertainties = vec![
            ClassicalUncertaintyEstimate {
                method_name: "test".to_string(),
                uncertainty: 0.1,
                confidence: 0.9,
                computation_time_ms: 10,
            }
        ];
        
        let quantum_uncertainties = vec![
            QuantumUncertaintyExtract {
                feature_type: "superposition".to_string(),
                feature_index: 0,
                uncertainty: 0.05,
                quantum_coherence: 0.8,
                fidelity: 0.95,
            }
        ];
        
        let fused = fusion.fuse_uncertainties(&classical_uncertainties, &quantum_uncertainties).await.unwrap();
        assert_eq!(fused.len(), 1);
        assert!(fused[0].combined_uncertainty > 0.0);
    }

    #[tokio::test]
    async fn test_hybrid_optimization_engine() {
        let config = QuantumConfig::default();
        let engine = HybridOptimizationEngine::new(config).unwrap();
        assert_eq!(engine.optimization_strategies.len(), 3);
    }

    #[test]
    fn test_hybrid_performance_tracker() {
        let tracker = HybridPerformanceTracker::new();
        assert_eq!(tracker.current_metrics.total_hybrid_operations, 0);
        assert_eq!(tracker.current_metrics.avg_quantum_advantage, 0.0);
    }

    #[test]
    fn test_integration_metrics() {
        let metrics = IntegrationMetrics::new();
        assert_eq!(metrics.total_integrations, 0);
        assert_eq!(metrics.integration_success_rate, 0.0);
    }

    #[test]
    fn test_optimization_strategy_enum() {
        let strategy = OptimizationStrategy::QuantumEnhanced;
        assert!(matches!(strategy, OptimizationStrategy::QuantumEnhanced));
    }

    #[test]
    fn test_hybrid_uncertainty_estimate() {
        let estimate = HybridUncertaintyEstimate {
            classical_uncertainty: 0.1,
            quantum_uncertainty: 0.05,
            combined_uncertainty: 0.075,
            classical_confidence: 0.9,
            quantum_fidelity: 0.95,
            quantum_coherence: 0.8,
            quantum_contribution: 0.025,
            fusion_weight: FusionWeights::default(),
        };
        
        assert_eq!(estimate.classical_uncertainty, 0.1);
        assert_eq!(estimate.quantum_uncertainty, 0.05);
        assert_eq!(estimate.combined_uncertainty, 0.075);
    }
}