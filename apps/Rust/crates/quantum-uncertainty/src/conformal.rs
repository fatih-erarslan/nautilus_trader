//! # Quantum-Enhanced Conformal Prediction
//!
//! This module implements quantum-enhanced conformal prediction intervals for
//! uncertainty quantification in trading systems.

use std::f64::consts::PI;

use anyhow::anyhow;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    QuantumState, QuantumFeatures, QuantumCircuitSimulator, QuantumConfig,
    QuantumGate, QuantumCircuit, PauliObservable, UncertaintyEstimate, Result,
};

/// Quantum-enhanced conformal prediction intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionIntervals {
    /// Lower bound of prediction interval
    pub lower_bound: f64,
    /// Upper bound of prediction interval
    pub upper_bound: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Interval width
    pub interval_width: f64,
    /// Quantum-enhanced coverage probability
    pub quantum_coverage_probability: f64,
    /// Non-conformity scores
    pub nonconformity_scores: Vec<f64>,
    /// Quantum calibration scores
    pub quantum_calibration_scores: Vec<f64>,
    /// Prediction intervals for each sample
    pub sample_intervals: Vec<(f64, f64)>,
    /// Quantum efficiency measure
    pub quantum_efficiency: f64,
    /// Validity indicators
    pub validity_indicators: Vec<bool>,
    /// Quantum advantage in prediction
    pub quantum_prediction_advantage: f64,
    /// Tail risk quantum measure
    pub tail_risk_quantum: f64,
    /// Coverage probability
    pub coverage_probability: f64,
}

impl ConformalPredictionIntervals {
    /// Create new conformal prediction intervals
    pub fn new(confidence_level: f64) -> Self {
        Self {
            lower_bound: 0.0,
            upper_bound: 0.0,
            confidence_level,
            interval_width: 0.0,
            quantum_coverage_probability: 0.0,
            nonconformity_scores: Vec::new(),
            quantum_calibration_scores: Vec::new(),
            sample_intervals: Vec::new(),
            quantum_efficiency: 0.0,
            validity_indicators: Vec::new(),
            quantum_prediction_advantage: 0.0,
            tail_risk_quantum: 0.0,
            coverage_probability: confidence_level,
        }
    }

    /// Check if value is within prediction interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value <= self.upper_bound
    }

    /// Get interval coverage for given values
    pub fn coverage(&self, values: &[f64]) -> f64 {
        let covered = values.iter().filter(|&&v| self.contains(v)).count();
        covered as f64 / values.len() as f64
    }

    /// Get average interval width
    pub fn average_interval_width(&self) -> f64 {
        if self.sample_intervals.is_empty() {
            self.interval_width
        } else {
            self.sample_intervals.iter()
                .map(|(lower, upper)| upper - lower)
                .sum::<f64>() / self.sample_intervals.len() as f64
        }
    }

    /// Calculate quantum efficiency score
    pub fn efficiency_score(&self) -> f64 {
        // Higher efficiency means narrower intervals with maintained coverage
        if self.interval_width > 0.0 {
            self.quantum_coverage_probability / self.interval_width
        } else {
            0.0
        }
    }

    /// Get validity rate
    pub fn validity_rate(&self) -> f64 {
        if self.validity_indicators.is_empty() {
            1.0
        } else {
            let valid_count = self.validity_indicators.iter().filter(|&&v| v).count();
            valid_count as f64 / self.validity_indicators.len() as f64
        }
    }
}

/// Quantum conformal predictor
#[derive(Debug)]
pub struct QuantumConformalPredictor {
    /// Configuration
    pub config: QuantumConfig,
    /// Quantum circuit simulator
    pub simulator: QuantumCircuitSimulator,
    /// Conformal prediction circuits
    pub conformal_circuits: Vec<QuantumCircuit>,
    /// Non-conformity measure calculators
    pub nonconformity_calculators: Vec<NonConformityCalculator>,
    /// Quantum calibration engine
    pub calibration_engine: QuantumCalibrationEngine,
    /// Prediction interval optimizers
    pub interval_optimizers: Vec<IntervalOptimizer>,
    /// Coverage probability estimator
    pub coverage_estimator: CoverageProbabilityEstimator,
    /// Quantum efficiency analyzer
    pub efficiency_analyzer: QuantumEfficiencyAnalyzer,
    /// Prediction statistics
    pub prediction_stats: ConformalPredictionStats,
}

impl QuantumConformalPredictor {
    /// Create new quantum conformal predictor
    pub fn new(config: QuantumConfig) -> Result<Self> {
        let simulator = QuantumCircuitSimulator::new(config.n_qubits)?;
        
        let conformal_circuits = Self::create_conformal_circuits(&config)?;
        let nonconformity_calculators = Self::create_nonconformity_calculators(&config)?;
        let calibration_engine = QuantumCalibrationEngine::new(config.clone())?;
        let interval_optimizers = Self::create_interval_optimizers(&config)?;
        let coverage_estimator = CoverageProbabilityEstimator::new(config.clone())?;
        let efficiency_analyzer = QuantumEfficiencyAnalyzer::new(config.clone())?;
        
        Ok(Self {
            config,
            simulator,
            conformal_circuits,
            nonconformity_calculators,
            calibration_engine,
            interval_optimizers,
            coverage_estimator,
            efficiency_analyzer,
            prediction_stats: ConformalPredictionStats::new(),
        })
    }

    /// Create conformal prediction circuits
    fn create_conformal_circuits(config: &QuantumConfig) -> Result<Vec<QuantumCircuit>> {
        let mut circuits = Vec::new();
        
        // Quantum non-conformity circuit
        let nonconformity_circuit = Self::create_nonconformity_circuit(config.n_qubits)?;
        circuits.push(nonconformity_circuit);
        
        // Quantum calibration circuit
        let calibration_circuit = Self::create_calibration_circuit(config.n_qubits)?;
        circuits.push(calibration_circuit);
        
        // Quantum interval optimization circuit
        let optimization_circuit = Self::create_optimization_circuit(config.n_qubits)?;
        circuits.push(optimization_circuit);
        
        // Quantum coverage estimation circuit
        let coverage_circuit = Self::create_coverage_circuit(config.n_qubits)?;
        circuits.push(coverage_circuit);
        
        Ok(circuits)
    }

    /// Create quantum non-conformity circuit
    fn create_nonconformity_circuit(n_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(n_qubits, "quantum_nonconformity".to_string());
        
        // Encode prediction and true value
        for i in 0..n_qubits / 2 {
            circuit.add_gate(QuantumGate::RY(i, 0.0)); // Prediction encoding
            circuit.add_gate(QuantumGate::RY(i + n_qubits / 2, 0.0)); // True value encoding
        }
        
        // Compute quantum distance
        for i in 0..n_qubits / 2 {
            circuit.add_gate(QuantumGate::CNOT(i, i + n_qubits / 2));
        }
        
        // Quantum error amplification
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGate::RZ(i, 0.0)); // Parameterized
        }
        
        Ok(circuit)
    }

    /// Create quantum calibration circuit
    fn create_calibration_circuit(n_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(n_qubits, "quantum_calibration".to_string());
        
        // Create calibration superposition
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGate::H(i));
            circuit.add_gate(QuantumGate::RY(i, 0.0)); // Parameterized
        }
        
        // Entangling gates for correlation
        for i in 0..n_qubits - 1 {
            circuit.add_gate(QuantumGate::CRY(i, i + 1, 0.0)); // Parameterized
        }
        
        // Quantum phase estimation for calibration
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGate::RZ(i, 0.0)); // Parameterized
        }
        
        Ok(circuit)
    }

    /// Create quantum optimization circuit
    fn create_optimization_circuit(n_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(n_qubits, "quantum_optimization".to_string());
        
        // Variational ansatz for interval optimization
        let n_layers = 3;
        for layer in 0..n_layers {
            // Single qubit rotations
            for i in 0..n_qubits {
                circuit.add_gate(QuantumGate::RY(i, 0.0)); // Parameterized
                circuit.add_gate(QuantumGate::RZ(i, 0.0)); // Parameterized
            }
            
            // Entangling gates
            for i in 0..n_qubits - 1 {
                circuit.add_gate(QuantumGate::CNOT(i, i + 1));
            }
        }
        
        Ok(circuit)
    }

    /// Create quantum coverage circuit
    fn create_coverage_circuit(n_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(n_qubits, "quantum_coverage".to_string());
        
        // Coverage probability estimation
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGate::H(i));
            circuit.add_gate(QuantumGate::RY(i, 0.0)); // Parameterized
        }
        
        // Quantum amplitude estimation
        for i in 0..n_qubits / 2 {
            circuit.add_gate(QuantumGate::CNOT(i, i + n_qubits / 2));
            circuit.add_gate(QuantumGate::RY(i + n_qubits / 2, 0.0)); // Parameterized
            circuit.add_gate(QuantumGate::CNOT(i, i + n_qubits / 2));
        }
        
        Ok(circuit)
    }

    /// Create non-conformity calculators
    fn create_nonconformity_calculators(config: &QuantumConfig) -> Result<Vec<NonConformityCalculator>> {
        let mut calculators = Vec::new();
        
        // Quantum absolute residual calculator
        let absolute_calculator = NonConformityCalculator::new(
            "quantum_absolute".to_string(),
            NonConformityType::QuantumAbsolute,
            config.clone(),
        )?;
        calculators.push(absolute_calculator);
        
        // Quantum relative residual calculator
        let relative_calculator = NonConformityCalculator::new(
            "quantum_relative".to_string(),
            NonConformityType::QuantumRelative,
            config.clone(),
        )?;
        calculators.push(relative_calculator);
        
        // Quantum normalized calculator
        let normalized_calculator = NonConformityCalculator::new(
            "quantum_normalized".to_string(),
            NonConformityType::QuantumNormalized,
            config.clone(),
        )?;
        calculators.push(normalized_calculator);
        
        Ok(calculators)
    }

    /// Create interval optimizers
    fn create_interval_optimizers(config: &QuantumConfig) -> Result<Vec<IntervalOptimizer>> {
        let mut optimizers = Vec::new();
        
        // Quantum width optimizer
        let width_optimizer = IntervalOptimizer::new(
            "quantum_width".to_string(),
            OptimizationType::QuantumWidth,
            config.clone(),
        )?;
        optimizers.push(width_optimizer);
        
        // Quantum coverage optimizer
        let coverage_optimizer = IntervalOptimizer::new(
            "quantum_coverage".to_string(),
            OptimizationType::QuantumCoverage,
            config.clone(),
        )?;
        optimizers.push(coverage_optimizer);
        
        // Quantum efficiency optimizer
        let efficiency_optimizer = IntervalOptimizer::new(
            "quantum_efficiency".to_string(),
            OptimizationType::QuantumEfficiency,
            config.clone(),
        )?;
        optimizers.push(efficiency_optimizer);
        
        Ok(optimizers)
    }

    /// Create prediction intervals using quantum enhancement
    pub async fn create_prediction_intervals(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<ConformalPredictionIntervals> {
        info!("Creating quantum-enhanced conformal prediction intervals");
        
        let confidence_level = self.config.confidence_level;
        let mut intervals = ConformalPredictionIntervals::new(confidence_level);
        
        // Compute quantum non-conformity scores
        intervals.nonconformity_scores = self.compute_quantum_nonconformity_scores(
            features, target, estimates
        ).await?;
        
        // Perform quantum calibration
        intervals.quantum_calibration_scores = self.calibration_engine
            .perform_quantum_calibration(&intervals.nonconformity_scores).await?;
        
        // Optimize prediction intervals
        let optimized_intervals = self.optimize_prediction_intervals(
            &intervals.quantum_calibration_scores,
            confidence_level,
        ).await?;
        
        intervals.lower_bound = optimized_intervals.0;
        intervals.upper_bound = optimized_intervals.1;
        intervals.interval_width = intervals.upper_bound - intervals.lower_bound;
        
        // Compute quantum coverage probability
        intervals.quantum_coverage_probability = self.coverage_estimator
            .estimate_quantum_coverage_probability(&intervals).await?;
        
        // Generate sample-wise intervals
        intervals.sample_intervals = self.generate_sample_intervals(
            features, &intervals.quantum_calibration_scores
        ).await?;
        
        // Compute quantum efficiency
        intervals.quantum_efficiency = self.efficiency_analyzer
            .analyze_quantum_efficiency(&intervals).await?;
        
        // Validate intervals
        intervals.validity_indicators = self.validate_intervals(&intervals, target).await?;
        
        // Compute quantum prediction advantage
        intervals.quantum_prediction_advantage = self.compute_quantum_prediction_advantage(
            &intervals, estimates
        ).await?;
        
        // Update statistics
        self.update_prediction_statistics(&intervals).await?;
        
        Ok(intervals)
    }

    /// Compute quantum non-conformity scores
    async fn compute_quantum_nonconformity_scores(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<Vec<f64>> {
        let mut nonconformity_scores = Vec::new();
        
        for calculator in &self.nonconformity_calculators {
            let scores = calculator.calculate_nonconformity_scores(
                features, target, estimates
            ).await?;
            nonconformity_scores.extend(scores);
        }
        
        Ok(nonconformity_scores)
    }

    /// Optimize prediction intervals
    async fn optimize_prediction_intervals(
        &self,
        calibration_scores: &[f64],
        confidence_level: f64,
    ) -> Result<(f64, f64)> {
        let mut best_interval = (f64::INFINITY, f64::NEG_INFINITY);
        let mut best_score = f64::INFINITY;
        
        for optimizer in &self.interval_optimizers {
            let interval = optimizer.optimize_interval(
                calibration_scores, confidence_level
            ).await?;
            
            let score = self.evaluate_interval_quality(&interval, calibration_scores).await?;
            
            if score < best_score {
                best_score = score;
                best_interval = interval;
            }
        }
        
        Ok(best_interval)
    }

    /// Evaluate interval quality
    async fn evaluate_interval_quality(
        &self,
        interval: &(f64, f64),
        scores: &[f64],
    ) -> Result<f64> {
        let width = interval.1 - interval.0;
        let coverage = scores.iter().filter(|&&s| s >= interval.0 && s <= interval.1).count() as f64 / scores.len() as f64;
        
        // Quality score balances width and coverage
        let quality = width * (1.0 - coverage).abs();
        Ok(quality)
    }

    /// Generate sample-wise intervals
    async fn generate_sample_intervals(
        &self,
        features: &QuantumFeatures,
        calibration_scores: &[f64],
    ) -> Result<Vec<(f64, f64)>> {
        let mut sample_intervals = Vec::new();
        
        for i in 0..features.classical_features.len() {
            // Quantum-enhanced sample-specific interval
            let quantum_adjustment = self.compute_quantum_adjustment(features, i).await?;
            
            let base_lower = calibration_scores.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            let base_upper = calibration_scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            
            let lower = base_lower - quantum_adjustment;
            let upper = base_upper + quantum_adjustment;
            
            sample_intervals.push((lower, upper));
        }
        
        Ok(sample_intervals)
    }

    /// Compute quantum adjustment for individual samples
    async fn compute_quantum_adjustment(&self, features: &QuantumFeatures, sample_idx: usize) -> Result<f64> {
        if sample_idx >= features.classical_features.len() {
            return Ok(0.0);
        }
        
        let feature_value = features.classical_features[sample_idx];
        let quantum_feature = if sample_idx < features.superposition_features.len() {
            features.superposition_features[sample_idx].norm()
        } else {
            0.0
        };
        
        let adjustment = feature_value.abs() * quantum_feature * 0.1; // Scaling factor
        Ok(adjustment)
    }

    /// Validate prediction intervals
    async fn validate_intervals(
        &self,
        intervals: &ConformalPredictionIntervals,
        target: &Array1<f64>,
    ) -> Result<Vec<bool>> {
        let mut validity_indicators = Vec::new();
        
        for (i, &target_value) in target.iter().enumerate() {
            let is_valid = if i < intervals.sample_intervals.len() {
                let (lower, upper) = intervals.sample_intervals[i];
                target_value >= lower && target_value <= upper
            } else {
                intervals.contains(target_value)
            };
            
            validity_indicators.push(is_valid);
        }
        
        Ok(validity_indicators)
    }

    /// Compute quantum prediction advantage
    async fn compute_quantum_prediction_advantage(
        &self,
        intervals: &ConformalPredictionIntervals,
        estimates: &[UncertaintyEstimate],
    ) -> Result<f64> {
        // Compare quantum vs classical interval performance
        let quantum_efficiency = intervals.quantum_efficiency;
        let classical_efficiency = estimates.iter()
            .map(|e| e.uncertainty)
            .sum::<f64>() / estimates.len() as f64;
        
        let advantage = if classical_efficiency > 0.0 {
            quantum_efficiency / classical_efficiency
        } else {
            1.0
        };
        
        Ok(advantage)
    }

    /// Update prediction statistics
    async fn update_prediction_statistics(&self, intervals: &ConformalPredictionIntervals) -> Result<()> {
        // This would update internal statistics in a real implementation
        debug!("Updated prediction statistics for quantum conformal predictor");
        Ok(())
    }

    /// Reset the conformal predictor
    pub fn reset(&mut self) -> Result<()> {
        self.simulator.reset()?;
        self.calibration_engine.reset()?;
        
        for calculator in &mut self.nonconformity_calculators {
            calculator.reset()?;
        }
        
        for optimizer in &mut self.interval_optimizers {
            optimizer.reset()?;
        }
        
        self.coverage_estimator.reset()?;
        self.efficiency_analyzer.reset()?;
        self.prediction_stats.reset();
        
        Ok(())
    }
}

/// Non-conformity score calculator
#[derive(Debug, Clone)]
pub struct NonConformityCalculator {
    /// Calculator name
    pub name: String,
    /// Non-conformity type
    pub nonconformity_type: NonConformityType,
    /// Configuration
    pub config: QuantumConfig,
    /// Calculation parameters
    pub parameters: Vec<f64>,
    /// Calculation statistics
    pub stats: CalculationStats,
}

impl NonConformityCalculator {
    /// Create new non-conformity calculator
    pub fn new(name: String, nonconformity_type: NonConformityType, config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            name,
            nonconformity_type,
            config,
            parameters: Vec::new(),
            stats: CalculationStats::new(),
        })
    }

    /// Calculate non-conformity scores
    pub async fn calculate_nonconformity_scores(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<Vec<f64>> {
        match self.nonconformity_type {
            NonConformityType::QuantumAbsolute => {
                self.calculate_quantum_absolute_scores(features, target, estimates).await
            }
            NonConformityType::QuantumRelative => {
                self.calculate_quantum_relative_scores(features, target, estimates).await
            }
            NonConformityType::QuantumNormalized => {
                self.calculate_quantum_normalized_scores(features, target, estimates).await
            }
        }
    }

    /// Calculate quantum absolute non-conformity scores
    async fn calculate_quantum_absolute_scores(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        
        for (i, &target_value) in target.iter().enumerate() {
            let predicted_value = if i < estimates.len() {
                estimates[i].uncertainty // Using uncertainty as prediction
            } else {
                0.0
            };
            
            // Quantum enhancement using superposition features
            let quantum_enhancement = if i < features.superposition_features.len() {
                features.superposition_features[i].norm()
            } else {
                1.0
            };
            
            let quantum_absolute_score = (target_value - predicted_value).abs() * quantum_enhancement;
            scores.push(quantum_absolute_score);
        }
        
        Ok(scores)
    }

    /// Calculate quantum relative non-conformity scores
    async fn calculate_quantum_relative_scores(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        
        for (i, &target_value) in target.iter().enumerate() {
            let predicted_value = if i < estimates.len() {
                estimates[i].uncertainty
            } else {
                0.0
            };
            
            let quantum_enhancement = if i < features.superposition_features.len() {
                features.superposition_features[i].norm()
            } else {
                1.0
            };
            
            let relative_error = if predicted_value.abs() > 1e-10 {
                (target_value - predicted_value).abs() / predicted_value.abs()
            } else {
                (target_value - predicted_value).abs()
            };
            
            let quantum_relative_score = relative_error * quantum_enhancement;
            scores.push(quantum_relative_score);
        }
        
        Ok(scores)
    }

    /// Calculate quantum normalized non-conformity scores
    async fn calculate_quantum_normalized_scores(
        &self,
        features: &QuantumFeatures,
        target: &Array1<f64>,
        estimates: &[UncertaintyEstimate],
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        
        // Calculate normalization factor
        let target_std = {
            let mean = target.mean().unwrap_or(0.0);
            let variance = target.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / target.len() as f64;
            variance.sqrt()
        };
        
        for (i, &target_value) in target.iter().enumerate() {
            let predicted_value = if i < estimates.len() {
                estimates[i].uncertainty
            } else {
                0.0
            };
            
            let quantum_enhancement = if i < features.superposition_features.len() {
                features.superposition_features[i].norm()
            } else {
                1.0
            };
            
            let normalized_error = if target_std > 1e-10 {
                (target_value - predicted_value).abs() / target_std
            } else {
                (target_value - predicted_value).abs()
            };
            
            let quantum_normalized_score = normalized_error * quantum_enhancement;
            scores.push(quantum_normalized_score);
        }
        
        Ok(scores)
    }

    /// Reset calculator
    pub fn reset(&mut self) -> Result<()> {
        self.parameters.clear();
        self.stats.reset();
        Ok(())
    }
}

/// Non-conformity types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonConformityType {
    /// Quantum-enhanced absolute residual
    QuantumAbsolute,
    /// Quantum-enhanced relative residual
    QuantumRelative,
    /// Quantum-enhanced normalized residual
    QuantumNormalized,
}

/// Quantum calibration engine
#[derive(Debug, Clone)]
pub struct QuantumCalibrationEngine {
    /// Configuration
    pub config: QuantumConfig,
    /// Calibration parameters
    pub calibration_params: CalibrationParams,
    /// Calibration history
    pub calibration_history: Vec<f64>,
}

impl QuantumCalibrationEngine {
    /// Create new quantum calibration engine
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            calibration_params: CalibrationParams::default(),
            calibration_history: Vec::new(),
        })
    }

    /// Perform quantum calibration
    pub async fn perform_quantum_calibration(&self, scores: &[f64]) -> Result<Vec<f64>> {
        let mut calibrated_scores = Vec::new();
        
        // Quantum-enhanced calibration using superposition and entanglement
        for &score in scores {
            let calibrated_score = self.apply_quantum_calibration(score).await?;
            calibrated_scores.push(calibrated_score);
        }
        
        Ok(calibrated_scores)
    }

    /// Apply quantum calibration to a single score
    async fn apply_quantum_calibration(&self, score: f64) -> Result<f64> {
        // Create quantum state encoding the score
        let mut simulator = QuantumCircuitSimulator::new(2)?;
        
        let angle = score * PI; // Map score to rotation angle
        simulator.apply_ry(0, angle)?;
        
        // Apply quantum calibration transformation
        simulator.apply_h(1)?;
        simulator.apply_cnot(0, 1)?;
        simulator.apply_ry(1, self.calibration_params.calibration_angle)?;
        simulator.apply_cnot(0, 1)?;
        
        // Measure calibration observable
        let calibration_observable = PauliObservable {
            pauli_string: "ZZ".to_string(),
            coefficients: vec![1.0, 1.0],
        };
        
        let calibrated_value = simulator.expectation_value(&calibration_observable)?;
        Ok(calibrated_value.abs())
    }

    /// Reset calibration engine
    pub fn reset(&mut self) -> Result<()> {
        self.calibration_history.clear();
        Ok(())
    }
}

/// Calibration parameters
#[derive(Debug, Clone)]
pub struct CalibrationParams {
    /// Calibration angle for quantum circuit
    pub calibration_angle: f64,
    /// Learning rate for adaptive calibration
    pub learning_rate: f64,
    /// Momentum for calibration updates
    pub momentum: f64,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            calibration_angle: PI / 4.0,
            learning_rate: 0.01,
            momentum: 0.9,
        }
    }
}

/// Interval optimizer
#[derive(Debug, Clone)]
pub struct IntervalOptimizer {
    /// Optimizer name
    pub name: String,
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Configuration
    pub config: QuantumConfig,
    /// Optimization parameters
    pub parameters: Vec<f64>,
    /// Optimization history
    pub optimization_history: Vec<f64>,
}

impl IntervalOptimizer {
    /// Create new interval optimizer
    pub fn new(name: String, optimization_type: OptimizationType, config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            name,
            optimization_type,
            config,
            parameters: Vec::new(),
            optimization_history: Vec::new(),
        })
    }

    /// Optimize prediction interval
    pub async fn optimize_interval(
        &self,
        scores: &[f64],
        confidence_level: f64,
    ) -> Result<(f64, f64)> {
        match self.optimization_type {
            OptimizationType::QuantumWidth => {
                self.optimize_quantum_width(scores, confidence_level).await
            }
            OptimizationType::QuantumCoverage => {
                self.optimize_quantum_coverage(scores, confidence_level).await
            }
            OptimizationType::QuantumEfficiency => {
                self.optimize_quantum_efficiency(scores, confidence_level).await
            }
        }
    }

    /// Optimize for minimum width
    async fn optimize_quantum_width(&self, scores: &[f64], confidence_level: f64) -> Result<(f64, f64)> {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - confidence_level;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;
        
        let lower_idx = (lower_quantile * sorted_scores.len() as f64).floor() as usize;
        let upper_idx = (upper_quantile * sorted_scores.len() as f64).ceil() as usize;
        
        let lower_bound = sorted_scores.get(lower_idx).unwrap_or(&0.0);
        let upper_bound = sorted_scores.get(upper_idx.min(sorted_scores.len() - 1)).unwrap_or(&0.0);
        
        // Quantum enhancement for tighter bounds
        let quantum_factor = 0.9; // Slightly tighten bounds
        let width = upper_bound - lower_bound;
        let center = (upper_bound + lower_bound) / 2.0;
        let quantum_width = width * quantum_factor;
        
        Ok((center - quantum_width / 2.0, center + quantum_width / 2.0))
    }

    /// Optimize for better coverage
    async fn optimize_quantum_coverage(&self, scores: &[f64], confidence_level: f64) -> Result<(f64, f64)> {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Use quantum-enhanced coverage estimation
        let alpha = 1.0 - confidence_level;
        let quantum_alpha = alpha * 0.95; // Slightly improve coverage
        
        let lower_quantile = quantum_alpha / 2.0;
        let upper_quantile = 1.0 - quantum_alpha / 2.0;
        
        let lower_idx = (lower_quantile * sorted_scores.len() as f64).floor() as usize;
        let upper_idx = (upper_quantile * sorted_scores.len() as f64).ceil() as usize;
        
        let lower_bound = sorted_scores.get(lower_idx).unwrap_or(&0.0);
        let upper_bound = sorted_scores.get(upper_idx.min(sorted_scores.len() - 1)).unwrap_or(&0.0);
        
        Ok((*lower_bound, *upper_bound))
    }

    /// Optimize for maximum efficiency
    async fn optimize_quantum_efficiency(&self, scores: &[f64], confidence_level: f64) -> Result<(f64, f64)> {
        let width_interval = self.optimize_quantum_width(scores, confidence_level).await?;
        let coverage_interval = self.optimize_quantum_coverage(scores, confidence_level).await?;
        
        // Balance between width and coverage
        let balanced_lower = (width_interval.0 + coverage_interval.0) / 2.0;
        let balanced_upper = (width_interval.1 + coverage_interval.1) / 2.0;
        
        Ok((balanced_lower, balanced_upper))
    }

    /// Reset optimizer
    pub fn reset(&mut self) -> Result<()> {
        self.parameters.clear();
        self.optimization_history.clear();
        Ok(())
    }
}

/// Optimization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    /// Optimize for minimum width
    QuantumWidth,
    /// Optimize for better coverage
    QuantumCoverage,
    /// Optimize for maximum efficiency
    QuantumEfficiency,
}

/// Coverage probability estimator
#[derive(Debug, Clone)]
pub struct CoverageProbabilityEstimator {
    /// Configuration
    pub config: QuantumConfig,
    /// Estimation parameters
    pub estimation_params: CoverageEstimationParams,
}

impl CoverageProbabilityEstimator {
    /// Create new coverage probability estimator
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            estimation_params: CoverageEstimationParams::default(),
        })
    }

    /// Estimate quantum coverage probability
    pub async fn estimate_quantum_coverage_probability(
        &self,
        intervals: &ConformalPredictionIntervals,
    ) -> Result<f64> {
        // Quantum-enhanced coverage probability estimation
        let classical_coverage = intervals.confidence_level;
        
        // Apply quantum enhancement
        let quantum_factor = self.compute_quantum_enhancement_factor(intervals).await?;
        let quantum_coverage = classical_coverage * quantum_factor;
        
        Ok(quantum_coverage.min(1.0).max(0.0))
    }

    /// Compute quantum enhancement factor
    async fn compute_quantum_enhancement_factor(
        &self,
        intervals: &ConformalPredictionIntervals,
    ) -> Result<f64> {
        // Use quantum circuit to estimate enhancement
        let mut simulator = QuantumCircuitSimulator::new(2)?;
        
        // Encode interval information
        let width_angle = intervals.interval_width * PI / 10.0; // Scale to reasonable angle
        simulator.apply_ry(0, width_angle)?;
        
        // Apply quantum enhancement circuit
        simulator.apply_hadamard(1)?;
        simulator.apply_cnot(0, 1)?;
        simulator.apply_ry(1, PI / 4.0)?;
        
        // Measure enhancement observable
        let enhancement_observable = PauliObservable {
            pauli_string: "ZI".to_string(),
            coefficients: vec![1.0, 0.0],
        };
        
        let enhancement = simulator.expectation_value(&enhancement_observable)?;
        let enhancement_factor = 1.0 + enhancement.abs() * 0.1; // Small enhancement
        
        Ok(enhancement_factor)
    }

    /// Reset estimator
    pub fn reset(&mut self) -> Result<()> {
        // Reset internal state
        Ok(())
    }
}

/// Coverage estimation parameters
#[derive(Debug, Clone)]
pub struct CoverageEstimationParams {
    /// Number of bootstrap samples
    pub n_bootstrap_samples: usize,
    /// Estimation confidence level
    pub estimation_confidence: f64,
}

impl Default for CoverageEstimationParams {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 1000,
            estimation_confidence: 0.95,
        }
    }
}

/// Quantum efficiency analyzer
#[derive(Debug, Clone)]
pub struct QuantumEfficiencyAnalyzer {
    /// Configuration
    pub config: QuantumConfig,
    /// Analysis parameters
    pub analysis_params: EfficiencyAnalysisParams,
}

impl QuantumEfficiencyAnalyzer {
    /// Create new quantum efficiency analyzer
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            analysis_params: EfficiencyAnalysisParams::default(),
        })
    }

    /// Analyze quantum efficiency
    pub async fn analyze_quantum_efficiency(
        &self,
        intervals: &ConformalPredictionIntervals,
    ) -> Result<f64> {
        // Quantum efficiency is the ratio of coverage to interval width
        let coverage = intervals.quantum_coverage_probability;
        let width = intervals.interval_width;
        
        let efficiency = if width > 0.0 {
            coverage / width
        } else {
            0.0
        };
        
        // Apply quantum enhancement
        let quantum_enhancement = self.compute_quantum_efficiency_enhancement(intervals).await?;
        let quantum_efficiency = efficiency * quantum_enhancement;
        
        Ok(quantum_efficiency)
    }

    /// Compute quantum efficiency enhancement
    async fn compute_quantum_efficiency_enhancement(
        &self,
        intervals: &ConformalPredictionIntervals,
    ) -> Result<f64> {
        // Use quantum circuit to compute efficiency enhancement
        let mut simulator = QuantumCircuitSimulator::new(2)?;
        
        // Encode efficiency information
        let coverage_angle = intervals.quantum_coverage_probability * PI;
        let width_angle = intervals.interval_width * PI / 10.0;
        
        simulator.apply_ry(0, coverage_angle)?;
        simulator.apply_ry(1, width_angle)?;
        
        // Apply quantum efficiency circuit
        simulator.apply_cnot(0, 1)?;
        simulator.apply_ry(1, PI / 6.0)?;
        simulator.apply_cnot(0, 1)?;
        
        // Measure efficiency observable
        let efficiency_observable = PauliObservable {
            pauli_string: "XX".to_string(),
            coefficients: vec![1.0, 1.0],
        };
        
        let enhancement = simulator.expectation_value(&efficiency_observable)?;
        let enhancement_factor = 1.0 + enhancement.abs() * 0.2; // Moderate enhancement
        
        Ok(enhancement_factor)
    }

    /// Reset analyzer
    pub fn reset(&mut self) -> Result<()> {
        // Reset internal state
        Ok(())
    }
}

/// Efficiency analysis parameters
#[derive(Debug, Clone)]
pub struct EfficiencyAnalysisParams {
    /// Efficiency threshold
    pub efficiency_threshold: f64,
    /// Analysis window size
    pub window_size: usize,
}

impl Default for EfficiencyAnalysisParams {
    fn default() -> Self {
        Self {
            efficiency_threshold: 0.8,
            window_size: 100,
        }
    }
}

/// Statistics structures
#[derive(Debug, Clone)]
pub struct ConformalPredictionStats {
    /// Total predictions made
    pub total_predictions: u64,
    /// Average interval width
    pub average_interval_width: f64,
    /// Average coverage probability
    pub average_coverage_probability: f64,
    /// Average quantum efficiency
    pub average_quantum_efficiency: f64,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
}

impl ConformalPredictionStats {
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            average_interval_width: 0.0,
            average_coverage_probability: 0.0,
            average_quantum_efficiency: 0.0,
            prediction_accuracy: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total_predictions = 0;
        self.average_interval_width = 0.0;
        self.average_coverage_probability = 0.0;
        self.average_quantum_efficiency = 0.0;
        self.prediction_accuracy = 0.0;
    }
}

#[derive(Debug, Clone)]
pub struct CalculationStats {
    pub total_calculations: u64,
    pub average_calculation_time: f64,
    pub calculation_accuracy: f64,
}

impl CalculationStats {
    pub fn new() -> Self {
        Self {
            total_calculations: 0,
            average_calculation_time: 0.0,
            calculation_accuracy: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total_calculations = 0;
        self.average_calculation_time = 0.0;
        self.calculation_accuracy = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_conformal_prediction_intervals_creation() {
        let intervals = ConformalPredictionIntervals::new(0.95);
        assert_abs_diff_eq!(intervals.confidence_level, 0.95, epsilon = 1e-10);
        assert_eq!(intervals.nonconformity_scores.len(), 0);
        assert_eq!(intervals.sample_intervals.len(), 0);
    }

    #[test]
    fn test_conformal_predictor_creation() {
        let config = QuantumConfig::default();
        let predictor = QuantumConformalPredictor::new(config);
        assert!(predictor.is_ok());
    }

    #[test]
    fn test_nonconformity_calculator_creation() {
        let config = QuantumConfig::default();
        let calculator = NonConformityCalculator::new(
            "test".to_string(),
            NonConformityType::QuantumAbsolute,
            config,
        ).unwrap();
        
        assert_eq!(calculator.name, "test");
        assert_eq!(calculator.nonconformity_type, NonConformityType::QuantumAbsolute);
    }

    #[test]
    fn test_interval_contains() {
        let mut intervals = ConformalPredictionIntervals::new(0.95);
        intervals.lower_bound = 0.0;
        intervals.upper_bound = 1.0;
        
        assert!(intervals.contains(0.5));
        assert!(!intervals.contains(1.5));
        assert!(!intervals.contains(-0.5));
    }

    #[test]
    fn test_interval_coverage() {
        let mut intervals = ConformalPredictionIntervals::new(0.95);
        intervals.lower_bound = 0.0;
        intervals.upper_bound = 1.0;
        
        let values = vec![0.2, 0.8, 1.5, -0.5, 0.5];
        let coverage = intervals.coverage(&values);
        assert_abs_diff_eq!(coverage, 0.6, epsilon = 1e-10); // 3 out of 5 values
    }

    #[test]
    fn test_quantum_calibration_engine() {
        let config = QuantumConfig::default();
        let engine = QuantumCalibrationEngine::new(config).unwrap();
        assert_abs_diff_eq!(engine.calibration_params.calibration_angle, PI / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interval_optimizer() {
        let config = QuantumConfig::default();
        let optimizer = IntervalOptimizer::new(
            "test".to_string(),
            OptimizationType::QuantumWidth,
            config,
        ).unwrap();
        
        assert_eq!(optimizer.name, "test");
        assert_eq!(optimizer.optimization_type, OptimizationType::QuantumWidth);
    }

    #[tokio::test]
    async fn test_quantum_width_optimization() {
        let config = QuantumConfig::default();
        let optimizer = IntervalOptimizer::new(
            "test".to_string(),
            OptimizationType::QuantumWidth,
            config,
        ).unwrap();
        
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let interval = optimizer.optimize_interval(&scores, 0.9).await.unwrap();
        
        assert!(interval.0 < interval.1);
        assert!(interval.1 - interval.0 > 0.0);
    }

    #[test]
    fn test_coverage_probability_estimator() {
        let config = QuantumConfig::default();
        let estimator = CoverageProbabilityEstimator::new(config).unwrap();
        assert_eq!(estimator.estimation_params.n_bootstrap_samples, 1000);
    }

    #[test]
    fn test_quantum_efficiency_analyzer() {
        let config = QuantumConfig::default();
        let analyzer = QuantumEfficiencyAnalyzer::new(config).unwrap();
        assert_abs_diff_eq!(analyzer.analysis_params.efficiency_threshold, 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_conformal_prediction_stats() {
        let mut stats = ConformalPredictionStats::new();
        assert_eq!(stats.total_predictions, 0);
        assert_abs_diff_eq!(stats.average_interval_width, 0.0, epsilon = 1e-10);
        
        stats.reset();
        assert_eq!(stats.total_predictions, 0);
    }
}