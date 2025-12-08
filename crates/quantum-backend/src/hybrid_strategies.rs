//! Quantum-Classical Hybrid Trading Strategies
//! 
//! Combines quantum optimization with classical machine learning
//! for advanced trading strategies.

use crate::{error::Result, types::*, QuantumBackend};
use quantum_core::{QuantumCircuit, QuantumState};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};

/// Hybrid strategy configuration
#[derive(Debug, Clone)]
pub struct HybridStrategyConfig {
    pub quantum_layers: usize,
    pub classical_layers: usize,
    pub feature_dimension: usize,
    pub quantum_encoding: EncodingType,
    pub optimization_objective: OptimizationObjective,
}

/// Quantum encoding types
#[derive(Debug, Clone)]
pub enum EncodingType {
    AmplitudeEncoding,
    AngleEncoding,
    BasisEncoding,
    IQPEncoding,
}

/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MaximizeSharpe,
    MinimizeRisk,
    MaximizeReturn,
    MinimizeDrawdown,
    CustomObjective(String),
}

/// Hybrid quantum-classical model
pub struct HybridTradingModel {
    quantum_backend: Arc<QuantumBackend>,
    config: HybridStrategyConfig,
    quantum_parameters: RwLock<Vec<f64>>,
    classical_parameters: RwLock<Array2<f64>>,
    performance_cache: RwLock<PerformanceCache>,
}

/// Performance tracking
struct PerformanceCache {
    sharpe_ratio: f64,
    max_drawdown: f64,
    total_return: f64,
    win_rate: f64,
    recent_predictions: Vec<TradingSignal>,
}

/// Trading signal from hybrid model
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub timestamp: u64,
    pub action: TradeAction,
    pub confidence: f64,
    pub quantum_contribution: f64,
    pub classical_contribution: f64,
    pub expected_return: f64,
    pub risk_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
}

impl HybridTradingModel {
    /// Create new hybrid trading model
    pub async fn new(
        quantum_backend: Arc<QuantumBackend>,
        config: HybridStrategyConfig,
    ) -> Result<Self> {
        let num_quantum_params = config.quantum_layers * config.feature_dimension * 2;
        let quantum_parameters = Self::initialize_quantum_params(num_quantum_params);
        
        let classical_shape = (config.classical_layers, config.feature_dimension);
        let classical_parameters = Self::initialize_classical_params(classical_shape);
        
        Ok(Self {
            quantum_backend,
            config,
            quantum_parameters: RwLock::new(quantum_parameters),
            classical_parameters: RwLock::new(classical_parameters),
            performance_cache: RwLock::new(PerformanceCache {
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                total_return: 0.0,
                win_rate: 0.0,
                recent_predictions: Vec::new(),
            }),
        })
    }
    
    /// Generate trading signal from market features
    pub async fn predict(
        &self,
        market_features: &Array1<f64>,
        market_context: &MarketContext,
    ) -> Result<TradingSignal> {
        let start = std::time::Instant::now();
        
        // Quantum feature processing
        let quantum_features = self.quantum_feature_extraction(market_features).await?;
        
        // Classical feature processing
        let classical_features = self.classical_feature_extraction(market_features)?;
        
        // Combine quantum and classical insights
        let combined_signal = self.combine_predictions(
            &quantum_features,
            &classical_features,
            market_context,
        )?;
        
        // Risk assessment using quantum algorithms
        let risk_score = self.quantum_risk_assessment(&quantum_features, market_context).await?;
        
        // Final trading decision
        let signal = self.make_trading_decision(combined_signal, risk_score, market_context)?;
        
        debug!("Hybrid prediction took {:?}", start.elapsed());
        
        // Cache performance
        self.update_performance_cache(&signal);
        
        Ok(signal)
    }
    
    /// Quantum feature extraction using variational circuit
    async fn quantum_feature_extraction(
        &self,
        features: &Array1<f64>,
    ) -> Result<QuantumFeatures> {
        let num_qubits = (features.len() as f64).log2().ceil() as usize;
        let mut circuit = QuantumCircuit::new(num_qubits);
        
        // Encode classical features into quantum state
        self.encode_features(&mut circuit, features)?;
        
        // Apply variational quantum circuit
        let params = self.quantum_parameters.read();
        self.apply_variational_layers(&mut circuit, &params)?;
        
        // Execute on quantum backend
        let result = self.quantum_backend.execute_circuit(&circuit).await?;
        
        // Extract quantum features from final state
        Ok(QuantumFeatures {
            amplitudes: result.state.amplitudes().to_vec(),
            probabilities: result.probabilities,
            entanglement_entropy: self.calculate_entanglement_entropy(&result.state)?,
            quantum_advantage_score: self.estimate_quantum_advantage(&circuit)?,
        })
    }
    
    /// Classical feature extraction
    fn classical_feature_extraction(&self, features: &Array1<f64>) -> Result<ClassicalFeatures> {
        let params = self.classical_parameters.read();
        
        // Apply classical neural network layers
        let mut activations = features.clone();
        
        for layer_weights in params.outer_iter() {
            activations = self.apply_classical_layer(&activations, &layer_weights)?;
        }
        
        Ok(ClassicalFeatures {
            processed_features: activations,
            feature_importance: self.calculate_feature_importance(features)?,
        })
    }
    
    /// Encode features into quantum circuit
    fn encode_features(&self, circuit: &mut QuantumCircuit, features: &Array1<f64>) -> Result<()> {
        match &self.config.quantum_encoding {
            EncodingType::AmplitudeEncoding => {
                // Normalize features
                let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
                let normalized: Vec<f64> = features.iter().map(|x| x / norm).collect();
                
                // Create superposition based on features
                for (i, &feat) in normalized.iter().enumerate() {
                    if i < circuit.num_qubits() {
                        circuit.add_gate(quantum_core::QuantumGate::ry(i, feat.acos() * 2.0));
                    }
                }
            }
            EncodingType::AngleEncoding => {
                // Encode features as rotation angles
                for (i, &feat) in features.iter().enumerate() {
                    if i < circuit.num_qubits() {
                        circuit.add_gate(quantum_core::QuantumGate::rx(i, feat));
                        circuit.add_gate(quantum_core::QuantumGate::rz(i, feat));
                    }
                }
            }
            EncodingType::IQPEncoding => {
                // IQP (Instantaneous Quantum Polynomial) encoding
                // Initial Hadamard layer
                for i in 0..circuit.num_qubits() {
                    circuit.add_gate(quantum_core::QuantumGate::hadamard(i));
                }
                
                // Feature-dependent diagonal gates
                for (i, &feat) in features.iter().enumerate() {
                    if i < circuit.num_qubits() - 1 {
                        // Two-qubit interactions
                        circuit.add_gate(quantum_core::QuantumGate::rz(i, feat));
                        circuit.add_gate(quantum_core::QuantumGate::cnot(i, i + 1));
                        circuit.add_gate(quantum_core::QuantumGate::rz(i + 1, feat));
                        circuit.add_gate(quantum_core::QuantumGate::cnot(i, i + 1));
                    }
                }
            }
            _ => {
                // Default encoding
                for (i, &feat) in features.iter().enumerate() {
                    if i < circuit.num_qubits() {
                        circuit.add_gate(quantum_core::QuantumGate::ry(i, feat));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply variational quantum layers
    fn apply_variational_layers(
        &self,
        circuit: &mut QuantumCircuit,
        params: &[f64],
    ) -> Result<()> {
        let mut param_idx = 0;
        
        for layer in 0..self.config.quantum_layers {
            // Parameterized rotations
            for q in 0..circuit.num_qubits() {
                if param_idx < params.len() {
                    circuit.add_gate(quantum_core::QuantumGate::ry(q, params[param_idx]));
                    param_idx += 1;
                }
                if param_idx < params.len() {
                    circuit.add_gate(quantum_core::QuantumGate::rz(q, params[param_idx]));
                    param_idx += 1;
                }
            }
            
            // Entangling layer
            for q in 0..circuit.num_qubits() - 1 {
                circuit.add_gate(quantum_core::QuantumGate::cnot(q, q + 1));
            }
            
            // Add controlled rotations for feature interactions
            if layer < self.config.quantum_layers - 1 {
                for q in (0..circuit.num_qubits() - 1).step_by(2) {
                    circuit.add_gate(quantum_core::QuantumGate::cz(q, q + 1));
                }
            }
        }
        
        Ok(())
    }
    
    /// Quantum risk assessment
    async fn quantum_risk_assessment(
        &self,
        quantum_features: &QuantumFeatures,
        context: &MarketContext,
    ) -> Result<f64> {
        // Use quantum algorithms for portfolio risk optimization
        let risk_hamiltonian = self.construct_risk_hamiltonian(context)?;
        
        // Run VQE to find minimum risk configuration
        let ansatz = quantum_core::QuantumAnsatz {
            ansatz_type: quantum_core::AnsatzType::HardwareEfficient,
            num_qubits: 4,
            depth: 2,
            entanglement: quantum_core::EntanglementType::Linear,
        };
        
        let vqe_result = self.quantum_backend.run_vqe(&risk_hamiltonian, &ansatz).await?;
        
        // Convert VQE result to risk score
        let base_risk = (-vqe_result.energy + 1.0) / 2.0; // Normalize to [0, 1]
        
        // Adjust based on quantum features
        let quantum_adjustment = quantum_features.entanglement_entropy * 0.1;
        
        Ok((base_risk + quantum_adjustment).clamp(0.0, 1.0))
    }
    
    /// Construct risk Hamiltonian
    fn construct_risk_hamiltonian(&self, context: &MarketContext) -> Result<quantum_core::QuantumHamiltonian> {
        let mut terms = vec![];
        
        // Volatility term
        let volatility_weight = context.volatility * 2.0;
        terms.push((
            volatility_weight,
            vec![quantum_core::PauliOperator::Z, quantum_core::PauliOperator::I],
        ));
        
        // Correlation term
        if context.correlation.abs() > 0.5 {
            terms.push((
                context.correlation,
                vec![quantum_core::PauliOperator::Z, quantum_core::PauliOperator::Z],
            ));
        }
        
        // Market regime term
        let regime_weight = match context.market_regime {
            MarketRegime::Trending => 0.3,
            MarketRegime::Ranging => 0.5,
            MarketRegime::Volatile => 0.8,
        };
        terms.push((
            regime_weight,
            vec![quantum_core::PauliOperator::X, quantum_core::PauliOperator::I],
        ));
        
        Ok(quantum_core::QuantumHamiltonian {
            terms,
            num_qubits: 2,
        })
    }
    
    /// Combine quantum and classical predictions
    fn combine_predictions(
        &self,
        quantum_features: &QuantumFeatures,
        classical_features: &ClassicalFeatures,
        context: &MarketContext,
    ) -> Result<CombinedSignal> {
        // Extract quantum signal strength
        let quantum_signal = quantum_features.probabilities.iter()
            .enumerate()
            .map(|(i, &p)| p * (i as f64 / quantum_features.probabilities.len() as f64))
            .sum::<f64>();
        
        // Extract classical signal strength
        let classical_signal = classical_features.processed_features.mean().unwrap_or(0.5);
        
        // Adaptive weighting based on market conditions
        let (quantum_weight, classical_weight) = self.calculate_adaptive_weights(context);
        
        let combined_strength = quantum_signal * quantum_weight + classical_signal * classical_weight;
        
        Ok(CombinedSignal {
            strength: combined_strength,
            quantum_contribution: quantum_signal * quantum_weight,
            classical_contribution: classical_signal * classical_weight,
            confidence: self.calculate_confidence(quantum_features, classical_features)?,
        })
    }
    
    /// Calculate adaptive weights for quantum/classical combination
    fn calculate_adaptive_weights(&self, context: &MarketContext) -> (f64, f64) {
        let performance = self.performance_cache.read();
        
        // Increase quantum weight in volatile/complex markets
        let base_quantum_weight = match context.market_regime {
            MarketRegime::Volatile => 0.7,
            MarketRegime::Trending => 0.4,
            MarketRegime::Ranging => 0.5,
        };
        
        // Adjust based on recent performance
        let performance_adjustment = if performance.sharpe_ratio > 1.5 {
            0.1
        } else if performance.sharpe_ratio < 0.5 {
            -0.1
        } else {
            0.0
        };
        
        let quantum_weight = (base_quantum_weight + performance_adjustment).clamp(0.2, 0.8);
        let classical_weight = 1.0 - quantum_weight;
        
        (quantum_weight, classical_weight)
    }
    
    /// Make final trading decision
    fn make_trading_decision(
        &self,
        signal: CombinedSignal,
        risk_score: f64,
        context: &MarketContext,
    ) -> Result<TradingSignal> {
        // Determine action based on signal strength and risk
        let risk_adjusted_signal = signal.strength * (1.0 - risk_score * 0.5);
        
        let action = if risk_adjusted_signal > 0.7 && risk_score < 0.3 {
            TradeAction::StrongBuy
        } else if risk_adjusted_signal > 0.55 {
            TradeAction::Buy
        } else if risk_adjusted_signal < 0.3 && risk_score > 0.7 {
            TradeAction::StrongSell
        } else if risk_adjusted_signal < 0.45 {
            TradeAction::Sell
        } else {
            TradeAction::Hold
        };
        
        // Calculate expected return
        let expected_return = self.calculate_expected_return(
            risk_adjusted_signal,
            context.volatility,
        );
        
        Ok(TradingSignal {
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            action,
            confidence: signal.confidence,
            quantum_contribution: signal.quantum_contribution,
            classical_contribution: signal.classical_contribution,
            expected_return,
            risk_score,
        })
    }
    
    /// Calculate expected return
    fn calculate_expected_return(&self, signal: f64, volatility: f64) -> f64 {
        let base_return = (signal - 0.5) * 0.02; // 2% max base return
        let volatility_adjustment = volatility * 0.5;
        
        base_return * (1.0 + volatility_adjustment)
    }
    
    /// Update performance cache
    fn update_performance_cache(&self, signal: &TradingSignal) {
        let mut cache = self.performance_cache.write();
        
        cache.recent_predictions.push(signal.clone());
        if cache.recent_predictions.len() > 100 {
            cache.recent_predictions.remove(0);
        }
        
        // Update performance metrics
        // This is simplified - real implementation would track actual returns
        if signal.action != TradeAction::Hold {
            let wins = cache.recent_predictions.iter()
                .filter(|s| s.expected_return > 0.0)
                .count();
            cache.win_rate = wins as f64 / cache.recent_predictions.len() as f64;
        }
    }
    
    /// Calculate entanglement entropy
    fn calculate_entanglement_entropy(&self, state: &QuantumState) -> Result<f64> {
        // Simplified entanglement entropy calculation
        let probs = state.probabilities();
        let entropy = -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.log2())
            .sum::<f64>();
        
        Ok(entropy)
    }
    
    /// Estimate quantum advantage
    fn estimate_quantum_advantage(&self, circuit: &QuantumCircuit) -> Result<f64> {
        // Estimate based on circuit properties
        let depth = circuit.depth();
        let entangling_gates = circuit.gates().iter()
            .filter(|g| g.control().is_some())
            .count();
        
        let advantage = (entangling_gates as f64 / depth as f64).min(1.0);
        Ok(advantage)
    }
    
    /// Calculate confidence score
    fn calculate_confidence(
        &self,
        quantum_features: &QuantumFeatures,
        classical_features: &ClassicalFeatures,
    ) -> Result<f64> {
        let quantum_confidence = 1.0 - quantum_features.entanglement_entropy / quantum_features.probabilities.len() as f64;
        let classical_confidence = classical_features.feature_importance.mean().unwrap_or(0.5);
        
        Ok((quantum_confidence + classical_confidence) / 2.0)
    }
    
    /// Apply classical layer
    fn apply_classical_layer(
        &self,
        input: &Array1<f64>,
        weights: &ndarray::ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        // Simple linear transformation with ReLU
        let output = input.iter()
            .zip(weights.iter())
            .map(|(x, w)| (x * w).max(0.0)) // ReLU activation
            .collect();
        
        Ok(Array1::from_vec(output))
    }
    
    /// Calculate feature importance
    fn calculate_feature_importance(&self, features: &Array1<f64>) -> Result<Array1<f64>> {
        // Simple variance-based importance
        let mean = features.mean().unwrap_or(0.0);
        let importance = features.mapv(|x| (x - mean).abs());
        let sum = importance.sum();
        
        if sum > 0.0 {
            Ok(importance / sum)
        } else {
            Ok(Array1::ones(features.len()) / features.len() as f64)
        }
    }
    
    /// Initialize quantum parameters
    fn initialize_quantum_params(num_params: usize) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..num_params)
            .map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI))
            .collect()
    }
    
    /// Initialize classical parameters
    fn initialize_classical_params(shape: (usize, usize)) -> Array2<f64> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        Array2::from_shape_fn(shape, |_| normal.sample(&mut rng))
    }
    
    /// Train the hybrid model
    pub async fn train(
        &self,
        training_data: &TradingDataset,
        epochs: usize,
    ) -> Result<TrainingMetrics> {
        info!("Training hybrid quantum-classical model for {} epochs", epochs);
        
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut training_history = Vec::new();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut predictions = Vec::new();
            
            // Mini-batch training
            for batch in training_data.batches(32) {
                // Forward pass
                let mut batch_predictions = Vec::new();
                for (features, label) in batch {
                    let signal = self.predict(features, &label.context).await?;
                    batch_predictions.push((signal, label));
                }
                
                // Calculate loss
                let loss = self.calculate_loss(&batch_predictions)?;
                epoch_loss += loss;
                
                // Backward pass - update parameters
                self.update_parameters(&batch_predictions, loss).await?;
                
                predictions.extend(batch_predictions);
            }
            
            // Evaluate epoch performance
            let metrics = self.evaluate_predictions(&predictions)?;
            
            if metrics.sharpe_ratio > best_sharpe {
                best_sharpe = metrics.sharpe_ratio;
                self.save_best_parameters()?;
            }
            
            training_history.push(metrics.clone());
            
            if epoch % 10 == 0 {
                info!("Epoch {}: Loss = {:.4}, Sharpe = {:.2}", 
                      epoch, epoch_loss, metrics.sharpe_ratio);
            }
        }
        
        Ok(TrainingMetrics {
            final_sharpe: best_sharpe,
            training_history,
            quantum_contribution_avg: self.calculate_avg_quantum_contribution()?,
        })
    }
    
    /// Calculate loss for predictions
    fn calculate_loss(&self, predictions: &[(TradingSignal, TradingLabel)]) -> Result<f64> {
        let mut total_loss = 0.0;
        
        for (signal, label) in predictions {
            // Prediction error
            let return_error = (signal.expected_return - label.actual_return).powi(2);
            
            // Risk penalty
            let risk_penalty = if label.actual_return < 0.0 {
                signal.risk_score * label.actual_return.abs()
            } else {
                0.0
            };
            
            // Action accuracy
            let action_loss = if signal.action != label.optimal_action {
                1.0 - signal.confidence
            } else {
                0.0
            };
            
            total_loss += return_error + risk_penalty + action_loss;
        }
        
        Ok(total_loss / predictions.len() as f64)
    }
    
    /// Update model parameters
    async fn update_parameters(
        &self,
        predictions: &[(TradingSignal, TradingLabel)],
        loss: f64,
    ) -> Result<()> {
        let learning_rate = 0.01;
        
        // Update quantum parameters using parameter shift rule
        let mut quantum_params = self.quantum_parameters.write();
        for i in 0..quantum_params.len() {
            // Estimate gradient using finite differences
            let gradient = self.estimate_quantum_gradient(i, predictions).await?;
            quantum_params[i] -= learning_rate * gradient;
        }
        
        // Update classical parameters
        let mut classical_params = self.classical_parameters.write();
        for ((i, j), param) in classical_params.indexed_iter_mut() {
            // Simple gradient approximation
            let gradient = loss * 0.1; // Simplified
            *param -= learning_rate * gradient;
        }
        
        Ok(())
    }
    
    /// Estimate quantum gradient
    async fn estimate_quantum_gradient(
        &self,
        param_idx: usize,
        predictions: &[(TradingSignal, TradingLabel)],
    ) -> Result<f64> {
        // Simplified gradient estimation
        // Real implementation would use parameter shift rule
        Ok(0.01)
    }
    
    /// Evaluate predictions
    fn evaluate_predictions(&self, predictions: &[(TradingSignal, TradingLabel)]) -> Result<PerformanceMetrics> {
        let returns: Vec<f64> = predictions.iter()
            .map(|(_, label)| label.actual_return)
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = (returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64)
            .sqrt();
        
        let sharpe_ratio = if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };
        
        let accuracy = predictions.iter()
            .filter(|(signal, label)| signal.action == label.optimal_action)
            .count() as f64 / predictions.len() as f64;
        
        Ok(PerformanceMetrics {
            sharpe_ratio,
            accuracy,
            mean_return,
            max_drawdown: self.calculate_max_drawdown(&returns),
        })
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        let mut cumulative = 0.0;
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        
        for &ret in returns {
            cumulative += ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak.max(1e-10);
            if dd > max_dd {
                max_dd = dd;
            }
        }
        
        max_dd
    }
    
    /// Save best parameters
    fn save_best_parameters(&self) -> Result<()> {
        // Implementation would save to disk
        Ok(())
    }
    
    /// Calculate average quantum contribution
    fn calculate_avg_quantum_contribution(&self) -> Result<f64> {
        let cache = self.performance_cache.read();
        
        if cache.recent_predictions.is_empty() {
            return Ok(0.0);
        }
        
        let total_quantum = cache.recent_predictions.iter()
            .map(|p| p.quantum_contribution)
            .sum::<f64>();
        
        Ok(total_quantum / cache.recent_predictions.len() as f64)
    }
}

// Supporting structures

#[derive(Debug, Clone)]
struct QuantumFeatures {
    amplitudes: Vec<quantum_core::ComplexAmplitude>,
    probabilities: Vec<f64>,
    entanglement_entropy: f64,
    quantum_advantage_score: f64,
}

#[derive(Debug, Clone)]
struct ClassicalFeatures {
    processed_features: Array1<f64>,
    feature_importance: Array1<f64>,
}

#[derive(Debug, Clone)]
struct CombinedSignal {
    strength: f64,
    quantum_contribution: f64,
    classical_contribution: f64,
    confidence: f64,
}

#[derive(Debug, Clone)]
pub struct MarketContext {
    pub volatility: f64,
    pub correlation: f64,
    pub market_regime: MarketRegime,
    pub volume_profile: VolumeProfile,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct VolumeProfile {
    pub current: f64,
    pub average: f64,
    pub trend: f64,
}

#[derive(Debug, Clone)]
pub struct TradingDataset {
    features: Vec<Array1<f64>>,
    labels: Vec<TradingLabel>,
}

#[derive(Debug, Clone)]
pub struct TradingLabel {
    actual_return: f64,
    optimal_action: TradeAction,
    context: MarketContext,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    sharpe_ratio: f64,
    accuracy: f64,
    mean_return: f64,
    max_drawdown: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    final_sharpe: f64,
    training_history: Vec<PerformanceMetrics>,
    quantum_contribution_avg: f64,
}

impl TradingDataset {
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Vec<(&Array1<f64>, &TradingLabel)>> {
        self.features.chunks(batch_size)
            .zip(self.labels.chunks(batch_size))
            .map(|(f_chunk, l_chunk)| {
                f_chunk.iter().zip(l_chunk.iter()).collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hybrid_model() {
        let quantum_backend = Arc::new(QuantumBackend::new().await.unwrap());
        
        let config = HybridStrategyConfig {
            quantum_layers: 2,
            classical_layers: 3,
            feature_dimension: 8,
            quantum_encoding: EncodingType::AngleEncoding,
            optimization_objective: OptimizationObjective::MaximizeSharpe,
        };
        
        let model = HybridTradingModel::new(quantum_backend, config).await.unwrap();
        
        // Test prediction
        let features = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.4]);
        let context = MarketContext {
            volatility: 0.2,
            correlation: 0.5,
            market_regime: MarketRegime::Trending,
            volume_profile: VolumeProfile {
                current: 1000000.0,
                average: 800000.0,
                trend: 0.1,
            },
        };
        
        let signal = model.predict(&features, &context).await.unwrap();
        
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.risk_score >= 0.0 && signal.risk_score <= 1.0);
        assert!(signal.quantum_contribution + signal.classical_contribution > 0.0);
    }
}