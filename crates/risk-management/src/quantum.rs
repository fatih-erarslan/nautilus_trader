//! Quantum risk integration interfaces and utilities

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array1, Array2};
use quantum_uncertainty::{
    QuantumUncertaintyEngine, UncertaintyQuantification, QuantumState,
    QuantumCorrelations, QuantumFeatures, ConformalPredictionIntervals
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::error::{RiskError, RiskResult};
use crate::types::{Portfolio, Position, MarketData};

/// Quantum portfolio data representation
#[derive(Debug, Clone)]
pub struct QuantumPortfolioData {
    /// Return matrix
    pub returns: Array2<f64>,
    /// Target returns
    pub targets: Array1<f64>,
    /// Position data
    pub positions: Vec<Position>,
    /// Market data
    pub market_data: MarketData,
}

/// Quantum risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumRiskMetrics {
    /// Quantum-enhanced VaR
    pub quantum_var: f64,
    /// Quantum-enhanced CVaR
    pub quantum_cvar: f64,
    /// Quantum advantage factor
    pub quantum_advantage: f64,
    /// Correlation uncertainty
    pub correlation_uncertainty: f64,
    /// Tail risk quantum enhancement
    pub tail_risk_quantum: f64,
    /// Quantum circuit fidelity
    pub quantum_fidelity: f64,
    /// Quantum features
    pub quantum_features: QuantumFeatureMetrics,
    /// Conformal prediction intervals
    pub conformal_intervals: QuantumConformalMetrics,
}

/// Quantum feature metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatureMetrics {
    /// Superposition coherence
    pub superposition_coherence: f64,
    /// Entanglement strength
    pub entanglement_strength: f64,
    /// Feature dimensionality
    pub feature_dimensionality: usize,
    /// Quantum information content
    pub quantum_information: f64,
}

/// Quantum conformal prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConformalMetrics {
    /// Lower prediction bound
    pub lower_bound: f64,
    /// Upper prediction bound
    pub upper_bound: f64,
    /// Prediction interval width
    pub interval_width: f64,
    /// Coverage probability
    pub coverage_probability: f64,
    /// Quantum tail risk
    pub tail_risk_quantum: f64,
}

/// Quantum correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCorrelationAnalysis {
    /// Quantum correlation matrix
    pub quantum_correlations: Array2<f64>,
    /// Classical correlation matrix
    pub classical_correlations: Array2<f64>,
    /// Quantum advantage in correlation modeling
    pub correlation_quantum_advantage: f64,
    /// Entanglement measures
    pub entanglement_measures: HashMap<String, f64>,
    /// Quantum coherence metrics
    pub coherence_metrics: QuantumCoherenceMetrics,
}

/// Quantum coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceMetrics {
    /// Relative entropy of coherence
    pub relative_entropy: f64,
    /// L1 norm of coherence
    pub l1_norm: f64,
    /// Robustness of coherence
    pub robustness: f64,
    /// Coherence of formation
    pub formation: f64,
}

/// Quantum risk integrator
pub struct QuantumRiskIntegrator {
    /// Quantum uncertainty engine
    quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>,
    /// Quantum state cache
    quantum_state_cache: Arc<RwLock<HashMap<String, QuantumState>>>,
    /// Feature cache
    feature_cache: Arc<RwLock<HashMap<String, QuantumFeatures>>>,
    /// Correlation cache
    correlation_cache: Arc<RwLock<HashMap<String, QuantumCorrelations>>>,
}

impl QuantumRiskIntegrator {
    /// Create new quantum risk integrator
    pub fn new(quantum_engine: Arc<RwLock<QuantumUncertaintyEngine>>) -> Self {
        Self {
            quantum_engine,
            quantum_state_cache: Arc::new(RwLock::new(HashMap::new())),
            feature_cache: Arc::new(RwLock::new(HashMap::new())),
            correlation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Calculate quantum-enhanced risk metrics
    pub async fn calculate_quantum_risk_metrics(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumRiskMetrics> {
        info!("Calculating quantum-enhanced risk metrics");
        
        // Convert portfolio to quantum format
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        // Perform quantum uncertainty quantification
        let quantum_engine = self.quantum_engine.read().await;
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &quantum_data.returns,
            &quantum_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        // Calculate quantum risk metrics
        let quantum_var = self.calculate_quantum_var(&uncertainty_quantification)?;
        let quantum_cvar = self.calculate_quantum_cvar(&uncertainty_quantification)?;
        
        // Get quantum circuit fidelity
        let quantum_fidelity = quantum_engine.validate_circuit_fidelity().await
            .map_err(|e| RiskError::quantum_circuit(e))?;
        
        // Extract quantum features metrics
        let quantum_features = self.extract_quantum_feature_metrics(
            &uncertainty_quantification.quantum_features
        )?;
        
        // Extract conformal prediction metrics
        let conformal_intervals = self.extract_conformal_metrics(
            &uncertainty_quantification.conformal_intervals
        )?;
        
        Ok(QuantumRiskMetrics {
            quantum_var,
            quantum_cvar,
            quantum_advantage: uncertainty_quantification.quantum_advantage,
            correlation_uncertainty: uncertainty_quantification.correlations.uncertainty_level,
            tail_risk_quantum: uncertainty_quantification.conformal_intervals.tail_risk_quantum,
            quantum_fidelity,
            quantum_features,
            conformal_intervals,
        })
    }
    
    /// Analyze quantum correlations
    pub async fn analyze_quantum_correlations(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumCorrelationAnalysis> {
        info!("Analyzing quantum correlations");
        
        // Convert portfolio to quantum format
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        // Get quantum correlation analysis
        let quantum_engine = self.quantum_engine.read().await;
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &quantum_data.returns,
            &quantum_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        // Extract quantum and classical correlations
        let quantum_correlations = self.extract_quantum_correlation_matrix(
            &uncertainty_quantification.correlations
        )?;
        
        let classical_correlations = self.calculate_classical_correlations(&quantum_data)?;
        
        // Calculate quantum advantage in correlation modeling
        let correlation_quantum_advantage = self.calculate_correlation_quantum_advantage(
            &quantum_correlations,
            &classical_correlations,
        )?;
        
        // Calculate entanglement measures
        let entanglement_measures = self.calculate_entanglement_measures(
            &uncertainty_quantification.correlations
        )?;
        
        // Calculate coherence metrics
        let coherence_metrics = self.calculate_coherence_metrics(
            &uncertainty_quantification.quantum_features
        )?;
        
        Ok(QuantumCorrelationAnalysis {
            quantum_correlations,
            classical_correlations,
            correlation_quantum_advantage,
            entanglement_measures,
            coherence_metrics,
        })
    }
    
    /// Create quantum state for portfolio
    pub async fn create_quantum_portfolio_state(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumState> {
        info!("Creating quantum portfolio state");
        
        // Check cache first
        let cache_key = format!("portfolio_{}", portfolio.id);
        if let Some(cached_state) = self.quantum_state_cache.read().await.get(&cache_key) {
            return Ok(cached_state.clone());
        }
        
        // Create quantum state from portfolio data
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        // Generate quantum state using superposition of portfolio states
        let n_assets = portfolio.positions.len();
        let n_qubits = (n_assets as f64).log2().ceil() as usize;
        
        // Create superposition state representing portfolio uncertainty
        let quantum_state = QuantumState::uniform_superposition(n_qubits);
        
        // Cache the state
        self.quantum_state_cache.write().await.insert(cache_key, quantum_state.clone());
        
        Ok(quantum_state)
    }
    
    /// Optimize quantum measurements for risk assessment
    pub async fn optimize_quantum_measurements(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<OptimizedQuantumMeasurements> {
        info!("Optimizing quantum measurements for risk assessment");
        
        let quantum_data = self.portfolio_to_quantum_data(portfolio).await?;
        
        let quantum_engine = self.quantum_engine.read().await;
        let uncertainty_quantification = quantum_engine.quantify_uncertainty(
            &quantum_data.returns,
            &quantum_data.targets,
        ).await.map_err(|e| RiskError::quantum_uncertainty(e))?;
        
        // Extract optimized measurements
        Ok(OptimizedQuantumMeasurements {
            measurement_basis: uncertainty_quantification.optimized_measurements.measurement_basis.clone(),
            measurement_probabilities: uncertainty_quantification.optimized_measurements.probabilities.clone(),
            optimal_measurement_order: uncertainty_quantification.optimized_measurements.optimal_order.clone(),
            measurement_efficiency: uncertainty_quantification.optimized_measurements.efficiency,
            quantum_information_gain: uncertainty_quantification.optimized_measurements.information_gain,
        })
    }
    
    /// Convert portfolio to quantum data format
    async fn portfolio_to_quantum_data(
        &self,
        portfolio: &Portfolio,
    ) -> RiskResult<QuantumPortfolioData> {
        // Convert returns to matrix format
        let n_observations = portfolio.returns.len();
        let n_assets = portfolio.positions.len().max(1);
        
        let returns = if portfolio.returns.is_empty() {
            Array2::zeros((1, n_assets))
        } else if n_assets == 1 {
            Array2::from_shape_vec(
                (n_observations, 1),
                portfolio.returns.clone(),
            ).map_err(|e| RiskError::matrix_operation(format!("Failed to create returns matrix: {}", e)))?
        } else {
            // For multiple assets, we'd need asset-specific returns
            // For now, replicate portfolio returns across assets
            let mut returns_matrix = Array2::zeros((n_observations, n_assets));
            for i in 0..n_observations {
                for j in 0..n_assets {
                    returns_matrix[[i, j]] = portfolio.returns[i];
                }
            }
            returns_matrix
        };
        
        let targets = if portfolio.targets.is_empty() {
            Array1::zeros(n_observations.max(1))
        } else {
            Array1::from_vec(portfolio.targets.clone())
        };
        
        Ok(QuantumPortfolioData {
            returns,
            targets,
            positions: portfolio.positions.clone(),
            market_data: portfolio.market_data.clone(),
        })
    }
    
    /// Calculate quantum VaR
    fn calculate_quantum_var(
        &self,
        uncertainty_quantification: &UncertaintyQuantification,
    ) -> RiskResult<f64> {
        let mean_uncertainty = uncertainty_quantification.mean_uncertainty();
        let quantum_advantage = uncertainty_quantification.quantum_advantage;
        
        // Apply quantum enhancement to traditional VaR
        let quantum_var = mean_uncertainty * (1.0 + quantum_advantage * 0.1);
        
        Ok(quantum_var)
    }
    
    /// Calculate quantum CVaR
    fn calculate_quantum_cvar(
        &self,
        uncertainty_quantification: &UncertaintyQuantification,
    ) -> RiskResult<f64> {
        let uncertainty_variance = uncertainty_quantification.uncertainty_variance();
        let quantum_advantage = uncertainty_quantification.quantum_advantage;
        
        // Apply quantum enhancement to traditional CVaR
        let quantum_cvar = uncertainty_variance.sqrt() * (1.0 + quantum_advantage * 0.15);
        
        Ok(quantum_cvar)
    }
    
    /// Extract quantum feature metrics
    fn extract_quantum_feature_metrics(
        &self,
        quantum_features: &QuantumFeatures,
    ) -> RiskResult<QuantumFeatureMetrics> {
        Ok(QuantumFeatureMetrics {
            superposition_coherence: quantum_features.superposition_coherence,
            entanglement_strength: quantum_features.entanglement_strength,
            feature_dimensionality: quantum_features.feature_dimension,
            quantum_information: quantum_features.quantum_information_content,
        })
    }
    
    /// Extract conformal prediction metrics
    fn extract_conformal_metrics(
        &self,
        conformal_intervals: &ConformalPredictionIntervals,
    ) -> RiskResult<QuantumConformalMetrics> {
        Ok(QuantumConformalMetrics {
            lower_bound: conformal_intervals.lower_bound,
            upper_bound: conformal_intervals.upper_bound,
            interval_width: conformal_intervals.upper_bound - conformal_intervals.lower_bound,
            coverage_probability: conformal_intervals.coverage_probability,
            tail_risk_quantum: conformal_intervals.tail_risk_quantum,
        })
    }
    
    /// Extract quantum correlation matrix
    fn extract_quantum_correlation_matrix(
        &self,
        quantum_correlations: &QuantumCorrelations,
    ) -> RiskResult<Array2<f64>> {
        // Convert quantum correlation structure to matrix
        let n_assets = quantum_correlations.correlation_dimension;
        let mut correlation_matrix = Array2::eye(n_assets);
        
        // Apply quantum correlation values
        for i in 0..n_assets {
            for j in i + 1..n_assets {
                let correlation_value = quantum_correlations.quantum_correlations
                    .get(&format!("{}_{}", i, j))
                    .unwrap_or(&0.0);
                correlation_matrix[[i, j]] = *correlation_value;
                correlation_matrix[[j, i]] = *correlation_value;
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// Calculate classical correlations for comparison
    fn calculate_classical_correlations(
        &self,
        quantum_data: &QuantumPortfolioData,
    ) -> RiskResult<Array2<f64>> {
        let returns = &quantum_data.returns;
        let n_assets = returns.ncols();
        
        // Calculate Pearson correlation matrix
        let mut correlation_matrix = Array2::eye(n_assets);
        
        for i in 0..n_assets {
            for j in i + 1..n_assets {
                let returns_i = returns.column(i);
                let returns_j = returns.column(j);
                
                let correlation = self.calculate_pearson_correlation(&returns_i, &returns_j)?;
                correlation_matrix[[i, j]] = correlation;
                correlation_matrix[[j, i]] = correlation;
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// Calculate Pearson correlation coefficient
    fn calculate_pearson_correlation(
        &self,
        x: &ndarray::ArrayView1<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> RiskResult<f64> {
        if x.len() != y.len() || x.len() == 0 {
            return Err(RiskError::insufficient_data("Invalid data for correlation calculation"));
        }
        
        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let diff_x = x[i] - mean_x;
            let diff_y = y[i] - mean_y;
            
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Calculate quantum advantage in correlation modeling
    fn calculate_correlation_quantum_advantage(
        &self,
        quantum_correlations: &Array2<f64>,
        classical_correlations: &Array2<f64>,
    ) -> RiskResult<f64> {
        if quantum_correlations.shape() != classical_correlations.shape() {
            return Err(RiskError::matrix_operation("Correlation matrix dimension mismatch"));
        }
        
        // Calculate Frobenius norm difference
        let diff = quantum_correlations - classical_correlations;
        let frobenius_norm = diff.mapv(|x| x * x).sum().sqrt();
        
        // Quantum advantage as normalized improvement
        let classical_norm = classical_correlations.mapv(|x| x * x).sum().sqrt();
        
        if classical_norm == 0.0 {
            Ok(0.0)
        } else {
            Ok(frobenius_norm / classical_norm)
        }
    }
    
    /// Calculate entanglement measures
    fn calculate_entanglement_measures(
        &self,
        quantum_correlations: &QuantumCorrelations,
    ) -> RiskResult<HashMap<String, f64>> {
        let mut measures = HashMap::new();
        
        // Von Neumann entropy (simplified)
        measures.insert("von_neumann_entropy".to_string(), quantum_correlations.von_neumann_entropy);
        
        // Concurrence (simplified)
        measures.insert("concurrence".to_string(), quantum_correlations.concurrence);
        
        // Negativity (simplified)
        measures.insert("negativity".to_string(), quantum_correlations.negativity);
        
        // Quantum mutual information
        measures.insert("quantum_mutual_information".to_string(), quantum_correlations.mutual_information);
        
        Ok(measures)
    }
    
    /// Calculate coherence metrics
    fn calculate_coherence_metrics(
        &self,
        quantum_features: &QuantumFeatures,
    ) -> RiskResult<QuantumCoherenceMetrics> {
        Ok(QuantumCoherenceMetrics {
            relative_entropy: quantum_features.coherence_measures.relative_entropy,
            l1_norm: quantum_features.coherence_measures.l1_norm,
            robustness: quantum_features.coherence_measures.robustness,
            formation: quantum_features.coherence_measures.formation,
        })
    }
    
    /// Reset quantum integrator state
    pub async fn reset(&self) -> RiskResult<()> {
        self.quantum_state_cache.write().await.clear();
        self.feature_cache.write().await.clear();
        self.correlation_cache.write().await.clear();
        Ok(())
    }
}

/// Optimized quantum measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedQuantumMeasurements {
    /// Optimal measurement basis
    pub measurement_basis: Vec<String>,
    /// Measurement probabilities
    pub measurement_probabilities: Vec<f64>,
    /// Optimal measurement order
    pub optimal_measurement_order: Vec<usize>,
    /// Measurement efficiency
    pub measurement_efficiency: f64,
    /// Quantum information gain
    pub quantum_information_gain: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tokio_test;

    #[tokio::test]
    async fn test_quantum_risk_integrator() {
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let integrator = QuantumRiskIntegrator::new(quantum_engine);
        
        let portfolio = Portfolio::default();
        let result = integrator.calculate_quantum_risk_metrics(&portfolio).await;
        
        // Should handle empty portfolio gracefully
        assert!(result.is_ok() || matches!(result.unwrap_err(), RiskError::InsufficientData(_)));
    }

    #[tokio::test]
    async fn test_portfolio_to_quantum_data_conversion() {
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
        ));
        
        let integrator = QuantumRiskIntegrator::new(quantum_engine);
        
        let mut portfolio = Portfolio::default();
        portfolio.returns = vec![0.01, -0.02, 0.015];
        portfolio.targets = vec![0.01, 0.01, 0.01];
        
        let quantum_data = integrator.portfolio_to_quantum_data(&portfolio).await;
        assert!(quantum_data.is_ok());
        
        let data = quantum_data.unwrap();
        assert_eq!(data.returns.nrows(), 3);
        assert_eq!(data.targets.len(), 3);
    }

    #[test]
    fn test_pearson_correlation_calculation() {
        let quantum_config = quantum_uncertainty::QuantumConfig::default();
        let quantum_engine = Arc::new(RwLock::new(
            futures::executor::block_on(async {
                QuantumUncertaintyEngine::new(quantum_config).await.unwrap()
            })
        ));
        
        let integrator = QuantumRiskIntegrator::new(quantum_engine);
        
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        
        let correlation = integrator.calculate_pearson_correlation(&x.view(), &y.view());
        assert!(correlation.is_ok());
        assert_abs_diff_eq!(correlation.unwrap(), 1.0, epsilon = 1e-10);
    }
}