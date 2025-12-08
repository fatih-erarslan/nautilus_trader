//! Quantum superposition detector for market data

use crate::types::*;
use crate::Result;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use tracing::{debug, info};

/// Quantum superposition detector that creates superposition of all possible price paths
#[derive(Clone)]
pub struct QuantumSuperposition {
    /// Configuration
    config: QuantumConfig,
    /// Quantum state preparation parameters
    state_prep_params: SuperpositionParams,
}

#[derive(Debug, Clone)]
struct SuperpositionParams {
    /// Number of basis states
    basis_states: usize,
    /// Coherence preservation factor
    coherence_factor: f64,
    /// Quantum noise level
    noise_level: f64,
    /// Phase randomization strength
    phase_randomization: f64,
}

impl QuantumSuperposition {
    /// Create a new quantum superposition detector
    pub async fn new(config: &QuantumConfig) -> Result<Self> {
        info!("Initializing Quantum Superposition Detector");

        let state_prep_params = SuperpositionParams {
            basis_states: config.max_superposition_states,
            coherence_factor: config.coherence_threshold,
            noise_level: 0.01, // 1% quantum noise
            phase_randomization: 0.1,
        };

        Ok(Self {
            config: config.clone(),
            state_prep_params,
        })
    }

    /// Create quantum superposition of all possible price paths
    pub async fn create_superposition(&self, market_data: &MarketData) -> Result<QuantumMarketData> {
        debug!("Creating quantum superposition for {} instruments", 
               market_data.price_history.len());

        // Step 1: Prepare basis states from market data
        let basis_states = self.prepare_basis_states(market_data).await?;
        
        // Step 2: Create quantum amplitudes for superposition
        let amplitudes = self.calculate_quantum_amplitudes(&basis_states).await?;
        
        // Step 3: Generate entanglement matrix
        let entanglement_matrix = self.generate_entanglement_matrix(&basis_states).await?;
        
        // Step 4: Calculate phase information
        let phase_matrix = self.calculate_phase_matrix(&basis_states).await?;
        
        // Step 5: Estimate coherence time
        let coherence_time_ms = self.estimate_coherence_time(market_data).await?;

        let quantum_data = QuantumMarketData {
            superposition_states: basis_states,
            amplitudes,
            entanglement_matrix,
            phase_matrix,
            classical_data: market_data.clone(),
            coherence_time_ms,
        };

        debug!("Created quantum superposition with {} states, coherence time: {:.2}ms",
               quantum_data.superposition_states.nrows(), coherence_time_ms);

        Ok(quantum_data)
    }

    /// Update superposition with new market data
    pub async fn update_superposition(
        &self, 
        current_quantum_data: &QuantumMarketData,
        new_market_data: &MarketData
    ) -> Result<QuantumMarketData> {
        
        // Create new superposition
        let new_quantum_data = self.create_superposition(new_market_data).await?;
        
        // Perform quantum evolution to transition from current to new state
        let evolved_quantum_data = self.evolve_quantum_state(current_quantum_data, &new_quantum_data).await?;
        
        Ok(evolved_quantum_data)
    }

    // Private helper methods

    async fn prepare_basis_states(&self, market_data: &MarketData) -> Result<Array2<Complex64>> {
        let num_instruments = market_data.price_history.len();
        let max_history_length = market_data.price_history.values()
            .map(|prices| prices.len())
            .max()
            .unwrap_or(0);

        if max_history_length == 0 {
            return Err(crate::QuantumError::Superposition("No price history available".to_string()));
        }

        let num_states = self.state_prep_params.basis_states.min(1000); // Limit for performance
        let mut basis_states = Array2::zeros((num_states, num_instruments));

        // Generate basis states representing different possible market scenarios
        for state_idx in 0..num_states {
            for (inst_idx, (_symbol, prices)) in market_data.price_history.iter().enumerate() {
                if prices.is_empty() {
                    continue;
                }

                // Create quantum state representing price evolution possibility
                let price_scenario = self.generate_price_scenario(prices, state_idx, num_states).await?;
                
                // Convert to complex amplitude with phase information
                let phase = (state_idx as f64 / num_states as f64) * 2.0 * PI;
                let amplitude = Complex64::new(
                    price_scenario * phase.cos(),
                    price_scenario * phase.sin(),
                );
                
                basis_states[[state_idx, inst_idx]] = amplitude;
            }
        }

        // Normalize states to maintain quantum unitarity
        self.normalize_quantum_states(&mut basis_states).await?;

        Ok(basis_states)
    }

    async fn generate_price_scenario(&self, prices: &[f64], state_idx: usize, total_states: usize) -> Result<f64> {
        if prices.is_empty() {
            return Ok(0.0);
        }

        let last_price = prices[prices.len() - 1];
        
        // Generate different price evolution scenarios based on state index
        let scenario_factor = (state_idx as f64 / total_states as f64 - 0.5) * 2.0; // Range: -1 to 1
        
        // Apply volatility estimation
        let volatility = self.estimate_volatility(prices).await?;
        
        // Create price scenario with quantum uncertainty
        let price_change = scenario_factor * volatility * last_price;
        let scenario_price = last_price + price_change;
        
        // Normalize relative to current price
        Ok(scenario_price / last_price)
    }

    async fn estimate_volatility(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 2 {
            return Ok(0.01); // Default 1% volatility
        }

        // Calculate historical volatility using returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|window| (window[1] / window[0]).ln())
            .collect();

        if returns.is_empty() {
            return Ok(0.01);
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    async fn calculate_quantum_amplitudes(&self, basis_states: &Array2<Complex64>) -> Result<Array1<Complex64>> {
        let num_states = basis_states.nrows();
        let mut amplitudes = Array1::zeros(num_states);

        // Calculate amplitudes based on quantum probability principles
        for state_idx in 0..num_states {
            let state_norm = basis_states.row(state_idx).iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt();

            if state_norm > 0.0 {
                // Apply quantum coherence weighting
                let coherence_weight = self.calculate_coherence_weight(state_idx, num_states).await?;
                let amplitude = Complex64::new(
                    (1.0 / num_states as f64).sqrt() * coherence_weight,
                    0.0
                );
                amplitudes[state_idx] = amplitude;
            }
        }

        // Normalize amplitudes to ensure probability conservation
        let total_probability: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if total_probability > 0.0 {
            let normalization_factor = 1.0 / total_probability.sqrt();
            amplitudes *= Complex64::new(normalization_factor, 0.0);
        }

        Ok(amplitudes)
    }

    async fn calculate_coherence_weight(&self, state_idx: usize, total_states: usize) -> Result<f64> {
        // Calculate coherence weight based on state's position in superposition
        let normalized_position = state_idx as f64 / total_states as f64;
        
        // Use Gaussian weighting centered around 0.5 (middle states have higher coherence)
        let distance_from_center = (normalized_position - 0.5).abs() * 2.0;
        let coherence_weight = (-distance_from_center.powi(2) / (2.0 * 0.3_f64.powi(2))).exp();
        
        // Apply coherence factor from configuration
        Ok(coherence_weight * self.state_prep_params.coherence_factor)
    }

    async fn generate_entanglement_matrix(&self, basis_states: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        let num_instruments = basis_states.ncols();
        let mut entanglement_matrix = Array2::zeros((num_instruments, num_instruments));

        // Calculate pairwise entanglement between instruments
        for i in 0..num_instruments {
            for j in 0..num_instruments {
                if i == j {
                    // Self-entanglement (identity)
                    entanglement_matrix[[i, j]] = Complex64::new(1.0, 0.0);
                } else {
                    // Cross-instrument entanglement
                    let entanglement_strength = self.calculate_entanglement_strength(
                        basis_states, i, j
                    ).await?;
                    
                    let phase = ((i + j) as f64 / num_instruments as f64) * PI;
                    entanglement_matrix[[i, j]] = Complex64::new(
                        entanglement_strength * phase.cos(),
                        entanglement_strength * phase.sin(),
                    );
                }
            }
        }

        Ok(entanglement_matrix)
    }

    async fn calculate_entanglement_strength(&self, basis_states: &Array2<Complex64>, inst_i: usize, inst_j: usize) -> Result<f64> {
        let num_states = basis_states.nrows();
        
        // Calculate correlation between quantum states of two instruments
        let mut correlation_sum = 0.0;
        let mut count = 0;

        for state_idx in 0..num_states {
            let state_i = basis_states[[state_idx, inst_i]];
            let state_j = basis_states[[state_idx, inst_j]];
            
            // Calculate quantum correlation
            let correlation = (state_i.conj() * state_j).re;
            correlation_sum += correlation;
            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }

        let avg_correlation = correlation_sum / count as f64;
        
        // Apply entanglement sensitivity from configuration
        Ok(avg_correlation.abs() * self.config.entanglement_sensitivity)
    }

    async fn calculate_phase_matrix(&self, basis_states: &Array2<Complex64>) -> Result<Array2<f64>> {
        let num_states = basis_states.nrows();
        let num_instruments = basis_states.ncols();
        let mut phase_matrix = Array2::zeros((num_states, num_instruments));

        // Extract phase information from quantum states
        for state_idx in 0..num_states {
            for inst_idx in 0..num_instruments {
                let complex_amplitude = basis_states[[state_idx, inst_idx]];
                let phase = complex_amplitude.arg(); // Get phase angle
                phase_matrix[[state_idx, inst_idx]] = phase;
            }
        }

        Ok(phase_matrix)
    }

    async fn estimate_coherence_time(&self, market_data: &MarketData) -> Result<f64> {
        // Estimate quantum coherence time based on market volatility and complexity
        let num_instruments = market_data.price_history.len();
        let avg_volatility = self.calculate_average_market_volatility(market_data).await?;
        
        // Higher volatility and more instruments reduce coherence time
        let base_coherence_ms = 1000.0; // 1 second base coherence
        let volatility_factor = 1.0 / (1.0 + avg_volatility * 10.0);
        let complexity_factor = 1.0 / (1.0 + num_instruments as f64 * 0.1);
        
        let coherence_time = base_coherence_ms * volatility_factor * complexity_factor;
        
        // Apply configuration coherence threshold
        Ok(coherence_time * self.config.coherence_threshold)
    }

    async fn calculate_average_market_volatility(&self, market_data: &MarketData) -> Result<f64> {
        let mut total_volatility = 0.0;
        let mut count = 0;

        for prices in market_data.price_history.values() {
            if !prices.is_empty() {
                let volatility = self.estimate_volatility(prices).await?;
                total_volatility += volatility;
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.01); // Default volatility
        }

        Ok(total_volatility / count as f64)
    }

    async fn normalize_quantum_states(&self, states: &mut Array2<Complex64>) -> Result<()> {
        let num_states = states.nrows();

        for state_idx in 0..num_states {
            let mut state_norm_sq = 0.0;
            
            // Calculate state norm
            for inst_idx in 0..states.ncols() {
                state_norm_sq += states[[state_idx, inst_idx]].norm_sqr();
            }

            // Normalize if non-zero
            if state_norm_sq > 0.0 {
                let norm_factor = 1.0 / state_norm_sq.sqrt();
                for inst_idx in 0..states.ncols() {
                    states[[state_idx, inst_idx]] *= norm_factor;
                }
            }
        }

        Ok(())
    }

    async fn evolve_quantum_state(
        &self,
        current_state: &QuantumMarketData,
        new_state: &QuantumMarketData,
    ) -> Result<QuantumMarketData> {
        
        // Simple linear interpolation for quantum evolution
        // In practice, this would use proper quantum evolution operators
        let evolution_factor = 0.7; // Weight towards new state
        
        let mut evolved_amplitudes = Array1::zeros(new_state.amplitudes.len());
        for i in 0..evolved_amplitudes.len() {
            if i < current_state.amplitudes.len() {
                evolved_amplitudes[i] = current_state.amplitudes[i] * (1.0 - evolution_factor) + 
                                      new_state.amplitudes[i] * evolution_factor;
            } else {
                evolved_amplitudes[i] = new_state.amplitudes[i];
            }
        }

        // Calculate evolved coherence time
        let evolved_coherence = current_state.coherence_time_ms * (1.0 - evolution_factor) + 
                               new_state.coherence_time_ms * evolution_factor;

        Ok(QuantumMarketData {
            superposition_states: new_state.superposition_states.clone(),
            amplitudes: evolved_amplitudes,
            entanglement_matrix: new_state.entanglement_matrix.clone(),
            phase_matrix: new_state.phase_matrix.clone(),
            classical_data: new_state.classical_data.clone(),
            coherence_time_ms: evolved_coherence,
        })
    }
}

/// Initialize quantum superposition system
pub async fn init() -> Result<()> {
    info!("Quantum Superposition subsystem initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use chrono::Utc;
    use ndarray::Array1;

    #[tokio::test]
    async fn test_superposition_creation() {
        let config = QuantumConfig::default();
        let detector = QuantumSuperposition::new(&config).await.unwrap();

        let mut price_history = HashMap::new();
        price_history.insert("BTCUSDT".to_string(), vec![50000.0, 51000.0, 49000.0]);
        price_history.insert("ETHUSDT".to_string(), vec![3000.0, 3100.0, 2900.0]);

        let market_data = MarketData {
            price_history,
            volume_data: HashMap::new(),
            timestamps: vec![Utc::now(); 3],
            features: ndarray::Array2::zeros((3, 2)),
            regime_indicators: Array1::zeros(3),
        };

        let quantum_data = detector.create_superposition(&market_data).await.unwrap();
        
        assert!(quantum_data.superposition_states.nrows() > 0);
        assert!(quantum_data.amplitudes.len() > 0);
        assert!(quantum_data.coherence_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_volatility_estimation() {
        let config = QuantumConfig::default();
        let detector = QuantumSuperposition::new(&config).await.unwrap();

        let prices = vec![100.0, 102.0, 98.0, 105.0, 95.0];
        let volatility = detector.estimate_volatility(&prices).await.unwrap();
        
        assert!(volatility > 0.0);
        assert!(volatility < 1.0); // Should be reasonable volatility
    }
}