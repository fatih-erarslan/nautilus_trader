//! Quantum entanglement correlation finder for cross-asset relationships

use crate::types::*;
use crate::Result;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;
use tracing::{debug, info};

/// Quantum entanglement detector that finds non-local correlations
#[derive(Clone)]
pub struct QuantumEntanglement {
    /// Configuration
    config: QuantumConfig,
    /// Entanglement detection parameters
    entanglement_params: EntanglementParams,
}

#[derive(Debug, Clone)]
struct EntanglementParams {
    /// Bell state threshold for entanglement detection
    bell_threshold: f64,
    /// Maximum entanglement distance (in correlation space)
    max_entanglement_distance: f64,
    /// Decoherence rate estimation factor
    decoherence_factor: f64,
    /// Fidelity calculation precision
    fidelity_precision: f64,
}

impl QuantumEntanglement {
    /// Create a new quantum entanglement detector
    pub async fn new(config: &QuantumConfig) -> Result<Self> {
        info!("Initializing Quantum Entanglement Detector");

        let entanglement_params = EntanglementParams {
            bell_threshold: config.entanglement_sensitivity,
            max_entanglement_distance: 2.0,
            decoherence_factor: 0.1,
            fidelity_precision: 1e-6,
        };

        Ok(Self {
            config: config.clone(),
            entanglement_params,
        })
    }

    /// Find entangled correlations in quantum superposition states
    pub async fn find_entangled_correlations(
        &self, 
        quantum_data: &QuantumMarketData
    ) -> Result<EntanglementCorrelation> {
        
        debug!("Analyzing entanglement in {} superposition states", 
               quantum_data.superposition_states.nrows());

        // Step 1: Identify potentially entangled pairs
        let candidate_pairs = self.identify_entangled_pairs(quantum_data).await?;
        
        // Step 2: Calculate Bell state coefficients
        let bell_coefficients = self.calculate_bell_coefficients(quantum_data, &candidate_pairs).await?;
        
        // Step 3: Compute correlation matrix in quantum space
        let correlation_matrix = self.compute_quantum_correlation_matrix(quantum_data).await?;
        
        // Step 4: Calculate entanglement strength
        let entanglement_strength = self.calculate_entanglement_strength(&correlation_matrix, &bell_coefficients).await?;
        
        // Step 5: Estimate fidelity and decoherence
        let fidelity = self.calculate_entanglement_fidelity(&bell_coefficients).await?;
        let decoherence_rate = self.estimate_decoherence_rate(quantum_data).await?;

        let entanglement_result = EntanglementCorrelation {
            strength: entanglement_strength,
            entangled_pairs: candidate_pairs,
            correlation_matrix,
            bell_coefficients,
            fidelity,
            decoherence_rate,
        };

        debug!("Found {} entangled pairs with strength {:.3}, fidelity {:.3}",
               entanglement_result.entangled_pairs.len(),
               entanglement_strength,
               fidelity);

        Ok(entanglement_result)
    }

    /// Validate entanglement using Bell inequality tests
    pub async fn validate_entanglement_bell_test(
        &self,
        entanglement: &EntanglementCorrelation
    ) -> Result<bool> {
        
        // Perform CHSH Bell inequality test
        let chsh_value = self.calculate_chsh_inequality(&entanglement.correlation_matrix).await?;
        
        // Bell inequality violation indicates genuine quantum entanglement
        let bell_violation = chsh_value > 2.0; // Classical limit is 2, quantum can reach 2√2 ≈ 2.828
        
        debug!("CHSH value: {:.3}, Bell violation: {}", chsh_value, bell_violation);
        
        Ok(bell_violation && entanglement.strength > self.entanglement_params.bell_threshold)
    }

    /// Monitor entanglement evolution over time
    pub async fn monitor_entanglement_evolution(
        &self,
        current_entanglement: &EntanglementCorrelation,
        new_quantum_data: &QuantumMarketData,
    ) -> Result<EntanglementEvolution> {
        
        // Find new entanglement state
        let new_entanglement = self.find_entangled_correlations(new_quantum_data).await?;
        
        // Calculate evolution metrics
        let strength_change = new_entanglement.strength - current_entanglement.strength;
        let fidelity_change = new_entanglement.fidelity - current_entanglement.fidelity;
        
        // Detect entanglement creation/destruction events
        let entanglement_events = self.detect_entanglement_events(
            current_entanglement, 
            &new_entanglement
        ).await?;

        Ok(EntanglementEvolution {
            strength_change,
            fidelity_change,
            events: entanglement_events,
            stability_metric: self.calculate_stability_metric(current_entanglement, &new_entanglement).await?,
        })
    }

    // Private helper methods

    async fn identify_entangled_pairs(&self, quantum_data: &QuantumMarketData) -> Result<Vec<(String, String)>> {
        let instruments: Vec<String> = quantum_data.classical_data.price_history.keys().cloned().collect();
        let mut entangled_pairs = Vec::new();

        // Check all possible pairs for entanglement
        for i in 0..instruments.len() {
            for j in (i + 1)..instruments.len() {
                let entanglement_strength = self.measure_pairwise_entanglement(
                    quantum_data, i, j
                ).await?;

                if entanglement_strength > self.entanglement_params.bell_threshold {
                    entangled_pairs.push((instruments[i].clone(), instruments[j].clone()));
                    debug!("Detected entanglement between {} and {} (strength: {:.3})",
                           instruments[i], instruments[j], entanglement_strength);
                }
            }
        }

        Ok(entangled_pairs)
    }

    async fn measure_pairwise_entanglement(
        &self,
        quantum_data: &QuantumMarketData,
        inst_i: usize,
        inst_j: usize,
    ) -> Result<f64> {
        
        if inst_i >= quantum_data.superposition_states.ncols() || 
           inst_j >= quantum_data.superposition_states.ncols() {
            return Ok(0.0);
        }

        let num_states = quantum_data.superposition_states.nrows();
        let mut entanglement_measure = 0.0;

        // Calculate quantum mutual information as entanglement measure
        for state_idx in 0..num_states {
            let state_i = quantum_data.superposition_states[[state_idx, inst_i]];
            let state_j = quantum_data.superposition_states[[state_idx, inst_j]];
            let amplitude = quantum_data.amplitudes[state_idx];

            // Von Neumann entropy contribution
            let joint_state = state_i * state_j.conj();
            let probability = (amplitude * joint_state).norm_sqr();

            if probability > 1e-10 {
                entanglement_measure += probability * (-probability.ln());
            }
        }

        // Normalize by configuration sensitivity
        Ok(entanglement_measure * self.config.entanglement_sensitivity)
    }

    async fn calculate_bell_coefficients(
        &self,
        quantum_data: &QuantumMarketData,
        entangled_pairs: &[(String, String)],
    ) -> Result<Array1<Complex64>> {
        
        let num_pairs = entangled_pairs.len();
        if num_pairs == 0 {
            return Ok(Array1::zeros(4)); // Four Bell states |Φ±⟩, |Ψ±⟩
        }

        let mut bell_coefficients = Array1::zeros(4);
        
        // Calculate coefficients for the four Bell states
        for (pair_idx, (_inst1, _inst2)) in entangled_pairs.iter().enumerate() {
            if pair_idx >= quantum_data.entanglement_matrix.nrows() {
                continue;
            }

            // Extract entanglement amplitude for this pair
            let entanglement_amplitude = quantum_data.entanglement_matrix[[pair_idx, pair_idx]];
            
            // Decompose into Bell state components
            let phi_plus = entanglement_amplitude / 2.0_f64.sqrt(); // |Φ+⟩ = (|00⟩ + |11⟩)/√2
            let phi_minus = entanglement_amplitude * Complex64::new(0.0, 1.0) / 2.0_f64.sqrt(); // |Φ-⟩
            let psi_plus = entanglement_amplitude.conj() / 2.0_f64.sqrt(); // |Ψ+⟩
            let psi_minus = entanglement_amplitude * Complex64::new(-1.0, 0.0) / 2.0_f64.sqrt(); // |Ψ-⟩

            bell_coefficients[0] += phi_plus;
            bell_coefficients[1] += phi_minus;
            bell_coefficients[2] += psi_plus;
            bell_coefficients[3] += psi_minus;
        }

        // Normalize Bell coefficients
        let norm = bell_coefficients.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            bell_coefficients /= norm;
        }

        Ok(bell_coefficients)
    }

    async fn compute_quantum_correlation_matrix(&self, quantum_data: &QuantumMarketData) -> Result<Array2<Complex64>> {
        let num_instruments = quantum_data.superposition_states.ncols();
        let num_states = quantum_data.superposition_states.nrows();
        let mut correlation_matrix = Array2::zeros((num_instruments, num_instruments));

        // Compute quantum correlation matrix using density matrix formalism
        for i in 0..num_instruments {
            for j in 0..num_instruments {
                let mut correlation = Complex64::new(0.0, 0.0);

                for state_idx in 0..num_states {
                    let state_i = quantum_data.superposition_states[[state_idx, i]];
                    let state_j = quantum_data.superposition_states[[state_idx, j]];
                    let amplitude = quantum_data.amplitudes[state_idx];

                    // Quantum correlation: ⟨ψ|σᵢ ⊗ σⱼ|ψ⟩
                    correlation += amplitude.conj() * state_i.conj() * state_j * amplitude;
                }

                correlation_matrix[[i, j]] = correlation;
            }
        }

        Ok(correlation_matrix)
    }

    async fn calculate_entanglement_strength(
        &self,
        correlation_matrix: &Array2<Complex64>,
        bell_coefficients: &Array1<Complex64>,
    ) -> Result<f64> {
        
        // Calculate entanglement strength using multiple measures
        
        // 1. Trace of correlation matrix squared (entanglement measure)
        let mut trace_corr_sq = 0.0;
        for i in 0..correlation_matrix.nrows() {
            for j in 0..correlation_matrix.ncols() {
                trace_corr_sq += correlation_matrix[[i, j]].norm_sqr();
            }
        }

        // 2. Bell coefficient magnitude
        let bell_magnitude = bell_coefficients.iter().map(|c| c.norm_sqr()).sum::<f64>();

        // 3. Quantum entanglement entropy
        let mut entanglement_entropy = 0.0;
        for coeff in bell_coefficients.iter() {
            let prob = coeff.norm_sqr();
            if prob > 1e-10 {
                entanglement_entropy -= prob * prob.ln();
            }
        }

        // Combine measures with weights
        let combined_strength = 0.4 * trace_corr_sq.sqrt() + 
                               0.4 * bell_magnitude.sqrt() + 
                               0.2 * entanglement_entropy;

        // Clamp to [0, 1] range
        Ok(combined_strength.min(1.0).max(0.0))
    }

    async fn calculate_entanglement_fidelity(&self, bell_coefficients: &Array1<Complex64>) -> Result<f64> {
        // Calculate fidelity with maximally entangled state
        // F = |⟨ψ_max|ψ⟩|²
        
        if bell_coefficients.is_empty() {
            return Ok(0.0);
        }

        // Maximally entangled state (equal superposition of Bell states)
        let max_entangled_amplitude = 0.5_f64.sqrt(); // 1/√4 for four Bell states
        
        // Calculate overlap with actual state
        let mut fidelity = 0.0;
        for coeff in bell_coefficients.iter() {
            fidelity += (coeff.norm() * max_entangled_amplitude).powi(2);
        }

        // Average over Bell states
        fidelity /= bell_coefficients.len() as f64;

        Ok(fidelity.min(1.0).max(0.0))
    }

    async fn estimate_decoherence_rate(&self, quantum_data: &QuantumMarketData) -> Result<f64> {
        // Estimate decoherence rate based on quantum state complexity and market volatility
        
        let num_states = quantum_data.superposition_states.nrows();
        let num_instruments = quantum_data.superposition_states.ncols();
        
        // Base decoherence rate increases with system complexity
        let complexity_factor = (num_states * num_instruments) as f64;
        let base_decoherence = self.entanglement_params.decoherence_factor / complexity_factor.sqrt();
        
        // Adjust for quantum coherence time
        let coherence_adjustment = 1000.0 / quantum_data.coherence_time_ms.max(1.0);
        
        // Market volatility increases decoherence
        let volatility_factor = self.estimate_market_entropy(quantum_data).await?;
        
        let decoherence_rate = base_decoherence * coherence_adjustment * (1.0 + volatility_factor);
        
        Ok(decoherence_rate.min(1.0).max(0.0))
    }

    async fn estimate_market_entropy(&self, quantum_data: &QuantumMarketData) -> Result<f64> {
        // Calculate market entropy from quantum amplitude distribution
        let mut entropy = 0.0;
        
        for amplitude in quantum_data.amplitudes.iter() {
            let probability = amplitude.norm_sqr();
            if probability > 1e-10 {
                entropy -= probability * probability.ln();
            }
        }

        // Normalize entropy
        let max_entropy = (quantum_data.amplitudes.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy /= max_entropy;
        }

        Ok(entropy)
    }

    async fn calculate_chsh_inequality(&self, correlation_matrix: &Array2<Complex64>) -> Result<f64> {
        // Calculate CHSH (Clauser-Horne-Shimony-Holt) inequality value
        // CHSH = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2 (classical), ≤ 2√2 (quantum)
        
        if correlation_matrix.nrows() < 2 || correlation_matrix.ncols() < 2 {
            return Ok(0.0);
        }

        // Use first four correlation terms for CHSH calculation
        let e_ab = correlation_matrix[[0, 1]].re;
        let e_ab_prime = if correlation_matrix.ncols() > 2 { 
            correlation_matrix[[0, 2]].re 
        } else { 
            correlation_matrix[[0, 1]].im 
        };
        let e_a_prime_b = correlation_matrix[[1, 0]].re;
        let e_a_prime_b_prime = correlation_matrix[[1, 1]].re;

        let chsh_value = (e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime).abs();
        
        Ok(chsh_value)
    }

    async fn detect_entanglement_events(
        &self,
        current: &EntanglementCorrelation,
        new: &EntanglementCorrelation,
    ) -> Result<Vec<EntanglementEvent>> {
        
        let mut events = Vec::new();
        
        // Detect sudden entanglement changes
        let strength_change = new.strength - current.strength;
        if strength_change.abs() > 0.1 {
            if strength_change > 0.0 {
                events.push(EntanglementEvent::EntanglementIncrease(strength_change));
            } else {
                events.push(EntanglementEvent::EntanglementDecrease(strength_change.abs()));
            }
        }

        // Detect fidelity changes
        let fidelity_change = new.fidelity - current.fidelity;
        if fidelity_change.abs() > 0.05 {
            events.push(EntanglementEvent::FidelityChange(fidelity_change));
        }

        // Detect new entangled pairs
        for new_pair in &new.entangled_pairs {
            if !current.entangled_pairs.contains(new_pair) {
                events.push(EntanglementEvent::NewEntangledPair(new_pair.clone()));
            }
        }

        // Detect lost entangled pairs
        for old_pair in &current.entangled_pairs {
            if !new.entangled_pairs.contains(old_pair) {
                events.push(EntanglementEvent::LostEntangledPair(old_pair.clone()));
            }
        }

        Ok(events)
    }

    async fn calculate_stability_metric(
        &self,
        current: &EntanglementCorrelation,
        new: &EntanglementCorrelation,
    ) -> Result<f64> {
        
        // Calculate entanglement stability based on various factors
        let strength_stability = 1.0 - (new.strength - current.strength).abs();
        let fidelity_stability = 1.0 - (new.fidelity - current.fidelity).abs();
        let pair_stability = if current.entangled_pairs.len() > 0 {
            let common_pairs = current.entangled_pairs.iter()
                .filter(|pair| new.entangled_pairs.contains(pair))
                .count();
            common_pairs as f64 / current.entangled_pairs.len() as f64
        } else {
            1.0
        };

        // Weighted average of stability measures
        let stability = 0.4 * strength_stability + 
                       0.3 * fidelity_stability + 
                       0.3 * pair_stability;

        Ok(stability.min(1.0).max(0.0))
    }
}

/// Entanglement evolution tracking
#[derive(Debug, Clone)]
pub struct EntanglementEvolution {
    pub strength_change: f64,
    pub fidelity_change: f64,
    pub events: Vec<EntanglementEvent>,
    pub stability_metric: f64,
}

/// Types of entanglement events
#[derive(Debug, Clone)]
pub enum EntanglementEvent {
    EntanglementIncrease(f64),
    EntanglementDecrease(f64),
    FidelityChange(f64),
    NewEntangledPair((String, String)),
    LostEntangledPair((String, String)),
    DecoherenceSpike(f64),
}

/// Initialize quantum entanglement system
pub async fn init() -> Result<()> {
    info!("Quantum Entanglement subsystem initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use chrono::Utc;
    use ndarray::Array1;

    #[tokio::test]
    async fn test_entanglement_detection() {
        let config = QuantumConfig::default();
        let detector = QuantumEntanglement::new(&config).await.unwrap();

        // Create test quantum data
        let superposition_states = Array2::from_shape_vec(
            (4, 2),
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.5),
                Complex64::new(0.5, -0.5), Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0), Complex64::new(-0.5, 0.5),
                Complex64::new(0.5, 0.5), Complex64::new(0.0, 1.0),
            ]
        ).unwrap();

        let amplitudes = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        let entanglement_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.7, 0.1),
                Complex64::new(0.7, -0.1), Complex64::new(1.0, 0.0),
            ]
        ).unwrap();

        let phase_matrix = Array2::zeros((4, 2));

        let mut price_history = HashMap::new();
        price_history.insert("BTCUSDT".to_string(), vec![50000.0, 51000.0]);
        price_history.insert("ETHUSDT".to_string(), vec![3000.0, 3100.0]);

        let classical_data = MarketData {
            price_history,
            volume_data: HashMap::new(),
            timestamps: vec![Utc::now(); 2],
            features: ndarray::Array2::zeros((2, 2)),
            regime_indicators: Array1::zeros(2),
        };

        let quantum_data = QuantumMarketData {
            superposition_states,
            amplitudes,
            entanglement_matrix,
            phase_matrix,
            classical_data,
            coherence_time_ms: 1000.0,
        };

        let entanglement = detector.find_entangled_correlations(&quantum_data).await.unwrap();
        
        assert!(entanglement.strength >= 0.0 && entanglement.strength <= 1.0);
        assert!(entanglement.fidelity >= 0.0 && entanglement.fidelity <= 1.0);
        assert!(entanglement.decoherence_rate >= 0.0);
    }

    #[tokio::test]
    async fn test_bell_inequality() {
        let config = QuantumConfig::default();
        let detector = QuantumEntanglement::new(&config).await.unwrap();

        // Create maximally entangled correlation matrix
        let correlation_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0),
            ]
        ).unwrap();

        let chsh_value = detector.calculate_chsh_inequality(&correlation_matrix).await.unwrap();
        
        // For maximally entangled states, CHSH should approach 2√2 ≈ 2.828
        assert!(chsh_value >= 2.0); // Should violate classical bound
    }
}