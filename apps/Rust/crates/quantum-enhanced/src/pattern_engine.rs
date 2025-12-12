//! Main quantum pattern detection engine

use crate::types::*;
use crate::quantum_superposition::QuantumSuperposition;
use crate::quantum_entanglement::QuantumEntanglement;
use crate::quantum_fourier::QuantumFourierTransform;
use crate::performance::PerformanceMonitor;
use crate::Result;

use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;
use std::collections::HashMap;
use tracing::{info, debug, warn};

/// Main quantum pattern detection engine
pub struct QuantumPatternEngine {
    /// Quantum superposition detector
    quantum_superposition_detector: QuantumSuperposition,
    /// Quantum entanglement correlation finder
    entanglement_correlation_finder: QuantumEntanglement,
    /// Quantum Fourier transform engine
    quantum_fourier_transform: QuantumFourierTransform,
    /// Performance monitoring
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    /// Configuration
    config: QuantumConfig,
    /// Pattern cache for efficiency
    pattern_cache: Arc<RwLock<HashMap<String, QuantumSignal>>>,
}

impl QuantumPatternEngine {
    /// Create a new quantum pattern engine
    pub async fn new(config: QuantumConfig) -> Result<Self> {
        info!("Initializing Quantum Pattern Engine with {} max superposition states", 
              config.max_superposition_states);

        let quantum_superposition_detector = QuantumSuperposition::new(&config).await?;
        let entanglement_correlation_finder = QuantumEntanglement::new(&config).await?;
        let quantum_fourier_transform = QuantumFourierTransform::new(&config).await?;
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));

        Ok(Self {
            quantum_superposition_detector,
            entanglement_correlation_finder,
            quantum_fourier_transform,
            performance_monitor,
            config,
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Detect quantum patterns using superposition of all possible states
    pub async fn detect_quantum_patterns(&self, market_data: &MarketData) -> Result<QuantumSignal> {
        let start_time = Instant::now();
        
        debug!("Starting quantum pattern detection for {} instruments", 
               market_data.price_history.len());

        // Step 1: Create quantum superposition of all possible price paths
        let superposition_states = self.quantum_superposition_detector
            .create_superposition(market_data)
            .await?;

        debug!("Created superposition with {} quantum states", 
               superposition_states.superposition_states.nrows());

        // Step 2: Apply quantum entanglement to find non-local correlations
        let entangled_patterns = self.entanglement_correlation_finder
            .find_entangled_correlations(&superposition_states)
            .await?;

        debug!("Found {} entangled correlation pairs", 
               entangled_patterns.entangled_pairs.len());

        // Step 3: Use Quantum Fourier Transform for frequency domain analysis
        let quantum_frequencies = self.quantum_fourier_transform
            .transform(&entangled_patterns)
            .await?;

        debug!("Extracted {} dominant frequencies", 
               quantum_frequencies.dominant_frequencies.len());

        // Step 4: Collapse superposition to classical signal
        let quantum_signal = self.collapse_to_classical_signal(
            &superposition_states,
            &entangled_patterns,
            &quantum_frequencies,
        ).await?;

        let execution_time = start_time.elapsed().as_micros() as u64;

        // Update performance metrics
        self.update_performance_metrics(execution_time, &quantum_signal).await?;

        info!("Quantum pattern detection completed in {}μs, signal strength: {:.3}, confidence: {:.3}",
              execution_time, quantum_signal.strength, quantum_signal.confidence);

        Ok(quantum_signal)
    }

    /// Detect multiple quantum patterns with ensemble approach
    pub async fn detect_ensemble_patterns(&self, market_data: &MarketData) -> Result<Vec<QuantumSignal>> {
        let mut ensemble_signals = Vec::new();
        
        // Create multiple quantum interpretations
        let quantum_configs = self.generate_ensemble_configs().await?;
        
        for config in quantum_configs {
            // Create temporary engine with different configuration
            let temp_engine = QuantumPatternEngine::new(config).await?;
            
            // Detect patterns with this configuration
            match temp_engine.detect_quantum_patterns(market_data).await {
                Ok(signal) => ensemble_signals.push(signal),
                Err(e) => warn!("Ensemble detection failed: {}", e),
            }
        }

        // Combine ensemble results
        let combined_signal = self.combine_ensemble_signals(ensemble_signals).await?;
        
        Ok(vec![combined_signal])
    }

    /// Validate quantum pattern against classical analysis
    pub async fn validate_quantum_pattern(
        &self, 
        quantum_signal: &QuantumSignal,
        market_data: &MarketData
    ) -> Result<QuantumValidationResult> {
        
        // Calculate classical correlation
        let classical_correlation = self.calculate_classical_correlation(quantum_signal, market_data).await?;
        
        // Estimate pattern persistence
        let persistence_time = self.estimate_pattern_persistence(quantum_signal).await?;
        
        // Determine signal validity
        let is_valid = quantum_signal.confidence > self.config.coherence_threshold &&
                      classical_correlation > 0.3 &&
                      persistence_time > 1000.0; // At least 1 second persistence

        Ok(QuantumValidationResult {
            is_valid,
            confidence: quantum_signal.confidence,
            classical_correlation,
            persistence_time_ms: persistence_time,
            signal_strength: quantum_signal.strength,
        })
    }

    /// Get real-time performance metrics
    pub async fn get_performance_metrics(&self) -> Result<QuantumPerformanceMetrics> {
        let monitor = self.performance_monitor.read().await;
        Ok(monitor.get_current_metrics())
    }

    /// Cache quantum pattern for efficiency
    pub async fn cache_pattern(&self, key: String, signal: QuantumSignal) -> Result<()> {
        let mut cache = self.pattern_cache.write().await;
        cache.insert(key, signal);
        
        // Limit cache size to prevent memory bloat
        if cache.len() > 1000 {
            let oldest_key = cache.keys().next().cloned();
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
        
        Ok(())
    }

    /// Retrieve cached pattern
    pub async fn get_cached_pattern(&self, key: &str) -> Option<QuantumSignal> {
        let cache = self.pattern_cache.read().await;
        cache.get(key).cloned()
    }

    // Private helper methods

    async fn collapse_to_classical_signal(
        &self,
        superposition_states: &QuantumMarketData,
        entangled_patterns: &EntanglementCorrelation,
        quantum_frequencies: &QuantumFourierResult,
    ) -> Result<QuantumSignal> {
        
        // Calculate signal strength from quantum amplitudes
        let signal_strength = self.calculate_signal_strength(superposition_states, quantum_frequencies).await?;
        
        // Determine pattern type from quantum characteristics
        let pattern_type = self.classify_pattern_type(entangled_patterns, quantum_frequencies).await?;
        
        // Calculate confidence from coherence and entanglement
        let confidence = (superposition_states.coherence_time_ms / 1000.0).min(1.0) * 
                        entangled_patterns.fidelity * 
                        quantum_frequencies.spectral_coherence;
        
        // Create quantum signal
        let mut signal = QuantumSignal::new(
            signal_strength,
            confidence,
            pattern_type,
            superposition_states.coherence_time_ms / 1000.0,
        );

        // Add entanglement information
        for (pair_idx, &(ref inst1, ref inst2)) in entangled_patterns.entangled_pairs.iter().enumerate() {
            let correlation_strength = entangled_patterns.correlation_matrix[[pair_idx, pair_idx]].norm();
            signal.entanglement_map.insert(
                format!("{}:{}", inst1, inst2),
                correlation_strength,
            );
        }

        // Add frequency signature
        signal.frequency_signature = quantum_frequencies.dominant_frequencies.clone().into();
        
        // Add affected instruments
        signal.affected_instruments = superposition_states.classical_data.price_history.keys().cloned().collect();

        Ok(signal)
    }

    async fn calculate_signal_strength(
        &self,
        superposition_states: &QuantumMarketData,
        quantum_frequencies: &QuantumFourierResult,
    ) -> Result<f64> {
        
        // Calculate weighted average of quantum amplitudes
        let amplitude_sum: f64 = superposition_states.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        let normalized_amplitude = amplitude_sum / superposition_states.amplitudes.len() as f64;
        
        // Weight by dominant frequency strength
        let frequency_weight = quantum_frequencies.dominant_frequencies.iter()
            .map(|&freq| freq.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        // Combine amplitude and frequency information
        let signal_strength = (normalized_amplitude * frequency_weight).tanh();
        
        // Apply sign based on phase information
        let phase_sign = if quantum_frequencies.phase_spectrum.iter().sum::<f64>() > 0.0 { 1.0 } else { -1.0 };
        
        Ok(signal_strength * phase_sign)
    }

    async fn classify_pattern_type(
        &self,
        entangled_patterns: &EntanglementCorrelation,
        quantum_frequencies: &QuantumFourierResult,
    ) -> Result<QuantumPatternType> {
        
        // Classify based on entanglement and frequency characteristics
        if entangled_patterns.strength > 0.8 && entangled_patterns.fidelity > 0.9 {
            Ok(QuantumPatternType::EntangledCorrelation)
        } else if quantum_frequencies.spectral_coherence > 0.8 {
            Ok(QuantumPatternType::CoherentOscillation)
        } else if entangled_patterns.decoherence_rate < 0.1 {
            Ok(QuantumPatternType::SuperpositionMomentum)
        } else if quantum_frequencies.dominant_frequencies.len() > 5 {
            Ok(QuantumPatternType::QuantumInterference)
        } else {
            Ok(QuantumPatternType::QuantumResonance)
        }
    }

    async fn generate_ensemble_configs(&self) -> Result<Vec<QuantumConfig>> {
        let mut configs = Vec::new();
        
        // Generate configurations with different quantum parameters
        for coherence_threshold in [0.5, 0.7, 0.9] {
            for entanglement_sensitivity in [0.3, 0.5, 0.7] {
                let mut config = self.config.clone();
                config.coherence_threshold = coherence_threshold;
                config.entanglement_sensitivity = entanglement_sensitivity;
                configs.push(config);
            }
        }
        
        Ok(configs)
    }

    async fn combine_ensemble_signals(&self, signals: Vec<QuantumSignal>) -> Result<QuantumSignal> {
        if signals.is_empty() {
            return Err(crate::QuantumError::PatternCollapse("No ensemble signals to combine".to_string()));
        }

        // Calculate weighted average of signals
        let total_confidence: f64 = signals.iter().map(|s| s.confidence).sum();
        let weighted_strength: f64 = signals.iter()
            .map(|s| s.strength * s.confidence)
            .sum::<f64>() / total_confidence;
        
        let avg_confidence = total_confidence / signals.len() as f64;
        let avg_coherence: f64 = signals.iter().map(|s| s.coherence).sum::<f64>() / signals.len() as f64;

        // Use the most common pattern type
        let mut pattern_counts: HashMap<QuantumPatternType, usize> = HashMap::new();
        for signal in &signals {
            *pattern_counts.entry(signal.pattern_type.clone()).or_insert(0) += 1;
        }
        
        let most_common_pattern = pattern_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(pattern, _)| pattern)
            .unwrap_or(QuantumPatternType::QuantumResonance);

        let mut combined_signal = QuantumSignal::new(
            weighted_strength,
            avg_confidence,
            most_common_pattern,
            avg_coherence,
        );

        // Combine entanglement maps
        for signal in signals {
            for (key, value) in signal.entanglement_map {
                let entry = combined_signal.entanglement_map.entry(key).or_insert(0.0);
                *entry = (*entry + value) / 2.0; // Average entanglement strengths
            }
        }

        Ok(combined_signal)
    }

    async fn calculate_classical_correlation(
        &self,
        quantum_signal: &QuantumSignal,
        market_data: &MarketData,
    ) -> Result<f64> {
        // Simple correlation calculation for validation
        // This would be more sophisticated in production
        Ok(quantum_signal.confidence * 0.8) // Placeholder
    }

    async fn estimate_pattern_persistence(&self, quantum_signal: &QuantumSignal) -> Result<f64> {
        // Estimate how long the pattern will persist based on coherence
        let base_persistence = quantum_signal.coherence * 5000.0; // Base persistence in ms
        let confidence_multiplier = quantum_signal.confidence;
        
        Ok(base_persistence * confidence_multiplier)
    }

    async fn update_performance_metrics(
        &self,
        execution_time_us: u64,
        quantum_signal: &QuantumSignal,
    ) -> Result<()> {
        let mut monitor = self.performance_monitor.write().await;
        monitor.record_detection(execution_time_us, quantum_signal.confidence);
        Ok(())
    }
}

impl Clone for QuantumPatternEngine {
    fn clone(&self) -> Self {
        // Create a new instance with the same configuration
        // Note: This is an async operation but clone must be sync
        // In practice, you might want to use Arc<QuantumPatternEngine> instead
        Self {
            quantum_superposition_detector: self.quantum_superposition_detector.clone(),
            entanglement_correlation_finder: self.entanglement_correlation_finder.clone(),
            quantum_fourier_transform: self.quantum_fourier_transform.clone(),
            performance_monitor: Arc::clone(&self.performance_monitor),
            config: self.config.clone(),
            pattern_cache: Arc::clone(&self.pattern_cache),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use chrono::Utc;
    use ndarray::Array2;

    #[tokio::test]
    async fn test_quantum_pattern_detection() {
        let config = QuantumConfig::default();
        let engine = QuantumPatternEngine::new(config).await.unwrap();

        // Create sample market data
        let mut price_history = HashMap::new();
        price_history.insert("BTCUSDT".to_string(), vec![50000.0, 51000.0, 49000.0, 52000.0]);
        price_history.insert("ETHUSDT".to_string(), vec![3000.0, 3100.0, 2900.0, 3200.0]);

        let mut volume_data = HashMap::new();
        volume_data.insert("BTCUSDT".to_string(), vec![100.0, 110.0, 90.0, 120.0]);
        volume_data.insert("ETHUSDT".to_string(), vec![200.0, 210.0, 190.0, 220.0]);

        let market_data = MarketData {
            price_history,
            volume_data,
            timestamps: vec![Utc::now(); 4],
            features: Array2::zeros((4, 2)),
            regime_indicators: Array1::zeros(4),
        };

        let signal = engine.detect_quantum_patterns(&market_data).await.unwrap();
        
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.strength >= -1.0 && signal.strength <= 1.0);
        assert!(signal.coherence >= 0.0);
    }

    #[tokio::test]
    async fn test_ensemble_detection() {
        let config = QuantumConfig::default();
        let engine = QuantumPatternEngine::new(config).await.unwrap();

        let mut price_history = HashMap::new();
        price_history.insert("BTCUSDT".to_string(), vec![50000.0, 51000.0]);

        let market_data = MarketData {
            price_history,
            volume_data: HashMap::new(),
            timestamps: vec![Utc::now(); 2],
            features: Array2::zeros((2, 1)),
            regime_indicators: Array1::zeros(2),
        };

        let signals = engine.detect_ensemble_patterns(&market_data).await.unwrap();
        
        assert!(!signals.is_empty());
        assert!(signals[0].confidence >= 0.0);
    }
}