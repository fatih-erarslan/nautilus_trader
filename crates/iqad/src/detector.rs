//! Main IQAD detector implementation

use crate::cache::{QuantumCache, CacheKey};
use crate::error::IqadResult;
use crate::hardware;
use crate::immune_system::ImmuneSystem;
use crate::quantum_circuits::{QuantumBackend, QuantumCircuits};
use crate::types::*;
use ndarray::Array1;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Main Immune-Inspired Quantum Anomaly Detector
pub struct ImmuneQuantumAnomalyDetector {
    /// Configuration
    config: IqadConfig,
    /// Quantum backend
    quantum_backend: Arc<Mutex<QuantumBackend>>,
    /// Quantum circuits
    quantum_circuits: Arc<QuantumCircuits>,
    /// Immune system
    immune_system: Arc<RwLock<ImmuneSystem>>,
    /// Quantum cache
    cache: Arc<QuantumCache>,
    /// Initialization status
    is_initialized: bool,
    /// Execution times for performance tracking
    execution_times: Arc<RwLock<Vec<f64>>>,
}

impl ImmuneQuantumAnomalyDetector {
    /// Create a new IQAD instance
    pub async fn new(config: IqadConfig) -> IqadResult<Self> {
        // Initialize logging
        Self::init_logging(&config.log_level)?;
        
        info!("Initializing IQAD with quantum dimension {}", config.quantum_dimension);
        
        // Detect hardware capabilities
        let hw_caps = hardware::detect_hardware();
        info!("Hardware capabilities: {:?}", hw_caps);
        
        // Create quantum backend
        let quantum_backend = QuantumBackend::new(
            config.quantum_dimension,
            config.quantum_shots,
        )?;
        
        // Create quantum circuits
        let quantum_circuits = QuantumCircuits::new(config.quantum_dimension);
        
        // Create immune system
        let immune_system = ImmuneSystem::new(
            config.num_detectors,
            config.max_self_patterns,
            config.max_anomaly_memory,
            config.mutation_rate,
            config.negative_selection_threshold,
        );
        
        // Create cache
        let cache = QuantumCache::new(config.cache_size as u64);
        
        let mut detector = Self {
            config,
            quantum_backend: Arc::new(Mutex::new(quantum_backend)),
            quantum_circuits: Arc::new(quantum_circuits),
            immune_system: Arc::new(RwLock::new(immune_system)),
            cache: Arc::new(cache),
            is_initialized: false,
            execution_times: Arc::new(RwLock::new(Vec::with_capacity(100))),
        };
        
        // Initialize
        detector.initialize().await?;
        
        Ok(detector)
    }
    
    /// Initialize the detector
    async fn initialize(&mut self) -> IqadResult<()> {
        if self.is_initialized {
            return Ok(());
        }
        
        info!("Initializing IQAD detector");
        
        // Initialize immune system detectors
        {
            let mut immune = self.immune_system.write();
            immune.initialize_detectors(2_usize.pow(self.config.quantum_dimension as u32));
        }
        
        self.is_initialized = true;
        info!("IQAD detector initialized successfully");
        
        Ok(())
    }
    
    /// Train on normal data
    pub async fn train_on_normal_data(&self, normal_patterns: Vec<std::collections::HashMap<String, f64>>) -> IqadResult<()> {
        info!("Training IQAD on {} normal patterns", normal_patterns.len());
        
        for pattern in normal_patterns {
            let encoded = self.encode_features(&pattern);
            self.learn_normal_pattern(encoded).await?;
        }
        
        Ok(())
    }
    
    /// Detect anomalies in features
    pub async fn detect_anomalies(
        &self,
        features: std::collections::HashMap<String, f64>,
        expected_behavior: Option<std::collections::HashMap<String, serde_json::Value>>,
        regime_transitions: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> IqadResult<AnomalyResult> {
        let start = Instant::now();
        
        debug!("Starting anomaly detection");
        
        // Encode features
        let encoded = self.encode_features(&features);
        
        // Calculate detector activation
        let activation = self.calculate_detector_activation(&encoded).await?;
        debug!("Detector activation: {:.4}", activation);
        
        // Calculate anomaly score
        let (anomaly_score, detector_affinities) = self.calculate_anomaly_score(&encoded).await?;
        debug!("Raw anomaly score: {:.4}", anomaly_score);
        
        // Apply sensitivity adjustment
        let adjusted_score = anomaly_score * self.config.sensitivity;
        
        // Determine threshold
        let mut threshold = 0.7;
        
        // Adjust threshold based on expected behavior
        if let Some(ref expected) = expected_behavior {
            if let Some(volatility) = expected.get("volatility").and_then(|v| v.as_f64()) {
                threshold += 0.1 * volatility;
            }
        }
        
        // Adjust threshold based on regime transitions
        if let Some(ref transitions) = regime_transitions {
            if let Some(prob) = transitions.get("probability").and_then(|v| v.as_f64()) {
                threshold -= 0.1 * prob;
            }
        }
        
        // Determine if anomaly
        let is_anomaly = adjusted_score > threshold;
        debug!("Adjusted score: {:.4}, threshold: {:.4}, is anomaly: {}", adjusted_score, threshold, is_anomaly);
        
        // Handle detection result
        if is_anomaly {
            let mut immune = self.immune_system.write();
            immune.memorize_anomaly(encoded, adjusted_score);
            info!("Anomaly detected with score: {:.4}", adjusted_score);
        } else if adjusted_score < 0.3 {
            self.learn_normal_pattern(encoded).await?;
            debug!("Normal pattern learned with score: {:.4}", adjusted_score);
        }
        
        // Track execution time
        let execution_time = start.elapsed().as_millis() as f64;
        self.track_execution_time(execution_time);
        
        // Determine time to event
        let time_to_event = if is_anomaly {
            if adjusted_score > 0.8 {
                Some(TimeToEvent::Imminent)
            } else if adjusted_score > threshold + 0.1 {
                Some(TimeToEvent::NearTerm)
            } else {
                Some(TimeToEvent::Potential)
            }
        } else {
            None
        };
        
        Ok(AnomalyResult {
            detected: is_anomaly,
            score: adjusted_score,
            threshold,
            confidence: (adjusted_score / threshold).min(1.0),
            detector_affinities: detector_affinities.into_iter().take(5).collect(),
            execution_time_ms: execution_time,
            time_to_event,
        })
    }
    
    /// Calculate tail probability for extreme events
    pub async fn calculate_tail_probability(&self, features: std::collections::HashMap<String, f64>, _quantile: f64) -> IqadResult<f64> {
        let encoded = self.encode_features(&features);
        
        // Get anomaly score
        let (anomaly_score, _) = self.calculate_anomaly_score(&encoded).await?;
        
        // Check against anomaly memory
        let memory_factor = {
            let immune = self.immune_system.read();
            immune.check_anomaly_memory(&encoded, |a, b| self.sync_quantum_affinity(a, b))
                .unwrap_or(0.0)
        };
        
        // Combine scores
        let combined_score = anomaly_score.max(memory_factor);
        
        // Apply extreme value scaling
        let tail_probability = combined_score.powf(2.0); // Square to emphasize extremes
        
        Ok(tail_probability.min(0.99))
    }
    
    /// Encode features into quantum-compatible vector
    fn encode_features(&self, features: &std::collections::HashMap<String, f64>) -> Array1<f64> {
        let key_features = vec![
            features.get("close").copied().unwrap_or(0.0),
            features.get("volume").copied().unwrap_or(0.0),
            features.get("volatility")
                .or_else(|| features.get("volatility_regime"))
                .copied()
                .unwrap_or(0.5),
            features.get("rsi_14")
                .or_else(|| features.get("rsi"))
                .copied()
                .unwrap_or(50.0) / 100.0,
            features.get("adx").copied().unwrap_or(15.0) / 100.0,
            features.get("trend").copied().unwrap_or(0.5),
            features.get("momentum").copied().unwrap_or(0.5),
            features.get("regime").copied().unwrap_or(0.5),
        ];
        
        Array1::from_vec(self.prepare_vector_for_quantum(key_features))
    }
    
    /// Prepare vector for quantum processing
    fn prepare_vector_for_quantum(&self, vec: Vec<f64>) -> Vec<f64> {
        let target_size = self.config.quantum_dimension;
        let mut result = vec;
        
        // Resize if needed
        if result.len() < target_size {
            result.resize(target_size, 0.0);
        } else if result.len() > target_size {
            result.truncate(target_size);
        }
        
        // Normalize to [0, 1]
        let min = result.iter().copied().fold(f64::INFINITY, f64::min);
        let max = result.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        if (max - min).abs() > 1e-10 {
            for val in &mut result {
                *val = (*val - min) / (max - min);
            }
        } else {
            result.fill(0.5);
        }
        
        result
    }
    
    /// Calculate quantum affinity (synchronous wrapper)
    fn sync_quantum_affinity(&self, pattern: &[f64], detector: &[f64]) -> f64 {
        // Software-simulated quantum affinity calculation
        // This is a synchronous version for use in closures
        if pattern.len() != detector.len() || pattern.is_empty() {
            return 0.0;
        }
        
        let mut overlap = 0.0;
        let mut norm_pattern = 0.0;
        let mut norm_detector = 0.0;
        
        for i in 0..pattern.len() {
            overlap += pattern[i] * detector[i];
            norm_pattern += pattern[i] * pattern[i];
            norm_detector += detector[i] * detector[i];
        }
        
        if norm_pattern > 0.0 && norm_detector > 0.0 {
            let affinity = overlap / (norm_pattern.sqrt() * norm_detector.sqrt());
            // Apply quantum interference pattern
            0.5 * (1.0 + affinity.cos())
        } else {
            0.0
        }
    }
    
    /// Calculate quantum affinity between pattern and detector
    async fn quantum_affinity(&self, pattern: &[f64], detector: &[f64]) -> IqadResult<f64> {
        // Check cache
        let cache_key = CacheKey::new("affinity", &[pattern, detector].concat());
        
        if let Some(cached) = self.cache.get(&cache_key).await {
            return Ok(cached[0]);
        }
        
        // Build quantum circuit
        let circuit = self.quantum_circuits.build_affinity_circuit(pattern, detector)?;
        
        // Execute circuit
        let mut backend = self.quantum_backend.lock().await;
        let measurements = backend.execute(&circuit)?;
        
        // Calculate affinity from measurements
        let affinity = measurements.iter().sum::<f64>() / measurements.len() as f64;
        
        // Cache result
        self.cache.insert(cache_key, vec![affinity]).await;
        
        Ok(affinity)
    }
    
    /// Calculate detector activation
    async fn calculate_detector_activation(&self, pattern: &Array1<f64>) -> IqadResult<f64> {
        let activation = {
            let mut immune = self.immune_system.write();
            immune.calculate_activation(pattern, |a, b| {
                // Use quantum distance
                self.sync_quantum_distance(a, b)
            })
        };
        
        Ok(activation)
    }
    
    /// Calculate quantum distance (synchronous wrapper)
    fn sync_quantum_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        // Simplified Euclidean distance as fallback
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    
    /// Calculate anomaly score
    async fn calculate_anomaly_score(&self, pattern: &Array1<f64>) -> IqadResult<(f64, Vec<f64>)> {
        let immune = self.immune_system.read();
        let detector_affinities = immune.get_detector_affinities(
            pattern,
            |a, b| self.sync_quantum_affinity(a, b),
            10,
        );
        
        let max_affinity = detector_affinities.iter().copied().fold(0.0_f64, f64::max);
        
        Ok((max_affinity, detector_affinities))
    }
    
    /// Learn a normal pattern
    async fn learn_normal_pattern(&self, pattern: Array1<f64>) -> IqadResult<()> {
        let mut immune = self.immune_system.write();
        immune.learn_normal_pattern(pattern.clone());
        
        // Update detectors
        immune.update_detectors(&pattern, |a, b| self.sync_quantum_affinity(a, b));
        
        Ok(())
    }
    
    /// Track execution time
    fn track_execution_time(&self, time_ms: f64) {
        let mut times = self.execution_times.write();
        times.push(time_ms);
        
        // Keep last 100 times
        if times.len() > 100 {
            times.remove(0);
        }
    }
    
    /// Get execution statistics
    pub fn get_execution_stats(&self) -> ExecutionStats {
        let times = self.execution_times.read();
        
        if times.is_empty() {
            return ExecutionStats::default();
        }
        
        let sum: f64 = times.iter().sum();
        let avg = sum / times.len() as f64;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        ExecutionStats {
            avg_time_ms: avg,
            min_time_ms: min,
            max_time_ms: max,
            count: times.len(),
        }
    }
    
    /// Clear cache
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
        info!("Cache cleared");
    }
    
    /// Reset detector
    pub async fn reset(&mut self) -> IqadResult<()> {
        info!("Resetting IQAD detector");
        
        // Clear immune system
        {
            let mut immune = self.immune_system.write();
            *immune = ImmuneSystem::new(
                self.config.num_detectors,
                self.config.max_self_patterns,
                self.config.max_anomaly_memory,
                self.config.mutation_rate,
                self.config.negative_selection_threshold,
            );
            immune.initialize_detectors(2_usize.pow(self.config.quantum_dimension as u32));
        }
        
        // Clear cache
        self.clear_cache().await;
        
        // Clear execution times
        self.execution_times.write().clear();
        
        Ok(())
    }
    
    /// Simplified detect method for benchmarks
    pub async fn detect(&self, data: &[f64]) -> IqadResult<AnomalyResult> {
        // Convert array data to feature map
        let mut features = std::collections::HashMap::new();
        for (i, &value) in data.iter().enumerate() {
            features.insert(format!("feature_{}", i), value);
        }
        
        self.detect_anomalies(features, None, None).await
    }
    
    /// Batch detection for benchmarks
    pub async fn detect_batch(&self, batch: &[Vec<f64>]) -> IqadResult<Vec<AnomalyResult>> {
        let mut results = Vec::with_capacity(batch.len());
        
        for data in batch {
            results.push(self.detect(data).await?);
        }
        
        Ok(results)
    }
    
    /// Initialize logging
    fn init_logging(level: &str) -> IqadResult<()> {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        
        let level = match level.to_uppercase().as_str() {
            "ERROR" => tracing::Level::ERROR,
            "WARN" => tracing::Level::WARN,
            "INFO" => tracing::Level::INFO,
            "DEBUG" => tracing::Level::DEBUG,
            "TRACE" => tracing::Level::TRACE,
            _ => tracing::Level::INFO,
        };
        
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new(
                format!("iqad={},roqoqo=warn", level)
            ))
            .with(tracing_subscriber::fmt::layer())
            .try_init()
            .ok(); // Ignore if already initialized
            
        Ok(())
    }
}

/// Execution statistics
#[derive(Debug, Default, Clone)]
pub struct ExecutionStats {
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub count: usize,
}

// Implement Clone for ImmuneQuantumAnomalyDetector
impl Clone for ImmuneQuantumAnomalyDetector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            quantum_backend: Arc::clone(&self.quantum_backend),
            quantum_circuits: Arc::clone(&self.quantum_circuits),
            immune_system: Arc::clone(&self.immune_system),
            cache: Arc::clone(&self.cache),
            is_initialized: self.is_initialized,
            execution_times: Arc::clone(&self.execution_times),
        }
    }
}

// Implement Send + Sync for async usage
unsafe impl Send for ImmuneQuantumAnomalyDetector {}
unsafe impl Sync for ImmuneQuantumAnomalyDetector {}