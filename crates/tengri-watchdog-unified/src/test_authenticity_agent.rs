//! Test Authenticity Agent
//! 
//! Advanced verification system for authentic data flows and real system interactions
//! Ensures all testing uses genuine data sources and authentic system behavior

use crate::{TENGRIError, TradingOperation, TENGRIOversightResult, ViolationType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use blake3::Hasher;
use sha3::{Digest, Sha3_256};

/// Test Authenticity Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAuthenticityConfig {
    pub require_entropy_analysis: bool,
    pub require_behavior_patterns: bool,
    pub require_temporal_consistency: bool,
    pub require_data_flow_validation: bool,
    pub entropy_threshold: f64,
    pub pattern_consistency_threshold: f64,
    pub temporal_variance_threshold: f64,
    pub data_flow_integrity_threshold: f64,
    pub enable_quantum_verification: bool,
    pub enable_behavioral_analysis: bool,
    pub suspicious_pattern_sensitivity: f64,
}

impl Default for TestAuthenticityConfig {
    fn default() -> Self {
        Self {
            require_entropy_analysis: true,
            require_behavior_patterns: true,
            require_temporal_consistency: true,
            require_data_flow_validation: true,
            entropy_threshold: 0.85,
            pattern_consistency_threshold: 0.90,
            temporal_variance_threshold: 0.15,
            data_flow_integrity_threshold: 0.95,
            enable_quantum_verification: true,
            enable_behavioral_analysis: true,
            suspicious_pattern_sensitivity: 0.80,
        }
    }
}

/// Data Flow Authenticity Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowAnalysis {
    pub flow_id: String,
    pub source_authenticity: f64,
    pub transformation_integrity: f64,
    pub temporal_consistency: f64,
    pub entropy_score: f64,
    pub behavioral_patterns: Vec<BehavioralPattern>,
    pub anomaly_indicators: Vec<AnomalyIndicator>,
    pub quantum_signature: Vec<u8>,
    pub authenticity_verdict: AuthenticityVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub confidence: f64,
    pub expected_range: (f64, f64),
    pub observed_value: f64,
    pub is_authentic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyIndicator {
    pub indicator_type: String,
    pub severity: AnomalySeverity,
    pub description: String,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticityVerdict {
    FullyAuthentic,
    MostlyAuthentic,
    QuestionableAuthenticity,
    LikelyInauthentic,
    DefinitelyInauthentic,
}

/// Entropy Analysis Engine
pub struct EntropyAnalyzer {
    config: TestAuthenticityConfig,
    entropy_baselines: HashMap<String, f64>,
    pattern_library: HashMap<String, Vec<f64>>,
}

impl EntropyAnalyzer {
    pub fn new(config: TestAuthenticityConfig) -> Self {
        Self {
            config,
            entropy_baselines: HashMap::new(),
            pattern_library: HashMap::new(),
        }
    }

    /// Calculate Shannon entropy of data
    pub fn calculate_shannon_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut frequency = HashMap::new();
        for &byte in data {
            *frequency.entry(byte).or_insert(0) += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for count in frequency.values() {
            let p = *count as f64 / len;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy / 8.0 // Normalize to [0, 1]
    }

    /// Calculate block entropy for temporal patterns
    pub fn calculate_block_entropy(&self, data: &[u8], block_size: usize) -> f64 {
        if data.len() < block_size {
            return self.calculate_shannon_entropy(data);
        }

        let mut block_entropies = Vec::new();
        for chunk in data.chunks(block_size) {
            block_entropies.push(self.calculate_shannon_entropy(chunk));
        }

        // Calculate variance in block entropies
        let mean = block_entropies.iter().sum::<f64>() / block_entropies.len() as f64;
        let variance = block_entropies.iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>() / block_entropies.len() as f64;

        mean * (1.0 - variance) // Higher variance suggests more authentic randomness
    }

    /// Analyze compression ratio for authenticity
    pub fn analyze_compression_ratio(&self, data: &[u8]) -> f64 {
        // Simple compression analysis using repetition detection
        let mut unique_sequences = HashSet::new();
        let window_size = 8;
        
        for window in data.windows(window_size) {
            unique_sequences.insert(window.to_vec());
        }

        if data.len() < window_size {
            return 1.0;
        }

        let compression_ratio = unique_sequences.len() as f64 / (data.len() - window_size + 1) as f64;
        compression_ratio.min(1.0)
    }
}

/// Behavioral Pattern Analyzer
pub struct BehavioralPatternAnalyzer {
    config: TestAuthenticityConfig,
    known_patterns: HashMap<String, Vec<BehavioralPattern>>,
    anomaly_detectors: HashMap<String, AnomalyDetector>,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    pub detector_type: String,
    pub threshold: f64,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub sample_count: usize,
}

impl BehavioralPatternAnalyzer {
    pub fn new(config: TestAuthenticityConfig) -> Self {
        Self {
            config,
            known_patterns: HashMap::new(),
            anomaly_detectors: HashMap::new(),
        }
    }

    /// Analyze behavioral patterns in data
    pub fn analyze_behavioral_patterns(&self, data: &[u8], context: &str) -> Vec<BehavioralPattern> {
        let mut patterns = Vec::new();

        // Analyze frequency patterns
        let frequency_pattern = self.analyze_frequency_patterns(data);
        patterns.push(frequency_pattern);

        // Analyze temporal patterns
        let temporal_pattern = self.analyze_temporal_patterns(data);
        patterns.push(temporal_pattern);

        // Analyze distribution patterns
        let distribution_pattern = self.analyze_distribution_patterns(data);
        patterns.push(distribution_pattern);

        patterns
    }

    fn analyze_frequency_patterns(&self, data: &[u8]) -> BehavioralPattern {
        let mut frequencies = HashMap::new();
        for &byte in data {
            *frequencies.entry(byte).or_insert(0) += 1;
        }

        let entropy = self.calculate_frequency_entropy(&frequencies, data.len());
        let expected_range = (0.7, 1.0); // Expected range for authentic data
        let is_authentic = entropy >= expected_range.0 && entropy <= expected_range.1;

        BehavioralPattern {
            pattern_type: "frequency".to_string(),
            frequency: entropy,
            confidence: 0.85,
            expected_range,
            observed_value: entropy,
            is_authentic,
        }
    }

    fn analyze_temporal_patterns(&self, data: &[u8]) -> BehavioralPattern {
        if data.len() < 4 {
            return BehavioralPattern {
                pattern_type: "temporal".to_string(),
                frequency: 0.0,
                confidence: 0.0,
                expected_range: (0.0, 1.0),
                observed_value: 0.0,
                is_authentic: false,
            };
        }

        // Calculate first-order differences
        let mut differences = Vec::new();
        for i in 1..data.len() {
            differences.push((data[i] as i16 - data[i-1] as i16).abs() as f64);
        }

        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = differences.iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>() / differences.len() as f64;

        let temporal_score = (variance.sqrt() / 255.0).min(1.0);
        let expected_range = (0.2, 0.8);
        let is_authentic = temporal_score >= expected_range.0 && temporal_score <= expected_range.1;

        BehavioralPattern {
            pattern_type: "temporal".to_string(),
            frequency: temporal_score,
            confidence: 0.80,
            expected_range,
            observed_value: temporal_score,
            is_authentic,
        }
    }

    fn analyze_distribution_patterns(&self, data: &[u8]) -> BehavioralPattern {
        // Chi-square test for uniform distribution
        let mut bins = vec![0; 256];
        for &byte in data {
            bins[byte as usize] += 1;
        }

        let expected = data.len() as f64 / 256.0;
        let mut chi_square = 0.0;
        for &observed in &bins {
            let diff = observed as f64 - expected;
            chi_square += diff * diff / expected;
        }

        // Normalize chi-square score
        let normalized_score = 1.0 - (chi_square / (data.len() as f64 * 256.0)).min(1.0);
        let expected_range = (0.4, 0.9);
        let is_authentic = normalized_score >= expected_range.0 && normalized_score <= expected_range.1;

        BehavioralPattern {
            pattern_type: "distribution".to_string(),
            frequency: normalized_score,
            confidence: 0.75,
            expected_range,
            observed_value: normalized_score,
            is_authentic,
        }
    }

    fn calculate_frequency_entropy(&self, frequencies: &HashMap<u8, usize>, total: usize) -> f64 {
        let mut entropy = 0.0;
        for &count in frequencies.values() {
            let p = count as f64 / total as f64;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy / 8.0 // Normalize to [0, 1]
    }

    /// Detect anomalies in behavioral patterns
    pub fn detect_anomalies(&self, patterns: &[BehavioralPattern]) -> Vec<AnomalyIndicator> {
        let mut anomalies = Vec::new();

        for pattern in patterns {
            if !pattern.is_authentic {
                let severity = if pattern.confidence > 0.9 {
                    AnomalySeverity::Critical
                } else if pattern.confidence > 0.7 {
                    AnomalySeverity::High
                } else if pattern.confidence > 0.5 {
                    AnomalySeverity::Medium
                } else {
                    AnomalySeverity::Low
                };

                anomalies.push(AnomalyIndicator {
                    indicator_type: format!("{}_anomaly", pattern.pattern_type),
                    severity,
                    description: format!("Behavioral pattern {} outside expected range", pattern.pattern_type),
                    confidence: pattern.confidence,
                    evidence: vec![
                        format!("Observed: {:.3}", pattern.observed_value),
                        format!("Expected: {:.3}-{:.3}", pattern.expected_range.0, pattern.expected_range.1),
                    ],
                });
            }
        }

        anomalies
    }
}

/// Quantum Signature Generator
pub struct QuantumSignatureGenerator {
    blake3_hasher: Hasher,
    sha3_hasher: Sha3_256,
}

impl QuantumSignatureGenerator {
    pub fn new() -> Self {
        Self {
            blake3_hasher: Hasher::new(),
            sha3_hasher: Sha3_256::new(),
        }
    }

    /// Generate quantum-resistant signature
    pub fn generate_signature(&mut self, data: &[u8]) -> Vec<u8> {
        // Multi-layer hash combination
        self.blake3_hasher.update(data);
        self.sha3_hasher.update(data);

        let blake3_hash = self.blake3_hasher.finalize();
        let sha3_hash = self.sha3_hasher.finalize();

        // Combine hashes with entropy mixing
        let mut combined = Vec::new();
        combined.extend_from_slice(blake3_hash.as_bytes());
        combined.extend_from_slice(&sha3_hash);

        // Add entropy-based mixing
        self.entropy_mix(&mut combined);

        combined
    }

    fn entropy_mix(&self, data: &mut Vec<u8>) {
        // Simple entropy mixing using XOR with pseudo-random sequence
        for (i, byte) in data.iter_mut().enumerate() {
            *byte ^= ((i * 17 + 37) % 256) as u8;
        }
    }
}

/// Test Authenticity Agent
pub struct TestAuthenticityAgent {
    config: TestAuthenticityConfig,
    entropy_analyzer: EntropyAnalyzer,
    behavioral_analyzer: BehavioralPatternAnalyzer,
    quantum_generator: Arc<RwLock<QuantumSignatureGenerator>>,
    analysis_history: Arc<RwLock<Vec<(DateTime<Utc>, DataFlowAnalysis)>>>,
    authenticity_scores: Arc<RwLock<HashMap<String, f64>>>,
}

impl TestAuthenticityAgent {
    /// Initialize Test Authenticity Agent
    pub async fn new(config: TestAuthenticityConfig) -> Result<Self, TENGRIError> {
        let entropy_analyzer = EntropyAnalyzer::new(config.clone());
        let behavioral_analyzer = BehavioralPatternAnalyzer::new(config.clone());
        let quantum_generator = Arc::new(RwLock::new(QuantumSignatureGenerator::new()));
        let analysis_history = Arc::new(RwLock::new(Vec::new()));
        let authenticity_scores = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            entropy_analyzer,
            behavioral_analyzer,
            quantum_generator,
            analysis_history,
            authenticity_scores,
        })
    }

    /// Verify test authenticity for trading operation
    pub async fn verify_authenticity(&self, operation: &TradingOperation) -> Result<TENGRIOversightResult, TENGRIError> {
        let verification_start = Instant::now();
        
        // Perform comprehensive authenticity analysis
        let analysis = self.perform_authenticity_analysis(operation).await?;
        
        // Record analysis history
        self.record_analysis_history(&analysis).await;
        
        // Update authenticity scores
        self.update_authenticity_scores(&analysis, operation).await;
        
        // Convert to oversight result
        let oversight_result = self.convert_to_oversight_result(&analysis).await?;

        let verification_duration = verification_start.elapsed();
        if verification_duration.as_millis() > 200 {
            tracing::warn!("Test authenticity verification exceeded 200ms: {:?}", verification_duration);
        }

        Ok(oversight_result)
    }

    /// Perform comprehensive authenticity analysis
    async fn perform_authenticity_analysis(&self, operation: &TradingOperation) -> Result<DataFlowAnalysis, TENGRIError> {
        let data = operation.data_source.as_bytes();
        
        // Entropy analysis
        let entropy_score = if self.config.require_entropy_analysis {
            self.entropy_analyzer.calculate_shannon_entropy(data)
        } else {
            1.0
        };

        // Behavioral pattern analysis
        let behavioral_patterns = if self.config.require_behavior_patterns {
            self.behavioral_analyzer.analyze_behavioral_patterns(data, &operation.agent_id)
        } else {
            Vec::new()
        };

        // Anomaly detection
        let anomaly_indicators = self.behavioral_analyzer.detect_anomalies(&behavioral_patterns);

        // Temporal consistency analysis
        let temporal_consistency = if self.config.require_temporal_consistency {
            self.analyze_temporal_consistency(data).await?
        } else {
            1.0
        };

        // Data flow validation
        let transformation_integrity = if self.config.require_data_flow_validation {
            self.validate_data_flow_integrity(data).await?
        } else {
            1.0
        };

        // Quantum signature generation
        let quantum_signature = if self.config.enable_quantum_verification {
            let mut generator = self.quantum_generator.write().await;
            generator.generate_signature(data)
        } else {
            Vec::new()
        };

        // Calculate overall authenticity score
        let source_authenticity = self.calculate_source_authenticity(&behavioral_patterns, entropy_score);
        
        // Determine authenticity verdict
        let authenticity_verdict = self.determine_authenticity_verdict(
            source_authenticity,
            transformation_integrity,
            temporal_consistency,
            &anomaly_indicators,
        );

        Ok(DataFlowAnalysis {
            flow_id: operation.id.to_string(),
            source_authenticity,
            transformation_integrity,
            temporal_consistency,
            entropy_score,
            behavioral_patterns,
            anomaly_indicators,
            quantum_signature,
            authenticity_verdict,
        })
    }

    /// Analyze temporal consistency
    async fn analyze_temporal_consistency(&self, data: &[u8]) -> Result<f64, TENGRIError> {
        if data.len() < 8 {
            return Ok(0.5); // Insufficient data for temporal analysis
        }

        // Calculate block entropy variance
        let block_entropy = self.entropy_analyzer.calculate_block_entropy(data, 8);
        
        // Calculate compression ratio
        let compression_ratio = self.entropy_analyzer.analyze_compression_ratio(data);
        
        // Combine scores
        let consistency_score = (block_entropy + compression_ratio) / 2.0;
        
        Ok(consistency_score.min(1.0))
    }

    /// Validate data flow integrity
    async fn validate_data_flow_integrity(&self, data: &[u8]) -> Result<f64, TENGRIError> {
        // Check for patterns indicating synthetic generation
        let synthetic_patterns = [
            b"mock", b"fake", b"test", b"dummy", b"sample", b"synthetic",
            b"generated", b"random", b"artificial", b"simulated"
        ];

        let mut synthetic_score = 0.0;
        for pattern in &synthetic_patterns {
            if data.windows(pattern.len()).any(|window| window == *pattern) {
                synthetic_score += 0.1;
            }
        }

        // Higher synthetic score means lower integrity
        let integrity_score = (1.0 - synthetic_score).max(0.0);
        
        Ok(integrity_score)
    }

    /// Calculate source authenticity score
    fn calculate_source_authenticity(&self, patterns: &[BehavioralPattern], entropy_score: f64) -> f64 {
        let pattern_score = if patterns.is_empty() {
            0.5
        } else {
            let authentic_count = patterns.iter().filter(|p| p.is_authentic).count() as f64;
            authentic_count / patterns.len() as f64
        };

        // Weighted combination
        (pattern_score * 0.6 + entropy_score * 0.4).min(1.0)
    }

    /// Determine authenticity verdict
    fn determine_authenticity_verdict(
        &self,
        source_authenticity: f64,
        transformation_integrity: f64,
        temporal_consistency: f64,
        anomaly_indicators: &[AnomalyIndicator],
    ) -> AuthenticityVerdict {
        let overall_score = (source_authenticity + transformation_integrity + temporal_consistency) / 3.0;
        
        let critical_anomalies = anomaly_indicators.iter()
            .filter(|a| matches!(a.severity, AnomalySeverity::Critical))
            .count();

        if critical_anomalies > 0 {
            AuthenticityVerdict::DefinitelyInauthentic
        } else if overall_score >= 0.9 {
            AuthenticityVerdict::FullyAuthentic
        } else if overall_score >= 0.7 {
            AuthenticityVerdict::MostlyAuthentic
        } else if overall_score >= 0.5 {
            AuthenticityVerdict::QuestionableAuthenticity
        } else if overall_score >= 0.3 {
            AuthenticityVerdict::LikelyInauthentic
        } else {
            AuthenticityVerdict::DefinitelyInauthentic
        }
    }

    /// Record analysis history
    async fn record_analysis_history(&self, analysis: &DataFlowAnalysis) {
        let mut history = self.analysis_history.write().await;
        history.push((Utc::now(), analysis.clone()));

        // Keep only last 1,000 entries
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }

    /// Update authenticity scores
    async fn update_authenticity_scores(&self, analysis: &DataFlowAnalysis, operation: &TradingOperation) {
        let mut scores = self.authenticity_scores.write().await;
        scores.insert(operation.agent_id.clone(), analysis.source_authenticity);
    }

    /// Convert to oversight result
    async fn convert_to_oversight_result(&self, analysis: &DataFlowAnalysis) -> Result<TENGRIOversightResult, TENGRIError> {
        match analysis.authenticity_verdict {
            AuthenticityVerdict::FullyAuthentic => Ok(TENGRIOversightResult::Approved),
            AuthenticityVerdict::MostlyAuthentic => Ok(TENGRIOversightResult::Warning {
                reason: "Minor authenticity concerns detected".to_string(),
                corrective_action: "Review data sources for authenticity".to_string(),
            }),
            AuthenticityVerdict::QuestionableAuthenticity => Ok(TENGRIOversightResult::Rejected {
                reason: "Questionable test authenticity detected".to_string(),
                emergency_action: crate::EmergencyAction::AlertOperators,
            }),
            AuthenticityVerdict::LikelyInauthentic => Ok(TENGRIOversightResult::Rejected {
                reason: "Likely inauthentic test data detected".to_string(),
                emergency_action: crate::EmergencyAction::QuarantineAgent {
                    agent_id: "authenticity_violation".to_string(),
                },
            }),
            AuthenticityVerdict::DefinitelyInauthentic => Ok(TENGRIOversightResult::CriticalViolation {
                violation_type: ViolationType::SyntheticData,
                immediate_shutdown: true,
                forensic_data: serde_json::to_vec(analysis).unwrap_or_default(),
            }),
        }
    }

    /// Get authenticity statistics
    pub async fn get_authenticity_stats(&self) -> Result<TestAuthenticityStats, TENGRIError> {
        let history = self.analysis_history.read().await;
        let scores = self.authenticity_scores.read().await;

        let total_analyses = history.len();
        let authentic_analyses = history.iter()
            .filter(|(_, a)| matches!(a.authenticity_verdict, AuthenticityVerdict::FullyAuthentic))
            .count();

        let average_authenticity = if !scores.is_empty() {
            scores.values().sum::<f64>() / scores.len() as f64
        } else {
            0.0
        };

        Ok(TestAuthenticityStats {
            total_analyses,
            authentic_analyses,
            authenticity_rate: if total_analyses > 0 { authentic_analyses as f64 / total_analyses as f64 } else { 0.0 },
            average_authenticity_score: average_authenticity,
            recent_analyses: history.iter().rev().take(50).cloned().collect(),
        })
    }
}

/// Authenticity Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAuthenticityStats {
    pub total_analyses: usize,
    pub authentic_analyses: usize,
    pub authenticity_rate: f64,
    pub average_authenticity_score: f64,
    pub recent_analyses: Vec<(DateTime<Utc>, DataFlowAnalysis)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperationType, RiskParameters};

    #[tokio::test]
    async fn test_entropy_analysis() {
        let config = TestAuthenticityConfig::default();
        let analyzer = EntropyAnalyzer::new(config);
        
        // Test with random data
        let random_data = b"a7b8c9d0e1f2a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8";
        let entropy = analyzer.calculate_shannon_entropy(random_data);
        assert!(entropy > 0.7);
        
        // Test with repetitive data
        let repetitive_data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let entropy = analyzer.calculate_shannon_entropy(repetitive_data);
        assert!(entropy < 0.1);
    }

    #[tokio::test]
    async fn test_behavioral_pattern_analysis() {
        let config = TestAuthenticityConfig::default();
        let analyzer = BehavioralPatternAnalyzer::new(config);
        
        let data = b"authentic_market_data_with_natural_variation_patterns";
        let patterns = analyzer.analyze_behavioral_patterns(data, "test_context");
        
        assert!(!patterns.is_empty());
        assert_eq!(patterns.len(), 3); // frequency, temporal, distribution
    }

    #[tokio::test]
    async fn test_quantum_signature_generation() {
        let mut generator = QuantumSignatureGenerator::new();
        let data = b"test_data_for_quantum_signature";
        let signature = generator.generate_signature(data);
        
        assert!(!signature.is_empty());
        assert!(signature.len() >= 64); // Combined hash length
    }

    #[tokio::test]
    async fn test_authenticity_agent() {
        let config = TestAuthenticityConfig::default();
        let agent = TestAuthenticityAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "authentic_market_data_feed_with_real_time_updates".to_string(),
            mathematical_model: "real_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.verify_authenticity(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Approved | TENGRIOversightResult::Warning { .. }));
    }

    #[tokio::test]
    async fn test_synthetic_data_detection() {
        let config = TestAuthenticityConfig::default();
        let agent = TestAuthenticityAgent::new(config).await.unwrap();
        
        let operation = TradingOperation {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            operation_type: OperationType::PlaceOrder,
            data_source: "mock_synthetic_generated_fake_dummy_test_data".to_string(),
            mathematical_model: "test_model".to_string(),
            risk_parameters: RiskParameters {
                max_position_size: 1000.0,
                stop_loss: Some(0.02),
                take_profit: Some(0.05),
                confidence_threshold: 0.95,
            },
            agent_id: "test_agent".to_string(),
        };

        let result = agent.verify_authenticity(&operation).await.unwrap();
        assert!(matches!(result, TENGRIOversightResult::Rejected { .. } | TENGRIOversightResult::CriticalViolation { .. }));
    }
}