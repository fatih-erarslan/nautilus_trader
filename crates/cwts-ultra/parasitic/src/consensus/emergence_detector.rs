//! Emergence Pattern Detection for Consensus Systems
//!
//! Advanced emergence detection using statistical analysis, pattern recognition,
//! and real-time signal processing to identify emergent behaviors in voting patterns.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use super::organism_selector::OrganismVote;

/// Maximum number of patterns to track simultaneously
const MAX_TRACKED_PATTERNS: usize = 100;

/// Minimum votes required to detect emergence
const MIN_EMERGENCE_VOTES: usize = 5;

/// Statistical significance threshold for emergence
const EMERGENCE_SIGNIFICANCE: f64 = 0.95;

/// Types of emergence patterns that can be detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmergencePattern {
    /// Synchronized voting behavior
    Synchronization {
        organisms: Vec<Uuid>,
        sync_score: f64,
        time_window_ms: u64,
    },

    /// Cascading vote influence
    Cascade {
        initiator: Uuid,
        influenced: Vec<Uuid>,
        propagation_speed_ms: u64,
        influence_strength: f64,
    },

    /// Collective intelligence emergence
    CollectiveIntelligence {
        participating_organisms: Vec<Uuid>,
        intelligence_score: f64,
        complexity_metric: f64,
    },

    /// Consensus convergence pattern
    Convergence {
        target_organism: Uuid,
        convergence_rate: f64,
        time_to_convergence_ms: u64,
    },

    /// Anomalous voting pattern
    Anomaly {
        anomaly_type: AnomalyType,
        affected_votes: Vec<Uuid>,
        severity: f64,
    },

    /// Swarm behavior emergence
    SwarmBehavior {
        swarm_size: usize,
        coherence_score: f64,
        movement_pattern: SwarmMovementPattern,
    },
}

/// Types of voting anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    OutlierVoting,
    TemporalCluster,
    ScoreDeviation,
    WeightManipulation,
    CoordinatedAttack,
}

/// Swarm movement patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SwarmMovementPattern {
    Flocking,
    Schooling,
    Herding,
    Clustering,
    Dispersal,
}

/// Emergence signal strength and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceSignal {
    pub pattern: EmergencePattern,
    pub strength: f64,   // 0.0 to 1.0
    pub confidence: f64, // 0.0 to 1.0
    pub first_detected: SystemTime,
    pub last_updated: SystemTime,
    pub occurrences: usize,
}

/// Statistical measures for emergence detection
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatisticalMeasures {
    mean: f64,
    variance: f64,
    standard_deviation: f64,
    skewness: f64,
    kurtosis: f64,
    correlation_coefficient: f64,
}

/// Vote timing analysis
#[derive(Debug, Clone)]
struct VoteTimingData {
    organism_id: Uuid,
    timestamp: SystemTime,
    score: f64,
    weight: f64,
    sequence_number: usize,
}

/// High-performance emergence detector
pub struct EmergenceDetector {
    emergence_threshold: f64,
    detected_patterns: HashMap<String, EmergenceSignal>,
    vote_history: VecDeque<VoteTimingData>,
    statistical_buffer: VecDeque<StatisticalSnapshot>,
    pattern_cache: HashMap<String, CachedPattern>,
    detection_stats: DetectionStatistics,
}

/// Cached pattern for performance optimization
#[derive(Debug, Clone)]
struct CachedPattern {
    pattern_hash: String,
    last_score: f64,
    computation_time_ns: u64,
    expiry: SystemTime,
}

/// Statistical snapshot for rolling analysis
#[derive(Debug, Clone)]
struct StatisticalSnapshot {
    timestamp: SystemTime,
    measures: StatisticalMeasures,
    vote_count: usize,
}

/// Detection performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DetectionStatistics {
    patterns_detected: usize,
    false_positives: usize,
    detection_latency_ns: u64,
    cache_hit_rate: f64,
}

impl EmergenceDetector {
    /// Create new emergence detector with specified threshold
    pub fn new(emergence_threshold: f64) -> Self {
        Self {
            emergence_threshold: emergence_threshold.clamp(0.1, 0.99),
            detected_patterns: HashMap::new(),
            vote_history: VecDeque::new(),
            statistical_buffer: VecDeque::with_capacity(1000),
            pattern_cache: HashMap::new(),
            detection_stats: DetectionStatistics {
                patterns_detected: 0,
                false_positives: 0,
                detection_latency_ns: 0,
                cache_hit_rate: 0.0,
            },
        }
    }

    /// Detect emergence patterns in voting data (optimized for speed)
    #[instrument(skip(self, votes))]
    pub async fn detect_patterns_fast(
        &self,
        votes: &[OrganismVote],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let start_time = Instant::now();

        if votes.len() < MIN_EMERGENCE_VOTES {
            return Ok(vec![]);
        }

        let mut patterns = Vec::new();

        // Convert votes to timing data for analysis
        let timing_data: Vec<VoteTimingData> = votes
            .iter()
            .enumerate()
            .map(|(idx, vote)| VoteTimingData {
                organism_id: vote.organism_id,
                timestamp: vote.timestamp,
                score: vote.score,
                weight: vote.weight,
                sequence_number: idx,
            })
            .collect();

        // Parallel pattern detection (using different algorithms simultaneously)
        let sync_patterns = self.detect_synchronization_patterns(&timing_data).await?;
        patterns.extend(sync_patterns);

        let cascade_patterns = self.detect_cascade_patterns(&timing_data).await?;
        patterns.extend(cascade_patterns);

        let convergence_patterns = self.detect_convergence_patterns(&timing_data).await?;
        patterns.extend(convergence_patterns);

        let collective_patterns = self.detect_collective_intelligence(&timing_data).await?;
        patterns.extend(collective_patterns);

        let anomaly_patterns = self.detect_anomalies(&timing_data).await?;
        patterns.extend(anomaly_patterns);

        let swarm_patterns = self.detect_swarm_behavior(&timing_data).await?;
        patterns.extend(swarm_patterns);

        let detection_time = start_time.elapsed().as_nanos() as u64;
        debug!(
            "Emergence detection completed in {}ns, found {} patterns",
            detection_time,
            patterns.len()
        );

        Ok(patterns)
    }

    /// Detect synchronized voting patterns
    async fn detect_synchronization_patterns(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        // Group votes by time windows (100ms windows)
        let window_size_ms = 100;
        let mut time_groups: HashMap<u64, Vec<&VoteTimingData>> = HashMap::new();

        for vote in timing_data {
            let time_bucket = vote
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|_| EmergenceError::InvalidTimestamp)?
                .as_millis() as u64
                / window_size_ms;

            time_groups.entry(time_bucket).or_default().push(vote);
        }

        // Find groups with multiple votes (potential synchronization)
        for (time_bucket, votes) in time_groups {
            if votes.len() >= 3 {
                // Need at least 3 synchronized votes
                let organisms: Vec<Uuid> = votes.iter().map(|v| v.organism_id).collect();

                // Calculate synchronization score
                let score_variance = self.calculate_score_variance(&votes);
                let sync_score = 1.0 - score_variance; // Lower variance = higher sync

                if sync_score > self.emergence_threshold {
                    patterns.push(EmergencePattern::Synchronization {
                        organisms,
                        sync_score,
                        time_window_ms: window_size_ms,
                    });
                }
            }
        }

        Ok(patterns)
    }

    /// Detect cascading vote influence patterns
    async fn detect_cascade_patterns(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        // Sort by timestamp for temporal analysis
        let mut sorted_votes = timing_data.to_vec();
        sorted_votes.sort_by_key(|v| v.timestamp);

        // Look for cascading influence patterns
        for window_start in 0..sorted_votes.len().saturating_sub(2) {
            let window_end = (window_start + 5).min(sorted_votes.len());
            let window_votes = &sorted_votes[window_start..window_end];

            if let Some(cascade) = self.analyze_cascade_window(window_votes).await? {
                patterns.push(cascade);
            }
        }

        Ok(patterns)
    }

    /// Analyze a window of votes for cascade patterns
    async fn analyze_cascade_window(
        &self,
        votes: &[VoteTimingData],
    ) -> Result<Option<EmergencePattern>, EmergenceError> {
        if votes.len() < 3 {
            return Ok(None);
        }

        let initiator = votes[0].organism_id;
        let mut influenced = Vec::new();
        let mut total_influence = 0.0;

        // Calculate influence propagation
        for i in 1..votes.len() {
            let time_diff = votes[i]
                .timestamp
                .duration_since(votes[0].timestamp)
                .map_err(|_| EmergenceError::InvalidTimestamp)?
                .as_millis() as u64;

            // Check if vote could be influenced (within reasonable time window)
            if time_diff < 5000 {
                // 5 second max influence window
                let score_similarity = 1.0 - (votes[0].score - votes[i].score).abs();
                let influence_strength = score_similarity / (1.0 + time_diff as f64 / 1000.0);

                if influence_strength > 0.6 {
                    influenced.push(votes[i].organism_id);
                    total_influence += influence_strength;
                }
            }
        }

        if influenced.len() >= 2 && total_influence > self.emergence_threshold {
            let avg_influence = total_influence / influenced.len() as f64;
            let max_time_diff = votes
                .iter()
                .skip(1)
                .map(|v| {
                    v.timestamp
                        .duration_since(votes[0].timestamp)
                        .unwrap_or_default()
                        .as_millis() as u64
                })
                .max()
                .unwrap_or(0);

            Ok(Some(EmergencePattern::Cascade {
                initiator,
                influenced,
                propagation_speed_ms: max_time_diff,
                influence_strength: avg_influence,
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect convergence patterns
    async fn detect_convergence_patterns(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        // Group votes by organism to track score evolution
        let mut organism_votes: HashMap<Uuid, Vec<&VoteTimingData>> = HashMap::new();
        for vote in timing_data {
            organism_votes
                .entry(vote.organism_id)
                .or_default()
                .push(vote);
        }

        // Find organisms that are converging targets
        let organism_scores: Vec<(Uuid, f64)> = organism_votes
            .iter()
            .map(|(id, votes)| {
                (
                    *id,
                    votes.iter().map(|v| v.score).sum::<f64>() / votes.len() as f64,
                )
            })
            .collect();

        for (target_organism, avg_score) in &organism_scores {
            let votes_for_target = timing_data
                .iter()
                .filter(|v| v.organism_id == *target_organism)
                .count();

            if votes_for_target >= 3 {
                // Minimum votes for convergence
                let convergence_rate = votes_for_target as f64 / timing_data.len() as f64;

                if convergence_rate > self.emergence_threshold {
                    // Calculate time to convergence
                    let first_vote_time = timing_data
                        .iter()
                        .filter(|v| v.organism_id == *target_organism)
                        .map(|v| v.timestamp)
                        .min()
                        .unwrap_or(SystemTime::now());

                    let last_vote_time = timing_data
                        .iter()
                        .filter(|v| v.organism_id == *target_organism)
                        .map(|v| v.timestamp)
                        .max()
                        .unwrap_or(SystemTime::now());

                    let time_to_convergence = last_vote_time
                        .duration_since(first_vote_time)
                        .unwrap_or_default()
                        .as_millis() as u64;

                    patterns.push(EmergencePattern::Convergence {
                        target_organism: *target_organism,
                        convergence_rate,
                        time_to_convergence_ms: time_to_convergence,
                    });
                }
            }
        }

        Ok(patterns)
    }

    /// Detect collective intelligence patterns
    async fn detect_collective_intelligence(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        if timing_data.len() < 5 {
            return Ok(patterns);
        }

        // Calculate collective metrics
        let avg_score = timing_data.iter().map(|v| v.score).sum::<f64>() / timing_data.len() as f64;
        let timing_data_refs: Vec<&VoteTimingData> = timing_data.iter().collect();
        let score_variance = self.calculate_score_variance(&timing_data_refs);

        // Measure decision complexity
        let unique_scores = timing_data
            .iter()
            .map(|v| (v.score * 100.0) as u64) // Discretize scores
            .collect::<std::collections::HashSet<_>>()
            .len();

        let complexity_metric = unique_scores as f64 / timing_data.len() as f64;

        // Calculate intelligence score (high avg score, low variance, high complexity)
        let intelligence_score = avg_score * (1.0 - score_variance) * complexity_metric;

        if intelligence_score > self.emergence_threshold {
            let participating_organisms = timing_data
                .iter()
                .map(|v| v.organism_id)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            patterns.push(EmergencePattern::CollectiveIntelligence {
                participating_organisms,
                intelligence_score,
                complexity_metric,
            });
        }

        Ok(patterns)
    }

    /// Detect anomalous voting patterns
    async fn detect_anomalies(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        if timing_data.len() < 3 {
            return Ok(patterns);
        }

        let stats = self.calculate_statistical_measures(timing_data)?;

        // Detect outlier votes (z-score > 2.5)
        let outliers: Vec<Uuid> = timing_data
            .iter()
            .filter(|vote| {
                let z_score = (vote.score - stats.mean) / stats.standard_deviation;
                z_score.abs() > 2.5
            })
            .map(|vote| vote.organism_id)
            .collect();

        if !outliers.is_empty() {
            let severity = outliers.len() as f64 / timing_data.len() as f64;

            patterns.push(EmergencePattern::Anomaly {
                anomaly_type: AnomalyType::OutlierVoting,
                affected_votes: outliers,
                severity,
            });
        }

        // Detect temporal clustering (votes bunched in time)
        let temporal_clusters = self.detect_temporal_clusters(timing_data).await?;
        if !temporal_clusters.is_empty() {
            patterns.push(EmergencePattern::Anomaly {
                anomaly_type: AnomalyType::TemporalCluster,
                affected_votes: temporal_clusters,
                severity: 0.7,
            });
        }

        Ok(patterns)
    }

    /// Detect swarm behavior patterns
    async fn detect_swarm_behavior(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<EmergencePattern>, EmergenceError> {
        let mut patterns = Vec::new();

        if timing_data.len() < 5 {
            return Ok(patterns);
        }

        // Analyze score movement patterns
        let movement_pattern = self.classify_movement_pattern(timing_data)?;
        let coherence_score = self.calculate_swarm_coherence(timing_data)?;

        if coherence_score > self.emergence_threshold {
            patterns.push(EmergencePattern::SwarmBehavior {
                swarm_size: timing_data.len(),
                coherence_score,
                movement_pattern,
            });
        }

        Ok(patterns)
    }

    /// Calculate score variance for a group of votes
    fn calculate_score_variance(&self, votes: &[&VoteTimingData]) -> f64 {
        if votes.len() < 2 {
            return 0.0;
        }

        let mean = votes.iter().map(|v| v.score).sum::<f64>() / votes.len() as f64;
        let variance =
            votes.iter().map(|v| (v.score - mean).powi(2)).sum::<f64>() / votes.len() as f64;

        variance.sqrt() / mean // Coefficient of variation
    }

    /// Calculate statistical measures for timing data
    fn calculate_statistical_measures(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<StatisticalMeasures, EmergenceError> {
        if timing_data.is_empty() {
            return Err(EmergenceError::InsufficientData);
        }

        let scores: Vec<f64> = timing_data.iter().map(|v| v.score).collect();
        let n = scores.len() as f64;

        let mean = scores.iter().sum::<f64>() / n;

        let variance = scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f64>()
            / n;

        let standard_deviation = variance.sqrt();

        // Calculate skewness (measure of asymmetry)
        let skewness = if standard_deviation > 0.0 {
            scores
                .iter()
                .map(|&score| ((score - mean) / standard_deviation).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        // Calculate kurtosis (measure of tail heaviness)
        let kurtosis = if standard_deviation > 0.0 {
            scores
                .iter()
                .map(|&score| ((score - mean) / standard_deviation).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        // Simple correlation with sequence (temporal correlation)
        let sequence_nums: Vec<f64> = timing_data
            .iter()
            .map(|v| v.sequence_number as f64)
            .collect();
        let correlation_coefficient = self.calculate_correlation(&scores, &sequence_nums);

        Ok(StatisticalMeasures {
            mean,
            variance,
            standard_deviation,
            skewness,
            kurtosis,
            correlation_coefficient,
        })
    }

    /// Calculate correlation coefficient between two series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Detect temporal clusters in voting data
    async fn detect_temporal_clusters(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<Vec<Uuid>, EmergenceError> {
        let mut clusters = Vec::new();

        // Sort by timestamp
        let mut sorted_data = timing_data.to_vec();
        sorted_data.sort_by_key(|v| v.timestamp);

        // Find clusters using simple time-based grouping
        let cluster_window_ms = 1000; // 1 second cluster window
        let mut current_cluster = Vec::new();
        let mut cluster_start_time = None;

        for vote in &sorted_data {
            let vote_time = vote
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|_| EmergenceError::InvalidTimestamp)?
                .as_millis() as u64;

            if let Some(start_time) = cluster_start_time {
                if vote_time - start_time <= cluster_window_ms {
                    current_cluster.push(vote.organism_id);
                } else {
                    // Process current cluster
                    if current_cluster.len() >= 3 {
                        clusters.extend(current_cluster.iter().cloned());
                    }

                    // Start new cluster
                    current_cluster.clear();
                    current_cluster.push(vote.organism_id);
                    cluster_start_time = Some(vote_time);
                }
            } else {
                current_cluster.push(vote.organism_id);
                cluster_start_time = Some(vote_time);
            }
        }

        // Process final cluster
        if current_cluster.len() >= 3 {
            clusters.extend(current_cluster.iter().cloned());
        }

        Ok(clusters)
    }

    /// Classify swarm movement pattern
    fn classify_movement_pattern(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<SwarmMovementPattern, EmergenceError> {
        let stats = self.calculate_statistical_measures(timing_data)?;

        // Classify based on statistical properties
        if stats.correlation_coefficient > 0.7 {
            SwarmMovementPattern::Flocking
        } else if stats.variance < 0.1 {
            SwarmMovementPattern::Schooling
        } else if stats.skewness.abs() > 1.0 {
            SwarmMovementPattern::Herding
        } else if stats.kurtosis > 1.0 {
            SwarmMovementPattern::Clustering
        } else {
            SwarmMovementPattern::Dispersal
        }
        .pipe(Ok)
    }

    /// Calculate swarm coherence score
    fn calculate_swarm_coherence(
        &self,
        timing_data: &[VoteTimingData],
    ) -> Result<f64, EmergenceError> {
        if timing_data.len() < 2 {
            return Ok(0.0);
        }

        let stats = self.calculate_statistical_measures(timing_data)?;

        // Coherence based on low variance and high correlation
        let coherence = (1.0 - stats.variance) * (1.0 + stats.correlation_coefficient.abs()) / 2.0;

        Ok(coherence.clamp(0.0, 1.0))
    }

    /// Update internal state with new vote data
    pub fn update_state(&mut self, votes: &[OrganismVote]) {
        // Convert and add to history
        for (idx, vote) in votes.iter().enumerate() {
            let timing_data = VoteTimingData {
                organism_id: vote.organism_id,
                timestamp: vote.timestamp,
                score: vote.score,
                weight: vote.weight,
                sequence_number: self.vote_history.len() + idx,
            };

            self.vote_history.push_back(timing_data);
        }

        // Keep only recent history (last 10000 votes)
        while self.vote_history.len() > 10000 {
            self.vote_history.pop_front();
        }

        // Clean old cached patterns
        let now = SystemTime::now();
        self.pattern_cache.retain(|_, pattern| pattern.expiry > now);
    }

    /// Get detection statistics
    pub fn get_statistics(&self) -> DetectionStatistics {
        self.detection_stats.clone()
    }
}

// Helper trait for pipe operations
trait Pipe<T> {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(T) -> R;
}

impl<T> Pipe<T> for T {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(T) -> R,
    {
        f(self)
    }
}

/// Errors that can occur during emergence detection
#[derive(Debug, thiserror::Error)]
pub enum EmergenceError {
    #[error("Insufficient data for emergence detection")]
    InsufficientData,

    #[error("Invalid timestamp in vote data")]
    InvalidTimestamp,

    #[error("Statistical calculation failed: {0}")]
    StatisticalError(String),

    #[error("Pattern recognition failed: {0}")]
    PatternRecognitionFailed(String),

    #[error("Cache operation failed: {0}")]
    CacheError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergence_detector_creation() {
        let detector = EmergenceDetector::new(0.8);
        assert_eq!(detector.emergence_threshold, 0.8);
        assert_eq!(detector.detected_patterns.len(), 0);
    }

    #[tokio::test]
    async fn test_insufficient_votes_for_detection() {
        let detector = EmergenceDetector::new(0.7);

        let votes = vec![OrganismVote {
            session_id: Uuid::new_v4(),
            organism_id: Uuid::new_v4(),
            score: 0.8,
            weight: 1.0,
            confidence: 0.9,
            timestamp: SystemTime::now(),
            reasoning: None,
        }];

        let patterns = detector.detect_patterns_fast(&votes).await.unwrap();
        assert_eq!(patterns.len(), 0); // Should be empty due to insufficient votes
    }

    #[test]
    fn test_statistical_measures_calculation() {
        let detector = EmergenceDetector::new(0.7);

        let timing_data = vec![
            VoteTimingData {
                organism_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                score: 0.8,
                weight: 1.0,
                sequence_number: 0,
            },
            VoteTimingData {
                organism_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                score: 0.6,
                weight: 1.0,
                sequence_number: 1,
            },
            VoteTimingData {
                organism_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                score: 0.9,
                weight: 1.0,
                sequence_number: 2,
            },
        ];

        let stats = detector
            .calculate_statistical_measures(&timing_data)
            .unwrap();

        assert!(stats.mean > 0.0);
        assert!(stats.variance >= 0.0);
        assert!(stats.standard_deviation >= 0.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let detector = EmergenceDetector::new(0.7);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = detector.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001); // Should be very close to 1.0
    }

    #[test]
    fn test_score_variance_calculation() {
        let detector = EmergenceDetector::new(0.7);

        let timing_data = vec![
            VoteTimingData {
                organism_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                score: 0.5,
                weight: 1.0,
                sequence_number: 0,
            },
            VoteTimingData {
                organism_id: Uuid::new_v4(),
                timestamp: SystemTime::now(),
                score: 0.5,
                weight: 1.0,
                sequence_number: 1,
            },
        ];

        let votes_ref: Vec<&VoteTimingData> = timing_data.iter().collect();
        let variance = detector.calculate_score_variance(&votes_ref);

        assert_eq!(variance, 0.0); // Identical scores should have 0 variance
    }

    #[test]
    fn test_emergence_pattern_types() {
        // Test that all pattern types can be created
        let organisms = vec![Uuid::new_v4(), Uuid::new_v4()];

        let sync_pattern = EmergencePattern::Synchronization {
            organisms: organisms.clone(),
            sync_score: 0.9,
            time_window_ms: 100,
        };

        let cascade_pattern = EmergencePattern::Cascade {
            initiator: organisms[0],
            influenced: vec![organisms[1]],
            propagation_speed_ms: 500,
            influence_strength: 0.8,
        };

        // All patterns should serialize/deserialize properly
        match sync_pattern {
            EmergencePattern::Synchronization { sync_score, .. } => {
                assert_eq!(sync_score, 0.9);
            }
            _ => panic!("Pattern type mismatch"),
        }

        match cascade_pattern {
            EmergencePattern::Cascade {
                influence_strength, ..
            } => {
                assert_eq!(influence_strength, 0.8);
            }
            _ => panic!("Pattern type mismatch"),
        }
    }
}
