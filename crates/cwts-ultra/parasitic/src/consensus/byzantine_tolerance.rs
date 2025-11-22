//! Byzantine Fault Tolerance for Consensus Voting
//!
//! Implementation of Byzantine fault-tolerant consensus algorithms to handle
//! malicious or faulty nodes in the organism voting system.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime};
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use super::organism_selector::OrganismVote;
use super::{BYZANTINE_THRESHOLD, MIN_CONSENSUS_PARTICIPANTS};

/// Maximum number of Byzantine faults we can tolerate
/// For Byzantine fault tolerance: n >= 3f + 1 where f is max faults
pub const MAX_BYZANTINE_FAULTS: usize = 16; // Allows up to 49 total nodes

/// Time window for detecting coordinated attacks (ms)
const ATTACK_DETECTION_WINDOW_MS: u64 = 5000;

/// Maximum allowed message frequency per node (votes per second)
const MAX_MESSAGE_FREQUENCY: f64 = 10.0;

/// Node states in Byzantine fault tolerance protocol
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeState {
    /// Node is functioning correctly
    Honest,

    /// Node behavior is suspicious but not confirmed malicious
    Suspicious {
        suspicion_score: f64,
        first_suspicious_time: SystemTime,
        suspicious_behaviors: Vec<SuspiciousBehavior>,
    },

    /// Node is confirmed to be Byzantine/malicious
    Byzantine {
        fault_type: ByzantineFaultType,
        detection_time: SystemTime,
        evidence: Vec<ByzantineEvidence>,
    },

    /// Node is offline or unresponsive
    Offline { last_seen: SystemTime },

    /// Node is temporarily quarantined
    Quarantined {
        quarantine_reason: String,
        quarantine_until: SystemTime,
    },
}

/// Types of Byzantine faults
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ByzantineFaultType {
    /// Node sends contradictory messages
    Equivocation,

    /// Node ignores protocol rules
    ProtocolViolation,

    /// Node coordinates with other malicious nodes
    Collusion,

    /// Node attempts to manipulate voting outcomes
    VoteManipulation,

    /// Node floods the system with messages
    SpamAttack,

    /// Node provides false information
    Misinformation,

    /// Generic malicious behavior
    MaliciousBehavior,
}

/// Suspicious behaviors that may indicate Byzantine faults
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SuspiciousBehavior {
    /// Voting patterns deviate significantly from expected
    AnomalousVoting {
        deviation_score: f64,
        expected_range: (f64, f64),
        actual_value: f64,
    },

    /// Message timing is suspicious
    SuspiciousTiming { frequency: f64, burst_size: usize },

    /// Votes appear coordinated with other nodes
    CoordinatedBehavior {
        coordinated_with: Vec<Uuid>,
        coordination_score: f64,
    },

    /// Vote values are statistically unlikely
    UnlikelyValues { statistical_probability: f64 },

    /// Node responds too quickly (potentially automated)
    SuperhumanResponse { response_time_ms: u64 },
}

/// Evidence of Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ByzantineEvidence {
    pub evidence_type: EvidenceType,
    pub timestamp: SystemTime,
    pub description: String,
    pub confidence: f64,          // 0.0 to 1.0
    pub supporting_data: Vec<u8>, // Serialized proof data
}

/// Types of evidence for Byzantine detection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvidenceType {
    ConflictingVotes,
    MessageFrequencyViolation,
    StatisticalAnomaly,
    CoordinationPatterns,
    ProtocolViolation,
    CryptographicProof,
}

/// Vote verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteVerification {
    pub is_valid: bool,
    pub verification_score: f64,
    pub anomaly_flags: Vec<AnomalyFlag>,
    pub processing_time_ns: u64,
}

/// Anomaly flags for vote verification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyFlag {
    TimestampAnomaly,
    ScoreOutOfRange,
    WeightManipulation,
    FrequencyViolation,
    PatternMatching,
    CryptographicFailure,
}

/// Historical vote record for Byzantine detection
#[derive(Debug, Clone)]
struct VoteRecord {
    vote: OrganismVote,
    verification: VoteVerification,
    hash: String,
}

/// Node behavior tracking
#[derive(Debug, Clone)]
struct NodeBehavior {
    organism_id: Uuid,
    vote_history: VecDeque<VoteRecord>,
    message_frequency: f64,
    last_message_time: SystemTime,
    suspicious_score: f64,
    total_votes: u64,
    verified_votes: u64,
}

/// Byzantine fault tolerance implementation
pub struct ByzantineTolerance {
    byzantine_threshold: f64,
    node_states: HashMap<Uuid, NodeState>,
    node_behaviors: HashMap<Uuid, NodeBehavior>,
    attack_detector: AttackDetector,
    fault_statistics: FaultStatistics,
}

/// Attack pattern detection
struct AttackDetector {
    coordination_matrix: HashMap<(Uuid, Uuid), f64>,
    timing_patterns: HashMap<Uuid, VecDeque<SystemTime>>,
    vote_patterns: HashMap<Uuid, VecDeque<f64>>,
}

/// Fault tolerance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultStatistics {
    pub total_nodes: usize,
    pub honest_nodes: usize,
    pub suspicious_nodes: usize,
    pub byzantine_nodes: usize,
    pub offline_nodes: usize,
    pub quarantined_nodes: usize,
    pub false_positive_rate: f64,
    pub detection_accuracy: f64,
    pub max_tolerable_faults: usize,
}

/// Main fault tolerance interface
pub trait FaultTolerance {
    /// Check if system can tolerate current number of faults
    fn can_tolerate_faults(&self) -> bool;

    /// Get minimum number of honest nodes required
    fn minimum_honest_nodes(&self) -> usize;

    /// Get current fault tolerance status
    fn fault_tolerance_status(&self) -> FaultToleranceStatus;
}

/// Fault tolerance system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceStatus {
    pub is_fault_tolerant: bool,
    pub fault_capacity: usize,
    pub current_faults: usize,
    pub safety_margin: i32,
    pub recommended_actions: Vec<String>,
}

impl ByzantineTolerance {
    /// Create new Byzantine fault tolerance system
    pub fn new(byzantine_threshold: f64) -> Self {
        Self {
            byzantine_threshold: byzantine_threshold.clamp(0.51, 0.99),
            node_states: HashMap::new(),
            node_behaviors: HashMap::new(),
            attack_detector: AttackDetector {
                coordination_matrix: HashMap::new(),
                timing_patterns: HashMap::new(),
                vote_patterns: HashMap::new(),
            },
            fault_statistics: FaultStatistics {
                total_nodes: 0,
                honest_nodes: 0,
                suspicious_nodes: 0,
                byzantine_nodes: 0,
                offline_nodes: 0,
                quarantined_nodes: 0,
                false_positive_rate: 0.0,
                detection_accuracy: 1.0,
                max_tolerable_faults: MAX_BYZANTINE_FAULTS,
            },
        }
    }

    /// Check if a vote appears to be Byzantine
    #[instrument(skip(self, vote))]
    pub async fn is_byzantine_vote(&self, vote: &OrganismVote) -> bool {
        let start_time = std::time::Instant::now();

        // Quick checks first (performance optimization)
        if !self.basic_vote_validation(vote) {
            return true;
        }

        // Check node state
        if let Some(state) = self.node_states.get(&vote.organism_id) {
            match state {
                NodeState::Byzantine { .. } => return true,
                NodeState::Quarantined { .. } => return true,
                _ => {}
            }
        }

        // Detailed verification
        let verification = self.verify_vote_comprehensive(vote).await;
        let processing_time = start_time.elapsed().as_nanos() as u64;

        debug!(
            "Vote verification completed in {}ns, valid: {}",
            processing_time, verification.is_valid
        );

        !verification.is_valid
    }

    /// Basic vote validation (fast checks)
    fn basic_vote_validation(&self, vote: &OrganismVote) -> bool {
        // Check score range
        if vote.score < 0.0 || vote.score > 1.0 {
            return false;
        }

        // Check weight range
        if vote.weight < 0.0 || vote.weight > 10.0 {
            return false;
        }

        // Check confidence range
        if vote.confidence < 0.0 || vote.confidence > 1.0 {
            return false;
        }

        // Check timestamp is reasonable (not too far in future/past)
        let now = SystemTime::now();
        let time_diff = vote
            .timestamp
            .duration_since(now)
            .unwrap_or_else(|_| now.duration_since(vote.timestamp).unwrap_or_default());

        if time_diff.as_secs() > 60 {
            // Allow 1 minute clock skew
            return false;
        }

        true
    }

    /// Comprehensive vote verification
    async fn verify_vote_comprehensive(&self, vote: &OrganismVote) -> VoteVerification {
        let start_time = std::time::Instant::now();
        let mut anomaly_flags = Vec::new();
        let mut verification_score = 1.0;

        // Check message frequency
        if let Some(behavior) = self.node_behaviors.get(&vote.organism_id) {
            if behavior.message_frequency > MAX_MESSAGE_FREQUENCY {
                anomaly_flags.push(AnomalyFlag::FrequencyViolation);
                verification_score *= 0.5;
            }
        }

        // Statistical analysis
        let statistical_score = self.analyze_vote_statistics(vote).await;
        if statistical_score < 0.3 {
            anomaly_flags.push(AnomalyFlag::PatternMatching);
            verification_score *= statistical_score;
        }

        // Check for coordination patterns
        if self.detect_coordination_patterns(vote).await {
            anomaly_flags.push(AnomalyFlag::PatternMatching);
            verification_score *= 0.3;
        }

        // Weight manipulation detection
        if self.detect_weight_manipulation(vote).await {
            anomaly_flags.push(AnomalyFlag::WeightManipulation);
            verification_score *= 0.2;
        }

        let processing_time = start_time.elapsed().as_nanos() as u64;
        let is_valid = verification_score >= self.byzantine_threshold;

        VoteVerification {
            is_valid,
            verification_score,
            anomaly_flags,
            processing_time_ns: processing_time,
        }
    }

    /// Analyze vote statistics for anomalies
    async fn analyze_vote_statistics(&self, vote: &OrganismVote) -> f64 {
        // Get historical voting patterns for this organism
        let behavior = match self.node_behaviors.get(&vote.organism_id) {
            Some(b) => b,
            None => return 0.8, // Neutral score for new nodes
        };

        if behavior.vote_history.is_empty() {
            return 0.8;
        }

        // Calculate statistical deviation from historical pattern
        let historical_scores: Vec<f64> = behavior
            .vote_history
            .iter()
            .map(|record| record.vote.score)
            .collect();

        let mean_score = historical_scores.iter().sum::<f64>() / historical_scores.len() as f64;
        let variance = historical_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<f64>()
            / historical_scores.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.8; // Constant voting pattern
        }

        // Calculate z-score
        let z_score = (vote.score - mean_score) / std_dev;

        // Convert z-score to probability (higher z-score = more suspicious)
        let statistical_score = 1.0 / (1.0 + z_score.abs() / 3.0);

        statistical_score.clamp(0.0, 1.0)
    }

    /// Detect coordination patterns between nodes
    async fn detect_coordination_patterns(&self, vote: &OrganismVote) -> bool {
        // Check if this vote aligns suspiciously with recent votes from other nodes
        let time_window = Duration::from_millis(ATTACK_DETECTION_WINDOW_MS);
        let vote_time = vote.timestamp;

        for (other_id, behavior) in &self.node_behaviors {
            if *other_id == vote.organism_id {
                continue;
            }

            // Check for recent similar votes
            for record in behavior.vote_history.iter().rev().take(10) {
                if let Ok(time_diff) = record
                    .vote
                    .timestamp
                    .duration_since(vote_time)
                    .or_else(|_| vote_time.duration_since(record.vote.timestamp))
                {
                    if time_diff <= time_window {
                        // Check similarity of scores
                        let score_diff = (record.vote.score - vote.score).abs();
                        let weight_diff = (record.vote.weight - vote.weight).abs();

                        if score_diff < 0.1 && weight_diff < 0.1 {
                            // Update coordination matrix
                            let key = (vote.organism_id, *other_id);
                            let coordination = self
                                .attack_detector
                                .coordination_matrix
                                .get(&key)
                                .unwrap_or(&0.0)
                                + 1.0;

                            if coordination > 3.0 {
                                // Threshold for coordination
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }

    /// Detect weight manipulation attempts
    async fn detect_weight_manipulation(&self, vote: &OrganismVote) -> bool {
        // Check if weight is disproportionate to organism's historical performance
        let behavior = match self.node_behaviors.get(&vote.organism_id) {
            Some(b) => b,
            None => return false, // Can't detect for new nodes
        };

        if behavior.vote_history.len() < 5 {
            return false; // Need history to detect manipulation
        }

        // Calculate expected weight based on historical performance
        let historical_weights: Vec<f64> = behavior
            .vote_history
            .iter()
            .map(|record| record.vote.weight)
            .collect();

        let avg_weight = historical_weights.iter().sum::<f64>() / historical_weights.len() as f64;
        let weight_deviation = (vote.weight - avg_weight).abs() / avg_weight.max(0.1);

        // Suspicious if weight deviates by more than 200%
        weight_deviation > 2.0
    }

    /// Update node behavior tracking
    pub fn update_node_behavior(&mut self, vote: &OrganismVote, verification: VoteVerification) {
        // Fix E0502: Calculate hash before mutable borrow of self
        let vote_hash = self.calculate_vote_hash(vote);
        let organism_id = vote.organism_id;
        let verification_clone = verification.clone();

        let behavior = self
            .node_behaviors
            .entry(vote.organism_id)
            .or_insert_with(|| NodeBehavior {
                organism_id: vote.organism_id,
                vote_history: VecDeque::new(),
                message_frequency: 0.0,
                last_message_time: SystemTime::now(),
                suspicious_score: 0.0,
                total_votes: 0,
                verified_votes: 0,
            });

        // Calculate message frequency
        let now = SystemTime::now();
        if let Ok(time_diff) = now.duration_since(behavior.last_message_time) {
            let time_diff_secs = time_diff.as_secs_f64();
            if time_diff_secs > 0.0 {
                behavior.message_frequency =
                    0.9 * behavior.message_frequency + 0.1 / time_diff_secs;
            }
        }
        behavior.last_message_time = now;

        let record = VoteRecord {
            vote: vote.clone(),
            verification: verification.clone(),
            hash: vote_hash,
        };

        behavior.vote_history.push_back(record);
        if behavior.vote_history.len() > 100 {
            behavior.vote_history.pop_front();
        }

        // Update statistics
        behavior.total_votes += 1;
        if verification.is_valid {
            behavior.verified_votes += 1;
        }

        // Update suspicious score
        let verification_rate = behavior.verified_votes as f64 / behavior.total_votes as f64;
        behavior.suspicious_score = 1.0 - verification_rate;
        let suspicious_score = behavior.suspicious_score;

        // Fix E0502/E0499: End mutable borrow before calling method
        drop(behavior);

        // Update node state based on behavior
        self.update_node_state(organism_id, suspicious_score, &verification_clone);
    }

    /// Update node state based on suspicious behavior
    fn update_node_state(
        &mut self,
        organism_id: Uuid,
        suspicious_score: f64,
        verification: &VoteVerification,
    ) {
        let current_state = self
            .node_states
            .get(&organism_id)
            .cloned()
            .unwrap_or(NodeState::Honest);

        let new_state = match current_state {
            NodeState::Honest => {
                if suspicious_score > 0.3 {
                    NodeState::Suspicious {
                        suspicion_score: suspicious_score,
                        first_suspicious_time: SystemTime::now(),
                        suspicious_behaviors: self.extract_suspicious_behaviors(verification),
                    }
                } else {
                    NodeState::Honest
                }
            }

            NodeState::Suspicious {
                suspicion_score: prev_score,
                first_suspicious_time,
                mut suspicious_behaviors,
            } => {
                let updated_score = 0.7 * prev_score + 0.3 * suspicious_score;

                if updated_score > 0.7 {
                    // Escalate to Byzantine
                    let evidence = self.compile_byzantine_evidence(&suspicious_behaviors);
                    NodeState::Byzantine {
                        fault_type: self.classify_fault_type(&suspicious_behaviors),
                        detection_time: SystemTime::now(),
                        evidence,
                    }
                } else if updated_score < 0.2 {
                    // Return to honest
                    NodeState::Honest
                } else {
                    // Update suspicious behaviors
                    suspicious_behaviors.extend(self.extract_suspicious_behaviors(verification));
                    NodeState::Suspicious {
                        suspicion_score: updated_score,
                        first_suspicious_time,
                        suspicious_behaviors,
                    }
                }
            }

            // Byzantine and quarantined nodes stay in their state
            NodeState::Byzantine { .. } | NodeState::Quarantined { .. } => current_state,

            NodeState::Offline { .. } => {
                // Node came back online
                NodeState::Honest
            }
        };

        self.node_states.insert(organism_id, new_state);
        self.update_statistics();
    }

    /// Extract suspicious behaviors from verification
    fn extract_suspicious_behaviors(
        &self,
        verification: &VoteVerification,
    ) -> Vec<SuspiciousBehavior> {
        let mut behaviors = Vec::new();

        for flag in &verification.anomaly_flags {
            let behavior = match flag {
                AnomalyFlag::FrequencyViolation => {
                    SuspiciousBehavior::SuspiciousTiming {
                        frequency: MAX_MESSAGE_FREQUENCY + 1.0, // Placeholder
                        burst_size: 1,
                    }
                }
                AnomalyFlag::PatternMatching => {
                    SuspiciousBehavior::CoordinatedBehavior {
                        coordinated_with: Vec::new(), // Would be filled with actual data
                        coordination_score: 0.8,
                    }
                }
                AnomalyFlag::WeightManipulation => SuspiciousBehavior::UnlikelyValues {
                    statistical_probability: 1.0 - verification.verification_score,
                },
                _ => continue,
            };

            behaviors.push(behavior);
        }

        behaviors
    }

    /// Compile evidence for Byzantine behavior
    fn compile_byzantine_evidence(
        &self,
        suspicious_behaviors: &[SuspiciousBehavior],
    ) -> Vec<ByzantineEvidence> {
        suspicious_behaviors
            .iter()
            .map(|behavior| {
                let (evidence_type, description, confidence) = match behavior {
                    SuspiciousBehavior::CoordinatedBehavior {
                        coordination_score, ..
                    } => (
                        EvidenceType::CoordinationPatterns,
                        "Node exhibits coordinated voting patterns".to_string(),
                        *coordination_score,
                    ),
                    SuspiciousBehavior::SuspiciousTiming { frequency, .. } => (
                        EvidenceType::MessageFrequencyViolation,
                        format!("Message frequency {} exceeds threshold", frequency),
                        0.9,
                    ),
                    SuspiciousBehavior::UnlikelyValues {
                        statistical_probability,
                    } => (
                        EvidenceType::StatisticalAnomaly,
                        "Vote values are statistically unlikely".to_string(),
                        1.0 - statistical_probability,
                    ),
                    _ => (
                        EvidenceType::ProtocolViolation,
                        "Generic protocol violation detected".to_string(),
                        0.7,
                    ),
                };

                ByzantineEvidence {
                    evidence_type,
                    timestamp: SystemTime::now(),
                    description,
                    confidence,
                    supporting_data: Vec::new(), // Would contain serialized proof data
                }
            })
            .collect()
    }

    /// Classify the type of Byzantine fault
    fn classify_fault_type(
        &self,
        suspicious_behaviors: &[SuspiciousBehavior],
    ) -> ByzantineFaultType {
        // Simple classification based on predominant behavior
        let mut coordination_count = 0;
        let mut timing_count = 0;
        let mut value_count = 0;

        for behavior in suspicious_behaviors {
            match behavior {
                SuspiciousBehavior::CoordinatedBehavior { .. } => coordination_count += 1,
                SuspiciousBehavior::SuspiciousTiming { .. } => timing_count += 1,
                SuspiciousBehavior::UnlikelyValues { .. } => value_count += 1,
                _ => {}
            }
        }

        if coordination_count > timing_count && coordination_count > value_count {
            ByzantineFaultType::Collusion
        } else if timing_count > value_count {
            ByzantineFaultType::SpamAttack
        } else if value_count > 0 {
            ByzantineFaultType::VoteManipulation
        } else {
            ByzantineFaultType::MaliciousBehavior
        }
    }

    /// Calculate hash of vote for integrity checking
    fn calculate_vote_hash(&self, vote: &OrganismVote) -> String {
        let mut hasher = Sha256::new();
        hasher.update(vote.organism_id.as_bytes());
        hasher.update(vote.score.to_be_bytes());
        hasher.update(vote.weight.to_be_bytes());
        hasher.update(vote.confidence.to_be_bytes());

        // Add timestamp (rounded to seconds to avoid minor variations)
        let timestamp_secs = vote
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        hasher.update(timestamp_secs.to_be_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Update fault tolerance statistics
    fn update_statistics(&mut self) {
        let mut stats = FaultStatistics {
            total_nodes: self.node_states.len(),
            honest_nodes: 0,
            suspicious_nodes: 0,
            byzantine_nodes: 0,
            offline_nodes: 0,
            quarantined_nodes: 0,
            false_positive_rate: self.fault_statistics.false_positive_rate,
            detection_accuracy: self.fault_statistics.detection_accuracy,
            max_tolerable_faults: MAX_BYZANTINE_FAULTS,
        };

        for state in self.node_states.values() {
            match state {
                NodeState::Honest => stats.honest_nodes += 1,
                NodeState::Suspicious { .. } => stats.suspicious_nodes += 1,
                NodeState::Byzantine { .. } => stats.byzantine_nodes += 1,
                NodeState::Offline { .. } => stats.offline_nodes += 1,
                NodeState::Quarantined { .. } => stats.quarantined_nodes += 1,
            }
        }

        self.fault_statistics = stats;
    }

    /// Get fault tolerance statistics
    pub fn get_statistics(&self) -> FaultStatistics {
        self.fault_statistics.clone()
    }

    /// Get node state
    pub fn get_node_state(&self, organism_id: &Uuid) -> Option<&NodeState> {
        self.node_states.get(organism_id)
    }

    /// Manually quarantine a node
    pub fn quarantine_node(&mut self, organism_id: Uuid, reason: String, duration: Duration) {
        let quarantine_until = SystemTime::now() + duration;

        self.node_states.insert(
            organism_id,
            NodeState::Quarantined {
                quarantine_reason: reason,
                quarantine_until,
            },
        );

        warn!(
            "Quarantined node {} until {:?}",
            organism_id, quarantine_until
        );
        self.update_statistics();
    }

    /// Remove expired quarantines
    pub fn cleanup_quarantines(&mut self) {
        let now = SystemTime::now();
        let mut to_remove = Vec::new();

        for (organism_id, state) in &self.node_states {
            if let NodeState::Quarantined {
                quarantine_until, ..
            } = state
            {
                if now >= *quarantine_until {
                    to_remove.push(*organism_id);
                }
            }
        }

        for organism_id in to_remove {
            self.node_states.insert(organism_id, NodeState::Honest);
            debug!("Released node {} from quarantine", organism_id);
        }

        if !self.node_states.is_empty() {
            self.update_statistics();
        }
    }
}

impl FaultTolerance for ByzantineTolerance {
    fn can_tolerate_faults(&self) -> bool {
        let total_faults = self.fault_statistics.byzantine_nodes
            + self.fault_statistics.offline_nodes
            + self.fault_statistics.quarantined_nodes;

        // For Byzantine fault tolerance: n >= 3f + 1
        let min_nodes_required = 3 * total_faults + 1;
        self.fault_statistics.total_nodes >= min_nodes_required
    }

    fn minimum_honest_nodes(&self) -> usize {
        let total_faults = self.fault_statistics.byzantine_nodes
            + self.fault_statistics.offline_nodes
            + self.fault_statistics.quarantined_nodes;

        // Need at least 2f + 1 honest nodes
        2 * total_faults + 1
    }

    fn fault_tolerance_status(&self) -> FaultToleranceStatus {
        let current_faults = self.fault_statistics.byzantine_nodes
            + self.fault_statistics.offline_nodes
            + self.fault_statistics.quarantined_nodes;

        let is_fault_tolerant = self.can_tolerate_faults();
        let safety_margin =
            self.fault_statistics.total_nodes as i32 - self.minimum_honest_nodes() as i32;

        let mut recommended_actions = Vec::new();

        if !is_fault_tolerant {
            recommended_actions
                .push("System cannot tolerate current faults - add more honest nodes".to_string());
        }

        if safety_margin < 5 {
            recommended_actions.push("Low safety margin - consider adding more nodes".to_string());
        }

        if self.fault_statistics.byzantine_nodes > 0 {
            recommended_actions.push(format!(
                "Remove {} Byzantine nodes from system",
                self.fault_statistics.byzantine_nodes
            ));
        }

        FaultToleranceStatus {
            is_fault_tolerant,
            fault_capacity: MAX_BYZANTINE_FAULTS,
            current_faults,
            safety_margin,
            recommended_actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byzantine_tolerance_creation() {
        let bt = ByzantineTolerance::new(0.67);
        assert_eq!(bt.byzantine_threshold, 0.67);
        assert_eq!(bt.node_states.len(), 0);
    }

    #[test]
    fn test_basic_vote_validation() {
        let bt = ByzantineTolerance::new(0.67);

        let valid_vote = OrganismVote {
            session_id: Uuid::new_v4(),
            organism_id: Uuid::new_v4(),
            score: 0.8,
            weight: 1.0,
            confidence: 0.9,
            timestamp: SystemTime::now(),
            reasoning: None,
        };

        assert!(bt.basic_vote_validation(&valid_vote));

        let invalid_vote = OrganismVote {
            score: 1.5, // Invalid score > 1.0
            ..valid_vote.clone()
        };

        assert!(!bt.basic_vote_validation(&invalid_vote));
    }

    #[test]
    fn test_fault_tolerance_calculations() {
        let bt = ByzantineTolerance::new(0.67);

        // Test minimum honest nodes calculation
        assert_eq!(bt.minimum_honest_nodes(), 1); // 2*0 + 1 = 1 (no faults initially)

        // Test fault tolerance status
        let status = bt.fault_tolerance_status();
        assert!(status.is_fault_tolerant); // Should be fault tolerant with no faults
        assert_eq!(status.current_faults, 0);
    }

    #[test]
    fn test_node_state_transitions() {
        let mut bt = ByzantineTolerance::new(0.67);
        let organism_id = Uuid::new_v4();

        // Start as honest
        assert_eq!(bt.get_node_state(&organism_id), None);

        // Quarantine node
        bt.quarantine_node(
            organism_id,
            "Test quarantine".to_string(),
            Duration::from_secs(60),
        );

        match bt.get_node_state(&organism_id) {
            Some(NodeState::Quarantined {
                quarantine_reason, ..
            }) => {
                assert_eq!(quarantine_reason, "Test quarantine");
            }
            _ => panic!("Node should be quarantined"),
        }
    }

    #[test]
    fn test_vote_hash_calculation() {
        let bt = ByzantineTolerance::new(0.67);

        let vote = OrganismVote {
            session_id: Uuid::new_v4(),
            organism_id: Uuid::new_v4(),
            score: 0.8,
            weight: 1.0,
            confidence: 0.9,
            timestamp: SystemTime::now(),
            reasoning: None,
        };

        let hash1 = bt.calculate_vote_hash(&vote);
        let hash2 = bt.calculate_vote_hash(&vote);

        // Same vote should produce same hash
        assert_eq!(hash1, hash2);
        assert!(hash1.len() > 0);
    }

    #[test]
    fn test_suspicious_behavior_extraction() {
        let bt = ByzantineTolerance::new(0.67);

        let verification = VoteVerification {
            is_valid: false,
            verification_score: 0.3,
            anomaly_flags: vec![
                AnomalyFlag::FrequencyViolation,
                AnomalyFlag::WeightManipulation,
            ],
            processing_time_ns: 1000,
        };

        let behaviors = bt.extract_suspicious_behaviors(&verification);
        assert_eq!(behaviors.len(), 2);

        match &behaviors[0] {
            SuspiciousBehavior::SuspiciousTiming { .. } => {}
            _ => panic!("Expected SuspiciousTiming behavior"),
        }

        match &behaviors[1] {
            SuspiciousBehavior::UnlikelyValues { .. } => {}
            _ => panic!("Expected UnlikelyValues behavior"),
        }
    }

    #[tokio::test]
    async fn test_statistical_analysis() {
        let mut bt = ByzantineTolerance::new(0.67);
        let organism_id = Uuid::new_v4();

        // Create some historical votes
        for i in 0..10 {
            let vote = OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id,
                score: 0.5 + i as f64 * 0.05, // Gradually increasing scores
                weight: 1.0,
                confidence: 0.9,
                timestamp: SystemTime::now(),
                reasoning: None,
            };

            let verification = VoteVerification {
                is_valid: true,
                verification_score: 0.9,
                anomaly_flags: vec![],
                processing_time_ns: 1000,
            };

            bt.update_node_behavior(&vote, verification);
        }

        // Test with normal vote
        let normal_vote = OrganismVote {
            session_id: Uuid::new_v4(),
            organism_id,
            score: 0.7, // Within expected range
            weight: 1.0,
            confidence: 0.9,
            timestamp: SystemTime::now(),
            reasoning: None,
        };

        let normal_score = bt.analyze_vote_statistics(&normal_vote).await;
        assert!(normal_score > 0.5);

        // Test with anomalous vote
        let anomalous_vote = OrganismVote {
            session_id: Uuid::new_v4(),
            organism_id,
            score: 0.1, // Far from historical pattern
            weight: 1.0,
            confidence: 0.9,
            timestamp: SystemTime::now(),
            reasoning: None,
        };

        let anomalous_score = bt.analyze_vote_statistics(&anomalous_vote).await;
        assert!(anomalous_score < normal_score);
    }
}
