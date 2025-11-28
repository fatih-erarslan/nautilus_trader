//! Byzantine Fault-Tolerant Consensus for CWTS Subsystem Risk Voting
//!
//! Implements a PBFT-inspired consensus mechanism adapted for CWTS where:
//! - Each subsystem acts as a "node" that votes on risk assessments
//! - Byzantine tolerance ensures corrupted/malfunctioning subsystems don't dominate
//! - Three-phase protocol: Pre-prepare (proposal), Prepare (voting), Commit (finalization)
//!
//! ## Scientific References
//!
//! - Castro & Liskov (1999): "Practical Byzantine Fault Tolerance"
//! - Lamport et al. (1982): "The Byzantine Generals Problem"
//! - Kahneman & Tversky (1979): "Prospect Theory" for decision weighting
//! - Arrow (1951): "Social Choice and Individual Values" for aggregation

use crate::core::{RiskLevel, Timestamp};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::coordinator::{SubsystemId, SubsystemRisk};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for BFT consensus in CWTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BftConsensusConfig {
    /// Maximum Byzantine subsystems tolerated (f in PBFT).
    /// With 5 subsystems, f=1 means we tolerate 1 Byzantine subsystem.
    pub max_byzantine: usize,

    /// Timeout for prepare phase in nanoseconds.
    pub prepare_timeout_ns: u64,

    /// Timeout for commit phase in nanoseconds.
    pub commit_timeout_ns: u64,

    /// Enable Byzantine detection heuristics.
    pub enable_byzantine_detection: bool,

    /// Threshold for declaring a subsystem Byzantine based on historical divergence.
    pub byzantine_divergence_threshold: f64,

    /// Rolling window for tracking subsystem behavior.
    pub behavior_window_size: usize,

    /// Minimum confidence for a vote to be counted.
    pub minimum_vote_confidence: f64,
}

impl Default for BftConsensusConfig {
    fn default() -> Self {
        Self {
            max_byzantine: 1, // Tolerate 1 out of 5 subsystems
            prepare_timeout_ns: 100_000,    // 100μs
            commit_timeout_ns: 50_000,      // 50μs
            enable_byzantine_detection: true,
            byzantine_divergence_threshold: 0.7, // 70% divergence from majority
            behavior_window_size: 100,
            minimum_vote_confidence: 0.1, // Very low minimum to allow all votes
        }
    }
}

// ============================================================================
// BFT Message Types
// ============================================================================

/// PBFT-style message for subsystem risk voting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BftRiskMessage {
    /// Pre-prepare: Initial risk proposal from coordinator.
    PrePrepare {
        /// View number (consensus round).
        view: u64,
        /// Sequence number for this proposal.
        sequence: u64,
        /// Digest of the proposal.
        digest: String,
        /// The actual risk proposal.
        proposal: RiskProposal,
    },

    /// Prepare: Subsystem acknowledges the proposal and provides assessment.
    Prepare {
        /// View number.
        view: u64,
        /// Sequence number.
        sequence: u64,
        /// Digest must match pre-prepare.
        digest: String,
        /// Subsystem casting this vote.
        subsystem: SubsystemId,
        /// The subsystem's risk assessment.
        assessment: SubsystemRisk,
        /// Signature (hash of assessment).
        signature: String,
    },

    /// Commit: Subsystem commits to the consensus decision.
    Commit {
        /// View number.
        view: u64,
        /// Sequence number.
        sequence: u64,
        /// Digest of agreed-upon decision.
        digest: String,
        /// Subsystem committing.
        subsystem: SubsystemId,
        /// Final agreed risk level.
        agreed_risk_level: RiskLevel,
        /// Signature.
        signature: String,
    },

    /// ViewChange: Request to change view (used when primary fails).
    ViewChange {
        /// New view number requested.
        new_view: u64,
        /// Subsystem requesting change.
        subsystem: SubsystemId,
        /// Reason for view change.
        reason: ViewChangeReason,
    },
}

/// Reason for requesting a view change.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ViewChangeReason {
    /// Primary subsystem timed out.
    PrimaryTimeout,
    /// Detected Byzantine behavior in primary.
    ByzantinePrimary,
    /// Inconsistent pre-prepare messages.
    InconsistentProposal,
    /// Network partition suspected.
    NetworkPartition,
}

/// A risk proposal submitted for BFT consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProposal {
    /// Unique proposal ID.
    pub id: u64,
    /// Context: symbol being evaluated (if any).
    pub symbol: Option<String>,
    /// Timestamp of proposal.
    pub timestamp: Timestamp,
    /// Initial risk assessment from coordinator.
    pub initial_assessment: RiskLevel,
    /// Metadata for subsystems to evaluate.
    pub context: ProposalContext,
}

/// Context provided to subsystems for evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalContext {
    /// Current portfolio size.
    pub portfolio_size: f64,
    /// Volatility estimate.
    pub volatility: f64,
    /// Recent drawdown.
    pub drawdown: f64,
    /// Market regime hint.
    pub regime_hint: Option<String>,
}

// ============================================================================
// Consensus Phase Tracking
// ============================================================================

/// Current phase of BFT consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusPhase {
    /// Initial state, waiting for proposal.
    Idle,
    /// Pre-prepare received, collecting prepare messages.
    PrePrepared,
    /// Enough prepares received, collecting commits.
    Prepared,
    /// Enough commits received, consensus finalized.
    Committed,
    /// Consensus failed (timeout or Byzantine detection).
    Failed,
}

/// State of an ongoing BFT consensus round.
#[derive(Debug, Clone)]
pub struct ConsensusRound {
    /// Current phase.
    pub phase: ConsensusPhase,
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// The proposal being voted on.
    pub proposal: Option<RiskProposal>,
    /// Prepare messages received.
    pub prepares: HashMap<SubsystemId, BftRiskMessage>,
    /// Commit messages received.
    pub commits: HashMap<SubsystemId, BftRiskMessage>,
    /// Start time of this round.
    pub start_time: Timestamp,
    /// Detected Byzantine subsystems in this round.
    pub byzantine_detected: Vec<SubsystemId>,
}

impl Default for ConsensusRound {
    fn default() -> Self {
        Self {
            phase: ConsensusPhase::Idle,
            view: 0,
            sequence: 0,
            proposal: None,
            prepares: HashMap::new(),
            commits: HashMap::new(),
            start_time: Timestamp::now(),
            byzantine_detected: Vec::new(),
        }
    }
}

// ============================================================================
// BFT Consensus Result
// ============================================================================

/// Result of BFT consensus on risk assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BftConsensusResult {
    /// Whether consensus was achieved.
    pub consensus_achieved: bool,
    /// The agreed-upon risk level (if consensus achieved).
    pub agreed_risk_level: RiskLevel,
    /// Aggregated risk score.
    pub aggregated_score: f64,
    /// Number of subsystems that participated.
    pub participant_count: usize,
    /// Number of agreeing subsystems.
    pub agreement_count: usize,
    /// Subsystems detected as potentially Byzantine.
    pub byzantine_suspects: Vec<SubsystemId>,
    /// Consensus proof for audit trail.
    pub proof: ConsensusProof,
    /// Total latency of consensus process in nanoseconds.
    pub latency_ns: u64,
}

/// Cryptographic proof of consensus for audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProof {
    /// View and sequence numbers.
    pub view: u64,
    pub sequence: u64,
    /// Merkle root of all prepare signatures.
    pub prepare_merkle_root: String,
    /// Merkle root of all commit signatures.
    pub commit_merkle_root: String,
    /// Final decision digest.
    pub decision_digest: String,
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Individual votes for transparency.
    pub votes: Vec<SubsystemVote>,
}

/// A single subsystem's vote record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemVote {
    /// Subsystem that voted.
    pub subsystem: SubsystemId,
    /// Risk level voted for.
    pub risk_level: RiskLevel,
    /// Risk score.
    pub risk_score: f64,
    /// Confidence in assessment.
    pub confidence: f64,
    /// Signature hash.
    pub signature: String,
}

// ============================================================================
// BFT Consensus Engine
// ============================================================================

/// Byzantine Fault-Tolerant consensus engine for CWTS.
#[derive(Debug)]
pub struct BftConsensusEngine {
    /// Configuration.
    config: BftConsensusConfig,
    /// Current view number.
    view: u64,
    /// Current sequence number.
    sequence: u64,
    /// Current consensus round state.
    current_round: ConsensusRound,
    /// Historical behavior for Byzantine detection.
    subsystem_history: HashMap<SubsystemId, Vec<HistoricalVote>>,
    /// Known Byzantine subsystems (accumulated over time).
    known_byzantine: Vec<SubsystemId>,
}

/// Historical vote record for Byzantine detection.
#[derive(Debug, Clone)]
struct HistoricalVote {
    /// Risk level voted.
    risk_level: RiskLevel,
    /// Risk score.
    risk_score: f64,
    /// Whether this vote diverged from majority.
    diverged_from_majority: bool,
    /// Timestamp.
    timestamp: Timestamp,
}

impl BftConsensusEngine {
    /// Create new BFT consensus engine.
    pub fn new(config: BftConsensusConfig) -> Self {
        Self {
            config,
            view: 0,
            sequence: 0,
            current_round: ConsensusRound::default(),
            subsystem_history: HashMap::new(),
            known_byzantine: Vec::new(),
        }
    }

    /// Start a new consensus round with a risk proposal.
    pub fn start_consensus(&mut self, context: ProposalContext) -> RiskProposal {
        self.sequence += 1;

        let proposal = RiskProposal {
            id: self.sequence,
            symbol: None,
            timestamp: Timestamp::now(),
            initial_assessment: RiskLevel::Normal, // Will be determined by consensus
            context,
        };

        self.current_round = ConsensusRound {
            phase: ConsensusPhase::PrePrepared,
            view: self.view,
            sequence: self.sequence,
            proposal: Some(proposal.clone()),
            prepares: HashMap::new(),
            commits: HashMap::new(),
            start_time: Timestamp::now(),
            byzantine_detected: Vec::new(),
        };

        proposal
    }

    /// Submit a subsystem's risk assessment (prepare phase).
    pub fn submit_assessment(&mut self, assessment: SubsystemRisk) -> Result<(), &'static str> {
        if self.current_round.phase != ConsensusPhase::PrePrepared
           && self.current_round.phase != ConsensusPhase::Prepared {
            return Err("Not in prepare phase");
        }

        // Check if subsystem is known Byzantine
        if self.known_byzantine.contains(&assessment.subsystem) {
            return Err("Subsystem marked as Byzantine");
        }

        // Check minimum confidence
        if assessment.confidence < self.config.minimum_vote_confidence {
            return Err("Confidence too low");
        }

        // Create prepare message
        let digest = self.compute_proposal_digest();
        let signature = self.sign_assessment(&assessment);

        let prepare = BftRiskMessage::Prepare {
            view: self.view,
            sequence: self.sequence,
            digest,
            subsystem: assessment.subsystem,
            assessment: assessment.clone(),
            signature,
        };

        self.current_round.prepares.insert(assessment.subsystem, prepare);

        // Check if we have enough prepares (2f + 1 for PBFT)
        let required = self.compute_prepare_threshold();
        if self.current_round.prepares.len() >= required {
            self.current_round.phase = ConsensusPhase::Prepared;
        }

        Ok(())
    }

    /// Attempt to finalize consensus (commit phase).
    pub fn finalize_consensus(&mut self) -> Result<BftConsensusResult, &'static str> {
        let start_time = Timestamp::now();

        if self.current_round.phase != ConsensusPhase::Prepared {
            return Err("Not in prepared phase");
        }

        // Collect all assessments
        let mut assessments: Vec<&SubsystemRisk> = Vec::new();
        for prepare in self.current_round.prepares.values() {
            if let BftRiskMessage::Prepare { assessment, .. } = prepare {
                assessments.push(assessment);
            }
        }

        if assessments.is_empty() {
            return Err("No assessments to finalize");
        }

        // Detect Byzantine behavior
        let byzantine_suspects = if self.config.enable_byzantine_detection {
            self.detect_byzantine_behavior(&assessments)
        } else {
            Vec::new()
        };

        // Filter out Byzantine suspects for final calculation
        let valid_assessments: Vec<&SubsystemRisk> = assessments
            .iter()
            .filter(|a| !byzantine_suspects.contains(&a.subsystem))
            .copied()
            .collect();

        if valid_assessments.is_empty() {
            return Err("All subsystems detected as Byzantine");
        }

        // Compute weighted consensus
        let (agreed_level, aggregated_score, agreement_count) =
            self.compute_weighted_consensus(&valid_assessments);

        // Generate commit messages
        for assessment in &valid_assessments {
            let commit = BftRiskMessage::Commit {
                view: self.view,
                sequence: self.sequence,
                digest: self.compute_proposal_digest(),
                subsystem: assessment.subsystem,
                agreed_risk_level: agreed_level,
                signature: self.sign_commit(assessment.subsystem, agreed_level),
            };
            self.current_round.commits.insert(assessment.subsystem, commit);
        }

        // Check commit threshold (2f + 1)
        let required_commits = self.compute_commit_threshold();
        let consensus_achieved = self.current_round.commits.len() >= required_commits;

        if consensus_achieved {
            self.current_round.phase = ConsensusPhase::Committed;
        } else {
            self.current_round.phase = ConsensusPhase::Failed;
        }

        // Build consensus proof
        let proof = self.build_consensus_proof(&valid_assessments, agreed_level);

        // Update subsystem history for future Byzantine detection
        self.update_subsystem_history(&assessments, agreed_level);

        // Mark detected Byzantine subsystems
        self.current_round.byzantine_detected = byzantine_suspects.clone();

        let end_time = Timestamp::now();
        let latency_ns = end_time.as_nanos().saturating_sub(start_time.as_nanos());

        Ok(BftConsensusResult {
            consensus_achieved,
            agreed_risk_level: agreed_level,
            aggregated_score,
            participant_count: assessments.len(),
            agreement_count,
            byzantine_suspects,
            proof,
            latency_ns,
        })
    }

    /// Compute weighted consensus from valid assessments.
    fn compute_weighted_consensus(&self, assessments: &[&SubsystemRisk]) -> (RiskLevel, f64, usize) {
        if assessments.is_empty() {
            return (RiskLevel::Normal, 0.0, 0);
        }

        // Weighted risk score aggregation
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for assessment in assessments {
            // Weight by confidence
            let weight = assessment.confidence;
            weighted_score += assessment.risk_score * weight;
            total_weight += weight;
        }

        let aggregated_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        // Determine risk level from aggregated score
        let agreed_level = Self::score_to_risk_level(aggregated_score);

        // Count how many agree with the final decision
        let agreement_count = assessments
            .iter()
            .filter(|a| a.risk_level == agreed_level ||
                       (a.risk_score - aggregated_score).abs() < 0.2)
            .count();

        (agreed_level, aggregated_score, agreement_count)
    }

    /// Detect Byzantine behavior in assessments.
    fn detect_byzantine_behavior(&self, assessments: &[&SubsystemRisk]) -> Vec<SubsystemId> {
        let mut suspects = Vec::new();

        if assessments.len() < 3 {
            return suspects; // Need at least 3 for meaningful detection
        }

        // Compute median risk score as reference
        let mut scores: Vec<f64> = assessments.iter().map(|a| a.risk_score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = scores[scores.len() / 2];

        // Check for outliers
        for assessment in assessments {
            let divergence = (assessment.risk_score - median).abs();

            // Also check historical divergence
            let historical_divergence = self.get_historical_divergence_rate(assessment.subsystem);

            // Mark as suspect if:
            // 1. Current divergence is high AND
            // 2. Historical divergence rate is above threshold
            if divergence > 0.5 && historical_divergence > self.config.byzantine_divergence_threshold {
                suspects.push(assessment.subsystem);
            }

            // Also check for obvious Byzantine signals
            if self.has_contradictory_signals(assessment) {
                suspects.push(assessment.subsystem);
            }
        }

        suspects.sort();
        suspects.dedup();
        suspects
    }

    /// Get historical divergence rate for a subsystem.
    fn get_historical_divergence_rate(&self, subsystem: SubsystemId) -> f64 {
        if let Some(history) = self.subsystem_history.get(&subsystem) {
            if history.is_empty() {
                return 0.0;
            }
            let diverged_count = history.iter().filter(|h| h.diverged_from_majority).count();
            diverged_count as f64 / history.len() as f64
        } else {
            0.0 // New subsystem, no history
        }
    }

    /// Check for contradictory signals in an assessment.
    fn has_contradictory_signals(&self, assessment: &SubsystemRisk) -> bool {
        // Risk level and score should be consistent
        let expected_level = Self::score_to_risk_level(assessment.risk_score);

        // Allow one level of difference (Normal vs Elevated is ok)
        let level_diff = (assessment.risk_level as i32 - expected_level as i32).abs();

        // Contradictory if more than 2 levels apart
        level_diff > 2
    }

    /// Update subsystem history after consensus.
    fn update_subsystem_history(&mut self, assessments: &[&SubsystemRisk], final_level: RiskLevel) {
        let final_score = Self::risk_level_to_score(final_level);

        for assessment in assessments {
            let diverged = (assessment.risk_score - final_score).abs() > 0.3;

            let vote = HistoricalVote {
                risk_level: assessment.risk_level,
                risk_score: assessment.risk_score,
                diverged_from_majority: diverged,
                timestamp: Timestamp::now(),
            };

            let history = self.subsystem_history
                .entry(assessment.subsystem)
                .or_insert_with(Vec::new);

            history.push(vote);

            // Trim to window size
            if history.len() > self.config.behavior_window_size {
                history.remove(0);
            }
        }
    }

    /// Build consensus proof for audit trail.
    fn build_consensus_proof(&self, assessments: &[&SubsystemRisk], final_level: RiskLevel) -> ConsensusProof {
        let votes: Vec<SubsystemVote> = assessments
            .iter()
            .map(|a| SubsystemVote {
                subsystem: a.subsystem,
                risk_level: a.risk_level,
                risk_score: a.risk_score,
                confidence: a.confidence,
                signature: self.sign_assessment(a),
            })
            .collect();

        let prepare_signatures: Vec<String> = votes.iter().map(|v| v.signature.clone()).collect();
        let prepare_merkle_root = self.compute_merkle_root(&prepare_signatures);

        let commit_signatures: Vec<String> = self.current_round.commits.values()
            .filter_map(|c| {
                if let BftRiskMessage::Commit { signature, .. } = c {
                    Some(signature.clone())
                } else {
                    None
                }
            })
            .collect();
        let commit_merkle_root = self.compute_merkle_root(&commit_signatures);

        let decision_digest = self.compute_decision_digest(final_level);

        ConsensusProof {
            view: self.view,
            sequence: self.sequence,
            prepare_merkle_root,
            commit_merkle_root,
            decision_digest,
            timestamp: Timestamp::now(),
            votes,
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Compute prepare threshold (2f for PBFT).
    fn compute_prepare_threshold(&self) -> usize {
        // With n nodes, need 2f messages where f is max Byzantine
        // For 5 subsystems with f=1: need 2 prepares
        2 * self.config.max_byzantine
    }

    /// Compute commit threshold (2f + 1 for PBFT).
    fn compute_commit_threshold(&self) -> usize {
        // Need 2f + 1 commits for finality
        2 * self.config.max_byzantine + 1
    }

    /// Compute proposal digest.
    fn compute_proposal_digest(&self) -> String {
        if let Some(ref proposal) = self.current_round.proposal {
            let content = format!("{}-{}-{}", self.view, self.sequence, proposal.id);
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            format!("{:x}", hasher.finalize())
        } else {
            "empty".to_string()
        }
    }

    /// Sign an assessment (hash-based signature).
    fn sign_assessment(&self, assessment: &SubsystemRisk) -> String {
        let content = format!(
            "{:?}-{}-{}-{}",
            assessment.subsystem,
            assessment.risk_score,
            assessment.confidence,
            self.sequence
        );
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Sign a commit message.
    fn sign_commit(&self, subsystem: SubsystemId, level: RiskLevel) -> String {
        let content = format!("{:?}-{:?}-{}-{}", subsystem, level, self.view, self.sequence);
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute Merkle root of signatures.
    fn compute_merkle_root(&self, signatures: &[String]) -> String {
        if signatures.is_empty() {
            return "empty".to_string();
        }
        let mut hasher = Sha256::new();
        for sig in signatures {
            hasher.update(sig.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Compute decision digest.
    fn compute_decision_digest(&self, level: RiskLevel) -> String {
        let content = format!("{:?}-{}-{}", level, self.view, self.sequence);
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Convert risk score to risk level.
    fn score_to_risk_level(score: f64) -> RiskLevel {
        match score {
            s if s >= 0.9 => RiskLevel::Emergency,
            s if s >= 0.75 => RiskLevel::Critical,
            s if s >= 0.5 => RiskLevel::High,
            s if s >= 0.25 => RiskLevel::Elevated,
            _ => RiskLevel::Normal,
        }
    }

    /// Convert risk level to approximate score.
    fn risk_level_to_score(level: RiskLevel) -> f64 {
        match level {
            RiskLevel::Normal => 0.1,
            RiskLevel::Elevated => 0.35,
            RiskLevel::High => 0.6,
            RiskLevel::Critical => 0.85,
            RiskLevel::Emergency => 0.95,
        }
    }

    /// Request view change (when current primary is Byzantine).
    pub fn request_view_change(&mut self, reason: ViewChangeReason) {
        self.view += 1;
        self.current_round = ConsensusRound::default();
        self.current_round.view = self.view;

        tracing::warn!(
            "View change requested: {:?}, new view: {}",
            reason,
            self.view
        );
    }

    /// Get known Byzantine subsystems.
    pub fn get_known_byzantine(&self) -> &[SubsystemId] {
        &self.known_byzantine
    }

    /// Mark a subsystem as Byzantine (manual override).
    pub fn mark_byzantine(&mut self, subsystem: SubsystemId) {
        if !self.known_byzantine.contains(&subsystem) {
            self.known_byzantine.push(subsystem);
            tracing::warn!("Subsystem {:?} marked as Byzantine", subsystem);
        }
    }

    /// Clear Byzantine status (after recovery/fix).
    pub fn clear_byzantine(&mut self, subsystem: SubsystemId) {
        self.known_byzantine.retain(|s| *s != subsystem);
    }

    /// Get current consensus round state.
    pub fn current_round(&self) -> &ConsensusRound {
        &self.current_round
    }

    /// Get current view number.
    pub fn current_view(&self) -> u64 {
        self.view
    }

    /// Get current sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.sequence
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assessment(subsystem: SubsystemId, risk_score: f64, confidence: f64) -> SubsystemRisk {
        SubsystemRisk {
            subsystem,
            risk_level: BftConsensusEngine::score_to_risk_level(risk_score),
            confidence,
            risk_score,
            position_factor: 1.0 - risk_score,
            reasoning: format!("Test assessment for {:?}", subsystem),
            timestamp: Timestamp::now(),
            latency_ns: 1000,
        }
    }

    #[test]
    fn test_bft_config_default() {
        let config = BftConsensusConfig::default();
        assert_eq!(config.max_byzantine, 1);
        assert!(config.enable_byzantine_detection);
    }

    #[test]
    fn test_start_consensus() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };

        let proposal = engine.start_consensus(context);
        assert_eq!(proposal.id, 1);
        assert_eq!(engine.current_round().phase, ConsensusPhase::PrePrepared);
    }

    #[test]
    fn test_submit_assessments() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };

        engine.start_consensus(context);

        // Submit assessments from multiple subsystems
        let assessments = vec![
            create_test_assessment(SubsystemId::Autopoiesis, 0.3, 0.9),
            create_test_assessment(SubsystemId::GameTheory, 0.35, 0.85),
            create_test_assessment(SubsystemId::Physics, 0.28, 0.8),
        ];

        for assessment in assessments {
            engine.submit_assessment(assessment).unwrap();
        }

        // Should have moved to Prepared phase (2f = 2, we submitted 3)
        assert_eq!(engine.current_round().phase, ConsensusPhase::Prepared);
    }

    #[test]
    fn test_finalize_consensus() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };

        engine.start_consensus(context);

        // Submit similar assessments (no Byzantine behavior)
        let assessments = vec![
            create_test_assessment(SubsystemId::Autopoiesis, 0.3, 0.9),
            create_test_assessment(SubsystemId::GameTheory, 0.35, 0.85),
            create_test_assessment(SubsystemId::Physics, 0.28, 0.8),
            create_test_assessment(SubsystemId::Neural, 0.32, 0.88),
        ];

        for assessment in assessments {
            engine.submit_assessment(assessment).unwrap();
        }

        let result = engine.finalize_consensus().unwrap();

        assert!(result.consensus_achieved);
        assert_eq!(result.agreed_risk_level, RiskLevel::Elevated);
        assert!(result.byzantine_suspects.is_empty());
        assert_eq!(result.participant_count, 4);
    }

    #[test]
    fn test_byzantine_detection() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        // First, build up history for one subsystem to be Byzantine
        for _ in 0..10 {
            let context = ProposalContext {
                portfolio_size: 1_000_000.0,
                volatility: 0.2,
                drawdown: 0.05,
                regime_hint: None,
            };

            engine.start_consensus(context);

            // Normal assessments
            engine.submit_assessment(create_test_assessment(
                SubsystemId::Autopoiesis, 0.3, 0.9
            )).unwrap();
            engine.submit_assessment(create_test_assessment(
                SubsystemId::GameTheory, 0.35, 0.85
            )).unwrap();
            engine.submit_assessment(create_test_assessment(
                SubsystemId::Physics, 0.28, 0.8
            )).unwrap();
            // Byzantine-like: always reports high risk when others report low
            engine.submit_assessment(create_test_assessment(
                SubsystemId::Neural, 0.85, 0.9
            )).unwrap();

            let _ = engine.finalize_consensus();
        }

        // After 10 rounds, Neural should be detected as Byzantine
        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };

        engine.start_consensus(context);

        engine.submit_assessment(create_test_assessment(
            SubsystemId::Autopoiesis, 0.3, 0.9
        )).unwrap();
        engine.submit_assessment(create_test_assessment(
            SubsystemId::GameTheory, 0.35, 0.85
        )).unwrap();
        engine.submit_assessment(create_test_assessment(
            SubsystemId::Physics, 0.28, 0.8
        )).unwrap();
        engine.submit_assessment(create_test_assessment(
            SubsystemId::Neural, 0.85, 0.9
        )).unwrap();

        let result = engine.finalize_consensus().unwrap();

        // Neural should be detected as Byzantine
        assert!(result.byzantine_suspects.contains(&SubsystemId::Neural));
    }

    #[test]
    fn test_risk_level_conversion() {
        assert_eq!(BftConsensusEngine::score_to_risk_level(0.1), RiskLevel::Normal);
        assert_eq!(BftConsensusEngine::score_to_risk_level(0.3), RiskLevel::Elevated);
        assert_eq!(BftConsensusEngine::score_to_risk_level(0.6), RiskLevel::High);
        assert_eq!(BftConsensusEngine::score_to_risk_level(0.8), RiskLevel::Critical);
        assert_eq!(BftConsensusEngine::score_to_risk_level(0.95), RiskLevel::Emergency);
    }

    #[test]
    fn test_view_change() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        assert_eq!(engine.current_view(), 0);

        engine.request_view_change(ViewChangeReason::PrimaryTimeout);

        assert_eq!(engine.current_view(), 1);
        assert_eq!(engine.current_round().phase, ConsensusPhase::Idle);
    }

    #[test]
    fn test_manual_byzantine_marking() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        engine.mark_byzantine(SubsystemId::Neural);
        assert!(engine.get_known_byzantine().contains(&SubsystemId::Neural));

        // Cannot submit from Byzantine subsystem
        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };
        engine.start_consensus(context);

        let assessment = create_test_assessment(SubsystemId::Neural, 0.3, 0.9);
        let result = engine.submit_assessment(assessment);
        assert!(result.is_err());

        // Clear Byzantine status
        engine.clear_byzantine(SubsystemId::Neural);
        assert!(!engine.get_known_byzantine().contains(&SubsystemId::Neural));
    }

    #[test]
    fn test_consensus_proof_generation() {
        let config = BftConsensusConfig::default();
        let mut engine = BftConsensusEngine::new(config);

        let context = ProposalContext {
            portfolio_size: 1_000_000.0,
            volatility: 0.2,
            drawdown: 0.05,
            regime_hint: None,
        };

        engine.start_consensus(context);

        engine.submit_assessment(create_test_assessment(
            SubsystemId::Autopoiesis, 0.3, 0.9
        )).unwrap();
        engine.submit_assessment(create_test_assessment(
            SubsystemId::GameTheory, 0.35, 0.85
        )).unwrap();
        engine.submit_assessment(create_test_assessment(
            SubsystemId::Physics, 0.28, 0.8
        )).unwrap();

        let result = engine.finalize_consensus().unwrap();

        // Verify proof structure
        assert!(!result.proof.prepare_merkle_root.is_empty());
        assert!(!result.proof.decision_digest.is_empty());
        assert_eq!(result.proof.votes.len(), 3);

        // All votes should have signatures
        for vote in &result.proof.votes {
            assert!(!vote.signature.is_empty());
        }
    }
}
