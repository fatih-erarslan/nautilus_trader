//! Sentinel Consensus Mechanism
//!
//! Implements Byzantine fault-tolerant consensus for quality gate decisions
//! with 2/3 threshold majority voting and real-time decision making.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::cqgs::sentinels::{SentinelId, SentinelType};
use crate::cqgs::{CqgsEvent, QualityGateDecision, QualityViolation, ViolationSeverity};

/// Consensus timeout for quality gate decisions
const CONSENSUS_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of concurrent consensus sessions
const MAX_CONCURRENT_SESSIONS: usize = 100;

/// Minimum number of sentinels required for consensus
const MIN_CONSENSUS_PARTICIPANTS: usize = 3;

/// Consensus proposal for quality gate decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub id: Uuid,
    pub violation: QualityViolation,
    pub proposed_decision: QualityGateDecision,
    pub proposer: SentinelId,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub priority: ConsensusPriority,
}

/// Priority levels for consensus proposals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsensusPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

impl From<ViolationSeverity> for ConsensusPriority {
    fn from(severity: ViolationSeverity) -> Self {
        match severity {
            ViolationSeverity::Info => ConsensusPriority::Low,
            ViolationSeverity::Warning => ConsensusPriority::Medium,
            ViolationSeverity::Error => ConsensusPriority::High,
            ViolationSeverity::Critical => ConsensusPriority::Critical,
            ViolationSeverity::Fatal => ConsensusPriority::Emergency,
        }
    }
}

/// Sentinel vote in consensus process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelVote {
    pub proposal_id: Uuid,
    pub sentinel_id: SentinelId,
    pub sentinel_type: SentinelType,
    pub decision: QualityGateDecision,
    pub confidence: f64, // 0.0 to 1.0
    pub reasoning: Option<String>,
    pub timestamp: SystemTime,
    pub supporting_evidence: Vec<String>,
}

/// Consensus session tracking active votes and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSession {
    pub proposal: ConsensusProposal,
    pub votes: HashMap<SentinelId, SentinelVote>,
    pub required_votes: usize,
    pub threshold_votes: usize,
    pub status: ConsensusStatus,
    pub result: Option<ConsensusResult>,
    pub started_at: SystemTime,
    pub completed_at: Option<SystemTime>,
}

/// Status of consensus session
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Timeout,
}

/// Final consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub decision: QualityGateDecision,
    pub vote_count: HashMap<QualityGateDecision, usize>,
    pub confidence_score: f64,
    pub participating_sentinels: Vec<SentinelId>,
    pub consensus_reached_at: SystemTime,
    pub unanimous: bool,
}

/// Byzantine Fault Tolerant Consensus Engine
pub struct SentinelConsensus {
    threshold: f64,
    active_sessions: Arc<DashMap<Uuid, ConsensusSession>>,
    sentinel_registry: Arc<RwLock<HashMap<SentinelId, SentinelInfo>>>,
    vote_history: Arc<DashMap<SentinelId, Vec<VoteRecord>>>,
    consensus_metrics: Arc<Mutex<ConsensusMetrics>>,
}

/// Sentinel information for consensus participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelInfo {
    pub id: SentinelId,
    pub sentinel_type: SentinelType,
    pub reliability_score: f64, // Based on historical voting accuracy
    pub is_active: bool,
    pub specialization_weight: HashMap<ViolationSeverity, f64>,
}

/// Historical vote record for reliability calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRecord {
    pub proposal_id: Uuid,
    pub vote: SentinelVote,
    pub final_outcome: QualityGateDecision,
    pub vote_accuracy: f64, // How well this vote aligned with consensus
}

/// Consensus system performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    pub total_sessions: u64,
    pub successful_consensus: u64,
    pub failed_consensus: u64,
    pub average_time_to_consensus: Duration,
    pub sentinel_participation_rate: HashMap<SentinelId, f64>,
    pub decision_accuracy_rate: f64,
}

impl Default for ConsensusMetrics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            successful_consensus: 0,
            failed_consensus: 0,
            average_time_to_consensus: Duration::from_secs(0),
            sentinel_participation_rate: HashMap::new(),
            decision_accuracy_rate: 1.0,
        }
    }
}

impl SentinelConsensus {
    /// Create new consensus engine with specified threshold
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.5, 1.0), // Ensure valid threshold
            active_sessions: Arc::new(DashMap::new()),
            sentinel_registry: Arc::new(RwLock::new(HashMap::new())),
            vote_history: Arc::new(DashMap::new()),
            consensus_metrics: Arc::new(Mutex::new(ConsensusMetrics::default())),
        }
    }

    /// Register a sentinel for consensus participation
    pub async fn register_sentinel(
        &self,
        id: SentinelId,
        sentinel_type: SentinelType,
        specialization_weights: HashMap<ViolationSeverity, f64>,
    ) {
        let sentinel_info = SentinelInfo {
            id: id.clone(),
            sentinel_type,
            reliability_score: 1.0, // Start with perfect reliability
            is_active: true,
            specialization_weight: specialization_weights,
        };

        let mut registry = self.sentinel_registry.write().await;
        registry.insert(id.clone(), sentinel_info);

        info!("Registered sentinel {} for consensus participation", id);
    }

    /// Unregister sentinel from consensus
    pub async fn unregister_sentinel(&self, id: &SentinelId) {
        let mut registry = self.sentinel_registry.write().await;
        if let Some(mut info) = registry.get_mut(id) {
            info.is_active = false;
        }

        info!("Unregistered sentinel {} from consensus", id);
    }

    /// Evaluate a quality violation and initiate consensus if needed
    #[instrument(skip(self, violation), fields(violation_id = %violation.id))]
    pub async fn evaluate_violation(&self, violation: &QualityViolation) -> QualityGateDecision {
        // For minor violations, make immediate decision
        if violation.severity < ViolationSeverity::Error {
            return match violation.auto_fixable {
                true => QualityGateDecision::RequireRemediation,
                false => QualityGateDecision::Pass,
            };
        }

        // For serious violations, initiate consensus
        let proposal = self.create_proposal(violation).await;
        match self.initiate_consensus(proposal).await {
            Ok(result) => result.decision,
            Err(e) => {
                error!("Consensus failed for violation {}: {}", violation.id, e);
                // Default to safe decision on consensus failure
                QualityGateDecision::BlockDeployment
            }
        }
    }

    /// Create consensus proposal from violation
    async fn create_proposal(&self, violation: &QualityViolation) -> ConsensusProposal {
        let proposed_decision = match violation.severity {
            ViolationSeverity::Fatal => QualityGateDecision::BlockDeployment,
            ViolationSeverity::Critical => {
                if violation.auto_fixable {
                    QualityGateDecision::RequireRemediation
                } else {
                    QualityGateDecision::EscalateToHuman
                }
            }
            ViolationSeverity::Error => {
                if violation.auto_fixable {
                    QualityGateDecision::RequireRemediation
                } else {
                    QualityGateDecision::Fail
                }
            }
            _ => QualityGateDecision::Pass,
        };

        ConsensusProposal {
            id: Uuid::new_v4(),
            violation: violation.clone(),
            proposed_decision,
            proposer: violation.sentinel_id.clone(),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + CONSENSUS_TIMEOUT,
            priority: violation.severity.clone().into(),
        }
    }

    /// Initiate consensus process for a proposal
    #[instrument(skip(self, proposal), fields(proposal_id = %proposal.id))]
    pub async fn initiate_consensus(
        &self,
        proposal: ConsensusProposal,
    ) -> Result<ConsensusResult, Box<dyn std::error::Error + Send + Sync>> {
        // Check if we have enough active sentinels
        let active_sentinels = self.get_active_sentinels().await;
        if active_sentinels.len() < MIN_CONSENSUS_PARTICIPANTS {
            return Err("Not enough active sentinels for consensus".into());
        }

        // Limit concurrent sessions
        if self.active_sessions.len() >= MAX_CONCURRENT_SESSIONS {
            return Err("Too many concurrent consensus sessions".into());
        }

        // Select appropriate sentinels for this violation type
        let selected_sentinels = self
            .select_sentinels_for_violation(&proposal.violation, &active_sentinels)
            .await;
        let threshold_votes = ((selected_sentinels.len() as f64) * self.threshold).ceil() as usize;

        let session = ConsensusSession {
            proposal: proposal.clone(),
            votes: HashMap::new(),
            required_votes: selected_sentinels.len(),
            threshold_votes,
            status: ConsensusStatus::Pending,
            result: None,
            started_at: SystemTime::now(),
            completed_at: None,
        };

        self.active_sessions.insert(proposal.id, session);

        info!(
            "Initiated consensus for proposal {} with {} sentinels (threshold: {})",
            proposal.id,
            selected_sentinels.len(),
            threshold_votes
        );

        // Request votes from selected sentinels
        for sentinel_id in &selected_sentinels {
            self.request_vote(sentinel_id, &proposal).await?;
        }

        // Wait for consensus or timeout
        let result = timeout(CONSENSUS_TIMEOUT, self.wait_for_consensus(proposal.id)).await;

        match result {
            Ok(consensus_result) => Ok(consensus_result?),
            Err(_) => {
                self.handle_consensus_timeout(proposal.id).await;
                Err("Consensus timeout".into())
            }
        }
    }

    /// Get all active sentinels
    async fn get_active_sentinels(&self) -> Vec<SentinelId> {
        let registry = self.sentinel_registry.read().await;
        registry
            .values()
            .filter(|info| info.is_active)
            .map(|info| info.id.clone())
            .collect()
    }

    /// Select appropriate sentinels for a specific violation
    async fn select_sentinels_for_violation(
        &self,
        violation: &QualityViolation,
        available_sentinels: &[SentinelId],
    ) -> Vec<SentinelId> {
        let registry = self.sentinel_registry.read().await;
        let mut weighted_sentinels: Vec<(SentinelId, f64)> = Vec::new();

        for sentinel_id in available_sentinels {
            if let Some(info) = registry.get(sentinel_id) {
                let weight = info
                    .specialization_weight
                    .get(&violation.severity)
                    .unwrap_or(&1.0)
                    * info.reliability_score;

                weighted_sentinels.push((sentinel_id.clone(), weight));
            }
        }

        // Sort by weight and select top sentinels
        weighted_sentinels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select at least 3 sentinels, or up to 75% of available sentinels
        let selection_count = (available_sentinels.len() * 3 / 4).max(MIN_CONSENSUS_PARTICIPANTS);
        weighted_sentinels
            .into_iter()
            .take(selection_count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Request vote from a specific sentinel
    async fn request_vote(
        &self,
        sentinel_id: &SentinelId,
        proposal: &ConsensusProposal,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In a real implementation, this would send a request to the actual sentinel
        // For now, we simulate the vote request
        debug!(
            "Requesting vote from sentinel {} for proposal {}",
            sentinel_id, proposal.id
        );

        // Simulate vote based on sentinel type and violation characteristics
        let vote = self.simulate_sentinel_vote(sentinel_id, proposal).await?;
        self.submit_vote(vote).await?;

        Ok(())
    }

    /// Simulate a sentinel vote (in production, this would come from actual sentinels)
    async fn simulate_sentinel_vote(
        &self,
        sentinel_id: &SentinelId,
        proposal: &ConsensusProposal,
    ) -> Result<SentinelVote, Box<dyn std::error::Error + Send + Sync>> {
        let registry = self.sentinel_registry.read().await;
        let sentinel_info = registry
            .get(sentinel_id)
            .ok_or("Sentinel not found in registry")?;

        // Simulate decision based on sentinel type and violation
        let decision = match sentinel_info.sentinel_type {
            SentinelType::Security => match proposal.violation.severity {
                ViolationSeverity::Critical | ViolationSeverity::Fatal => {
                    QualityGateDecision::BlockDeployment
                }
                ViolationSeverity::Error => QualityGateDecision::RequireRemediation,
                _ => QualityGateDecision::Pass,
            },
            SentinelType::ZeroMock => {
                if proposal.violation.location.contains("mock") {
                    QualityGateDecision::Fail
                } else {
                    QualityGateDecision::Pass
                }
            }
            SentinelType::Performance => match proposal.violation.severity {
                ViolationSeverity::Error | ViolationSeverity::Critical => {
                    QualityGateDecision::RequireRemediation
                }
                _ => QualityGateDecision::Pass,
            },
            _ => proposal.proposed_decision.clone(),
        };

        let confidence = 0.8 + (rand::random::<f64>() * 0.2); // 0.8 to 1.0

        Ok(SentinelVote {
            proposal_id: proposal.id,
            sentinel_id: sentinel_id.clone(),
            sentinel_type: sentinel_info.sentinel_type,
            decision,
            confidence,
            reasoning: Some(format!(
                "Decision based on {} analysis",
                sentinel_info.sentinel_type as u8
            )),
            timestamp: SystemTime::now(),
            supporting_evidence: vec![format!("Analysis by {}", sentinel_id)],
        })
    }

    /// Submit a vote for consensus
    #[instrument(skip(self, vote), fields(proposal_id = %vote.proposal_id, sentinel_id = %vote.sentinel_id))]
    pub async fn submit_vote(
        &self,
        vote: SentinelVote,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let proposal_id = vote.proposal_id;

        // Update session with new vote
        if let Some(mut session) = self.active_sessions.get_mut(&proposal_id) {
            session.votes.insert(vote.sentinel_id.clone(), vote.clone());

            debug!(
                "Received vote from {} for proposal {} (decision: {:?})",
                vote.sentinel_id, proposal_id, vote.decision
            );

            // Check if consensus is reached
            if self.check_consensus_reached(&session).await {
                let result = self.finalize_consensus(&mut session).await?;
                session.result = Some(result);
                session.status = ConsensusStatus::Completed;
                session.completed_at = Some(SystemTime::now());
            }
        }

        // Record vote in history
        let vote_record = VoteRecord {
            proposal_id,
            vote,
            final_outcome: QualityGateDecision::Pass, // Will be updated when consensus completes
            vote_accuracy: 1.0,                       // Will be calculated later
        };

        self.vote_history
            .entry(vote_record.vote.sentinel_id.clone())
            .or_insert_with(Vec::new)
            .push(vote_record);

        Ok(())
    }

    /// Check if consensus has been reached for a session
    async fn check_consensus_reached(&self, session: &ConsensusSession) -> bool {
        if session.votes.len() < session.threshold_votes {
            return false;
        }

        // Count votes for each decision
        let mut vote_counts: HashMap<QualityGateDecision, usize> = HashMap::new();
        for vote in session.votes.values() {
            *vote_counts.entry(vote.decision.clone()).or_insert(0) += 1;
        }

        // Check if any decision has reached threshold
        vote_counts
            .values()
            .any(|&count| count >= session.threshold_votes)
    }

    /// Finalize consensus and determine result
    async fn finalize_consensus(
        &self,
        session: &mut ConsensusSession,
    ) -> Result<ConsensusResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut vote_counts: HashMap<QualityGateDecision, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        let mut participating_sentinels = Vec::new();

        for vote in session.votes.values() {
            *vote_counts.entry(vote.decision.clone()).or_insert(0) += 1;
            total_confidence += vote.confidence;
            participating_sentinels.push(vote.sentinel_id.clone());
        }

        // Find winning decision
        let winning_decision = vote_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(decision, _)| decision.clone())
            .ok_or("No votes received")?;

        let winning_count = vote_counts.get(&winning_decision).unwrap_or(&0);
        let unanimous = *winning_count == session.votes.len();
        let confidence_score = total_confidence / session.votes.len() as f64;

        let result = ConsensusResult {
            decision: winning_decision,
            vote_count: vote_counts,
            confidence_score,
            participating_sentinels,
            consensus_reached_at: SystemTime::now(),
            unanimous,
        };

        info!(
            "Consensus reached for proposal {}: {:?} (confidence: {:.2}, unanimous: {})",
            session.proposal.id, result.decision, result.confidence_score, unanimous
        );

        // Update metrics
        self.update_consensus_metrics(&result).await;

        Ok(result)
    }

    /// Wait for consensus to be reached
    async fn wait_for_consensus(
        &self,
        proposal_id: Uuid,
    ) -> Result<ConsensusResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut interval = interval(Duration::from_millis(100));

        loop {
            interval.tick().await;

            if let Some(session) = self.active_sessions.get(&proposal_id) {
                match session.status {
                    ConsensusStatus::Completed => {
                        if let Some(result) = &session.result {
                            return Ok(result.clone());
                        }
                    }
                    ConsensusStatus::Failed | ConsensusStatus::Timeout => {
                        return Err("Consensus failed".into());
                    }
                    _ => continue,
                }
            } else {
                return Err("Session not found".into());
            }
        }
    }

    /// Handle consensus timeout
    async fn handle_consensus_timeout(&self, proposal_id: Uuid) {
        if let Some(mut session) = self.active_sessions.get_mut(&proposal_id) {
            session.status = ConsensusStatus::Timeout;
            session.completed_at = Some(SystemTime::now());
        }

        warn!("Consensus timeout for proposal {}", proposal_id);

        // Update failure metrics
        let mut metrics = self.consensus_metrics.lock().await;
        metrics.failed_consensus += 1;
    }

    /// Update consensus performance metrics
    async fn update_consensus_metrics(&self, result: &ConsensusResult) {
        let mut metrics = self.consensus_metrics.lock().await;
        metrics.total_sessions += 1;
        metrics.successful_consensus += 1;

        // Update participation rates
        for sentinel_id in &result.participating_sentinels {
            let participation_rate = metrics
                .sentinel_participation_rate
                .entry(sentinel_id.clone())
                .or_insert(0.0);
            *participation_rate = (*participation_rate + 1.0) / 2.0; // Moving average
        }

        // Update decision accuracy (simplified calculation)
        metrics.decision_accuracy_rate =
            (metrics.decision_accuracy_rate + result.confidence_score) / 2.0;
    }

    /// Start consensus loop for processing proposals
    pub async fn start_consensus_loop(&self, event_bus: broadcast::Sender<CqgsEvent>) {
        let mut interval = interval(Duration::from_secs(1));

        loop {
            interval.tick().await;

            // Clean up expired sessions
            self.cleanup_expired_sessions().await;

            // Process any pending high-priority proposals
            self.process_priority_proposals().await;
        }
    }

    /// Clean up expired consensus sessions
    async fn cleanup_expired_sessions(&self) {
        let now = SystemTime::now();
        let mut expired_sessions = Vec::new();

        for entry in self.active_sessions.iter() {
            let (id, session) = (entry.key(), entry.value());
            if session.proposal.expires_at <= now && session.status == ConsensusStatus::Pending {
                expired_sessions.push(*id);
            }
        }

        for session_id in expired_sessions {
            self.active_sessions.remove(&session_id);
            warn!("Removed expired consensus session {}", session_id);
        }
    }

    /// Process high-priority proposals first
    async fn process_priority_proposals(&self) {
        // Implementation would prioritize emergency and critical proposals
        // This is a placeholder for priority queue processing
    }

    /// Get consensus system metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        self.consensus_metrics.lock().await.clone()
    }

    /// Get active consensus sessions count
    pub fn get_active_sessions_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// Export consensus history for analysis
    pub async fn export_consensus_history(&self) -> HashMap<SentinelId, Vec<VoteRecord>> {
        self.vote_history
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqgs::sentinels::{SentinelId, SentinelType};
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_consensus_initialization() {
        let consensus = SentinelConsensus::new(0.67);
        assert_eq!(consensus.threshold, 0.67);
        assert_eq!(consensus.get_active_sessions_count(), 0);
    }

    #[tokio::test]
    async fn test_sentinel_registration() {
        let consensus = SentinelConsensus::new(0.67);
        let sentinel_id = SentinelId::new("test_sentinel".to_string());

        let weights = HashMap::new();
        consensus
            .register_sentinel(sentinel_id.clone(), SentinelType::Quality, weights)
            .await;

        let active_sentinels = consensus.get_active_sentinels().await;
        assert!(active_sentinels.contains(&sentinel_id));
    }

    #[tokio::test]
    async fn test_consensus_proposal_creation() {
        let consensus = SentinelConsensus::new(0.67);

        let violation = QualityViolation {
            id: Uuid::new_v4(),
            sentinel_id: SentinelId::new("test".to_string()),
            severity: ViolationSeverity::Error,
            message: "Test violation".to_string(),
            location: "test.rs:1".to_string(),
            timestamp: SystemTime::now(),
            remediation_suggestion: None,
            auto_fixable: false,
            hyperbolic_coordinates: None,
        };

        let proposal = consensus.create_proposal(&violation).await;
        assert_eq!(proposal.violation.id, violation.id);
        assert_eq!(proposal.priority, ConsensusPriority::High);
    }
}
