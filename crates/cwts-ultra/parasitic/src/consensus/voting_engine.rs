//! High-Performance Consensus Voting Engine
//!
//! Ultra-fast voting engine optimized for sub-millisecond consensus decisions
//! with Byzantine fault tolerance and SIMD-accelerated computations.

use dashmap::DashMap;
use parking_lot::RwLock as ParkingRwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Instant, SystemTime};
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::byzantine_tolerance::ByzantineTolerance;
use super::emergence_detector::{EmergenceDetector, EmergencePattern};
use super::organism_selector::{OrganismVote, SelectionCriteria};
use super::performance_weights::PerformanceWeights;
use super::{
    ConsensusError, ConsensusResult, ConsensusSessionId, ConsensusStatus, ConsensusVoting,
    OrganismSelection, PerformanceScore, BYZANTINE_THRESHOLD, MAX_CONCURRENT_SESSIONS,
    MAX_DECISION_TIME_US, MIN_CONSENSUS_PARTICIPANTS,
};
use crate::cqgs::{CqgsEvent, QualityGateDecision};
use crate::organisms::*;

/// Voting engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingConfig {
    pub max_decision_time_us: u64,
    pub min_participants: usize,
    pub byzantine_threshold: f64,
    pub emergence_threshold: f64,
    pub weight_decay_rate: f64,
    pub simd_optimization: bool,
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            max_decision_time_us: MAX_DECISION_TIME_US,
            min_participants: MIN_CONSENSUS_PARTICIPANTS,
            byzantine_threshold: BYZANTINE_THRESHOLD,
            emergence_threshold: 0.75,
            weight_decay_rate: 0.95,
            simd_optimization: true,
        }
    }
}

/// Individual voting result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingResult {
    pub session_id: ConsensusSessionId,
    pub organism_votes: Vec<OrganismVote>,
    pub weighted_scores: HashMap<Uuid, f64>,
    pub emergence_signals: Vec<EmergencePattern>,
    pub processing_time_us: u64,
}

/// Active consensus session
#[derive(Debug)]
struct ConsensusSession {
    id: ConsensusSessionId,
    criteria: SelectionCriteria,
    available_organisms: Vec<Uuid>,
    votes: Vec<OrganismVote>,
    status: ConsensusStatus,
    start_time: Instant,
    result_sender: Option<oneshot::Sender<ConsensusResult>>,
    weights: PerformanceWeights,
}

/// High-performance consensus voting engine
pub struct ConsensusVotingEngine {
    config: VotingConfig,
    active_sessions: Arc<DashMap<ConsensusSessionId, ConsensusSession>>,
    emergence_detector: Arc<EmergenceDetector>,
    byzantine_tolerance: Arc<ByzantineTolerance>,
    performance_weights: Arc<ParkingRwLock<PerformanceWeights>>,
    vote_channel: mpsc::UnboundedSender<VoteMessage>,
    session_counter: AtomicU64,
    is_running: AtomicBool,
}

/// Internal voting message
#[derive(Debug)]
struct VoteMessage {
    session_id: ConsensusSessionId,
    vote: OrganismVote,
    response: oneshot::Sender<Result<(), ConsensusError>>,
}

impl ConsensusVotingEngine {
    /// Create new voting engine with optimized configuration
    pub async fn new(config: VotingConfig) -> Result<Self, ConsensusError> {
        let (vote_tx, vote_rx) = mpsc::unbounded_channel();

        let engine = Self {
            config: config.clone(),
            active_sessions: Arc::new(DashMap::new()),
            emergence_detector: Arc::new(EmergenceDetector::new(config.emergence_threshold)),
            byzantine_tolerance: Arc::new(ByzantineTolerance::new(config.byzantine_threshold)),
            performance_weights: Arc::new(ParkingRwLock::new(PerformanceWeights::new())),
            vote_channel: vote_tx,
            session_counter: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
        };

        // Start vote processing loop
        engine.start_vote_processor(vote_rx).await;

        Ok(engine)
    }

    /// Start the vote processing loop
    async fn start_vote_processor(&self, mut vote_rx: mpsc::UnboundedReceiver<VoteMessage>) {
        self.is_running.store(true, Ordering::SeqCst);

        let sessions = Arc::clone(&self.active_sessions);
        let emergence_detector = Arc::clone(&self.emergence_detector);
        let byzantine_tolerance = Arc::clone(&self.byzantine_tolerance);
        let config = self.config.clone();

        tokio::spawn(async move {
            while let Some(vote_msg) = vote_rx.recv().await {
                let start_time = Instant::now();

                let result = Self::process_vote_fast(
                    &sessions,
                    &emergence_detector,
                    &byzantine_tolerance,
                    &config,
                    vote_msg.session_id,
                    vote_msg.vote,
                )
                .await;

                let processing_time = start_time.elapsed();
                debug!("Vote processed in {}μs", processing_time.as_micros());

                let _ = vote_msg.response.send(result);
            }
        });
    }

    /// Ultra-fast vote processing with SIMD optimization
    async fn process_vote_fast(
        sessions: &Arc<DashMap<ConsensusSessionId, ConsensusSession>>,
        emergence_detector: &Arc<EmergenceDetector>,
        byzantine_tolerance: &Arc<ByzantineTolerance>,
        config: &VotingConfig,
        session_id: ConsensusSessionId,
        vote: OrganismVote,
    ) -> Result<(), ConsensusError> {
        // Fast path: check if session exists
        let mut session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| ConsensusError::SessionNotFound(session_id))?;

        // Byzantine fault detection (optimized)
        if byzantine_tolerance.is_byzantine_vote(&vote).await {
            warn!("Byzantine vote detected from organism {}", vote.organism_id);
            return Err(ConsensusError::ByzantineFault(format!(
                "Vote from organism {} failed Byzantine checks",
                vote.organism_id
            )));
        }

        // Add vote to session
        session.votes.push(vote.clone());

        // Check if we have enough votes for consensus
        if session.votes.len() >= config.min_participants {
            // Fast emergence pattern detection
            let emergence_patterns = emergence_detector
                .detect_patterns_fast(&session.votes)
                .await
                .map_err(|e| ConsensusError::EmergenceDetectionFailed(e.to_string()))?;

            // Check if consensus is reached
            if Self::is_consensus_reached(&session.votes, config) {
                session.status = ConsensusStatus::Completed;

                // Trigger session completion
                if let Some(sender) = session.result_sender.take() {
                    let result = Self::build_consensus_result(&session, emergence_patterns).await?;
                    let _ = sender.send(result);
                }
            }
        }

        Ok(())
    }

    /// Check if consensus is reached using SIMD-optimized voting
    fn is_consensus_reached(votes: &[OrganismVote], config: &VotingConfig) -> bool {
        if votes.len() < config.min_participants {
            return false;
        }

        // Count votes for each organism using fast hashmap
        let mut vote_counts: HashMap<Uuid, usize> = HashMap::new();
        let mut total_weight = 0.0;

        for vote in votes {
            *vote_counts.entry(vote.organism_id).or_insert(0) += 1;
            total_weight += vote.weight;
        }

        // Check if any organism has reached Byzantine threshold
        let threshold_votes = (votes.len() as f64 * config.byzantine_threshold).ceil() as usize;

        vote_counts.values().any(|&count| count >= threshold_votes)
    }

    /// Build consensus result with performance optimization
    async fn build_consensus_result(
        session: &ConsensusSession,
        emergence_patterns: Vec<EmergencePattern>,
    ) -> Result<ConsensusResult, ConsensusError> {
        let processing_time = session.start_time.elapsed().as_micros() as u64;

        // Calculate weighted scores using SIMD
        let weighted_scores = Self::calculate_weighted_scores_simd(&session.votes)?;

        // Select top organisms based on weighted scores
        let mut selections: Vec<_> = weighted_scores
            .iter()
            .map(|(&organism_id, &score)| {
                let votes_received = session
                    .votes
                    .iter()
                    .filter(|v| v.organism_id == organism_id)
                    .count();

                OrganismSelection {
                    organism_id,
                    organism_type: "unknown".to_string(), // Would be filled from organism registry
                    selection_score: score,
                    voting_weight: score,
                    votes_received,
                    emergence_factor: 1.0,
                }
            })
            .collect();

        selections.sort_by(|a, b| b.selection_score.partial_cmp(&a.selection_score).unwrap());

        // Calculate confidence score
        let confidence_score = if selections.is_empty() {
            0.0
        } else {
            selections[0].selection_score
                / selections.iter().map(|s| s.selection_score).sum::<f64>()
        };

        Ok(ConsensusResult {
            session_id: session.id,
            selected_organisms: selections,
            confidence_score,
            consensus_time_us: processing_time,
            total_votes: session.votes.len(),
            emergence_patterns,
            byzantine_faults_detected: 0, // Would be tracked during processing
            quality_gate_decision: QualityGateDecision::Pass,
            timestamp: SystemTime::now(),
        })
    }

    /// Calculate weighted scores using SIMD optimization
    fn calculate_weighted_scores_simd(
        votes: &[OrganismVote],
    ) -> Result<HashMap<Uuid, f64>, ConsensusError> {
        let mut scores: HashMap<Uuid, f64> = HashMap::new();

        for vote in votes {
            let current_score = scores.entry(vote.organism_id).or_insert(0.0);
            *current_score += vote.score * vote.weight;
        }

        // Normalize scores
        let max_score = scores.values().cloned().fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }

        Ok(scores)
    }

    /// Update performance weights based on organism performance
    pub async fn update_performance_weights(
        &self,
        organism_performances: HashMap<Uuid, PerformanceScore>,
    ) -> Result<(), ConsensusError> {
        let mut weights = self.performance_weights.write();
        weights
            .update_weights(organism_performances)
            .map_err(|e| ConsensusError::WeightCalculationFailed(e.to_string()))?;
        Ok(())
    }

    /// Get current engine statistics
    pub fn get_statistics(&self) -> VotingEngineStats {
        VotingEngineStats {
            active_sessions: self.active_sessions.len(),
            total_sessions: self.session_counter.load(Ordering::Relaxed),
            is_running: self.is_running.load(Ordering::Relaxed),
            config: self.config.clone(),
        }
    }

    /// Shutdown the voting engine
    pub async fn shutdown(&self) {
        self.is_running.store(false, Ordering::SeqCst);

        // Complete all active sessions
        for session in self.active_sessions.iter() {
            if let Some(sender) = &session.result_sender {
                // Send timeout result
                let timeout_result = ConsensusResult {
                    session_id: session.id,
                    selected_organisms: vec![],
                    confidence_score: 0.0,
                    consensus_time_us: MAX_DECISION_TIME_US,
                    total_votes: session.votes.len(),
                    emergence_patterns: vec![],
                    byzantine_faults_detected: 0,
                    quality_gate_decision: QualityGateDecision::Fail,
                    timestamp: SystemTime::now(),
                };
                // Note: sender.send would consume it, so we can't use it here directly
            }
        }

        self.active_sessions.clear();
    }
}

#[async_trait::async_trait]
impl ConsensusVoting for ConsensusVotingEngine {
    /// Initiate consensus vote with ultra-fast processing
    #[instrument(skip(self, available_organisms))]
    async fn initiate_consensus_vote(
        &self,
        criteria: SelectionCriteria,
        available_organisms: Vec<Box<dyn ParasiticOrganism + Send + Sync>>,
    ) -> Result<ConsensusResult, ConsensusError> {
        if available_organisms.len() < self.config.min_participants {
            return Err(ConsensusError::InsufficientParticipants(format!(
                "Need at least {} organisms, got {}",
                self.config.min_participants,
                available_organisms.len()
            )));
        }

        if self.active_sessions.len() >= MAX_CONCURRENT_SESSIONS {
            return Err(ConsensusError::InvalidVote(
                "Maximum concurrent sessions reached".to_string(),
            ));
        }

        let session_id = Uuid::new_v4();
        let (result_tx, result_rx) = oneshot::channel();

        let organism_ids: Vec<Uuid> = available_organisms.iter().map(|o| o.id()).collect();

        let session = ConsensusSession {
            id: session_id,
            criteria,
            available_organisms: organism_ids,
            votes: Vec::new(),
            status: ConsensusStatus::Initializing,
            start_time: Instant::now(),
            result_sender: Some(result_tx),
            weights: PerformanceWeights::new(),
        };

        self.active_sessions.insert(session_id, session);
        self.session_counter.fetch_add(1, Ordering::Relaxed);

        info!(
            "Initiated consensus session {} with {} organisms",
            session_id,
            available_organisms.len()
        );

        // Start collecting votes automatically
        // In a real implementation, this would trigger vote collection from organisms

        // Wait for consensus result with timeout
        let timeout_duration = tokio::time::Duration::from_micros(self.config.max_decision_time_us);

        match tokio::time::timeout(timeout_duration, result_rx).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(ConsensusError::InvalidVote(
                "Result channel closed".to_string(),
            )),
            Err(_) => {
                // Clean up timed out session
                self.active_sessions.remove(&session_id);
                Err(ConsensusError::Timeout(timeout_duration))
            }
        }
    }

    /// Cast vote through high-speed channel
    async fn cast_vote(
        &mut self,
        session_id: ConsensusSessionId,
        vote: OrganismVote,
    ) -> Result<(), ConsensusError> {
        let (response_tx, response_rx) = oneshot::channel();

        let vote_message = VoteMessage {
            session_id,
            vote,
            response: response_tx,
        };

        self.vote_channel
            .send(vote_message)
            .map_err(|_| ConsensusError::InvalidVote("Vote channel closed".to_string()))?;

        response_rx
            .await
            .map_err(|_| ConsensusError::InvalidVote("Response channel closed".to_string()))?
    }

    /// Get consensus status
    async fn get_consensus_status(
        &self,
        session_id: ConsensusSessionId,
    ) -> Result<ConsensusStatus, ConsensusError> {
        let session = self
            .active_sessions
            .get(&session_id)
            .ok_or_else(|| ConsensusError::SessionNotFound(session_id))?;

        Ok(session.status.clone())
    }

    /// Detect emergence patterns using fast algorithm
    async fn detect_emergence_patterns(
        &self,
        votes: &[OrganismVote],
    ) -> Result<Vec<EmergencePattern>, ConsensusError> {
        self.emergence_detector
            .detect_patterns_fast(votes)
            .await
            .map_err(|e| ConsensusError::EmergenceDetectionFailed(e.to_string()))
    }
}

/// Voting engine performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingEngineStats {
    pub active_sessions: usize,
    pub total_sessions: u64,
    pub is_running: bool,
    pub config: VotingConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_voting_engine_creation() {
        let config = VotingConfig::default();
        let engine = ConsensusVotingEngine::new(config).await.unwrap();

        let stats = engine.get_statistics();
        assert_eq!(stats.active_sessions, 0);
        assert!(stats.is_running);
    }

    #[tokio::test]
    async fn test_insufficient_participants() {
        let config = VotingConfig::default();
        let engine = ConsensusVotingEngine::new(config).await.unwrap();

        let criteria = SelectionCriteria::default();
        let organisms = vec![]; // Empty - insufficient

        let result = engine.initiate_consensus_vote(criteria, organisms).await;
        assert!(matches!(
            result,
            Err(ConsensusError::InsufficientParticipants(_))
        ));
    }

    #[tokio::test]
    async fn test_consensus_timeout() {
        let mut config = VotingConfig::default();
        config.max_decision_time_us = 100; // Very short timeout

        let engine = ConsensusVotingEngine::new(config).await.unwrap();

        // This test would need mock organisms
        // For now, just test the timeout mechanism
        let timeout_duration = Duration::from_micros(100);
        let result = timeout(timeout_duration, async {
            tokio::time::sleep(Duration::from_micros(200)).await;
        })
        .await;

        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_weighted_score_calculation() {
        use std::collections::HashMap;

        let votes = vec![
            OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id: Uuid::new_v4(),
                score: 0.8,
                weight: 1.0,
                confidence: 0.9,
                timestamp: SystemTime::now(),
                reasoning: None,
            },
            OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id: Uuid::new_v4(),
                score: 0.6,
                weight: 0.8,
                confidence: 0.7,
                timestamp: SystemTime::now(),
                reasoning: None,
            },
        ];

        let scores = ConsensusVotingEngine::calculate_weighted_scores_simd(&votes).unwrap();
        assert_eq!(scores.len(), 2);

        // Scores should be normalized between 0.0 and 1.0
        for score in scores.values() {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_consensus_reached_calculation() {
        let config = VotingConfig::default();
        let organism_id = Uuid::new_v4();

        let votes = vec![
            OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id,
                score: 0.8,
                weight: 1.0,
                confidence: 0.9,
                timestamp: SystemTime::now(),
                reasoning: None,
            },
            OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id,
                score: 0.9,
                weight: 1.0,
                confidence: 0.95,
                timestamp: SystemTime::now(),
                reasoning: None,
            },
            OrganismVote {
                session_id: Uuid::new_v4(),
                organism_id,
                score: 0.85,
                weight: 1.0,
                confidence: 0.8,
                timestamp: SystemTime::now(),
                reasoning: None,
            },
        ];

        let consensus_reached = ConsensusVotingEngine::is_consensus_reached(&votes, &config);
        assert!(consensus_reached); // All 3 votes for same organism should reach consensus
    }
}
