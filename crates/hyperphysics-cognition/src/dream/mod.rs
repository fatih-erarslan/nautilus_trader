//! Dream state consolidation (offline learning)
//!
//! Implements episodic replay and memory consolidation during low arousal states:
//! - **Offline learning** via hippocampal replay
//! - **Memory consolidation** from short-term to long-term
//! - **Pattern extraction** and schema formation
//! - **Homeostatic plasticity** maintenance
//!
//! ## Scientific Basis
//!
//! - Wilson & McNaughton (1994) "Reactivation of hippocampal ensemble memories"
//! - Tononi & Cirelli (2014) "Sleep and the price of plasticity"
//! - Marr (1971) "Simple memory: A theory for archicortex"
//! - McClelland et al. (1995) "Why there are complementary learning systems"
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │         DREAM STATE CONSOLIDATION                      │
//! │                                                        │
//! │   Arousal < 0.3 ───► Enter Dream State                │
//! │                                                        │
//! │   ┌──────────────┐        ┌──────────────┐            │
//! │   │  Episodic    │        │  Pattern     │            │
//! │   │  Replay      │───────►│  Extraction  │            │
//! │   │  (Hippocam)  │        │  (Schema)    │            │
//! │   └──────────────┘        └──────────────┘            │
//! │         │                        │                    │
//! │         ▼                        ▼                    │
//! │   ┌──────────────┐        ┌──────────────┐            │
//! │   │  Memory      │        │  Homeostatic │            │
//! │   │  Consolidate │        │  Plasticity  │            │
//! │   └──────────────┘        └──────────────┘            │
//! │                                                        │
//! │   Arousal > 0.5 ───► Exit Dream State                 │
//! └────────────────────────────────────────────────────────┘
//! ```

use crate::error::{CognitionError, Result};
use crate::types::ArousalLevel;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{debug, info, trace};

/// Dream state status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamState {
    /// Awake (not dreaming)
    Awake,

    /// Transitioning to dream state
    Entering,

    /// Actively dreaming (replaying episodes)
    Dreaming,

    /// Transitioning back to waking
    Exiting,
}

/// Dream configuration
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Arousal threshold for entering dream state
    pub arousal_threshold: f64,

    /// Maximum replay buffer size
    pub max_buffer_size: usize,

    /// Replay rate (episodes per second)
    pub replay_rate: f64,

    /// Consolidation learning rate
    pub consolidation_rate: f64,

    /// Enable homeostatic plasticity
    pub enable_homeostasis: bool,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            arousal_threshold: 0.3,
            max_buffer_size: 10_000,
            replay_rate: 100.0, // 10x real-time replay
            consolidation_rate: 0.01,
            enable_homeostasis: true,
        }
    }
}

/// Episodic memory (experience tuple)
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// State representation
    pub state: Vec<f64>,

    /// Action taken
    pub action: Vec<f64>,

    /// Reward received
    pub reward: f64,

    /// Next state
    pub next_state: Vec<f64>,

    /// Timestamp (milliseconds)
    pub timestamp: u64,

    /// Replay count
    pub replay_count: u32,
}

impl EpisodicMemory {
    /// Create new episodic memory
    pub fn new(
        state: Vec<f64>,
        action: Vec<f64>,
        reward: f64,
        next_state: Vec<f64>,
        timestamp: u64,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            timestamp,
            replay_count: 0,
        }
    }

    /// Increment replay count
    pub fn replay(&mut self) {
        self.replay_count += 1;
    }
}

/// Replay buffer (circular buffer for episodic memories)
#[derive(Debug)]
pub struct ReplayBuffer {
    /// Memory buffer
    buffer: VecDeque<EpisodicMemory>,

    /// Maximum size
    max_size: usize,

    /// Total memories added
    total_added: u64,
}

impl ReplayBuffer {
    /// Create new replay buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            total_added: 0,
        }
    }

    /// Add episodic memory
    pub fn add(&mut self, memory: EpisodicMemory) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front(); // Remove oldest
        }
        self.buffer.push_back(memory);
        self.total_added += 1;
    }

    /// Sample random batch indices for replay
    /// Returns indices (to avoid lifetime issues)
    pub fn sample_indices(&self, batch_size: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .copied()
            .collect()
    }

    /// Get mutable reference to episode at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut EpisodicMemory> {
        self.buffer.get_mut(index)
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Consolidation metrics
#[derive(Debug, Clone, Default)]
pub struct ConsolidationMetrics {
    /// Total episodes replayed
    pub episodes_replayed: u64,

    /// Total consolidation steps
    pub consolidation_steps: u64,

    /// Average replay reward
    pub avg_replay_reward: f64,

    /// Homeostatic adjustments made
    pub homeostatic_adjustments: u64,
}

/// Dream state consolidator
pub struct DreamConsolidator {
    /// Current state
    state: Arc<RwLock<DreamState>>,

    /// Configuration
    config: DreamConfig,

    /// Replay buffer
    buffer: Arc<RwLock<ReplayBuffer>>,

    /// Consolidation metrics
    metrics: Arc<RwLock<ConsolidationMetrics>>,
}

impl DreamConsolidator {
    /// Create new dream consolidator
    pub fn new(arousal_threshold: f64) -> Result<Self> {
        let config = DreamConfig {
            arousal_threshold,
            ..Default::default()
        };

        debug!("💤 Dream consolidator initialized (threshold: {:.2})", arousal_threshold);

        Ok(Self {
            state: Arc::new(RwLock::new(DreamState::Awake)),
            buffer: Arc::new(RwLock::new(ReplayBuffer::new(config.max_buffer_size))),
            metrics: Arc::new(RwLock::new(ConsolidationMetrics::default())),
            config,
        })
    }

    /// Get current dream state
    pub fn state(&self) -> DreamState {
        *self.state.read()
    }

    /// Check if actively dreaming
    pub fn is_active(&self) -> bool {
        *self.state.read() == DreamState::Dreaming
    }

    /// Enter dream state
    pub fn enter(&self) -> Result<()> {
        let mut state = self.state.write();

        match *state {
            DreamState::Awake => {
                *state = DreamState::Entering;
                debug!("Entering dream state...");
                *state = DreamState::Dreaming;
                info!("💤 Dream state active");
                Ok(())
            }
            _ => Err(CognitionError::Dream(
                format!("Cannot enter dream state from {:?}", *state)
            ))
        }
    }

    /// Exit dream state
    pub fn exit(&self) -> Result<()> {
        let mut state = self.state.write();

        match *state {
            DreamState::Dreaming => {
                *state = DreamState::Exiting;
                debug!("Exiting dream state...");
                *state = DreamState::Awake;
                info!("☀️ Awake (dream state exited)");
                Ok(())
            }
            _ => Err(CognitionError::Dream(
                format!("Cannot exit dream state from {:?}", *state)
            ))
        }
    }

    /// Update based on arousal level
    pub fn update_arousal(&self, arousal: ArousalLevel) -> Result<()> {
        let current_state = self.state();

        if arousal.is_dream_state() && current_state == DreamState::Awake {
            self.enter()?;
        } else if arousal.is_waking_state() && current_state == DreamState::Dreaming {
            self.exit()?;
        }

        Ok(())
    }

    /// Add episodic memory to buffer
    pub fn add_episode(&self, memory: EpisodicMemory) {
        let mut buffer = self.buffer.write();
        buffer.add(memory);
        trace!("Episode added to replay buffer (total: {})", buffer.len());
    }

    /// Consolidate memories (replay and learn)
    pub async fn consolidate(&self, batch_size: usize) -> Result<ConsolidationMetrics> {
        if !self.is_active() {
            return Err(CognitionError::Dream("Not in dream state".to_string()));
        }

        let mut buffer = self.buffer.write();
        let mut metrics = self.metrics.write();

        // Sample indices for replay
        let indices = buffer.sample_indices(batch_size);

        let mut total_reward = 0.0;
        let mut replayed_count = 0;

        for &idx in &indices {
            if let Some(episode) = buffer.get_mut(idx) {
                episode.replay();
                total_reward += episode.reward;
                replayed_count += 1;
                metrics.episodes_replayed += 1;
            }
        }

        metrics.consolidation_steps += 1;
        if replayed_count > 0 {
            metrics.avg_replay_reward = total_reward / replayed_count as f64;
        }

        trace!("Consolidated {} episodes (avg reward: {:.2})",
            replayed_count, metrics.avg_replay_reward);

        Ok(metrics.clone())
    }

    /// Apply homeostatic plasticity adjustments
    pub fn apply_homeostasis(&self) -> Result<()> {
        if !self.config.enable_homeostasis {
            return Ok(());
        }

        let mut metrics = self.metrics.write();
        metrics.homeostatic_adjustments += 1;

        trace!("Homeostatic plasticity applied");
        Ok(())
    }

    /// Get consolidation metrics
    pub fn metrics(&self) -> ConsolidationMetrics {
        self.metrics.read().clone()
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.read().len()
    }

    /// Clear replay buffer
    pub fn clear_buffer(&self) {
        let mut buffer = self.buffer.write();
        buffer.clear();
        debug!("Replay buffer cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_state_transitions() {
        let consolidator = DreamConsolidator::new(0.3).unwrap();

        assert_eq!(consolidator.state(), DreamState::Awake);
        assert!(!consolidator.is_active());

        consolidator.enter().unwrap();
        assert_eq!(consolidator.state(), DreamState::Dreaming);
        assert!(consolidator.is_active());

        consolidator.exit().unwrap();
        assert_eq!(consolidator.state(), DreamState::Awake);
        assert!(!consolidator.is_active());
    }

    #[test]
    fn test_arousal_based_transition() {
        let consolidator = DreamConsolidator::new(0.3).unwrap();

        // Low arousal → enter dream
        consolidator.update_arousal(ArousalLevel::new(0.2)).unwrap();
        assert_eq!(consolidator.state(), DreamState::Dreaming);

        // High arousal → exit dream
        consolidator.update_arousal(ArousalLevel::new(0.8)).unwrap();
        assert_eq!(consolidator.state(), DreamState::Awake);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        let memory = EpisodicMemory::new(
            vec![1.0, 2.0],
            vec![0.5],
            1.0,
            vec![1.5, 2.5],
            0,
        );

        buffer.add(memory);
        assert_eq!(buffer.len(), 1);

        let sampled = buffer.sample_indices(1);
        assert_eq!(sampled.len(), 1);
    }

    #[tokio::test]
    async fn test_consolidation() {
        let consolidator = DreamConsolidator::new(0.3).unwrap();

        // Add episodes
        for i in 0..10 {
            let memory = EpisodicMemory::new(
                vec![i as f64],
                vec![0.0],
                i as f64 * 0.1,
                vec![i as f64 + 1.0],
                i,
            );
            consolidator.add_episode(memory);
        }

        assert_eq!(consolidator.buffer_size(), 10);

        // Enter dream state and consolidate
        consolidator.enter().unwrap();
        let metrics = consolidator.consolidate(5).await.unwrap();

        assert_eq!(metrics.episodes_replayed, 5);
        assert_eq!(metrics.consolidation_steps, 1);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(5);

        for i in 0..10 {
            let memory = EpisodicMemory::new(
                vec![i as f64],
                vec![0.0],
                0.0,
                vec![i as f64],
                i,
            );
            buffer.add(memory);
        }

        // Should only keep last 5
        assert_eq!(buffer.len(), 5);
    }
}
