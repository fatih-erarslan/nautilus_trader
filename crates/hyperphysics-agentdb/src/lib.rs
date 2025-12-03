//! # HyperPhysics AgentDB
//!
//! Native Rust integration of AgentDB for trading memory and causal learning.
//! Built on ruvector-core's AgenticDB with trading-specific extensions.
//!
//! ## Features
//!
//! - **Trading Episodes**: Store and retrieve trading experiences with outcomes
//! - **Strategy Skills**: Semantic search over consolidated trading patterns
//! - **Causal Learning**: Learn cause-effect relationships between market events and P&L
//! - **RL Sessions**: Store reinforcement learning experiences for trading agents
//!
//! ## Enhanced Features (v0.2.0)
//!
//! - **GNN-Enhanced Search**: Graph neural networks for self-improving pattern matching
//! - **Hyperbolic Attention**: Poincaré disk geometry for hierarchical market structures
//! - **MoE Strategy Routing**: Mixture of Experts for intelligent strategy selection
//! - **pBit Regime Detection**: Probabilistic bit dynamics for market regime classification
//! - **Hopfield Pattern Completion**: Complete partial trading patterns from memory
//! - **Tiny Dancer Routing**: FastGRNN neural routing with circuit breakers
//!
//! ## Usage
//!
//! ```ignore
//! use hyperphysics_agentdb::{TradingAgentDB, TradingEpisode, TradeAction, MarketContext};
//!
//! // Create database
//! let config = AgentDBConfig {
//!     db_path: "./trading_memory".into(),
//!     dimensions: 768,
//!     ..Default::default()
//! };
//! let db = TradingAgentDB::new(config)?;
//!
//! // Store episode (symbol-agnostic)
//! let episode = TradingEpisode::new(
//!     "ETH-PERP",           // Any trading pair
//!     TradeAction::Long,
//!     MarketContext::default(),
//! );
//! db.store_episode(episode)?;
//!
//! // Recall similar trades across all pairs
//! let similar = db.recall_similar("momentum breakout high volume", 10)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `gnn` - GNN-enhanced search with EWC forgetting mitigation (default)
//! - `attention` - Hyperbolic, MoE, and graph attention mechanisms (default)
//! - `routing` - Tiny Dancer FastGRNN routing (default)
//! - `pbit` - HyperPhysics pBit dynamics integration
//! - `full` - All features enabled

pub mod trading;
pub mod alpha;
pub mod embeddings;
pub mod error;
pub mod storage;
pub mod vector;

// Enhanced modules (feature-gated)
#[cfg(feature = "gnn")]
pub mod gnn;

#[cfg(feature = "attention")]
pub mod attention;

#[cfg(feature = "pbit")]
pub mod pbit_integration;

#[cfg(feature = "routing")]
pub mod routing;

pub use trading::*;
pub use alpha::*;
pub use error::*;
pub use storage::{TradingStorage, StrategyPerformance};

// Re-export enhanced modules
#[cfg(feature = "gnn")]
pub use gnn::{GnnTradingMemory, GnnConfig, GnnTrainingStats, CausalGraphGnn};

#[cfg(feature = "attention")]
pub use attention::{
    TradingPatternAttention, AttentionSystemConfig, AttentionStats,
    TradingAttentionPipeline, ExpertRoutingResult,
};

#[cfg(feature = "pbit")]
pub use pbit_integration::{
    PBitRegimeDetector, RegimeDetection, RegimeDetectorStats,
    TradingHopfieldNetwork, PBitPortfolioOptimizer,
};

#[cfg(feature = "routing")]
pub use routing::{
    TradingStrategyRouter, TradingRouterConfig, StrategyCandidate,
    RouterStats, RoutingResult,
};

use ruvector_core::agenticdb::AgenticDB;
use ruvector_core::types::DbOptions;
use std::path::Path;

/// Main trading AgentDB interface
pub struct TradingAgentDB {
    inner: AgenticDB,
    config: AgentDBConfig,
}

/// Configuration for TradingAgentDB
#[derive(Debug, Clone)]
pub struct AgentDBConfig {
    /// Database storage path
    pub db_path: String,
    /// Vector embedding dimensions
    pub dimensions: usize,
    /// Auto-consolidate successful patterns into skills
    pub auto_consolidate: bool,
    /// Minimum episodes before consolidation
    pub consolidation_threshold: usize,
    /// Enable causal learning
    pub causal_learning_enabled: bool,
    /// Minimum confidence for causal edges
    pub causal_confidence_threshold: f64,
}

impl Default for AgentDBConfig {
    fn default() -> Self {
        Self {
            db_path: "./hyperphysics_agentdb".into(),
            dimensions: 768,
            auto_consolidate: true,
            consolidation_threshold: 10,
            causal_learning_enabled: true,
            causal_confidence_threshold: 0.7,
        }
    }
}

impl TradingAgentDB {
    /// Create new TradingAgentDB with configuration
    pub fn new(config: AgentDBConfig) -> Result<Self> {
        let mut db_options = DbOptions::default();
        db_options.storage_path = config.db_path.clone();
        db_options.dimensions = config.dimensions;

        let inner = AgenticDB::new(db_options)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))?;

        Ok(Self { inner, config })
    }

    /// Open database at path with default config
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = AgentDBConfig {
            db_path: path.as_ref().to_string_lossy().to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Open with custom dimensions
    pub fn open_with_dimensions<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self> {
        let config = AgentDBConfig {
            db_path: path.as_ref().to_string_lossy().to_string(),
            dimensions,
            ..Default::default()
        };
        Self::new(config)
    }

    // ============ Episode API ============

    /// Store a trading episode
    pub fn store_episode(&self, episode: TradingEpisode) -> Result<String> {
        let task = episode.task_description();
        let actions = episode.actions_as_strings();
        let observations = episode.observations_as_strings();
        let critique = episode.critique.clone();

        self.inner
            .store_episode(task, actions, observations, critique)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Recall similar episodes by semantic query
    pub fn recall_similar(&self, query: &str, k: usize) -> Result<Vec<storage::ReflexionEpisode>> {
        self.inner
            .retrieve_similar_episodes(query, k)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Recall with utility scoring (similarity + causal relevance)
    pub fn recall_with_utility(
        &self,
        query: &str,
        k: usize,
        alpha: f64,  // similarity weight
        beta: f64,   // causal uplift weight
        gamma: f64,  // latency penalty weight
    ) -> Result<Vec<storage::UtilitySearchResult>> {
        self.inner
            .query_with_utility(query, k, alpha, beta, gamma)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    // ============ Skill API ============

    /// Create a trading skill
    pub fn create_skill(&self, skill: StrategySkill) -> Result<String> {
        self.inner
            .create_skill(skill.name, skill.description, skill.parameters, skill.example_trades)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Search for applicable skills
    pub fn search_skills(&self, context: &str, k: usize) -> Result<Vec<storage::Skill>> {
        self.inner
            .search_skills(context, k)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Auto-consolidate action sequences into skills
    pub fn consolidate_patterns(
        &self,
        action_sequences: Vec<Vec<String>>,
    ) -> Result<Vec<String>> {
        self.inner
            .auto_consolidate(action_sequences, self.config.consolidation_threshold)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    // ============ Causal Learning API ============

    /// Add a causal relationship
    pub fn add_causal_edge(
        &self,
        causes: Vec<String>,
        effects: Vec<String>,
        confidence: f64,
        context: &str,
    ) -> Result<String> {
        if confidence < self.config.causal_confidence_threshold {
            return Err(AgentDBError::InsufficientConfidence {
                required: self.config.causal_confidence_threshold,
                actual: confidence,
            });
        }

        self.inner
            .add_causal_edge(causes, effects, confidence, context.to_string())
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Query causal factors for an outcome
    pub fn query_causal_factors(
        &self,
        outcome: &str,
        k: usize,
    ) -> Result<Vec<storage::UtilitySearchResult>> {
        self.inner
            .query_with_utility(outcome, k, 0.4, 0.5, 0.1)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    // ============ RL Session API ============

    /// Start a learning session
    pub fn start_session(
        &self,
        algorithm: &str,
        state_dim: usize,
        action_dim: usize,
    ) -> Result<String> {
        self.inner
            .start_session(algorithm.to_string(), state_dim, action_dim)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Add experience to session
    pub fn add_experience(
        &self,
        session_id: &str,
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f64,
        next_state: Vec<f32>,
        done: bool,
    ) -> Result<()> {
        self.inner
            .add_experience(session_id, state, action, reward, next_state, done)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Predict action with confidence
    pub fn predict_action(
        &self,
        session_id: &str,
        state: Vec<f32>,
    ) -> Result<storage::Prediction> {
        self.inner
            .predict_with_confidence(session_id, state)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Result<Option<storage::LearningSession>> {
        self.inner
            .get_session(session_id)
            .map_err(|e| AgentDBError::DatabaseError(e.to_string()))
    }

    // ============ Utility ============

    /// Get configuration
    pub fn config(&self) -> &AgentDBConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_db() -> TradingAgentDB {
        let dir = tempdir().unwrap();
        let config = AgentDBConfig {
            db_path: dir.path().join("test").to_string_lossy().to_string(),
            dimensions: 128,
            ..Default::default()
        };
        TradingAgentDB::new(config).unwrap()
    }

    #[test]
    fn test_create_db() {
        let _db = create_test_db();
    }

    #[test]
    fn test_store_and_recall_episode() {
        let db = create_test_db();

        // Create episode with any symbol (pair-agnostic)
        let episode = TradingEpisode {
            symbol: "TEST-PAIR".into(),
            action: TradeAction::Long,
            entry_price: 100.0,
            exit_price: Some(105.0),
            position_size: 0.1,
            pnl: 0.05,
            pnl_absolute: 5.0,
            duration_secs: 3600,
            entry_context: MarketContext::default(),
            exit_context: None,
            signals: vec!["signal_a".into(), "signal_b".into()],
            strategy: "test_strategy".into(),
            critique: "Test critique for momentum pattern".into(),
            timestamp: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
        };

        let id = db.store_episode(episode).unwrap();
        assert!(!id.is_empty());

        // Recall by semantic query
        let results = db.recall_similar("momentum pattern", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_causal_edge() {
        let db = create_test_db();

        let edge_id = db.add_causal_edge(
            vec!["high_volume".into(), "breakout".into()],
            vec!["positive_return".into()],
            0.85,
            "Volume-confirmed breakout pattern",
        ).unwrap();

        assert!(!edge_id.is_empty());
    }

    #[test]
    fn test_causal_confidence_threshold() {
        let db = create_test_db();

        // Should fail - below threshold
        let result = db.add_causal_edge(
            vec!["weak_signal".into()],
            vec!["random_outcome".into()],
            0.3,
            "Low confidence pattern",
        );

        assert!(matches!(result, Err(AgentDBError::InsufficientConfidence { .. })));
    }

    #[test]
    fn test_skill_creation() {
        let db = create_test_db();

        let skill = StrategySkill {
            name: "breakout_strategy".into(),
            description: "Enter on volume-confirmed breakout above resistance".into(),
            parameters: HashMap::new(),
            example_trades: vec!["Entry on breakout".into()],
            win_rate: 0.0,
            avg_pnl: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            trade_count: 0,
        };

        let skill_id = db.create_skill(skill).unwrap();
        assert!(!skill_id.is_empty());

        let results = db.search_skills("breakout volume", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_learning_session() {
        let db = create_test_db();

        let session_id = db.start_session("PPO", 10, 3).unwrap();
        assert!(!session_id.is_empty());

        db.add_experience(
            &session_id,
            vec![0.1; 10],  // state
            vec![0.5, 0.3, 0.2],  // action
            0.5,  // reward
            vec![0.2; 10],  // next_state
            false,  // done
        ).unwrap();

        let session = db.get_session(&session_id).unwrap();
        assert!(session.is_some());
        assert_eq!(session.unwrap().experiences.len(), 1);
    }
}
