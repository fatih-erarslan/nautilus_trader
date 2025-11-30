//! Agent memory storage using AgenticDB's 5-table schema

use crate::error::{Result, VectorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "storage")]
use ruvector_core::AgenticDB;

/// Trading strategy episode for reflexion memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEpisode {
    /// Unique episode ID
    pub id: String,
    /// Strategy that was executed
    pub strategy_name: String,
    /// Market conditions at entry
    pub entry_conditions: HashMap<String, f64>,
    /// Actions taken (entry, exits, adjustments)
    pub actions: Vec<String>,
    /// Observed outcomes
    pub outcomes: Vec<String>,
    /// Self-critique / what could be improved
    pub critique: String,
    /// PnL result
    pub pnl: f64,
    /// Sharpe ratio for episode
    pub sharpe: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Embedding for similarity search
    pub embedding: Vec<f32>,
    /// Timestamp
    pub timestamp: i64,
}

/// Consolidated trading skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSkill {
    /// Unique skill ID
    pub id: String,
    /// Skill name
    pub name: String,
    /// Description of the skill
    pub description: String,
    /// Required market conditions
    pub preconditions: HashMap<String, String>,
    /// Entry rules
    pub entry_rules: Vec<String>,
    /// Exit rules
    pub exit_rules: Vec<String>,
    /// Risk parameters
    pub risk_params: HashMap<String, f64>,
    /// Historical success rate
    pub success_rate: f64,
    /// Average return when successful
    pub avg_return: f64,
    /// Number of times used
    pub usage_count: usize,
    /// Embedding for search
    pub embedding: Vec<f32>,
}

/// Causal relationship in market dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCausalEdge {
    /// Unique edge ID
    pub id: String,
    /// Cause events (hypergraph - multiple causes)
    pub causes: Vec<String>,
    /// Effect events (hypergraph - multiple effects)
    pub effects: Vec<String>,
    /// Confidence in relationship (0-1)
    pub confidence: f64,
    /// Typical lag time in milliseconds
    pub lag_ms: i64,
    /// Market context where this applies
    pub context: String,
    /// Number of observations supporting this
    pub observation_count: usize,
    /// Embedding for search
    pub embedding: Vec<f32>,
}

/// RL learning session for strategy optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingLearningSession {
    /// Session ID
    pub id: String,
    /// RL algorithm used
    pub algorithm: String,
    /// State features used
    pub state_features: Vec<String>,
    /// Action space definition
    pub action_space: Vec<String>,
    /// Reward function description
    pub reward_function: String,
    /// Experiences collected
    pub experiences: Vec<TradingExperience>,
    /// Final policy performance
    pub final_sharpe: f64,
    /// Training episodes completed
    pub episodes_completed: usize,
    /// Timestamp
    pub created_at: i64,
}

/// Single RL experience tuple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingExperience {
    /// State embedding
    pub state: Vec<f32>,
    /// Action taken
    pub action: Vec<f32>,
    /// Reward received
    pub reward: f64,
    /// Next state embedding
    pub next_state: Vec<f32>,
    /// Terminal state
    pub done: bool,
    /// Info dict
    pub info: HashMap<String, f64>,
}

/// Agent memory interface for trading agents
#[cfg(feature = "storage")]
pub struct AgentMemory {
    db: AgenticDB,
}

#[cfg(feature = "storage")]
impl AgentMemory {
    /// Create new agent memory store
    pub fn new(path: &str, dimensions: usize) -> Result<Self> {
        let mut options = ruvector_core::types::DbOptions::default();
        options.storage_path = path.to_string();
        options.dimensions = dimensions;

        let db = AgenticDB::new(options)?;
        Ok(Self { db })
    }

    /// Store a strategy episode for reflexion
    pub fn store_episode(&self, episode: &StrategyEpisode) -> Result<String> {
        let critique = format!(
            "Strategy: {} | PnL: {:.2} | Sharpe: {:.2} | {}",
            episode.strategy_name, episode.pnl, episode.sharpe, episode.critique
        );

        self.db
            .store_episode(
                episode.strategy_name.clone(),
                episode.actions.clone(),
                episode.outcomes.clone(),
                critique,
            )
            .map_err(VectorError::from)
    }

    /// Retrieve similar past episodes
    pub fn retrieve_similar_episodes(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<ruvector_core::agenticdb::ReflexionEpisode>> {
        self.db
            .retrieve_similar_episodes(query, k)
            .map_err(VectorError::from)
    }

    /// Create a trading skill
    pub fn create_skill(&self, skill: &TradingSkill) -> Result<String> {
        let description = format!(
            "{}: {} | Success: {:.1}% | Avg Return: {:.2}%",
            skill.name,
            skill.description,
            skill.success_rate * 100.0,
            skill.avg_return * 100.0
        );

        let mut params = HashMap::new();
        for (key, value) in &skill.risk_params {
            params.insert(key.clone(), value.to_string());
        }

        let examples = skill.entry_rules.clone();

        self.db
            .create_skill(skill.name.clone(), description, params, examples)
            .map_err(VectorError::from)
    }

    /// Search for relevant trading skills
    pub fn search_skills(
        &self,
        market_condition: &str,
        k: usize,
    ) -> Result<Vec<ruvector_core::agenticdb::Skill>> {
        self.db
            .search_skills(market_condition, k)
            .map_err(VectorError::from)
    }

    /// Add a causal relationship
    pub fn add_causal_edge(&self, edge: &MarketCausalEdge) -> Result<String> {
        self.db
            .add_causal_edge(
                edge.causes.clone(),
                edge.effects.clone(),
                edge.confidence,
                edge.context.clone(),
            )
            .map_err(VectorError::from)
    }

    /// Query causal relationships with utility function
    pub fn query_causal_with_utility(
        &self,
        query: &str,
        k: usize,
        similarity_weight: f64,
        causal_weight: f64,
        latency_weight: f64,
    ) -> Result<Vec<ruvector_core::agenticdb::UtilitySearchResult>> {
        self.db
            .query_with_utility(query, k, similarity_weight, causal_weight, latency_weight)
            .map_err(VectorError::from)
    }

    /// Start a learning session
    pub fn start_learning_session(
        &self,
        algorithm: &str,
        state_dim: usize,
        action_dim: usize,
    ) -> Result<String> {
        self.db
            .start_session(algorithm.to_string(), state_dim, action_dim)
            .map_err(VectorError::from)
    }

    /// Add an experience to learning session
    pub fn add_experience(
        &self,
        session_id: &str,
        experience: &TradingExperience,
    ) -> Result<()> {
        self.db
            .add_experience(
                session_id,
                experience.state.clone(),
                experience.action.clone(),
                experience.reward,
                experience.next_state.clone(),
                experience.done,
            )
            .map_err(VectorError::from)
    }

    /// Get prediction with confidence interval
    pub fn predict_action(
        &self,
        session_id: &str,
        state: Vec<f32>,
    ) -> Result<ruvector_core::agenticdb::Prediction> {
        self.db
            .predict_with_confidence(session_id, state)
            .map_err(VectorError::from)
    }
}

/// Placeholder when storage feature is not enabled
#[cfg(not(feature = "storage"))]
pub struct AgentMemory;

#[cfg(not(feature = "storage"))]
impl AgentMemory {
    /// Create placeholder (requires storage feature)
    pub fn new(_path: &str, _dimensions: usize) -> Result<Self> {
        Err(VectorError::Configuration(
            "AgentMemory requires 'storage' feature".to_string()
        ))
    }
}
