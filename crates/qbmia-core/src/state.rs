//! State management and persistence

use crate::error::{QBMIAError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// State manager for agent persistence
#[derive(Debug, Clone)]
pub struct StateManager {
    agent_id: String,
    checkpoint_dir: String,
}

/// Agent state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub agent_id: String,
    pub timestamp: String,
    pub config: serde_json::Value,
    pub quantum_states: String,
    pub memory_state: String,
    pub component_states: HashMap<String, serde_json::Value>,
    pub performance_metrics: serde_json::Value,
    pub last_decision: Option<serde_json::Value>,
}

impl StateManager {
    /// Create a new state manager
    pub fn new(agent_id: String, checkpoint_dir: String) -> Self {
        Self {
            agent_id,
            checkpoint_dir,
        }
    }
    
    /// Save agent state to checkpoint
    pub async fn save_checkpoint(&self, state: AgentState, filepath: Option<&str>) -> Result<String> {
        // Ensure checkpoint directory exists
        fs::create_dir_all(&self.checkpoint_dir).await?;
        
        let filename = if let Some(path) = filepath {
            path.to_string()
        } else {
            format!("{}/checkpoint_{}.json", self.checkpoint_dir, chrono::Utc::now().format("%Y%m%d_%H%M%S"))
        };
        
        let json_data = serde_json::to_string_pretty(&state)?;
        fs::write(&filename, json_data).await?;
        
        Ok(filename)
    }
    
    /// Load agent state from checkpoint
    pub async fn load_checkpoint(&self, filepath: &str) -> Result<AgentState> {
        let content = fs::read_to_string(filepath).await?;
        let state: AgentState = serde_json::from_str(&content)?;
        Ok(state)
    }
    
    /// Get information about the last checkpoint
    pub fn get_last_checkpoint_info(&self) -> Option<String> {
        // Would implement checkpoint discovery logic
        None
    }
}