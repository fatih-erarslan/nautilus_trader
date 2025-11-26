//! Core types for the HiveMind system

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for an agent
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of agent in the hive
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    /// Research and analysis agent
    Researcher,

    /// Code implementation agent
    Coder,

    /// Testing and validation agent
    Tester,

    /// Architecture and design agent
    Architect,

    /// Code review agent
    Reviewer,

    /// Performance optimization agent
    Optimizer,

    /// Documentation agent
    Documenter,

    /// Coordination agent
    Coordinator,

    /// Custom agent type
    Custom(String),
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentType::Researcher => write!(f, "Researcher"),
            AgentType::Coder => write!(f, "Coder"),
            AgentType::Tester => write!(f, "Tester"),
            AgentType::Architect => write!(f, "Architect"),
            AgentType::Reviewer => write!(f, "Reviewer"),
            AgentType::Optimizer => write!(f, "Optimizer"),
            AgentType::Documenter => write!(f, "Documenter"),
            AgentType::Coordinator => write!(f, "Coordinator"),
            AgentType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// A task to be executed by the hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub description: String,
    pub priority: TaskPriority,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Task {
    pub fn new(description: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description,
            priority: TaskPriority::Medium,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Priority level for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub output: String,
    pub agent_id: AgentId,
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

impl TaskResult {
    pub fn success(task_id: String, agent_id: AgentId, output: String) -> Self {
        Self {
            task_id,
            success: true,
            output,
            agent_id,
            completed_at: chrono::Utc::now(),
        }
    }

    pub fn failure(task_id: String, agent_id: AgentId, error: String) -> Self {
        Self {
            task_id,
            success: false,
            output: error,
            agent_id,
            completed_at: chrono::Utc::now(),
        }
    }
}

/// Agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub can_code: bool,
    pub can_test: bool,
    pub can_review: bool,
    pub can_research: bool,
    pub can_optimize: bool,
    pub can_document: bool,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            can_code: false,
            can_test: false,
            can_review: false,
            can_research: false,
            can_optimize: false,
            can_document: false,
        }
    }
}

impl AgentCapabilities {
    pub fn from_agent_type(agent_type: &AgentType) -> Self {
        let mut caps = Self::default();

        match agent_type {
            AgentType::Researcher => caps.can_research = true,
            AgentType::Coder => caps.can_code = true,
            AgentType::Tester => caps.can_test = true,
            AgentType::Reviewer => caps.can_review = true,
            AgentType::Optimizer => caps.can_optimize = true,
            AgentType::Documenter => caps.can_document = true,
            AgentType::Architect => {
                caps.can_code = true;
                caps.can_review = true;
                caps.can_research = true;
            }
            AgentType::Coordinator => {
                // Coordinator can do everything at a basic level
                caps.can_code = true;
                caps.can_test = true;
                caps.can_review = true;
                caps.can_research = true;
                caps.can_optimize = true;
                caps.can_document = true;
            }
            AgentType::Custom(_) => {
                // Custom agents get basic capabilities
                caps.can_research = true;
            }
        }

        caps
    }
}
