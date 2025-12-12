//! Compatibility types for existing codebase
//! 
//! This module provides the types that the risk-management and other crates
//! expect to find in mcp-orchestration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Re-export from main types file if it exists
pub use crate::types::*;

/// Unique identifier for agents in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Generate a new random agent ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create an agent ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of agents in the TENGRI trading swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Risk management agent
    Risk,
    /// Neural network forecasting agent
    Neural,
    /// Quantum uncertainty agent
    Quantum,
    /// Market data agent
    MarketData,
    /// Order execution agent
    OrderExecution,
    /// Portfolio management agent
    Portfolio,
    /// Technical analysis agent
    TechnicalAnalysis,
    /// Sentiment analysis agent
    SentimentAnalysis,
    /// News processing agent
    NewsProcessing,
    /// Backtesting agent
    Backtesting,
    /// Performance monitoring agent
    PerformanceMonitoring,
    /// Memory management agent
    MemoryManagement,
    /// Configuration management agent
    ConfigManagement,
    /// Logging agent
    Logging,
    /// Metrics collection agent
    MetricsCollection,
    /// Alert management agent
    AlertManagement,
    /// System health agent
    SystemHealth,
    /// Load balancing agent
    LoadBalancing,
    /// Fault tolerance agent
    FaultTolerance,
    /// MCP orchestration agent
    McpOrchestration,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Critical tasks that must be processed immediately
    Critical = 0,
    /// High priority tasks
    High = 1,
    /// Medium priority tasks
    Medium = 2,
    /// Low priority tasks
    Low = 3,
}

/// Unique identifier for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Generate a new random task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create a task ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentType {
    /// Get all agent types in the swarm
    pub fn all() -> Vec<Self> {
        vec![
            Self::Risk,
            Self::Neural,
            Self::Quantum,
            Self::MarketData,
            Self::OrderExecution,
            Self::Portfolio,
            Self::TechnicalAnalysis,
            Self::SentimentAnalysis,
            Self::NewsProcessing,
            Self::Backtesting,
            Self::PerformanceMonitoring,
            Self::MemoryManagement,
            Self::ConfigManagement,
            Self::Logging,
            Self::MetricsCollection,
            Self::AlertManagement,
            Self::SystemHealth,
            Self::LoadBalancing,
            Self::FaultTolerance,
            Self::McpOrchestration,
        ]
    }
    
    /// Get the priority level for this agent type
    pub fn priority(&self) -> TaskPriority {
        match self {
            Self::Risk => TaskPriority::Critical,
            Self::Neural => TaskPriority::High,
            Self::Quantum => TaskPriority::High,
            Self::MarketData => TaskPriority::High,
            Self::OrderExecution => TaskPriority::Critical,
            Self::Portfolio => TaskPriority::High,
            Self::TechnicalAnalysis => TaskPriority::Medium,
            Self::SentimentAnalysis => TaskPriority::Medium,
            Self::NewsProcessing => TaskPriority::Medium,
            Self::Backtesting => TaskPriority::Low,
            Self::PerformanceMonitoring => TaskPriority::Medium,
            Self::MemoryManagement => TaskPriority::High,
            Self::ConfigManagement => TaskPriority::Medium,
            Self::Logging => TaskPriority::Low,
            Self::MetricsCollection => TaskPriority::Low,
            Self::AlertManagement => TaskPriority::High,
            Self::SystemHealth => TaskPriority::High,
            Self::LoadBalancing => TaskPriority::Medium,
            Self::FaultTolerance => TaskPriority::High,
            Self::McpOrchestration => TaskPriority::Critical,
        }
    }
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Risk => write!(f, "Risk"),
            Self::Neural => write!(f, "Neural"),
            Self::Quantum => write!(f, "Quantum"),
            Self::MarketData => write!(f, "MarketData"),
            Self::OrderExecution => write!(f, "OrderExecution"),
            Self::Portfolio => write!(f, "Portfolio"),
            Self::TechnicalAnalysis => write!(f, "TechnicalAnalysis"),
            Self::SentimentAnalysis => write!(f, "SentimentAnalysis"),
            Self::NewsProcessing => write!(f, "NewsProcessing"),
            Self::Backtesting => write!(f, "Backtesting"),
            Self::PerformanceMonitoring => write!(f, "PerformanceMonitoring"),
            Self::MemoryManagement => write!(f, "MemoryManagement"),
            Self::ConfigManagement => write!(f, "ConfigManagement"),
            Self::Logging => write!(f, "Logging"),
            Self::MetricsCollection => write!(f, "MetricsCollection"),
            Self::AlertManagement => write!(f, "AlertManagement"),
            Self::SystemHealth => write!(f, "SystemHealth"),
            Self::LoadBalancing => write!(f, "LoadBalancing"),
            Self::FaultTolerance => write!(f, "FaultTolerance"),
            Self::McpOrchestration => write!(f, "McpOrchestration"),
        }
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Medium => write!(f, "Medium"),
            Self::Low => write!(f, "Low"),
        }
    }
}