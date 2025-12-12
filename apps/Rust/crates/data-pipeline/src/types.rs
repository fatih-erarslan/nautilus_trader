//! Common types used throughout the data pipeline

use serde::{Deserialize, Serialize};

/// Data item structure for pipeline input
#[derive(Debug, Clone)]
pub struct DataItem {
    pub symbol: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub text: Option<String>,
    pub raw_data: Vec<u8>,
}

impl Default for DataItem {
    fn default() -> Self {
        Self {
            symbol: "UNKNOWN".to_string(),
            timestamp: chrono::Utc::now(),
            price: 0.0,
            volume: 0.0,
            bid: None,
            ask: None,
            text: None,
            raw_data: Vec::new(),
        }
    }
}

// ============ OMEGA-2: MASS TYPE GENERATION ============

/// Agent discovery service
#[derive(Debug, Clone)]
pub struct AgentDiscoveryService {
    pub service_id: String,
    pub discovery_protocol: String,
    pub registered_agents: Vec<String>,
}

/// Adaptation requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRequirement {
    pub requirement_type: String,
    pub priority: u32,
    pub target_metric: String,
    pub threshold: f64,
}

/// Communication layer
#[derive(Debug, Clone)]
pub struct CommunicationLayer {
    pub protocol: String,
    pub endpoints: Vec<String>,
    pub encryption_enabled: bool,
}

/// Agent interface
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: String,
    pub agent_type: String,
    pub status: String,
    pub capabilities: Vec<String>,
}