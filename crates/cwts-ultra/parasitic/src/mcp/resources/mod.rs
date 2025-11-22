//! MCP Resources for Parasitic Pairlist System
//! Provides resource definitions and management for MCP tools

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource types available in the parasitic pairlist system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// Market data for trading pairs
    PairData,
    /// Organism performance metrics
    OrganismMetrics,
    /// System configuration
    SystemConfig,
    /// Performance statistics
    PerformanceStats,
}

/// MCP Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
    pub resource_type: ResourceType,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Resource manager for the MCP system
pub struct ResourceManager {
    resources: HashMap<String, Resource>,
}

impl ResourceManager {
    pub fn new() -> Self {
        let mut manager = Self {
            resources: HashMap::new(),
        };
        
        manager.register_default_resources();
        manager
    }
    
    /// Register default resources for the parasitic pairlist system
    fn register_default_resources(&mut self) {
        // Pair data resource
        self.register_resource(Resource {
            uri: "parasitic://pairs".to_string(),
            name: "Trading Pairs".to_string(),
            description: "Current trading pair data and market information".to_string(),
            mime_type: "application/json".to_string(),
            resource_type: ResourceType::PairData,
            metadata: HashMap::new(),
        });
        
        // Organism metrics resource
        self.register_resource(Resource {
            uri: "parasitic://organisms/metrics".to_string(),
            name: "Organism Metrics".to_string(),
            description: "Performance metrics for all active organisms".to_string(),
            mime_type: "application/json".to_string(),
            resource_type: ResourceType::OrganismMetrics,
            metadata: HashMap::new(),
        });
        
        // System configuration resource
        self.register_resource(Resource {
            uri: "parasitic://system/config".to_string(),
            name: "System Configuration".to_string(),
            description: "Current system configuration and settings".to_string(),
            mime_type: "application/json".to_string(),
            resource_type: ResourceType::SystemConfig,
            metadata: HashMap::new(),
        });
        
        // Performance statistics resource
        self.register_resource(Resource {
            uri: "parasitic://performance/stats".to_string(),
            name: "Performance Statistics".to_string(),
            description: "Real-time performance statistics for all operations".to_string(),
            mime_type: "application/json".to_string(),
            resource_type: ResourceType::PerformanceStats,
            metadata: HashMap::new(),
        });
    }
    
    /// Register a new resource
    pub fn register_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.uri.clone(), resource);
    }
    
    /// Get resource by URI
    pub fn get_resource(&self, uri: &str) -> Option<&Resource> {
        self.resources.get(uri)
    }
    
    /// List all available resources
    pub fn list_resources(&self) -> Vec<&Resource> {
        self.resources.values().collect()
    }
    
    /// Get resources by type
    pub fn get_resources_by_type(&self, resource_type: &ResourceType) -> Vec<&Resource> {
        self.resources.values()
            .filter(|resource| std::mem::discriminant(&resource.resource_type) == std::mem::discriminant(resource_type))
            .collect()
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}