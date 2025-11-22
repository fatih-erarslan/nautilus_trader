//! MCP Tools registry for Parasitic Pairlist System
//! 
//! Registers all 10 MCP tool handlers with proper JSON schema validation
//! and performance monitoring. Follows blueprint specification exactly.

use crate::mcp::ParasiticPairlistManager;
use crate::mcp::handlers::*;
use crate::{Result, Error};
use serde_json::{json, Value};
use std::sync::Arc;
use std::collections::HashMap;

/// MCP Tool definition
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub handler: Box<dyn ToolHandler>,
}

/// Trait for MCP tool handlers
#[async_trait::async_trait]
pub trait ToolHandler: Send + Sync {
    /// Handle the tool request
    async fn handle(&self, input: Value) -> Result<Value>;
    
    /// Validate input against schema
    async fn validate_input(&self, input: &Value) -> Result<()>;
    
    /// Check if WebSocket subscriptions are supported
    fn supports_websocket(&self) -> bool;
    
    /// Subscribe to real-time updates via WebSocket
    async fn subscribe(&self, subscription_data: Value) -> Result<String>;
    
    /// Unsubscribe from updates
    async fn unsubscribe(&self, subscription_id: &str) -> Result<bool>;
}

/// Main tools registry for the parasitic pairlist system
pub struct ParasiticPairlistTools {
    manager: Arc<ParasiticPairlistManager>,
    tools: HashMap<String, Tool>,
}

impl ParasiticPairlistTools {
    /// Create new tools registry with manager
    pub fn new(manager: Arc<ParasiticPairlistManager>) -> Self {
        let mut tools = Self {
            manager: manager.clone(),
            tools: HashMap::new(),
        };
        
        tools.register_all_tools();
        tools
    }
    
    /// Register all MCP tools
    fn register_all_tools(&mut self) {
        // Tool 1: Scan for parasitic opportunities
        self.register_tool(Tool {
            name: "scan_parasitic_opportunities".to_string(),
            description: "Scan all pairs for parasitic trading opportunities".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "min_volume": {"type": "number", "minimum": 0},
                    "organisms": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "enum": ["komodo", "electric_eel", "octopus", "platypus", "anglerfish", "cuckoo", "cordyceps", "tardigrade"]
                    },
                    "risk_limit": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["min_volume"]
            }),
            handler: Box::new(ParasiticScanHandler::new(self.manager.clone())),
        });
        
        // Tool 2: Detect whale nests
        self.register_tool(Tool {
            name: "detect_whale_nests".to_string(),
            description: "Find pairs with whale activity suitable for cuckoo parasitism".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "min_whale_size": {"type": "number", "minimum": 0},
                    "vulnerability_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["min_whale_size"]
            }),
            handler: Box::new(WhaleNestDetectorHandler::new(self.manager.clone())),
        });
        
        // Tool 3: Identify zombie pairs
        self.register_tool(Tool {
            name: "identify_zombie_pairs".to_string(),
            description: "Find algorithmic trading patterns for cordyceps exploitation".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "min_predictability": {"type": "number", "minimum": 0, "maximum": 1},
                    "pattern_depth": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "required": ["min_predictability"]
            }),
            handler: Box::new(ZombiePairHandler::new(self.manager.clone())),
        });
        
        // Tool 4: Analyze mycelial correlations
        self.register_tool(Tool {
            name: "analyze_mycelial_network".to_string(),
            description: "Build correlation network between pairs".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "correlation_threshold": {"type": "number", "minimum": -1, "maximum": 1},
                    "network_depth": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["correlation_threshold"]
            }),
            handler: Box::new(MycelialNetworkHandler::new(self.manager.clone())),
        });
        
        // Tool 5: Deploy camouflage
        self.register_tool(Tool {
            name: "activate_octopus_camouflage".to_string(),
            description: "Dynamically adapt pair selection to avoid detection".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "threat_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"]
                    },
                    "camouflage_pattern": {
                        "type": "string", 
                        "enum": ["mimetic", "disruptive", "transparent", "adaptive"]
                    }
                },
                "required": ["threat_level"]
            }),
            handler: Box::new(CamouflageHandler::new(self.manager.clone())),
        });
        
        // Tool 6: Set anglerfish lure
        self.register_tool(Tool {
            name: "deploy_anglerfish_lure".to_string(),
            description: "Create artificial activity to attract traders".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "lure_pairs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "intensity": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["lure_pairs", "intensity"]
            }),
            handler: Box::new(AnglerfishLureHandler::new(self.manager.clone())),
        });
        
        // Tool 7: Track wounded pairs
        self.register_tool(Tool {
            name: "track_wounded_pairs".to_string(),
            description: "Persistently track high-volatility pairs".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "volatility_threshold": {"type": "number", "minimum": 0},
                    "tracking_duration": {"type": "integer", "minimum": 1000} // minimum 1 second
                },
                "required": ["volatility_threshold"]
            }),
            handler: Box::new(KomodoTrackerHandler::new(self.manager.clone())),
        });
        
        // Tool 8: Enter cryptobiosis
        self.register_tool(Tool {
            name: "enter_cryptobiosis".to_string(),
            description: "Enter dormant state during extreme conditions".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "trigger_conditions": {"type": "object"},
                    "revival_conditions": {"type": "object"}
                },
                "required": ["trigger_conditions", "revival_conditions"]
            }),
            handler: Box::new(TardigradeHandler::new(self.manager.clone())),
        });
        
        // Tool 9: Generate market shock
        self.register_tool(Tool {
            name: "electric_shock".to_string(),
            description: "Generate market disruption to reveal hidden liquidity".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "shock_pairs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "voltage": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["shock_pairs", "voltage"]
            }),
            handler: Box::new(ElectricEelHandler::new(self.manager.clone())),
        });
        
        // Tool 10: Detect subtle signals
        self.register_tool(Tool {
            name: "electroreception_scan".to_string(),
            description: "Detect subtle order flow signals".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "sensitivity": {"type": "number", "minimum": 0, "maximum": 1},
                    "frequency_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["sensitivity", "frequency_range"]
            }),
            handler: Box::new(PlatypusHandler::new(self.manager.clone())),
        });
    }
    
    /// Register a single tool
    fn register_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }
    
    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }
    
    /// List all available tools
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
    
    /// Execute tool with input validation and performance tracking
    pub async fn execute_tool(&self, tool_name: &str, input: Value) -> Result<Value> {
        let start_time = std::time::Instant::now();
        
        let tool = self.tools.get(tool_name)
            .ok_or_else(|| Error::Configuration(format!("Tool not found: {}", tool_name)))?;
        
        // Validate input schema
        tool.handler.validate_input(&input).await?;
        
        // Execute the tool
        let result = tool.handler.handle(input).await;
        
        // Record performance
        let duration_ns = start_time.elapsed().as_nanos() as u64;
        let success = result.is_ok();
        self.manager.record_performance(tool_name, duration_ns, success).await;
        
        result
    }
    
    /// Get manager reference for direct access
    pub fn get_manager(&self) -> Arc<ParasiticPairlistManager> {
        self.manager.clone()
    }
    
    /// Get performance statistics for all tools
    pub async fn get_performance_stats(&self) -> Result<HashMap<String, crate::mcp::OperationStats>> {
        self.manager.get_performance_stats().await
    }
    
    /// Create WebSocket subscription for a tool
    pub async fn subscribe_to_tool(&self, tool_name: &str, subscription_id: String, parameters: Value) -> Result<String> {
        let tool = self.tools.get(tool_name)
            .ok_or_else(|| Error::Configuration(format!("Tool not found: {}", tool_name)))?;
        
        if !tool.handler.supports_websocket() {
            return Err(Error::Configuration(format!("Tool {} does not support WebSocket subscriptions", tool_name)));
        }
        
        // Add subscription to manager
        self.manager.add_subscription(subscription_id.clone(), tool_name.to_string(), parameters.clone()).await?;
        
        // Create subscription with handler
        tool.handler.subscribe(parameters).await
    }
    
    /// Remove WebSocket subscription
    pub async fn unsubscribe_from_tool(&self, tool_name: &str, subscription_id: &str) -> Result<bool> {
        let tool = self.tools.get(tool_name)
            .ok_or_else(|| Error::Configuration(format!("Tool not found: {}", tool_name)))?;
        
        // Remove from manager
        let manager_result = self.manager.remove_subscription(subscription_id).await?;
        
        // Remove from handler
        let handler_result = tool.handler.unsubscribe(subscription_id).await?;
        
        Ok(manager_result && handler_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::ManagerConfig;
    
    #[tokio::test]
    async fn test_tools_registration() {
        let config = ManagerConfig::default();
        let manager = Arc::new(ParasiticPairlistManager::new(config).await.unwrap());
        let tools = ParasiticPairlistTools::new(manager);
        
        // Verify all 10 tools are registered
        let tool_names = tools.list_tools();
        assert_eq!(tool_names.len(), 10);
        
        // Verify specific tools exist
        assert!(tools.get_tool("scan_parasitic_opportunities").is_some());
        assert!(tools.get_tool("detect_whale_nests").is_some());
        assert!(tools.get_tool("identify_zombie_pairs").is_some());
        assert!(tools.get_tool("analyze_mycelial_network").is_some());
        assert!(tools.get_tool("activate_octopus_camouflage").is_some());
        assert!(tools.get_tool("deploy_anglerfish_lure").is_some());
        assert!(tools.get_tool("track_wounded_pairs").is_some());
        assert!(tools.get_tool("enter_cryptobiosis").is_some());
        assert!(tools.get_tool("electric_shock").is_some());
        assert!(tools.get_tool("electroreception_scan").is_some());
    }
    
    #[tokio::test]
    async fn test_tool_execution_with_validation() {
        let config = ManagerConfig::default();
        let manager = Arc::new(ParasiticPairlistManager::new(config).await.unwrap());
        let tools = ParasiticPairlistTools::new(manager);
        
        // Test valid input
        let valid_input = json!({
            "min_volume": 1000.0,
            "organisms": ["platypus", "octopus"],
            "risk_limit": 0.1
        });
        
        let result = tools.execute_tool("scan_parasitic_opportunities", valid_input).await;
        assert!(result.is_ok(), "Valid input should succeed: {:?}", result.err());
        
        // Test invalid input (missing required field)
        let invalid_input = json!({
            "organisms": ["platypus"]
            // missing min_volume
        });
        
        let result = tools.execute_tool("scan_parasitic_opportunities", invalid_input).await;
        assert!(result.is_err(), "Invalid input should fail");
    }
}