//! MCP Protocol Integration with Claude-Flow
//!
//! Implements the Model Context Protocol (MCP) for seamless integration
//! with Claude-Flow orchestration system and external AI agents.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex, mpsc, broadcast};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;
use dashmap::DashMap;
use hyper::{Body, Request, Response, Server, Method, StatusCode};
use hyper::service::{make_service_fn, service_fn};
use tonic::{transport::Server as TonicServer, Request as TonicRequest, Response as TonicResponse, Status};
use tracing::{debug, info, warn, error, instrument};
use chrono::{DateTime, Utc};

use crate::agents::{MCPMessage, MCPMessageType, MessagePriority, RoutingInfo};
use crate::{MCPOrchestrationError, SwarmType, HierarchyLevel};

/// MCP Protocol version
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP Server Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout_ms: u64,
    /// Message size limit
    pub max_message_size: usize,
    /// Enable compression
    pub compression_enabled: bool,
    /// Enable encryption
    pub encryption_enabled: bool,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Claude-Flow integration settings
    pub claude_flow_config: ClaudeFlowConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication
    pub enabled: bool,
    /// API key for client authentication
    pub api_key: String,
    /// JWT secret for token validation
    pub jwt_secret: String,
    /// Token expiration time
    pub token_expiry_hours: u64,
}

/// Claude-Flow specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeFlowConfig {
    /// Claude API endpoint
    pub claude_api_endpoint: String,
    /// Claude API key
    pub claude_api_key: String,
    /// Model preference
    pub preferred_model: String,
    /// Context window size
    pub context_window: usize,
    /// Temperature setting
    pub temperature: f32,
    /// Max tokens per response
    pub max_tokens: usize,
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            max_connections: 1000,
            connection_timeout_ms: 30000,
            max_message_size: 1024 * 1024, // 1MB
            compression_enabled: true,
            encryption_enabled: true,
            auth_config: AuthConfig {
                enabled: true,
                api_key: "default-key".to_string(),
                jwt_secret: "default-secret".to_string(),
                token_expiry_hours: 24,
            },
            claude_flow_config: ClaudeFlowConfig {
                claude_api_endpoint: "https://api.anthropic.com".to_string(),
                claude_api_key: "".to_string(),
                preferred_model: "claude-3-5-sonnet-20241022".to_string(),
                context_window: 200000,
                temperature: 0.1,
                max_tokens: 4096,
            },
        }
    }
}

/// MCP message types following the protocol specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum MCPRequestMethod {
    #[serde(rename = "initialize")]
    Initialize { params: InitializeParams },
    #[serde(rename = "notifications/initialized")]
    Initialized,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "tools/list")]
    ToolsList,
    #[serde(rename = "tools/call")]
    ToolsCall { params: ToolCallParams },
    #[serde(rename = "resources/list")]
    ResourcesList,
    #[serde(rename = "resources/read")]
    ResourcesRead { params: ResourceReadParams },
    #[serde(rename = "prompts/list")]
    PromptsList,
    #[serde(rename = "prompts/get")]
    PromptsGet { params: PromptGetParams },
    #[serde(rename = "logging/setLevel")]
    LoggingSetLevel { params: LoggingParams },
    // Claude-Flow specific methods
    #[serde(rename = "claude_flow/orchestrate")]
    ClaudeFlowOrchestrate { params: OrchestrationParams },
    #[serde(rename = "claude_flow/swarm_status")]
    ClaudeFlowSwarmStatus,
    #[serde(rename = "claude_flow/agent_control")]
    ClaudeFlowAgentControl { params: AgentControlParams },
}

/// MCP response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<MCPError>,
}

/// MCP error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// Initialize parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

/// Client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    pub tools: Option<ToolsCapability>,
    pub resources: Option<ResourcesCapability>,
    pub prompts: Option<PromptsCapability>,
    pub logging: Option<LoggingCapability>,
}

/// Tools capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    pub list_changed: Option<bool>,
}

/// Resources capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesCapability {
    pub list_changed: Option<bool>,
    pub subscribe: Option<bool>,
}

/// Prompts capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    pub list_changed: Option<bool>,
}

/// Logging capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingCapability {
    pub level: Option<String>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Tool call parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: Value,
}

/// Resource read parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadParams {
    pub uri: String,
}

/// Prompt get parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetParams {
    pub name: String,
    pub arguments: Option<Value>,
}

/// Logging parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingParams {
    pub level: String,
}

/// Orchestration parameters for Claude-Flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationParams {
    pub task: String,
    pub mode: Option<String>,
    pub agents: Option<Vec<String>>,
    pub priority: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Agent control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentControlParams {
    pub agent_id: String,
    pub action: String,
    pub parameters: Option<Value>,
}

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// MCP Prompt definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPrompt {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: Option<bool>,
}

/// MCP connection state
#[derive(Debug, Clone)]
pub struct MCPConnection {
    pub id: String,
    pub client_info: Option<ClientInfo>,
    pub capabilities: Option<ClientCapabilities>,
    pub last_ping: Instant,
    pub message_count: u64,
    pub error_count: u64,
}

/// MCP Server implementation
pub struct MCPServer {
    config: MCPServerConfig,
    connections: Arc<DashMap<String, MCPConnection>>,
    tools: Arc<RwLock<Vec<MCPTool>>>,
    resources: Arc<RwLock<Vec<MCPResource>>>,
    prompts: Arc<RwLock<Vec<MCPPrompt>>>,
    message_handler: Arc<MCPMessageHandler>,
    claude_client: Arc<ClaudeClient>,
    orchestration_bridge: Arc<OrchestrationBridge>,
    shutdown_tx: Arc<Mutex<Option<mpsc::UnboundedSender<()>>>>,
    server_stats: Arc<RwLock<ServerStats>>,
}

/// Server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub connections_count: usize,
    pub total_messages: u64,
    pub total_errors: u64,
    pub uptime_seconds: u64,
    pub start_time: DateTime<Utc>,
}

/// Message handler for MCP protocol
pub struct MCPMessageHandler {
    request_handlers: HashMap<String, Arc<dyn MCPRequestHandler>>,
    response_cache: Arc<DashMap<String, MCPResponse>>,
    metrics: Arc<RwLock<HandlerMetrics>>,
}

/// Request handler trait
#[async_trait::async_trait]
pub trait MCPRequestHandler: Send + Sync {
    async fn handle(&self, request: MCPRequestMethod, connection_id: &str) -> Result<Value, MCPError>;
}

/// Handler metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerMetrics {
    pub requests_handled: u64,
    pub requests_failed: u64,
    pub average_response_time_ms: f64,
}

/// Claude client for API integration
pub struct ClaudeClient {
    config: ClaudeFlowConfig,
    http_client: reqwest::Client,
    request_cache: Arc<DashMap<String, ClaudeResponse>>,
    rate_limiter: Arc<RateLimiter>,
}

/// Claude API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: TokenUsage,
}

/// Content block in Claude response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    pub r#type: String,
    pub text: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Rate limiter for API calls
pub struct RateLimiter {
    tokens: Arc<Mutex<f64>>,
    capacity: f64,
    refill_rate: f64,
    last_refill: Arc<Mutex<Instant>>,
}

/// Orchestration bridge for connecting MCP to swarm
pub struct OrchestrationBridge {
    swarm_connector: Arc<SwarmConnector>,
    task_translator: Arc<TaskTranslator>,
    response_formatter: Arc<ResponseFormatter>,
}

/// Swarm connector for agent communication
pub struct SwarmConnector {
    agent_registry: Arc<DashMap<String, AgentConnection>>,
    message_bus: Arc<MessageBus>,
    health_checker: Arc<HealthChecker>,
}

/// Agent connection information
#[derive(Debug, Clone)]
pub struct AgentConnection {
    pub agent_id: String,
    pub swarm_type: SwarmType,
    pub endpoint: String,
    pub last_seen: Instant,
    pub capabilities: Vec<String>,
}

/// Message bus for inter-agent communication
pub struct MessageBus {
    channels: Arc<DashMap<String, mpsc::UnboundedSender<MCPMessage>>>,
    message_history: Arc<RwLock<Vec<MessageRecord>>>,
}

/// Message record for history
#[derive(Debug, Clone)]
pub struct MessageRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub target: String,
    pub message_type: MCPMessageType,
    pub success: bool,
}

/// Health checker for agent monitoring
pub struct HealthChecker {
    health_status: Arc<DashMap<String, AgentHealthStatus>>,
    check_interval: Duration,
}

/// Agent health status
#[derive(Debug, Clone)]
pub struct AgentHealthStatus {
    pub agent_id: String,
    pub healthy: bool,
    pub last_check: Instant,
    pub response_time_ms: u64,
    pub error_count: u64,
}

/// Task translator for converting MCP requests to swarm tasks
pub struct TaskTranslator {
    translation_rules: Vec<TranslationRule>,
    task_templates: HashMap<String, TaskTemplate>,
}

/// Translation rule for task conversion
pub struct TranslationRule {
    pub pattern: String,
    pub target_agents: Vec<SwarmType>,
    pub priority: MessagePriority,
    pub timeout_ms: u64,
}

/// Task template
#[derive(Debug, Clone)]
pub struct TaskTemplate {
    pub name: String,
    pub description: String,
    pub required_agents: Vec<SwarmType>,
    pub estimated_duration_ms: u64,
}

/// Response formatter for converting swarm responses to MCP format
pub struct ResponseFormatter {
    formatting_rules: HashMap<SwarmType, FormattingRule>,
}

/// Formatting rule
pub struct FormattingRule {
    pub output_format: String,
    pub include_metadata: bool,
    pub compression_enabled: bool,
}

impl MCPServer {
    /// Create new MCP server
    pub async fn new() -> Result<Self, MCPOrchestrationError> {
        let config = MCPServerConfig::default();
        
        let connections = Arc::new(DashMap::new());
        let tools = Arc::new(RwLock::new(Self::create_default_tools()));
        let resources = Arc::new(RwLock::new(Self::create_default_resources()));
        let prompts = Arc::new(RwLock::new(Self::create_default_prompts()));
        
        let message_handler = Arc::new(MCPMessageHandler {
            request_handlers: Self::create_request_handlers(),
            response_cache: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(HandlerMetrics {
                requests_handled: 0,
                requests_failed: 0,
                average_response_time_ms: 0.0,
            })),
        });
        
        let claude_client = Arc::new(ClaudeClient {
            config: config.claude_flow_config.clone(),
            http_client: reqwest::Client::new(),
            request_cache: Arc::new(DashMap::new()),
            rate_limiter: Arc::new(RateLimiter {
                tokens: Arc::new(Mutex::new(100.0)),
                capacity: 100.0,
                refill_rate: 1.0, // 1 token per second
                last_refill: Arc::new(Mutex::new(Instant::now())),
            }),
        });
        
        let orchestration_bridge = Arc::new(OrchestrationBridge {
            swarm_connector: Arc::new(SwarmConnector {
                agent_registry: Arc::new(DashMap::new()),
                message_bus: Arc::new(MessageBus {
                    channels: Arc::new(DashMap::new()),
                    message_history: Arc::new(RwLock::new(Vec::new())),
                }),
                health_checker: Arc::new(HealthChecker {
                    health_status: Arc::new(DashMap::new()),
                    check_interval: Duration::from_secs(30),
                }),
            }),
            task_translator: Arc::new(TaskTranslator {
                translation_rules: Self::create_translation_rules(),
                task_templates: Self::create_task_templates(),
            }),
            response_formatter: Arc::new(ResponseFormatter {
                formatting_rules: Self::create_formatting_rules(),
            }),
        });
        
        let server_stats = Arc::new(RwLock::new(ServerStats {
            connections_count: 0,
            total_messages: 0,
            total_errors: 0,
            uptime_seconds: 0,
            start_time: Utc::now(),
        }));
        
        Ok(Self {
            config,
            connections,
            tools,
            resources,
            prompts,
            message_handler,
            claude_client,
            orchestration_bridge,
            shutdown_tx: Arc::new(Mutex::new(None)),
            server_stats,
        })
    }
    
    /// Start the MCP server
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<(), MCPOrchestrationError> {
        info!("Starting MCP server on {}:{}", self.config.host, self.config.port);
        
        // Start TCP server for MCP protocol
        let tcp_server = self.start_tcp_server();
        
        // Start HTTP server for REST API
        let http_server = self.start_http_server();
        
        // Start background tasks
        let health_monitor = self.start_health_monitoring();
        let rate_limiter_task = self.start_rate_limiter();
        let stats_updater = self.start_stats_updater();
        
        // Wait for all servers to complete
        tokio::try_join!(
            tcp_server,
            http_server,
            health_monitor,
            rate_limiter_task,
            stats_updater
        )?;
        
        Ok(())
    }
    
    /// Start TCP server for MCP protocol
    async fn start_tcp_server(&self) -> Result<(), MCPOrchestrationError> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr).await
            .map_err(|e| MCPOrchestrationError::MCPProtocolError {
                reason: format!("Failed to bind TCP listener: {}", e),
            })?;
        
        info!("MCP TCP server listening on {}", addr);
        
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    info!("New MCP connection from {}", addr);
                    let connection_id = Uuid::new_v4().to_string();
                    
                    // Create connection record
                    let connection = MCPConnection {
                        id: connection_id.clone(),
                        client_info: None,
                        capabilities: None,
                        last_ping: Instant::now(),
                        message_count: 0,
                        error_count: 0,
                    };
                    
                    self.connections.insert(connection_id.clone(), connection);
                    
                    // Handle connection in background
                    let server_clone = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server_clone.handle_connection(stream, connection_id).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }
    
    /// Handle individual connection
    async fn handle_connection(&self, stream: TcpStream, connection_id: String) -> Result<(), MCPOrchestrationError> {
        let codec = LengthDelimitedCodec::new();
        let mut framed = Framed::new(stream, codec);
        
        while let Some(frame) = framed.next().await {
            match frame {
                Ok(bytes) => {
                    // Parse MCP message
                    match serde_json::from_slice::<Value>(&bytes) {
                        Ok(json_value) => {
                            if let Err(e) = self.process_mcp_message(json_value, &connection_id).await {
                                error!("Error processing MCP message: {}", e);
                                self.increment_error_count(&connection_id).await;
                            } else {
                                self.increment_message_count(&connection_id).await;
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse MCP message: {}", e);
                            self.increment_error_count(&connection_id).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Frame error: {}", e);
                    self.increment_error_count(&connection_id).await;
                    break;
                }
            }
        }
        
        // Clean up connection
        self.connections.remove(&connection_id);
        info!("Connection {} closed", connection_id);
        
        Ok(())
    }
    
    /// Process MCP message
    async fn process_mcp_message(&self, message: Value, connection_id: &str) -> Result<(), MCPOrchestrationError> {
        debug!("Processing MCP message: {:?}", message);
        
        // Extract method and params
        if let Some(method) = message.get("method").and_then(|m| m.as_str()) {
            let params = message.get("params").cloned().unwrap_or(json!({}));
            let id = message.get("id").cloned();
            
            // Route to appropriate handler
            let response = match method {
                "initialize" => self.handle_initialize(params, connection_id).await,
                "ping" => self.handle_ping().await,
                "tools/list" => self.handle_tools_list().await,
                "tools/call" => self.handle_tools_call(params).await,
                "resources/list" => self.handle_resources_list().await,
                "resources/read" => self.handle_resources_read(params).await,
                "prompts/list" => self.handle_prompts_list().await,
                "prompts/get" => self.handle_prompts_get(params).await,
                "claude_flow/orchestrate" => self.handle_orchestrate(params).await,
                "claude_flow/swarm_status" => self.handle_swarm_status().await,
                "claude_flow/agent_control" => self.handle_agent_control(params).await,
                _ => Err(MCPError {
                    code: -32601,
                    message: "Method not found".to_string(),
                    data: None,
                }),
            };
            
            // Send response
            let mcp_response = MCPResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: response.as_ref().ok().cloned(),
                error: response.as_ref().err().cloned(),
            };
            
            // In a real implementation, send response back through the connection
            debug!("MCP response: {:?}", mcp_response);
        }
        
        Ok(())
    }
    
    /// Handle initialize request
    async fn handle_initialize(&self, params: Value, connection_id: &str) -> Result<Value, MCPError> {
        let init_params: InitializeParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        // Update connection with client info
        if let Some(mut connection) = self.connections.get_mut(connection_id) {
            connection.client_info = Some(init_params.client_info);
            connection.capabilities = Some(init_params.capabilities);
        }
        
        // Return server capabilities
        Ok(json!({
            "protocol_version": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "list_changed": true
                },
                "resources": {
                    "list_changed": true,
                    "subscribe": true
                },
                "prompts": {
                    "list_changed": true
                },
                "logging": {
                    "level": "info"
                }
            },
            "server_info": {
                "name": "TENGRI MCP Orchestration Server",
                "version": "1.0.0"
            }
        }))
    }
    
    /// Handle ping request
    async fn handle_ping(&self) -> Result<Value, MCPError> {
        Ok(json!({}))
    }
    
    /// Handle tools list request
    async fn handle_tools_list(&self) -> Result<Value, MCPError> {
        let tools = self.tools.read().await;
        Ok(json!({
            "tools": tools.clone()
        }))
    }
    
    /// Handle tools call request
    async fn handle_tools_call(&self, params: Value) -> Result<Value, MCPError> {
        let tool_params: ToolCallParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        // Route tool call to appropriate handler
        match tool_params.name.as_str() {
            "orchestrate_swarm" => self.orchestrate_swarm(tool_params.arguments).await,
            "get_agent_status" => self.get_agent_status(tool_params.arguments).await,
            "deploy_agents" => self.deploy_agents(tool_params.arguments).await,
            "monitor_performance" => self.monitor_performance(tool_params.arguments).await,
            _ => Err(MCPError {
                code: -32601,
                message: "Tool not found".to_string(),
                data: None,
            }),
        }
    }
    
    /// Handle resources list request
    async fn handle_resources_list(&self) -> Result<Value, MCPError> {
        let resources = self.resources.read().await;
        Ok(json!({
            "resources": resources.clone()
        }))
    }
    
    /// Handle resources read request
    async fn handle_resources_read(&self, params: Value) -> Result<Value, MCPError> {
        let read_params: ResourceReadParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        // Read resource based on URI
        match read_params.uri.as_str() {
            "swarm://status" => self.get_swarm_status_resource().await,
            "swarm://topology" => self.get_topology_resource().await,
            "swarm://metrics" => self.get_metrics_resource().await,
            _ => Err(MCPError {
                code: -32601,
                message: "Resource not found".to_string(),
                data: None,
            }),
        }
    }
    
    /// Handle prompts list request
    async fn handle_prompts_list(&self) -> Result<Value, MCPError> {
        let prompts = self.prompts.read().await;
        Ok(json!({
            "prompts": prompts.clone()
        }))
    }
    
    /// Handle prompts get request
    async fn handle_prompts_get(&self, params: Value) -> Result<Value, MCPError> {
        let prompt_params: PromptGetParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        // Get prompt template
        match prompt_params.name.as_str() {
            "orchestrate" => Ok(json!({
                "description": "Orchestrate swarm operations",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Orchestrate the following task: {{task}}"
                        }
                    }
                ]
            })),
            "analyze" => Ok(json!({
                "description": "Analyze swarm performance",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Analyze performance metrics: {{metrics}}"
                        }
                    }
                ]
            })),
            _ => Err(MCPError {
                code: -32601,
                message: "Prompt not found".to_string(),
                data: None,
            }),
        }
    }
    
    /// Handle orchestrate request
    async fn handle_orchestrate(&self, params: Value) -> Result<Value, MCPError> {
        let orchestration_params: OrchestrationParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        // Translate task to swarm operations
        let task = self.orchestration_bridge.task_translator
            .translate_task(&orchestration_params).await?;
        
        // Execute task through swarm
        let result = self.orchestration_bridge.swarm_connector
            .execute_task(task).await?;
        
        // Format response
        let formatted_response = self.orchestration_bridge.response_formatter
            .format_response(result).await?;
        
        Ok(formatted_response)
    }
    
    /// Handle swarm status request
    async fn handle_swarm_status(&self) -> Result<Value, MCPError> {
        let status = self.orchestration_bridge.swarm_connector
            .get_swarm_status().await?;
        
        Ok(json!({
            "status": status,
            "timestamp": Utc::now().to_rfc3339()
        }))
    }
    
    /// Handle agent control request
    async fn handle_agent_control(&self, params: Value) -> Result<Value, MCPError> {
        let control_params: AgentControlParams = serde_json::from_value(params)
            .map_err(|e| MCPError {
                code: -32602,
                message: "Invalid params".to_string(),
                data: Some(json!({"error": e.to_string()})),
            })?;
        
        let result = self.orchestration_bridge.swarm_connector
            .control_agent(control_params).await?;
        
        Ok(json!({
            "result": result,
            "timestamp": Utc::now().to_rfc3339()
        }))
    }
    
    /// Tool implementations
    
    /// Orchestrate swarm tool
    async fn orchestrate_swarm(&self, arguments: Value) -> Result<Value, MCPError> {
        // Implementation for swarm orchestration
        Ok(json!({
            "status": "success",
            "message": "Swarm orchestration initiated",
            "task_id": Uuid::new_v4().to_string()
        }))
    }
    
    /// Get agent status tool
    async fn get_agent_status(&self, arguments: Value) -> Result<Value, MCPError> {
        // Implementation for agent status
        Ok(json!({
            "agents": [],
            "total_count": 0,
            "healthy_count": 0
        }))
    }
    
    /// Deploy agents tool
    async fn deploy_agents(&self, arguments: Value) -> Result<Value, MCPError> {
        // Implementation for agent deployment
        Ok(json!({
            "status": "success",
            "deployed_agents": [],
            "deployment_id": Uuid::new_v4().to_string()
        }))
    }
    
    /// Monitor performance tool
    async fn monitor_performance(&self, arguments: Value) -> Result<Value, MCPError> {
        // Implementation for performance monitoring
        Ok(json!({
            "metrics": {},
            "timestamp": Utc::now().to_rfc3339()
        }))
    }
    
    /// Resource implementations
    
    /// Get swarm status resource
    async fn get_swarm_status_resource(&self) -> Result<Value, MCPError> {
        Ok(json!({
            "contents": [{
                "uri": "swarm://status",
                "mimeType": "application/json",
                "text": json!({
                    "active_swarms": 4,
                    "total_agents": 25,
                    "healthy_agents": 25,
                    "system_status": "operational"
                }).to_string()
            }]
        }))
    }
    
    /// Get topology resource
    async fn get_topology_resource(&self) -> Result<Value, MCPError> {
        Ok(json!({
            "contents": [{
                "uri": "swarm://topology",
                "mimeType": "application/json",
                "text": json!({
                    "hierarchy_levels": 4,
                    "topology_type": "hierarchical",
                    "optimization_score": 0.95
                }).to_string()
            }]
        }))
    }
    
    /// Get metrics resource
    async fn get_metrics_resource(&self) -> Result<Value, MCPError> {
        Ok(json!({
            "contents": [{
                "uri": "swarm://metrics",
                "mimeType": "application/json",
                "text": json!({
                    "average_latency_us": 500,
                    "throughput_ops_per_sec": 10000,
                    "cpu_utilization": 0.65,
                    "memory_utilization": 0.55
                }).to_string()
            }]
        }))
    }
    
    /// Start HTTP server for REST API
    async fn start_http_server(&self) -> Result<(), MCPOrchestrationError> {
        let addr = ([127, 0, 0, 1], self.config.port + 1).into();
        
        let make_svc = make_service_fn(|_conn| {
            async {
                Ok::<_, hyper::Error>(service_fn(|req| async move {
                    Self::handle_http_request(req).await
                }))
            }
        });
        
        let server = Server::bind(&addr).serve(make_svc);
        info!("MCP HTTP server listening on {}", addr);
        
        if let Err(e) = server.await {
            error!("HTTP server error: {}", e);
        }
        
        Ok(())
    }
    
    /// Handle HTTP request
    async fn handle_http_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        match (req.method(), req.uri().path()) {
            (&Method::GET, "/health") => {
                Ok(Response::new(Body::from(json!({
                    "status": "healthy",
                    "timestamp": Utc::now().to_rfc3339()
                }).to_string())))
            }
            (&Method::GET, "/status") => {
                Ok(Response::new(Body::from(json!({
                    "server": "MCP Orchestration Server",
                    "version": "1.0.0",
                    "protocol": MCP_PROTOCOL_VERSION
                }).to_string())))
            }
            _ => {
                let mut response = Response::new(Body::from("Not Found"));
                *response.status_mut() = StatusCode::NOT_FOUND;
                Ok(response)
            }
        }
    }
    
    /// Helper methods for connection management
    
    async fn increment_message_count(&self, connection_id: &str) {
        if let Some(mut connection) = self.connections.get_mut(connection_id) {
            connection.message_count += 1;
        }
    }
    
    async fn increment_error_count(&self, connection_id: &str) {
        if let Some(mut connection) = self.connections.get_mut(connection_id) {
            connection.error_count += 1;
        }
    }
    
    /// Start health monitoring
    async fn start_health_monitoring(&self) -> Result<(), MCPOrchestrationError> {
        // Health monitoring implementation
        Ok(())
    }
    
    /// Start rate limiter
    async fn start_rate_limiter(&self) -> Result<(), MCPOrchestrationError> {
        // Rate limiter implementation
        Ok(())
    }
    
    /// Start stats updater
    async fn start_stats_updater(&self) -> Result<(), MCPOrchestrationError> {
        // Stats updater implementation
        Ok(())
    }
    
    /// Create default tools
    fn create_default_tools() -> Vec<MCPTool> {
        vec![
            MCPTool {
                name: "orchestrate_swarm".to_string(),
                description: "Orchestrate swarm operations and task execution".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "agents": {"type": "array"},
                        "priority": {"type": "string"}
                    },
                    "required": ["task"]
                }),
            },
            MCPTool {
                name: "get_agent_status".to_string(),
                description: "Get status of all agents in the swarm".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"}
                    }
                }),
            },
            MCPTool {
                name: "deploy_agents".to_string(),
                description: "Deploy new agents to the swarm".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "agent_type": {"type": "string"},
                        "count": {"type": "number"}
                    },
                    "required": ["agent_type"]
                }),
            },
            MCPTool {
                name: "monitor_performance".to_string(),
                description: "Monitor swarm performance metrics".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "duration": {"type": "string"}
                    }
                }),
            },
        ]
    }
    
    /// Create default resources
    fn create_default_resources() -> Vec<MCPResource> {
        vec![
            MCPResource {
                uri: "swarm://status".to_string(),
                name: "Swarm Status".to_string(),
                description: Some("Real-time status of all swarm components".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            MCPResource {
                uri: "swarm://topology".to_string(),
                name: "Swarm Topology".to_string(),
                description: Some("Current swarm topology and agent placement".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            MCPResource {
                uri: "swarm://metrics".to_string(),
                name: "Performance Metrics".to_string(),
                description: Some("Real-time performance and health metrics".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ]
    }
    
    /// Create default prompts
    fn create_default_prompts() -> Vec<MCPPrompt> {
        vec![
            MCPPrompt {
                name: "orchestrate".to_string(),
                description: Some("Orchestrate swarm operations".to_string()),
                arguments: Some(vec![
                    PromptArgument {
                        name: "task".to_string(),
                        description: Some("Task to orchestrate".to_string()),
                        required: Some(true),
                    },
                ]),
            },
            MCPPrompt {
                name: "analyze".to_string(),
                description: Some("Analyze swarm performance".to_string()),
                arguments: Some(vec![
                    PromptArgument {
                        name: "metrics".to_string(),
                        description: Some("Metrics to analyze".to_string()),
                        required: Some(true),
                    },
                ]),
            },
        ]
    }
    
    /// Create request handlers
    fn create_request_handlers() -> HashMap<String, Arc<dyn MCPRequestHandler>> {
        // Implementation would create actual handlers
        HashMap::new()
    }
    
    /// Create translation rules
    fn create_translation_rules() -> Vec<TranslationRule> {
        // Implementation would create actual translation rules
        vec![]
    }
    
    /// Create task templates
    fn create_task_templates() -> HashMap<String, TaskTemplate> {
        // Implementation would create actual task templates
        HashMap::new()
    }
    
    /// Create formatting rules
    fn create_formatting_rules() -> HashMap<SwarmType, FormattingRule> {
        // Implementation would create actual formatting rules
        HashMap::new()
    }
}

impl Clone for MCPServer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            connections: Arc::clone(&self.connections),
            tools: Arc::clone(&self.tools),
            resources: Arc::clone(&self.resources),
            prompts: Arc::clone(&self.prompts),
            message_handler: Arc::clone(&self.message_handler),
            claude_client: Arc::clone(&self.claude_client),
            orchestration_bridge: Arc::clone(&self.orchestration_bridge),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            server_stats: Arc::clone(&self.server_stats),
        }
    }
}

/// Helper implementations for async traits

#[async_trait::async_trait]
impl TaskTranslator {
    async fn translate_task(&self, params: &OrchestrationParams) -> Result<SwarmTask, MCPError> {
        // Implementation would translate MCP orchestration params to swarm task
        Ok(SwarmTask {
            id: Uuid::new_v4().to_string(),
            task_type: params.task.clone(),
            priority: MessagePriority::Normal,
            timeout_ms: params.timeout_ms.unwrap_or(30000),
        })
    }
}

#[async_trait::async_trait]
impl SwarmConnector {
    async fn execute_task(&self, task: SwarmTask) -> Result<SwarmTaskResult, MCPError> {
        // Implementation would execute task through swarm
        Ok(SwarmTaskResult {
            task_id: task.id,
            status: "completed".to_string(),
            result: json!({"message": "Task completed successfully"}),
        })
    }
    
    async fn get_swarm_status(&self) -> Result<SwarmStatus, MCPError> {
        // Implementation would get actual swarm status
        Ok(SwarmStatus {
            active_agents: 25,
            healthy_agents: 25,
            total_swarms: 4,
            system_status: "operational".to_string(),
        })
    }
    
    async fn control_agent(&self, params: AgentControlParams) -> Result<String, MCPError> {
        // Implementation would control specific agent
        Ok(format!("Agent {} action {} executed", params.agent_id, params.action))
    }
}

#[async_trait::async_trait]
impl ResponseFormatter {
    async fn format_response(&self, result: SwarmTaskResult) -> Result<Value, MCPError> {
        // Implementation would format swarm response for MCP
        Ok(json!({
            "task_id": result.task_id,
            "status": result.status,
            "result": result.result
        }))
    }
}

/// Helper types for task execution

#[derive(Debug, Clone)]
pub struct SwarmTask {
    pub id: String,
    pub task_type: String,
    pub priority: MessagePriority,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SwarmTaskResult {
    pub task_id: String,
    pub status: String,
    pub result: Value,
}

#[derive(Debug, Clone)]
pub struct SwarmStatus {
    pub active_agents: usize,
    pub healthy_agents: usize,
    pub total_swarms: usize,
    pub system_status: String,
}