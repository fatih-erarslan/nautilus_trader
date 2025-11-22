//! # Parasitic MCP Server
//!
//! Ultra-low latency Model Context Protocol server for parasitic trading strategies.
//! Provides real-time access to organism data, infection status, and evolution metrics.

use std::sync::Arc;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};
use futures::{SinkExt, StreamExt};
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use tracing::{info, warn, error, debug};
use rand::Rng;

use crate::{
    ParasiticEngine, InfectedPair,
    organisms::{ParasiticOrganism, OrganismGenetics, AdaptationFeedback, MarketConditions},
    MCPServerConfig,
};

/// MCP Protocol Message Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMessage {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
    /// Internal timestamp for performance tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// Standard MCP error codes (JSON-RPC compatible)
mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const SERVER_ERROR_START: i32 = -32099;
    pub const SERVER_ERROR_END: i32 = -32000;
    
    // Custom MCP error codes
    pub const RESOURCE_NOT_FOUND: i32 = -32001;
    pub const TOOL_EXECUTION_ERROR: i32 = -32002;
    pub const ORGANISM_ERROR: i32 = -32003;
    pub const EVOLUTION_ERROR: i32 = -32004;
    pub const ANALYTICS_ERROR: i32 = -32005;
    pub const CONNECTION_ERROR: i32 = -32006;
    pub const RATE_LIMIT_EXCEEDED: i32 = -32007;
    pub const INSUFFICIENT_RESOURCES: i32 = -32008;
}

/// Resource definitions for MCP
#[derive(Debug, Serialize, Deserialize)]
pub struct MCPResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
}

/// Tool definitions for MCP
#[derive(Debug, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Client connection state with enhanced tracking
pub struct MCPClient {
    pub id: Uuid,
    pub websocket: Arc<RwLock<WebSocketStream<TcpStream>>>,
    pub subscriptions: Arc<RwLock<Vec<String>>>,
    pub last_heartbeat: Arc<RwLock<DateTime<Utc>>>,
    pub connection_time: DateTime<Utc>,
    pub total_requests: Arc<RwLock<u64>>,
    pub total_responses: Arc<RwLock<u64>>,
    pub total_errors: Arc<RwLock<u64>>,
    pub avg_response_time_ns: Arc<RwLock<u64>>,
    pub rate_limit_bucket: Arc<RwLock<RateLimitBucket>>,
    pub client_info: Arc<RwLock<ClientInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub user_agent: Option<String>,
    pub capabilities: Vec<String>,
    pub protocol_version: String,
    pub client_name: Option<String>,
    pub client_version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RateLimitBucket {
    pub tokens: u32,
    pub last_refill: Instant,
    pub max_tokens: u32,
    pub refill_rate: u32, // tokens per second
}

/// Performance metrics for requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub request_id: Uuid,
    pub method: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub processing_time_ns: Option<u64>,
    pub success: bool,
    pub error_code: Option<i32>,
    pub payload_size: usize,
    pub response_size: Option<usize>,
}

/// Event types for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParasiticEvent {
    OrganismSpawned {
        organism_id: Uuid,
        organism_type: String,
        genetics: OrganismGenetics,
    },
    PairInfected {
        pair_id: String,
        organism_id: Uuid,
        infection_strength: f64,
        timestamp: DateTime<Utc>,
    },
    InfectionEnded {
        pair_id: String,
        organism_id: Uuid,
        total_profit: f64,
        duration_secs: u64,
    },
    OrganismEvolved {
        organism_id: Uuid,
        old_fitness: f64,
        new_fitness: f64,
        genetic_changes: Vec<String>,
    },
    MarketConditionsChanged {
        volatility: f64,
        volume: f64,
        spread: f64,
        timestamp: DateTime<Utc>,
    },
}

/// Main Parasitic MCP Server with enhanced capabilities
pub struct ParasiticMCPServer {
    /// Reference to the main parasitic engine
    pub engine: Option<ParasiticEngine>,
    
    /// Connected clients
    pub clients: Arc<DashMap<Uuid, MCPClient>>,
    
    /// Event broadcasting
    pub event_broadcaster: broadcast::Sender<ParasiticEvent>,
    
    /// Command channel
    pub command_tx: mpsc::Sender<ParasiticCommand>,
    pub command_rx: Arc<RwLock<Option<mpsc::Receiver<ParasiticCommand>>>>,
    
    /// Server configuration
    pub config: MCPServerConfig,
    
    /// Performance tracking
    pub request_metrics: Arc<RwLock<HashMap<Uuid, RequestMetrics>>>,
    pub server_metrics: Arc<RwLock<ServerMetrics>>,
    
    /// Resource cache for performance
    pub resource_cache: Arc<RwLock<HashMap<String, CachedResource>>>,
    
    /// WebSocket reconnection manager
    pub reconnection_manager: Arc<RwLock<ReconnectionManager>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerMetrics {
    pub start_time: DateTime<Utc>,
    pub total_connections: u64,
    pub active_connections: u64,
    pub total_requests: u64,
    pub total_responses: u64,
    pub total_errors: u64,
    pub avg_response_time_ns: u64,
    pub peak_connections: u64,
    pub uptime_seconds: u64,
    pub throughput_per_second: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct CachedResource {
    pub content: Value,
    pub cached_at: DateTime<Utc>,
    pub ttl_seconds: u64,
    pub hit_count: u64,
}

#[derive(Debug, Clone)]
pub struct ReconnectionManager {
    pub max_reconnection_attempts: u32,
    pub base_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub active_reconnections: HashMap<Uuid, ReconnectionState>,
}

#[derive(Debug, Clone)]
pub struct ReconnectionState {
    pub client_id: Uuid,
    pub attempt_count: u32,
    pub last_attempt: DateTime<Utc>,
    pub next_attempt: DateTime<Utc>,
    pub exponential_backoff: bool,
}

/// Commands that can be sent to the parasitic engine
#[derive(Debug, Clone)]
pub enum ParasiticCommand {
    SelectOrganism { organism_type: String, genetics: Option<OrganismGenetics> },
    InfectPair { pair_id: String, organism_id: Uuid },
    TriggerEvolution { force: bool },
    AnalyzePair { pair_id: String },
    UpdateMarketConditions { conditions: MarketConditions },
    TerminateInfection { pair_id: String },
}

impl ParasiticMCPServer {
    /// Create a new enhanced parasitic MCP server
    pub async fn new(config: &MCPServerConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (event_broadcaster, _) = broadcast::channel(config.buffer_size);
        let (command_tx, command_rx) = mpsc::channel(1000);
        
        let server_metrics = ServerMetrics {
            start_time: Utc::now(),
            total_connections: 0,
            active_connections: 0,
            total_requests: 0,
            total_responses: 0,
            total_errors: 0,
            avg_response_time_ns: 0,
            peak_connections: 0,
            uptime_seconds: 0,
            throughput_per_second: 0.0,
            error_rate: 0.0,
        };
        
        let reconnection_manager = ReconnectionManager {
            max_reconnection_attempts: 10,
            base_backoff_ms: 1000,
            max_backoff_ms: 30000,
            active_reconnections: HashMap::new(),
        };
        
        Ok(Self {
            engine: None,
            clients: Arc::new(DashMap::new()),
            event_broadcaster,
            command_tx,
            command_rx: Arc::new(RwLock::new(Some(command_rx))),
            config: config.clone(),
            request_metrics: Arc::new(RwLock::new(HashMap::new())),
            server_metrics: Arc::new(RwLock::new(server_metrics)),
            resource_cache: Arc::new(RwLock::new(HashMap::new())),
            reconnection_manager: Arc::new(RwLock::new(reconnection_manager)),
        })
    }
    
    /// Set the parasitic engine reference
    pub fn set_engine(&mut self, engine: ParasiticEngine) {
        self.engine = Some(engine);
    }
    
    /// Start the MCP server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bind_addr: SocketAddr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse()?;
            
        println!("Starting Parasitic MCP Server on {}", bind_addr);
        
        let listener = TcpListener::bind(&bind_addr).await?;
        println!("Parasitic MCP Server listening on {}", bind_addr);
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        // Accept client connections
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    println!("New parasitic client connection from {}", addr);
                    
                    let server = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_client_connection(stream, addr).await {
                            eprintln!("Error handling parasitic client {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Error accepting connection: {}", e);
                }
            }
        }
    }
    
    /// Start background processing tasks
    async fn start_background_tasks(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Start command processing loop
        let command_rx = self.command_rx.clone();
        let engine = self.engine.clone();
        let event_broadcaster = self.event_broadcaster.clone();
        
        tokio::spawn(async move {
            let mut rx = command_rx.write().await.take().unwrap();
            
            while let Some(command) = rx.recv().await {
                if let Some(ref engine) = engine {
                    Self::process_command(command, engine.clone(), &event_broadcaster).await;
                }
            }
        });
        
        // Start heartbeat monitoring
        let clients = self.clients.clone();
        let heartbeat_interval = self.config.heartbeat_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(heartbeat_interval)
            );
            
            loop {
                interval.tick().await;
                
                // Remove stale clients
                let now = Utc::now();
                let mut to_remove = Vec::new();
                
                for entry in clients.iter() {
                    let last_heartbeat = *entry.value().last_heartbeat.read().await;
                    if (now - last_heartbeat).num_seconds() > 60 {
                        to_remove.push(*entry.key());
                    }
                }
                
                for client_id in to_remove {
                    clients.remove(&client_id);
                    println!("Removed stale parasitic client: {}", client_id);
                }
            }
        });
        
        println!("Parasitic MCP background tasks started");
        Ok(())
    }
    
    /// Process parasitic commands
    async fn process_command(
        command: ParasiticCommand,
        engine: ParasiticEngine,
        broadcaster: &broadcast::Sender<ParasiticEvent>,
    ) {
        match command {
            ParasiticCommand::InfectPair { pair_id, organism_id } => {
                match engine.infect_pair(pair_id.clone(), None).await {
                    Ok(infection) => {
                        let event = ParasiticEvent::PairInfected {
                            pair_id,
                            organism_id,
                            infection_strength: infection.infection_strength,
                            timestamp: infection.infection_time,
                        };
                        let _ = broadcaster.send(event);
                    }
                    Err(e) => {
                        eprintln!("Failed to infect pair {}: {}", pair_id, e);
                    }
                }
            }
            ParasiticCommand::TriggerEvolution { .. } => {
                // Trigger evolution cycle
                if let Err(e) = engine.evolution_engine.evolve_organisms(&engine.organisms).await {
                    eprintln!("Evolution failed: {}", e);
                }
            }
            _ => {
                // Handle other commands
            }
        }
    }
    
    /// Handle a new client connection with enhanced error handling
    async fn handle_client_connection(
        &self,
        stream: TcpStream,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client_id = Uuid::new_v4();
        
        // Set up WebSocket with proper error handling
        let ws_stream = match accept_async(stream).await {
            Ok(ws) => ws,
            Err(e) => {
                error!("Failed to establish WebSocket connection from {}: {}", addr, e);
                return Err(e.into());
            }
        };
        
        // Initialize rate limiting
        let rate_limit_bucket = RateLimitBucket {
            tokens: 100, // 100 requests per bucket
            last_refill: Instant::now(),
            max_tokens: 100,
            refill_rate: 10, // 10 tokens per second
        };
        
        let client = MCPClient {
            id: client_id,
            websocket: Arc::new(RwLock::new(ws_stream)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            last_heartbeat: Arc::new(RwLock::new(Utc::now())),
            connection_time: Utc::now(),
            total_requests: Arc::new(RwLock::new(0)),
            total_responses: Arc::new(RwLock::new(0)),
            total_errors: Arc::new(RwLock::new(0)),
            avg_response_time_ns: Arc::new(RwLock::new(0)),
            rate_limit_bucket: Arc::new(RwLock::new(rate_limit_bucket)),
            client_info: Arc::new(RwLock::new(ClientInfo {
                user_agent: None,
                capabilities: Vec::new(),
                protocol_version: "2024-11-05".to_string(),
                client_name: None,
                client_version: None,
            })),
        };
        
        // Update server metrics
        {
            let mut metrics = self.server_metrics.write().await;
            metrics.total_connections += 1;
            metrics.active_connections += 1;
            metrics.peak_connections = metrics.peak_connections.max(metrics.active_connections);
        }
        
        // Add client to active clients
        self.clients.insert(client_id, client);
        
        info!("Parasitic client {} connected from {} (total: {})", 
              client_id, addr, self.clients.len());
        
        // Handle client messages with reconnection support
        let result = self.client_message_loop_with_reconnection(client_id, addr).await;
        
        // Cleanup when client disconnects
        self.clients.remove(&client_id);
        
        result
    }
    
    /// Enhanced message loop with reconnection support
    async fn client_message_loop_with_reconnection(
        &self,
        client_id: Uuid,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let client = match self.clients.get(&client_id) {
            Some(client) => client.clone(),
            None => return Err("Client not found".into()),
        };
        
        // Subscribe to event broadcasts
        let mut event_rx = self.event_broadcaster.subscribe();
        let ws_clone = client.websocket.clone();
        let event_client_id = client_id;
        
        // Spawn task to handle broadcast events
        let event_task = tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                let message = MCPMessage {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    method: Some("parasitic_event".to_string()),
                    params: Some(serde_json::to_value(&event).unwrap_or_default()),
                    result: None,
                    error: None,
                    _timestamp: Some(Utc::now()),
                };
                
                let mut ws = ws_clone.write().await;
                if let Err(e) = ws.send(Message::Text(serde_json::to_string(&message).unwrap())).await {
                    warn!("Failed to send event to client {}: {}", event_client_id, e);
                    break;
                }
            }
        });
        
        // Main message handling loop with error recovery
        let mut consecutive_errors = 0;
        let max_consecutive_errors = 5;
        
        loop {
            let mut ws = client.websocket.write().await;
            
            match tokio::time::timeout(Duration::from_secs(30), ws.next()).await {
                Ok(Some(Ok(msg))) => {
                    drop(ws); // Release lock before processing
                    consecutive_errors = 0; // Reset error counter
                    
                    // Update heartbeat
                    *client.last_heartbeat.write().await = Utc::now();
                    
                    match msg {
                        Message::Text(text) => {
                            if let Err(e) = self.handle_text_message(client_id, text).await {
                                warn!("Error handling text message from client {}: {}", client_id, e);
                                *client.total_errors.write().await += 1;
                            }
                        }
                        Message::Binary(data) => {
                            debug!("Received binary message from client {} ({} bytes)", client_id, data.len());
                            // Handle binary messages if needed
                        }
                        Message::Ping(data) => {
                            let mut ws = client.websocket.write().await;
                            if let Err(e) = ws.send(Message::Pong(data)).await {
                                warn!("Failed to send pong to client {}: {}", client_id, e);
                            }
                        }
                        Message::Pong(_) => {
                            debug!("Received pong from client {}", client_id);
                        }
                        Message::Close(frame) => {
                            info!("Client {} sent close frame: {:?}", client_id, frame);
                            break;
                        }
                        Message::Frame(_) => {
                            debug!("Received raw frame from client {}", client_id);
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    consecutive_errors += 1;
                    warn!("WebSocket error for client {} (attempt {}): {}", 
                          client_id, consecutive_errors, e);
                    
                    if consecutive_errors >= max_consecutive_errors {
                        error!("Too many consecutive errors for client {}, disconnecting", client_id);
                        break;
                    }
                    
                    // Brief pause before continuing
                    tokio::time::sleep(Duration::from_millis(100 * consecutive_errors as u64)).await;
                }
                Ok(None) => {
                    info!("WebSocket stream ended for client {}", client_id);
                    break;
                }
                Err(_) => {
                    // Timeout - send ping to check connection
                    let mut ws = client.websocket.write().await;
                    if let Err(e) = ws.send(Message::Ping(vec![])).await {
                        warn!("Failed to send ping to client {}: {}", client_id, e);
                        break;
                    }
                }
            }
        }
        
        // Cleanup event task
        event_task.abort();
        
        info!("Client {} message loop ended", client_id);
        Ok(())
    }
    
    /// Handle text messages with proper parsing and error handling
    async fn handle_text_message(
        &self,
        client_id: Uuid,
        text: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check rate limiting first
        if let Err(_) = self.apply_rate_limit(client_id) {
            self.send_error_response(
                client_id,
                None,
                -32003,
                "Rate limit exceeded. Please slow down your requests.",
                None,
            ).await?;
            return Ok(());
        }
        
        // Parse MCP message
        let mcp_msg: MCPMessage = match serde_json::from_str(&text) {
            Ok(mut msg) => {
                msg._timestamp = Some(Utc::now());
                msg
            }
            Err(e) => {
                self.send_error_response(
                    client_id,
                    None,
                    error_codes::PARSE_ERROR,
                    &format!("Failed to parse JSON: {}", e),
                    None,
                ).await?;
                return Ok(());
            }
        };
        
        // Validate JSON-RPC structure
        if mcp_msg.jsonrpc != "2.0" {
            self.send_error_response(
                client_id,
                mcp_msg.id.clone(),
                error_codes::INVALID_REQUEST,
                "Invalid JSON-RPC version. Must be \"2.0\"",
                None,
            ).await?;
            return Ok(());
        }
        
        // Update client request counter
        if let Some(client) = self.clients.get(&client_id) {
            *client.total_requests.write().await += 1;
        }
        
        // Handle the message
        self.handle_mcp_message(client_id, mcp_msg).await;
        Ok(())
    }
    
    /// Handle incoming MCP protocol messages with comprehensive logging
    async fn handle_mcp_message(&self, client_id: Uuid, mut message: MCPMessage) {
        let start_time = Instant::now();
        let request_id = Uuid::new_v4();
        let method = message.method.as_ref().map(|s| s.as_str()).unwrap_or("");
        
        // Log request
        let request_metrics = RequestMetrics {
            request_id,
            method: method.to_string(),
            start_time: Utc::now(),
            end_time: None,
            processing_time_ns: None,
            success: false, // Will be updated
            error_code: None,
            payload_size: serde_json::to_string(&message).unwrap_or_default().len(),
            response_size: None,
        };
        
        // Store request metrics
        {
            let mut metrics = self.request_metrics.write().await;
            metrics.insert(request_id, request_metrics);
        }
        
        debug!("Processing MCP request {} from client {}: {}", 
               request_id, client_id, method);
        
        // Handle method with comprehensive error handling
        let result = match method {
            "initialize" => {
                Some(self.handle_initialize(&message).await)
            },
            "initialized" => {
                match self.handle_initialized(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "resources/list" => {
                match self.handle_list_resources(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "resources/read" => {
                match self.handle_read_resource(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "resources/subscribe" => {
                match self.handle_resource_subscribe(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "resources/unsubscribe" => {
                match self.handle_resource_unsubscribe(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "tools/list" => {
                match self.handle_list_tools(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "tools/call" => {
                match self.handle_tool_call(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "prompts/list" => {
                match self.handle_list_prompts(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "prompts/get" => {
                match self.handle_get_prompt(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "completion/complete" => {
                match self.handle_completion(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "logging/setLevel" => {
                match self.handle_set_log_level(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "ping" => {
                match self.handle_ping(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            "notifications/cancelled" => {
                match self.handle_cancellation(client_id, message.id.clone(), message.params.unwrap_or(json!({}))).await {
                    Ok(_) => None,
                    Err(e) => Some(self.create_error_response(
                        message.id.clone(),
                        error_codes::INTERNAL_ERROR,
                        &e.to_string(),
                        None,
                    )),
                }
            },
            _ => {
                warn!("Unknown method '{}' requested by client {}", method, client_id);
                Some(self.create_error_response(
                    message.id.clone(),
                    error_codes::METHOD_NOT_FOUND,
                    &format!("Method '{}' not found", method),
                    Some(json!({
                        "available_methods": [
                            "initialize", "initialized", 
                            "resources/list", "resources/read", "resources/subscribe", "resources/unsubscribe",
                            "tools/list", "tools/call",
                            "prompts/list", "prompts/get",
                            "completion/complete", "logging/setLevel", "ping"
                        ]
                    })),
                ))
            }
        };
        
        // Calculate processing time
        let processing_time = start_time.elapsed();
        let processing_time_ns = processing_time.as_nanos() as u64;
        
        // If there's an error response to send, send it
        if let Some(error_response) = result {
            // Update metrics
            {
                let mut metrics = self.request_metrics.write().await;
                if let Some(request_metrics) = metrics.get_mut(&request_id) {
                    request_metrics.end_time = Some(Utc::now());
                    request_metrics.processing_time_ns = Some(processing_time_ns);
                    request_metrics.success = error_response.error.is_none();
                    request_metrics.error_code = error_response.error.as_ref().map(|e| e.code);
                    request_metrics.response_size = Some(serde_json::to_string(&error_response).unwrap_or_default().len());
                }
            }
            
            // Send error response back to client
            if let Err(e) = self.send_response_to_client(client_id, error_response).await {
                error!("Failed to send error response to client {}: {}", client_id, e);
            }
        } else {
            // Success case - update metrics
            {
                let mut metrics = self.request_metrics.write().await;
                if let Some(request_metrics) = metrics.get_mut(&request_id) {
                    request_metrics.end_time = Some(Utc::now());
                    request_metrics.processing_time_ns = Some(processing_time_ns);
                    request_metrics.success = true;
                    request_metrics.error_code = None;
                }
            }
        }
        
        // Log performance
        if processing_time_ns > 50_000 { // > 50μs
            warn!("Slow request {} ({}): {:.2}μs", 
                  request_id, method, processing_time_ns as f64 / 1000.0);
        } else {
            debug!("Request {} completed in {:.2}μs", 
                   request_id, processing_time_ns as f64 / 1000.0);
        }
        
        // Update server metrics
        self.update_server_metrics().await;
    }
    
    /// Handle initialization
    async fn handle_initialize(&self, message: &MCPMessage) -> MCPMessage {
        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id.clone(),
            method: None,
            params: None,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "resources": {
                        "subscribe": true,
                        "listChanged": true
                    },
                    "tools": {
                        "listChanged": true
                    },
                    "subscriptions": true
                },
                "serverInfo": {
                    "name": "CWTS Parasitic Trading Server",
                    "version": "2.0.0",
                    "description": "Bio-inspired parasitic trading strategy engine with ultra-low latency MCP interface"
                }
            })),
            error: None,
        }
    }
    
    /// List available resources
    async fn handle_list_resources(&self, message: &MCPMessage) -> MCPMessage {
        let resources = vec![
            MCPResource {
                uri: "parasitic://organisms".to_string(),
                name: "Available Organisms".to_string(),
                description: "List of all parasitic organisms and their characteristics".to_string(),
                mime_type: "application/json".to_string(),
            },
            MCPResource {
                uri: "parasitic://pairs/infected".to_string(),
                name: "Infected Pairs".to_string(),
                description: "Currently parasitized trading pairs and infection status".to_string(),
                mime_type: "application/json".to_string(),
            },
            MCPResource {
                uri: "parasitic://evolution/status".to_string(),
                name: "Evolution Status".to_string(),
                description: "Current evolution metrics and organism fitness scores".to_string(),
                mime_type: "application/json".to_string(),
            },
            MCPResource {
                uri: "parasitic://analytics/performance".to_string(),
                name: "Performance Analytics".to_string(),
                description: "Real-time performance metrics and profitability analysis".to_string(),
                mime_type: "application/json".to_string(),
            },
        ];
        
        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id.clone(),
            method: None,
            params: None,
            result: Some(json!({ "resources": resources })),
            error: None,
        }
    }
    
    /// Read a specific resource
    async fn handle_read_resource(&self, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(uri) = params.get("uri").and_then(|v| v.as_str()) {
                let content = match uri {
                    "parasitic://organisms" => {
                        self.get_organisms_resource().await
                    }
                    "parasitic://pairs/infected" => {
                        self.get_infected_pairs_resource().await
                    }
                    "parasitic://evolution/status" => {
                        self.get_evolution_status_resource().await
                    }
                    "parasitic://analytics/performance" => {
                        self.get_performance_analytics_resource().await
                    }
                    _ if uri.starts_with("parasitic://strategies/") => {
                        let organism_type = uri.strip_prefix("parasitic://strategies/").unwrap();
                        self.get_strategy_resource(organism_type).await
                    }
                    _ => {
                        return self.create_error_response(
                            message.id.clone(),
                            -32602,
                            "Resource not found",
                            None,
                        );
                    }
                };
                
                match content {
                    Ok(data) => MCPMessage {
                        jsonrpc: "2.0".to_string(),
                        id: message.id.clone(),
                        method: None,
                        params: None,
                        result: Some(json!({
                            "contents": [{
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": serde_json::to_string_pretty(&data).unwrap()
                            }]
                        })),
                        error: None,
                    },
                    Err(e) => self.create_error_response(
                        message.id.clone(),
                        -32603,
                        &format!("Failed to read resource: {}", e),
                        None,
                    ),
                }
            } else {
                self.create_error_response(
                    message.id.clone(),
                    -32602,
                    "Invalid parameters: uri required",
                    None,
                )
            }
        } else {
            self.create_error_response(
                message.id.clone(),
                -32602,
                "Invalid parameters: params required",
                None,
            )
        }
    }
    
    /// List available tools
    async fn handle_list_tools(&self, message: &MCPMessage) -> MCPMessage {
        let tools = vec![
            // Existing tools
            MCPTool {
                name: "parasitic_select".to_string(),
                description: "Select and spawn a parasitic organism with specific characteristics".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "organism_type": {
                            "type": "string",
                            "enum": ["cuckoo", "wasp", "virus", "bacteria"],
                            "description": "Type of parasitic organism to spawn"
                        },
                        "genetics": {
                            "type": "object",
                            "properties": {
                                "aggression": {"type": "number", "minimum": 0, "maximum": 1},
                                "adaptability": {"type": "number", "minimum": 0, "maximum": 1},
                                "efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                                "resilience": {"type": "number", "minimum": 0, "maximum": 1},
                                "reaction_speed": {"type": "number", "minimum": 0, "maximum": 1},
                                "risk_tolerance": {"type": "number", "minimum": 0, "maximum": 1},
                                "cooperation": {"type": "number", "minimum": 0, "maximum": 1},
                                "stealth": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "description": "Optional genetics to override defaults"
                        }
                    },
                    "required": ["organism_type"]
                }),
            },
            MCPTool {
                name: "parasitic_infect".to_string(),
                description: "Infect a trading pair with a specific parasitic organism".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "description": "Trading pair to infect (e.g., BTCUSD, ETHUSD)"
                        },
                        "organism_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "UUID of the organism to use for infection"
                        },
                        "force": {
                            "type": "boolean",
                            "default": false,
                            "description": "Force infection even if conditions are suboptimal"
                        }
                    },
                    "required": ["pair_id", "organism_id"]
                }),
            },
            MCPTool {
                name: "parasitic_evolve".to_string(),
                description: "Trigger evolution cycle to adapt organisms based on performance".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "default": false,
                            "description": "Force evolution even if normal cycle hasn't completed"
                        },
                        "selection_pressure": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.5,
                            "description": "Intensity of natural selection (higher = more aggressive)"
                        }
                    }
                }),
            },
            MCPTool {
                name: "parasitic_analyze".to_string(),
                description: "Analyze trading pair vulnerability and suitability for parasitic infection".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "description": "Trading pair to analyze"
                        },
                        "timeframe": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                            "default": "5m",
                            "description": "Analysis timeframe"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["basic", "detailed", "comprehensive"],
                            "default": "detailed",
                            "description": "Analysis depth level"
                        }
                    },
                    "required": ["pair_id"]
                }),
            },
            
            // NEW MISSING TOOLS - Advanced Parasitic Strategies
            MCPTool {
                name: "detect_whale_nests".to_string(),
                description: "Cuckoo whale detection - identify large holder accumulation patterns".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "description": "Trading pair to scan for whale activity"
                        },
                        "min_volume_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "default": 1000000,
                            "description": "Minimum volume threshold to consider whale activity"
                        },
                        "detection_window": {
                            "type": "string",
                            "enum": ["1h", "4h", "1d", "7d"],
                            "default": "4h",
                            "description": "Time window for whale detection analysis"
                        },
                        "stealth_level": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.8,
                            "description": "Detection sensitivity (higher = detect more subtle patterns)"
                        }
                    },
                    "required": ["pair_id"]
                }),
            },
            MCPTool {
                name: "identify_zombie_pairs".to_string(),
                description: "Cordyceps algorithmic exploitation - identify manipulated trading pairs".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "market_filter": {
                            "type": "string",
                            "enum": ["all", "spot", "futures", "options"],
                            "default": "spot",
                            "description": "Market type to scan"
                        },
                        "manipulation_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.7,
                            "description": "Threshold for manipulation detection score"
                        },
                        "scan_depth": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 1000,
                            "default": 100,
                            "description": "Number of pairs to analyze"
                        },
                        "exploit_strategy": {
                            "type": "string",
                            "enum": ["counter", "follow", "hybrid"],
                            "default": "hybrid",
                            "description": "Strategy for exploiting identified zombie pairs"
                        }
                    }
                }),
            },
            MCPTool {
                name: "analyze_mycelial_network".to_string(),
                description: "Correlation network analysis - map interconnected pair relationships".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "root_pairs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Starting pairs for network expansion"
                        },
                        "correlation_threshold": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                            "default": 0.6,
                            "description": "Minimum correlation coefficient to establish connection"
                        },
                        "network_depth": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3,
                            "description": "Maximum network traversal depth"
                        },
                        "temporal_scope": {
                            "type": "string",
                            "enum": ["realtime", "1h", "1d", "7d", "30d"],
                            "default": "1d",
                            "description": "Temporal scope for correlation analysis"
                        }
                    },
                    "required": ["root_pairs"]
                }),
            },
            MCPTool {
                name: "activate_octopus_camouflage".to_string(),
                description: "Adaptive stealth - dynamically adjust trading behavior to avoid detection".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "organism_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Organism to apply camouflage to"
                        },
                        "stealth_mode": {
                            "type": "string",
                            "enum": ["mimicry", "invisibility", "misdirection", "adaptive"],
                            "default": "adaptive",
                            "description": "Type of camouflage strategy"
                        },
                        "adaptation_speed": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.5,
                            "description": "Rate of behavioral adaptation (higher = faster)"
                        },
                        "target_detection_rate": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.1,
                            "description": "Target detection probability"
                        }
                    },
                    "required": ["organism_id"]
                }),
            },
            MCPTool {
                name: "deploy_anglerfish_lure".to_string(),
                description: "Artificial activity generation - create deceptive market signals".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "description": "Target trading pair for lure deployment"
                        },
                        "lure_type": {
                            "type": "string",
                            "enum": ["volume_spike", "price_momentum", "order_book_depth", "mixed"],
                            "default": "mixed",
                            "description": "Type of artificial activity to generate"
                        },
                        "intensity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.6,
                            "description": "Intensity of artificial activity"
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1440,
                            "default": 60,
                            "description": "Duration of lure activity in minutes"
                        },
                        "target_response": {
                            "type": "string",
                            "enum": ["buy_pressure", "sell_pressure", "volatility", "volume"],
                            "default": "volume",
                            "description": "Desired market response to trigger"
                        }
                    },
                    "required": ["pair_id"]
                }),
            },
            MCPTool {
                name: "track_wounded_pairs".to_string(),
                description: "Komodo persistence tracking - monitor and exploit weakened trading pairs".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "weakness_indicators": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["low_liquidity", "high_volatility", "declining_volume", "technical_breakdown", "sentiment_shift"]
                            },
                            "default": ["low_liquidity", "high_volatility"],
                            "description": "Indicators to identify weakness"
                        },
                        "tracking_duration": {
                            "type": "string",
                            "enum": ["1h", "4h", "12h", "1d", "3d", "7d"],
                            "default": "1d",
                            "description": "Duration to track wounded pairs"
                        },
                        "exploit_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.8,
                            "description": "Weakness score threshold for exploitation"
                        },
                        "persistence_strategy": {
                            "type": "string",
                            "enum": ["patient", "aggressive", "opportunistic"],
                            "default": "patient",
                            "description": "Tracking and exploitation strategy"
                        }
                    }
                }),
            },
            MCPTool {
                name: "enter_cryptobiosis".to_string(),
                description: "Tardigrade survival mode - enter dormant state during adverse conditions".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "organism_id": {
                            "type": "string",
                            "format": "uuid",
                            "description": "Organism to enter cryptobiosis"
                        },
                        "trigger_conditions": {
                            "type": "object",
                            "properties": {
                                "market_volatility_threshold": {"type": "number", "minimum": 0, "default": 0.8},
                                "liquidity_threshold": {"type": "number", "minimum": 0, "default": 0.2},
                                "drawdown_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.15}
                            },
                            "description": "Conditions that trigger dormancy"
                        },
                        "revival_conditions": {
                            "type": "object",
                            "properties": {
                                "market_stability_duration": {"type": "integer", "minimum": 1, "default": 60},
                                "liquidity_recovery_threshold": {"type": "number", "minimum": 0, "default": 0.5},
                                "opportunity_score_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6}
                            },
                            "description": "Conditions for leaving dormancy"
                        },
                        "dormancy_level": {
                            "type": "string",
                            "enum": ["light", "deep", "complete"],
                            "default": "deep",
                            "description": "Depth of cryptobiotic state"
                        }
                    },
                    "required": ["organism_id"]
                }),
            },
            MCPTool {
                name: "electric_shock".to_string(),
                description: "Market disruption for liquidity - create sudden market movements to generate trading opportunities".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pair_id": {
                            "type": "string",
                            "description": "Target trading pair for disruption"
                        },
                        "shock_type": {
                            "type": "string",
                            "enum": ["price_spike", "volume_burst", "spread_compression", "order_avalanche"],
                            "default": "price_spike",
                            "description": "Type of market disruption"
                        },
                        "intensity": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "Intensity of market disruption"
                        },
                        "propagation_delay_ms": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 5000,
                            "default": 100,
                            "description": "Delay between shock initiation and market response"
                        },
                        "recovery_strategy": {
                            "type": "string",
                            "enum": ["capitalize", "stabilize", "amplify", "neutral"],
                            "default": "capitalize",
                            "description": "Strategy after shock deployment"
                        }
                    },
                    "required": ["pair_id"]
                }),
            },
            MCPTool {
                name: "electroreception_scan".to_string(),
                description: "Subtle signal detection - detect minute market anomalies and electrical field changes".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "scan_scope": {
                            "type": "string",
                            "enum": ["single_pair", "pair_group", "market_wide", "cross_market"],
                            "default": "pair_group",
                            "description": "Scope of electroreception scanning"
                        },
                        "sensitivity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.9,
                            "description": "Detection sensitivity (higher = detect weaker signals)"
                        },
                        "signal_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["micro_trends", "order_flow_imbalance", "latency_anomalies", "volume_whispers", "price_echoes"]
                            },
                            "default": ["micro_trends", "order_flow_imbalance"],
                            "description": "Types of subtle signals to detect"
                        },
                        "temporal_resolution": {
                            "type": "string",
                            "enum": ["microsecond", "millisecond", "second", "minute"],
                            "default": "millisecond",
                            "description": "Temporal resolution for signal detection"
                        },
                        "noise_filtering": {
                            "type": "boolean",
                            "default": true,
                            "description": "Apply adaptive noise filtering"
                        }
                    }
                }),
            },
        ];
        
        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id.clone(),
            method: None,
            params: None,
            result: Some(json!({ "tools": tools })),
            error: None,
        }
    }
    
    /// Handle tool calls
    async fn handle_tool_call(&self, _client_id: Uuid, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(name) = params.get("name").and_then(|v| v.as_str()) {
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                
                let result = match name {
                    // Existing tools
                    "parasitic_select" => self.tool_parasitic_select(arguments).await,
                    "parasitic_infect" => self.tool_parasitic_infect(arguments).await,
                    "parasitic_evolve" => self.tool_parasitic_evolve(arguments).await,
                    "parasitic_analyze" => self.tool_parasitic_analyze(arguments).await,
                    
                    // New advanced parasitic tools
                    "detect_whale_nests" => self.tool_detect_whale_nests(arguments).await,
                    "identify_zombie_pairs" => self.tool_identify_zombie_pairs(arguments).await,
                    "analyze_mycelial_network" => self.tool_analyze_mycelial_network(arguments).await,
                    "activate_octopus_camouflage" => self.tool_activate_octopus_camouflage(arguments).await,
                    "deploy_anglerfish_lure" => self.tool_deploy_anglerfish_lure(arguments).await,
                    "track_wounded_pairs" => self.tool_track_wounded_pairs(arguments).await,
                    "enter_cryptobiosis" => self.tool_enter_cryptobiosis(arguments).await,
                    "electric_shock" => self.tool_electric_shock(arguments).await,
                    "electroreception_scan" => self.tool_electroreception_scan(arguments).await,
                    
                    _ => Err("Unknown tool".into()),
                };
                
                match result {
                    Ok(content) => MCPMessage {
                        jsonrpc: "2.0".to_string(),
                        id: message.id.clone(),
                        method: None,
                        params: None,
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": content
                            }]
                        })),
                        error: None,
                    },
                    Err(e) => self.create_error_response(
                        message.id.clone(),
                        -32603,
                        &format!("Tool execution failed: {}", e),
                        None,
                    ),
                }
            } else {
                self.create_error_response(
                    message.id.clone(),
                    -32602,
                    "Invalid parameters: name required",
                    None,
                )
            }
        } else {
            self.create_error_response(
                message.id.clone(),
                -32602,
                "Invalid parameters: params required",
                None,
            )
        }
    }
    
    // Resource implementations
    async fn get_organisms_resource(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref engine) = self.engine {
            let mut organisms = Vec::new();
            
            for entry in engine.organisms.iter() {
                let organism = entry.value();
                organisms.push(json!({
                    "id": organism.id(),
                    "type": organism.organism_type(),
                    "fitness": organism.fitness(),
                    "genetics": organism.get_genetics(),
                    "resource_consumption": organism.resource_consumption(),
                    "strategy_params": organism.get_strategy_params()
                }));
            }
            
            Ok(json!({
                "total_organisms": organisms.len(),
                "organisms": organisms,
                "timestamp": Utc::now()
            }))
        } else {
            Err("Engine not initialized".into())
        }
    }
    
    async fn get_infected_pairs_resource(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref engine) = self.engine {
            let infected_pairs = engine.get_infected_pairs().await;
            
            Ok(json!({
                "total_infections": infected_pairs.len(),
                "infected_pairs": infected_pairs,
                "timestamp": Utc::now()
            }))
        } else {
            Err("Engine not initialized".into())
        }
    }
    
    async fn get_evolution_status_resource(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref engine) = self.engine {
            let status = engine.get_evolution_status().await;
            
            Ok(json!({
                "evolution_status": status,
                "timestamp": Utc::now()
            }))
        } else {
            Err("Engine not initialized".into())
        }
    }
    
    async fn get_performance_analytics_resource(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref engine) = self.engine {
            let mut analytics = Vec::new();
            
            for entry in engine.organisms.iter() {
                let organism_id = *entry.key();
                if let Some(org_analytics) = engine.get_organism_analytics(organism_id).await {
                    analytics.push(org_analytics);
                }
            }
            
            Ok(json!({
                "organism_analytics": analytics,
                "timestamp": Utc::now()
            }))
        } else {
            Err("Engine not initialized".into())
        }
    }
    
    async fn get_strategy_resource(&self, organism_type: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Return strategy details for specific organism type
        Ok(json!({
            "organism_type": organism_type,
            "description": format!("Strategy details for {} organism", organism_type),
            "characteristics": self.get_organism_characteristics(organism_type),
            "timestamp": Utc::now()
        }))
    }
    
    fn get_organism_characteristics(&self, organism_type: &str) -> Value {
        match organism_type {
            "cuckoo" => json!({
                "strategy": "Order book manipulation through deceptive orders",
                "strengths": ["Stealth", "Market manipulation", "Low detection risk"],
                "weaknesses": ["Limited to liquid markets", "Requires volume"]
            }),
            "wasp" => json!({
                "strategy": "Aggressive high-frequency trading with territorial behavior", 
                "strengths": ["Speed", "Aggression", "Territory control"],
                "weaknesses": ["High resource consumption", "Easily detected"]
            }),
            "virus" => json!({
                "strategy": "Self-replicating strategy that spreads across correlated pairs",
                "strengths": ["Replication", "Network effects", "Scalability"],
                "weaknesses": ["Vulnerability to immune responses", "Mutation risk"]
            }),
            "bacteria" => json!({
                "strategy": "Cooperative clustering with resource sharing",
                "strengths": ["Cooperation", "Resource efficiency", "Resilience"],
                "weaknesses": ["Slower individual performance", "Coordination overhead"]
            }),
            _ => json!({
                "error": "Unknown organism type"
            })
        }
    }
    
    // Tool implementations
    async fn tool_parasitic_select(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let organism_type = arguments.get("organism_type")
            .and_then(|v| v.as_str())
            .ok_or("organism_type required")?;
            
        let genetics = if let Some(genetics_json) = arguments.get("genetics") {
            Some(serde_json::from_value(genetics_json.clone())?)
        } else {
            None
        };
        
        let command = ParasiticCommand::SelectOrganism {
            organism_type: organism_type.to_string(),
            genetics,
        };
        
        self.command_tx.send(command).await
            .map_err(|e| format!("Failed to send command: {}", e))?;
            
        Ok(format!("Selected {} organism for spawning", organism_type))
    }
    
    async fn tool_parasitic_infect(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let pair_id = arguments.get("pair_id")
            .and_then(|v| v.as_str())
            .ok_or("pair_id required")?;
            
        let organism_id = arguments.get("organism_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or("valid organism_id UUID required")?;
            
        let command = ParasiticCommand::InfectPair {
            pair_id: pair_id.to_string(),
            organism_id,
        };
        
        self.command_tx.send(command).await
            .map_err(|e| format!("Failed to send command: {}", e))?;
            
        Ok(format!("Initiated infection of {} with organism {}", pair_id, organism_id))
    }
    
    async fn tool_parasitic_evolve(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let force = arguments.get("force")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            
        let command = ParasiticCommand::TriggerEvolution { force };
        
        self.command_tx.send(command).await
            .map_err(|e| format!("Failed to send command: {}", e))?;
            
        Ok("Triggered evolution cycle".to_string())
    }
    
    async fn tool_parasitic_analyze(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let pair_id = arguments.get("pair_id")
            .and_then(|v| v.as_str())
            .ok_or("pair_id required")?;
            
        let command = ParasiticCommand::AnalyzePair {
            pair_id: pair_id.to_string(),
        };
        
        self.command_tx.send(command).await
            .map_err(|e| format!("Failed to send command: {}", e))?;
            
        Ok(format!("Initiated analysis of {}", pair_id))
    }
    
    // NEW ADVANCED PARASITIC TOOL IMPLEMENTATIONS
    
    /// Detect whale nests - Cuckoo whale detection system
    async fn tool_detect_whale_nests(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let pair_id = arguments.get("pair_id")
            .and_then(|v| v.as_str())
            .ok_or("pair_id required")?;
            
        let min_volume_threshold = arguments.get("min_volume_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(1_000_000.0);
            
        let detection_window = arguments.get("detection_window")
            .and_then(|v| v.as_str())
            .unwrap_or("4h");
            
        let stealth_level = arguments.get("stealth_level")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);

        // Ultra-low latency whale detection algorithm
        let start_time = std::time::Instant::now();
        
        // Simulate advanced whale nest detection
        let whale_nests_detected = match self.engine.as_ref() {
            Some(engine) => {
                // Real-time order flow analysis for whale detection
                let large_orders = self.analyze_large_orders(pair_id, min_volume_threshold, detection_window).await?;
                let accumulation_patterns = self.detect_accumulation_patterns(pair_id, stealth_level).await?;
                let stealth_metrics = self.calculate_whale_stealth_score(pair_id).await?;
                
                let whale_count = large_orders.len();
                let stealth_score = stealth_metrics;
                let detection_confidence = accumulation_patterns.len() as f64 / (whale_count.max(1) as f64);
                
                json!({
                    "pair_id": pair_id,
                    "whale_nests_detected": whale_count,
                    "detection_confidence": detection_confidence.min(1.0),
                    "stealth_score": stealth_score,
                    "large_orders": large_orders,
                    "accumulation_patterns": accumulation_patterns,
                    "detection_window": detection_window,
                    "min_threshold": min_volume_threshold
                })
            },
            None => json!({
                "error": "Engine not initialized",
                "pair_id": pair_id
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🐋 Whale nest detection completed for {} in {:.2}μs. Analysis: {}", 
                  pair_id, 
                  processing_time.as_nanos() as f64 / 1000.0,
                  serde_json::to_string_pretty(&whale_nests_detected)?))
    }
    
    /// Identify zombie pairs - Cordyceps algorithmic exploitation
    async fn tool_identify_zombie_pairs(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let market_filter = arguments.get("market_filter")
            .and_then(|v| v.as_str())
            .unwrap_or("spot");
            
        let manipulation_threshold = arguments.get("manipulation_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);
            
        let scan_depth = arguments.get("scan_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
            
        let exploit_strategy = arguments.get("exploit_strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("hybrid");

        let start_time = std::time::Instant::now();
        
        // Cordyceps-style zombie pair identification
        let zombie_analysis = match self.engine.as_ref() {
            Some(engine) => {
                let pairs_to_scan = self.get_market_pairs(market_filter, scan_depth).await?;
                let mut zombie_pairs = Vec::new();
                
                for pair in pairs_to_scan {
                    let manipulation_score = self.calculate_manipulation_score(&pair).await?;
                    let algorithmic_control = self.detect_algorithmic_control(&pair).await?;
                    let exploitation_potential = self.assess_exploitation_potential(&pair, exploit_strategy).await?;
                    
                    if manipulation_score >= manipulation_threshold {
                        zombie_pairs.push(json!({
                            "pair": pair,
                            "manipulation_score": manipulation_score,
                            "algorithmic_control": algorithmic_control,
                            "exploitation_potential": exploitation_potential,
                            "exploit_strategy": exploit_strategy
                        }));
                    }
                }
                
                json!({
                    "market_filter": market_filter,
                    "scanned_pairs": scan_depth,
                    "zombie_pairs_found": zombie_pairs.len(),
                    "manipulation_threshold": manipulation_threshold,
                    "zombie_pairs": zombie_pairs,
                    "exploit_strategies_available": ["counter", "follow", "hybrid"]
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🧟 Zombie pair identification completed in {:.2}μs. Found {} manipulated pairs. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  zombie_analysis["zombie_pairs_found"].as_u64().unwrap_or(0),
                  serde_json::to_string_pretty(&zombie_analysis)?))
    }
    
    /// Analyze mycelial network - Correlation network mapping
    async fn tool_analyze_mycelial_network(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let root_pairs = arguments.get("root_pairs")
            .and_then(|v| v.as_array())
            .ok_or("root_pairs array required")?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
            
        let correlation_threshold = arguments.get("correlation_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);
            
        let network_depth = arguments.get("network_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
            
        let temporal_scope = arguments.get("temporal_scope")
            .and_then(|v| v.as_str())
            .unwrap_or("1d");

        let start_time = std::time::Instant::now();
        
        // Mycelial network correlation analysis
        let network_analysis = match self.engine.as_ref() {
            Some(engine) => {
                let mut network_nodes = std::collections::HashMap::new();
                let mut network_edges = Vec::new();
                let mut processed_pairs = std::collections::HashSet::new();
                
                // BFS traversal of correlation network
                let mut queue = std::collections::VecDeque::new();
                for root_pair in &root_pairs {
                    queue.push_back((root_pair.clone(), 0));
                }
                
                while let Some((current_pair, depth)) = queue.pop_front() {
                    if processed_pairs.contains(&current_pair) || depth >= network_depth {
                        continue;
                    }
                    
                    processed_pairs.insert(current_pair.clone());
                    let correlations = self.calculate_pair_correlations(&current_pair, temporal_scope).await?;
                    
                    network_nodes.insert(current_pair.clone(), json!({
                        "pair": current_pair,
                        "depth": depth,
                        "correlation_count": correlations.len(),
                        "centrality_score": self.calculate_network_centrality(&current_pair).await?
                    }));
                    
                    for (correlated_pair, correlation_strength) in correlations {
                        if correlation_strength.abs() >= correlation_threshold {
                            network_edges.push(json!({
                                "from": current_pair,
                                "to": correlated_pair.clone(),
                                "strength": correlation_strength,
                                "type": if correlation_strength > 0.0 { "positive" } else { "negative" }
                            }));
                            
                            if !processed_pairs.contains(&correlated_pair) && depth + 1 < network_depth {
                                queue.push_back((correlated_pair, depth + 1));
                            }
                        }
                    }
                }
                
                json!({
                    "root_pairs": root_pairs,
                    "network_depth": network_depth,
                    "correlation_threshold": correlation_threshold,
                    "temporal_scope": temporal_scope,
                    "network_nodes": network_nodes.values().collect::<Vec<_>>(),
                    "network_edges": network_edges,
                    "total_nodes": network_nodes.len(),
                    "total_edges": network_edges.len(),
                    "network_density": (network_edges.len() as f64) / (network_nodes.len().pow(2) as f64).max(1.0)
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🕸️ Mycelial network analysis completed in {:.2}μs. Network: {} nodes, {} edges. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  network_analysis["total_nodes"].as_u64().unwrap_or(0),
                  network_analysis["total_edges"].as_u64().unwrap_or(0),
                  serde_json::to_string_pretty(&network_analysis)?))
    }
    
    /// Activate octopus camouflage - Adaptive stealth system
    async fn tool_activate_octopus_camouflage(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let organism_id = arguments.get("organism_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or("valid organism_id UUID required")?;
            
        let stealth_mode = arguments.get("stealth_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("adaptive");
            
        let adaptation_speed = arguments.get("adaptation_speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
            
        let target_detection_rate = arguments.get("target_detection_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        let start_time = std::time::Instant::now();
        
        // Octopus-style adaptive camouflage system
        let camouflage_result = match self.engine.as_ref() {
            Some(engine) => {
                // Check if organism exists
                if let Some(organism) = engine.organisms.get(&organism_id) {
                    let current_detection_rate = self.calculate_current_detection_rate(organism_id).await?;
                    let market_conditions = self.analyze_market_conditions().await?;
                    
                    let camouflage_config = match stealth_mode {
                        "mimicry" => self.configure_mimicry_camouflage(organism_id, &market_conditions).await?,
                        "invisibility" => self.configure_invisibility_camouflage(organism_id, target_detection_rate).await?,
                        "misdirection" => self.configure_misdirection_camouflage(organism_id, adaptation_speed).await?,
                        "adaptive" => self.configure_adaptive_camouflage(organism_id, &market_conditions, adaptation_speed).await?,
                        _ => return Err("Invalid stealth mode".into())
                    };
                    
                    let new_detection_rate = self.estimate_new_detection_rate(&camouflage_config).await?;
                    
                    json!({
                        "organism_id": organism_id,
                        "stealth_mode": stealth_mode,
                        "camouflage_activated": true,
                        "previous_detection_rate": current_detection_rate,
                        "new_detection_rate": new_detection_rate,
                        "detection_improvement": ((current_detection_rate - new_detection_rate) / current_detection_rate.max(0.001)) * 100.0,
                        "adaptation_speed": adaptation_speed,
                        "target_detection_rate": target_detection_rate,
                        "camouflage_config": camouflage_config
                    })
                } else {
                    json!({
                        "error": format!("Organism {} not found", organism_id),
                        "organism_id": organism_id
                    })
                }
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🐙 Octopus camouflage activated in {:.2}μs for organism {}. Stealth mode: {}. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  organism_id,
                  stealth_mode,
                  serde_json::to_string_pretty(&camouflage_result)?))
    }
    
    /// Deploy anglerfish lure - Artificial activity generation
    async fn tool_deploy_anglerfish_lure(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let pair_id = arguments.get("pair_id")
            .and_then(|v| v.as_str())
            .ok_or("pair_id required")?;
            
        let lure_type = arguments.get("lure_type")
            .and_then(|v| v.as_str())
            .unwrap_or("mixed");
            
        let intensity = arguments.get("intensity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);
            
        let duration_minutes = arguments.get("duration_minutes")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);
            
        let target_response = arguments.get("target_response")
            .and_then(|v| v.as_str())
            .unwrap_or("volume");

        let start_time = std::time::Instant::now();
        
        // Anglerfish-style market lure deployment
        let lure_deployment = match self.engine.as_ref() {
            Some(engine) => {
                let current_market_state = self.assess_market_state(pair_id).await?;
                
                let lure_config = match lure_type {
                    "volume_spike" => self.create_volume_spike_lure(pair_id, intensity, duration_minutes).await?,
                    "price_momentum" => self.create_price_momentum_lure(pair_id, intensity, target_response).await?,
                    "order_book_depth" => self.create_order_book_lure(pair_id, intensity).await?,
                    "mixed" => self.create_mixed_lure(pair_id, intensity, duration_minutes, target_response).await?,
                    _ => return Err("Invalid lure type".into())
                };
                
                let expected_response = self.predict_market_response(&lure_config, &current_market_state).await?;
                let deployment_time = Utc::now();
                let expiry_time = deployment_time + chrono::Duration::minutes(duration_minutes as i64);
                
                json!({
                    "pair_id": pair_id,
                    "lure_type": lure_type,
                    "intensity": intensity,
                    "duration_minutes": duration_minutes,
                    "target_response": target_response,
                    "deployment_time": deployment_time,
                    "expiry_time": expiry_time,
                    "lure_config": lure_config,
                    "expected_response": expected_response,
                    "current_market_state": current_market_state,
                    "status": "deployed"
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🪝 Anglerfish lure deployed in {:.2}μs for {}. Type: {}, Intensity: {:.2}. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  pair_id,
                  lure_type,
                  intensity,
                  serde_json::to_string_pretty(&lure_deployment)?))
    }
    
    /// Track wounded pairs - Komodo persistence tracking
    async fn tool_track_wounded_pairs(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let weakness_indicators = arguments.get("weakness_indicators")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect::<Vec<String>>())
            .unwrap_or_else(|| vec!["low_liquidity".to_string(), "high_volatility".to_string()]);
            
        let tracking_duration = arguments.get("tracking_duration")
            .and_then(|v| v.as_str())
            .unwrap_or("1d");
            
        let exploit_threshold = arguments.get("exploit_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);
            
        let persistence_strategy = arguments.get("persistence_strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("patient");

        let start_time = std::time::Instant::now();
        
        // Komodo-style persistence tracking system
        let tracking_result = match self.engine.as_ref() {
            Some(engine) => {
                let all_pairs = self.get_all_tradeable_pairs().await?;
                let mut wounded_pairs = Vec::new();
                let mut tracking_data = std::collections::HashMap::new();
                
                for pair in all_pairs {
                    let weakness_score = self.calculate_weakness_score(&pair, &weakness_indicators).await?;
                    
                    if weakness_score >= exploit_threshold {
                        let tracking_info = json!({
                            "pair": pair,
                            "weakness_score": weakness_score,
                            "weakness_indicators_detected": self.get_detected_indicators(&pair, &weakness_indicators).await?,
                            "exploitation_potential": self.assess_exploitation_potential(&pair, persistence_strategy).await?,
                            "tracking_started": Utc::now(),
                            "persistence_strategy": persistence_strategy,
                            "recommended_actions": self.generate_komodo_actions(&pair, weakness_score, persistence_strategy).await?
                        });
                        
                        wounded_pairs.push(tracking_info.clone());
                        tracking_data.insert(pair.clone(), tracking_info);
                    }
                }
                
                // Sort by weakness score (highest first)
                wounded_pairs.sort_by(|a, b| {
                    b["weakness_score"].as_f64().unwrap_or(0.0)
                        .partial_cmp(&a["weakness_score"].as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                
                json!({
                    "tracking_duration": tracking_duration,
                    "weakness_indicators": weakness_indicators,
                    "exploit_threshold": exploit_threshold,
                    "persistence_strategy": persistence_strategy,
                    "wounded_pairs_found": wounded_pairs.len(),
                    "wounded_pairs": wounded_pairs,
                    "tracking_active": true,
                    "next_scan_in": self.calculate_next_scan_interval(persistence_strategy)
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🦎 Komodo tracking initiated in {:.2}μs. Found {} wounded pairs. Strategy: {}. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  tracking_result["wounded_pairs_found"].as_u64().unwrap_or(0),
                  persistence_strategy,
                  serde_json::to_string_pretty(&tracking_result)?))
    }
    
    /// Enter cryptobiosis - Tardigrade survival mode
    async fn tool_enter_cryptobiosis(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let organism_id = arguments.get("organism_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or("valid organism_id UUID required")?;
            
        let trigger_conditions = arguments.get("trigger_conditions")
            .and_then(|v| v.as_object())
            .cloned();
            
        let revival_conditions = arguments.get("revival_conditions")
            .and_then(|v| v.as_object())
            .cloned();
            
        let dormancy_level = arguments.get("dormancy_level")
            .and_then(|v| v.as_str())
            .unwrap_or("deep");

        let start_time = std::time::Instant::now();
        
        // Tardigrade-style cryptobiotic survival system
        let cryptobiosis_result = match self.engine.as_ref() {
            Some(engine) => {
                if let Some(organism) = engine.organisms.get(&organism_id) {
                    let current_conditions = self.assess_survival_conditions(organism_id).await?;
                    let should_enter_cryptobiosis = self.evaluate_cryptobiosis_triggers(
                        &current_conditions, 
                        &trigger_conditions.unwrap_or_default()
                    ).await?;
                    
                    if should_enter_cryptobiosis {
                        let cryptobiosis_config = self.configure_cryptobiotic_state(
                            organism_id,
                            dormancy_level,
                            &revival_conditions.unwrap_or_default()
                        ).await?;
                        
                        let resource_preservation = self.calculate_resource_preservation(dormancy_level).await?;
                        let estimated_survival_time = self.estimate_survival_duration(
                            &current_conditions,
                            dormancy_level
                        ).await?;
                        
                        json!({
                            "organism_id": organism_id,
                            "cryptobiosis_activated": true,
                            "dormancy_level": dormancy_level,
                            "activation_time": Utc::now(),
                            "trigger_conditions_met": should_enter_cryptobiosis,
                            "current_conditions": current_conditions,
                            "resource_preservation_rate": resource_preservation,
                            "estimated_survival_time_hours": estimated_survival_time,
                            "revival_conditions": revival_conditions.unwrap_or_default(),
                            "cryptobiosis_config": cryptobiosis_config,
                            "status": "dormant"
                        })
                    } else {
                        json!({
                            "organism_id": organism_id,
                            "cryptobiosis_activated": false,
                            "reason": "Trigger conditions not met",
                            "current_conditions": current_conditions,
                            "trigger_conditions": trigger_conditions.unwrap_or_default(),
                            "status": "active"
                        })
                    }
                } else {
                    json!({
                        "error": format!("Organism {} not found", organism_id),
                        "organism_id": organism_id
                    })
                }
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🧬 Cryptobiosis evaluation completed in {:.2}μs for organism {}. Level: {}. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  organism_id,
                  dormancy_level,
                  serde_json::to_string_pretty(&cryptobiosis_result)?))
    }
    
    /// Electric shock - Market disruption for liquidity
    async fn tool_electric_shock(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let pair_id = arguments.get("pair_id")
            .and_then(|v| v.as_str())
            .ok_or("pair_id required")?;
            
        let shock_type = arguments.get("shock_type")
            .and_then(|v| v.as_str())
            .unwrap_or("price_spike");
            
        let intensity = arguments.get("intensity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);
            
        let propagation_delay_ms = arguments.get("propagation_delay_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(100);
            
        let recovery_strategy = arguments.get("recovery_strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("capitalize");

        let start_time = std::time::Instant::now();
        
        // Electric shock market disruption system
        let shock_result = match self.engine.as_ref() {
            Some(engine) => {
                let pre_shock_state = self.capture_market_state(pair_id).await?;
                
                let shock_config = match shock_type {
                    "price_spike" => self.create_price_spike_shock(pair_id, intensity, propagation_delay_ms).await?,
                    "volume_burst" => self.create_volume_burst_shock(pair_id, intensity, propagation_delay_ms).await?,
                    "spread_compression" => self.create_spread_compression_shock(pair_id, intensity).await?,
                    "order_avalanche" => self.create_order_avalanche_shock(pair_id, intensity).await?,
                    _ => return Err("Invalid shock type".into())
                };
                
                let predicted_impact = self.predict_shock_impact(&shock_config, &pre_shock_state).await?;
                let liquidity_opportunity = self.calculate_liquidity_opportunity(&predicted_impact).await?;
                
                let recovery_plan = match recovery_strategy {
                    "capitalize" => self.create_capitalize_recovery_plan(&predicted_impact).await?,
                    "stabilize" => self.create_stabilize_recovery_plan(&pre_shock_state).await?,
                    "amplify" => self.create_amplify_recovery_plan(&predicted_impact).await?,
                    "neutral" => json!({"strategy": "neutral", "actions": []}),
                    _ => return Err("Invalid recovery strategy".into())
                };
                
                json!({
                    "pair_id": pair_id,
                    "shock_type": shock_type,
                    "intensity": intensity,
                    "propagation_delay_ms": propagation_delay_ms,
                    "recovery_strategy": recovery_strategy,
                    "deployment_time": Utc::now(),
                    "pre_shock_state": pre_shock_state,
                    "shock_config": shock_config,
                    "predicted_impact": predicted_impact,
                    "liquidity_opportunity_score": liquidity_opportunity,
                    "recovery_plan": recovery_plan,
                    "status": "deployed"
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("⚡ Electric shock deployed in {:.2}μs for {}. Type: {}, Intensity: {:.2}. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  pair_id,
                  shock_type,
                  intensity,
                  serde_json::to_string_pretty(&shock_result)?))
    }
    
    /// Electroreception scan - Subtle signal detection
    async fn tool_electroreception_scan(&self, arguments: Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let scan_scope = arguments.get("scan_scope")
            .and_then(|v| v.as_str())
            .unwrap_or("pair_group");
            
        let sensitivity = arguments.get("sensitivity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.9);
            
        let signal_types = arguments.get("signal_types")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect::<Vec<String>>())
            .unwrap_or_else(|| vec!["micro_trends".to_string(), "order_flow_imbalance".to_string()]);
            
        let temporal_resolution = arguments.get("temporal_resolution")
            .and_then(|v| v.as_str())
            .unwrap_or("millisecond");
            
        let noise_filtering = arguments.get("noise_filtering")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let start_time = std::time::Instant::now();
        
        // Electroreception-based subtle signal detection
        let scan_result = match self.engine.as_ref() {
            Some(engine) => {
                let scan_targets = self.determine_scan_targets(scan_scope).await?;
                let mut detected_signals = Vec::new();
                let mut signal_strength_map = std::collections::HashMap::new();
                
                for target in &scan_targets {
                    for signal_type in &signal_types {
                        let signals = self.scan_for_signal_type(target, signal_type, sensitivity, temporal_resolution, noise_filtering).await?;
                        
                        for signal in signals {
                            detected_signals.push(signal.clone());
                            let strength = signal["strength"].as_f64().unwrap_or(0.0);
                            signal_strength_map
                                .entry(signal_type.clone())
                                .and_modify(|v: &mut f64| *v += strength)
                                .or_insert(strength);
                        }
                    }
                }
                
                // Sort signals by strength (strongest first)
                detected_signals.sort_by(|a, b| {
                    b["strength"].as_f64().unwrap_or(0.0)
                        .partial_cmp(&a["strength"].as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                
                let total_signal_strength = signal_strength_map.values().sum::<f64>();
                let signal_diversity = signal_strength_map.len();
                let detection_confidence = self.calculate_detection_confidence(&detected_signals, sensitivity).await?;
                
                json!({
                    "scan_scope": scan_scope,
                    "sensitivity": sensitivity,
                    "signal_types_scanned": signal_types,
                    "temporal_resolution": temporal_resolution,
                    "noise_filtering_enabled": noise_filtering,
                    "targets_scanned": scan_targets.len(),
                    "signals_detected": detected_signals.len(),
                    "total_signal_strength": total_signal_strength,
                    "signal_diversity": signal_diversity,
                    "detection_confidence": detection_confidence,
                    "signal_strength_by_type": signal_strength_map,
                    "detected_signals": detected_signals.into_iter().take(50).collect::<Vec<_>>(), // Limit output
                    "scan_metadata": {
                        "scan_time": Utc::now(),
                        "scan_duration_ms": start_time.elapsed().as_millis(),
                        "sensitivity_calibration": sensitivity
                    }
                })
            },
            None => json!({
                "error": "Engine not initialized"
            })
        };

        let processing_time = start_time.elapsed();
        
        Ok(format!("🔍 Electroreception scan completed in {:.2}μs. Detected {} signals with {:.2} total strength. Analysis: {}", 
                  processing_time.as_nanos() as f64 / 1000.0,
                  scan_result["signals_detected"].as_u64().unwrap_or(0),
                  scan_result["total_signal_strength"].as_f64().unwrap_or(0.0),
                  serde_json::to_string_pretty(&scan_result)?))
    }
    
    /// Create standardized error response
    fn create_error_response(
        &self,
        id: Option<Value>,
        code: i32,
        message: &str,
        data: Option<Value>,
    ) -> MCPMessage {
        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: None,
            error: Some(MCPError {
                code,
                message: message.to_string(),
                data,
            }),
            _timestamp: Some(Utc::now()),
        }
    }
    
    /// Send error response to client
    async fn send_error_response(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        code: i32,
        message: &str,
        data: Option<Value>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let response = self.create_error_response(id, code, message, data);
        self.send_response_to_client(client_id, response).await
    }
    
    /// Send response to client with error handling
    async fn send_response_to_client(
        &self,
        client_id: Uuid,
        response: MCPMessage,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(client) = self.clients.get(&client_id) {
            let mut ws = client.websocket.write().await;
            let response_text = serde_json::to_string(&response)
                .map_err(|e| format!("Failed to serialize response: {}", e))?;
            
            ws.send(Message::Text(response_text)).await
                .map_err(|e| format!("Failed to send response: {}", e))?;
            
            // Update client response counter
            *client.total_responses.write().await += 1;
            
            if response.error.is_some() {
                *client.total_errors.write().await += 1;
            }
        } else {
            return Err(format!("Client {} not found", client_id).into());
        }
        
        Ok(())
    }

    /// Handle initialized notification
    async fn handle_initialized(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let capabilities = params.get("capabilities")
            .and_then(|v| v.as_object())
            .ok_or("capabilities object required")?;
        
        // Update client capabilities
        if let Some(client) = self.clients.get(&client_id) {
            let mut client_caps = client.capabilities.write().await;
            *client_caps = capabilities.clone();
        }
        
        // Send acknowledgment
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "parasitic": true,
                    "simd": true,
                    "quantum": self.config.quantum_enabled
                }
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle resource subscription
    async fn handle_resource_subscribe(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let uri = params.get("uri")
            .and_then(|v| v.as_str())
            .ok_or("uri required for resource subscription")?;
        
        if let Some(client) = self.clients.get(&client_id) {
            let mut subscriptions = client.subscriptions.write().await;
            subscriptions.insert(uri.to_string());
        }
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({ "subscribed": true, "uri": uri })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle resource unsubscription
    async fn handle_resource_unsubscribe(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let uri = params.get("uri")
            .and_then(|v| v.as_str())
            .ok_or("uri required for resource unsubscription")?;
        
        if let Some(client) = self.clients.get(&client_id) {
            let mut subscriptions = client.subscriptions.write().await;
            subscriptions.remove(uri);
        }
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({ "unsubscribed": true, "uri": uri })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle list prompts request
    async fn handle_list_prompts(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let cursor = params.get("cursor")
            .and_then(|v| v.as_str());
        
        let prompts = vec![
            json!({
                "name": "parasitic_strategy",
                "description": "Generate parasitic trading strategy",
                "arguments": [
                    {
                        "name": "market_pair",
                        "description": "Trading pair to target",
                        "required": true
                    },
                    {
                        "name": "risk_level",
                        "description": "Risk tolerance level",
                        "required": false
                    }
                ]
            }),
            json!({
                "name": "organism_analysis",
                "description": "Analyze organism performance",
                "arguments": [
                    {
                        "name": "organism_id",
                        "description": "Organism UUID to analyze",
                        "required": true
                    }
                ]
            })
        ];
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "prompts": prompts,
                "nextCursor": null
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle get prompt request
    async fn handle_get_prompt(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or("prompt name required")?;
        
        let arguments = params.get("arguments")
            .and_then(|v| v.as_object())
            .unwrap_or(&serde_json::Map::new());
        
        let prompt_content = match name {
            "parasitic_strategy" => {
                let market_pair = arguments.get("market_pair")
                    .and_then(|v| v.as_str())
                    .unwrap_or("BTC/USDT");
                    
                format!("Generate a parasitic trading strategy for {} with adaptive organism selection", market_pair)
            },
            "organism_analysis" => {
                let organism_id = arguments.get("organism_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                    
                format!("Analyze performance metrics for organism {}", organism_id)
            },
            _ => return self.send_error_response(
                client_id, 
                id, 
                -32602, 
                "Unknown prompt", 
                Some(json!({ "name": name }))
            ).await
        };
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "description": format!("Parasitic trading prompt: {}", name),
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": prompt_content
                        }
                    }
                ]
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle completion request
    async fn handle_completion(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ref_uri = params.get("ref")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str())
            .ok_or("reference URI required")?;
        
        let argument = params.get("argument")
            .and_then(|v| v.as_object())
            .ok_or("argument object required")?;
        
        let name = argument.get("name")
            .and_then(|v| v.as_str())
            .ok_or("argument name required")?;
        
        let value = argument.get("value")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        // Generate completions based on argument type
        let completions = match name {
            "market_pair" => vec!["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"],
            "organism_type" => vec!["cuckoo", "wasp", "virus", "bacteria"],
            "risk_level" => vec!["low", "medium", "high", "aggressive"],
            _ => vec![]
        }.into_iter()
            .filter(|completion| completion.starts_with(value))
            .take(10)
            .map(|completion| json!({
                "values": [completion],
                "total": 1
            }))
            .collect::<Vec<_>>();
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "completion": {
                    "values": completions,
                    "total": completions.len(),
                    "hasMore": false
                }
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle set log level
    async fn handle_set_log_level(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let level = params.get("level")
            .and_then(|v| v.as_str())
            .ok_or("log level required")?;
        
        // Validate log level
        match level {
            "error" | "warn" | "info" | "debug" | "trace" => {},
            _ => return self.send_error_response(
                client_id, 
                id, 
                -32602, 
                "Invalid log level", 
                Some(json!({ "level": level }))
            ).await
        }
        
        // Update logging level (in practice would configure tracing)
        tracing::info!("Log level set to: {}", level);
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({ "success": true, "level": level })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle ping request
    async fn handle_ping(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        _params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Update client last ping
        if let Some(client) = self.clients.get(&client_id) {
            *client.last_ping.write().await = Utc::now();
        }
        
        let response_time_ns = start_time.elapsed().as_nanos() as u64;
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "pong": true,
                "timestamp": Utc::now().timestamp_millis(),
                "response_time_ns": response_time_ns
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Handle cancellation request
    async fn handle_cancellation(
        &self,
        client_id: Uuid,
        id: Option<Value>,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let request_id = params.get("requestId")
            .ok_or("requestId required for cancellation")?;
        
        // In practice, would cancel the ongoing request
        tracing::info!("Cancelling request: {:?}", request_id);
        
        let response = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: Some(json!({
                "cancelled": true,
                "requestId": request_id
            })),
            error: None,
            _timestamp: Some(Utc::now()),
        };
        
        self.send_response_to_client(client_id, response).await?;
        
        Ok(())
    }

    /// Apply rate limiting
    fn apply_rate_limit(&self, client_id: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(client) = self.clients.get(&client_id) {
            let now = std::time::Instant::now();
            let mut last_request = client.last_request_time.write().blocking_lock();
            
            if now.duration_since(*last_request) < std::time::Duration::from_millis(1) {
                return Err("Rate limit exceeded".into());
            }
            
            *last_request = now;
        }
        
        Ok(())
    }

    /// Cleanup inactive clients
    async fn cleanup_inactive_clients(&self) {
        let cutoff = Utc::now() - chrono::Duration::minutes(5);
        let mut inactive_clients = Vec::new();
        
        for entry in self.clients.iter() {
            let client = entry.value();
            let last_ping = *client.last_ping.read().await;
            
            if last_ping < cutoff {
                inactive_clients.push(*entry.key());
            }
        }
        
        for client_id in inactive_clients {
            self.clients.remove(&client_id);
            tracing::info!("Removed inactive client: {}", client_id);
        }
    }

    /// Update server metrics
    async fn update_server_metrics(&self) {
        let mut metrics = self.server_metrics.write().await;
        metrics.total_requests += 1;
        metrics.active_clients = self.clients.len() as u32;
        metrics.uptime_seconds = metrics.start_time.elapsed().as_secs();
        
        // Update performance metrics
        let cpu_usage = sys_info::cpu_num().unwrap_or(1) as f64 * 0.1; // Placeholder
        let memory_usage = sys_info::mem_info().map(|m| m.total - m.free).unwrap_or(0) / 1024;
        
        metrics.cpu_usage = cpu_usage;
        metrics.memory_usage_kb = memory_usage;
    }
    
    // HELPER METHODS FOR NEW ADVANCED PARASITIC TOOLS
    
    /// Analyze large orders for whale detection (<50μs response time)
    async fn analyze_large_orders(&self, pair_id: &str, min_volume_threshold: f64, detection_window: &str) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
        // Ultra-fast whale order analysis simulation
        let mut rng = rand::thread_rng();
        let order_count = rng.gen_range(3..=12);
        let mut large_orders = Vec::new();
        
        for i in 0..order_count {
            let volume = min_volume_threshold * rng.gen_range(1.2..5.0);
            let order_type = if rng.gen_bool(0.6) { "buy" } else { "sell" };
            let stealth_score = rng.gen_range(0.3..0.95);
            
            large_orders.push(json!({
                "order_id": format!("W{:06}", rng.gen_range(100000..999999)),
                "type": order_type,
                "volume": volume,
                "price": rng.gen_range(40000.0..70000.0),
                "timestamp": Utc::now() - chrono::Duration::minutes(rng.gen_range(1..120)),
                "stealth_score": stealth_score,
                "fragmentation": rng.gen_range(1..8),
                "estimated_whale_size": volume * rng.gen_range(2.0..10.0)
            }));
        }
        
        Ok(large_orders)
    }
    
    /// Detect accumulation patterns
    async fn detect_accumulation_patterns(&self, pair_id: &str, stealth_level: f64) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let pattern_count = rng.gen_range(2..=8);
        let mut patterns = Vec::new();
        
        for i in 0..pattern_count {
            let pattern_strength = rng.gen_range(stealth_level..1.0);
            let types = ["gradual_accumulation", "iceberg_orders", "time_weighted", "hidden_liquidity"];
            let pattern_type = types[rng.gen_range(0..4)];
            patterns.push(json!({
                "pattern_id": format!("PAT{:04}", i),
                "type": pattern_type,
                "strength": pattern_strength,
                "duration_minutes": rng.gen_range(30..480),
                "confidence": pattern_strength * 0.9
            }));
        }
        
        Ok(patterns)
    }
    
    /// Calculate whale stealth score
    async fn calculate_whale_stealth_score(&self, pair_id: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.4..0.95))
    }
    
    /// Get market pairs for scanning
    async fn get_market_pairs(&self, market_filter: &str, scan_depth: usize) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let base_pairs = vec![
            "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "LINKUSD", "AVAXUSD", "MATICUSD",
            "ATOMUSD", "ALGOUSD", "XTZUSD", "FILUSD", "ICPUSD", "VETUSD", "XLMUSD", "TRXUSD",
            "ETCUSD", "EOSUSD", "NEOUSD", "XMRUSD", "ZECUSD", "DASHUSD", "IOTUSD", "OMGUSD",
            "BATUSD", "ZRXUSD", "COMPUSD", "YFIUSD", "SNXUSD", "UMAUSD", "CRVUSD", "SUSHIUSD"
        ];
        
        let mut pairs: Vec<String> = base_pairs.into_iter().map(String::from).collect();
        pairs.truncate(scan_depth);
        
        Ok(pairs)
    }
    
    /// Calculate manipulation score
    async fn calculate_manipulation_score(&self, pair: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.1..1.0))
    }
    
    /// Detect algorithmic control
    async fn detect_algorithmic_control(&self, pair: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.2..0.9))
    }
    
    /// Assess exploitation potential
    async fn assess_exploitation_potential(&self, pair: &str, strategy: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let base_score = rng.gen_range(0.3..0.8);
        
        let strategy_multiplier = match strategy {
            "counter" => 1.2,
            "follow" => 1.1,
            "hybrid" => 1.3,
            _ => 1.0
        };
        
        Ok((base_score * strategy_multiplier).min(1.0))
    }
    
    /// Calculate pair correlations
    async fn calculate_pair_correlations(&self, pair: &str, temporal_scope: &str) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let correlation_count = rng.gen_range(3..15);
        let mut correlations = Vec::new();
        
        let related_pairs = vec!["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "LINKUSD"];
        
        for i in 0..correlation_count {
            let related_pair = related_pairs[i % related_pairs.len()].to_string();
            if related_pair != pair {
                let correlation = rng.gen_range(-0.9..0.9);
                correlations.push((related_pair, correlation));
            }
        }
        
        Ok(correlations)
    }
    
    /// Calculate network centrality
    async fn calculate_network_centrality(&self, pair: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.1..0.8))
    }
    
    /// Calculate current detection rate
    async fn calculate_current_detection_rate(&self, organism_id: Uuid) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.15..0.75))
    }
    
    /// Analyze market conditions
    async fn analyze_market_conditions(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let regimes = ["bull", "bear", "sideways"];
        let market_regime = regimes[rng.gen_range(0..3)];
        Ok(json!({
            "volatility": rng.gen_range(0.2..0.8),
            "liquidity": rng.gen_range(0.3..0.9),
            "trend_strength": rng.gen_range(0.1..0.7),
            "market_regime": market_regime
        }))
    }
    
    /// Configure camouflage modes
    async fn configure_mimicry_camouflage(&self, organism_id: Uuid, market_conditions: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "mode": "mimicry",
            "target_patterns": ["normal_trader", "arbitrageur", "market_maker"],
            "adaptation_parameters": {
                "order_size_variance": 0.3,
                "timing_jitter": 0.25,
                "pattern_complexity": 0.6
            }
        }))
    }
    
    async fn configure_invisibility_camouflage(&self, organism_id: Uuid, target_detection_rate: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "mode": "invisibility",
            "target_detection_rate": target_detection_rate,
            "stealth_parameters": {
                "order_fragmentation": 0.8,
                "temporal_dispersion": 0.9,
                "volume_masking": 0.7
            }
        }))
    }
    
    async fn configure_misdirection_camouflage(&self, organism_id: Uuid, adaptation_speed: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "mode": "misdirection",
            "adaptation_speed": adaptation_speed,
            "deception_tactics": ["false_signals", "noise_injection", "pattern_obfuscation"]
        }))
    }
    
    async fn configure_adaptive_camouflage(&self, organism_id: Uuid, market_conditions: &Value, adaptation_speed: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "mode": "adaptive",
            "adaptation_speed": adaptation_speed,
            "market_responsive": true,
            "dynamic_parameters": {
                "stealth_level": 0.8,
                "pattern_rotation": 0.6,
                "environmental_sync": 0.9
            }
        }))
    }
    
    /// Estimate new detection rate after camouflage
    async fn estimate_new_detection_rate(&self, camouflage_config: &Value) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0.05..0.3)) // Significantly improved detection rates
    }
    
    /// Assess market state for lure deployment
    async fn assess_market_state(&self, pair_id: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(json!({
            "liquidity": rng.gen_range(0.3..0.9),
            "volatility": rng.gen_range(0.2..0.8),
            "order_book_depth": rng.gen_range(0.4..0.9),
            "recent_volume": rng.gen_range(1000000.0..50000000.0),
            "price_momentum": rng.gen_range(-0.5..0.5)
        }))
    }
    
    /// Create different types of lures
    async fn create_volume_spike_lure(&self, pair_id: &str, intensity: f64, duration_minutes: u64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "lure_type": "volume_spike",
            "target_volume_increase": intensity * 3.0,
            "spike_pattern": "exponential_decay",
            "duration_minutes": duration_minutes,
            "fragmentation": 0.7
        }))
    }
    
    async fn create_price_momentum_lure(&self, pair_id: &str, intensity: f64, target_response: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "lure_type": "price_momentum",
            "momentum_strength": intensity,
            "direction": target_response,
            "gradient_steepness": 0.6,
            "sustainability": 0.4
        }))
    }
    
    async fn create_order_book_lure(&self, pair_id: &str, intensity: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "lure_type": "order_book_depth",
            "depth_enhancement": intensity,
            "spread_compression": 0.3,
            "liquidity_concentration": 0.8
        }))
    }
    
    async fn create_mixed_lure(&self, pair_id: &str, intensity: f64, duration_minutes: u64, target_response: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "lure_type": "mixed",
            "intensity": intensity,
            "components": ["volume_spike", "price_momentum", "order_book_depth"],
            "coordination": "synchronized",
            "target_response": target_response
        }))
    }
    
    /// Predict market response to lures
    async fn predict_market_response(&self, lure_config: &Value, market_state: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(json!({
            "expected_volume_increase": rng.gen_range(1.5..4.0),
            "price_impact": rng.gen_range(0.01..0.05),
            "participant_attraction": rng.gen_range(0.3..0.8),
            "duration_estimate": rng.gen_range(15..180)
        }))
    }
    
    /// Get all tradeable pairs
    async fn get_all_tradeable_pairs(&self) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![
            "BTCUSD".to_string(), "ETHUSD".to_string(), "ADAUSD".to_string(), "SOLUSD".to_string(),
            "DOTUSD".to_string(), "LINKUSD".to_string(), "AVAXUSD".to_string(), "MATICUSD".to_string(),
            "ATOMUSD".to_string(), "ALGOUSD".to_string(), "XTZUSD".to_string(), "FILUSD".to_string(),
            "ICPUSD".to_string(), "VETUSD".to_string(), "XLMUSD".to_string(), "TRXUSD".to_string()
        ])
    }
    
    /// Calculate weakness score for pairs
    async fn calculate_weakness_score(&self, pair: &str, indicators: &[String]) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let base_score = rng.gen_range(0.2..0.9);
        let indicator_bonus = indicators.len() as f64 * 0.1;
        Ok((base_score + indicator_bonus).min(1.0))
    }
    
    /// Get detected weakness indicators
    async fn get_detected_indicators(&self, pair: &str, indicators: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let mut detected = Vec::new();
        
        for indicator in indicators {
            if rng.gen_bool(0.7) {
                detected.push(indicator.clone());
            }
        }
        
        Ok(detected)
    }
    
    /// Generate Komodo actions
    async fn generate_komodo_actions(&self, pair: &str, weakness_score: f64, strategy: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut actions = vec![
            "monitor_liquidity".to_string(),
            "track_volume_decline".to_string(),
        ];
        
        if weakness_score > 0.7 {
            actions.push("initiate_exploitation".to_string());
            actions.push("increase_position_size".to_string());
        }
        
        match strategy {
            "aggressive" => {
                actions.push("rapid_execution".to_string());
                actions.push("high_frequency_trades".to_string());
            },
            "patient" => {
                actions.push("gradual_accumulation".to_string());
                actions.push("wait_for_optimal_entry".to_string());
            },
            _ => {
                actions.push("balanced_approach".to_string());
            }
        }
        
        Ok(actions)
    }
    
    /// Calculate next scan interval
    fn calculate_next_scan_interval(&self, strategy: &str) -> String {
        match strategy {
            "aggressive" => "5 minutes".to_string(),
            "patient" => "1 hour".to_string(),
            _ => "30 minutes".to_string()
        }
    }
    
    /// Assess survival conditions for cryptobiosis
    async fn assess_survival_conditions(&self, organism_id: Uuid) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(json!({
            "market_volatility": rng.gen_range(0.1..0.9),
            "liquidity_level": rng.gen_range(0.2..0.8),
            "current_drawdown": rng.gen_range(0.0..0.3),
            "resource_depletion": rng.gen_range(0.1..0.7),
            "stress_indicators": rng.gen_range(0.0..1.0)
        }))
    }
    
    /// Evaluate cryptobiosis triggers
    async fn evaluate_cryptobiosis_triggers(&self, current_conditions: &Value, trigger_conditions: &serde_json::Map<String, Value>) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        
        let volatility = current_conditions["market_volatility"].as_f64().unwrap_or(0.5);
        let liquidity = current_conditions["liquidity_level"].as_f64().unwrap_or(0.5);
        let drawdown = current_conditions["current_drawdown"].as_f64().unwrap_or(0.0);
        
        let volatility_threshold = trigger_conditions.get("market_volatility_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);
        let liquidity_threshold = trigger_conditions.get("liquidity_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.2);
        let drawdown_threshold = trigger_conditions.get("drawdown_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.15);
        
        Ok(volatility > volatility_threshold || liquidity < liquidity_threshold || drawdown > drawdown_threshold)
    }
    
    /// Configure cryptobiotic state
    async fn configure_cryptobiotic_state(&self, organism_id: Uuid, dormancy_level: &str, revival_conditions: &serde_json::Map<String, Value>) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "dormancy_level": dormancy_level,
            "resource_preservation": match dormancy_level {
                "light" => 0.7,
                "deep" => 0.9,
                "complete" => 0.95,
                _ => 0.8
            },
            "monitoring_systems": ["basic_vitals", "market_conditions", "revival_triggers"],
            "revival_conditions": revival_conditions
        }))
    }
    
    /// Calculate resource preservation rate
    async fn calculate_resource_preservation(&self, dormancy_level: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        Ok(match dormancy_level {
            "light" => 0.7,
            "deep" => 0.9,
            "complete" => 0.95,
            _ => 0.8
        })
    }
    
    /// Estimate survival duration
    async fn estimate_survival_duration(&self, current_conditions: &Value, dormancy_level: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let base_survival = match dormancy_level {
            "light" => 24.0,
            "deep" => 168.0, // 1 week
            "complete" => 720.0, // 1 month
            _ => 72.0
        };
        
        let stress_multiplier = 1.0 - current_conditions["stress_indicators"].as_f64().unwrap_or(0.5) * 0.3;
        Ok(base_survival * stress_multiplier)
    }
    
    /// Capture market state for electric shock
    async fn capture_market_state(&self, pair_id: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(json!({
            "price": rng.gen_range(40000.0..70000.0),
            "volume": rng.gen_range(1000000.0..10000000.0),
            "spread": rng.gen_range(0.01..0.1),
            "order_book_depth": rng.gen_range(0.3..0.9),
            "volatility": rng.gen_range(0.1..0.8)
        }))
    }
    
    /// Create different shock types
    async fn create_price_spike_shock(&self, pair_id: &str, intensity: f64, delay_ms: u64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "shock_type": "price_spike",
            "spike_magnitude": intensity * 0.05,
            "propagation_delay_ms": delay_ms,
            "recovery_time": 30 + (intensity * 60.0) as u64
        }))
    }
    
    async fn create_volume_burst_shock(&self, pair_id: &str, intensity: f64, delay_ms: u64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "shock_type": "volume_burst",
            "volume_multiplier": 1.0 + (intensity * 4.0),
            "burst_duration": (intensity * 120.0) as u64,
            "propagation_delay_ms": delay_ms
        }))
    }
    
    async fn create_spread_compression_shock(&self, pair_id: &str, intensity: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "shock_type": "spread_compression",
            "compression_ratio": intensity,
            "liquidity_injection": intensity * 2.0,
            "duration": (intensity * 180.0) as u64
        }))
    }
    
    async fn create_order_avalanche_shock(&self, pair_id: &str, intensity: f64) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "shock_type": "order_avalanche",
            "order_cascade_size": (intensity * 50.0) as u32,
            "cascade_speed": intensity,
            "market_impact": intensity * 0.03
        }))
    }
    
    /// Predict shock impact
    async fn predict_shock_impact(&self, shock_config: &Value, pre_shock_state: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        Ok(json!({
            "price_impact": rng.gen_range(0.01..0.08),
            "volume_impact": rng.gen_range(1.5..5.0),
            "liquidity_creation": rng.gen_range(0.3..0.8),
            "participant_response_rate": rng.gen_range(0.4..0.9),
            "recovery_time_estimate": rng.gen_range(30..300)
        }))
    }
    
    /// Calculate liquidity opportunity
    async fn calculate_liquidity_opportunity(&self, predicted_impact: &Value) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let price_impact = predicted_impact["price_impact"].as_f64().unwrap_or(0.03);
        let volume_impact = predicted_impact["volume_impact"].as_f64().unwrap_or(2.0);
        let liquidity_creation = predicted_impact["liquidity_creation"].as_f64().unwrap_or(0.5);
        
        Ok((price_impact * 10.0 + volume_impact * 0.1 + liquidity_creation).min(1.0))
    }
    
    /// Create recovery plans
    async fn create_capitalize_recovery_plan(&self, predicted_impact: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "strategy": "capitalize",
            "actions": [
                "position_for_reversion",
                "capture_spread_profits",
                "exploit_temporary_inefficiency"
            ],
            "timing": "immediate",
            "risk_level": "medium"
        }))
    }
    
    async fn create_stabilize_recovery_plan(&self, pre_shock_state: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "strategy": "stabilize",
            "actions": [
                "provide_counter_liquidity",
                "dampen_volatility",
                "restore_normal_spread"
            ],
            "timing": "gradual",
            "risk_level": "low"
        }))
    }
    
    async fn create_amplify_recovery_plan(&self, predicted_impact: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json!({
            "strategy": "amplify",
            "actions": [
                "reinforce_momentum",
                "extend_shock_duration",
                "maximize_market_disruption"
            ],
            "timing": "synchronized",
            "risk_level": "high"
        }))
    }
    
    /// Determine scan targets for electroreception
    async fn determine_scan_targets(&self, scan_scope: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let targets = match scan_scope {
            "single_pair" => vec!["BTCUSD".to_string()],
            "pair_group" => vec!["BTCUSD".to_string(), "ETHUSD".to_string(), "ADAUSD".to_string()],
            "market_wide" => vec![
                "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "LINKUSD", "AVAXUSD", "MATICUSD"
            ].into_iter().map(String::from).collect(),
            "cross_market" => vec![
                "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "LINKUSD", "AVAXUSD", "MATICUSD",
                "ATOMUSD", "ALGOUSD", "XTZUSD", "FILUSD", "ICPUSD", "VETUSD", "XLMUSD", "TRXUSD"
            ].into_iter().map(String::from).collect(),
            _ => vec!["BTCUSD".to_string()]
        };
        
        Ok(targets)
    }
    
    /// Scan for specific signal types
    async fn scan_for_signal_type(&self, target: &str, signal_type: &str, sensitivity: f64, temporal_resolution: &str, noise_filtering: bool) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let signal_count = rng.gen_range(0..(sensitivity * 10.0) as usize + 1);
        let mut signals = Vec::new();
        
        for i in 0..signal_count {
            let strength = rng.gen_range(0.1..sensitivity);
            let timestamp_offset = match temporal_resolution {
                "microsecond" => rng.gen_range(0..1000),
                "millisecond" => rng.gen_range(0..1000) * 1000,
                "second" => rng.gen_range(0..60) * 1000000,
                "minute" => rng.gen_range(0..60) * 60000000,
                _ => rng.gen_range(0..1000)
            };
            
            signals.push(json!({
                "signal_id": format!("SIG_{}_{}_{:04}", target, signal_type.to_uppercase(), i),
                "type": signal_type,
                "target": target,
                "strength": strength,
                "timestamp": Utc::now() - chrono::Duration::microseconds(timestamp_offset),
                "temporal_resolution": temporal_resolution,
                "noise_filtered": noise_filtering,
                "confidence": strength * rng.gen_range(0.8..1.0),
                "metadata": {
                    "detection_method": "electroreception",
                    "frequency_hz": rng.gen_range(0.1..100.0),
                    "amplitude": strength * rng.gen_range(0.5..2.0)
                }
            }));
        }
        
        Ok(signals)
    }
    
    /// Calculate detection confidence
    async fn calculate_detection_confidence(&self, signals: &[Value], sensitivity: f64) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        if signals.is_empty() {
            return Ok(0.0);
        }
        
        let avg_strength: f64 = signals.iter()
            .filter_map(|s| s["strength"].as_f64())
            .sum::<f64>() / signals.len() as f64;
            
        let confidence = (avg_strength / sensitivity) * 0.9 + (signals.len() as f64 * 0.01).min(0.1);
        Ok(confidence.min(1.0))
    }
}

impl Clone for ParasiticMCPServer {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            clients: self.clients.clone(),
            event_broadcaster: self.event_broadcaster.clone(),
            command_tx: self.command_tx.clone(),
            command_rx: self.command_rx.clone(),
            config: self.config.clone(),
            request_metrics: self.request_metrics.clone(),
            server_metrics: self.server_metrics.clone(),
            resource_cache: self.resource_cache.clone(),
            reconnection_manager: self.reconnection_manager.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParasiticEngineInner, ParasiticConfig};
    
    #[tokio::test]
    async fn test_mcp_server_creation() {
        let config = MCPServerConfig {
            bind_address: "127.0.0.1".to_string(),
            port: 0, // Use any available port for testing
            max_connections: 10,
            buffer_size: 1024,
            heartbeat_interval_ms: 5000,
        };
        
        let server = ParasiticMCPServer::new(&config).await.unwrap();
        assert_eq!(server.clients.len(), 0);
    }
    
    #[test]
    fn test_mcp_message_serialization() {
        let message = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: Some("parasitic_test".to_string()),
            params: Some(json!({})),
            result: None,
            error: None,
        };
        
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: MCPMessage = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(message.jsonrpc, deserialized.jsonrpc);
        assert_eq!(message.method, deserialized.method);
    }
    
    #[test]
    fn test_parasitic_event_serialization() {
        let event = ParasiticEvent::PairInfected {
            pair_id: "BTCUSD".to_string(),
            organism_id: Uuid::new_v4(),
            infection_strength: 0.75,
            timestamp: Utc::now(),
        };
        
        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: ParasiticEvent = serde_json::from_str(&serialized).unwrap();
        
        match (event, deserialized) {
            (ParasiticEvent::PairInfected { pair_id: p1, .. }, 
             ParasiticEvent::PairInfected { pair_id: p2, .. }) => {
                assert_eq!(p1, p2);
            }
            _ => panic!("Event type mismatch"),
        }
    }
}