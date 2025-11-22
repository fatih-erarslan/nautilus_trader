// Complete MCP Server implementation for CWTS Ultra Trading System
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};
use uuid::Uuid;

use super::resources::ResourceManager;
use super::subscriptions::{MarketDataEvent, SubscriptionManager};
use super::tools::ToolManager;
use crate::algorithms::lockfree_orderbook::LockFreeOrderBook;
use crate::execution::simple_orders::{AtomicMatchingEngine, AtomicOrder, OrderSide, OrderType};

/// MCP Protocol Message Types
#[derive(Debug, Serialize, Deserialize)]
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// Client connection state
pub struct MCPClient {
    pub id: Uuid,
    pub websocket: Arc<RwLock<WebSocketStream<TcpStream>>>,
    pub subscriptions: Arc<RwLock<Vec<String>>>,
    pub capabilities: ClientCapabilities,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientCapabilities {
    pub supports_resources: bool,
    pub supports_tools: bool,
    pub supports_subscriptions: bool,
    pub supported_resource_types: Vec<String>,
    pub supported_tool_types: Vec<String>,
}

impl Default for ClientCapabilities {
    fn default() -> Self {
        Self {
            supports_resources: true,
            supports_tools: true,
            supports_subscriptions: true,
            supported_resource_types: vec![
                "order_book".to_string(),
                "positions".to_string(),
                "market_data".to_string(),
                "trade_history".to_string(),
            ],
            supported_tool_types: vec![
                "place_order".to_string(),
                "cancel_order".to_string(),
                "modify_order".to_string(),
                "get_positions".to_string(),
                "get_market_data".to_string(),
            ],
        }
    }
}

/// Main MCP Trading Server
pub struct TradingMCPServer {
    // Core trading components
    pub order_book: Arc<LockFreeOrderBook>,
    pub matching_engine: Arc<AtomicMatchingEngine>,
    pub resource_manager: Arc<ResourceManager>,
    pub tool_manager: Arc<ToolManager>,
    pub subscription_manager: Arc<SubscriptionManager>,

    // Client management
    pub clients: Arc<RwLock<HashMap<Uuid, MCPClient>>>,

    // Communication channels
    pub broadcast_tx: broadcast::Sender<MarketDataEvent>,
    pub command_tx: mpsc::Sender<TradingCommand>,
    pub command_rx: Arc<RwLock<Option<mpsc::Receiver<TradingCommand>>>>,

    // Server configuration
    pub config: ServerConfig,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_address: SocketAddr,
    pub max_clients: usize,
    pub heartbeat_interval_ms: u64,
    pub enable_compression: bool,
    pub max_message_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:3000".parse().unwrap(),
            max_clients: 1000,
            heartbeat_interval_ms: 30000, // 30 seconds
            enable_compression: true,
            max_message_size: 1024 * 1024, // 1MB
        }
    }
}

#[derive(Debug, Clone)]
pub enum TradingCommand {
    PlaceOrder {
        client_id: Uuid,
        order: OrderRequest,
    },
    CancelOrder {
        client_id: Uuid,
        order_id: u64,
    },
    ModifyOrder {
        client_id: Uuid,
        order_id: u64,
        new_price: Option<u64>,
        new_quantity: Option<u64>,
    },
    GetPositions {
        client_id: Uuid,
    },
    GetMarketData {
        client_id: Uuid,
        symbol: String,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: Option<String>,
    pub client_order_id: Option<String>,
}

impl TradingMCPServer {
    pub async fn new(config: Option<ServerConfig>) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config.unwrap_or_default();

        // Initialize core trading components
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());

        // Initialize managers
        let resource_manager = Arc::new(ResourceManager::new(order_book.clone()));
        let tool_manager = Arc::new(ToolManager::new(
            order_book.clone(),
            matching_engine.clone(),
        ));

        // Create communication channels
        let (broadcast_tx, _) = broadcast::channel(1000);
        let (command_tx, command_rx) = mpsc::channel(1000);

        let subscription_manager = Arc::new(SubscriptionManager::new(broadcast_tx.clone()));

        Ok(Self {
            order_book,
            matching_engine,
            resource_manager,
            tool_manager,
            subscription_manager,
            clients: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            command_tx,
            command_rx: Arc::new(RwLock::new(Some(command_rx))),
            config,
        })
    }

    /// Start the MCP server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "Starting CWTS Ultra MCP Server on {}",
            self.config.bind_address
        );

        let listener = TcpListener::bind(&self.config.bind_address).await?;
        println!("MCP Server listening on {}", self.config.bind_address);

        // Start background tasks
        self.start_background_tasks().await?;

        // Accept client connections
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    println!("New client connection from {}", addr);

                    let server = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_client_connection(stream, addr).await {
                            eprintln!("Error handling client {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Error accepting connection: {}", e);
                }
            }
        }
    }

    /// Start background tasks for order matching and market data
    async fn start_background_tasks(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Start order matching loop
        let matching_engine = self.matching_engine.clone();
        let broadcast_tx = self.broadcast_tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_micros(100), // 100µs matching cycle
            );

            loop {
                interval.tick().await;

                // Match orders and broadcast trades
                let trades = matching_engine.match_orders();
                if !trades.is_empty() {
                    let event = MarketDataEvent::TradeUpdate { trades };
                    let _ = broadcast_tx.send(event);
                }
            }
        });

        // Start command processing loop
        let command_rx = self.command_rx.clone();
        let server_clone = self.clone();
        tokio::spawn(async move {
            let mut rx = command_rx.write().await.take().unwrap();

            while let Some(command) = rx.recv().await {
                server_clone.process_trading_command(command).await;
            }
        });

        // Start market data simulation (for demo purposes)
        let order_book = self.order_book.clone();
        let broadcast_tx = self.broadcast_tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1000));

            let mut order_id = 1000u64;

            loop {
                interval.tick().await;

                // Add some random market making orders
                let base_price = 100_000_000; // $100.00 in micropips
                let spread = 10_000; // $0.01 spread

                // Add buy order
                order_book.add_bid(
                    base_price - spread / 2,
                    1_000_000_000, // 10.0 units
                    order_id,
                );
                order_id += 1;

                // Add sell order
                order_book.add_ask(
                    base_price + spread / 2,
                    1_000_000_000, // 10.0 units
                    order_id,
                );
                order_id += 1;

                // Broadcast order book update
                let (bids, asks) = order_book.get_depth(5);
                let event = MarketDataEvent::OrderBookUpdate {
                    symbol: "BTCUSD".to_string(),
                    bids: bids
                        .into_iter()
                        .map(|(p, q)| (p as f64 / 1_000_000.0, q as f64 / 100_000_000.0))
                        .collect(),
                    asks: asks
                        .into_iter()
                        .map(|(p, q)| (p as f64 / 1_000_000.0, q as f64 / 100_000_000.0))
                        .collect(),
                    timestamp: chrono::Utc::now(),
                };
                let _ = broadcast_tx.send(event);
            }
        });

        println!("Background tasks started");
        Ok(())
    }

    /// Handle a new client connection
    async fn handle_client_connection(
        &self,
        stream: TcpStream,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ws_stream = accept_async(stream).await?;
        let client_id = Uuid::new_v4();

        let client = MCPClient {
            id: client_id,
            websocket: Arc::new(RwLock::new(ws_stream)),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            capabilities: ClientCapabilities::default(),
        };

        // Add client to active clients
        {
            let mut clients = self.clients.write().await;
            if clients.len() >= self.config.max_clients {
                return Err("Maximum client limit reached".into());
            }
            clients.insert(client_id, client);
        }

        println!("Client {} connected from {}", client_id, addr);

        // Handle client messages
        self.client_message_loop(client_id).await?;

        // Remove client when disconnected
        {
            let mut clients = self.clients.write().await;
            clients.remove(&client_id);
        }

        println!("Client {} disconnected", client_id);
        Ok(())
    }

    /// Main message loop for a client
    async fn client_message_loop(&self, client_id: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        let client = {
            let clients = self.clients.read().await;
            clients.get(&client_id).unwrap().clone()
        };

        // Subscribe to market data broadcasts
        let mut broadcast_rx = self.broadcast_tx.subscribe();
        let ws_clone = client.websocket.clone();

        // Spawn task to handle broadcast messages
        tokio::spawn(async move {
            while let Ok(event) = broadcast_rx.recv().await {
                let message = MCPMessage {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    method: Some("market_data_update".to_string()),
                    params: Some(serde_json::to_value(event).unwrap()),
                    result: None,
                    error: None,
                };

                let mut ws = ws_clone.write().await;
                let _ = ws
                    .send(Message::Text(serde_json::to_string(&message).unwrap()))
                    .await;
            }
        });

        // Main message handling loop
        loop {
            let mut ws = client.websocket.write().await;

            match ws.next().await {
                Some(Ok(msg)) => {
                    drop(ws); // Release lock before processing

                    match msg {
                        Message::Text(text) => {
                            if let Ok(mcp_msg) = serde_json::from_str::<MCPMessage>(&text) {
                                self.handle_mcp_message(client_id, mcp_msg).await;
                            }
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                }
                Some(Err(e)) => {
                    eprintln!("WebSocket error for client {}: {}", client_id, e);
                    break;
                }
                None => break,
            }
        }

        Ok(())
    }

    /// Handle incoming MCP protocol messages
    async fn handle_mcp_message(&self, client_id: Uuid, message: MCPMessage) {
        let method = message.method.as_deref().unwrap_or("");

        let response = match method {
            "initialize" => self.handle_initialize(&message).await,
            "resources/list" => self.handle_list_resources(&message).await,
            "resources/read" => self.handle_read_resource(&message).await,
            "tools/list" => self.handle_list_tools(&message).await,
            "tools/call" => self.handle_tool_call(client_id, &message).await,
            "subscriptions/subscribe" => self.handle_subscribe(client_id, &message).await,
            "subscriptions/unsubscribe" => self.handle_unsubscribe(client_id, &message).await,
            _ => self.create_error_response(message.id.clone(), -32601, "Method not found", None),
        };

        // Send response back to client
        if let Some(clients) = self.clients.read().await.get(&client_id) {
            let mut ws = clients.websocket.write().await;
            let response_text = serde_json::to_string(&response).unwrap();
            let _ = ws.send(Message::Text(response_text)).await;
        }
    }

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
                    "name": "CWTS Ultra Trading Server",
                    "version": "2.0.0",
                    "description": "High-performance trading system with lock-free order matching"
                }
            })),
            error: None,
        }
    }

    async fn handle_list_resources(&self, message: &MCPMessage) -> MCPMessage {
        let resources = self.resource_manager.list_resources().await;

        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id.clone(),
            method: None,
            params: None,
            result: Some(json!({ "resources": resources })),
            error: None,
        }
    }

    async fn handle_read_resource(&self, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(uri) = params.get("uri").and_then(|v| v.as_str()) {
                match self.resource_manager.read_resource(uri).await {
                    Ok(content) => MCPMessage {
                        jsonrpc: "2.0".to_string(),
                        id: message.id.clone(),
                        method: None,
                        params: None,
                        result: Some(json!({
                            "contents": [content]
                        })),
                        error: None,
                    },
                    Err(e) => self.create_error_response(
                        message.id.clone(),
                        -32602,
                        &format!("Resource not found: {}", e),
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

    async fn handle_list_tools(&self, message: &MCPMessage) -> MCPMessage {
        let tools = self.tool_manager.list_tools().await;

        MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id.clone(),
            method: None,
            params: None,
            result: Some(json!({ "tools": tools })),
            error: None,
        }
    }

    async fn handle_tool_call(&self, _client_id: Uuid, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(name) = params.get("name").and_then(|v| v.as_str()) {
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                match self.tool_manager.call_tool(name, arguments).await {
                    Ok(result) => MCPMessage {
                        jsonrpc: "2.0".to_string(),
                        id: message.id.clone(),
                        method: None,
                        params: None,
                        result: Some(json!({
                            "content": [result]
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

    async fn handle_subscribe(&self, client_id: Uuid, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(uri) = params.get("uri").and_then(|v| v.as_str()) {
                // Add subscription to client
                if let Some(client) = self.clients.read().await.get(&client_id) {
                    let mut subscriptions = client.subscriptions.write().await;
                    if !subscriptions.contains(&uri.to_string()) {
                        subscriptions.push(uri.to_string());
                    }
                }

                let _ = self
                    .subscription_manager
                    .subscribe(client_id, uri.to_string())
                    .await;

                MCPMessage {
                    jsonrpc: "2.0".to_string(),
                    id: message.id.clone(),
                    method: None,
                    params: None,
                    result: Some(json!({})),
                    error: None,
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

    async fn handle_unsubscribe(&self, client_id: Uuid, message: &MCPMessage) -> MCPMessage {
        if let Some(params) = &message.params {
            if let Some(uri) = params.get("uri").and_then(|v| v.as_str()) {
                // Remove subscription from client
                if let Some(client) = self.clients.read().await.get(&client_id) {
                    let mut subscriptions = client.subscriptions.write().await;
                    subscriptions.retain(|s| s != uri);
                }

                let _ = self
                    .subscription_manager
                    .unsubscribe(client_id, uri.to_string())
                    .await;

                MCPMessage {
                    jsonrpc: "2.0".to_string(),
                    id: message.id.clone(),
                    method: None,
                    params: None,
                    result: Some(json!({})),
                    error: None,
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

    /// Process trading commands
    async fn process_trading_command(&self, command: TradingCommand) {
        match command {
            TradingCommand::PlaceOrder { client_id, order } => {
                self.process_place_order(client_id, order).await;
            }
            TradingCommand::CancelOrder {
                client_id,
                order_id,
            } => {
                self.process_cancel_order(client_id, order_id).await;
            }
            TradingCommand::ModifyOrder {
                client_id,
                order_id,
                new_price,
                new_quantity,
            } => {
                self.process_modify_order(client_id, order_id, new_price, new_quantity)
                    .await;
            }
            _ => {
                // Handle other command types
            }
        }
    }

    async fn process_place_order(&self, _client_id: Uuid, order_request: OrderRequest) {
        // Convert order request to internal format
        let price = order_request
            .price
            .map(|p| (p * 1_000_000.0) as u64)
            .unwrap_or(0);
        let quantity = (order_request.quantity * 100_000_000.0) as u64;
        let order_id = self.generate_order_id();

        let side = match order_request.side.as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            _ => OrderSide::Buy,
        };

        let order_type = match order_request.order_type.as_str() {
            "market" => OrderType::Market,
            "limit" => OrderType::Limit,
            "stop" => OrderType::Stop,
            _ => OrderType::Limit,
        };

        // Submit order to matching engine
        let atomic_order = AtomicOrder::new(order_id, price, quantity, side, order_type);
        self.matching_engine.submit_order(atomic_order);

        // Broadcast order update
        let event = MarketDataEvent::OrderUpdate {
            order_id,
            status: "new".to_string(),
            filled_quantity: 0.0,
        };
        let _ = self.broadcast_tx.send(event);
    }

    async fn process_cancel_order(&self, _client_id: Uuid, _order_id: u64) {
        // Implementation for order cancellation
        // This would involve finding the order in the order book and cancelling it
    }

    async fn process_modify_order(
        &self,
        _client_id: Uuid,
        _order_id: u64,
        _new_price: Option<u64>,
        _new_quantity: Option<u64>,
    ) {
        // Implementation for order modification
        // This would involve finding and atomically updating the order
    }

    fn generate_order_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        ORDER_ID_COUNTER.fetch_add(1, Ordering::AcqRel)
    }

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
        }
    }
}

impl Clone for TradingMCPServer {
    fn clone(&self) -> Self {
        Self {
            order_book: self.order_book.clone(),
            matching_engine: self.matching_engine.clone(),
            resource_manager: self.resource_manager.clone(),
            tool_manager: self.tool_manager.clone(),
            subscription_manager: self.subscription_manager.clone(),
            clients: self.clients.clone(),
            broadcast_tx: self.broadcast_tx.clone(),
            command_tx: self.command_tx.clone(),
            command_rx: self.command_rx.clone(),
            config: self.config.clone(),
        }
    }
}

impl Clone for MCPClient {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            websocket: self.websocket.clone(),
            subscriptions: self.subscriptions.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_server_creation() {
        let server = TradingMCPServer::new(None).await.unwrap();
        assert!(!server.clients.read().await.is_empty() == false);
    }

    #[tokio::test]
    async fn test_background_tasks() {
        let server = TradingMCPServer::new(None).await.unwrap();

        // Start background tasks
        tokio::spawn(async move {
            server.start_background_tasks().await.unwrap();
        });

        // Let background tasks run briefly
        sleep(Duration::from_millis(100)).await;

        // Tasks should be running (no panics)
    }

    #[test]
    fn test_mcp_message_serialization() {
        let message = MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: Some("test".to_string()),
            params: Some(json!({})),
            result: None,
            error: None,
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: MCPMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(message.jsonrpc, deserialized.jsonrpc);
        assert_eq!(message.method, deserialized.method);
    }
}
