// Tool Manager for CWTS Ultra MCP Server
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
// Removed unused Uuid import

use crate::algorithms::lockfree_orderbook::LockFreeOrderBook;
use crate::execution::simple_orders::{AtomicMatchingEngine, AtomicOrder, OrderSide, OrderType};

/// Trading tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
    pub annotations: Option<HashMap<String, Value>>,
}

/// Order placement request
#[derive(Debug, Serialize, Deserialize)]
pub struct PlaceOrderRequest {
    pub symbol: String,
    pub side: String,       // "buy" or "sell"
    pub order_type: String, // "market", "limit", "stop"
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: Option<String>, // "GTC", "IOC", "FOK"
    pub client_order_id: Option<String>,
}

/// Order modification request
#[derive(Debug, Serialize, Deserialize)]
pub struct ModifyOrderRequest {
    pub order_id: u64,
    pub new_price: Option<f64>,
    pub new_quantity: Option<f64>,
}

/// Position query parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct GetPositionsRequest {
    pub symbol: Option<String>,
    pub include_pnl: Option<bool>,
}

/// Market data query parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct GetMarketDataRequest {
    pub symbol: String,
    pub depth: Option<usize>, // Order book depth
    pub include_trades: Option<bool>,
}

/// Risk analysis request
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyzeRiskRequest {
    pub symbol: Option<String>,
    pub time_horizon: Option<String>,  // "1d", "1w", "1m"
    pub confidence_level: Option<f64>, // 0.95, 0.99, etc.
}

/// Tool Manager
pub struct ToolManager {
    order_book: Arc<LockFreeOrderBook>,
    #[allow(dead_code)]
    matching_engine: Arc<AtomicMatchingEngine>,
    active_orders: Arc<tokio::sync::RwLock<HashMap<u64, AtomicOrder>>>,
}

impl ToolManager {
    pub fn new(
        order_book: Arc<LockFreeOrderBook>,
        matching_engine: Arc<AtomicMatchingEngine>,
    ) -> Self {
        Self {
            order_book,
            matching_engine,
            active_orders: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// List all available tools
    pub async fn list_tools(&self) -> Vec<TradingTool> {
        vec![
            TradingTool {
                name: "place_order".to_string(),
                description: "Place a new trading order (buy/sell)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol (e.g., BTCUSD)"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Order side"
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop"],
                            "description": "Order type"
                        },
                        "quantity": {
                            "type": "number",
                            "minimum": 0.0001,
                            "description": "Order quantity"
                        },
                        "price": {
                            "type": "number",
                            "minimum": 0.01,
                            "description": "Order price (required for limit orders)"
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["GTC", "IOC", "FOK"],
                            "description": "Time in force"
                        },
                        "client_order_id": {
                            "type": "string",
                            "description": "Optional client-provided order ID"
                        }
                    },
                    "required": ["symbol", "side", "order_type", "quantity"]
                }),
            },
            TradingTool {
                name: "cancel_order".to_string(),
                description: "Cancel an existing order".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "integer",
                            "description": "Order ID to cancel"
                        }
                    },
                    "required": ["order_id"]
                }),
            },
            TradingTool {
                name: "modify_order".to_string(),
                description: "Modify price or quantity of existing order".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "integer",
                            "description": "Order ID to modify"
                        },
                        "new_price": {
                            "type": "number",
                            "minimum": 0.01,
                            "description": "New order price"
                        },
                        "new_quantity": {
                            "type": "number",
                            "minimum": 0.0001,
                            "description": "New order quantity"
                        }
                    },
                    "required": ["order_id"]
                }),
            },
            TradingTool {
                name: "get_positions".to_string(),
                description: "Get current trading positions".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Filter by symbol (optional)"
                        },
                        "include_pnl": {
                            "type": "boolean",
                            "description": "Include P&L calculations"
                        }
                    }
                }),
            },
            TradingTool {
                name: "get_market_data".to_string(),
                description: "Get real-time market data and order book".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "depth": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Order book depth levels"
                        },
                        "include_trades": {
                            "type": "boolean",
                            "description": "Include recent trades"
                        }
                    },
                    "required": ["symbol"]
                }),
            },
            TradingTool {
                name: "analyze_risk".to_string(),
                description: "Perform portfolio and position risk analysis".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Analyze specific symbol (optional)"
                        },
                        "time_horizon": {
                            "type": "string",
                            "enum": ["1d", "1w", "1m"],
                            "description": "Risk analysis time horizon"
                        },
                        "confidence_level": {
                            "type": "number",
                            "minimum": 0.9,
                            "maximum": 0.999,
                            "description": "VaR confidence level"
                        }
                    }
                }),
            },
            TradingTool {
                name: "get_order_status".to_string(),
                description: "Get detailed status of an order".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "integer",
                            "description": "Order ID to query"
                        }
                    },
                    "required": ["order_id"]
                }),
            },
            TradingTool {
                name: "calculate_profit_loss".to_string(),
                description: "Calculate P&L for a hypothetical trade".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "entry_price": {
                            "type": "number",
                            "description": "Entry price"
                        },
                        "exit_price": {
                            "type": "number",
                            "description": "Exit price"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Position size"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "Position side"
                        }
                    },
                    "required": ["symbol", "entry_price", "exit_price", "quantity", "side"]
                }),
            },
        ]
    }

    /// Execute a tool call
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolResult, String> {
        match name {
            "place_order" => self.place_order(arguments).await,
            "cancel_order" => self.cancel_order(arguments).await,
            "modify_order" => self.modify_order(arguments).await,
            "get_positions" => self.get_positions(arguments).await,
            "get_market_data" => self.get_market_data(arguments).await,
            "analyze_risk" => self.analyze_risk(arguments).await,
            "get_order_status" => self.get_order_status(arguments).await,
            "calculate_profit_loss" => self.calculate_profit_loss(arguments).await,
            _ => Err(format!("Tool not found: {}", name)),
        }
    }

    async fn place_order(&self, arguments: Value) -> Result<ToolResult, String> {
        let request: PlaceOrderRequest =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {}", e))?;

        // Validate symbol
        if request.symbol != "BTCUSD" {
            return Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!("Error: Unsupported symbol '{}'", request.symbol),
                    annotations: None,
                }],
                is_error: true,
            });
        }

        // Generate order ID
        let order_id = self.generate_order_id();

        // Convert to internal format
        let price = match request.order_type.as_str() {
            "limit" | "stop" => request
                .price
                .ok_or("Price required for limit/stop orders")?,
            "market" => {
                let (bid, ask) = self.order_book.get_spread();
                if request.side == "buy" {
                    ask as f64 / 1_000_000.0
                } else {
                    bid as f64 / 1_000_000.0
                }
            }
            _ => return Err("Invalid order type".to_string()),
        };

        let internal_price = (price * 1_000_000.0) as u64;
        let internal_quantity = (request.quantity * 100_000_000.0) as u64;

        let side = match request.side.as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            _ => return Err("Invalid side".to_string()),
        };

        let order_type = match request.order_type.as_str() {
            "market" => OrderType::Market,
            "limit" => OrderType::Limit,
            "stop" => OrderType::Stop,
            _ => return Err("Invalid order type".to_string()),
        };

        // Create atomic order
        let atomic_order = AtomicOrder::new(
            order_id,
            internal_price,
            internal_quantity,
            side,
            order_type,
        );

        // Store order for tracking
        {
            let mut active_orders = self.active_orders.write().await;
            active_orders.insert(order_id, atomic_order);
        }

        // Submit to matching engine (simulated)
        let success = match side {
            OrderSide::Buy => self
                .order_book
                .add_bid(internal_price, internal_quantity, order_id),
            OrderSide::Sell => self
                .order_book
                .add_ask(internal_price, internal_quantity, order_id),
        };

        if success {
            let response = json!({
                "order_id": order_id,
                "status": "accepted",
                "symbol": request.symbol,
                "side": request.side,
                "order_type": request.order_type,
                "quantity": request.quantity,
                "price": price,
                "timestamp": Utc::now(),
                "client_order_id": request.client_order_id
            });

            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!(
                        "Order placed successfully:\n{}",
                        serde_json::to_string_pretty(&response).unwrap()
                    ),
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("order_id".to_string(), json!(order_id));
                        annotations.insert("status".to_string(), json!("accepted"));
                        annotations
                    }),
                }],
                is_error: false,
            })
        } else {
            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: "Error: Failed to place order".to_string(),
                    annotations: None,
                }],
                is_error: true,
            })
        }
    }

    async fn cancel_order(&self, arguments: Value) -> Result<ToolResult, String> {
        let order_id: u64 = arguments
            .get("order_id")
            .and_then(|v| v.as_u64())
            .ok_or("Missing or invalid order_id")?;

        // Find and cancel order
        let result = {
            let active_orders = self.active_orders.read().await;
            if let Some(order) = active_orders.get(&order_id) {
                order.atomic_cancel()
            } else {
                false
            }
        };

        if result {
            // Remove from active orders
            {
                let mut active_orders = self.active_orders.write().await;
                active_orders.remove(&order_id);
            }

            let response = json!({
                "order_id": order_id,
                "status": "cancelled",
                "timestamp": Utc::now()
            });

            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!(
                        "Order cancelled successfully:\n{}",
                        serde_json::to_string_pretty(&response).unwrap()
                    ),
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("order_id".to_string(), json!(order_id));
                        annotations.insert("status".to_string(), json!("cancelled"));
                        annotations
                    }),
                }],
                is_error: false,
            })
        } else {
            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!("Error: Order {} not found or already cancelled", order_id),
                    annotations: None,
                }],
                is_error: true,
            })
        }
    }

    async fn modify_order(&self, arguments: Value) -> Result<ToolResult, String> {
        let request: ModifyOrderRequest =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {}", e))?;

        let active_orders = self.active_orders.read().await;
        if let Some(order) = active_orders.get(&request.order_id) {
            let mut modified = false;

            if let Some(new_price) = request.new_price {
                let internal_price = (new_price * 1_000_000.0) as u64;
                if order.atomic_modify_price(internal_price) {
                    modified = true;
                }
            }

            if let Some(new_quantity) = request.new_quantity {
                let internal_quantity = (new_quantity * 100_000_000.0) as u64;
                if order.atomic_modify_quantity(internal_quantity) {
                    modified = true;
                }
            }

            if modified {
                let response = json!({
                    "order_id": request.order_id,
                    "status": "modified",
                    "new_price": request.new_price,
                    "new_quantity": request.new_quantity,
                    "timestamp": Utc::now()
                });

                Ok(ToolResult {
                    content: vec![ToolContent {
                        content_type: "text".to_string(),
                        text: format!(
                            "Order modified successfully:\n{}",
                            serde_json::to_string_pretty(&response).unwrap()
                        ),
                        annotations: Some({
                            let mut annotations = HashMap::new();
                            annotations.insert("order_id".to_string(), json!(request.order_id));
                            annotations.insert("status".to_string(), json!("modified"));
                            annotations
                        }),
                    }],
                    is_error: false,
                })
            } else {
                Ok(ToolResult {
                    content: vec![ToolContent {
                        content_type: "text".to_string(),
                        text: "Error: Failed to modify order".to_string(),
                        annotations: None,
                    }],
                    is_error: true,
                })
            }
        } else {
            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!("Error: Order {} not found", request.order_id),
                    annotations: None,
                }],
                is_error: true,
            })
        }
    }

    async fn get_positions(&self, _arguments: Value) -> Result<ToolResult, String> {
        // Simulated positions data
        let positions = json!({
            "positions": [
                {
                    "symbol": "BTCUSD",
                    "side": "long",
                    "size": 0.5,
                    "entry_price": 45000.0,
                    "current_price": 46500.0,
                    "unrealized_pnl": 750.0,
                    "margin_used": 2250.0,
                    "timestamp": Utc::now()
                }
            ],
            "total_unrealized_pnl": 750.0,
            "total_margin_used": 2250.0,
            "position_count": 1
        });

        Ok(ToolResult {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: format!(
                    "Current positions:\n{}",
                    serde_json::to_string_pretty(&positions).unwrap()
                ),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("position_count".to_string(), json!(1));
                    annotations.insert("total_pnl".to_string(), json!(750.0));
                    annotations
                }),
            }],
            is_error: false,
        })
    }

    async fn get_market_data(&self, arguments: Value) -> Result<ToolResult, String> {
        let request: GetMarketDataRequest =
            serde_json::from_value(arguments).map_err(|e| format!("Invalid arguments: {}", e))?;

        if request.symbol != "BTCUSD" {
            return Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!("Error: Unsupported symbol '{}'", request.symbol),
                    annotations: None,
                }],
                is_error: true,
            });
        }

        let depth = request.depth.unwrap_or(5);
        let (bids, asks) = self.order_book.get_depth(depth);
        let (best_bid, best_ask) = self.order_book.get_spread();

        // Convert to external format
        let bid_levels: Vec<(f64, f64)> = bids
            .into_iter()
            .map(|(price, qty)| (price as f64 / 1_000_000.0, qty as f64 / 100_000_000.0))
            .collect();

        let ask_levels: Vec<(f64, f64)> = asks
            .into_iter()
            .map(|(price, qty)| (price as f64 / 1_000_000.0, qty as f64 / 100_000_000.0))
            .collect();

        let market_data = json!({
            "symbol": request.symbol,
            "timestamp": Utc::now(),
            "best_bid": best_bid as f64 / 1_000_000.0,
            "best_ask": best_ask as f64 / 1_000_000.0,
            "spread": (best_ask - best_bid) as f64 / 1_000_000.0,
            "mid_price": (best_bid + best_ask) as f64 / 2_000_000.0,
            "order_book": {
                "bids": bid_levels,
                "asks": ask_levels
            },
            "last_price": (best_bid + best_ask) as f64 / 2_000_000.0,
            "volume_24h": 12_345_678.90,
            "price_change_24h": 2.35
        });

        Ok(ToolResult {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: format!(
                    "Market data for {}:\n{}",
                    request.symbol,
                    serde_json::to_string_pretty(&market_data).unwrap()
                ),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("symbol".to_string(), json!(request.symbol));
                    annotations.insert(
                        "spread".to_string(),
                        json!((best_ask - best_bid) as f64 / 1_000_000.0),
                    );
                    annotations
                }),
            }],
            is_error: false,
        })
    }

    async fn analyze_risk(&self, _arguments: Value) -> Result<ToolResult, String> {
        // Simulated risk analysis
        let risk_analysis = json!({
            "portfolio_value": 50000.0,
            "total_exposure": 22500.0,
            "leverage": 0.45,
            "var_95_1d": 950.0,
            "cvar_95_1d": 1216.0,
            "max_drawdown": -2500.0,
            "sharpe_ratio": 1.85,
            "position_risk": {
                "concentration_risk": "low",
                "largest_position_pct": 4.5,
                "currency_exposure": {
                    "USD": 95.5,
                    "BTC": 4.5
                }
            },
            "recommendations": [
                "Consider diversifying cryptocurrency exposure",
                "Current leverage is within acceptable limits",
                "Portfolio volatility is manageable"
            ],
            "timestamp": Utc::now()
        });

        Ok(ToolResult {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: format!(
                    "Risk Analysis:\n{}",
                    serde_json::to_string_pretty(&risk_analysis).unwrap()
                ),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("var_95".to_string(), json!(950.0));
                    annotations.insert("leverage".to_string(), json!(0.45));
                    annotations.insert("risk_level".to_string(), json!("moderate"));
                    annotations
                }),
            }],
            is_error: false,
        })
    }

    async fn get_order_status(&self, arguments: Value) -> Result<ToolResult, String> {
        let order_id: u64 = arguments
            .get("order_id")
            .and_then(|v| v.as_u64())
            .ok_or("Missing or invalid order_id")?;

        let active_orders = self.active_orders.read().await;
        if let Some(order) = active_orders.get(&order_id) {
            let order_info = json!({
                "order_id": order_id,
                "price": order.price.load(std::sync::atomic::Ordering::Acquire) as f64 / 1_000_000.0,
                "quantity": order.quantity.load(std::sync::atomic::Ordering::Acquire) as f64 / 100_000_000.0,
                "filled_quantity": order.filled_quantity.load(std::sync::atomic::Ordering::Acquire) as f64 / 100_000_000.0,
                "status": match order.status.load(std::sync::atomic::Ordering::Acquire) {
                    0 => "new",
                    1 => "partially_filled",
                    2 => "filled",
                    3 => "cancelled",
                    _ => "unknown"
                },
                "is_active": order.is_active.load(std::sync::atomic::Ordering::Acquire),
                "created_time": order.created_ns.load(std::sync::atomic::Ordering::Acquire),
                "updated_time": order.updated_ns.load(std::sync::atomic::Ordering::Acquire)
            });

            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!(
                        "Order status:\n{}",
                        serde_json::to_string_pretty(&order_info).unwrap()
                    ),
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("order_id".to_string(), json!(order_id));
                        annotations
                    }),
                }],
                is_error: false,
            })
        } else {
            Ok(ToolResult {
                content: vec![ToolContent {
                    content_type: "text".to_string(),
                    text: format!("Error: Order {} not found", order_id),
                    annotations: None,
                }],
                is_error: true,
            })
        }
    }

    async fn calculate_profit_loss(&self, arguments: Value) -> Result<ToolResult, String> {
        let entry_price: f64 = arguments
            .get("entry_price")
            .and_then(|v| v.as_f64())
            .ok_or("Missing or invalid entry_price")?;

        let exit_price: f64 = arguments
            .get("exit_price")
            .and_then(|v| v.as_f64())
            .ok_or("Missing or invalid exit_price")?;

        let quantity: f64 = arguments
            .get("quantity")
            .and_then(|v| v.as_f64())
            .ok_or("Missing or invalid quantity")?;

        let side: &str = arguments
            .get("side")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid side")?;

        let symbol: &str = arguments
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid symbol")?;

        // Calculate P&L
        let pnl = match side {
            "buy" => (exit_price - entry_price) * quantity,
            "sell" => (entry_price - exit_price) * quantity,
            _ => return Err("Invalid side".to_string()),
        };

        let pnl_percentage = (pnl / (entry_price * quantity)).abs() * 100.0;
        let commission = (entry_price * quantity + exit_price * quantity) * 0.001; // 0.1% commission
        let net_pnl = pnl - commission;

        let pnl_calc = json!({
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_pnl": pnl,
            "commission": commission,
            "net_pnl": net_pnl,
            "pnl_percentage": pnl_percentage,
            "is_profitable": net_pnl > 0.0,
            "calculation_time": Utc::now()
        });

        Ok(ToolResult {
            content: vec![ToolContent {
                content_type: "text".to_string(),
                text: format!(
                    "P&L Calculation:\n{}",
                    serde_json::to_string_pretty(&pnl_calc).unwrap()
                ),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("net_pnl".to_string(), json!(net_pnl));
                    annotations.insert("profitable".to_string(), json!(net_pnl > 0.0));
                    annotations
                }),
            }],
            is_error: false,
        })
    }

    fn generate_order_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        ORDER_ID_COUNTER.fetch_add(1, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_manager_creation() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());
        let tool_manager = ToolManager::new(order_book, matching_engine);

        let tools = tool_manager.list_tools().await;
        assert!(!tools.is_empty());
        assert!(tools.len() >= 8);
    }

    #[tokio::test]
    async fn test_place_order_tool() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());
        let tool_manager = ToolManager::new(order_book, matching_engine);

        let arguments = json!({
            "symbol": "BTCUSD",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.1,
            "price": 45000.0
        });

        let result = tool_manager
            .call_tool("place_order", arguments)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn test_get_market_data_tool() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());
        let tool_manager = ToolManager::new(order_book.clone(), matching_engine);

        // Add some test data to order book
        order_book.add_bid(45_000_000_000, 100_000_000, 1);
        order_book.add_ask(45_500_000_000, 200_000_000, 2);

        let arguments = json!({
            "symbol": "BTCUSD",
            "depth": 5
        });

        let result = tool_manager
            .call_tool("get_market_data", arguments)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn test_calculate_profit_loss_tool() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());
        let tool_manager = ToolManager::new(order_book, matching_engine);

        let arguments = json!({
            "symbol": "BTCUSD",
            "entry_price": 45000.0,
            "exit_price": 46000.0,
            "quantity": 0.5,
            "side": "buy"
        });

        let result = tool_manager
            .call_tool("calculate_profit_loss", arguments)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn test_invalid_tool() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let matching_engine = Arc::new(AtomicMatchingEngine::new());
        let tool_manager = ToolManager::new(order_book, matching_engine);

        let result = tool_manager.call_tool("invalid_tool", json!({})).await;
        assert!(result.is_err());
    }
}
