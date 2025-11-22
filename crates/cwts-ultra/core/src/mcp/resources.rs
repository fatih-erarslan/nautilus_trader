// Resource Manager for CWTS Ultra MCP Server
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use crate::algorithms::lockfree_orderbook::LockFreeOrderBook;
// Removed unused EngineStats import

/// Trading resource types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
    pub annotations: Option<HashMap<String, Value>>,
}

/// Resource content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    pub uri: String,
    pub mime_type: String,
    pub text: Option<String>,
    pub blob: Option<Vec<u8>>,
    pub annotations: Option<HashMap<String, Value>>,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub spread: f64,
    pub mid_price: f64,
    pub depth_levels: usize,
}

/// Position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: String, // "long" or "short"
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub margin_used: f64,
    pub timestamp: DateTime<Utc>,
}

/// Market data tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataTick {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume: f64,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    pub close: f64,
}

/// Trade execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub trade_id: u64,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub side: String,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub buy_order_id: u64,
    pub sell_order_id: u64,
}

/// Account summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSummary {
    pub account_id: String,
    pub timestamp: DateTime<Utc>,
    pub total_equity: f64,
    pub available_margin: f64,
    pub used_margin: f64,
    pub free_margin: f64,
    pub margin_level: f64,
    pub total_pnl: f64,
    pub daily_pnl: f64,
    pub position_count: usize,
    pub order_count: usize,
}

/// Resource Manager
pub struct ResourceManager {
    order_book: Arc<LockFreeOrderBook>,
    positions: Arc<tokio::sync::RwLock<HashMap<String, Position>>>,
    trade_history: Arc<tokio::sync::RwLock<Vec<TradeRecord>>>,
    market_data: Arc<tokio::sync::RwLock<HashMap<String, MarketDataTick>>>,
    account: Arc<tokio::sync::RwLock<AccountSummary>>,
}

impl ResourceManager {
    pub fn new(order_book: Arc<LockFreeOrderBook>) -> Self {
        let account = AccountSummary {
            account_id: "CWTS_ULTRA_001".to_string(),
            timestamp: Utc::now(),
            total_equity: 50000.0, // $50,000 starting capital
            available_margin: 48000.0,
            used_margin: 2000.0,
            free_margin: 46000.0,
            margin_level: 2400.0, // 2400%
            total_pnl: 0.0,
            daily_pnl: 0.0,
            position_count: 0,
            order_count: 0,
        };

        Self {
            order_book,
            positions: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            trade_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            market_data: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            account: Arc::new(tokio::sync::RwLock::new(account)),
        }
    }

    /// List all available resources
    pub async fn list_resources(&self) -> Vec<TradingResource> {
        vec![
            TradingResource {
                uri: "trading://order_book/BTCUSD".to_string(),
                name: "BTC/USD Order Book".to_string(),
                description: "Real-time order book for Bitcoin/USD pair".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("symbol".to_string(), json!("BTCUSD"));
                    annotations.insert("asset_class".to_string(), json!("cryptocurrency"));
                    annotations.insert("update_frequency".to_string(), json!("real-time"));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://positions".to_string(),
                name: "Trading Positions".to_string(),
                description: "Current trading positions across all symbols".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("type".to_string(), json!("positions"));
                    annotations.insert("real_time".to_string(), json!(true));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://market_data/BTCUSD".to_string(),
                name: "BTC/USD Market Data".to_string(),
                description: "Real-time market data including bid/ask/last prices".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("symbol".to_string(), json!("BTCUSD"));
                    annotations.insert("data_type".to_string(), json!("tick"));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://trades/history".to_string(),
                name: "Trade History".to_string(),
                description: "Historical trade execution records".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("type".to_string(), json!("trade_history"));
                    annotations.insert("sortable".to_string(), json!(true));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://account/summary".to_string(),
                name: "Account Summary".to_string(),
                description: "Account balance, equity, margin, and P&L information".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("type".to_string(), json!("account"));
                    annotations.insert("sensitive".to_string(), json!(true));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://engine/stats".to_string(),
                name: "Matching Engine Statistics".to_string(),
                description: "Real-time matching engine performance metrics".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("type".to_string(), json!("engine_stats"));
                    annotations.insert("performance".to_string(), json!(true));
                    annotations
                }),
            },
            TradingResource {
                uri: "trading://risk/metrics".to_string(),
                name: "Risk Metrics".to_string(),
                description: "Portfolio risk analysis and exposure metrics".to_string(),
                mime_type: "application/json".to_string(),
                annotations: Some({
                    let mut annotations = HashMap::new();
                    annotations.insert("type".to_string(), json!("risk_metrics"));
                    annotations.insert("calculation_method".to_string(), json!("var_cvar"));
                    annotations
                }),
            },
        ]
    }

    /// Read specific resource content
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContent, String> {
        match uri {
            "trading://order_book/BTCUSD" => {
                let snapshot = self.get_order_book_snapshot("BTCUSD").await?;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&snapshot).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("timestamp".to_string(), json!(snapshot.timestamp));
                        annotations.insert("spread".to_string(), json!(snapshot.spread));
                        annotations.insert("levels".to_string(), json!(snapshot.depth_levels));
                        annotations
                    }),
                })
            }
            "trading://positions" => {
                let positions = self.get_all_positions().await;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&positions).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("count".to_string(), json!(positions.len()));
                        annotations.insert("timestamp".to_string(), json!(Utc::now()));
                        annotations
                    }),
                })
            }
            "trading://market_data/BTCUSD" => {
                let market_data = self.get_market_data("BTCUSD").await?;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&market_data).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("timestamp".to_string(), json!(market_data.timestamp));
                        annotations.insert(
                            "spread".to_string(),
                            json!(market_data.ask - market_data.bid),
                        );
                        annotations
                    }),
                })
            }
            "trading://trades/history" => {
                let trade_history = self.get_trade_history(100).await; // Last 100 trades
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&trade_history).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("count".to_string(), json!(trade_history.len()));
                        annotations.insert("period".to_string(), json!("latest_100"));
                        annotations
                    }),
                })
            }
            "trading://account/summary" => {
                let account = self.get_account_summary().await;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&account).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("timestamp".to_string(), json!(account.timestamp));
                        annotations.insert("currency".to_string(), json!("USD"));
                        annotations
                    }),
                })
            }
            "trading://engine/stats" => {
                let stats = self.get_engine_stats().await;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&stats).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("timestamp".to_string(), json!(Utc::now()));
                        annotations.insert("engine_type".to_string(), json!("lock_free_atomic"));
                        annotations
                    }),
                })
            }
            "trading://risk/metrics" => {
                let risk_metrics = self.calculate_risk_metrics().await;
                Ok(ResourceContent {
                    uri: uri.to_string(),
                    mime_type: "application/json".to_string(),
                    text: Some(serde_json::to_string_pretty(&risk_metrics).unwrap()),
                    blob: None,
                    annotations: Some({
                        let mut annotations = HashMap::new();
                        annotations.insert("calculation_time".to_string(), json!(Utc::now()));
                        annotations.insert("confidence_level".to_string(), json!(0.95));
                        annotations
                    }),
                })
            }
            _ => Err(format!("Resource not found: {}", uri)),
        }
    }

    /// Get order book snapshot for a symbol
    async fn get_order_book_snapshot(&self, symbol: &str) -> Result<OrderBookSnapshot, String> {
        if symbol != "BTCUSD" {
            return Err("Symbol not supported".to_string());
        }

        let (bids, asks) = self.order_book.get_depth(10);
        let (best_bid, best_ask) = self.order_book.get_spread();

        // Convert internal format to external format
        let bid_levels: Vec<(f64, f64)> = bids
            .into_iter()
            .map(|(price, qty)| (price as f64 / 1_000_000.0, qty as f64 / 100_000_000.0))
            .collect();

        let ask_levels: Vec<(f64, f64)> = asks
            .into_iter()
            .map(|(price, qty)| (price as f64 / 1_000_000.0, qty as f64 / 100_000_000.0))
            .collect();

        let spread = if best_ask > best_bid {
            (best_ask - best_bid) as f64 / 1_000_000.0
        } else {
            0.0
        };

        let mid_price = if best_bid > 0 && best_ask > 0 {
            (best_bid + best_ask) as f64 / 2_000_000.0
        } else {
            0.0
        };

        Ok(OrderBookSnapshot {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids: bid_levels,
            asks: ask_levels,
            spread,
            mid_price,
            depth_levels: 10,
        })
    }

    /// Get all current positions
    async fn get_all_positions(&self) -> Vec<Position> {
        let positions = self.positions.read().await;
        positions.values().cloned().collect()
    }

    /// Get market data for a symbol
    async fn get_market_data(&self, symbol: &str) -> Result<MarketDataTick, String> {
        let market_data = self.market_data.read().await;

        if let Some(tick) = market_data.get(symbol) {
            Ok(tick.clone())
        } else {
            // Generate simulated market data
            let (best_bid, best_ask) = self.order_book.get_spread();
            let bid = best_bid as f64 / 1_000_000.0;
            let ask = best_ask as f64 / 1_000_000.0;
            let last = (bid + ask) / 2.0;

            Ok(MarketDataTick {
                symbol: symbol.to_string(),
                timestamp: Utc::now(),
                bid,
                ask,
                last,
                volume: 1_234_567.89, // Simulated volume
                high: last * 1.02,
                low: last * 0.98,
                open: last * 0.995,
                close: last,
            })
        }
    }

    /// Get trade history
    async fn get_trade_history(&self, limit: usize) -> Vec<TradeRecord> {
        let trade_history = self.trade_history.read().await;
        trade_history.iter().rev().take(limit).cloned().collect()
    }

    /// Get account summary
    async fn get_account_summary(&self) -> AccountSummary {
        let mut account = self.account.write().await;

        // Update timestamp and recalculate metrics
        account.timestamp = Utc::now();

        // Calculate current P&L from positions
        let positions = self.positions.read().await;
        account.total_pnl = positions.values().map(|p| p.unrealized_pnl).sum();
        account.position_count = positions.len();

        // Update margin calculations
        account.used_margin = positions.values().map(|p| p.margin_used).sum();
        account.free_margin = account.total_equity - account.used_margin;
        account.available_margin = account.free_margin * 0.8; // 80% of free margin
        account.margin_level = if account.used_margin > 0.0 {
            (account.total_equity / account.used_margin) * 100.0
        } else {
            999999.0 // No margin used
        };

        account.clone()
    }

    /// Get matching engine statistics
    async fn get_engine_stats(&self) -> serde_json::Value {
        // This would get real stats from the matching engine
        // For now, return simulated stats
        json!({
            "timestamp": Utc::now(),
            "orders_processed": 15847,
            "trades_executed": 1234,
            "average_latency_us": 2.3,
            "max_latency_us": 12.7,
            "orders_per_second": 125000,
            "matches_per_second": 8500,
            "memory_usage_mb": 45.2,
            "cpu_usage_percent": 12.5,
            "active_buy_orders": 42,
            "active_sell_orders": 38,
            "total_volume_24h": 2_456_789.12
        })
    }

    /// Calculate risk metrics
    async fn calculate_risk_metrics(&self) -> serde_json::Value {
        let positions = self.positions.read().await;
        let account = self.account.read().await;

        // Calculate portfolio-level risk metrics
        let total_exposure: f64 = positions
            .values()
            .map(|p| p.size.abs() * p.current_price)
            .sum();
        let leverage = if account.total_equity > 0.0 {
            total_exposure / account.total_equity
        } else {
            0.0
        };

        // Calculate position-level metrics
        let long_exposure: f64 = positions
            .values()
            .filter(|p| p.side == "long")
            .map(|p| p.size * p.current_price)
            .sum();

        let short_exposure: f64 = positions
            .values()
            .filter(|p| p.side == "short")
            .map(|p| p.size.abs() * p.current_price)
            .sum();

        // Simplified VaR calculation (1-day, 95% confidence)
        let portfolio_volatility = 0.02; // 2% daily volatility assumption
        let var_95 = account.total_equity * portfolio_volatility * 1.645;
        let cvar_95 = var_95 * 1.28; // Expected shortfall

        json!({
            "timestamp": Utc::now(),
            "portfolio_value": account.total_equity,
            "total_exposure": total_exposure,
            "leverage": leverage,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "gross_exposure": long_exposure + short_exposure,
            "margin_utilization": account.used_margin / account.total_equity,
            "var_95_1d": var_95,
            "cvar_95_1d": cvar_95,
            "max_drawdown": -2500.0, // Simulated max drawdown
            "sharpe_ratio": 1.85, // Simulated Sharpe ratio
            "position_count": positions.len(),
            "concentration_risk": {
                "largest_position_pct": 15.2,
                "top_3_positions_pct": 38.7
            },
            "currency_exposure": {
                "USD": account.total_equity,
                "BTC": 0.0,
                "ETH": 0.0
            }
        })
    }

    /// Add a new position
    pub async fn add_position(&self, position: Position) {
        let mut positions = self.positions.write().await;
        positions.insert(position.symbol.clone(), position);
    }

    /// Update market data
    pub async fn update_market_data(&self, symbol: &str, tick: MarketDataTick) {
        let mut market_data = self.market_data.write().await;
        market_data.insert(symbol.to_string(), tick);
    }

    /// Add trade record
    pub async fn add_trade_record(&self, trade: TradeRecord) {
        let mut trade_history = self.trade_history.write().await;
        trade_history.push(trade);

        // Keep only last 10,000 trades to manage memory
        if trade_history.len() > 10_000 {
            trade_history.drain(0..1000); // Remove oldest 1,000 trades
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_manager_creation() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let resource_manager = ResourceManager::new(order_book);

        let resources = resource_manager.list_resources().await;
        assert!(!resources.is_empty());
        assert!(resources.len() >= 7);
    }

    #[tokio::test]
    async fn test_order_book_resource() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let resource_manager = ResourceManager::new(order_book.clone());

        // Add some orders to the book
        order_book.add_bid(99_500_000, 1_000_000_000, 1); // $99.50, 10.0 units
        order_book.add_ask(100_500_000, 2_000_000_000, 2); // $100.50, 20.0 units

        let content = resource_manager
            .read_resource("trading://order_book/BTCUSD")
            .await
            .unwrap();

        assert_eq!(content.mime_type, "application/json");
        assert!(content.text.is_some());

        let snapshot: OrderBookSnapshot = serde_json::from_str(&content.text.unwrap()).unwrap();
        assert_eq!(snapshot.symbol, "BTCUSD");
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }

    #[tokio::test]
    async fn test_account_summary_resource() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let resource_manager = ResourceManager::new(order_book);

        let content = resource_manager
            .read_resource("trading://account/summary")
            .await
            .unwrap();

        assert_eq!(content.mime_type, "application/json");
        assert!(content.text.is_some());

        let account: AccountSummary = serde_json::from_str(&content.text.unwrap()).unwrap();
        assert_eq!(account.account_id, "CWTS_ULTRA_001");
        assert!(account.total_equity > 0.0);
    }

    #[tokio::test]
    async fn test_position_management() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let resource_manager = ResourceManager::new(order_book);

        let position = Position {
            symbol: "BTCUSD".to_string(),
            side: "long".to_string(),
            size: 1.5,
            entry_price: 45000.0,
            current_price: 46000.0,
            unrealized_pnl: 1500.0,
            realized_pnl: 0.0,
            margin_used: 4500.0,
            timestamp: Utc::now(),
        };

        resource_manager.add_position(position.clone()).await;

        let positions = resource_manager.get_all_positions().await;
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].symbol, "BTCUSD");
    }

    #[tokio::test]
    async fn test_invalid_resource() {
        let order_book = Arc::new(LockFreeOrderBook::new());
        let resource_manager = ResourceManager::new(order_book);

        let result = resource_manager
            .read_resource("trading://invalid/resource")
            .await;

        assert!(result.is_err());
    }
}
