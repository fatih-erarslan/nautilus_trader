// Subscription Manager for CWTS Ultra MCP Server
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

use crate::execution::simple_orders::Trade;

/// Market data events that can be subscribed to
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MarketDataEvent {
    /// Order book depth update
    OrderBookUpdate {
        symbol: String,
        bids: Vec<(f64, f64)>, // (price, quantity)
        asks: Vec<(f64, f64)>,
        timestamp: DateTime<Utc>,
    },
    /// Trade execution update
    TradeUpdate { trades: Vec<Trade> },
    /// Price tick update
    PriceUpdate {
        symbol: String,
        bid: f64,
        ask: f64,
        last: f64,
        volume: f64,
        timestamp: DateTime<Utc>,
    },
    /// Order status change
    OrderUpdate {
        order_id: u64,
        status: String,
        filled_quantity: f64,
    },
    /// Position update
    PositionUpdate {
        symbol: String,
        size: f64,
        unrealized_pnl: f64,
        timestamp: DateTime<Utc>,
    },
    /// Account balance update
    AccountUpdate {
        total_equity: f64,
        available_margin: f64,
        timestamp: DateTime<Utc>,
    },
    /// Risk alert
    RiskAlert {
        alert_type: String,
        message: String,
        severity: String, // "info", "warning", "critical"
        timestamp: DateTime<Utc>,
    },
    /// Market status update
    MarketStatus {
        symbol: String,
        status: String, // "open", "closed", "halted"
        timestamp: DateTime<Utc>,
    },
}

/// Subscription information
#[derive(Debug, Clone)]
pub struct Subscription {
    pub client_id: Uuid,
    pub uri: String,
    pub event_type: SubscriptionType,
    pub filters: Option<HashMap<String, String>>,
    pub created_at: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

/// Types of subscriptions available
#[derive(Debug, Clone, PartialEq)]
pub enum SubscriptionType {
    OrderBook,
    Trades,
    Prices,
    Orders,
    Positions,
    Account,
    RiskAlerts,
    MarketStatus,
}

impl SubscriptionType {
    pub fn from_uri(uri: &str) -> Option<Self> {
        match uri {
            uri if uri.starts_with("trading://order_book/") => Some(Self::OrderBook),
            uri if uri.starts_with("trading://trades/") => Some(Self::Trades),
            uri if uri.starts_with("trading://prices/") => Some(Self::Prices),
            uri if uri.starts_with("trading://orders/") => Some(Self::Orders),
            uri if uri.starts_with("trading://positions/") => Some(Self::Positions),
            uri if uri.starts_with("trading://account/") => Some(Self::Account),
            uri if uri.starts_with("trading://risk/") => Some(Self::RiskAlerts),
            uri if uri.starts_with("trading://market_status/") => Some(Self::MarketStatus),
            _ => None,
        }
    }
}

/// Subscription statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionStats {
    pub total_subscriptions: usize,
    pub active_clients: usize,
    pub events_sent_total: u64,
    pub events_sent_last_hour: u64,
    pub subscription_breakdown: HashMap<String, usize>,
    pub top_subscribed_symbols: Vec<(String, usize)>,
}

/// Subscription Manager
pub struct SubscriptionManager {
    /// Active subscriptions indexed by client ID
    subscriptions: Arc<RwLock<HashMap<Uuid, Vec<Subscription>>>>,

    /// Broadcast sender for market data events
    broadcast_tx: broadcast::Sender<MarketDataEvent>,

    /// Statistics tracking
    events_sent_total: Arc<tokio::sync::RwLock<u64>>,
    events_sent_last_hour: Arc<tokio::sync::RwLock<u64>>,

    /// Event filters and routing
    event_filters: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl SubscriptionManager {
    pub fn new(broadcast_tx: broadcast::Sender<MarketDataEvent>) -> Self {
        let manager = Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            events_sent_total: Arc::new(tokio::sync::RwLock::new(0)),
            events_sent_last_hour: Arc::new(tokio::sync::RwLock::new(0)),
            event_filters: Arc::new(RwLock::new(HashMap::new())),
        };

        // Start background statistics reset task
        manager.start_stats_reset_task();

        manager
    }

    /// Subscribe a client to a specific URI
    pub async fn subscribe(&self, client_id: Uuid, uri: String) -> Result<(), String> {
        let subscription_type = SubscriptionType::from_uri(&uri)
            .ok_or_else(|| format!("Invalid subscription URI: {}", uri))?;

        // Parse filters from URI query parameters
        let filters = self.parse_uri_filters(&uri);

        let subscription = Subscription {
            client_id,
            uri: uri.clone(),
            event_type: subscription_type,
            filters,
            created_at: Utc::now(),
            last_update: Utc::now(),
        };

        // Add subscription to client's list
        let mut subscriptions = self.subscriptions.write().await;
        let client_subs = subscriptions.entry(client_id).or_insert_with(Vec::new);

        // Check for duplicate subscription
        if !client_subs.iter().any(|s| s.uri == uri) {
            client_subs.push(subscription);

            // Update event filters for efficient routing
            self.update_event_filters(&uri).await;

            println!("Client {} subscribed to {}", client_id, uri);
            Ok(())
        } else {
            Ok(()) // Already subscribed, no error
        }
    }

    /// Unsubscribe a client from a specific URI
    pub async fn unsubscribe(&self, client_id: Uuid, uri: String) -> Result<(), String> {
        let mut subscriptions = self.subscriptions.write().await;

        if let Some(client_subs) = subscriptions.get_mut(&client_id) {
            client_subs.retain(|s| s.uri != uri);

            // Remove client entry if no subscriptions left
            if client_subs.is_empty() {
                subscriptions.remove(&client_id);
            }

            println!("Client {} unsubscribed from {}", client_id, uri);
            Ok(())
        } else {
            Err("Client not found".to_string())
        }
    }

    /// Unsubscribe a client from all URIs (when client disconnects)
    pub async fn unsubscribe_all(&self, client_id: Uuid) {
        let mut subscriptions = self.subscriptions.write().await;
        if let Some(client_subs) = subscriptions.remove(&client_id) {
            println!(
                "Client {} unsubscribed from {} subscriptions",
                client_id,
                client_subs.len()
            );
        }
    }

    /// Get all subscriptions for a client
    pub async fn get_client_subscriptions(&self, client_id: Uuid) -> Vec<Subscription> {
        let subscriptions = self.subscriptions.read().await;
        subscriptions.get(&client_id).cloned().unwrap_or_default()
    }

    /// Broadcast an event to all relevant subscribers
    pub async fn broadcast_event(&self, event: MarketDataEvent) {
        // Update statistics
        {
            let mut total = self.events_sent_total.write().await;
            *total += 1;
        }
        {
            let mut hour = self.events_sent_last_hour.write().await;
            *hour += 1;
        }

        // Send to broadcast channel
        if let Err(e) = self.broadcast_tx.send(event.clone()) {
            eprintln!("Failed to broadcast event: {}", e);
        }

        // Log significant events
        match &event {
            MarketDataEvent::TradeUpdate { trades } if !trades.is_empty() => {
                println!("Broadcasted {} trade updates", trades.len());
            }
            MarketDataEvent::RiskAlert {
                severity, message, ..
            } if severity == "critical" => {
                println!("CRITICAL RISK ALERT: {}", message);
            }
            _ => {} // Don't log routine updates
        }
    }

    /// Get subscription statistics
    pub async fn get_statistics(&self) -> SubscriptionStats {
        let subscriptions = self.subscriptions.read().await;
        let total_subscriptions: usize = subscriptions.values().map(|v| v.len()).sum();
        let active_clients = subscriptions.len();

        let events_sent_total = *self.events_sent_total.read().await;
        let events_sent_last_hour = *self.events_sent_last_hour.read().await;

        // Calculate subscription breakdown by type
        let mut subscription_breakdown = HashMap::new();
        let mut symbol_counts = HashMap::new();

        for client_subs in subscriptions.values() {
            for sub in client_subs {
                let type_name = format!("{:?}", sub.event_type);
                *subscription_breakdown.entry(type_name).or_insert(0) += 1;

                // Extract symbol from URI for statistics
                if let Some(symbol) = self.extract_symbol_from_uri(&sub.uri) {
                    *symbol_counts.entry(symbol).or_insert(0) += 1;
                }
            }
        }

        // Get top subscribed symbols
        let mut top_subscribed_symbols: Vec<(String, usize)> = symbol_counts.into_iter().collect();
        top_subscribed_symbols.sort_by(|a, b| b.1.cmp(&a.1));
        top_subscribed_symbols.truncate(10); // Top 10

        SubscriptionStats {
            total_subscriptions,
            active_clients,
            events_sent_total,
            events_sent_last_hour,
            subscription_breakdown,
            top_subscribed_symbols,
        }
    }

    /// Check if a client is subscribed to a specific event type
    pub async fn is_subscribed(&self, client_id: Uuid, event_type: SubscriptionType) -> bool {
        let subscriptions = self.subscriptions.read().await;

        if let Some(client_subs) = subscriptions.get(&client_id) {
            client_subs.iter().any(|s| s.event_type == event_type)
        } else {
            false
        }
    }

    /// Get clients subscribed to a specific event type
    pub async fn get_subscribers(&self, event_type: SubscriptionType) -> Vec<Uuid> {
        let subscriptions = self.subscriptions.read().await;
        let mut subscribers = Vec::new();

        for (&client_id, client_subs) in subscriptions.iter() {
            if client_subs.iter().any(|s| s.event_type == event_type) {
                subscribers.push(client_id);
            }
        }

        subscribers
    }

    /// Parse URI query parameters for filters
    fn parse_uri_filters(&self, uri: &str) -> Option<HashMap<String, String>> {
        if let Some(query_start) = uri.find('?') {
            let query = &uri[query_start + 1..];
            let mut filters = HashMap::new();

            for pair in query.split('&') {
                if let Some(eq_pos) = pair.find('=') {
                    let key = pair[..eq_pos].to_string();
                    let value = pair[eq_pos + 1..].to_string();
                    filters.insert(key, value);
                }
            }

            if !filters.is_empty() {
                return Some(filters);
            }
        }
        None
    }

    /// Update event filters for efficient routing
    async fn update_event_filters(&self, uri: &str) {
        if let Some(symbol) = self.extract_symbol_from_uri(uri) {
            let mut filters = self.event_filters.write().await;
            let symbol_subs = filters.entry(symbol.clone()).or_insert_with(Vec::new);

            if !symbol_subs.contains(&uri.to_string()) {
                symbol_subs.push(uri.to_string());
            }
        }
    }

    /// Extract symbol from URI
    fn extract_symbol_from_uri(&self, uri: &str) -> Option<String> {
        // Extract symbol from URIs like "trading://order_book/BTCUSD"
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 3 {
            // Remove query parameters if present
            let symbol = parts[parts.len() - 1];
            if let Some(query_start) = symbol.find('?') {
                Some(symbol[..query_start].to_string())
            } else {
                Some(symbol.to_string())
            }
        } else {
            None
        }
    }

    /// Start background task to reset hourly statistics
    fn start_stats_reset_task(&self) {
        let events_sent_last_hour = self.events_sent_last_hour.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(3600), // 1 hour
            );

            loop {
                interval.tick().await;

                // Reset hourly counter
                {
                    let mut hour = events_sent_last_hour.write().await;
                    *hour = 0;
                }

                println!("Hourly subscription statistics reset");
            }
        });
    }

    /// Create sample events for testing
    pub async fn create_sample_events(&self) {
        // Sample order book update
        let order_book_event = MarketDataEvent::OrderBookUpdate {
            symbol: "BTCUSD".to_string(),
            bids: vec![(45000.0, 1.5), (44950.0, 2.0), (44900.0, 0.8)],
            asks: vec![(45050.0, 1.2), (45100.0, 1.8), (45150.0, 0.9)],
            timestamp: Utc::now(),
        };
        self.broadcast_event(order_book_event).await;

        // Sample price update
        let price_event = MarketDataEvent::PriceUpdate {
            symbol: "BTCUSD".to_string(),
            bid: 45000.0,
            ask: 45050.0,
            last: 45025.0,
            volume: 12345.67,
            timestamp: Utc::now(),
        };
        self.broadcast_event(price_event).await;

        // Sample account update
        let account_event = MarketDataEvent::AccountUpdate {
            total_equity: 50250.0,
            available_margin: 47800.0,
            timestamp: Utc::now(),
        };
        self.broadcast_event(account_event).await;
    }
}

/// Helper functions for event filtering and routing
impl SubscriptionManager {
    /// Check if an event matches a subscription's filters
    pub fn event_matches_subscription(
        &self,
        event: &MarketDataEvent,
        subscription: &Subscription,
    ) -> bool {
        // Check event type compatibility
        let event_type_matches = match (event, &subscription.event_type) {
            (MarketDataEvent::OrderBookUpdate { .. }, SubscriptionType::OrderBook) => true,
            (MarketDataEvent::TradeUpdate { .. }, SubscriptionType::Trades) => true,
            (MarketDataEvent::PriceUpdate { .. }, SubscriptionType::Prices) => true,
            (MarketDataEvent::OrderUpdate { .. }, SubscriptionType::Orders) => true,
            (MarketDataEvent::PositionUpdate { .. }, SubscriptionType::Positions) => true,
            (MarketDataEvent::AccountUpdate { .. }, SubscriptionType::Account) => true,
            (MarketDataEvent::RiskAlert { .. }, SubscriptionType::RiskAlerts) => true,
            (MarketDataEvent::MarketStatus { .. }, SubscriptionType::MarketStatus) => true,
            _ => false,
        };

        if !event_type_matches {
            return false;
        }

        // Apply additional filters if present
        if let Some(filters) = &subscription.filters {
            // Example: symbol filter
            if let Some(filter_symbol) = filters.get("symbol") {
                let event_symbol = match event {
                    MarketDataEvent::OrderBookUpdate { symbol, .. } => Some(symbol),
                    MarketDataEvent::PriceUpdate { symbol, .. } => Some(symbol),
                    MarketDataEvent::PositionUpdate { symbol, .. } => Some(symbol),
                    MarketDataEvent::MarketStatus { symbol, .. } => Some(symbol),
                    _ => None,
                };

                if let Some(symbol) = event_symbol {
                    if symbol != filter_symbol {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Example: severity filter for risk alerts
            if let Some(filter_severity) = filters.get("severity") {
                if let MarketDataEvent::RiskAlert { severity, .. } = event {
                    if severity != filter_severity {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn test_subscription_manager_creation() {
        let (tx, _rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_subscriptions, 0);
        assert_eq!(stats.active_clients, 0);
    }

    #[tokio::test]
    async fn test_client_subscription() {
        let (tx, _rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);
        let client_id = Uuid::new_v4();

        // Subscribe to order book
        let result = manager
            .subscribe(client_id, "trading://order_book/BTCUSD".to_string())
            .await;
        assert!(result.is_ok());

        // Check subscription
        let subs = manager.get_client_subscriptions(client_id).await;
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].uri, "trading://order_book/BTCUSD");

        // Check statistics
        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_subscriptions, 1);
        assert_eq!(stats.active_clients, 1);
    }

    #[tokio::test]
    async fn test_subscription_with_filters() {
        let (tx, _rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);
        let client_id = Uuid::new_v4();

        // Subscribe with filters
        let uri = "trading://risk/alerts?severity=critical".to_string();
        let result = manager.subscribe(client_id, uri).await;
        assert!(result.is_ok());

        let subs = manager.get_client_subscriptions(client_id).await;
        assert_eq!(subs.len(), 1);
        assert!(subs[0].filters.is_some());

        let filters = subs[0].filters.as_ref().unwrap();
        assert_eq!(filters.get("severity"), Some(&"critical".to_string()));
    }

    #[tokio::test]
    async fn test_event_broadcasting() {
        let (tx, mut rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);

        let event = MarketDataEvent::PriceUpdate {
            symbol: "BTCUSD".to_string(),
            bid: 45000.0,
            ask: 45050.0,
            last: 45025.0,
            volume: 1000.0,
            timestamp: Utc::now(),
        };

        manager.broadcast_event(event.clone()).await;

        // Receive the broadcasted event
        let received_event = rx.recv().await.unwrap();
        match received_event {
            MarketDataEvent::PriceUpdate {
                symbol, bid, ask, ..
            } => {
                assert_eq!(symbol, "BTCUSD");
                assert_eq!(bid, 45000.0);
                assert_eq!(ask, 45050.0);
            }
            _ => panic!("Unexpected event type"),
        }
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let (tx, _rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);
        let client_id = Uuid::new_v4();

        // Subscribe and then unsubscribe
        manager
            .subscribe(client_id, "trading://order_book/BTCUSD".to_string())
            .await
            .unwrap();

        let result = manager
            .unsubscribe(client_id, "trading://order_book/BTCUSD".to_string())
            .await;
        assert!(result.is_ok());

        let subs = manager.get_client_subscriptions(client_id).await;
        assert_eq!(subs.len(), 0);
    }

    #[tokio::test]
    async fn test_event_filtering() {
        let (tx, _rx) = broadcast::channel(100);
        let manager = SubscriptionManager::new(tx);

        // Create subscription with symbol filter
        let subscription = Subscription {
            client_id: Uuid::new_v4(),
            uri: "trading://prices/BTCUSD?symbol=BTCUSD".to_string(),
            event_type: SubscriptionType::Prices,
            filters: {
                let mut filters = HashMap::new();
                filters.insert("symbol".to_string(), "BTCUSD".to_string());
                Some(filters)
            },
            created_at: Utc::now(),
            last_update: Utc::now(),
        };

        // Test matching event
        let matching_event = MarketDataEvent::PriceUpdate {
            symbol: "BTCUSD".to_string(),
            bid: 45000.0,
            ask: 45050.0,
            last: 45025.0,
            volume: 1000.0,
            timestamp: Utc::now(),
        };

        assert!(manager.event_matches_subscription(&matching_event, &subscription));

        // Test non-matching event
        let non_matching_event = MarketDataEvent::PriceUpdate {
            symbol: "ETHUSD".to_string(),
            bid: 3000.0,
            ask: 3050.0,
            last: 3025.0,
            volume: 500.0,
            timestamp: Utc::now(),
        };

        assert!(!manager.event_matches_subscription(&non_matching_event, &subscription));
    }
}
