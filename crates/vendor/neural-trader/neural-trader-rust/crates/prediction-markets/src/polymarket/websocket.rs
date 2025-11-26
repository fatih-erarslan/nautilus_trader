//! WebSocket streaming for Polymarket real-time data

use crate::error::{PredictionMarketError, Result};
use crate::models::{Subscription, WebSocketMessage};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::broadcast;
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
use tracing::{debug, error, info, warn};
use url::Url;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

const DEFAULT_WS_URL: &str = "wss://ws.polymarket.com";

/// WebSocket subscription manager
pub struct PolymarketStream {
    ws_url: String,
    subscriptions: Arc<DashMap<String, Subscription>>,
    broadcast_tx: broadcast::Sender<WebSocketMessage>,
}

impl PolymarketStream {
    /// Create a new WebSocket stream manager
    pub fn new(ws_url: Option<String>) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);

        Self {
            ws_url: ws_url.unwrap_or_else(|| DEFAULT_WS_URL.to_string()),
            subscriptions: Arc::new(DashMap::new()),
            broadcast_tx,
        }
    }

    /// Connect to WebSocket server
    pub async fn connect(&self) -> Result<WsStream> {
        info!("Connecting to WebSocket: {}", self.ws_url);

        let url = Url::parse(&self.ws_url)
            .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

        let (ws_stream, _) = connect_async(url)
            .await
            .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

        info!("WebSocket connected");
        Ok(ws_stream)
    }

    /// Subscribe to market orderbook updates
    pub async fn subscribe_orderbook(
        &self,
        ws: &mut WsStream,
        market_id: &str,
        outcome_id: &str,
    ) -> Result<()> {
        let subscription = Subscription {
            channel: "orderbook".to_string(),
            market_id: Some(market_id.to_string()),
            outcome_id: Some(outcome_id.to_string()),
        };

        let sub_id = format!("orderbook:{}:{}", market_id, outcome_id);
        self.subscriptions.insert(sub_id.clone(), subscription.clone());

        let message = json!({
            "type": "subscribe",
            "channel": "orderbook",
            "market_id": market_id,
            "outcome_id": outcome_id,
        });

        ws.send(Message::Text(message.to_string()))
            .await
            .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

        info!("Subscribed to orderbook: {} / {}", market_id, outcome_id);
        Ok(())
    }

    /// Subscribe to market updates
    pub async fn subscribe_market(&self, ws: &mut WsStream, market_id: &str) -> Result<()> {
        let subscription = Subscription {
            channel: "market".to_string(),
            market_id: Some(market_id.to_string()),
            outcome_id: None,
        };

        let sub_id = format!("market:{}", market_id);
        self.subscriptions.insert(sub_id.clone(), subscription.clone());

        let message = json!({
            "type": "subscribe",
            "channel": "market",
            "market_id": market_id,
        });

        ws.send(Message::Text(message.to_string()))
            .await
            .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

        info!("Subscribed to market updates: {}", market_id);
        Ok(())
    }

    /// Subscribe to trade updates
    pub async fn subscribe_trades(
        &self,
        ws: &mut WsStream,
        market_id: &str,
        outcome_id: &str,
    ) -> Result<()> {
        let subscription = Subscription {
            channel: "trades".to_string(),
            market_id: Some(market_id.to_string()),
            outcome_id: Some(outcome_id.to_string()),
        };

        let sub_id = format!("trades:{}:{}", market_id, outcome_id);
        self.subscriptions.insert(sub_id.clone(), subscription.clone());

        let message = json!({
            "type": "subscribe",
            "channel": "trades",
            "market_id": market_id,
            "outcome_id": outcome_id,
        });

        ws.send(Message::Text(message.to_string()))
            .await
            .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

        info!("Subscribed to trades: {} / {}", market_id, outcome_id);
        Ok(())
    }

    /// Unsubscribe from a channel
    pub async fn unsubscribe(
        &self,
        ws: &mut WsStream,
        sub_id: &str,
    ) -> Result<()> {
        if let Some((_, subscription)) = self.subscriptions.remove(sub_id) {
            let message = json!({
                "type": "unsubscribe",
                "channel": subscription.channel,
                "market_id": subscription.market_id,
                "outcome_id": subscription.outcome_id,
            });

            ws.send(Message::Text(message.to_string()))
                .await
                .map_err(|e| PredictionMarketError::WebSocketError(e.to_string()))?;

            info!("Unsubscribed from: {}", sub_id);
        }

        Ok(())
    }

    /// Get a receiver for broadcast messages
    pub fn subscribe_updates(&self) -> broadcast::Receiver<WebSocketMessage> {
        self.broadcast_tx.subscribe()
    }

    /// Process incoming WebSocket messages
    pub async fn process_message(&self, message: Message) -> Result<()> {
        match message {
            Message::Text(text) => {
                debug!("Received WebSocket message: {}", text);

                match serde_json::from_str::<WebSocketMessage>(&text) {
                    Ok(ws_msg) => {
                        if let Err(e) = self.broadcast_tx.send(ws_msg) {
                            warn!("Failed to broadcast message: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse WebSocket message: {}", e);
                    }
                }
            }
            Message::Binary(_) => {
                debug!("Received binary WebSocket message");
            }
            Message::Ping(_data) => {
                debug!("Received ping, sending pong");
                // Pong is handled automatically by tokio-tungstenite
            }
            Message::Pong(_) => {
                debug!("Received pong");
            }
            Message::Close(frame) => {
                info!("WebSocket closing: {:?}", frame);
            }
            _ => {}
        }

        Ok(())
    }

    /// Run the WebSocket message processing loop
    pub async fn run(&self, mut ws: WsStream) -> Result<()> {
        info!("Starting WebSocket message processing loop");

        while let Some(msg_result) = ws.next().await {
            match msg_result {
                Ok(message) => {
                    if let Err(e) = self.process_message(message).await {
                        error!("Error processing message: {}", e);
                    }
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    return Err(PredictionMarketError::WebSocketError(e.to_string()));
                }
            }
        }

        warn!("WebSocket connection closed");
        Ok(())
    }

    /// Get active subscriptions
    pub fn get_subscriptions(&self) -> Vec<(String, Subscription)> {
        self.subscriptions
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Clear all subscriptions
    pub fn clear_subscriptions(&self) {
        self.subscriptions.clear();
        info!("Cleared all subscriptions");
    }
}

/// WebSocket stream builder for easy configuration
pub struct StreamBuilder {
    ws_url: Option<String>,
}

impl StreamBuilder {
    pub fn new() -> Self {
        Self { ws_url: None }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = Some(url.into());
        self
    }

    pub fn build(self) -> PolymarketStream {
        PolymarketStream::new(self.ws_url)
    }
}

impl Default for StreamBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_builder() {
        let stream = StreamBuilder::new()
            .with_url("wss://test.com")
            .build();

        assert_eq!(stream.ws_url, "wss://test.com");
    }

    #[test]
    fn test_subscription_management() {
        let stream = PolymarketStream::new(None);

        let sub = Subscription {
            channel: "test".to_string(),
            market_id: Some("market1".to_string()),
            outcome_id: None,
        };

        stream.subscriptions.insert("test:market1".to_string(), sub.clone());

        assert_eq!(stream.subscriptions.len(), 1);
        assert!(stream.subscriptions.contains_key("test:market1"));

        stream.clear_subscriptions();
        assert_eq!(stream.subscriptions.len(), 0);
    }
}
