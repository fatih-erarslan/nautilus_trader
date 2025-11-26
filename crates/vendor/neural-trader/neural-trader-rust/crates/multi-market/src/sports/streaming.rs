//! Real-time Odds Streaming
//!
//! WebSocket-based streaming of live odds updates

use crate::error::{MultiMarketError, Result};
use crate::sports::odds_api::{Event, Sport};
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// Odds update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OddsUpdate {
    /// Event identifier
    pub event_id: String,
    /// Sport
    pub sport: String,
    /// Updated event data
    pub event: Event,
    /// Update timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Stream subscription filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamFilter {
    /// Sports to subscribe to
    pub sports: Vec<Sport>,
    /// Event IDs to subscribe to (optional)
    pub event_ids: Option<Vec<String>>,
    /// Bookmakers to filter (optional)
    pub bookmakers: Option<Vec<String>>,
}

/// Odds streaming client
#[async_trait]
pub trait OddsStreamClient: Send + Sync {
    /// Start streaming odds updates
    async fn start_streaming(&mut self, filter: StreamFilter) -> Result<()>;

    /// Stop streaming
    async fn stop_streaming(&mut self) -> Result<()>;

    /// Subscribe to odds updates
    fn subscribe(&self) -> broadcast::Receiver<OddsUpdate>;

    /// Check if streaming is active
    fn is_streaming(&self) -> bool;
}

/// Polling-based odds streamer (fallback for APIs without WebSocket)
pub struct PollingOddsStreamer {
    api_client: Arc<crate::sports::odds_api::OddsApiClient>,
    update_sender: broadcast::Sender<OddsUpdate>,
    is_running: Arc<RwLock<bool>>,
    poll_interval_secs: u64,
}

impl PollingOddsStreamer {
    /// Create new polling streamer
    pub fn new(api_client: crate::sports::odds_api::OddsApiClient, poll_interval_secs: u64) -> Self {
        let (update_sender, _) = broadcast::channel(1000);

        Self {
            api_client: Arc::new(api_client),
            update_sender,
            is_running: Arc::new(RwLock::new(false)),
            poll_interval_secs,
        }
    }

    /// Poll odds for a sport
    async fn poll_sport(&self, sport: Sport, filter: &StreamFilter) -> Result<Vec<OddsUpdate>> {
        let events = self
            .api_client
            .get_odds(
                sport,
                &[
                    crate::sports::odds_api::Market::H2h,
                    crate::sports::odds_api::Market::Spreads,
                    crate::sports::odds_api::Market::Totals,
                ],
                &["us", "uk", "au"],
            )
            .await?;

        let mut updates = Vec::new();

        for event in events {
            // Apply event ID filter if specified
            if let Some(ref event_ids) = filter.event_ids {
                if !event_ids.contains(&event.id) {
                    continue;
                }
            }

            // Apply bookmaker filter if specified
            if let Some(ref bookmakers) = filter.bookmakers {
                let event_bookmakers: Vec<String> = event
                    .bookmaker_odds
                    .values()
                    .flatten()
                    .map(|odds| odds.bookmaker.clone())
                    .collect();

                if !bookmakers
                    .iter()
                    .any(|b| event_bookmakers.contains(b))
                {
                    continue;
                }
            }

            updates.push(OddsUpdate {
                event_id: event.id.clone(),
                sport: event.sport.clone(),
                event,
                timestamp: chrono::Utc::now(),
            });
        }

        Ok(updates)
    }

    /// Start polling loop
    async fn polling_loop(self: Arc<Self>, filter: StreamFilter) {
        info!("Starting polling loop for odds updates");

        while *self.is_running.read().await {
            for sport in &filter.sports {
                match self.poll_sport(*sport, &filter).await {
                    Ok(updates) => {
                        for update in updates {
                            if let Err(e) = self.update_sender.send(update) {
                                debug!("No receivers for update: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error polling odds for {:?}: {}", sport, e);
                    }
                }
            }

            // Wait before next poll
            tokio::time::sleep(tokio::time::Duration::from_secs(self.poll_interval_secs)).await;
        }

        info!("Polling loop stopped");
    }
}

#[async_trait]
impl OddsStreamClient for PollingOddsStreamer {
    async fn start_streaming(&mut self, filter: StreamFilter) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(MultiMarketError::ValidationError(
                "Streaming already active".to_string(),
            ));
        }

        *is_running = true;
        drop(is_running);

        // Spawn polling task
        let self_arc = Arc::new(self.clone());
        tokio::spawn(async move {
            self_arc.polling_loop(filter).await;
        });

        info!("Started polling-based odds streaming");
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        info!("Stopped odds streaming");
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<OddsUpdate> {
        self.update_sender.subscribe()
    }

    fn is_streaming(&self) -> bool {
        // Use try_read to avoid blocking
        self.is_running.try_read().map(|r| *r).unwrap_or(false)
    }
}

impl Clone for PollingOddsStreamer {
    fn clone(&self) -> Self {
        Self {
            api_client: Arc::clone(&self.api_client),
            update_sender: self.update_sender.clone(),
            is_running: Arc::clone(&self.is_running),
            poll_interval_secs: self.poll_interval_secs,
        }
    }
}

/// WebSocket-based odds streamer (for APIs with WebSocket support)
pub struct WebSocketOddsStreamer {
    ws_url: String,
    update_sender: broadcast::Sender<OddsUpdate>,
    is_running: Arc<RwLock<bool>>,
}

impl WebSocketOddsStreamer {
    /// Create new WebSocket streamer
    pub fn new(ws_url: String) -> Self {
        let (update_sender, _) = broadcast::channel(1000);

        Self {
            ws_url,
            update_sender,
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// WebSocket connection loop
    async fn websocket_loop(self: Arc<Self>, filter: StreamFilter) {
        info!("Connecting to WebSocket: {}", self.ws_url);

        loop {
            if !*self.is_running.read().await {
                break;
            }

            match connect_async(&self.ws_url).await {
                Ok((ws_stream, _)) => {
                    info!("WebSocket connected");

                    let (mut write, mut read) = ws_stream.split();

                    // Send subscription message
                    let sub_msg = serde_json::to_string(&filter).unwrap();
                    if let Err(e) = write.send(Message::Text(sub_msg)).await {
                        error!("Failed to send subscription: {}", e);
                        continue;
                    }

                    // Process incoming messages
                    while let Some(msg) = read.next().await {
                        if !*self.is_running.read().await {
                            break;
                        }

                        match msg {
                            Ok(Message::Text(text)) => {
                                match serde_json::from_str::<OddsUpdate>(&text) {
                                    Ok(update) => {
                                        if let Err(e) = self.update_sender.send(update) {
                                            debug!("No receivers for update: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse update: {}", e);
                                    }
                                }
                            }
                            Ok(Message::Ping(data)) => {
                                if let Err(e) = write.send(Message::Pong(data)).await {
                                    error!("Failed to send pong: {}", e);
                                    break;
                                }
                            }
                            Ok(Message::Close(_)) => {
                                info!("WebSocket closed by server");
                                break;
                            }
                            Err(e) => {
                                error!("WebSocket error: {}", e);
                                break;
                            }
                            _ => {}
                        }
                    }

                    info!("WebSocket connection closed");
                }
                Err(e) => {
                    error!("Failed to connect to WebSocket: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }

            // Reconnect delay
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        info!("WebSocket loop stopped");
    }
}

#[async_trait]
impl OddsStreamClient for WebSocketOddsStreamer {
    async fn start_streaming(&mut self, filter: StreamFilter) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(MultiMarketError::ValidationError(
                "Streaming already active".to_string(),
            ));
        }

        *is_running = true;
        drop(is_running);

        // Spawn WebSocket task
        let self_arc = Arc::new(self.clone());
        tokio::spawn(async move {
            self_arc.websocket_loop(filter).await;
        });

        info!("Started WebSocket odds streaming");
        Ok(())
    }

    async fn stop_streaming(&mut self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        *is_running = false;
        info!("Stopped WebSocket streaming");
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<OddsUpdate> {
        self.update_sender.subscribe()
    }

    fn is_streaming(&self) -> bool {
        self.is_running.try_read().map(|r| *r).unwrap_or(false)
    }
}

impl Clone for WebSocketOddsStreamer {
    fn clone(&self) -> Self {
        Self {
            ws_url: self.ws_url.clone(),
            update_sender: self.update_sender.clone(),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_polling_streamer_creation() {
        let api_client = crate::sports::odds_api::OddsApiClient::new("test_key");
        let streamer = PollingOddsStreamer::new(api_client, 30);

        assert!(!streamer.is_streaming());
    }

    #[tokio::test]
    async fn test_websocket_streamer_creation() {
        let streamer = WebSocketOddsStreamer::new("wss://example.com/odds".to_string());

        assert!(!streamer.is_streaming());
    }

    #[tokio::test]
    async fn test_subscription() {
        let api_client = crate::sports::odds_api::OddsApiClient::new("test_key");
        let streamer = PollingOddsStreamer::new(api_client, 30);

        let mut receiver = streamer.subscribe();

        // Test that receiver is ready
        assert!(receiver.try_recv().is_err()); // No messages yet
    }
}
