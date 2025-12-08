//! Cross-scale communication protocols

use crate::{
    config::{PadsConfig, CommunicationConfig},
    error::{PadsError, Result},
    monitoring::PadsMonitor,
    types::*,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, broadcast, RwLock};
use tracing::{debug, info, warn};

/// Manages cross-scale communication
pub struct CrossScaleCommunicator {
    config: Arc<PadsConfig>,
    monitor: Arc<PadsMonitor>,
    channels: DashMap<ChannelId, CommunicationChannel>,
    message_router: Arc<MessageRouter>,
    protocol_handler: Arc<ProtocolHandler>,
}

/// Channel identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct ChannelId {
    from_scale: ScaleLevel,
    to_scale: ScaleLevel,
    channel_type: ChannelType,
}

/// Channel type
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
enum ChannelType {
    Upward,
    Downward,
    Lateral,
    Broadcast,
}

/// Communication channel
struct CommunicationChannel {
    id: ChannelId,
    sender: mpsc::Sender<Message>,
    receiver: Arc<RwLock<mpsc::Receiver<Message>>>,
    metrics: ChannelMetrics,
}

/// Message structure
#[derive(Debug, Clone)]
struct Message {
    id: String,
    from_scale: ScaleLevel,
    to_scale: ScaleLevel,
    message_type: MessageType,
    payload: serde_json::Value,
    timestamp: chrono::DateTime<chrono::Utc>,
    priority: f64,
}

/// Message type
#[derive(Debug, Clone)]
enum MessageType {
    ScaleEffect(ScaleEffect),
    Coordination(CoordinationMessage),
    Notification(NotificationMessage),
    Query(QueryMessage),
    Response(ResponseMessage),
}

/// Coordination message
#[derive(Debug, Clone)]
struct CoordinationMessage {
    action: String,
    parameters: serde_json::Value,
    requires_response: bool,
}

/// Notification message
#[derive(Debug, Clone)]
struct NotificationMessage {
    event_type: String,
    details: serde_json::Value,
}

/// Query message
#[derive(Debug, Clone)]
struct QueryMessage {
    query_type: String,
    parameters: serde_json::Value,
}

/// Response message
#[derive(Debug, Clone)]
struct ResponseMessage {
    query_id: String,
    result: serde_json::Value,
    success: bool,
}

/// Channel metrics
#[derive(Debug, Clone, Default)]
struct ChannelMetrics {
    messages_sent: u64,
    messages_received: u64,
    errors: u64,
    avg_latency_ms: f64,
}

/// Message router
struct MessageRouter {
    routing_table: DashMap<ScaleLevel, Vec<ChannelId>>,
    broadcast_channel: broadcast::Sender<Message>,
}

/// Protocol handler
struct ProtocolHandler {
    compression_enabled: bool,
    encryption_enabled: bool,
}

impl CrossScaleCommunicator {
    /// Create new communicator
    pub async fn new(config: Arc<PadsConfig>, monitor: Arc<PadsMonitor>) -> Result<Self> {
        let channels = DashMap::new();
        
        let (broadcast_tx, _) = broadcast::channel(1000);
        let message_router = Arc::new(MessageRouter {
            routing_table: DashMap::new(),
            broadcast_channel: broadcast_tx,
        });
        
        let protocol_handler = Arc::new(ProtocolHandler {
            compression_enabled: config.comm_config.enable_compression,
            encryption_enabled: config.comm_config.enable_encryption,
        });
        
        Ok(Self {
            config,
            monitor,
            channels,
            message_router,
            protocol_handler,
        })
    }
    
    /// Setup communication channels
    pub async fn setup_channels(&self) -> Result<()> {
        info!("Setting up cross-scale communication channels");
        
        // Create channels between all scale pairs
        let scales = vec![ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro];
        
        for from_scale in &scales {
            for to_scale in &scales {
                if from_scale != to_scale {
                    // Determine channel type
                    let channel_type = match (from_scale, to_scale) {
                        (ScaleLevel::Micro, ScaleLevel::Meso) |
                        (ScaleLevel::Meso, ScaleLevel::Macro) => ChannelType::Upward,
                        (ScaleLevel::Macro, ScaleLevel::Meso) |
                        (ScaleLevel::Meso, ScaleLevel::Micro) => ChannelType::Downward,
                        _ => ChannelType::Lateral,
                    };
                    
                    self.create_channel(*from_scale, *to_scale, channel_type).await?;
                }
            }
            
            // Create broadcast channel for each scale
            self.create_channel(*from_scale, *from_scale, ChannelType::Broadcast).await?;
        }
        
        // Setup routing table
        self.setup_routing_table().await?;
        
        Ok(())
    }
    
    /// Create a communication channel
    async fn create_channel(
        &self,
        from_scale: ScaleLevel,
        to_scale: ScaleLevel,
        channel_type: ChannelType
    ) -> Result<()> {
        let buffer_size = self.config.comm_config.channel_buffer_size;
        let (tx, rx) = mpsc::channel(buffer_size);
        
        let channel_id = ChannelId {
            from_scale,
            to_scale,
            channel_type,
        };
        
        let channel = CommunicationChannel {
            id: channel_id.clone(),
            sender: tx,
            receiver: Arc::new(RwLock::new(rx)),
            metrics: ChannelMetrics::default(),
        };
        
        self.channels.insert(channel_id, channel);
        
        debug!("Created channel {:?} -> {:?} ({:?})", from_scale, to_scale, channel_type);
        
        Ok(())
    }
    
    /// Setup routing table
    async fn setup_routing_table(&self) -> Result<()> {
        for channel_ref in self.channels.iter() {
            let channel_id = channel_ref.key();
            let from_scale = channel_id.from_scale;
            
            self.message_router.routing_table
                .entry(from_scale)
                .or_insert_with(Vec::new)
                .push(channel_id.clone());
        }
        
        Ok(())
    }
    
    /// Broadcast scale transition
    pub async fn broadcast_scale_transition(&self, from_scale: ScaleLevel) -> Result<()> {
        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            from_scale,
            to_scale: from_scale, // Broadcast to all
            message_type: MessageType::Notification(NotificationMessage {
                event_type: "scale_transition".to_string(),
                details: serde_json::json!({
                    "from_scale": from_scale,
                    "timestamp": chrono::Utc::now(),
                }),
            }),
            timestamp: chrono::Utc::now(),
            priority: 1.0,
        };
        
        self.message_router.broadcast_channel.send(message)
            .map_err(|_| PadsError::comm("Broadcast failed"))?;
        
        Ok(())
    }
    
    /// Propagate effects upward
    pub async fn propagate_upward(&self, effects: &[ScaleEffect]) -> Result<()> {
        for effect in effects {
            let message = Message {
                id: uuid::Uuid::new_v4().to_string(),
                from_scale: effect.target_scale,
                to_scale: self.get_upward_scale(effect.target_scale)?,
                message_type: MessageType::ScaleEffect(effect.clone()),
                timestamp: chrono::Utc::now(),
                priority: effect.magnitude,
            };
            
            self.send_message(message).await?;
        }
        
        Ok(())
    }
    
    /// Propagate effects downward
    pub async fn propagate_downward(&self, effects: &[ScaleEffect]) -> Result<()> {
        for effect in effects {
            let message = Message {
                id: uuid::Uuid::new_v4().to_string(),
                from_scale: effect.target_scale,
                to_scale: self.get_downward_scale(effect.target_scale)?,
                message_type: MessageType::ScaleEffect(effect.clone()),
                timestamp: chrono::Utc::now(),
                priority: effect.magnitude,
            };
            
            self.send_message(message).await?;
        }
        
        Ok(())
    }
    
    /// Coordinate with specific scale
    pub async fn coordinate_with_scale(
        &self,
        target_scale: PanarchyScale,
        result: &DecisionResult
    ) -> Result<()> {
        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            from_scale: result.scale_level,
            to_scale: target_scale.level,
            message_type: MessageType::Coordination(CoordinationMessage {
                action: "sync_decision".to_string(),
                parameters: serde_json::to_value(result)?,
                requires_response: false,
            }),
            timestamp: chrono::Utc::now(),
            priority: 0.5,
        };
        
        self.send_message(message).await?;
        
        Ok(())
    }
    
    /// Send a message
    async fn send_message(&self, message: Message) -> Result<()> {
        let channel_id = ChannelId {
            from_scale: message.from_scale,
            to_scale: message.to_scale,
            channel_type: self.determine_channel_type(&message),
        };
        
        let channel = self.channels.get(&channel_id)
            .ok_or_else(|| PadsError::comm("Channel not found"))?;
        
        // Apply protocol handling
        let processed_message = self.protocol_handler.process_outgoing(message)?;
        
        // Send with timeout
        let timeout = self.config.comm_config.message_timeout;
        match tokio::time::timeout(timeout, channel.sender.send(processed_message)).await {
            Ok(Ok(_)) => {
                self.monitor.record_message_sent(channel_id.from_scale, channel_id.to_scale);
                Ok(())
            }
            Ok(Err(_)) => Err(PadsError::ChannelSend),
            Err(_) => Err(PadsError::timeout("Message send timeout")),
        }
    }
    
    /// Determine channel type for message
    fn determine_channel_type(&self, message: &Message) -> ChannelType {
        match (&message.from_scale, &message.to_scale) {
            (from, to) if from == to => ChannelType::Broadcast,
            (ScaleLevel::Micro, ScaleLevel::Meso) |
            (ScaleLevel::Meso, ScaleLevel::Macro) => ChannelType::Upward,
            (ScaleLevel::Macro, ScaleLevel::Meso) |
            (ScaleLevel::Meso, ScaleLevel::Micro) => ChannelType::Downward,
            _ => ChannelType::Lateral,
        }
    }
    
    /// Get upward scale
    fn get_upward_scale(&self, scale: ScaleLevel) -> Result<ScaleLevel> {
        match scale {
            ScaleLevel::Micro => Ok(ScaleLevel::Meso),
            ScaleLevel::Meso => Ok(ScaleLevel::Macro),
            ScaleLevel::Macro => Err(PadsError::comm("No upward scale from Macro")),
        }
    }
    
    /// Get downward scale
    fn get_downward_scale(&self, scale: ScaleLevel) -> Result<ScaleLevel> {
        match scale {
            ScaleLevel::Macro => Ok(ScaleLevel::Meso),
            ScaleLevel::Meso => Ok(ScaleLevel::Micro),
            ScaleLevel::Micro => Err(PadsError::comm("No downward scale from Micro")),
        }
    }
    
    /// Reconnect all channels
    pub async fn reconnect_all(&self) -> Result<()> {
        warn!("Reconnecting all communication channels");
        
        // Clear existing channels
        self.channels.clear();
        
        // Re-setup channels
        self.setup_channels().await?;
        
        Ok(())
    }
    
    /// Get communication status
    pub async fn get_status(&self) -> Result<CommunicationStatus> {
        let active_channels = self.channels.len();
        
        let (total_sent, total_received, total_errors) = self.channels.iter()
            .map(|ch| {
                let metrics = &ch.value().metrics;
                (metrics.messages_sent, metrics.messages_received, metrics.errors)
            })
            .fold((0, 0, 0), |(s, r, e), (sent, recv, err)| {
                (s + sent, r + recv, e + err)
            });
        
        let message_rate = total_sent as f64 / 60.0; // Per minute
        let error_rate = if total_sent > 0 {
            total_errors as f64 / total_sent as f64
        } else {
            0.0
        };
        
        Ok(CommunicationStatus {
            active_channels,
            message_rate,
            error_rate,
            avg_latency_ms: 5.0, // Would calculate actual
        })
    }
}

impl ProtocolHandler {
    /// Process outgoing message
    fn process_outgoing(&self, mut message: Message) -> Result<Message> {
        if self.compression_enabled {
            // Apply compression (simplified)
            debug!("Compressing message {}", message.id);
        }
        
        if self.encryption_enabled {
            // Apply encryption (simplified)
            debug!("Encrypting message {}", message.id);
        }
        
        Ok(message)
    }
    
    /// Process incoming message
    fn process_incoming(&self, mut message: Message) -> Result<Message> {
        if self.encryption_enabled {
            // Decrypt (simplified)
            debug!("Decrypting message {}", message.id);
        }
        
        if self.compression_enabled {
            // Decompress (simplified)
            debug!("Decompressing message {}", message.id);
        }
        
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_channel_setup() {
        let config = Arc::new(PadsConfig::default());
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await.unwrap());
        let comm = CrossScaleCommunicator::new(config, monitor).await.unwrap();
        
        assert!(comm.setup_channels().await.is_ok());
        assert!(comm.channels.len() > 0);
    }
}