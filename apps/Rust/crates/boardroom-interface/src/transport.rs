//! Transport layer implementations for Redis and ZeroMQ

use crate::message::Message;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Transport trait for message passing
#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    /// Connect to the transport
    async fn connect(&mut self) -> Result<()>;

    /// Disconnect from the transport
    async fn disconnect(&mut self) -> Result<()>;

    /// Send a message
    async fn send(&self, message: &Message) -> Result<()>;

    /// Receive a message
    async fn receive(&self) -> Result<Option<Message>>;

    /// Subscribe to a channel/topic
    async fn subscribe(&self, topic: &str) -> Result<()>;

    /// Unsubscribe from a channel/topic
    async fn unsubscribe(&self, topic: &str) -> Result<()>;

    /// Check if connected
    async fn is_connected(&self) -> bool;
}

/// Redis-based transport
pub struct RedisTransport {
    client: Option<redis::Client>,
    connection: Arc<Mutex<Option<redis::aio::MultiplexedConnection>>>,
    pubsub: Arc<Mutex<Option<redis::aio::PubSub>>>,
    config: RedisConfig,
}

#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
    pub timeout_ms: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            pool_size: 10,
            timeout_ms: 5000,
        }
    }
}

impl RedisTransport {
    /// Create new Redis transport
    pub fn new(config: RedisConfig) -> Self {
        Self {
            client: None,
            connection: Arc::new(Mutex::new(None)),
            pubsub: Arc::new(Mutex::new(None)),
            config,
        }
    }

    /// Get a connection
    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        let mut conn_guard = self.connection.lock().await;
        
        if let Some(conn) = conn_guard.as_ref() {
            Ok(conn.clone())
        } else {
            anyhow::bail!("Not connected to Redis")
        }
    }
}

#[async_trait::async_trait]
impl Transport for RedisTransport {
    async fn connect(&mut self) -> Result<()> {
        let client = redis::Client::open(self.config.url.as_str())?;
        let conn = client.get_multiplexed_async_connection().await?;
        let pubsub = client.get_async_pubsub().await?;
        
        self.client = Some(client);
        *self.connection.lock().await = Some(conn);
        *self.pubsub.lock().await = Some(pubsub);
        
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        *self.connection.lock().await = None;
        *self.pubsub.lock().await = None;
        self.client = None;
        Ok(())
    }

    async fn send(&self, message: &Message) -> Result<()> {
        let mut conn = self.get_connection().await?;
        let serialized = serde_json::to_string(message)?;
        
        // Send to specific agent or broadcast
        let channel = if let Some(to) = &message.to {
            format!("agent:{}", to)
        } else {
            "broadcast".to_string()
        };
        
        redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(&serialized)
            .query_async(&mut conn)
            .await?;
        
        Ok(())
    }

    async fn receive(&self) -> Result<Option<Message>> {
        let mut pubsub_guard = self.pubsub.lock().await;
        
        if let Some(pubsub) = pubsub_guard.as_mut() {
            match pubsub.on_message().next().await {
                Some(msg) => {
                    let payload: String = msg.get_payload()?;
                    let message: Message = serde_json::from_str(&payload)?;
                    Ok(Some(message))
                }
                None => Ok(None),
            }
        } else {
            anyhow::bail!("PubSub not initialized")
        }
    }

    async fn subscribe(&self, topic: &str) -> Result<()> {
        let mut pubsub_guard = self.pubsub.lock().await;
        
        if let Some(pubsub) = pubsub_guard.as_mut() {
            pubsub.subscribe(topic).await?;
            Ok(())
        } else {
            anyhow::bail!("PubSub not initialized")
        }
    }

    async fn unsubscribe(&self, topic: &str) -> Result<()> {
        let mut pubsub_guard = self.pubsub.lock().await;
        
        if let Some(pubsub) = pubsub_guard.as_mut() {
            pubsub.unsubscribe(topic).await?;
            Ok(())
        } else {
            anyhow::bail!("PubSub not initialized")
        }
    }

    async fn is_connected(&self) -> bool {
        self.connection.lock().await.is_some()
    }
}

/// ZeroMQ-based transport
pub struct ZmqTransport {
    context: zeromq::Context,
    socket: Arc<Mutex<Option<zeromq::Socket>>>,
    config: ZmqConfig,
}

#[derive(Debug, Clone)]
pub struct ZmqConfig {
    pub endpoint: String,
    pub socket_type: ZmqSocketType,
    pub high_water_mark: i32,
    pub timeout_ms: i32,
}

#[derive(Debug, Clone, Copy)]
pub enum ZmqSocketType {
    Pub,
    Sub,
    Push,
    Pull,
    Dealer,
    Router,
}

impl Default for ZmqConfig {
    fn default() -> Self {
        Self {
            endpoint: "tcp://127.0.0.1:5555".to_string(),
            socket_type: ZmqSocketType::Dealer,
            high_water_mark: 1000,
            timeout_ms: 5000,
        }
    }
}

impl ZmqTransport {
    /// Create new ZeroMQ transport
    pub fn new(config: ZmqConfig) -> Self {
        Self {
            context: zeromq::Context::new(),
            socket: Arc::new(Mutex::new(None)),
            config,
        }
    }

    fn get_socket_type(&self) -> zeromq::SocketType {
        match self.config.socket_type {
            ZmqSocketType::Pub => zeromq::SocketType::PUB,
            ZmqSocketType::Sub => zeromq::SocketType::SUB,
            ZmqSocketType::Push => zeromq::SocketType::PUSH,
            ZmqSocketType::Pull => zeromq::SocketType::PULL,
            ZmqSocketType::Dealer => zeromq::SocketType::DEALER,
            ZmqSocketType::Router => zeromq::SocketType::ROUTER,
        }
    }
}

#[async_trait::async_trait]
impl Transport for ZmqTransport {
    async fn connect(&mut self) -> Result<()> {
        let socket = self.context.socket(self.get_socket_type())?;
        
        // Set socket options
        socket.set_rcvhwm(self.config.high_water_mark)?;
        socket.set_sndhwm(self.config.high_water_mark)?;
        socket.set_rcvtimeo(self.config.timeout_ms)?;
        socket.set_sndtimeo(self.config.timeout_ms)?;
        
        // Connect or bind based on socket type
        match self.config.socket_type {
            ZmqSocketType::Router | ZmqSocketType::Pull | ZmqSocketType::Sub => {
                socket.bind(&self.config.endpoint)?;
            }
            _ => {
                socket.connect(&self.config.endpoint)?;
            }
        }
        
        *self.socket.lock().await = Some(socket);
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        if let Some(socket) = self.socket.lock().await.take() {
            drop(socket);
        }
        Ok(())
    }

    async fn send(&self, message: &Message) -> Result<()> {
        let socket_guard = self.socket.lock().await;
        
        if let Some(socket) = socket_guard.as_ref() {
            let serialized = bincode::serialize(message)?;
            socket.send(&serialized, 0)?;
            Ok(())
        } else {
            anyhow::bail!("Socket not connected")
        }
    }

    async fn receive(&self) -> Result<Option<Message>> {
        let socket_guard = self.socket.lock().await;
        
        if let Some(socket) = socket_guard.as_ref() {
            match socket.recv_bytes(zeromq::DONTWAIT) {
                Ok(bytes) => {
                    let message: Message = bincode::deserialize(&bytes)?;
                    Ok(Some(message))
                }
                Err(zeromq::Error::EAGAIN) => Ok(None),
                Err(e) => Err(e.into()),
            }
        } else {
            anyhow::bail!("Socket not connected")
        }
    }

    async fn subscribe(&self, topic: &str) -> Result<()> {
        let socket_guard = self.socket.lock().await;
        
        if let Some(socket) = socket_guard.as_ref() {
            if matches!(self.config.socket_type, ZmqSocketType::Sub) {
                socket.set_subscribe(topic.as_bytes())?;
            }
            Ok(())
        } else {
            anyhow::bail!("Socket not connected")
        }
    }

    async fn unsubscribe(&self, topic: &str) -> Result<()> {
        let socket_guard = self.socket.lock().await;
        
        if let Some(socket) = socket_guard.as_ref() {
            if matches!(self.config.socket_type, ZmqSocketType::Sub) {
                socket.set_unsubscribe(topic.as_bytes())?;
            }
            Ok(())
        } else {
            anyhow::bail!("Socket not connected")
        }
    }

    async fn is_connected(&self) -> bool {
        self.socket.lock().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_transport_creation() {
        let config = RedisConfig::default();
        let transport = RedisTransport::new(config);
        assert!(!transport.is_connected().await);
    }

    #[tokio::test]
    async fn test_zmq_transport_creation() {
        let config = ZmqConfig::default();
        let transport = ZmqTransport::new(config);
        assert!(!transport.is_connected().await);
    }
}