//! # Multi-Transport Messaging System
//!
//! Adaptive routing between Redis and ZeroMQ transports with failover

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use redis::AsyncCommands;
use zmq::{Context, Socket, SocketType};

/// Multi-transport messaging system
#[derive(Debug)]
pub struct MultiTransportMessaging {
    /// Configuration
    config: MessagingConfig,
    
    /// Redis transport
    redis_transport: Arc<RwLock<Option<RedisTransport>>>,
    
    /// ZeroMQ transport
    zmq_transport: Arc<RwLock<Option<ZmqTransport>>>,
    
    /// Message router
    router: Arc<RwLock<MessageRouter>>,
    
    /// Transport health monitor
    health_monitor: Arc<RwLock<TransportHealthMonitor>>,
    
    /// Registered components
    components: Arc<RwLock<HashMap<String, ComponentInfo>>>,
    
    /// Message queue
    message_queue: Arc<RwLock<MessageQueue>>,
}

/// Messaging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagingConfig {
    /// Redis configuration
    pub redis: RedisConfig,
    
    /// ZeroMQ configuration
    pub zmq: ZmqConfig,
    
    /// Routing configuration
    pub routing: RoutingConfig,
    
    /// Message queue configuration
    pub queue: QueueConfig,
    
    /// Health monitoring configuration
    pub health: HealthConfig,
}

impl Default for MessagingConfig {
    fn default() -> Self {
        Self {
            redis: RedisConfig::default(),
            zmq: ZmqConfig::default(),
            routing: RoutingConfig::default(),
            queue: QueueConfig::default(),
            health: HealthConfig::default(),
        }
    }
}

/// Redis transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: u32,
    pub connection_timeout: Duration,
    pub retry_attempts: u32,
    pub retry_delay: Duration,
    pub enabled: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            pool_size: 10,
            connection_timeout: Duration::from_secs(5),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(500),
            enabled: true,
        }
    }
}

/// ZeroMQ transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZmqConfig {
    pub dealer_endpoint: String,
    pub router_endpoint: String,
    pub pub_endpoint: String,
    pub sub_endpoint: String,
    pub high_water_mark: i32,
    pub linger: i32,
    pub enabled: bool,
}

impl Default for ZmqConfig {
    fn default() -> Self {
        Self {
            dealer_endpoint: "tcp://localhost:5555".to_string(),
            router_endpoint: "tcp://localhost:5556".to_string(),
            pub_endpoint: "tcp://localhost:5557".to_string(),
            sub_endpoint: "tcp://localhost:5558".to_string(),
            high_water_mark: 1000,
            linger: 100,
            enabled: true,
        }
    }
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub default_transport: TransportType,
    pub fallback_transport: TransportType,
    pub adaptive_routing: bool,
    pub load_balancing: bool,
    pub failover_threshold: f64,
    pub failover_timeout: Duration,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_transport: TransportType::Redis,
            fallback_transport: TransportType::ZeroMQ,
            adaptive_routing: true,
            load_balancing: true,
            failover_threshold: 0.1,
            failover_timeout: Duration::from_secs(30),
        }
    }
}

/// Message queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    pub max_queue_size: usize,
    pub message_timeout: Duration,
    pub batch_size: usize,
    pub batch_timeout: Duration,
    pub persistence: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            message_timeout: Duration::from_secs(30),
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
            persistence: true,
        }
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub check_interval: Duration,
    pub timeout: Duration,
    pub max_failures: u32,
    pub recovery_timeout: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            max_failures: 3,
            recovery_timeout: Duration::from_secs(60),
        }
    }
}

/// Transport type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportType {
    Redis,
    ZeroMQ,
}

/// Message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message ID
    pub id: String,
    
    /// Message type
    pub message_type: String,
    
    /// Source component
    pub source: String,
    
    /// Target component(s)
    pub target: MessageTarget,
    
    /// Message payload
    pub payload: serde_json::Value,
    
    /// Priority (0.0 = low, 1.0 = high)
    pub priority: f64,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Expiration time
    pub expires_at: Option<Instant>,
    
    /// Retry count
    pub retry_count: u32,
    
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Message target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageTarget {
    Component(String),
    Group(String),
    Broadcast,
}

/// Redis transport implementation
#[derive(Debug)]
pub struct RedisTransport {
    client: redis::Client,
    connection: Option<redis::aio::Connection>,
    config: RedisConfig,
}

impl RedisTransport {
    pub async fn new(config: RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())?;
        
        Ok(Self {
            client,
            connection: None,
            config,
        })
    }
    
    pub async fn connect(&mut self) -> Result<()> {
        let conn = self.client.get_async_connection().await?;
        self.connection = Some(conn);
        Ok(())
    }
    
    pub async fn send_message(&mut self, message: &Message) -> Result<()> {
        if let Some(conn) = &mut self.connection {
            let serialized = serde_json::to_string(message)?;
            
            match &message.target {
                MessageTarget::Component(target) => {
                    let key = format!("component:{}", target);
                    conn.lpush(key, serialized).await?;
                }
                MessageTarget::Group(group) => {
                    let key = format!("group:{}", group);
                    conn.publish(key, serialized).await?;
                }
                MessageTarget::Broadcast => {
                    conn.publish("broadcast", serialized).await?;
                }
            }
        }
        Ok(())
    }
    
    pub async fn receive_message(&mut self, component_id: &str) -> Result<Option<Message>> {
        if let Some(conn) = &mut self.connection {
            let key = format!("component:{}", component_id);
            let result: Option<String> = conn.brpop(key, 1).await?;
            
            if let Some(serialized) = result {
                let message: Message = serde_json::from_str(&serialized)?;
                return Ok(Some(message));
            }
        }
        Ok(None)
    }
    
    pub async fn health_check(&mut self) -> Result<bool> {
        if let Some(conn) = &mut self.connection {
            let result: String = conn.ping().await?;
            Ok(result == "PONG")
        } else {
            Ok(false)
        }
    }
}

/// ZeroMQ transport implementation
#[derive(Debug)]
pub struct ZmqTransport {
    context: Context,
    dealer: Option<Socket>,
    router: Option<Socket>,
    publisher: Option<Socket>,
    subscriber: Option<Socket>,
    config: ZmqConfig,
}

impl ZmqTransport {
    pub fn new(config: ZmqConfig) -> Result<Self> {
        let context = Context::new();
        
        Ok(Self {
            context,
            dealer: None,
            router: None,
            publisher: None,
            subscriber: None,
            config,
        })
    }
    
    pub async fn connect(&mut self) -> Result<()> {
        // Create DEALER socket
        let dealer = self.context.socket(SocketType::DEALER)?;
        dealer.set_sndhwm(self.config.high_water_mark)?;
        dealer.set_rcvhwm(self.config.high_water_mark)?;
        dealer.set_linger(self.config.linger)?;
        dealer.connect(&self.config.dealer_endpoint)?;
        self.dealer = Some(dealer);
        
        // Create ROUTER socket
        let router = self.context.socket(SocketType::ROUTER)?;
        router.set_sndhwm(self.config.high_water_mark)?;
        router.set_rcvhwm(self.config.high_water_mark)?;
        router.set_linger(self.config.linger)?;
        router.bind(&self.config.router_endpoint)?;
        self.router = Some(router);
        
        // Create PUB socket
        let publisher = self.context.socket(SocketType::PUB)?;
        publisher.set_sndhwm(self.config.high_water_mark)?;
        publisher.set_linger(self.config.linger)?;
        publisher.bind(&self.config.pub_endpoint)?;
        self.publisher = Some(publisher);
        
        // Create SUB socket
        let subscriber = self.context.socket(SocketType::SUB)?;
        subscriber.set_rcvhwm(self.config.high_water_mark)?;
        subscriber.connect(&self.config.sub_endpoint)?;
        subscriber.set_subscribe(b"")?; // Subscribe to all messages
        self.subscriber = Some(subscriber);
        
        Ok(())
    }
    
    pub async fn send_message(&mut self, message: &Message) -> Result<()> {
        let serialized = serde_json::to_string(message)?;
        
        match &message.target {
            MessageTarget::Component(target) => {
                if let Some(dealer) = &self.dealer {
                    dealer.send(target, zmq::SNDMORE)?;
                    dealer.send(serialized, 0)?;
                }
            }
            MessageTarget::Group(group) => {
                if let Some(publisher) = &self.publisher {
                    publisher.send(group, zmq::SNDMORE)?;
                    publisher.send(serialized, 0)?;
                }
            }
            MessageTarget::Broadcast => {
                if let Some(publisher) = &self.publisher {
                    publisher.send("broadcast", zmq::SNDMORE)?;
                    publisher.send(serialized, 0)?;
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn receive_message(&mut self, component_id: &str) -> Result<Option<Message>> {
        if let Some(router) = &self.router {
            // Non-blocking receive
            let mut msg = zmq::Message::new();
            match router.recv(&mut msg, zmq::DONTWAIT) {
                Ok(_) => {
                    let identity = msg.as_str().unwrap_or("");
                    if identity == component_id {
                        // Get the actual message
                        let mut payload = zmq::Message::new();
                        router.recv(&mut payload, 0)?;
                        let serialized = payload.as_str().unwrap_or("");
                        let message: Message = serde_json::from_str(serialized)?;
                        return Ok(Some(message));
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // No message available
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }
    
    pub async fn health_check(&mut self) -> Result<bool> {
        // Simple health check - try to create a test socket
        let test_socket = self.context.socket(SocketType::REQ)?;
        test_socket.set_linger(0)?;
        Ok(true)
    }
}

/// Message router for adaptive transport selection
#[derive(Debug)]
pub struct MessageRouter {
    config: RoutingConfig,
    transport_metrics: HashMap<TransportType, TransportMetrics>,
    current_transport: TransportType,
    failover_state: FailoverState,
}

/// Transport metrics
#[derive(Debug, Clone)]
pub struct TransportMetrics {
    pub success_rate: f64,
    pub avg_latency: Duration,
    pub throughput: f64,
    pub error_count: u64,
    pub last_success: Instant,
}

/// Failover state
#[derive(Debug, Clone)]
pub struct FailoverState {
    pub is_active: bool,
    pub failed_transport: Option<TransportType>,
    pub failover_started: Option<Instant>,
    pub recovery_attempts: u32,
}

impl MessageRouter {
    pub fn new(config: RoutingConfig) -> Self {
        let mut transport_metrics = HashMap::new();
        transport_metrics.insert(TransportType::Redis, TransportMetrics::default());
        transport_metrics.insert(TransportType::ZeroMQ, TransportMetrics::default());
        
        Self {
            current_transport: config.default_transport,
            config,
            transport_metrics,
            failover_state: FailoverState {
                is_active: false,
                failed_transport: None,
                failover_started: None,
                recovery_attempts: 0,
            },
        }
    }
    
    pub fn select_transport(&mut self, message: &Message) -> TransportType {
        // If in failover state, use fallback transport
        if self.failover_state.is_active {
            return self.config.fallback_transport;
        }
        
        // If adaptive routing is disabled, use default transport
        if !self.config.adaptive_routing {
            return self.config.default_transport;
        }
        
        // Select transport based on metrics and message characteristics
        let redis_metrics = self.transport_metrics.get(&TransportType::Redis).unwrap();
        let zmq_metrics = self.transport_metrics.get(&TransportType::ZeroMQ).unwrap();
        
        // High priority messages prefer the transport with lower latency
        if message.priority > 0.8 {
            if redis_metrics.avg_latency < zmq_metrics.avg_latency {
                TransportType::Redis
            } else {
                TransportType::ZeroMQ
            }
        } else {
            // Regular messages use the transport with higher success rate
            if redis_metrics.success_rate > zmq_metrics.success_rate {
                TransportType::Redis
            } else {
                TransportType::ZeroMQ
            }
        }
    }
    
    pub fn record_success(&mut self, transport: TransportType, latency: Duration) {
        if let Some(metrics) = self.transport_metrics.get_mut(&transport) {
            metrics.success_rate = (metrics.success_rate * 0.95) + (1.0 * 0.05);
            metrics.avg_latency = Duration::from_millis(
                (metrics.avg_latency.as_millis() as f64 * 0.95 + latency.as_millis() as f64 * 0.05) as u64
            );
            metrics.last_success = Instant::now();
        }
    }
    
    pub fn record_failure(&mut self, transport: TransportType) {
        if let Some(metrics) = self.transport_metrics.get_mut(&transport) {
            metrics.success_rate = metrics.success_rate * 0.95;
            metrics.error_count += 1;
            
            // Check if we need to failover
            if metrics.success_rate < self.config.failover_threshold && !self.failover_state.is_active {
                self.initiate_failover(transport);
            }
        }
    }
    
    fn initiate_failover(&mut self, failed_transport: TransportType) {
        info!("Initiating failover from {:?} to {:?}", failed_transport, self.config.fallback_transport);
        
        self.failover_state = FailoverState {
            is_active: true,
            failed_transport: Some(failed_transport),
            failover_started: Some(Instant::now()),
            recovery_attempts: 0,
        };
    }
    
    pub fn check_recovery(&mut self) -> bool {
        if !self.failover_state.is_active {
            return false;
        }
        
        if let Some(failed_transport) = self.failover_state.failed_transport {
            if let Some(metrics) = self.transport_metrics.get(&failed_transport) {
                if metrics.success_rate > self.config.failover_threshold * 2.0 {
                    info!("Recovery detected for transport {:?}", failed_transport);
                    self.failover_state.is_active = false;
                    self.current_transport = failed_transport;
                    return true;
                }
            }
        }
        
        false
    }
}

impl Default for TransportMetrics {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            avg_latency: Duration::from_millis(10),
            throughput: 100.0,
            error_count: 0,
            last_success: Instant::now(),
        }
    }
}

/// Transport health monitor
#[derive(Debug)]
pub struct TransportHealthMonitor {
    redis_health: TransportHealth,
    zmq_health: TransportHealth,
    config: HealthConfig,
}

/// Transport health status
#[derive(Debug, Clone)]
pub struct TransportHealth {
    pub is_healthy: bool,
    pub last_check: Instant,
    pub failure_count: u32,
    pub recovery_time: Option<Instant>,
}

impl TransportHealthMonitor {
    pub fn new(config: HealthConfig) -> Self {
        Self {
            redis_health: TransportHealth::default(),
            zmq_health: TransportHealth::default(),
            config,
        }
    }
    
    pub async fn check_health(&mut self, redis: &mut Option<RedisTransport>, zmq: &mut Option<ZmqTransport>) {
        // Check Redis health
        if let Some(redis_transport) = redis {
            match redis_transport.health_check().await {
                Ok(healthy) => {
                    self.redis_health.is_healthy = healthy;
                    self.redis_health.last_check = Instant::now();
                    if healthy {
                        self.redis_health.failure_count = 0;
                    }
                }
                Err(_) => {
                    self.redis_health.is_healthy = false;
                    self.redis_health.failure_count += 1;
                    self.redis_health.last_check = Instant::now();
                }
            }
        }
        
        // Check ZMQ health
        if let Some(zmq_transport) = zmq {
            match zmq_transport.health_check().await {
                Ok(healthy) => {
                    self.zmq_health.is_healthy = healthy;
                    self.zmq_health.last_check = Instant::now();
                    if healthy {
                        self.zmq_health.failure_count = 0;
                    }
                }
                Err(_) => {
                    self.zmq_health.is_healthy = false;
                    self.zmq_health.failure_count += 1;
                    self.zmq_health.last_check = Instant::now();
                }
            }
        }
    }
    
    pub fn get_health_status(&self, transport: TransportType) -> &TransportHealth {
        match transport {
            TransportType::Redis => &self.redis_health,
            TransportType::ZeroMQ => &self.zmq_health,
        }
    }
}

impl Default for TransportHealth {
    fn default() -> Self {
        Self {
            is_healthy: true,
            last_check: Instant::now(),
            failure_count: 0,
            recovery_time: None,
        }
    }
}

/// Message queue for buffering and batching
#[derive(Debug)]
pub struct MessageQueue {
    queue: Vec<Message>,
    config: QueueConfig,
}

impl MessageQueue {
    pub fn new(config: QueueConfig) -> Self {
        Self {
            queue: Vec::new(),
            config,
        }
    }
    
    pub fn enqueue(&mut self, message: Message) -> Result<()> {
        if self.queue.len() >= self.config.max_queue_size {
            return Err(anyhow::anyhow!("Queue is full"));
        }
        
        self.queue.push(message);
        Ok(())
    }
    
    pub fn dequeue(&mut self) -> Option<Message> {
        if self.queue.is_empty() {
            return None;
        }
        
        // Remove expired messages
        let now = Instant::now();
        self.queue.retain(|msg| {
            if let Some(expires_at) = msg.expires_at {
                expires_at > now
            } else {
                true
            }
        });
        
        // Return highest priority message
        if let Some(pos) = self.queue.iter().position(|msg| {
            msg.priority == self.queue.iter().map(|m| m.priority).fold(0.0, f64::max)
        }) {
            Some(self.queue.remove(pos))
        } else {
            None
        }
    }
    
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Component information
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub id: String,
    pub transport_preference: Option<TransportType>,
    pub message_types: Vec<String>,
    pub last_activity: Instant,
    pub is_active: bool,
}

impl MultiTransportMessaging {
    pub async fn new(config: MessagingConfig) -> Result<Self> {
        let redis_transport = if config.redis.enabled {
            Some(RedisTransport::new(config.redis.clone()).await?)
        } else {
            None
        };
        
        let zmq_transport = if config.zmq.enabled {
            Some(ZmqTransport::new(config.zmq.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            router: Arc::new(RwLock::new(MessageRouter::new(config.routing.clone()))),
            health_monitor: Arc::new(RwLock::new(TransportHealthMonitor::new(config.health.clone()))),
            redis_transport: Arc::new(RwLock::new(redis_transport)),
            zmq_transport: Arc::new(RwLock::new(zmq_transport)),
            components: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(RwLock::new(MessageQueue::new(config.queue.clone()))),
            config,
        })
    }
    
    pub async fn start(&mut self) -> Result<()> {
        // Connect transports
        if let Some(redis) = &mut *self.redis_transport.write().await {
            redis.connect().await?;
        }
        
        if let Some(zmq) = &mut *self.zmq_transport.write().await {
            zmq.connect().await?;
        }
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        // Stop health monitoring and cleanup
        Ok(())
    }
    
    pub async fn register_component(&mut self, component_id: &str) -> Result<()> {
        let component_info = ComponentInfo {
            id: component_id.to_string(),
            transport_preference: None,
            message_types: Vec::new(),
            last_activity: Instant::now(),
            is_active: true,
        };
        
        self.components.write().await.insert(component_id.to_string(), component_info);
        Ok(())
    }
    
    pub async fn send_message(&mut self, message: Message) -> Result<()> {
        let transport = self.router.write().await.select_transport(&message);
        let start_time = Instant::now();
        
        let result = match transport {
            TransportType::Redis => {
                if let Some(redis) = &mut *self.redis_transport.write().await {
                    redis.send_message(&message).await
                } else {
                    Err(anyhow::anyhow!("Redis transport not available"))
                }
            }
            TransportType::ZeroMQ => {
                if let Some(zmq) = &mut *self.zmq_transport.write().await {
                    zmq.send_message(&message).await
                } else {
                    Err(anyhow::anyhow!("ZeroMQ transport not available"))
                }
            }
        };
        
        let latency = start_time.elapsed();
        
        // Record metrics
        match result {
            Ok(_) => {
                self.router.write().await.record_success(transport, latency);
            }
            Err(e) => {
                self.router.write().await.record_failure(transport);
                return Err(e);
            }
        }
        
        Ok(())
    }
    
    async fn start_health_monitoring(&self) -> Result<()> {
        let health_monitor = Arc::clone(&self.health_monitor);
        let redis_transport = Arc::clone(&self.redis_transport);
        let zmq_transport = Arc::clone(&self.zmq_transport);
        let router = Arc::clone(&self.router);
        let interval = self.config.health.check_interval;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                // Check transport health
                {
                    let mut monitor = health_monitor.write().await;
                    let mut redis = redis_transport.write().await;
                    let mut zmq = zmq_transport.write().await;
                    
                    monitor.check_health(&mut *redis, &mut *zmq).await;
                }
                
                // Check for recovery
                router.write().await.check_recovery();
            }
        });
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_message_router() {
        let config = RoutingConfig::default();
        let mut router = MessageRouter::new(config);
        
        let message = Message {
            id: "test".to_string(),
            message_type: "test".to_string(),
            source: "test".to_string(),
            target: MessageTarget::Component("test".to_string()),
            payload: serde_json::Value::Null,
            priority: 0.5,
            timestamp: Instant::now(),
            expires_at: None,
            retry_count: 0,
            metadata: HashMap::new(),
        };
        
        let transport = router.select_transport(&message);
        assert_eq!(transport, TransportType::Redis);
    }
    
    #[tokio::test]
    async fn test_message_queue() {
        let config = QueueConfig::default();
        let mut queue = MessageQueue::new(config);
        
        let message = Message {
            id: "test".to_string(),
            message_type: "test".to_string(),
            source: "test".to_string(),
            target: MessageTarget::Component("test".to_string()),
            payload: serde_json::Value::Null,
            priority: 0.5,
            timestamp: Instant::now(),
            expires_at: None,
            retry_count: 0,
            metadata: HashMap::new(),
        };
        
        assert!(queue.enqueue(message).is_ok());
        assert_eq!(queue.len(), 1);
        
        let dequeued = queue.dequeue();
        assert!(dequeued.is_some());
        assert_eq!(queue.len(), 0);
    }
}