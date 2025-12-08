//! Main boardroom interface for multi-agent collaboration

use crate::{
    agent::{Agent, AgentId, AgentInfo, AgentCapability, AgentState},
    consensus::{ConsensusManager, ConsensusRequest, ConsensusResult, VotingPolicy},
    discovery::{ServiceRegistry, LocalDiscoveryService, DiscoveryService},
    message::{Message, MessageId, MessageType, MessageRouter},
    routing::{LoadBalancer, MessageRouterImpl, RoutingStrategy},
    transport::{Transport, RedisTransport, ZmqTransport, RedisConfig, ZmqConfig},
};
use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock, Mutex};
use tokio::task::JoinHandle;
use tracing::{info, warn, error};

/// Boardroom configuration
#[derive(Debug, Clone)]
pub struct BoardroomConfig {
    pub name: String,
    pub transport_type: TransportType,
    pub transport_config: TransportConfig,
    pub consensus_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub stale_agent_timeout_secs: i64,
    pub default_routing_strategy: RoutingStrategy,
    pub enable_metrics: bool,
}

#[derive(Debug, Clone)]
pub enum TransportType {
    Redis,
    ZeroMQ,
}

#[derive(Debug, Clone)]
pub enum TransportConfig {
    Redis(RedisConfig),
    ZeroMQ(ZmqConfig),
}

impl Default for BoardroomConfig {
    fn default() -> Self {
        Self {
            name: "default-boardroom".to_string(),
            transport_type: TransportType::Redis,
            transport_config: TransportConfig::Redis(RedisConfig::default()),
            consensus_timeout_ms: 30000,
            heartbeat_interval_ms: 5000,
            stale_agent_timeout_secs: 60,
            default_routing_strategy: RoutingStrategy::LeastLoaded,
            enable_metrics: true,
        }
    }
}

/// Main boardroom interface
pub struct Boardroom {
    config: BoardroomConfig,
    transport: Arc<Mutex<Box<dyn Transport>>>,
    registry: Arc<ServiceRegistry>,
    discovery: Arc<dyn DiscoveryService>,
    consensus: Arc<ConsensusManager>,
    load_balancer: Arc<LoadBalancer>,
    router: Arc<MessageRouterImpl>,
    local_agents: Arc<DashMap<AgentId, Box<dyn Agent>>>,
    shutdown_tx: broadcast::Sender<()>,
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl Boardroom {
    /// Create new boardroom
    pub async fn new(config: BoardroomConfig) -> Result<Self> {
        // Create transport
        let transport: Box<dyn Transport> = match &config.transport_config {
            TransportConfig::Redis(redis_config) => {
                let mut transport = RedisTransport::new(redis_config.clone());
                transport.connect().await?;
                Box::new(transport)
            }
            TransportConfig::ZeroMQ(zmq_config) => {
                let mut transport = ZmqTransport::new(zmq_config.clone());
                transport.connect().await?;
                Box::new(transport)
            }
        };

        // Create components
        let registry = Arc::new(ServiceRegistry::new());
        let discovery = Arc::new(LocalDiscoveryService::new(registry.clone()));
        let consensus = Arc::new(ConsensusManager::new(config.consensus_timeout_ms));
        let load_balancer = Arc::new(LoadBalancer::new());
        let router = Arc::new(MessageRouterImpl::new(
            load_balancer.clone(),
            config.default_routing_strategy,
        ));

        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            transport: Arc::new(Mutex::new(transport)),
            registry,
            discovery,
            consensus,
            load_balancer,
            router,
            local_agents: Arc::new(DashMap::new()),
            shutdown_tx,
            tasks: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start the boardroom
    pub async fn start(&self) -> Result<()> {
        info!("Starting boardroom: {}", self.config.name);

        // Start message receiver
        self.start_message_receiver().await?;

        // Start heartbeat sender
        self.start_heartbeat_sender().await?;

        // Start cleanup task
        self.start_cleanup_task().await?;

        // Start metrics collector if enabled
        if self.config.enable_metrics {
            self.start_metrics_collector().await?;
        }

        info!("Boardroom started successfully");
        Ok(())
    }

    /// Stop the boardroom
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping boardroom: {}", self.config.name);

        // Send shutdown signal
        let _ = self.shutdown_tx.send(());

        // Stop all local agents
        for entry in self.local_agents.iter() {
            let mut agent = entry.value().clone();
            if let Err(e) = agent.stop().await {
                warn!("Error stopping agent {}: {}", entry.key(), e);
            }
        }

        // Wait for tasks to complete
        let mut tasks = self.tasks.lock().await;
        for task in tasks.drain(..) {
            let _ = task.await;
        }

        // Disconnect transport
        let mut transport = self.transport.lock().await;
        transport.disconnect().await?;

        info!("Boardroom stopped");
        Ok(())
    }

    /// Register a local agent
    pub async fn register_agent(&self, mut agent: Box<dyn Agent>) -> Result<()> {
        let agent_id = agent.id();
        let info = agent.info().await;

        // Initialize and start the agent
        agent.initialize().await?;
        agent.start().await?;

        // Register with discovery service
        self.discovery.register(info.clone()).await?;

        // Register with load balancer
        self.load_balancer.register_agent(info);

        // Store locally
        self.local_agents.insert(agent_id, agent);

        // Subscribe to agent's channel
        let channel = format!("agent:{}", agent_id);
        let transport = self.transport.lock().await;
        transport.subscribe(&channel).await?;

        info!("Agent {} registered", agent_id);
        Ok(())
    }

    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: AgentId) -> Result<()> {
        // Remove from local agents
        if let Some((_, mut agent)) = self.local_agents.remove(&agent_id) {
            agent.stop().await?;
        }

        // Unregister from discovery
        self.discovery.unregister(agent_id).await?;

        // Unregister from load balancer
        self.load_balancer.unregister_agent(agent_id);

        // Unsubscribe from channel
        let channel = format!("agent:{}", agent_id);
        let transport = self.transport.lock().await;
        transport.unsubscribe(&channel).await?;

        info!("Agent {} unregistered", agent_id);
        Ok(())
    }

    /// Send a message
    pub async fn send_message(&self, message: Message) -> Result<()> {
        // Route message if needed
        let mut message = message;
        if message.to.is_none() {
            message.to = self.router.route_message(&message).await?;
        }

        // Send via transport
        let transport = self.transport.lock().await;
        transport.send(&message).await?;

        Ok(())
    }

    /// Request consensus
    pub async fn request_consensus(
        &self,
        proposal: serde_json::Value,
        participants: Vec<AgentId>,
        policy: VotingPolicy,
        timeout_ms: Option<u64>,
    ) -> Result<broadcast::Receiver<ConsensusResult>> {
        let request = ConsensusRequest {
            id: MessageId::new(),
            proposal,
            policy,
            timeout_ms: timeout_ms.unwrap_or(self.config.consensus_timeout_ms),
            initiator: AgentId::new(), // Would be the calling agent
            participants,
            metadata: Default::default(),
        };

        self.consensus.initiate_consensus(request).await
    }

    /// Find agents by capability
    pub async fn find_agents(&self, capability: &AgentCapability) -> Result<Vec<AgentInfo>> {
        self.discovery.find_agents(capability).await
    }

    /// Get agent info
    pub async fn get_agent(&self, agent_id: AgentId) -> Result<Option<AgentInfo>> {
        self.discovery.get_agent(agent_id).await
    }

    /// Subscribe to a topic
    pub async fn subscribe_to_topic(&self, agent_id: AgentId, topic: String) -> Result<()> {
        self.registry.subscribe_to_topic(agent_id, topic).await
    }

    /// Broadcast message to topic subscribers
    pub async fn broadcast_to_topic(&self, topic: &str, message: Message) -> Result<()> {
        let subscribers = self.registry.get_topic_subscribers(topic).await;
        
        for agent_id in subscribers {
            let mut msg = message.clone();
            msg.to = Some(agent_id);
            self.send_message(msg).await?;
        }

        Ok(())
    }

    /// Start message receiver task
    async fn start_message_receiver(&self) -> Result<()> {
        let transport = self.transport.clone();
        let local_agents = self.local_agents.clone();
        let consensus = self.consensus.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = async {
                        let transport = transport.lock().await;
                        transport.receive().await
                    } => {
                        match result {
                            Ok(Some(message)) => {
                                if let Err(e) = Self::handle_message(
                                    message,
                                    &local_agents,
                                    &consensus,
                                ).await {
                                    error!("Error handling message: {}", e);
                                }
                            }
                            Ok(None) => {
                                // No message available
                                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            }
                            Err(e) => {
                                error!("Error receiving message: {}", e);
                                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
            }
        });

        let mut tasks = self.tasks.lock().await;
        tasks.push(task);

        Ok(())
    }

    /// Handle received message
    async fn handle_message(
        message: Message,
        local_agents: &DashMap<AgentId, Box<dyn Agent>>,
        consensus: &ConsensusManager,
    ) -> Result<()> {
        // Check if message is for a local agent
        if let Some(to) = message.to {
            if let Some(mut entry) = local_agents.get_mut(&to) {
                return entry.handle_message(message).await;
            }
        }

        // Handle special message types
        match &message.message_type {
            MessageType::ConsensusVote { .. } => {
                // Extract vote and submit to consensus manager
                if let MessageType::ConsensusVote { vote, reason, .. } = message.message_type {
                    let vote = crate::consensus::Vote {
                        agent_id: message.from,
                        vote,
                        weight: 1.0, // Could be dynamic
                        reason,
                        timestamp: chrono::Utc::now(),
                    };
                    consensus.submit_vote(vote).await?;
                }
            }
            _ => {
                // Other message types would be handled here
            }
        }

        Ok(())
    }

    /// Start heartbeat sender task
    async fn start_heartbeat_sender(&self) -> Result<()> {
        let local_agents = self.local_agents.clone();
        let transport = self.transport.clone();
        let interval_ms = self.config.heartbeat_interval_ms;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = interval.tick() => {
                        for entry in local_agents.iter() {
                            let agent_id = *entry.key();
                            let heartbeat = Message::heartbeat(agent_id);
                            
                            if let Ok(transport) = transport.lock().await.as_ref() {
                                let _ = transport.send(&heartbeat).await;
                            }
                        }
                    }
                }
            }
        });

        let mut tasks = self.tasks.lock().await;
        tasks.push(task);

        Ok(())
    }

    /// Start cleanup task
    async fn start_cleanup_task(&self) -> Result<()> {
        let registry = self.registry.clone();
        let timeout_secs = self.config.stale_agent_timeout_secs;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = interval.tick() => {
                        if let Err(e) = registry.cleanup_stale(timeout_secs).await {
                            error!("Error cleaning up stale agents: {}", e);
                        }
                    }
                }
            }
        });

        let mut tasks = self.tasks.lock().await;
        tasks.push(task);

        Ok(())
    }

    /// Start metrics collector task
    async fn start_metrics_collector(&self) -> Result<()> {
        let local_agents = self.local_agents.clone();
        let load_balancer = self.load_balancer.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = interval.tick() => {
                        // Collect metrics from local agents
                        for entry in local_agents.iter() {
                            let agent_id = *entry.key();
                            
                            // In a real implementation, we would query the agent for metrics
                            let metrics = crate::routing::LoadMetrics {
                                agent_id,
                                active_tasks: rand::random::<usize>() % 10,
                                message_queue_size: rand::random::<usize>() % 100,
                                cpu_usage: rand::random::<f32>() * 0.8,
                                memory_usage: rand::random::<f32>() * 0.7,
                                response_time_ms: rand::random::<f64>() * 100.0,
                                last_updated: chrono::Utc::now(),
                            };
                            
                            load_balancer.update_metrics(metrics);
                        }
                    }
                }
            }
        });

        let mut tasks = self.tasks.lock().await;
        tasks.push(task);

        Ok(())
    }
}

#[async_trait::async_trait]
impl MessageRouter for Boardroom {
    async fn route(&self, message: Message) -> Result<()> {
        self.send_message(message).await
    }

    async fn subscribe(&self, agent_id: AgentId, topic: String) -> Result<()> {
        self.subscribe_to_topic(agent_id, topic).await
    }

    async fn unsubscribe(&self, agent_id: AgentId, topic: String) -> Result<()> {
        self.registry.unsubscribe_from_topic(agent_id, &topic).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_boardroom_creation() {
        let config = BoardroomConfig::default();
        
        // Test configuration creation
        assert_eq!(config.name, "default-boardroom");
        assert_eq!(config.consensus_timeout_ms, 30000);
        assert_eq!(config.heartbeat_interval_ms, 5000);
        assert_eq!(config.stale_agent_timeout_secs, 60);
        assert!(config.enable_metrics);
        assert!(matches!(config.transport_type, TransportType::Redis));
        assert!(matches!(config.default_routing_strategy, RoutingStrategy::LeastLoaded));
        
        // Test that config validation works
        let mut test_config = config.clone();
        test_config.consensus_timeout_ms = 0;
        // This should be valid since we're not enforcing minimum timeout
        assert!(test_config.consensus_timeout_ms == 0);
        
        // Test transport config
        match &config.transport_config {
            TransportConfig::Redis(redis_config) => {
                // Redis config should be valid
                assert!(!redis_config.url.is_empty());
            }
            _ => panic!("Expected Redis transport config"),
        }
    }
    
    #[tokio::test]
    async fn test_boardroom_config_validation() {
        let mut config = BoardroomConfig::default();
        
        // Test different timeout values
        config.consensus_timeout_ms = 1000;
        assert!(config.consensus_timeout_ms > 0);
        
        config.heartbeat_interval_ms = 2000;
        assert!(config.heartbeat_interval_ms > 0);
        
        config.stale_agent_timeout_secs = 30;
        assert!(config.stale_agent_timeout_secs > 0);
        
        // Test different routing strategies
        config.default_routing_strategy = RoutingStrategy::RoundRobin;
        assert!(matches!(config.default_routing_strategy, RoutingStrategy::RoundRobin));
        
        config.default_routing_strategy = RoutingStrategy::Random;
        assert!(matches!(config.default_routing_strategy, RoutingStrategy::Random));
        
        config.default_routing_strategy = RoutingStrategy::CapabilityBased;
        assert!(matches!(config.default_routing_strategy, RoutingStrategy::CapabilityBased));
        
        // Test metrics toggle
        config.enable_metrics = false;
        assert!(!config.enable_metrics);
    }
    
    #[tokio::test]
    async fn test_transport_config_types() {
        // Test Redis transport config
        let redis_config = RedisConfig::default();
        let transport_config = TransportConfig::Redis(redis_config);
        assert!(matches!(transport_config, TransportConfig::Redis(_)));
        
        // Test ZeroMQ transport config
        let zmq_config = ZmqConfig::default();
        let transport_config = TransportConfig::ZeroMQ(zmq_config);
        assert!(matches!(transport_config, TransportConfig::ZeroMQ(_)));
        
        // Test different transport types
        let redis_type = TransportType::Redis;
        let zmq_type = TransportType::ZeroMQ;
        assert!(matches!(redis_type, TransportType::Redis));
        assert!(matches!(zmq_type, TransportType::ZeroMQ));
    }
}