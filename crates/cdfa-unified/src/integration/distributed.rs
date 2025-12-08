//! Distributed coordination system for multi-node CDFA operations
//!
//! This module provides comprehensive distributed computing capabilities:
//! - Multi-node coordination and synchronization
//! - Distributed consensus algorithms
//! - Load balancing and work distribution
//! - Fault tolerance and failure recovery
//! - Distributed state management
//! - Leader election and cluster management

use std::{
    collections::{HashMap, HashSet, BTreeMap},
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{RwLock, Mutex, Notify, mpsc, oneshot},
    time::{interval, sleep},
};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use sha2::{Sha256, Digest};
use crate::{
    error::{CdfaError, Result},
    integration::{
        redis_connector::{RedisPool, RedisClusterCoordinator},
        messaging::{RedisMessageBroker, Message, MessageType, MessagePriority, DeliveryMode},
    },
};

/// Node status in the distributed cluster
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is active and healthy
    Active,
    /// Node is degraded but still functional
    Degraded,
    /// Node is temporarily unavailable
    Unavailable,
    /// Node has been marked as failed
    Failed,
    /// Node is leaving the cluster gracefully
    Leaving,
}

/// Node role in the distributed system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    /// Leader node coordinating the cluster
    Leader,
    /// Follower node executing tasks
    Follower,
    /// Observer node monitoring the cluster
    Observer,
    /// Worker node specialized for computation
    Worker,
    /// Gateway node handling external connections
    Gateway,
}

/// Node capabilities and resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Available CPU cores
    pub cpu_cores: u32,
    /// Available memory in MB
    pub memory_mb: u64,
    /// GPU availability
    pub has_gpu: bool,
    /// Supported SIMD instruction sets
    pub simd_features: Vec<String>,
    /// Network bandwidth capacity
    pub network_bandwidth_mbps: u32,
    /// Specialized computing capabilities
    pub specializations: Vec<String>,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: u32,
}

/// Node information in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub id: String,
    /// Node role in the cluster
    pub role: NodeRole,
    /// Current node status
    pub status: NodeStatus,
    /// Node capabilities and resources
    pub capabilities: NodeCapabilities,
    /// Node network address
    pub address: String,
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
    /// Node uptime in seconds
    pub uptime: u64,
    /// Current task count
    pub current_tasks: u32,
    /// Total completed tasks
    pub completed_tasks: u64,
    /// Node version/build information
    pub version: String,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl NodeInfo {
    /// Check if node is considered healthy
    pub fn is_healthy(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let heartbeat_age = now.saturating_sub(self.last_heartbeat);
        
        matches!(self.status, NodeStatus::Active | NodeStatus::Degraded) 
            && heartbeat_age < 30 // 30 second heartbeat timeout
    }

    /// Calculate load factor (0.0 to 1.0)
    pub fn load_factor(&self) -> f64 {
        if self.capabilities.max_concurrent_tasks == 0 {
            return 1.0;
        }
        self.current_tasks as f64 / self.capabilities.max_concurrent_tasks as f64
    }
}

/// Consensus proposal for cluster coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    /// Unique proposal identifier
    pub id: Uuid,
    /// Proposal type
    pub proposal_type: ProposalType,
    /// Proposer node ID
    pub proposer: String,
    /// Proposal data
    pub data: serde_json::Value,
    /// Proposal timestamp
    pub timestamp: u64,
    /// Required votes for acceptance
    pub required_votes: u32,
    /// Current vote count
    pub votes: HashMap<String, bool>, // node_id -> vote
    /// Proposal status
    pub status: ProposalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalType {
    /// Leader election proposal
    LeaderElection,
    /// Configuration change proposal
    ConfigurationChange,
    /// Task distribution proposal
    TaskDistribution,
    /// Node addition/removal proposal
    MembershipChange,
    /// Emergency cluster operation
    EmergencyAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Proposal is being voted on
    Pending,
    /// Proposal has been accepted
    Accepted,
    /// Proposal has been rejected
    Rejected,
    /// Proposal has expired
    Expired,
}

/// Distributed task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTask {
    /// Unique task identifier
    pub id: Uuid,
    /// Task type identifier
    pub task_type: String,
    /// Task priority level
    pub priority: u32,
    /// Task payload data
    pub payload: serde_json::Value,
    /// Required node capabilities
    pub requirements: NodeCapabilities,
    /// Maximum execution time in seconds
    pub timeout: u64,
    /// Task dependencies
    pub dependencies: Vec<Uuid>,
    /// Assigned node ID
    pub assigned_node: Option<String>,
    /// Task status
    pub status: TaskStatus,
    /// Creation timestamp
    pub created_at: u64,
    /// Started timestamp
    pub started_at: Option<u64>,
    /// Completed timestamp
    pub completed_at: Option<u64>,
    /// Task result
    pub result: Option<serde_json::Value>,
    /// Error information
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is waiting for assignment
    Pending,
    /// Task has been assigned to a node
    Assigned,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

/// Cluster configuration and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster name/identifier
    pub cluster_id: String,
    /// Minimum number of nodes for quorum
    pub min_nodes: u32,
    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,
    /// Node timeout before marking as failed
    pub node_timeout: u64,
    /// Leader election timeout
    pub election_timeout: u64,
    /// Maximum task execution time
    pub max_task_timeout: u64,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Consensus requirements
    pub consensus_threshold: f64, // 0.5 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded node assignment
    LeastLoaded,
    /// Capability-based assignment
    CapabilityBased,
    /// Random assignment
    Random,
    /// Custom strategy
    Custom(String),
}

/// Distributed coordination manager
pub struct DistributedCoordinator {
    /// Local node information
    node_info: Arc<RwLock<NodeInfo>>,
    /// Cluster configuration
    config: ClusterConfig,
    /// Redis connection pool
    redis_pool: Arc<RedisPool>,
    /// Message broker for communication
    message_broker: Arc<RwLock<RedisMessageBroker>>,
    /// Known cluster nodes
    cluster_nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    /// Current leader node
    current_leader: Arc<RwLock<Option<String>>>,
    /// Pending tasks queue
    pending_tasks: Arc<RwLock<BTreeMap<u32, Vec<DistributedTask>>>>, // priority -> tasks
    /// Running tasks
    running_tasks: Arc<RwLock<HashMap<Uuid, DistributedTask>>>,
    /// Consensus proposals
    active_proposals: Arc<RwLock<HashMap<Uuid, ConsensusProposal>>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Task assignment notifier
    task_notify: Arc<Notify>,
    /// Statistics
    stats: Arc<RwLock<CoordinatorStats>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinatorStats {
    pub total_tasks_processed: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub active_nodes: u32,
    pub total_proposals: u64,
    pub accepted_proposals: u64,
    pub rejected_proposals: u64,
    pub leadership_changes: u32,
    pub cluster_uptime: u64,
    pub average_task_duration: f64,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub async fn new(
        node_info: NodeInfo,
        config: ClusterConfig,
        redis_pool: Arc<RedisPool>,
    ) -> Result<Self> {
        let node_id = node_info.id.clone();
        let message_broker = Arc::new(RwLock::new(
            RedisMessageBroker::new(redis_pool.clone(), node_id.clone())
        ));

        let coordinator = Self {
            node_info: Arc::new(RwLock::new(node_info)),
            config,
            redis_pool,
            message_broker,
            cluster_nodes: Arc::new(RwLock::new(HashMap::new())),
            current_leader: Arc::new(RwLock::new(None)),
            pending_tasks: Arc::new(RwLock::new(BTreeMap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            active_proposals: Arc::new(RwLock::new(HashMap::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            task_notify: Arc::new(Notify::new()),
            stats: Arc::new(RwLock::new(CoordinatorStats::default())),
        };

        // Initialize coordinator
        coordinator.initialize().await?;

        Ok(coordinator)
    }

    /// Initialize the coordinator
    async fn initialize(&self) -> Result<()> {
        // Register node in cluster
        self.register_node().await?;

        // Start background tasks
        self.start_heartbeat_task().await;
        self.start_leader_election_task().await;
        self.start_task_processor().await;
        self.start_consensus_processor().await;
        self.start_cluster_monitor().await;

        log::info!("Distributed coordinator initialized for node: {}", 
                  self.node_info.read().await.id);

        Ok(())
    }

    /// Register this node in the cluster
    async fn register_node(&self) -> Result<()> {
        let node_info = self.node_info.read().await;
        let key = format!("cdfa:cluster:{}:nodes:{}", self.config.cluster_id, node_info.id);
        
        let node_data = serde_json::to_string(&*node_info)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize node info: {}", e)))?;

        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let _: () = locked_conn.connection
            .set_ex(&key, node_data, self.config.heartbeat_interval * 3)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to register node: {}", e)))?;

        log::info!("Node registered: {}", node_info.id);
        Ok(())
    }

    /// Start heartbeat background task
    async fn start_heartbeat_task(&self) {
        let node_info = self.node_info.clone();
        let config = self.config.clone();
        let redis_pool = self.redis_pool.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.heartbeat_interval));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                if let Err(e) = Self::send_heartbeat(&node_info, &config, &redis_pool).await {
                    log::error!("Failed to send heartbeat: {}", e);
                }
            }
        });
    }

    /// Send heartbeat to maintain node presence
    async fn send_heartbeat(
        node_info: &Arc<RwLock<NodeInfo>>,
        config: &ClusterConfig,
        redis_pool: &Arc<RedisPool>,
    ) -> Result<()> {
        let mut node = node_info.write().await;
        node.last_heartbeat = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        let key = format!("cdfa:cluster:{}:nodes:{}", config.cluster_id, node.id);
        let node_data = serde_json::to_string(&*node)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize node info: {}", e)))?;

        let conn = redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let _: () = locked_conn.connection
            .set_ex(&key, node_data, config.heartbeat_interval * 3)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to send heartbeat: {}", e)))?;

        Ok(())
    }

    /// Start leader election background task
    async fn start_leader_election_task(&self) {
        let node_info = self.node_info.clone();
        let config = self.config.clone();
        let redis_pool = self.redis_pool.clone();
        let current_leader = self.current_leader.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.election_timeout / 2));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                if let Err(e) = Self::check_leader_election(
                    &node_info,
                    &config,
                    &redis_pool,
                    &current_leader,
                    &cluster_nodes,
                ).await {
                    log::error!("Leader election check failed: {}", e);
                }
            }
        });
    }

    /// Check and potentially trigger leader election
    async fn check_leader_election(
        node_info: &Arc<RwLock<NodeInfo>>,
        config: &ClusterConfig,
        redis_pool: &Arc<RedisPool>,
        current_leader: &Arc<RwLock<Option<String>>>,
        cluster_nodes: &Arc<RwLock<HashMap<String, NodeInfo>>>,
    ) -> Result<()> {
        // Update cluster nodes list
        Self::update_cluster_nodes(config, redis_pool, cluster_nodes).await?;

        let nodes = cluster_nodes.read().await;
        let healthy_nodes: Vec<_> = nodes.values()
            .filter(|node| node.is_healthy())
            .collect();

        // Check if we have enough nodes for quorum
        if healthy_nodes.len() < config.min_nodes as usize {
            log::warn!("Insufficient healthy nodes for quorum: {} < {}", 
                      healthy_nodes.len(), config.min_nodes);
            return Ok(());
        }

        // Check current leader health
        let leader_healthy = if let Some(leader_id) = current_leader.read().await.as_ref() {
            healthy_nodes.iter().any(|node| &node.id == leader_id)
        } else {
            false
        };

        // Start election if no healthy leader
        if !leader_healthy {
            Self::start_leader_election(node_info, config, redis_pool, current_leader).await?;
        }

        Ok(())
    }

    /// Start a new leader election
    async fn start_leader_election(
        node_info: &Arc<RwLock<NodeInfo>>,
        config: &ClusterConfig,
        redis_pool: &Arc<RedisPool>,
        current_leader: &Arc<RwLock<Option<String>>>,
    ) -> Result<()> {
        let node = node_info.read().await;
        
        // Only eligible nodes can become leader
        if !matches!(node.role, NodeRole::Leader | NodeRole::Follower) {
            return Ok(());
        }

        let election_key = format!("cdfa:cluster:{}:election", config.cluster_id);
        let candidate_data = serde_json::json!({
            "candidate": node.id,
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "capabilities": node.capabilities
        });

        let conn = redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        // Try to set election key with NX (only if not exists)
        let result: Option<String> = locked_conn.connection
            .set_nx(&election_key, candidate_data.to_string())
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to start election: {}", e)))?;

        if result.is_some() {
            log::info!("Started leader election for node: {}", node.id);
            
            // Set election timeout
            let _: () = locked_conn.connection
                .expire(&election_key, config.election_timeout as i64)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to set election timeout: {}", e)))?;

            // Wait for election result
            sleep(Duration::from_secs(config.election_timeout)).await;
            
            // Check if we won the election
            let winner: Option<String> = locked_conn.connection
                .get(&election_key)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to get election result: {}", e)))?;

            if let Some(winner_data) = winner {
                if let Ok(election_result) = serde_json::from_str::<serde_json::Value>(&winner_data) {
                    if let Some(winner_id) = election_result.get("candidate").and_then(|v| v.as_str()) {
                        *current_leader.write().await = Some(winner_id.to_string());
                        
                        if winner_id == node.id {
                            log::info!("Won leadership election: {}", node.id);
                        } else {
                            log::info!("Leadership election won by: {}", winner_id);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Update cluster nodes from Redis
    async fn update_cluster_nodes(
        config: &ClusterConfig,
        redis_pool: &Arc<RedisPool>,
        cluster_nodes: &Arc<RwLock<HashMap<String, NodeInfo>>>,
    ) -> Result<()> {
        let pattern = format!("cdfa:cluster:{}:nodes:*", config.cluster_id);
        
        let conn = redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let keys: Vec<String> = locked_conn.connection
            .keys(&pattern)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to get node keys: {}", e)))?;

        let mut updated_nodes = HashMap::new();

        for key in keys {
            let node_data: Option<String> = locked_conn.connection
                .get(&key)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to get node data: {}", e)))?;

            if let Some(data) = node_data {
                match serde_json::from_str::<NodeInfo>(&data) {
                    Ok(node_info) => {
                        updated_nodes.insert(node_info.id.clone(), node_info);
                    }
                    Err(e) => {
                        log::warn!("Failed to deserialize node info: {}", e);
                    }
                }
            }
        }

        *cluster_nodes.write().await = updated_nodes;
        Ok(())
    }

    /// Start task processor background task
    async fn start_task_processor(&self) {
        let node_info = self.node_info.clone();
        let current_leader = self.current_leader.clone();
        let pending_tasks = self.pending_tasks.clone();
        let running_tasks = self.running_tasks.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        let task_notify = self.task_notify.clone();
        let shutdown = self.shutdown.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                // Wait for task notification or timeout
                tokio::select! {
                    _ = task_notify.notified() => {},
                    _ = sleep(Duration::from_secs(5)) => {},
                }

                let node = node_info.read().await;
                let is_leader = if let Some(leader_id) = current_leader.read().await.as_ref() {
                    leader_id == &node.id
                } else {
                    false
                };

                if is_leader {
                    // Leader assigns tasks to nodes
                    if let Err(e) = Self::assign_tasks(
                        &pending_tasks,
                        &cluster_nodes,
                        &config,
                    ).await {
                        log::error!("Failed to assign tasks: {}", e);
                    }
                } else {
                    // Worker processes assigned tasks
                    if let Err(e) = Self::process_tasks(
                        &node,
                        &running_tasks,
                    ).await {
                        log::error!("Failed to process tasks: {}", e);
                    }
                }
            }
        });
    }

    /// Assign tasks to available nodes (leader function)
    async fn assign_tasks(
        pending_tasks: &Arc<RwLock<BTreeMap<u32, Vec<DistributedTask>>>>,
        cluster_nodes: &Arc<RwLock<HashMap<String, NodeInfo>>>,
        config: &ClusterConfig,
    ) -> Result<()> {
        let mut tasks = pending_tasks.write().await;
        let nodes = cluster_nodes.read().await;

        // Get available nodes
        let available_nodes: Vec<_> = nodes.values()
            .filter(|node| node.is_healthy() && node.load_factor() < 0.8)
            .collect();

        if available_nodes.is_empty() {
            return Ok(());
        }

        // Process tasks by priority (highest first)
        for (priority, task_list) in tasks.iter_mut().rev() {
            let mut i = 0;
            while i < task_list.len() && !available_nodes.is_empty() {
                let task = &mut task_list[i];
                
                // Find suitable node for task
                if let Some(assigned_node) = Self::find_suitable_node(&available_nodes, task, config) {
                    task.assigned_node = Some(assigned_node.id.clone());
                    task.status = TaskStatus::Assigned;
                    
                    log::debug!("Assigned task {} to node {}", task.id, assigned_node.id);
                    
                    // Remove from pending and move to assigned
                    let assigned_task = task_list.remove(i);
                    // TODO: Add to assigned tasks tracking
                } else {
                    i += 1;
                }
            }
        }

        // Remove empty priority queues
        tasks.retain(|_, task_list| !task_list.is_empty());

        Ok(())
    }

    /// Find suitable node for a task
    fn find_suitable_node(
        available_nodes: &[&NodeInfo],
        task: &DistributedTask,
        config: &ClusterConfig,
    ) -> Option<&NodeInfo> {
        match config.load_balancing {
            LoadBalancingStrategy::LeastLoaded => {
                available_nodes.iter()
                    .min_by(|a, b| a.load_factor().partial_cmp(&b.load_factor()).unwrap())
                    .copied()
            }
            LoadBalancingStrategy::CapabilityBased => {
                // Find node with best capability match
                available_nodes.iter()
                    .filter(|node| Self::node_meets_requirements(node, &task.requirements))
                    .min_by(|a, b| a.load_factor().partial_cmp(&b.load_factor()).unwrap())
                    .copied()
            }
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (would need state tracking in real implementation)
                available_nodes.first().copied()
            }
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..available_nodes.len());
                Some(available_nodes[index])
            }
            LoadBalancingStrategy::Custom(_) => {
                // Custom strategy implementation would go here
                available_nodes.first().copied()
            }
        }
    }

    /// Check if node meets task requirements
    fn node_meets_requirements(node: &NodeInfo, requirements: &NodeCapabilities) -> bool {
        node.capabilities.cpu_cores >= requirements.cpu_cores &&
        node.capabilities.memory_mb >= requirements.memory_mb &&
        (!requirements.has_gpu || node.capabilities.has_gpu) &&
        node.capabilities.max_concurrent_tasks >= requirements.max_concurrent_tasks
    }

    /// Process assigned tasks (worker function)
    async fn process_tasks(
        node: &NodeInfo,
        running_tasks: &Arc<RwLock<HashMap<Uuid, DistributedTask>>>,
    ) -> Result<()> {
        // This would be implemented to actually execute tasks
        // For now, just log that we're processing
        let tasks = running_tasks.read().await;
        let node_tasks: Vec<_> = tasks.values()
            .filter(|task| task.assigned_node.as_ref() == Some(&node.id))
            .collect();

        if !node_tasks.is_empty() {
            log::debug!("Processing {} tasks on node {}", node_tasks.len(), node.id);
        }

        Ok(())
    }

    /// Start consensus processor background task
    async fn start_consensus_processor(&self) {
        let active_proposals = self.active_proposals.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                if let Err(e) = Self::process_consensus(
                    &active_proposals,
                    &cluster_nodes,
                    &config,
                ).await {
                    log::error!("Consensus processing failed: {}", e);
                }
            }
        });
    }

    /// Process active consensus proposals
    async fn process_consensus(
        active_proposals: &Arc<RwLock<HashMap<Uuid, ConsensusProposal>>>,
        cluster_nodes: &Arc<RwLock<HashMap<String, NodeInfo>>>,
        config: &ClusterConfig,
    ) -> Result<()> {
        let mut proposals = active_proposals.write().await;
        let nodes = cluster_nodes.read().await;
        
        let healthy_node_count = nodes.values().filter(|node| node.is_healthy()).count() as u32;
        let required_votes = ((healthy_node_count as f64) * config.consensus_threshold).ceil() as u32;

        let mut completed_proposals = Vec::new();

        for (proposal_id, proposal) in proposals.iter_mut() {
            if proposal.status != ProposalStatus::Pending {
                continue;
            }

            // Check if proposal has expired
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            if now > proposal.timestamp + 300 { // 5 minute timeout
                proposal.status = ProposalStatus::Expired;
                completed_proposals.push(*proposal_id);
                continue;
            }

            // Count votes
            let yes_votes = proposal.votes.values().filter(|&&vote| vote).count() as u32;
            let no_votes = proposal.votes.values().filter(|&&vote| !vote).count() as u32;

            if yes_votes >= required_votes {
                proposal.status = ProposalStatus::Accepted;
                completed_proposals.push(*proposal_id);
                log::info!("Proposal {} accepted with {} votes", proposal_id, yes_votes);
            } else if no_votes > healthy_node_count - required_votes {
                proposal.status = ProposalStatus::Rejected;
                completed_proposals.push(*proposal_id);
                log::info!("Proposal {} rejected with {} no votes", proposal_id, no_votes);
            }
        }

        // Remove completed proposals
        for proposal_id in completed_proposals {
            proposals.remove(&proposal_id);
        }

        Ok(())
    }

    /// Start cluster monitor background task
    async fn start_cluster_monitor(&self) {
        let cluster_nodes = self.cluster_nodes.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let nodes = cluster_nodes.read().await;
                let mut stats_guard = stats.write().await;
                
                stats_guard.active_nodes = nodes.values()
                    .filter(|node| node.is_healthy())
                    .count() as u32;
                
                stats_guard.cluster_uptime += 10; // 10 second interval
                
                log::debug!("Cluster status: {} active nodes", stats_guard.active_nodes);
            }
        });
    }

    /// Submit a new distributed task
    pub async fn submit_task(&self, mut task: DistributedTask) -> Result<Uuid> {
        task.id = Uuid::new_v4();
        task.status = TaskStatus::Pending;
        task.created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let priority = task.priority;
        {
            let mut pending = self.pending_tasks.write().await;
            pending.entry(priority).or_insert_with(Vec::new).push(task);
        }

        // Notify task processor
        self.task_notify.notify_one();

        let mut stats = self.stats.write().await;
        stats.total_tasks_processed += 1;

        Ok(task.id)
    }

    /// Submit a consensus proposal
    pub async fn submit_proposal(&self, proposal: ConsensusProposal) -> Result<Uuid> {
        let proposal_id = proposal.id;
        
        {
            let mut proposals = self.active_proposals.write().await;
            proposals.insert(proposal_id, proposal);
        }

        let mut stats = self.stats.write().await;
        stats.total_proposals += 1;

        log::info!("Submitted consensus proposal: {}", proposal_id);
        Ok(proposal_id)
    }

    /// Vote on a consensus proposal
    pub async fn vote_on_proposal(&self, proposal_id: Uuid, vote: bool) -> Result<()> {
        let node_id = self.node_info.read().await.id.clone();
        
        {
            let mut proposals = self.active_proposals.write().await;
            if let Some(proposal) = proposals.get_mut(&proposal_id) {
                proposal.votes.insert(node_id, vote);
                log::debug!("Voted {} on proposal {}", if vote { "yes" } else { "no" }, proposal_id);
            } else {
                return Err(CdfaError::invalid_input(format!("Proposal not found: {}", proposal_id)));
            }
        }

        Ok(())
    }

    /// Get current coordinator statistics
    pub async fn get_stats(&self) -> CoordinatorStats {
        self.stats.read().await.clone()
    }

    /// Get cluster status
    pub async fn get_cluster_status(&self) -> ClusterStatus {
        let nodes = self.cluster_nodes.read().await;
        let leader = self.current_leader.read().await.clone();
        let pending_tasks = self.pending_tasks.read().await;
        let running_tasks = self.running_tasks.read().await;

        let total_pending_tasks = pending_tasks.values().map(|tasks| tasks.len()).sum();

        ClusterStatus {
            cluster_id: self.config.cluster_id.clone(),
            leader_node: leader,
            total_nodes: nodes.len() as u32,
            healthy_nodes: nodes.values().filter(|node| node.is_healthy()).count() as u32,
            pending_tasks: total_pending_tasks,
            running_tasks: running_tasks.len(),
            uptime: self.stats.read().await.cluster_uptime,
        }
    }

    /// Gracefully shutdown the coordinator
    pub async fn shutdown(&self) -> Result<()> {
        log::info!("Shutting down distributed coordinator");
        
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Deregister node
        let node_info = self.node_info.read().await;
        let key = format!("cdfa:cluster:{}:nodes:{}", self.config.cluster_id, node_info.id);
        
        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;
        
        let _: () = locked_conn.connection
            .del(&key)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to deregister node: {}", e)))?;

        Ok(())
    }
}

/// Cluster status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub cluster_id: String,
    pub leader_node: Option<String>,
    pub total_nodes: u32,
    pub healthy_nodes: u32,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_health_check() {
        let mut node = NodeInfo {
            id: "test-node".to_string(),
            role: NodeRole::Worker,
            status: NodeStatus::Active,
            capabilities: NodeCapabilities {
                cpu_cores: 4,
                memory_mb: 8192,
                has_gpu: false,
                simd_features: vec!["avx2".to_string()],
                network_bandwidth_mbps: 1000,
                specializations: vec![],
                max_concurrent_tasks: 10,
            },
            address: "localhost:8080".to_string(),
            last_heartbeat: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            uptime: 3600,
            current_tasks: 5,
            completed_tasks: 100,
            version: "1.0.0".to_string(),
            metadata: HashMap::new(),
        };

        assert!(node.is_healthy());
        assert_eq!(node.load_factor(), 0.5);

        // Test unhealthy node (old heartbeat)
        node.last_heartbeat = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 60;
        assert!(!node.is_healthy());
    }

    #[test]
    fn test_consensus_proposal() {
        let proposal = ConsensusProposal {
            id: Uuid::new_v4(),
            proposal_type: ProposalType::LeaderElection,
            proposer: "node1".to_string(),
            data: serde_json::json!({"candidate": "node1"}),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            required_votes: 3,
            votes: HashMap::new(),
            status: ProposalStatus::Pending,
        };

        assert_eq!(proposal.status, ProposalStatus::Pending);
        assert!(proposal.votes.is_empty());
    }

    #[test]
    fn test_task_creation() {
        let task = DistributedTask {
            id: Uuid::new_v4(),
            task_type: "computation".to_string(),
            priority: 1,
            payload: serde_json::json!({"input": "test"}),
            requirements: NodeCapabilities {
                cpu_cores: 2,
                memory_mb: 1024,
                has_gpu: false,
                simd_features: vec![],
                network_bandwidth_mbps: 100,
                specializations: vec![],
                max_concurrent_tasks: 1,
            },
            timeout: 300,
            dependencies: vec![],
            assigned_node: None,
            status: TaskStatus::Pending,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
        };

        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.assigned_node.is_none());
    }
}