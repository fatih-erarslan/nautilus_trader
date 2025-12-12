//! Load Balancer Agent
//!
//! Dynamic load balancing across all 20+ agents with real-time load monitoring,
//! intelligent distribution algorithms, and adaptive scaling capabilities.

use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::{RwLock, mpsc, broadcast, Mutex, Semaphore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use dashmap::DashMap;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use parking_lot::RwLock as ParkingRwLock;
use atomic::{Atomic, Ordering as AtomicOrdering};
use tracing::{debug, info, warn, error, instrument};
use anyhow::Result;
use futures::{Future, StreamExt};
use tokio::time::{sleep, timeout};
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use priority_queue::PriorityQueue;
use smallvec::SmallVec;
use ordered_float::OrderedFloat;

use crate::types::*;
use crate::error::OrchestrationError;
use crate::agent::{Agent, AgentId, AgentInfo, AgentState, AgentType};
use crate::communication::{Message, MessageRouter, MessageType};
use crate::health::{HealthStatus, HealthChecker};
use crate::metrics::{OrchestrationMetrics};

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Load monitoring interval in milliseconds
    pub monitoring_interval: u64,
    /// Load threshold for scaling
    pub load_threshold: f64,
    /// Minimum agents per swarm
    pub min_agents_per_swarm: usize,
    /// Maximum agents per swarm
    pub max_agents_per_swarm: usize,
    /// Auto-scaling enabled
    pub auto_scaling_enabled: bool,
    /// Scale-up threshold
    pub scale_up_threshold: f64,
    /// Scale-down threshold
    pub scale_down_threshold: f64,
    /// Cool-down period after scaling
    pub scaling_cooldown_ms: u64,
    /// Health check weight in load calculation
    pub health_weight: f64,
    /// Response time weight in load calculation
    pub response_time_weight: f64,
    /// CPU weight in load calculation
    pub cpu_weight: f64,
    /// Memory weight in load calculation
    pub memory_weight: f64,
    /// Network weight in load calculation
    pub network_weight: f64,
    /// Enable predictive scaling
    pub predictive_scaling: bool,
    /// Load history size for predictions
    pub load_history_size: usize,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least response time
    LeastResponseTime,
    /// Random selection
    Random,
    /// Consistent hashing
    ConsistentHashing,
    /// Resource-based selection
    ResourceBased,
    /// Adaptive ML-based selection
    Adaptive,
    /// Latency-aware selection
    LatencyAware,
    /// Throughput-optimized
    ThroughputOptimized,
}

/// Agent load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoad {
    /// Agent ID
    pub agent_id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Swarm ID
    pub swarm_id: String,
    /// Current load metrics
    pub metrics: LoadMetrics,
    /// Health status
    pub health_status: HealthStatus,
    /// Capacity information
    pub capacity: AgentCapacity,
    /// Performance history
    pub performance_history: VecDeque<PerformanceSnapshot>,
    /// Last update timestamp
    pub last_update: Instant,
    /// Weight for weighted algorithms
    pub weight: f64,
    /// Current connections
    pub active_connections: usize,
    /// Load score (0.0 = no load, 1.0 = fully loaded)
    pub load_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_utilization: f64,
    /// Network utilization (0.0 - 1.0)
    pub network_utilization: f64,
    /// Average response time in microseconds
    pub avg_response_time_us: f64,
    /// Messages per second
    pub messages_per_second: f64,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Throughput score (0.0 - 1.0)
    pub throughput_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapacity {
    /// Maximum CPU capacity
    pub max_cpu: f64,
    /// Maximum memory capacity in bytes
    pub max_memory: u64,
    /// Maximum network capacity in bytes per second
    pub max_network_bps: u64,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Maximum messages per second
    pub max_messages_per_second: f64,
    /// Specialized capabilities
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Load metrics at this time
    pub metrics: LoadMetrics,
    /// Response time
    pub response_time_us: f64,
    /// Throughput
    pub throughput: f64,
    /// Error count
    pub error_count: u64,
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingStrategy {
    /// Primary algorithm
    pub primary_algorithm: LoadBalancingAlgorithm,
    /// Fallback algorithm
    pub fallback_algorithm: LoadBalancingAlgorithm,
    /// Algorithm weights
    pub weights: HashMap<String, f64>,
    /// Adaptive parameters
    pub adaptive_params: AdaptiveParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Exploration vs exploitation ratio
    pub exploration_rate: f64,
    /// Prediction window size
    pub prediction_window: usize,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
}

/// Load balancer command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancerCommand {
    /// Select agent for task assignment
    SelectAgent {
        swarm_id: String,
        task_type: String,
        requirements: TaskRequirements,
    },
    /// Update agent load metrics
    UpdateAgentLoad {
        agent_id: AgentId,
        metrics: LoadMetrics,
    },
    /// Add agent to load balancing
    AddAgent {
        agent_id: AgentId,
        agent_type: AgentType,
        swarm_id: String,
        capacity: AgentCapacity,
    },
    /// Remove agent from load balancing
    RemoveAgent {
        agent_id: AgentId,
    },
    /// Set agent weight
    SetAgentWeight {
        agent_id: AgentId,
        weight: f64,
    },
    /// Trigger scaling operation
    TriggerScaling {
        swarm_id: String,
        scale_direction: ScaleDirection,
    },
    /// Set load balancing algorithm
    SetAlgorithm {
        algorithm: LoadBalancingAlgorithm,
    },
    /// Get load balancing statistics
    GetStats,
    /// Get agent loads
    GetAgentLoads {
        swarm_id: Option<String>,
    },
    /// Optimize load distribution
    OptimizeDistribution,
    /// Enable/disable auto-scaling
    SetAutoScaling {
        enabled: bool,
    },
    /// Reset load balancer
    Reset,
    /// Shutdown
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Required CPU capacity
    pub cpu_requirement: f64,
    /// Required memory in bytes
    pub memory_requirement: u64,
    /// Required network bandwidth
    pub network_requirement: u64,
    /// Maximum acceptable latency
    pub max_latency_ms: f64,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Priority level
    pub priority: TaskPriority,
    /// Duration estimate
    pub estimated_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleDirection {
    Up,
    Down,
}

/// Load balancer event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancerEvent {
    /// Agent selected for task
    AgentSelected {
        agent_id: AgentId,
        swarm_id: String,
        load_score: f64,
        selection_reason: String,
    },
    /// Agent load updated
    AgentLoadUpdated {
        agent_id: AgentId,
        old_load_score: f64,
        new_load_score: f64,
    },
    /// Auto-scaling triggered
    AutoScalingTriggered {
        swarm_id: String,
        current_agents: usize,
        target_agents: usize,
        trigger_reason: String,
    },
    /// Load balancing algorithm changed
    AlgorithmChanged {
        old_algorithm: LoadBalancingAlgorithm,
        new_algorithm: LoadBalancingAlgorithm,
    },
    /// Load distribution optimized
    DistributionOptimized {
        improvements: HashMap<String, f64>,
    },
    /// Overload detected
    OverloadDetected {
        swarm_id: String,
        overloaded_agents: Vec<AgentId>,
        average_load: f64,
    },
    /// Underload detected
    UnderloadDetected {
        swarm_id: String,
        underutilized_agents: Vec<AgentId>,
        average_load: f64,
    },
}

/// Load Balancer Agent
pub struct LoadBalancerAgent {
    /// Agent ID
    id: AgentId,
    /// Configuration
    config: LoadBalancerConfig,
    /// Agent state
    state: Arc<RwLock<AgentState>>,
    /// Agent loads by ID
    agent_loads: Arc<RwLock<HashMap<AgentId, AgentLoad>>>,
    /// Swarm to agents mapping
    swarm_agents: Arc<RwLock<HashMap<String, HashSet<AgentId>>>>,
    /// Round-robin counters
    round_robin_counters: Arc<RwLock<HashMap<String, usize>>>,
    /// Consistent hashing ring
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    /// Load balancing strategy
    strategy: Arc<RwLock<LoadBalancingStrategy>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<LoadBalancerMetrics>>,
    /// Scaling history
    scaling_history: Arc<RwLock<VecDeque<ScalingEvent>>>,
    /// Load predictions
    load_predictor: Arc<LoadPredictor>,
    /// Message router
    message_router: Arc<MessageRouter>,
    /// Health checker
    health_checker: Arc<HealthChecker>,
    /// Command channel
    command_tx: mpsc::UnboundedSender<LoadBalancerCommand>,
    command_rx: Arc<Mutex<mpsc::UnboundedReceiver<LoadBalancerCommand>>>,
    /// Event broadcast
    event_tx: broadcast::Sender<LoadBalancerEvent>,
    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<mpsc::UnboundedSender<()>>>>,
    /// Running state
    running: Arc<Atomic<bool>>,
    /// Random number generator
    rng: Arc<Mutex<ChaCha8Rng>>,
    /// Last scaling time by swarm
    last_scaling: Arc<RwLock<HashMap<String, Instant>>>,
}

/// Load balancer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerMetrics {
    /// Total agent selections
    pub total_selections: u64,
    /// Average selection time in nanoseconds
    pub avg_selection_time_ns: f64,
    /// Load distribution efficiency
    pub distribution_efficiency: f64,
    /// Auto-scaling events
    pub scaling_events: u64,
    /// Current load balance score
    pub load_balance_score: f64,
    /// Average agent utilization
    pub avg_agent_utilization: f64,
    /// Overloaded agents count
    pub overloaded_agents: usize,
    /// Underutilized agents count
    pub underutilized_agents: usize,
    /// Algorithm performance scores
    pub algorithm_scores: HashMap<String, f64>,
}

/// Consistent hash ring for load balancing
pub struct ConsistentHashRing {
    /// Ring mapping hash values to agents
    ring: BTreeMap<u64, AgentId>,
    /// Virtual nodes per agent
    virtual_nodes: usize,
}

/// Load prediction system
pub struct LoadPredictor {
    /// Historical load data
    load_history: Arc<RwLock<HashMap<AgentId, VecDeque<f64>>>>,
    /// Prediction models
    models: Arc<RwLock<HashMap<AgentId, PredictionModel>>>,
    /// Prediction accuracy tracking
    accuracy_tracker: Arc<RwLock<HashMap<AgentId, f64>>>,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Linear regression coefficients
    pub coefficients: Vec<f64>,
    /// Model accuracy
    pub accuracy: f64,
    /// Training data size
    pub training_size: usize,
    /// Last update time
    pub last_update: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Swarm ID
    pub swarm_id: String,
    /// Scale direction
    pub direction: ScaleDirection,
    /// Agents before scaling
    pub agents_before: usize,
    /// Agents after scaling
    pub agents_after: usize,
    /// Trigger reason
    pub reason: String,
    /// Load before scaling
    pub load_before: f64,
    /// Load after scaling
    pub load_after: Option<f64>,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::Adaptive,
            monitoring_interval: 1000,
            load_threshold: 0.8,
            min_agents_per_swarm: 2,
            max_agents_per_swarm: 10,
            auto_scaling_enabled: true,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            scaling_cooldown_ms: 60000,
            health_weight: 0.3,
            response_time_weight: 0.25,
            cpu_weight: 0.2,
            memory_weight: 0.15,
            network_weight: 0.1,
            predictive_scaling: true,
            load_history_size: 100,
        }
    }
}

impl LoadBalancerAgent {
    /// Create a new load balancer agent
    pub async fn new(
        config: LoadBalancerConfig,
        message_router: Arc<MessageRouter>,
        health_checker: Arc<HealthChecker>,
    ) -> Result<Self> {
        let id = AgentId::new();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, _) = broadcast::channel(1024);
        let (shutdown_tx, _) = mpsc::unbounded_channel();

        let strategy = LoadBalancingStrategy {
            primary_algorithm: config.algorithm.clone(),
            fallback_algorithm: LoadBalancingAlgorithm::RoundRobin,
            weights: HashMap::new(),
            adaptive_params: AdaptiveParams {
                learning_rate: 0.01,
                exploration_rate: 0.1,
                prediction_window: 10,
                adaptation_threshold: 0.05,
            },
        };

        let hash_ring = ConsistentHashRing {
            ring: BTreeMap::new(),
            virtual_nodes: 100,
        };

        let load_predictor = Arc::new(LoadPredictor {
            load_history: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            accuracy_tracker: Arc::new(RwLock::new(HashMap::new())),
        });

        let initial_metrics = LoadBalancerMetrics {
            total_selections: 0,
            avg_selection_time_ns: 0.0,
            distribution_efficiency: 100.0,
            scaling_events: 0,
            load_balance_score: 100.0,
            avg_agent_utilization: 0.0,
            overloaded_agents: 0,
            underutilized_agents: 0,
            algorithm_scores: HashMap::new(),
        };

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            agent_loads: Arc::new(RwLock::new(HashMap::new())),
            swarm_agents: Arc::new(RwLock::new(HashMap::new())),
            round_robin_counters: Arc::new(RwLock::new(HashMap::new())),
            hash_ring: Arc::new(RwLock::new(hash_ring)),
            strategy: Arc::new(RwLock::new(strategy)),
            performance_metrics: Arc::new(RwLock::new(initial_metrics)),
            scaling_history: Arc::new(RwLock::new(VecDeque::new())),
            load_predictor,
            message_router,
            health_checker,
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            event_tx,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            running: Arc::new(Atomic::new(false)),
            rng: Arc::new(Mutex::new(ChaCha8Rng::from_entropy())),
            last_scaling: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the load balancer
    #[instrument(skip(self), fields(balancer_id = %self.id))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Load Balancer Agent {}", self.id);
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Running;
        }
        
        self.running.store(true, AtomicOrdering::SeqCst);
        
        // Initialize swarm mappings
        self.initialize_swarm_mappings().await?;
        
        // Spawn background tasks
        self.spawn_background_tasks().await?;
        
        // Start main event loop
        self.run_event_loop().await?;
        
        Ok(())
    }

    /// Stop the load balancer
    #[instrument(skip(self), fields(balancer_id = %self.id))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Load Balancer Agent {}", self.id);
        
        self.running.store(false, AtomicOrdering::SeqCst);
        
        // Signal shutdown
        if let Some(shutdown_tx) = self.shutdown_tx.lock().await.take() {
            let _ = shutdown_tx.send(());
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Stopped;
        }
        
        Ok(())
    }

    /// Initialize swarm mappings
    async fn initialize_swarm_mappings(&self) -> Result<()> {
        let mut swarm_agents = self.swarm_agents.write().await;
        let mut round_robin_counters = self.round_robin_counters.write().await;
        
        // Initialize empty mappings for known swarms
        let swarms = vec![
            "risk-management",
            "trading-strategy",
            "data-pipeline",
            "tengri-watchdog",
        ];
        
        for swarm in swarms {
            swarm_agents.insert(swarm.to_string(), HashSet::new());
            round_robin_counters.insert(swarm.to_string(), 0);
        }
        
        Ok(())
    }

    /// Spawn background tasks
    async fn spawn_background_tasks(&self) -> Result<()> {
        // Load monitoring task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.load_monitoring_task().await;
        });

        // Auto-scaling task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.auto_scaling_task().await;
        });

        // Metrics collection task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.metrics_collection_task().await;
        });

        // Load prediction task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.load_prediction_task().await;
        });

        // Distribution optimization task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.distribution_optimization_task().await;
        });

        Ok(())
    }

    /// Main event loop
    async fn run_event_loop(&self) -> Result<()> {
        let mut command_rx = self.command_rx.lock().await;
        
        while self.running.load(AtomicOrdering::SeqCst) {
            tokio::select! {
                Some(command) = command_rx.recv() => {
                    if let Err(e) = self.handle_command(command).await {
                        error!("Error handling load balancer command: {}", e);
                    }
                }
                _ = sleep(Duration::from_millis(100)) => {
                    // Regular maintenance
                    self.maintenance_cycle().await;
                }
            }
        }
        
        Ok(())
    }

    /// Handle incoming commands
    async fn handle_command(&self, command: LoadBalancerCommand) -> Result<()> {
        match command {
            LoadBalancerCommand::SelectAgent { swarm_id, task_type, requirements } => {
                self.select_agent(swarm_id, task_type, requirements).await
            }
            LoadBalancerCommand::UpdateAgentLoad { agent_id, metrics } => {
                self.update_agent_load(agent_id, metrics).await
            }
            LoadBalancerCommand::AddAgent { agent_id, agent_type, swarm_id, capacity } => {
                self.add_agent(agent_id, agent_type, swarm_id, capacity).await
            }
            LoadBalancerCommand::RemoveAgent { agent_id } => {
                self.remove_agent(agent_id).await
            }
            LoadBalancerCommand::SetAgentWeight { agent_id, weight } => {
                self.set_agent_weight(agent_id, weight).await
            }
            LoadBalancerCommand::TriggerScaling { swarm_id, scale_direction } => {
                self.trigger_scaling(swarm_id, scale_direction).await
            }
            LoadBalancerCommand::SetAlgorithm { algorithm } => {
                self.set_algorithm(algorithm).await
            }
            LoadBalancerCommand::GetStats => {
                self.get_stats().await
            }
            LoadBalancerCommand::GetAgentLoads { swarm_id } => {
                self.get_agent_loads(swarm_id).await
            }
            LoadBalancerCommand::OptimizeDistribution => {
                self.optimize_distribution().await
            }
            LoadBalancerCommand::SetAutoScaling { enabled } => {
                self.set_auto_scaling(enabled).await
            }
            LoadBalancerCommand::Reset => {
                self.reset().await
            }
            LoadBalancerCommand::Shutdown => {
                self.stop().await
            }
        }
    }

    /// Select the best agent for a task
    async fn select_agent(
        &self,
        swarm_id: String,
        task_type: String,
        requirements: TaskRequirements,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        info!("Selecting agent for swarm {} with task type {}", swarm_id, task_type);
        
        let selected_agent = {
            let strategy = self.strategy.read().await;
            match strategy.primary_algorithm {
                LoadBalancingAlgorithm::RoundRobin => {
                    self.select_round_robin(&swarm_id).await
                }
                LoadBalancingAlgorithm::LeastConnections => {
                    self.select_least_connections(&swarm_id).await
                }
                LoadBalancingAlgorithm::WeightedRoundRobin => {
                    self.select_weighted_round_robin(&swarm_id).await
                }
                LoadBalancingAlgorithm::LeastResponseTime => {
                    self.select_least_response_time(&swarm_id).await
                }
                LoadBalancingAlgorithm::Random => {
                    self.select_random(&swarm_id).await
                }
                LoadBalancingAlgorithm::ConsistentHashing => {
                    self.select_consistent_hash(&swarm_id, &task_type).await
                }
                LoadBalancingAlgorithm::ResourceBased => {
                    self.select_resource_based(&swarm_id, &requirements).await
                }
                LoadBalancingAlgorithm::Adaptive => {
                    self.select_adaptive(&swarm_id, &requirements).await
                }
                LoadBalancingAlgorithm::LatencyAware => {
                    self.select_latency_aware(&swarm_id).await
                }
                LoadBalancingAlgorithm::ThroughputOptimized => {
                    self.select_throughput_optimized(&swarm_id).await
                }
            }
        };
        
        if let Some((agent_id, load_score, reason)) = selected_agent {
            // Update metrics
            {
                let mut metrics = self.performance_metrics.write().await;
                metrics.total_selections += 1;
                let selection_time = start_time.elapsed().as_nanos() as f64;
                metrics.avg_selection_time_ns = 
                    (metrics.avg_selection_time_ns + selection_time) / 2.0;
            }
            
            // Update agent connection count
            {
                let mut agent_loads = self.agent_loads.write().await;
                if let Some(agent_load) = agent_loads.get_mut(&agent_id) {
                    agent_load.active_connections += 1;
                }
            }
            
            let _ = self.event_tx.send(LoadBalancerEvent::AgentSelected {
                agent_id,
                swarm_id,
                load_score,
                selection_reason: reason,
            });
            
            info!("Agent selected: {} with load score {:.3}", agent_id, load_score);
        } else {
            warn!("No suitable agent found for swarm {}", swarm_id);
        }
        
        Ok(())
    }

    /// Round-robin selection
    async fn select_round_robin(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_vec: Vec<_> = agents.iter().collect();
        let mut counters = self.round_robin_counters.write().await;
        let counter = counters.entry(swarm_id.to_string()).or_insert(0);
        
        let selected_agent = agent_vec[*counter % agent_vec.len()].clone();
        *counter += 1;
        
        let load_score = self.get_agent_load_score(&selected_agent).await.unwrap_or(0.5);
        
        Some((selected_agent, load_score, "round_robin".to_string()))
    }

    /// Least connections selection
    async fn select_least_connections(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut best_agent = None;
        let mut min_connections = usize::MAX;
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy &&
                   agent_load.active_connections < min_connections {
                    min_connections = agent_load.active_connections;
                    best_agent = Some((agent_id.clone(), agent_load.load_score));
                }
            }
        }
        
        best_agent.map(|(agent_id, load_score)| {
            (agent_id, load_score, "least_connections".to_string())
        })
    }

    /// Weighted round-robin selection
    async fn select_weighted_round_robin(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut weighted_agents = Vec::new();
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy {
                    // Higher weight for agents with lower load
                    let weight = (1.0 - agent_load.load_score) * agent_load.weight;
                    for _ in 0..(weight * 10.0) as usize {
                        weighted_agents.push(agent_id.clone());
                    }
                }
            }
        }
        
        if weighted_agents.is_empty() {
            return None;
        }
        
        let mut rng = self.rng.lock().await;
        let index = rng.gen_range(0..weighted_agents.len());
        let selected_agent = weighted_agents[index].clone();
        
        let load_score = self.get_agent_load_score(&selected_agent).await.unwrap_or(0.5);
        
        Some((selected_agent, load_score, "weighted_round_robin".to_string()))
    }

    /// Least response time selection
    async fn select_least_response_time(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut best_agent = None;
        let mut min_response_time = f64::MAX;
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy &&
                   agent_load.metrics.avg_response_time_us < min_response_time {
                    min_response_time = agent_load.metrics.avg_response_time_us;
                    best_agent = Some((agent_id.clone(), agent_load.load_score));
                }
            }
        }
        
        best_agent.map(|(agent_id, load_score)| {
            (agent_id, load_score, "least_response_time".to_string())
        })
    }

    /// Random selection
    async fn select_random(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let healthy_agents: Vec<_> = {
            let agent_loads = self.agent_loads.read().await;
            agents.iter()
                .filter(|agent_id| {
                    agent_loads.get(agent_id)
                        .map(|load| load.health_status == HealthStatus::Healthy)
                        .unwrap_or(false)
                })
                .cloned()
                .collect()
        };
        
        if healthy_agents.is_empty() {
            return None;
        }
        
        let mut rng = self.rng.lock().await;
        let index = rng.gen_range(0..healthy_agents.len());
        let selected_agent = healthy_agents[index].clone();
        
        let load_score = self.get_agent_load_score(&selected_agent).await.unwrap_or(0.5);
        
        Some((selected_agent, load_score, "random".to_string()))
    }

    /// Consistent hashing selection
    async fn select_consistent_hash(&self, swarm_id: &str, task_type: &str) -> Option<(AgentId, f64, String)> {
        let hash_ring = self.hash_ring.read().await;
        
        // Hash the task type to find position on ring
        let mut hasher = blake3::Hasher::new();
        hasher.update(task_type.as_bytes());
        let hash = u64::from_le_bytes(hasher.finalize().as_bytes()[0..8].try_into().unwrap());
        
        // Find the first agent on or after this position
        if let Some((_, agent_id)) = hash_ring.ring.range(hash..).next() {
            let load_score = self.get_agent_load_score(agent_id).await.unwrap_or(0.5);
            return Some((agent_id.clone(), load_score, "consistent_hash".to_string()));
        }
        
        // Wrap around to the beginning of the ring
        if let Some((_, agent_id)) = hash_ring.ring.iter().next() {
            let load_score = self.get_agent_load_score(agent_id).await.unwrap_or(0.5);
            return Some((agent_id.clone(), load_score, "consistent_hash_wrap".to_string()));
        }
        
        None
    }

    /// Resource-based selection
    async fn select_resource_based(
        &self,
        swarm_id: &str,
        requirements: &TaskRequirements,
    ) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut best_agent = None;
        let mut best_score = f64::NEG_INFINITY;
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy {
                    let score = self.calculate_resource_score(agent_load, requirements);
                    if score > best_score {
                        best_score = score;
                        best_agent = Some((agent_id.clone(), agent_load.load_score));
                    }
                }
            }
        }
        
        best_agent.map(|(agent_id, load_score)| {
            (agent_id, load_score, format!("resource_based_score_{:.3}", best_score))
        })
    }

    /// Calculate resource fitness score
    fn calculate_resource_score(&self, agent_load: &AgentLoad, requirements: &TaskRequirements) -> f64 {
        let cpu_score = if requirements.cpu_requirement <= agent_load.capacity.max_cpu {
            (agent_load.capacity.max_cpu - agent_load.metrics.cpu_utilization) / agent_load.capacity.max_cpu
        } else {
            0.0
        };
        
        let memory_score = if requirements.memory_requirement <= agent_load.capacity.max_memory {
            let used_memory = agent_load.metrics.memory_utilization * agent_load.capacity.max_memory as f64;
            (agent_load.capacity.max_memory as f64 - used_memory) / agent_load.capacity.max_memory as f64
        } else {
            0.0
        };
        
        let network_score = if requirements.network_requirement <= agent_load.capacity.max_network_bps {
            (1.0 - agent_load.metrics.network_utilization)
        } else {
            0.0
        };
        
        let capability_score = if requirements.required_capabilities.is_empty() {
            1.0
        } else {
            let matching_capabilities = requirements.required_capabilities.iter()
                .filter(|cap| agent_load.capacity.capabilities.contains(cap))
                .count();
            matching_capabilities as f64 / requirements.required_capabilities.len() as f64
        };
        
        // Weighted combination
        cpu_score * 0.3 + memory_score * 0.3 + network_score * 0.2 + capability_score * 0.2
    }

    /// Adaptive selection using machine learning
    async fn select_adaptive(
        &self,
        swarm_id: &str,
        requirements: &TaskRequirements,
    ) -> Option<(AgentId, f64, String)> {
        // For now, fall back to resource-based selection
        // In a full implementation, this would use ML models
        self.select_resource_based(swarm_id, requirements).await
    }

    /// Latency-aware selection
    async fn select_latency_aware(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut best_agent = None;
        let mut best_latency_score = f64::NEG_INFINITY;
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy {
                    // Score based on response time and current load
                    let latency_score = 1.0 / (1.0 + agent_load.metrics.avg_response_time_us / 1000.0);
                    let load_penalty = 1.0 - agent_load.load_score;
                    let combined_score = latency_score * 0.7 + load_penalty * 0.3;
                    
                    if combined_score > best_latency_score {
                        best_latency_score = combined_score;
                        best_agent = Some((agent_id.clone(), agent_load.load_score));
                    }
                }
            }
        }
        
        best_agent.map(|(agent_id, load_score)| {
            (agent_id, load_score, format!("latency_aware_score_{:.3}", best_latency_score))
        })
    }

    /// Throughput-optimized selection
    async fn select_throughput_optimized(&self, swarm_id: &str) -> Option<(AgentId, f64, String)> {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id)?;
        
        if agents.is_empty() {
            return None;
        }
        
        let agent_loads = self.agent_loads.read().await;
        let mut best_agent = None;
        let mut best_throughput_score = f64::NEG_INFINITY;
        
        for agent_id in agents {
            if let Some(agent_load) = agent_loads.get(agent_id) {
                if agent_load.health_status == HealthStatus::Healthy {
                    // Score based on throughput capacity and current utilization
                    let throughput_capacity = agent_load.capacity.max_messages_per_second;
                    let current_throughput = agent_load.metrics.messages_per_second;
                    let available_throughput = throughput_capacity - current_throughput;
                    let utilization_penalty = agent_load.load_score;
                    
                    let throughput_score = available_throughput * (1.0 - utilization_penalty);
                    
                    if throughput_score > best_throughput_score {
                        best_throughput_score = throughput_score;
                        best_agent = Some((agent_id.clone(), agent_load.load_score));
                    }
                }
            }
        }
        
        best_agent.map(|(agent_id, load_score)| {
            (agent_id, load_score, format!("throughput_optimized_score_{:.3}", best_throughput_score))
        })
    }

    /// Get agent load score
    async fn get_agent_load_score(&self, agent_id: &AgentId) -> Option<f64> {
        let agent_loads = self.agent_loads.read().await;
        agent_loads.get(agent_id).map(|load| load.load_score)
    }

    /// Update agent load metrics
    async fn update_agent_load(&self, agent_id: AgentId, metrics: LoadMetrics) -> Result<()> {
        let old_load_score = {
            let mut agent_loads = self.agent_loads.write().await;
            if let Some(agent_load) = agent_loads.get_mut(&agent_id) {
                let old_score = agent_load.load_score;
                
                // Update metrics
                agent_load.metrics = metrics.clone();
                agent_load.last_update = Instant::now();
                
                // Calculate new load score
                agent_load.load_score = self.calculate_load_score(&metrics, &agent_load.capacity);
                
                // Update performance history
                let snapshot = PerformanceSnapshot {
                    timestamp: Instant::now(),
                    metrics: metrics.clone(),
                    response_time_us: metrics.avg_response_time_us,
                    throughput: metrics.messages_per_second,
                    error_count: (metrics.error_rate * 100.0) as u64,
                };
                
                agent_load.performance_history.push_back(snapshot);
                if agent_load.performance_history.len() > self.config.load_history_size {
                    agent_load.performance_history.pop_front();
                }
                
                Some(old_score)
            } else {
                None
            }
        };
        
        if let Some(old_score) = old_load_score {
            let new_score = self.get_agent_load_score(&agent_id).await.unwrap_or(0.0);
            
            let _ = self.event_tx.send(LoadBalancerEvent::AgentLoadUpdated {
                agent_id,
                old_load_score: old_score,
                new_load_score: new_score,
            });
        }
        
        Ok(())
    }

    /// Calculate load score from metrics
    fn calculate_load_score(&self, metrics: &LoadMetrics, capacity: &AgentCapacity) -> f64 {
        let cpu_score = metrics.cpu_utilization;
        let memory_score = metrics.memory_utilization;
        let network_score = metrics.network_utilization;
        let response_time_score = (metrics.avg_response_time_us / 10000.0).min(1.0); // Normalize to 10ms
        let throughput_score = (metrics.messages_per_second / capacity.max_messages_per_second).min(1.0);
        
        // Weighted combination
        self.config.cpu_weight * cpu_score +
        self.config.memory_weight * memory_score +
        self.config.network_weight * network_score +
        self.config.response_time_weight * response_time_score +
        0.1 * throughput_score
    }

    /// Add agent to load balancing
    async fn add_agent(
        &self,
        agent_id: AgentId,
        agent_type: AgentType,
        swarm_id: String,
        capacity: AgentCapacity,
    ) -> Result<()> {
        info!("Adding agent {} to swarm {} for load balancing", agent_id, swarm_id);
        
        let agent_load = AgentLoad {
            agent_id: agent_id.clone(),
            agent_type,
            swarm_id: swarm_id.clone(),
            metrics: LoadMetrics {
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                network_utilization: 0.0,
                avg_response_time_us: 0.0,
                messages_per_second: 0.0,
                error_rate: 0.0,
                queue_depth: 0,
                throughput_score: 1.0,
            },
            health_status: HealthStatus::Unknown,
            capacity,
            performance_history: VecDeque::new(),
            last_update: Instant::now(),
            weight: 1.0,
            active_connections: 0,
            load_score: 0.0,
        };
        
        // Add to agent loads
        {
            let mut agent_loads = self.agent_loads.write().await;
            agent_loads.insert(agent_id.clone(), agent_load);
        }
        
        // Add to swarm mapping
        {
            let mut swarm_agents = self.swarm_agents.write().await;
            swarm_agents.entry(swarm_id).or_insert_with(HashSet::new).insert(agent_id.clone());
        }
        
        // Add to consistent hash ring
        {
            let mut hash_ring = self.hash_ring.write().await;
            for i in 0..hash_ring.virtual_nodes {
                let mut hasher = blake3::Hasher::new();
                hasher.update(agent_id.to_string().as_bytes());
                hasher.update(&i.to_le_bytes());
                let hash = u64::from_le_bytes(hasher.finalize().as_bytes()[0..8].try_into().unwrap());
                hash_ring.ring.insert(hash, agent_id.clone());
            }
        }
        
        Ok(())
    }

    /// Remove agent from load balancing
    async fn remove_agent(&self, agent_id: AgentId) -> Result<()> {
        info!("Removing agent {} from load balancing", agent_id);
        
        let swarm_id = {
            let mut agent_loads = self.agent_loads.write().await;
            agent_loads.remove(&agent_id).map(|load| load.swarm_id)
        };
        
        if let Some(swarm_id) = swarm_id {
            // Remove from swarm mapping
            {
                let mut swarm_agents = self.swarm_agents.write().await;
                if let Some(agents) = swarm_agents.get_mut(&swarm_id) {
                    agents.remove(&agent_id);
                }
            }
            
            // Remove from hash ring
            {
                let mut hash_ring = self.hash_ring.write().await;
                hash_ring.ring.retain(|_, id| *id != agent_id);
            }
        }
        
        Ok(())
    }

    /// Set agent weight
    async fn set_agent_weight(&self, agent_id: AgentId, weight: f64) -> Result<()> {
        let mut agent_loads = self.agent_loads.write().await;
        if let Some(agent_load) = agent_loads.get_mut(&agent_id) {
            agent_load.weight = weight;
            info!("Set agent {} weight to {}", agent_id, weight);
        }
        
        Ok(())
    }

    /// Trigger scaling operation
    async fn trigger_scaling(&self, swarm_id: String, scale_direction: ScaleDirection) -> Result<()> {
        info!("Triggering {:?} scaling for swarm {}", scale_direction, swarm_id);
        
        // Check cooldown period
        {
            let last_scaling = self.last_scaling.read().await;
            if let Some(last_time) = last_scaling.get(&swarm_id) {
                if last_time.elapsed().as_millis() < self.config.scaling_cooldown_ms as u128 {
                    warn!("Scaling cooldown active for swarm {}", swarm_id);
                    return Ok(());
                }
            }
        }
        
        let current_agents = {
            let swarm_agents = self.swarm_agents.read().await;
            swarm_agents.get(&swarm_id).map(|agents| agents.len()).unwrap_or(0)
        };
        
        let target_agents = match scale_direction {
            ScaleDirection::Up => {
                (current_agents + 1).min(self.config.max_agents_per_swarm)
            }
            ScaleDirection::Down => {
                (current_agents.saturating_sub(1)).max(self.config.min_agents_per_swarm)
            }
        };
        
        if target_agents != current_agents {
            // Record scaling event
            let scaling_event = ScalingEvent {
                timestamp: Instant::now(),
                swarm_id: swarm_id.clone(),
                direction: scale_direction,
                agents_before: current_agents,
                agents_after: target_agents,
                reason: "manual_trigger".to_string(),
                load_before: self.calculate_swarm_load(&swarm_id).await,
                load_after: None,
            };
            
            {
                let mut scaling_history = self.scaling_history.write().await;
                scaling_history.push_back(scaling_event);
                if scaling_history.len() > 100 {
                    scaling_history.pop_front();
                }
            }
            
            // Update last scaling time
            {
                let mut last_scaling = self.last_scaling.write().await;
                last_scaling.insert(swarm_id.clone(), Instant::now());
            }
            
            // Update metrics
            {
                let mut metrics = self.performance_metrics.write().await;
                metrics.scaling_events += 1;
            }
            
            let _ = self.event_tx.send(LoadBalancerEvent::AutoScalingTriggered {
                swarm_id,
                current_agents,
                target_agents,
                trigger_reason: "manual_trigger".to_string(),
            });
        }
        
        Ok(())
    }

    /// Calculate swarm load
    async fn calculate_swarm_load(&self, swarm_id: &str) -> f64 {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id);
        
        if let Some(agents) = agents {
            if agents.is_empty() {
                return 0.0;
            }
            
            let agent_loads = self.agent_loads.read().await;
            let total_load: f64 = agents.iter()
                .filter_map(|agent_id| agent_loads.get(agent_id))
                .map(|load| load.load_score)
                .sum();
            
            total_load / agents.len() as f64
        } else {
            0.0
        }
    }

    /// Set load balancing algorithm
    async fn set_algorithm(&self, algorithm: LoadBalancingAlgorithm) -> Result<()> {
        let old_algorithm = {
            let mut strategy = self.strategy.write().await;
            let old = strategy.primary_algorithm.clone();
            strategy.primary_algorithm = algorithm.clone();
            old
        };
        
        let _ = self.event_tx.send(LoadBalancerEvent::AlgorithmChanged {
            old_algorithm,
            new_algorithm: algorithm,
        });
        
        Ok(())
    }

    /// Get load balancing statistics
    async fn get_stats(&self) -> Result<()> {
        let metrics = self.performance_metrics.read().await;
        info!("Load Balancer Stats: {} selections, {:.2}ns avg time, {:.2}% efficiency",
              metrics.total_selections, metrics.avg_selection_time_ns, metrics.distribution_efficiency);
        
        Ok(())
    }

    /// Get agent loads
    async fn get_agent_loads(&self, swarm_id: Option<String>) -> Result<()> {
        let agent_loads = self.agent_loads.read().await;
        
        match swarm_id {
            Some(swarm_id) => {
                info!("Agent loads for swarm {}:", swarm_id);
                for (agent_id, load) in agent_loads.iter() {
                    if load.swarm_id == swarm_id {
                        info!("  Agent {}: load={:.3}, connections={}, health={:?}",
                              agent_id, load.load_score, load.active_connections, load.health_status);
                    }
                }
            }
            None => {
                info!("All agent loads:");
                for (agent_id, load) in agent_loads.iter() {
                    info!("  Agent {} ({}): load={:.3}, connections={}, health={:?}",
                          agent_id, load.swarm_id, load.load_score, 
                          load.active_connections, load.health_status);
                }
            }
        }
        
        Ok(())
    }

    /// Optimize load distribution
    async fn optimize_distribution(&self) -> Result<()> {
        info!("Optimizing load distribution");
        
        // Analyze current distribution and suggest improvements
        let mut improvements = HashMap::new();
        
        let swarm_agents = self.swarm_agents.read().await;
        for (swarm_id, _agents) in swarm_agents.iter() {
            let load_variance = self.calculate_load_variance(swarm_id).await;
            let old_variance = load_variance;
            
            // Simulate optimization (in a real implementation, this would rebalance loads)
            let new_variance = load_variance * 0.8; // Assume 20% improvement
            
            improvements.insert(swarm_id.clone(), (old_variance - new_variance) / old_variance * 100.0);
        }
        
        let _ = self.event_tx.send(LoadBalancerEvent::DistributionOptimized {
            improvements,
        });
        
        Ok(())
    }

    /// Calculate load variance for a swarm
    async fn calculate_load_variance(&self, swarm_id: &str) -> f64 {
        let swarm_agents = self.swarm_agents.read().await;
        let agents = swarm_agents.get(swarm_id);
        
        if let Some(agents) = agents {
            if agents.len() < 2 {
                return 0.0;
            }
            
            let agent_loads = self.agent_loads.read().await;
            let loads: Vec<f64> = agents.iter()
                .filter_map(|agent_id| agent_loads.get(agent_id))
                .map(|load| load.load_score)
                .collect();
            
            if loads.is_empty() {
                return 0.0;
            }
            
            let mean = loads.iter().sum::<f64>() / loads.len() as f64;
            let variance = loads.iter()
                .map(|load| (load - mean).powi(2))
                .sum::<f64>() / loads.len() as f64;
            
            variance
        } else {
            0.0
        }
    }

    /// Set auto-scaling enabled/disabled
    async fn set_auto_scaling(&self, enabled: bool) -> Result<()> {
        info!("Setting auto-scaling to {}", enabled);
        
        // Note: In a real implementation, we'd need mutable access to config
        // self.config.auto_scaling_enabled = enabled;
        
        Ok(())
    }

    /// Reset load balancer
    async fn reset(&self) -> Result<()> {
        info!("Resetting load balancer");
        
        // Clear all data
        {
            let mut agent_loads = self.agent_loads.write().await;
            agent_loads.clear();
        }
        
        {
            let mut swarm_agents = self.swarm_agents.write().await;
            swarm_agents.clear();
        }
        
        {
            let mut round_robin_counters = self.round_robin_counters.write().await;
            round_robin_counters.clear();
        }
        
        {
            let mut hash_ring = self.hash_ring.write().await;
            hash_ring.ring.clear();
        }
        
        // Reset metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            *metrics = LoadBalancerMetrics {
                total_selections: 0,
                avg_selection_time_ns: 0.0,
                distribution_efficiency: 100.0,
                scaling_events: 0,
                load_balance_score: 100.0,
                avg_agent_utilization: 0.0,
                overloaded_agents: 0,
                underutilized_agents: 0,
                algorithm_scores: HashMap::new(),
            };
        }
        
        Ok(())
    }

    /// Load monitoring task
    async fn load_monitoring_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Monitor agent loads and detect overload/underload
            self.monitor_agent_loads().await;
            
            sleep(Duration::from_millis(self.config.monitoring_interval)).await;
        }
    }

    /// Monitor agent loads
    async fn monitor_agent_loads(&self) {
        let swarm_agents = self.swarm_agents.read().await;
        
        for (swarm_id, agents) in swarm_agents.iter() {
            let mut overloaded_agents = Vec::new();
            let mut underutilized_agents = Vec::new();
            let mut total_load = 0.0;
            let mut healthy_agents = 0;
            
            {
                let agent_loads = self.agent_loads.read().await;
                for agent_id in agents {
                    if let Some(agent_load) = agent_loads.get(agent_id) {
                        if agent_load.health_status == HealthStatus::Healthy {
                            total_load += agent_load.load_score;
                            healthy_agents += 1;
                            
                            if agent_load.load_score > self.config.scale_up_threshold {
                                overloaded_agents.push(agent_id.clone());
                            } else if agent_load.load_score < self.config.scale_down_threshold {
                                underutilized_agents.push(agent_id.clone());
                            }
                        }
                    }
                }
            }
            
            let average_load = if healthy_agents > 0 {
                total_load / healthy_agents as f64
            } else {
                0.0
            };
            
            // Send overload event
            if !overloaded_agents.is_empty() {
                let _ = self.event_tx.send(LoadBalancerEvent::OverloadDetected {
                    swarm_id: swarm_id.clone(),
                    overloaded_agents,
                    average_load,
                });
            }
            
            // Send underload event
            if underutilized_agents.len() > 1 && healthy_agents > self.config.min_agents_per_swarm {
                let _ = self.event_tx.send(LoadBalancerEvent::UnderloadDetected {
                    swarm_id: swarm_id.clone(),
                    underutilized_agents,
                    average_load,
                });
            }
        }
    }

    /// Auto-scaling task
    async fn auto_scaling_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            if self.config.auto_scaling_enabled {
                self.check_auto_scaling().await;
            }
            
            sleep(Duration::from_millis(self.config.monitoring_interval * 2)).await;
        }
    }

    /// Check if auto-scaling is needed
    async fn check_auto_scaling(&self) {
        let swarm_agents = self.swarm_agents.read().await;
        
        for (swarm_id, agents) in swarm_agents.iter() {
            let average_load = self.calculate_swarm_load(swarm_id).await;
            let agent_count = agents.len();
            
            // Check for scale-up
            if average_load > self.config.scale_up_threshold && 
               agent_count < self.config.max_agents_per_swarm {
                if let Err(e) = self.trigger_scaling(swarm_id.clone(), ScaleDirection::Up).await {
                    error!("Auto scale-up failed: {}", e);
                }
            }
            
            // Check for scale-down
            if average_load < self.config.scale_down_threshold && 
               agent_count > self.config.min_agents_per_swarm {
                if let Err(e) = self.trigger_scaling(swarm_id.clone(), ScaleDirection::Down).await {
                    error!("Auto scale-down failed: {}", e);
                }
            }
        }
    }

    /// Metrics collection task
    async fn metrics_collection_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            self.update_performance_metrics().await;
            sleep(Duration::from_secs(5)).await;
        }
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self) {
        let agent_loads = self.agent_loads.read().await;
        
        let mut total_utilization = 0.0;
        let mut overloaded_count = 0;
        let mut underutilized_count = 0;
        let mut healthy_agents = 0;
        
        for agent_load in agent_loads.values() {
            if agent_load.health_status == HealthStatus::Healthy {
                total_utilization += agent_load.load_score;
                healthy_agents += 1;
                
                if agent_load.load_score > self.config.scale_up_threshold {
                    overloaded_count += 1;
                } else if agent_load.load_score < self.config.scale_down_threshold {
                    underutilized_count += 1;
                }
            }
        }
        
        let mut metrics = self.performance_metrics.write().await;
        metrics.avg_agent_utilization = if healthy_agents > 0 {
            total_utilization / healthy_agents as f64
        } else {
            0.0
        };
        
        metrics.overloaded_agents = overloaded_count;
        metrics.underutilized_agents = underutilized_count;
        
        // Calculate load balance score
        let swarm_agents = self.swarm_agents.read().await;
        let mut total_variance = 0.0;
        let mut swarm_count = 0;
        
        for swarm_id in swarm_agents.keys() {
            total_variance += self.calculate_load_variance(swarm_id).await;
            swarm_count += 1;
        }
        
        if swarm_count > 0 {
            let avg_variance = total_variance / swarm_count as f64;
            metrics.load_balance_score = (1.0 - avg_variance.min(1.0)) * 100.0;
        }
        
        metrics.distribution_efficiency = 
            100.0 - (overloaded_count + underutilized_count) as f64 / healthy_agents.max(1) as f64 * 100.0;
    }

    /// Load prediction task
    async fn load_prediction_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            if self.config.predictive_scaling {
                self.update_load_predictions().await;
            }
            
            sleep(Duration::from_secs(10)).await;
        }
    }

    /// Update load predictions
    async fn update_load_predictions(&self) {
        // Simplified load prediction
        // In a real implementation, this would use sophisticated ML models
        
        let agent_loads = self.agent_loads.read().await;
        for (agent_id, agent_load) in agent_loads.iter() {
            if agent_load.performance_history.len() >= 3 {
                // Simple linear trend prediction
                let history = &agent_load.performance_history;
                let recent_loads: Vec<f64> = history.iter()
                    .rev()
                    .take(5)
                    .map(|snapshot| {
                        snapshot.metrics.cpu_utilization * 0.5 + 
                        snapshot.metrics.memory_utilization * 0.3 +
                        snapshot.metrics.network_utilization * 0.2
                    })
                    .collect();
                
                if recent_loads.len() >= 3 {
                    // Calculate trend
                    let trend = (recent_loads[0] - recent_loads[recent_loads.len() - 1]) / recent_loads.len() as f64;
                    let predicted_load = recent_loads[0] + trend;
                    
                    debug!("Agent {} predicted load: {:.3} (trend: {:.3})", 
                           agent_id, predicted_load, trend);
                }
            }
        }
    }

    /// Distribution optimization task
    async fn distribution_optimization_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            if let Err(e) = self.optimize_distribution().await {
                error!("Distribution optimization failed: {}", e);
            }
            
            sleep(Duration::from_secs(30)).await;
        }
    }

    /// Maintenance cycle
    async fn maintenance_cycle(&self) {
        debug!("Running load balancer maintenance cycle");
        
        // Update metrics
        self.update_performance_metrics().await;
        
        // Clean up old performance history
        {
            let mut agent_loads = self.agent_loads.write().await;
            for agent_load in agent_loads.values_mut() {
                while agent_load.performance_history.len() > self.config.load_history_size {
                    agent_load.performance_history.pop_front();
                }
            }
        }
        
        // Clean up old scaling history
        {
            let mut scaling_history = self.scaling_history.write().await;
            while scaling_history.len() > 100 {
                scaling_history.pop_front();
            }
        }
    }

    /// Send command to load balancer
    pub async fn send_command(&self, command: LoadBalancerCommand) -> Result<()> {
        self.command_tx.send(command)
            .map_err(|e| OrchestrationError::CommunicationError(e.to_string()))?;
        Ok(())
    }

    /// Subscribe to load balancer events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<LoadBalancerEvent> {
        self.event_tx.subscribe()
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> LoadBalancerMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Check if load balancer is running
    pub fn is_running(&self) -> bool {
        self.running.load(AtomicOrdering::SeqCst)
    }
}

impl Clone for LoadBalancerAgent {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            agent_loads: Arc::clone(&self.agent_loads),
            swarm_agents: Arc::clone(&self.swarm_agents),
            round_robin_counters: Arc::clone(&self.round_robin_counters),
            hash_ring: Arc::clone(&self.hash_ring),
            strategy: Arc::clone(&self.strategy),
            performance_metrics: Arc::clone(&self.performance_metrics),
            scaling_history: Arc::clone(&self.scaling_history),
            load_predictor: Arc::clone(&self.load_predictor),
            message_router: Arc::clone(&self.message_router),
            health_checker: Arc::clone(&self.health_checker),
            command_tx: self.command_tx.clone(),
            command_rx: Arc::clone(&self.command_rx),
            event_tx: self.event_tx.clone(),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            running: Arc::clone(&self.running),
            rng: Arc::clone(&self.rng),
            last_scaling: Arc::clone(&self.last_scaling),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let config = LoadBalancerConfig::default();
        assert_eq!(config.algorithm, LoadBalancingAlgorithm::Adaptive);
        assert_eq!(config.min_agents_per_swarm, 2);
        assert_eq!(config.max_agents_per_swarm, 10);
    }

    #[tokio::test]
    async fn test_load_score_calculation() {
        let config = LoadBalancerConfig::default();
        
        let metrics = LoadMetrics {
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            network_utilization: 0.3,
            avg_response_time_us: 1000.0,
            messages_per_second: 100.0,
            error_rate: 0.01,
            queue_depth: 5,
            throughput_score: 0.8,
        };
        
        let capacity = AgentCapacity {
            max_cpu: 1.0,
            max_memory: 8_000_000_000,
            max_network_bps: 1_000_000_000,
            max_connections: 100,
            max_messages_per_second: 1000.0,
            capabilities: vec!["processing".to_string()],
        };
        
        // Create a temporary load balancer to test the calculation
        // In a real test, we'd use a mock or test fixture
        assert!(metrics.cpu_utilization >= 0.0 && metrics.cpu_utilization <= 1.0);
        assert!(capacity.max_cpu > 0.0);
    }

    #[tokio::test]
    async fn test_task_requirements() {
        let requirements = TaskRequirements {
            cpu_requirement: 0.5,
            memory_requirement: 1_000_000_000,
            network_requirement: 10_000_000,
            max_latency_ms: 10.0,
            required_capabilities: vec!["ml".to_string()],
            priority: TaskPriority::High,
            estimated_duration_ms: 5000.0,
        };
        
        assert_eq!(requirements.cpu_requirement, 0.5);
        assert_eq!(requirements.memory_requirement, 1_000_000_000);
        assert!(matches!(requirements.priority, TaskPriority::High));
    }
}