//! Ultra-Low Latency Message Router
//!
//! Implements sub-microsecond message routing with hierarchical topology
//! optimization and intelligent load balancing for swarm coordination.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use dashmap::DashMap;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use lockfree::queue::Queue;
use parking_lot::RwLock as ParkingRwLock;
use flume::{Receiver as FlumeReceiver, Sender as FlumeSender};
use kanal::{Receiver as KanalReceiver, Sender as KanalSender};
use priority_queue::PriorityQueue;
use smallvec::SmallVec;
use bimap::BiMap;
use tracing::{debug, info, warn, error, instrument, span, Level};
use metrics::{counter, histogram, gauge};
use chrono::{DateTime, Utc};

use crate::agents::{MCPMessage, MCPMessageType, MessagePriority, RoutingInfo};
use crate::{MCPOrchestrationError, SwarmType, HierarchyLevel};

/// Router configuration for ultra-low latency operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRouterConfig {
    /// Target latency in nanoseconds (sub-1μs = 1000ns)
    pub target_latency_ns: u64,
    /// Maximum message queue size per route
    pub max_queue_size: usize,
    /// Number of worker threads per CPU core
    pub workers_per_core: usize,
    /// Routing table optimization interval
    pub optimization_interval_ms: u64,
    /// Enable adaptive routing
    pub adaptive_routing: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
    /// Message batching size
    pub batch_size: usize,
    /// Priority queue depth
    pub priority_queue_depth: usize,
    /// Enable zero-copy optimization
    pub zero_copy_enabled: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
}

impl Default for MessageRouterConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 1000, // 1 microsecond
            max_queue_size: 1000000,
            workers_per_core: 2,
            optimization_interval_ms: 1000,
            adaptive_routing: true,
            compression_threshold: 1024,
            batch_size: 100,
            priority_queue_depth: 1000,
            zero_copy_enabled: true,
            memory_pool_size: 10000,
        }
    }
}

/// Route entry for optimized message routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEntry {
    pub source: String,
    pub target: String,
    pub path: SmallVec<[String; 8]>, // Most paths are short
    pub latency_ns: u64,
    pub bandwidth_mbps: u64,
    pub congestion_level: f64,
    pub reliability_score: f64,
    pub last_updated: DateTime<Utc>,
    pub use_count: AtomicU64,
}

/// Route metrics for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetrics {
    pub total_messages: u64,
    pub successful_messages: u64,
    pub failed_messages: u64,
    pub average_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
    pub p95_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub throughput_msg_per_sec: f64,
    pub bandwidth_utilization: f64,
    pub error_rate: f64,
}

/// Message batch for optimized processing
#[derive(Debug, Clone)]
pub struct MessageBatch {
    pub messages: SmallVec<[MCPMessage; 100]>,
    pub batch_id: String,
    pub priority: MessagePriority,
    pub created_at: Instant,
    pub target_latency_ns: u64,
}

/// Routing table with lockfree operations
pub struct RoutingTable {
    /// Direct route lookup (source -> target -> route)
    direct_routes: Arc<DashMap<String, DashMap<String, RouteEntry>>>,
    /// Hierarchical routing cache
    hierarchy_cache: Arc<DashMap<String, SmallVec<[String; 4]>>>,
    /// Swarm-based routing
    swarm_routes: Arc<DashMap<SwarmType, Vec<String>>>,
    /// Priority routes for critical messages
    priority_routes: Arc<DashMap<String, RouteEntry>>,
    /// Route metrics
    route_metrics: Arc<DashMap<String, RouteMetrics>>,
    /// Last optimization timestamp
    last_optimization: Arc<AtomicU64>,
}

/// Message Router with ultra-low latency optimization
pub struct MessageRouter {
    config: MessageRouterConfig,
    routing_table: Arc<RoutingTable>,
    message_pools: Arc<MessagePools>,
    worker_pool: Arc<WorkerPool>,
    latency_tracker: Arc<LatencyTracker>,
    congestion_controller: Arc<CongestionController>,
    route_optimizer: Arc<RouteOptimizer>,
    priority_scheduler: Arc<PriorityScheduler>,
    compression_engine: Arc<CompressionEngine>,
    metrics_collector: Arc<RouterMetricsCollector>,
    shutdown_signal: Arc<Mutex<Option<Sender<()>>>>,
}

/// Memory pools for zero-copy message handling
pub struct MessagePools {
    /// Message object pool
    message_pool: Arc<Queue<Box<MCPMessage>>>,
    /// Batch object pool
    batch_pool: Arc<Queue<Box<MessageBatch>>>,
    /// Buffer pool for serialization
    buffer_pool: Arc<Queue<Vec<u8>>>,
    /// Route cache pool
    route_cache_pool: Arc<Queue<SmallVec<[String; 8]>>>,
}

/// Worker pool for parallel message processing
pub struct WorkerPool {
    /// Worker threads
    workers: Vec<WorkerThread>,
    /// Work distribution channel
    work_sender: FlumeSender<WorkItem>,
    work_receiver: Arc<Mutex<FlumeReceiver<WorkItem>>>,
    /// Load balancer
    load_balancer: Arc<WorkerLoadBalancer>,
}

/// Individual worker thread
pub struct WorkerThread {
    id: usize,
    thread_handle: tokio::task::JoinHandle<()>,
    local_queue: Arc<Queue<WorkItem>>,
    metrics: Arc<WorkerMetrics>,
}

/// Work item for processing
#[derive(Debug, Clone)]
pub enum WorkItem {
    RouteMessage(MCPMessage),
    RouteBatch(MessageBatch),
    OptimizeRoutes,
    UpdateMetrics,
    CompressMessage(MCPMessage),
    DecompressMessage(Vec<u8>),
}

/// Worker metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub worker_id: usize,
    pub messages_processed: AtomicU64,
    pub processing_time_ns: AtomicU64,
    pub queue_depth: AtomicU64,
    pub cpu_utilization: f64,
}

/// Worker load balancer
pub struct WorkerLoadBalancer {
    worker_loads: Arc<DashMap<usize, f64>>,
    assignment_strategy: LoadBalancingStrategy,
    last_assignment: Arc<AtomicU64>,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    PowerOfTwoChoices,
    ConsistentHashing,
    LatencyBased,
}

/// Latency tracker for sub-microsecond monitoring
pub struct LatencyTracker {
    /// Latency measurements per route
    route_latencies: Arc<DashMap<String, LatencyHistogram>>,
    /// Global latency statistics
    global_latency: Arc<RwLock<LatencyStats>>,
    /// Latency targets
    targets: Arc<DashMap<String, u64>>,
    /// SLA violations
    sla_violations: Arc<AtomicU64>,
}

/// Latency histogram for detailed analysis
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    pub buckets: Vec<AtomicU64>,
    pub bucket_boundaries: Vec<u64>,
    pub total_samples: AtomicU64,
    pub sum_ns: AtomicU64,
    pub min_ns: AtomicU64,
    pub max_ns: AtomicU64,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean_ns: u64,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub stddev_ns: f64,
}

/// Congestion controller for adaptive flow control
pub struct CongestionController {
    /// Congestion levels per route
    congestion_levels: Arc<DashMap<String, f64>>,
    /// Flow control settings
    flow_control: Arc<RwLock<FlowControlSettings>>,
    /// Backpressure signals
    backpressure_queue: Arc<Queue<BackpressureSignal>>,
    /// Adaptive algorithms
    algorithms: Vec<Box<dyn CongestionAlgorithm>>,
}

/// Flow control settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlSettings {
    pub max_inflight_messages: usize,
    pub congestion_window: usize,
    pub slow_start_threshold: usize,
    pub timeout_ms: u64,
    pub retransmission_timeout_ms: u64,
}

/// Backpressure signal
#[derive(Debug, Clone)]
pub struct BackpressureSignal {
    pub route_id: String,
    pub congestion_level: f64,
    pub recommended_action: BackpressureAction,
    pub timestamp: Instant,
}

/// Backpressure actions
#[derive(Debug, Clone)]
pub enum BackpressureAction {
    ReduceRate(f64),
    SwitchRoute(String),
    BufferMessages,
    DropLowPriority,
    RequestMoreCapacity,
}

/// Congestion algorithm trait
pub trait CongestionAlgorithm: Send + Sync {
    fn detect_congestion(&self, metrics: &RouteMetrics) -> f64;
    fn recommend_action(&self, congestion_level: f64) -> BackpressureAction;
    fn update_parameters(&self, feedback: &CongestionFeedback);
}

/// Congestion feedback
#[derive(Debug, Clone)]
pub struct CongestionFeedback {
    pub route_id: String,
    pub latency_change: f64,
    pub throughput_change: f64,
    pub packet_loss: f64,
}

/// Route optimizer for dynamic path selection
pub struct RouteOptimizer {
    /// Optimization algorithms
    algorithms: Vec<Box<dyn RoutingAlgorithm>>,
    /// Route quality scores
    route_scores: Arc<DashMap<String, f64>>,
    /// Optimization history
    optimization_history: Arc<RwLock<Vec<OptimizationEvent>>>,
    /// A* pathfinding cache
    pathfinding_cache: Arc<DashMap<String, PathfindingResult>>,
}

/// Routing algorithm trait
pub trait RoutingAlgorithm: Send + Sync {
    fn calculate_best_path(&self, source: &str, target: &str, table: &RoutingTable) -> Option<SmallVec<[String; 8]>>;
    fn algorithm_name(&self) -> &str;
    fn optimization_weight(&self) -> f64;
}

/// Optimization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: OptimizationEventType,
    pub route_id: String,
    pub old_latency_ns: u64,
    pub new_latency_ns: u64,
    pub improvement_pct: f64,
}

/// Optimization event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationEventType {
    RouteDiscovered,
    RouteOptimized,
    RouteRemoved,
    PathUpdated,
    MetricsImproved,
}

/// Pathfinding result cache
#[derive(Debug, Clone)]
pub struct PathfindingResult {
    pub path: SmallVec<[String; 8]>,
    pub cost: f64,
    pub heuristic: f64,
    pub cached_at: Instant,
    pub validity_duration: Duration,
}

/// Priority scheduler for message ordering
pub struct PriorityScheduler {
    /// Priority queues per message type
    priority_queues: Arc<DashMap<MessagePriority, PriorityQueue<String, i64>>>,
    /// Scheduling policies
    policies: Arc<RwLock<SchedulingPolicies>>,
    /// Deadline tracking
    deadlines: Arc<DashMap<String, Instant>>,
    /// Scheduler metrics
    metrics: Arc<SchedulerMetrics>,
}

/// Scheduling policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingPolicies {
    pub critical_queue_weight: f64,
    pub high_queue_weight: f64,
    pub normal_queue_weight: f64,
    pub low_queue_weight: f64,
    pub deadline_enforcement: bool,
    pub starvation_prevention: bool,
}

/// Scheduler metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    pub critical_messages_scheduled: AtomicU64,
    pub high_messages_scheduled: AtomicU64,
    pub normal_messages_scheduled: AtomicU64,
    pub low_messages_scheduled: AtomicU64,
    pub deadline_violations: AtomicU64,
    pub average_scheduling_delay_ns: AtomicU64,
}

/// Compression engine for message optimization
pub struct CompressionEngine {
    /// Compression algorithms
    algorithms: HashMap<String, Box<dyn CompressionAlgorithm>>,
    /// Compression ratios
    compression_ratios: Arc<DashMap<String, f64>>,
    /// Compression cache
    compression_cache: Arc<DashMap<String, CompressedMessage>>,
    /// Algorithm selector
    algorithm_selector: Arc<CompressionSelector>,
}

/// Compression algorithm trait
pub trait CompressionAlgorithm: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError>;
    fn algorithm_name(&self) -> &str;
    fn compression_level(&self) -> u8;
}

/// Compression error
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
    #[error("Invalid algorithm: {0}")]
    InvalidAlgorithm(String),
}

/// Compressed message
#[derive(Debug, Clone)]
pub struct CompressedMessage {
    pub original_size: usize,
    pub compressed_data: Vec<u8>,
    pub algorithm: String,
    pub compression_ratio: f64,
    pub compressed_at: Instant,
}

/// Compression algorithm selector
pub struct CompressionSelector {
    selection_strategy: CompressionStrategy,
    performance_cache: Arc<DashMap<String, CompressionPerformance>>,
}

/// Compression strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    Fastest,
    SmallestSize,
    Balanced,
    Adaptive,
}

/// Compression performance metrics
#[derive(Debug, Clone)]
pub struct CompressionPerformance {
    pub compression_time_ns: u64,
    pub decompression_time_ns: u64,
    pub compression_ratio: f64,
    pub cpu_cost: f64,
}

/// Router metrics collector
pub struct RouterMetricsCollector {
    /// Message metrics
    message_metrics: Arc<DashMap<String, MessageMetrics>>,
    /// Route metrics
    route_metrics: Arc<DashMap<String, RouteMetrics>>,
    /// Global metrics
    global_metrics: Arc<RwLock<GlobalRouterMetrics>>,
    /// Collection interval
    collection_interval: Duration,
}

/// Message metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetrics {
    pub message_type: MCPMessageType,
    pub total_count: AtomicU64,
    pub success_count: AtomicU64,
    pub failure_count: AtomicU64,
    pub average_size_bytes: AtomicU64,
    pub processing_time_ns: AtomicU64,
}

/// Global router metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRouterMetrics {
    pub total_messages_routed: u64,
    pub messages_per_second: f64,
    pub average_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub route_optimization_count: u64,
    pub congestion_events: u64,
    pub compression_savings_bytes: u64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
}

impl MessageRouter {
    /// Create new ultra-low latency message router
    pub async fn new() -> Result<Self, MCPOrchestrationError> {
        let config = MessageRouterConfig::default();
        
        let routing_table = Arc::new(RoutingTable {
            direct_routes: Arc::new(DashMap::new()),
            hierarchy_cache: Arc::new(DashMap::new()),
            swarm_routes: Arc::new(DashMap::new()),
            priority_routes: Arc::new(DashMap::new()),
            route_metrics: Arc::new(DashMap::new()),
            last_optimization: Arc::new(AtomicU64::new(0)),
        });
        
        let message_pools = Arc::new(MessagePools {
            message_pool: Arc::new(Queue::new()),
            batch_pool: Arc::new(Queue::new()),
            buffer_pool: Arc::new(Queue::new()),
            route_cache_pool: Arc::new(Queue::new()),
        });
        
        // Initialize memory pools
        Self::initialize_memory_pools(&message_pools, &config).await?;
        
        let (work_sender, work_receiver) = flume::unbounded();
        let worker_pool = Arc::new(WorkerPool {
            workers: Vec::new(),
            work_sender,
            work_receiver: Arc::new(Mutex::new(work_receiver)),
            load_balancer: Arc::new(WorkerLoadBalancer {
                worker_loads: Arc::new(DashMap::new()),
                assignment_strategy: LoadBalancingStrategy::PowerOfTwoChoices,
                last_assignment: Arc::new(AtomicU64::new(0)),
            }),
        });
        
        let latency_tracker = Arc::new(LatencyTracker {
            route_latencies: Arc::new(DashMap::new()),
            global_latency: Arc::new(RwLock::new(LatencyStats {
                mean_ns: 0,
                median_ns: 0,
                p95_ns: 0,
                p99_ns: 0,
                p999_ns: 0,
                min_ns: u64::MAX,
                max_ns: 0,
                stddev_ns: 0.0,
            })),
            targets: Arc::new(DashMap::new()),
            sla_violations: Arc::new(AtomicU64::new(0)),
        });
        
        let congestion_controller = Arc::new(CongestionController {
            congestion_levels: Arc::new(DashMap::new()),
            flow_control: Arc::new(RwLock::new(FlowControlSettings {
                max_inflight_messages: 1000,
                congestion_window: 100,
                slow_start_threshold: 50,
                timeout_ms: 1000,
                retransmission_timeout_ms: 3000,
            })),
            backpressure_queue: Arc::new(Queue::new()),
            algorithms: vec![], // Initialize with actual algorithms
        });
        
        let route_optimizer = Arc::new(RouteOptimizer {
            algorithms: vec![], // Initialize with actual algorithms
            route_scores: Arc::new(DashMap::new()),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            pathfinding_cache: Arc::new(DashMap::new()),
        });
        
        let priority_scheduler = Arc::new(PriorityScheduler {
            priority_queues: Arc::new(DashMap::new()),
            policies: Arc::new(RwLock::new(SchedulingPolicies {
                critical_queue_weight: 1.0,
                high_queue_weight: 0.8,
                normal_queue_weight: 0.5,
                low_queue_weight: 0.2,
                deadline_enforcement: true,
                starvation_prevention: true,
            })),
            deadlines: Arc::new(DashMap::new()),
            metrics: Arc::new(SchedulerMetrics {
                critical_messages_scheduled: AtomicU64::new(0),
                high_messages_scheduled: AtomicU64::new(0),
                normal_messages_scheduled: AtomicU64::new(0),
                low_messages_scheduled: AtomicU64::new(0),
                deadline_violations: AtomicU64::new(0),
                average_scheduling_delay_ns: AtomicU64::new(0),
            }),
        });
        
        let compression_engine = Arc::new(CompressionEngine {
            algorithms: Self::create_compression_algorithms(),
            compression_ratios: Arc::new(DashMap::new()),
            compression_cache: Arc::new(DashMap::new()),
            algorithm_selector: Arc::new(CompressionSelector {
                selection_strategy: CompressionStrategy::Adaptive,
                performance_cache: Arc::new(DashMap::new()),
            }),
        });
        
        let metrics_collector = Arc::new(RouterMetricsCollector {
            message_metrics: Arc::new(DashMap::new()),
            route_metrics: Arc::new(DashMap::new()),
            global_metrics: Arc::new(RwLock::new(GlobalRouterMetrics {
                total_messages_routed: 0,
                messages_per_second: 0.0,
                average_latency_ns: 0,
                p99_latency_ns: 0,
                route_optimization_count: 0,
                congestion_events: 0,
                compression_savings_bytes: 0,
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
            })),
            collection_interval: Duration::from_millis(100),
        });
        
        Ok(Self {
            config,
            routing_table,
            message_pools,
            worker_pool,
            latency_tracker,
            congestion_controller,
            route_optimizer,
            priority_scheduler,
            compression_engine,
            metrics_collector,
            shutdown_signal: Arc::new(Mutex::new(None)),
        })
    }
    
    /// Initialize memory pools for zero-copy operations
    async fn initialize_memory_pools(
        pools: &MessagePools,
        config: &MessageRouterConfig,
    ) -> Result<(), MCPOrchestrationError> {
        // Pre-allocate message objects
        for _ in 0..config.memory_pool_size {
            let message = Box::new(MCPMessage {
                id: String::new(),
                source: String::new(),
                target: String::new(),
                message_type: MCPMessageType::Command,
                payload: serde_json::Value::Null,
                timestamp: Utc::now(),
                priority: MessagePriority::Normal,
                routing_info: RoutingInfo {
                    route_id: String::new(),
                    hop_count: 0,
                    max_hops: 8,
                    latency_target_us: config.target_latency_ns / 1000,
                    compression_enabled: false,
                },
            });
            pools.message_pool.push(message);
        }
        
        // Pre-allocate batch objects
        for _ in 0..config.memory_pool_size / 10 {
            let batch = Box::new(MessageBatch {
                messages: SmallVec::new(),
                batch_id: String::new(),
                priority: MessagePriority::Normal,
                created_at: Instant::now(),
                target_latency_ns: config.target_latency_ns,
            });
            pools.batch_pool.push(batch);
        }
        
        // Pre-allocate buffers
        for _ in 0..config.memory_pool_size {
            let buffer = Vec::with_capacity(1024);
            pools.buffer_pool.push(buffer);
        }
        
        // Pre-allocate route caches
        for _ in 0..config.memory_pool_size {
            let route_cache = SmallVec::new();
            pools.route_cache_pool.push(route_cache);
        }
        
        Ok(())
    }
    
    /// Initialize routing with hierarchical topology
    pub async fn initialize_routes(&self) -> Result<(), MCPOrchestrationError> {
        info!("Initializing ultra-low latency routing table");
        
        // Clear existing routes
        self.routing_table.direct_routes.clear();
        self.routing_table.hierarchy_cache.clear();
        self.routing_table.swarm_routes.clear();
        
        // Initialize hierarchical routes
        self.initialize_hierarchical_routes().await?;
        
        // Initialize swarm routes
        self.initialize_swarm_routes().await?;
        
        // Initialize priority routes
        self.initialize_priority_routes().await?;
        
        // Start route optimization
        self.start_route_optimization().await?;
        
        info!("Routing table initialized with ultra-low latency optimization");
        Ok(())
    }
    
    /// Initialize hierarchical routes
    async fn initialize_hierarchical_routes(&self) -> Result<(), MCPOrchestrationError> {
        // Define hierarchy levels and their interconnections
        let hierarchy_map = vec![
            ("orchestrator", HierarchyLevel::Orchestrator, vec![]),
            ("coordinator_0", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
            ("coordinator_1", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
            ("coordinator_2", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
            ("coordinator_3", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
            ("coordinator_4", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
            ("coordinator_5", HierarchyLevel::SwarmCoordinator, vec!["orchestrator"]),
        ];
        
        // Create direct routes between hierarchy levels
        for (node, level, parents) in hierarchy_map {
            let routes = self.routing_table.direct_routes.entry(node.to_string()).or_insert_with(DashMap::new);
            
            // Create routes to parents
            for parent in parents {
                let route_entry = RouteEntry {
                    source: node.to_string(),
                    target: parent.to_string(),
                    path: SmallVec::from_slice(&[node.to_string(), parent.to_string()]),
                    latency_ns: 100, // 100ns for direct hierarchical connection
                    bandwidth_mbps: 10000,
                    congestion_level: 0.0,
                    reliability_score: 0.99,
                    last_updated: Utc::now(),
                    use_count: AtomicU64::new(0),
                };
                
                routes.insert(parent.to_string(), route_entry);
                
                // Update hierarchy cache
                self.routing_table.hierarchy_cache.insert(
                    format!("{}_{}", node, parent),
                    SmallVec::from_slice(&[node.to_string(), parent.to_string()]),
                );
            }
        }
        
        Ok(())
    }
    
    /// Initialize swarm-specific routes
    async fn initialize_swarm_routes(&self) -> Result<(), MCPOrchestrationError> {
        let swarm_configs = vec![
            (SwarmType::RiskManagement, vec!["coordinator_0", "risk_agent_0", "risk_agent_1", "risk_agent_2", "risk_agent_3", "risk_agent_4"]),
            (SwarmType::TradingStrategy, vec!["coordinator_1", "trading_agent_0", "trading_agent_1", "trading_agent_2", "trading_agent_3", "trading_agent_4", "trading_agent_5"]),
            (SwarmType::DataPipeline, vec!["coordinator_2", "data_agent_0", "data_agent_1", "data_agent_2", "data_agent_3", "data_agent_4", "data_agent_5"]),
            (SwarmType::TENGRIWatchdog, vec!["coordinator_3", "tengri_agent_0", "tengri_agent_1", "tengri_agent_2", "tengri_agent_3", "tengri_agent_4", "tengri_agent_5", "tengri_agent_6", "tengri_agent_7"]),
            (SwarmType::QuantumML, vec!["coordinator_4"]),
            (SwarmType::MCPOrchestration, vec!["coordinator_5", "service_0", "service_1", "service_2", "service_3"]),
        ];
        
        for (swarm_type, agents) in swarm_configs {
            self.routing_table.swarm_routes.insert(swarm_type, agents.iter().map(|s| s.to_string()).collect());
            
            // Create optimized intra-swarm routes
            for (i, agent1) in agents.iter().enumerate() {
                let routes = self.routing_table.direct_routes.entry(agent1.to_string()).or_insert_with(DashMap::new);
                
                for (j, agent2) in agents.iter().enumerate() {
                    if i != j {
                        let latency_ns = if i == 0 || j == 0 { 50 } else { 200 }; // Coordinator has lower latency
                        
                        let route_entry = RouteEntry {
                            source: agent1.to_string(),
                            target: agent2.to_string(),
                            path: SmallVec::from_slice(&[agent1.to_string(), agent2.to_string()]),
                            latency_ns,
                            bandwidth_mbps: 5000,
                            congestion_level: 0.0,
                            reliability_score: 0.98,
                            last_updated: Utc::now(),
                            use_count: AtomicU64::new(0),
                        };
                        
                        routes.insert(agent2.to_string(), route_entry);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Initialize priority routes for critical messages
    async fn initialize_priority_routes(&self) -> Result<(), MCPOrchestrationError> {
        // Create ultra-fast routes for critical messages
        let critical_routes = vec![
            ("orchestrator", "coordinator_0", 50), // Risk management priority
            ("orchestrator", "coordinator_3", 50), // TENGRI watchdog priority
            ("coordinator_0", "risk_agent_0", 25), // Primary risk agent
            ("coordinator_3", "tengri_agent_0", 25), // Primary watchdog agent
        ];
        
        for (source, target, latency_ns) in critical_routes {
            let route_entry = RouteEntry {
                source: source.to_string(),
                target: target.to_string(),
                path: SmallVec::from_slice(&[source.to_string(), target.to_string()]),
                latency_ns,
                bandwidth_mbps: 20000,
                congestion_level: 0.0,
                reliability_score: 0.999,
                last_updated: Utc::now(),
                use_count: AtomicU64::new(0),
            };
            
            self.routing_table.priority_routes.insert(
                format!("{}_{}", source, target),
                route_entry,
            );
        }
        
        Ok(())
    }
    
    /// Start route optimization background task
    async fn start_route_optimization(&self) -> Result<(), MCPOrchestrationError> {
        let router_clone = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(router_clone.config.optimization_interval_ms));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = router_clone.optimize_routes().await {
                    error!("Route optimization error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Route message with ultra-low latency
    #[instrument(skip(self, message), fields(message_id = %message.id, source = %message.source, target = %message.target))]
    pub async fn route_message(&self, message: MCPMessage) -> Result<(), MCPOrchestrationError> {
        let start_time = Instant::now();
        
        // Fast path for priority messages
        if matches!(message.priority, MessagePriority::Critical) {
            return self.route_priority_message(message).await;
        }
        
        // Check if message should be compressed
        let message = if self.should_compress_message(&message) {
            self.compress_message(message).await?
        } else {
            message
        };
        
        // Find optimal route
        let route = self.find_optimal_route(&message.source, &message.target).await?;
        
        // Route through the path
        self.route_through_path(message, route).await?;
        
        // Record latency
        let latency_ns = start_time.elapsed().as_nanos() as u64;
        self.record_latency(&format!("{}_{}", message.source, message.target), latency_ns).await;
        
        // Check SLA compliance
        if latency_ns > self.config.target_latency_ns {
            self.latency_tracker.sla_violations.fetch_add(1, Ordering::SeqCst);
            warn!("SLA violation: {}ns > {}ns", latency_ns, self.config.target_latency_ns);
        }
        
        counter!("messages_routed", 1);
        histogram!("message_routing_latency_ns", latency_ns as f64);
        
        Ok(())
    }
    
    /// Route priority message through dedicated fast path
    async fn route_priority_message(&self, message: MCPMessage) -> Result<(), MCPOrchestrationError> {
        let start_time = Instant::now();
        
        // Look for dedicated priority route
        let priority_key = format!("{}_{}", message.source, message.target);
        if let Some(route) = self.routing_table.priority_routes.get(&priority_key) {
            // Use priority route
            self.route_through_path(message, route.path.clone()).await?;
        } else {
            // Use fastest available route
            let route = self.find_fastest_route(&message.source, &message.target).await?;
            self.route_through_path(message, route).await?;
        }
        
        let latency_ns = start_time.elapsed().as_nanos() as u64;
        histogram!("priority_message_latency_ns", latency_ns as f64);
        
        Ok(())
    }
    
    /// Find optimal route between source and target
    async fn find_optimal_route(&self, source: &str, target: &str) -> Result<SmallVec<[String; 8]>, MCPOrchestrationError> {
        // Check cache first
        let cache_key = format!("{}_{}", source, target);
        if let Some(cached_route) = self.routing_table.hierarchy_cache.get(&cache_key) {
            return Ok(cached_route.clone());
        }
        
        // Check direct route
        if let Some(source_routes) = self.routing_table.direct_routes.get(source) {
            if let Some(route_entry) = source_routes.get(target) {
                let route = route_entry.path.clone();
                self.routing_table.hierarchy_cache.insert(cache_key, route.clone());
                return Ok(route);
            }
        }
        
        // Use pathfinding algorithm
        let route = self.calculate_shortest_path(source, target).await?;
        self.routing_table.hierarchy_cache.insert(cache_key, route.clone());
        
        Ok(route)
    }
    
    /// Find fastest route (for priority messages)
    async fn find_fastest_route(&self, source: &str, target: &str) -> Result<SmallVec<[String; 8]>, MCPOrchestrationError> {
        let mut best_route = SmallVec::new();
        let mut best_latency = u64::MAX;
        
        // Check all possible routes and select the fastest
        if let Some(source_routes) = self.routing_table.direct_routes.get(source) {
            for route_entry in source_routes.iter() {
                if route_entry.key() == target && route_entry.value().latency_ns < best_latency {
                    best_latency = route_entry.value().latency_ns;
                    best_route = route_entry.value().path.clone();
                }
            }
        }
        
        if best_route.is_empty() {
            // Fallback to pathfinding
            best_route = self.calculate_shortest_path(source, target).await?;
        }
        
        Ok(best_route)
    }
    
    /// Calculate shortest path using A* algorithm
    async fn calculate_shortest_path(&self, source: &str, target: &str) -> Result<SmallVec<[String; 8]>, MCPOrchestrationError> {
        // Simplified A* implementation
        // In a real implementation, this would use a proper graph library
        
        // For now, use hierarchical routing logic
        let path = if source.starts_with("orchestrator") {
            // Route from orchestrator to coordinators or agents
            if target.starts_with("coordinator") {
                SmallVec::from_slice(&[source.to_string(), target.to_string()])
            } else {
                // Route through appropriate coordinator
                let coordinator = self.get_coordinator_for_agent(target);
                SmallVec::from_slice(&[source.to_string(), coordinator, target.to_string()])
            }
        } else if source.starts_with("coordinator") {
            // Route from coordinator
            if target.starts_with("orchestrator") {
                SmallVec::from_slice(&[source.to_string(), target.to_string()])
            } else if target.starts_with("coordinator") {
                // Route through orchestrator
                SmallVec::from_slice(&[source.to_string(), "orchestrator".to_string(), target.to_string()])
            } else {
                // Direct to agent in same swarm or through orchestrator
                SmallVec::from_slice(&[source.to_string(), target.to_string()])
            }
        } else {
            // Route from agent
            let source_coordinator = self.get_coordinator_for_agent(source);
            if target.starts_with("coordinator") || target.starts_with("orchestrator") {
                SmallVec::from_slice(&[source.to_string(), source_coordinator, target.to_string()])
            } else {
                let target_coordinator = self.get_coordinator_for_agent(target);
                if source_coordinator == target_coordinator {
                    // Same swarm
                    SmallVec::from_slice(&[source.to_string(), target.to_string()])
                } else {
                    // Different swarms - route through orchestrator
                    SmallVec::from_slice(&[
                        source.to_string(),
                        source_coordinator,
                        "orchestrator".to_string(),
                        target_coordinator,
                        target.to_string(),
                    ])
                }
            }
        };
        
        Ok(path)
    }
    
    /// Get coordinator for a given agent
    fn get_coordinator_for_agent(&self, agent: &str) -> String {
        match agent {
            s if s.starts_with("risk_") => "coordinator_0".to_string(),
            s if s.starts_with("trading_") => "coordinator_1".to_string(),
            s if s.starts_with("data_") => "coordinator_2".to_string(),
            s if s.starts_with("tengri_") => "coordinator_3".to_string(),
            s if s.starts_with("quantum_") => "coordinator_4".to_string(),
            s if s.starts_with("service_") => "coordinator_5".to_string(),
            _ => "coordinator_0".to_string(), // Default fallback
        }
    }
    
    /// Route message through specified path
    async fn route_through_path(&self, message: MCPMessage, path: SmallVec<[String; 8]>) -> Result<(), MCPOrchestrationError> {
        if path.len() < 2 {
            return Err(MCPOrchestrationError::MessageRoutingError {
                reason: "Invalid path: must have at least source and target".to_string(),
            });
        }
        
        // For each hop in the path
        for i in 0..path.len() - 1 {
            let current_hop = &path[i];
            let next_hop = &path[i + 1];
            
            // Record hop latency
            let hop_start = Instant::now();
            
            // Simulate message forwarding (in real implementation, this would be actual network communication)
            self.forward_message_to_hop(&message, current_hop, next_hop).await?;
            
            let hop_latency = hop_start.elapsed().as_nanos() as u64;
            
            // Update route metrics
            let route_key = format!("{}_{}", current_hop, next_hop);
            self.update_route_metrics(&route_key, hop_latency, true).await;
        }
        
        Ok(())
    }
    
    /// Forward message to next hop
    async fn forward_message_to_hop(&self, _message: &MCPMessage, _current: &str, _next: &str) -> Result<(), MCPOrchestrationError> {
        // Simulate ultra-fast message forwarding
        // In a real implementation, this would use:
        // - Zero-copy networking
        // - RDMA for lowest latency
        // - Custom protocols
        // - Hardware acceleration
        
        // For now, simulate minimal processing time
        tokio::time::sleep(Duration::from_nanos(10)).await;
        
        Ok(())
    }
    
    /// Check if message should be compressed
    fn should_compress_message(&self, message: &MCPMessage) -> bool {
        if !self.config.zero_copy_enabled {
            return false;
        }
        
        // Don't compress critical messages to save latency
        if matches!(message.priority, MessagePriority::Critical) {
            return false;
        }
        
        // Compress large messages
        let estimated_size = message.id.len() + message.source.len() + message.target.len() + 100; // Rough estimate
        estimated_size > self.config.compression_threshold
    }
    
    /// Compress message
    async fn compress_message(&self, message: MCPMessage) -> Result<MCPMessage, MCPOrchestrationError> {
        // Implementation would compress the message payload
        // For now, just mark it as compressed
        let mut compressed_message = message;
        compressed_message.routing_info.compression_enabled = true;
        Ok(compressed_message)
    }
    
    /// Record latency measurement
    async fn record_latency(&self, route_id: &str, latency_ns: u64) {
        // Update route-specific latency
        let histogram = self.latency_tracker.route_latencies
            .entry(route_id.to_string())
            .or_insert_with(|| LatencyHistogram {
                buckets: vec![AtomicU64::new(0); 20], // 20 buckets
                bucket_boundaries: vec![
                    100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
                    200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, u64::MAX
                ],
                total_samples: AtomicU64::new(0),
                sum_ns: AtomicU64::new(0),
                min_ns: AtomicU64::new(u64::MAX),
                max_ns: AtomicU64::new(0),
            });
        
        // Find appropriate bucket
        let bucket_index = histogram.bucket_boundaries
            .iter()
            .position(|&boundary| latency_ns <= boundary)
            .unwrap_or(histogram.buckets.len() - 1);
        
        histogram.buckets[bucket_index].fetch_add(1, Ordering::SeqCst);
        histogram.total_samples.fetch_add(1, Ordering::SeqCst);
        histogram.sum_ns.fetch_add(latency_ns, Ordering::SeqCst);
        
        // Update min/max
        let current_min = histogram.min_ns.load(Ordering::SeqCst);
        if latency_ns < current_min {
            histogram.min_ns.store(latency_ns, Ordering::SeqCst);
        }
        
        let current_max = histogram.max_ns.load(Ordering::SeqCst);
        if latency_ns > current_max {
            histogram.max_ns.store(latency_ns, Ordering::SeqCst);
        }
        
        // Update global latency statistics
        self.update_global_latency_stats().await;
    }
    
    /// Update global latency statistics
    async fn update_global_latency_stats(&self) {
        // Calculate global statistics from all route histograms
        let mut total_samples = 0u64;
        let mut total_sum = 0u64;
        let mut global_min = u64::MAX;
        let mut global_max = 0u64;
        
        for histogram in self.latency_tracker.route_latencies.iter() {
            let samples = histogram.total_samples.load(Ordering::SeqCst);
            let sum = histogram.sum_ns.load(Ordering::SeqCst);
            let min = histogram.min_ns.load(Ordering::SeqCst);
            let max = histogram.max_ns.load(Ordering::SeqCst);
            
            total_samples += samples;
            total_sum += sum;
            if min < global_min {
                global_min = min;
            }
            if max > global_max {
                global_max = max;
            }
        }
        
        if total_samples > 0 {
            let mut global_stats = self.latency_tracker.global_latency.write().await;
            global_stats.mean_ns = total_sum / total_samples;
            global_stats.min_ns = global_min;
            global_stats.max_ns = global_max;
            // Note: Median, p95, p99, p999, and stddev would require more complex calculations
        }
    }
    
    /// Update route metrics
    async fn update_route_metrics(&self, route_id: &str, latency_ns: u64, success: bool) {
        let metrics = self.routing_table.route_metrics
            .entry(route_id.to_string())
            .or_insert_with(|| RouteMetrics {
                total_messages: 0,
                successful_messages: 0,
                failed_messages: 0,
                average_latency_ns: 0,
                min_latency_ns: u64::MAX,
                max_latency_ns: 0,
                p95_latency_ns: 0,
                p99_latency_ns: 0,
                throughput_msg_per_sec: 0.0,
                bandwidth_utilization: 0.0,
                error_rate: 0.0,
            });
        
        metrics.total_messages += 1;
        if success {
            metrics.successful_messages += 1;
        } else {
            metrics.failed_messages += 1;
        }
        
        // Update latency statistics
        if latency_ns < metrics.min_latency_ns {
            metrics.min_latency_ns = latency_ns;
        }
        if latency_ns > metrics.max_latency_ns {
            metrics.max_latency_ns = latency_ns;
        }
        
        // Update average latency (simple moving average)
        metrics.average_latency_ns = (metrics.average_latency_ns * (metrics.total_messages - 1) + latency_ns) / metrics.total_messages;
        
        // Update error rate
        metrics.error_rate = metrics.failed_messages as f64 / metrics.total_messages as f64;
    }
    
    /// Optimize routes based on performance metrics
    async fn optimize_routes(&self) -> Result<(), MCPOrchestrationError> {
        debug!("Starting route optimization");
        
        let start_time = Instant::now();
        
        // Analyze current route performance
        let optimization_candidates = self.identify_optimization_candidates().await;
        
        // Apply optimizations
        for candidate in optimization_candidates {
            self.apply_route_optimization(candidate).await?;
        }
        
        // Update optimization timestamp
        self.routing_table.last_optimization.store(
            start_time.elapsed().as_nanos() as u64,
            Ordering::SeqCst,
        );
        
        debug!("Route optimization completed in {:?}", start_time.elapsed());
        Ok(())
    }
    
    /// Identify routes that could benefit from optimization
    async fn identify_optimization_candidates(&self) -> Vec<RouteOptimizationCandidate> {
        let mut candidates = Vec::new();
        
        for route_entry in self.routing_table.route_metrics.iter() {
            let route_id = route_entry.key();
            let metrics = route_entry.value();
            
            // Check if route exceeds latency target
            if metrics.average_latency_ns > self.config.target_latency_ns {
                candidates.push(RouteOptimizationCandidate {
                    route_id: route_id.clone(),
                    optimization_type: OptimizationType::ReduceLatency,
                    current_performance: metrics.average_latency_ns as f64,
                    target_performance: self.config.target_latency_ns as f64,
                    priority: if metrics.average_latency_ns > self.config.target_latency_ns * 2 {
                        OptimizationPriority::High
                    } else {
                        OptimizationPriority::Medium
                    },
                });
            }
            
            // Check for high error rate
            if metrics.error_rate > 0.01 { // 1% error rate threshold
                candidates.push(RouteOptimizationCandidate {
                    route_id: route_id.clone(),
                    optimization_type: OptimizationType::ImproveReliability,
                    current_performance: metrics.error_rate,
                    target_performance: 0.001, // 0.1% target
                    priority: OptimizationPriority::High,
                });
            }
        }
        
        // Sort by priority
        candidates.sort_by(|a, b| {
            match (a.priority.clone(), b.priority.clone()) {
                (OptimizationPriority::High, OptimizationPriority::High) => 
                    a.current_performance.partial_cmp(&b.current_performance).unwrap_or(std::cmp::Ordering::Equal),
                (OptimizationPriority::High, _) => std::cmp::Ordering::Less,
                (_, OptimizationPriority::High) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            }
        });
        
        candidates
    }
    
    /// Apply route optimization
    async fn apply_route_optimization(&self, candidate: RouteOptimizationCandidate) -> Result<(), MCPOrchestrationError> {
        match candidate.optimization_type {
            OptimizationType::ReduceLatency => {
                self.optimize_route_latency(&candidate.route_id).await?;
            }
            OptimizationType::ImproveReliability => {
                self.optimize_route_reliability(&candidate.route_id).await?;
            }
            OptimizationType::IncreaseCapacity => {
                self.optimize_route_capacity(&candidate.route_id).await?;
            }
        }
        
        Ok(())
    }
    
    /// Optimize route latency
    async fn optimize_route_latency(&self, route_id: &str) -> Result<(), MCPOrchestrationError> {
        // Parse route ID to get source and target
        let parts: Vec<&str> = route_id.split('_').collect();
        if parts.len() != 2 {
            return Ok(());
        }
        
        let source = parts[0];
        let target = parts[1];
        
        // Try to find a more direct route
        let alternative_route = self.find_alternative_route(source, target).await?;
        
        // Update routing table if alternative is better
        if let Some(alt_route) = alternative_route {
            if let Some(source_routes) = self.routing_table.direct_routes.get(source) {
                if let Some(mut current_route) = source_routes.get_mut(target) {
                    if alt_route.latency_ns < current_route.latency_ns {
                        current_route.path = alt_route.path;
                        current_route.latency_ns = alt_route.latency_ns;
                        current_route.last_updated = Utc::now();
                        
                        info!("Optimized route {} -> {}: {}ns -> {}ns", 
                              source, target, current_route.latency_ns, alt_route.latency_ns);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Find alternative route with potentially better performance
    async fn find_alternative_route(&self, source: &str, target: &str) -> Result<Option<RouteEntry>, MCPOrchestrationError> {
        // Check if there are intermediate hops that could be bypassed
        let current_route = self.find_optimal_route(source, target).await?;
        
        if current_route.len() > 2 {
            // Try direct connection
            let direct_route = RouteEntry {
                source: source.to_string(),
                target: target.to_string(),
                path: SmallVec::from_slice(&[source.to_string(), target.to_string()]),
                latency_ns: 300, // Assume direct connection has higher base latency but fewer hops
                bandwidth_mbps: 1000,
                congestion_level: 0.0,
                reliability_score: 0.95,
                last_updated: Utc::now(),
                use_count: AtomicU64::new(0),
            };
            
            return Ok(Some(direct_route));
        }
        
        Ok(None)
    }
    
    /// Optimize route reliability
    async fn optimize_route_reliability(&self, route_id: &str) -> Result<(), MCPOrchestrationError> {
        // Implementation would add redundancy or failover routes
        debug!("Optimizing reliability for route: {}", route_id);
        Ok(())
    }
    
    /// Optimize route capacity
    async fn optimize_route_capacity(&self, route_id: &str) -> Result<(), MCPOrchestrationError> {
        // Implementation would increase bandwidth or add parallel paths
        debug!("Optimizing capacity for route: {}", route_id);
        Ok(())
    }
    
    /// Get routing statistics
    pub async fn get_routing_statistics(&self) -> RoutingStatistics {
        let global_stats = self.latency_tracker.global_latency.read().await;
        let sla_violations = self.latency_tracker.sla_violations.load(Ordering::SeqCst);
        
        let mut total_routes = 0;
        let mut active_routes = 0;
        let mut total_messages = 0;
        
        for route_metrics in self.routing_table.route_metrics.iter() {
            total_routes += 1;
            total_messages += route_metrics.total_messages;
            if route_metrics.total_messages > 0 {
                active_routes += 1;
            }
        }
        
        RoutingStatistics {
            total_routes,
            active_routes,
            total_messages,
            average_latency: Duration::from_nanos(global_stats.mean_ns),
            messages_per_second: total_messages as f64 / 60.0, // Rough estimate
            sla_violations,
            optimization_count: 0, // Would track actual optimizations
        }
    }
    
    /// Create compression algorithms
    fn create_compression_algorithms() -> HashMap<String, Box<dyn CompressionAlgorithm>> {
        // Implementation would create actual compression algorithms
        HashMap::new()
    }
}

impl Clone for MessageRouter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            routing_table: Arc::clone(&self.routing_table),
            message_pools: Arc::clone(&self.message_pools),
            worker_pool: Arc::clone(&self.worker_pool),
            latency_tracker: Arc::clone(&self.latency_tracker),
            congestion_controller: Arc::clone(&self.congestion_controller),
            route_optimizer: Arc::clone(&self.route_optimizer),
            priority_scheduler: Arc::clone(&self.priority_scheduler),
            compression_engine: Arc::clone(&self.compression_engine),
            metrics_collector: Arc::clone(&self.metrics_collector),
            shutdown_signal: Arc::clone(&self.shutdown_signal),
        }
    }
}

/// Route optimization candidate
#[derive(Debug, Clone)]
pub struct RouteOptimizationCandidate {
    pub route_id: String,
    pub optimization_type: OptimizationType,
    pub current_performance: f64,
    pub target_performance: f64,
    pub priority: OptimizationPriority,
}

/// Optimization types
#[derive(Debug, Clone)]
pub enum OptimizationType {
    ReduceLatency,
    ImproveReliability,
    IncreaseCapacity,
}

/// Optimization priorities
#[derive(Debug, Clone)]
pub enum OptimizationPriority {
    High,
    Medium,
    Low,
}

/// Routing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStatistics {
    pub total_routes: usize,
    pub active_routes: usize,
    pub total_messages: u64,
    pub average_latency: Duration,
    pub messages_per_second: f64,
    pub sla_violations: u64,
    pub optimization_count: u64,
}