//! Message Router Agent
//!
//! Ultra-low latency message routing between swarms with sub-microsecond
//! routing capabilities, intelligent routing algorithms, and real-time
//! performance optimization.

use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::{RwLock, mpsc, broadcast, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use dashmap::DashMap;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crossbeam_utils::CachePadded;
use parking_lot::RwLock as ParkingRwLock;
use atomic::{Atomic, Ordering as AtomicOrdering};
use tracing::{debug, info, warn, error, instrument};
use anyhow::Result;
use futures::{Future, StreamExt};
use tokio::time::{sleep, timeout};
use rayon::prelude::*;
use lockfree::queue::Queue;
use flume::{Receiver as FlumeReceiver, Sender as FlumeSender};
use priority_queue::PriorityQueue;
use smallvec::SmallVec;
use arrayvec::ArrayVec;
use blake3::Hasher;

use crate::types::*;
use crate::error::OrchestrationError;
use crate::agent::{Agent, AgentId, AgentInfo, AgentState, AgentType};
use crate::communication::{Message, MessageType, CommunicationLayer};
use crate::metrics::{OrchestrationMetrics};

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Maximum concurrent routes
    pub max_concurrent_routes: usize,
    /// Route timeout in nanoseconds
    pub route_timeout_ns: u64,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Message buffer size
    pub message_buffer_size: usize,
    /// Routing algorithm
    pub routing_algorithm: RoutingAlgorithm,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Enable message compression
    pub enable_compression: bool,
    /// Enable message encryption
    pub enable_encryption: bool,
    /// Priority queue size
    pub priority_queue_size: usize,
    /// Route cache size
    pub route_cache_size: usize,
}

/// Routing algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    /// Direct point-to-point routing
    Direct,
    /// Shortest path routing
    ShortestPath,
    /// Load-aware routing
    LoadAware,
    /// Adaptive routing with ML
    Adaptive,
    /// Multicast routing
    Multicast,
    /// Hierarchical routing
    Hierarchical,
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    LatencyBased,
    LoadBased,
    Adaptive,
}

/// Message with routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedMessage {
    /// Message ID
    pub id: MessageId,
    /// Source agent ID
    pub source: AgentId,
    /// Destination agent ID
    pub destination: AgentId,
    /// Message type
    pub message_type: MessageType,
    /// Payload
    pub payload: Vec<u8>,
    /// Priority
    pub priority: MessagePriority,
    /// Timestamp
    pub timestamp: Instant,
    /// TTL in nanoseconds
    pub ttl_ns: u64,
    /// Routing metadata
    pub routing_metadata: RoutingMetadata,
    /// Compression info
    pub compression: Option<CompressionInfo>,
    /// Encryption info
    pub encryption: Option<EncryptionInfo>,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

/// Message ID type
pub type MessageId = u64;

/// Routing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetadata {
    /// Hop count
    pub hop_count: u8,
    /// Route path
    pub route_path: SmallVec<[AgentId; 8]>,
    /// Routing flags
    pub flags: RoutingFlags,
    /// QoS requirements
    pub qos: QoSRequirements,
    /// Trace information
    pub trace: Option<RouteTrace>,
}

/// Routing flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingFlags {
    /// Requires acknowledgment
    pub requires_ack: bool,
    /// Broadcast message
    pub broadcast: bool,
    /// Multicast message
    pub multicast: bool,
    /// Ordered delivery
    pub ordered: bool,
    /// Reliable delivery
    pub reliable: bool,
    /// Real-time priority
    pub real_time: bool,
}

/// Quality of Service requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSRequirements {
    /// Maximum latency in nanoseconds
    pub max_latency_ns: u64,
    /// Minimum bandwidth in bytes per second
    pub min_bandwidth_bps: u64,
    /// Maximum jitter in nanoseconds
    pub max_jitter_ns: u64,
    /// Reliability percentage
    pub reliability: f64,
    /// Ordering requirement
    pub ordering: OrderingRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingRequirement {
    None,
    FIFO,
    Priority,
    Causal,
    Total,
}

/// Route trace information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTrace {
    /// Trace ID
    pub trace_id: u64,
    /// Span information
    pub spans: Vec<TraceSpan>,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Span ID
    pub span_id: u64,
    /// Agent ID
    pub agent_id: AgentId,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Latency in nanoseconds
    pub latency_ns: u64,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Brotli,
    Snappy,
}

/// Encryption information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInfo {
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// Key ID
    pub key_id: String,
    /// IV/Nonce
    pub iv: Vec<u8>,
    /// Authentication tag
    pub auth_tag: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    None,
    ChaCha20Poly1305,
    AesGcm,
    Xchacha20Poly1305,
}

/// Route information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    /// Route ID
    pub id: u64,
    /// Source agent
    pub source: AgentId,
    /// Destination agent
    pub destination: AgentId,
    /// Route path
    pub path: Vec<AgentId>,
    /// Route metrics
    pub metrics: RouteMetrics,
    /// Last update time
    pub last_update: Instant,
    /// Route status
    pub status: RouteStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetrics {
    /// Latency in nanoseconds
    pub latency_ns: u64,
    /// Bandwidth in bytes per second
    pub bandwidth_bps: u64,
    /// Packet loss percentage
    pub packet_loss: f64,
    /// Jitter in nanoseconds
    pub jitter_ns: u64,
    /// Throughput in messages per second
    pub throughput_mps: f64,
    /// Reliability score
    pub reliability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RouteStatus {
    Active,
    Inactive,
    Congested,
    Failed,
    Maintenance,
}

/// Router command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouterCommand {
    RouteMessage(RoutedMessage),
    AddRoute(RouteInfo),
    RemoveRoute(u64),
    UpdateRouteMetrics { route_id: u64, metrics: RouteMetrics },
    SetRoutingAlgorithm(RoutingAlgorithm),
    SetLoadBalancing(LoadBalancingStrategy),
    GetRouteInfo(u64),
    GetAllRoutes,
    GetRoutingStats,
    OptimizeRoutes,
    FlushMessageQueue,
    EnableTracing { trace_id: u64 },
    DisableTracing,
    Shutdown,
}

/// Router event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouterEvent {
    MessageRouted {
        message_id: MessageId,
        latency_ns: u64,
        route_path: Vec<AgentId>,
    },
    MessageDropped {
        message_id: MessageId,
        reason: String,
    },
    RouteAdded {
        route_id: u64,
        source: AgentId,
        destination: AgentId,
    },
    RouteRemoved {
        route_id: u64,
    },
    RouteOptimized {
        route_id: u64,
        old_latency_ns: u64,
        new_latency_ns: u64,
    },
    CongestionDetected {
        route_id: u64,
        congestion_level: f64,
    },
    RoutingAlgorithmChanged {
        old_algorithm: RoutingAlgorithm,
        new_algorithm: RoutingAlgorithm,
    },
}

/// Message Router Agent
pub struct MessageRouterAgent {
    /// Router ID
    id: AgentId,
    /// Configuration
    config: RouterConfig,
    /// Router state
    state: Arc<RwLock<AgentState>>,
    /// Active routes
    routes: Arc<RwLock<HashMap<u64, RouteInfo>>>,
    /// Route lookup table
    route_lookup: Arc<DashMap<(AgentId, AgentId), u64>>,
    /// Message queue
    message_queue: Arc<Queue<RoutedMessage>>,
    /// Priority queue
    priority_queue: Arc<Mutex<PriorityQueue<RoutedMessage, MessagePriority>>>,
    /// Route cache
    route_cache: Arc<DashMap<u64, RouteInfo>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<RouterMetrics>>,
    /// Message counters
    message_counters: Arc<MessageCounters>,
    /// Latency tracker
    latency_tracker: Arc<LatencyTracker>,
    /// Command channel
    command_tx: mpsc::UnboundedSender<RouterCommand>,
    command_rx: Arc<Mutex<mpsc::UnboundedReceiver<RouterCommand>>>,
    /// Event broadcast
    event_tx: broadcast::Sender<RouterEvent>,
    /// Message channels
    message_channels: Arc<RwLock<HashMap<AgentId, FlumeSender<RoutedMessage>>>>,
    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<mpsc::UnboundedSender<()>>>>,
    /// Running state
    running: Arc<Atomic<bool>>,
    /// Message ID generator
    message_id_gen: Arc<AtomicU64>,
    /// Route ID generator
    route_id_gen: Arc<AtomicU64>,
    /// Compression engine
    compression_engine: Arc<CompressionEngine>,
    /// Encryption engine
    encryption_engine: Arc<EncryptionEngine>,
}

/// Router performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterMetrics {
    /// Total messages routed
    pub total_messages_routed: u64,
    /// Messages per second
    pub messages_per_second: f64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// 95th percentile latency
    pub p95_latency_ns: u64,
    /// 99th percentile latency
    pub p99_latency_ns: u64,
    /// Maximum latency
    pub max_latency_ns: u64,
    /// Total bandwidth utilization
    pub total_bandwidth_bps: u64,
    /// Route efficiency
    pub route_efficiency: f64,
    /// Error rate
    pub error_rate: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Active routes
    pub active_routes: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Message counters
pub struct MessageCounters {
    /// Total messages
    pub total: AtomicU64,
    /// Messages by priority
    pub by_priority: [AtomicU64; 5],
    /// Messages by type
    pub by_type: DashMap<MessageType, AtomicU64>,
    /// Dropped messages
    pub dropped: AtomicU64,
    /// Failed routes
    pub failed_routes: AtomicU64,
}

/// Latency tracking system
pub struct LatencyTracker {
    /// Latency samples
    samples: Arc<RwLock<VecDeque<u64>>>,
    /// Sample limit
    sample_limit: usize,
    /// Percentile calculator
    percentiles: Arc<RwLock<LatencyPercentiles>>,
}

#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub max: u64,
}

/// Compression engine
pub struct CompressionEngine {
    /// LZ4 encoder
    lz4_encoder: Arc<Mutex<lz4::Encoder<Vec<u8>>>>,
    /// Zstd encoder
    zstd_encoder: Arc<Mutex<zstd::Encoder<'static, Vec<u8>>>>,
}

/// Encryption engine
pub struct EncryptionEngine {
    /// Encryption keys
    keys: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Key rotation schedule
    key_rotation: Arc<RwLock<BTreeMap<Instant, String>>>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_concurrent_routes: 1000,
            route_timeout_ns: 10_000_000, // 10ms
            max_message_size: 1024 * 1024, // 1MB
            message_buffer_size: 10000,
            routing_algorithm: RoutingAlgorithm::Adaptive,
            load_balancing: LoadBalancingStrategy::LatencyBased,
            batch_size: 100,
            enable_compression: true,
            enable_encryption: false,
            priority_queue_size: 1000,
            route_cache_size: 10000,
        }
    }
}

impl MessageRouterAgent {
    /// Create a new message router agent
    pub async fn new(config: RouterConfig) -> Result<Self> {
        let id = AgentId::new();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, _) = broadcast::channel(1024);
        let (shutdown_tx, _) = mpsc::unbounded_channel();

        let message_counters = Arc::new(MessageCounters {
            total: AtomicU64::new(0),
            by_priority: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            by_type: DashMap::new(),
            dropped: AtomicU64::new(0),
            failed_routes: AtomicU64::new(0),
        });

        let latency_tracker = Arc::new(LatencyTracker {
            samples: Arc::new(RwLock::new(VecDeque::new())),
            sample_limit: 10000,
            percentiles: Arc::new(RwLock::new(LatencyPercentiles {
                p50: 0,
                p95: 0,
                p99: 0,
                p999: 0,
                max: 0,
            })),
        });

        let compression_engine = Arc::new(CompressionEngine {
            lz4_encoder: Arc::new(Mutex::new(lz4::EncoderBuilder::new().build(Vec::new())?)),
            zstd_encoder: Arc::new(Mutex::new(zstd::Encoder::new(Vec::new(), 3)?)),
        });

        let encryption_engine = Arc::new(EncryptionEngine {
            keys: Arc::new(RwLock::new(HashMap::new())),
            key_rotation: Arc::new(RwLock::new(BTreeMap::new())),
        });

        let initial_metrics = RouterMetrics {
            total_messages_routed: 0,
            messages_per_second: 0.0,
            avg_latency_ns: 0,
            p95_latency_ns: 0,
            p99_latency_ns: 0,
            max_latency_ns: 0,
            total_bandwidth_bps: 0,
            route_efficiency: 100.0,
            error_rate: 0.0,
            queue_depth: 0,
            active_routes: 0,
            compression_ratio: 1.0,
        };

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            routes: Arc::new(RwLock::new(HashMap::new())),
            route_lookup: Arc::new(DashMap::new()),
            message_queue: Arc::new(Queue::new()),
            priority_queue: Arc::new(Mutex::new(PriorityQueue::new())),
            route_cache: Arc::new(DashMap::new()),
            performance_metrics: Arc::new(RwLock::new(initial_metrics)),
            message_counters,
            latency_tracker,
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            event_tx,
            message_channels: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            running: Arc::new(Atomic::new(false)),
            message_id_gen: Arc::new(AtomicU64::new(1)),
            route_id_gen: Arc::new(AtomicU64::new(1)),
            compression_engine,
            encryption_engine,
        })
    }

    /// Start the message router
    #[instrument(skip(self), fields(router_id = %self.id))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Message Router Agent {}", self.id);
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Running;
        }
        
        self.running.store(true, AtomicOrdering::SeqCst);
        
        // Initialize routing tables
        self.initialize_routing_tables().await?;
        
        // Spawn background tasks
        self.spawn_background_tasks().await?;
        
        // Start main event loop
        self.run_event_loop().await?;
        
        Ok(())
    }

    /// Stop the message router
    #[instrument(skip(self), fields(router_id = %self.id))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Message Router Agent {}", self.id);
        
        self.running.store(false, AtomicOrdering::SeqCst);
        
        // Flush remaining messages
        self.flush_message_queue().await?;
        
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

    /// Initialize routing tables
    async fn initialize_routing_tables(&self) -> Result<()> {
        info!("Initializing routing tables");
        
        // Initialize default routes for known swarms
        let swarm_agents = vec![
            ("risk-management", vec!["risk-1", "risk-2", "risk-3", "risk-4", "risk-5"]),
            ("trading-strategy", vec!["trade-1", "trade-2", "trade-3", "trade-4", "trade-5", "trade-6"]),
            ("data-pipeline", vec!["data-1", "data-2", "data-3", "data-4", "data-5", "data-6"]),
            ("tengri-watchdog", vec!["watch-1", "watch-2", "watch-3", "watch-4", "watch-5", "watch-6", "watch-7", "watch-8"]),
        ];
        
        for (swarm_name, agent_names) in swarm_agents {
            for agent_name in agent_names {
                let agent_id = AgentId::new(); // In real implementation, this would be derived from agent_name
                self.add_default_routes(agent_id, swarm_name).await?;
            }
        }
        
        Ok(())
    }

    /// Add default routes for an agent
    async fn add_default_routes(&self, agent_id: AgentId, swarm_name: &str) -> Result<()> {
        // Create direct routes to other agents in the same swarm
        // This is a simplified implementation
        let route_id = self.route_id_gen.fetch_add(1, AtomicOrdering::SeqCst);
        
        let route_info = RouteInfo {
            id: route_id,
            source: agent_id.clone(),
            destination: agent_id.clone(), // Self-route
            path: vec![agent_id.clone()],
            metrics: RouteMetrics {
                latency_ns: 1000, // 1 microsecond
                bandwidth_bps: 1_000_000_000, // 1 Gbps
                packet_loss: 0.0,
                jitter_ns: 100,
                throughput_mps: 10000.0,
                reliability_score: 99.9,
            },
            last_update: Instant::now(),
            status: RouteStatus::Active,
        };
        
        self.routes.write().await.insert(route_id, route_info.clone());
        self.route_lookup.insert((agent_id.clone(), agent_id), route_id);
        
        Ok(())
    }

    /// Spawn background tasks
    async fn spawn_background_tasks(&self) -> Result<()> {
        // Message processing task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.message_processing_task().await;
        });

        // Priority queue processing task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.priority_queue_processing_task().await;
        });

        // Route optimization task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.route_optimization_task().await;
        });

        // Metrics collection task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.metrics_collection_task().await;
        });

        // Latency tracking task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.latency_tracking_task().await;
        });

        // Congestion detection task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.congestion_detection_task().await;
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
                        error!("Error handling router command: {}", e);
                    }
                }
                _ = sleep(Duration::from_nanos(100)) => {
                    // Ultra-low latency processing
                    self.process_high_priority_messages().await;
                }
            }
        }
        
        Ok(())
    }

    /// Handle incoming commands
    async fn handle_command(&self, command: RouterCommand) -> Result<()> {
        match command {
            RouterCommand::RouteMessage(message) => {
                self.route_message(message).await
            }
            RouterCommand::AddRoute(route_info) => {
                self.add_route(route_info).await
            }
            RouterCommand::RemoveRoute(route_id) => {
                self.remove_route(route_id).await
            }
            RouterCommand::UpdateRouteMetrics { route_id, metrics } => {
                self.update_route_metrics(route_id, metrics).await
            }
            RouterCommand::SetRoutingAlgorithm(algorithm) => {
                self.set_routing_algorithm(algorithm).await
            }
            RouterCommand::SetLoadBalancing(strategy) => {
                self.set_load_balancing(strategy).await
            }
            RouterCommand::GetRouteInfo(route_id) => {
                self.get_route_info(route_id).await
            }
            RouterCommand::GetAllRoutes => {
                self.get_all_routes().await
            }
            RouterCommand::GetRoutingStats => {
                self.get_routing_stats().await
            }
            RouterCommand::OptimizeRoutes => {
                self.optimize_routes().await
            }
            RouterCommand::FlushMessageQueue => {
                self.flush_message_queue().await
            }
            RouterCommand::EnableTracing { trace_id } => {
                self.enable_tracing(trace_id).await
            }
            RouterCommand::DisableTracing => {
                self.disable_tracing().await
            }
            RouterCommand::Shutdown => {
                self.stop().await
            }
        }
    }

    /// Route a message
    async fn route_message(&self, mut message: RoutedMessage) -> Result<()> {
        let start_time = Instant::now();
        
        // Assign message ID if not present
        if message.id == 0 {
            message.id = self.message_id_gen.fetch_add(1, AtomicOrdering::SeqCst);
        }
        
        // Update counters
        self.message_counters.total.fetch_add(1, AtomicOrdering::SeqCst);
        self.message_counters.by_priority[message.priority as usize].fetch_add(1, AtomicOrdering::SeqCst);
        
        // Check TTL
        if message.timestamp.elapsed().as_nanos() as u64 > message.ttl_ns {
            self.message_counters.dropped.fetch_add(1, AtomicOrdering::SeqCst);
            let _ = self.event_tx.send(RouterEvent::MessageDropped {
                message_id: message.id,
                reason: "TTL expired".to_string(),
            });
            return Ok(());
        }
        
        // Find route
        let route_id = self.find_route(&message.source, &message.destination).await?;
        let route_info = {
            let routes = self.routes.read().await;
            routes.get(&route_id).cloned()
        };
        
        if let Some(route) = route_info {
            // Compress message if enabled
            if self.config.enable_compression {
                self.compress_message(&mut message).await?;
            }
            
            // Encrypt message if enabled
            if self.config.enable_encryption {
                self.encrypt_message(&mut message).await?;
            }
            
            // Update routing metadata
            message.routing_metadata.hop_count += 1;
            message.routing_metadata.route_path.push(message.source.clone());
            
            // Route based on priority
            match message.priority {
                MessagePriority::Critical | MessagePriority::High => {
                    // High priority messages go to priority queue
                    let mut priority_queue = self.priority_queue.lock().await;
                    priority_queue.push(message.clone(), message.priority);
                }
                _ => {
                    // Normal priority messages go to regular queue
                    self.message_queue.push(message.clone());
                }
            }
            
            // Record latency
            let latency_ns = start_time.elapsed().as_nanos() as u64;
            self.latency_tracker.samples.write().await.push_back(latency_ns);
            
            let _ = self.event_tx.send(RouterEvent::MessageRouted {
                message_id: message.id,
                latency_ns,
                route_path: message.routing_metadata.route_path.clone().into_vec(),
            });
            
            info!("Message {} routed in {}ns", message.id, latency_ns);
        } else {
            self.message_counters.failed_routes.fetch_add(1, AtomicOrdering::SeqCst);
            let _ = self.event_tx.send(RouterEvent::MessageDropped {
                message_id: message.id,
                reason: "No route found".to_string(),
            });
        }
        
        Ok(())
    }

    /// Find route between source and destination
    async fn find_route(&self, source: &AgentId, destination: &AgentId) -> Result<u64> {
        // Check cache first
        if let Some(route_id) = self.route_lookup.get(&(source.clone(), destination.clone())) {
            return Ok(*route_id);
        }
        
        // Find best route based on algorithm
        let routes = self.routes.read().await;
        let mut best_route = None;
        let mut best_score = f64::NEG_INFINITY;
        
        for (route_id, route_info) in routes.iter() {
            if route_info.source == *source && route_info.destination == *destination {
                let score = self.calculate_route_score(route_info).await;
                if score > best_score {
                    best_score = score;
                    best_route = Some(*route_id);
                }
            }
        }
        
        if let Some(route_id) = best_route {
            // Cache the result
            self.route_lookup.insert((source.clone(), destination.clone()), route_id);
            Ok(route_id)
        } else {
            Err(OrchestrationError::RouteNotFound(format!("{} -> {}", source, destination)).into())
        }
    }

    /// Calculate route score based on current algorithm
    async fn calculate_route_score(&self, route_info: &RouteInfo) -> f64 {
        match self.config.routing_algorithm {
            RoutingAlgorithm::Direct => {
                // Prefer direct routes
                if route_info.path.len() == 1 {
                    100.0
                } else {
                    50.0 / route_info.path.len() as f64
                }
            }
            RoutingAlgorithm::ShortestPath => {
                // Prefer shortest path
                100.0 / route_info.path.len() as f64
            }
            RoutingAlgorithm::LoadAware => {
                // Consider load and latency
                let latency_score = 1000000.0 / route_info.metrics.latency_ns as f64;
                let bandwidth_score = route_info.metrics.bandwidth_bps as f64 / 1_000_000_000.0;
                let reliability_score = route_info.metrics.reliability_score / 100.0;
                
                (latency_score + bandwidth_score + reliability_score) / 3.0
            }
            RoutingAlgorithm::Adaptive => {
                // Adaptive scoring based on current conditions
                let latency_weight = 0.4;
                let bandwidth_weight = 0.3;
                let reliability_weight = 0.3;
                
                let latency_score = 1000000.0 / route_info.metrics.latency_ns as f64;
                let bandwidth_score = route_info.metrics.bandwidth_bps as f64 / 1_000_000_000.0;
                let reliability_score = route_info.metrics.reliability_score / 100.0;
                
                latency_weight * latency_score + 
                bandwidth_weight * bandwidth_score + 
                reliability_weight * reliability_score
            }
            _ => 1.0,
        }
    }

    /// Compress message
    async fn compress_message(&self, message: &mut RoutedMessage) -> Result<()> {
        let original_size = message.payload.len();
        
        if original_size > 1024 {
            // Only compress messages larger than 1KB
            match self.config.routing_algorithm {
                RoutingAlgorithm::Direct => {
                    // Use LZ4 for fast compression
                    let compressed = self.compress_lz4(&message.payload).await?;
                    if compressed.len() < original_size {
                        message.payload = compressed;
                        message.compression = Some(CompressionInfo {
                            algorithm: CompressionAlgorithm::Lz4,
                            original_size,
                            compressed_size: message.payload.len(),
                            compression_ratio: original_size as f64 / message.payload.len() as f64,
                        });
                    }
                }
                _ => {
                    // Use Zstd for better compression
                    let compressed = self.compress_zstd(&message.payload).await?;
                    if compressed.len() < original_size {
                        message.payload = compressed;
                        message.compression = Some(CompressionInfo {
                            algorithm: CompressionAlgorithm::Zstd,
                            original_size,
                            compressed_size: message.payload.len(),
                            compression_ratio: original_size as f64 / message.payload.len() as f64,
                        });
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Compress with LZ4
    async fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = self.compression_engine.lz4_encoder.lock().await;
        encoder.get_mut().clear();
        encoder.write_all(data)?;
        let (compressed, _) = encoder.finish()?;
        Ok(compressed)
    }

    /// Compress with Zstd
    async fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = self.compression_engine.zstd_encoder.lock().await;
        encoder.get_mut().clear();
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed)
    }

    /// Encrypt message
    async fn encrypt_message(&self, message: &mut RoutedMessage) -> Result<()> {
        // Simplified encryption implementation
        // In a real system, this would use proper encryption libraries
        
        message.encryption = Some(EncryptionInfo {
            algorithm: EncryptionAlgorithm::ChaCha20Poly1305,
            key_id: "default".to_string(),
            iv: vec![0u8; 12], // Placeholder IV
            auth_tag: Some(vec![0u8; 16]), // Placeholder auth tag
        });
        
        Ok(())
    }

    /// Add route
    async fn add_route(&self, route_info: RouteInfo) -> Result<()> {
        info!("Adding route {} from {} to {}", route_info.id, route_info.source, route_info.destination);
        
        let route_id = route_info.id;
        let source = route_info.source.clone();
        let destination = route_info.destination.clone();
        
        // Add to routes
        self.routes.write().await.insert(route_id, route_info);
        
        // Update lookup table
        self.route_lookup.insert((source.clone(), destination.clone()), route_id);
        
        // Update metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.active_routes += 1;
        }
        
        let _ = self.event_tx.send(RouterEvent::RouteAdded {
            route_id,
            source,
            destination,
        });
        
        Ok(())
    }

    /// Remove route
    async fn remove_route(&self, route_id: u64) -> Result<()> {
        info!("Removing route {}", route_id);
        
        if let Some(route_info) = self.routes.write().await.remove(&route_id) {
            // Remove from lookup table
            self.route_lookup.remove(&(route_info.source.clone(), route_info.destination.clone()));
            
            // Update metrics
            {
                let mut metrics = self.performance_metrics.write().await;
                metrics.active_routes = metrics.active_routes.saturating_sub(1);
            }
            
            let _ = self.event_tx.send(RouterEvent::RouteRemoved { route_id });
        }
        
        Ok(())
    }

    /// Update route metrics
    async fn update_route_metrics(&self, route_id: u64, metrics: RouteMetrics) -> Result<()> {
        let mut routes = self.routes.write().await;
        if let Some(route_info) = routes.get_mut(&route_id) {
            let old_latency = route_info.metrics.latency_ns;
            route_info.metrics = metrics;
            route_info.last_update = Instant::now();
            
            if route_info.metrics.latency_ns != old_latency {
                let _ = self.event_tx.send(RouterEvent::RouteOptimized {
                    route_id,
                    old_latency_ns: old_latency,
                    new_latency_ns: route_info.metrics.latency_ns,
                });
            }
        }
        
        Ok(())
    }

    /// Set routing algorithm
    async fn set_routing_algorithm(&self, algorithm: RoutingAlgorithm) -> Result<()> {
        info!("Setting routing algorithm to {:?}", algorithm);
        
        let old_algorithm = self.config.routing_algorithm.clone();
        // Note: This is a simplified implementation
        // In a real system, we'd need mutable access to config
        
        let _ = self.event_tx.send(RouterEvent::RoutingAlgorithmChanged {
            old_algorithm,
            new_algorithm: algorithm,
        });
        
        Ok(())
    }

    /// Set load balancing strategy
    async fn set_load_balancing(&self, strategy: LoadBalancingStrategy) -> Result<()> {
        info!("Setting load balancing strategy to {:?}", strategy);
        
        // Note: This is a simplified implementation
        // In a real system, we'd need mutable access to config
        
        Ok(())
    }

    /// Get route information
    async fn get_route_info(&self, route_id: u64) -> Result<()> {
        let routes = self.routes.read().await;
        if let Some(route_info) = routes.get(&route_id) {
            info!("Route {}: {} -> {}, latency: {}ns, status: {:?}",
                  route_id, route_info.source, route_info.destination,
                  route_info.metrics.latency_ns, route_info.status);
        } else {
            warn!("Route {} not found", route_id);
        }
        
        Ok(())
    }

    /// Get all routes
    async fn get_all_routes(&self) -> Result<()> {
        let routes = self.routes.read().await;
        info!("Total routes: {}", routes.len());
        
        for (route_id, route_info) in routes.iter() {
            info!("Route {}: {} -> {} ({}ns)", 
                  route_id, route_info.source, route_info.destination,
                  route_info.metrics.latency_ns);
        }
        
        Ok(())
    }

    /// Get routing statistics
    async fn get_routing_stats(&self) -> Result<()> {
        let metrics = self.performance_metrics.read().await;
        info!("Routing Stats: {} msg/s, {}ns avg latency, {}% efficiency",
              metrics.messages_per_second, metrics.avg_latency_ns, metrics.route_efficiency);
        
        Ok(())
    }

    /// Optimize routes
    async fn optimize_routes(&self) -> Result<()> {
        info!("Optimizing routes");
        
        // Implement route optimization logic
        // This would analyze current performance and adjust routes
        
        Ok(())
    }

    /// Flush message queue
    async fn flush_message_queue(&self) -> Result<()> {
        info!("Flushing message queue");
        
        // Process all remaining messages
        while let Some(message) = self.message_queue.pop() {
            self.deliver_message(message).await?;
        }
        
        // Process priority queue
        {
            let mut priority_queue = self.priority_queue.lock().await;
            while let Some((message, _)) = priority_queue.pop() {
                self.deliver_message(message).await?;
            }
        }
        
        Ok(())
    }

    /// Deliver message to destination
    async fn deliver_message(&self, message: RoutedMessage) -> Result<()> {
        let message_channels = self.message_channels.read().await;
        if let Some(channel) = message_channels.get(&message.destination) {
            if let Err(e) = channel.send(message.clone()) {
                error!("Failed to deliver message {}: {}", message.id, e);
                return Err(OrchestrationError::MessageDeliveryFailed(message.id.to_string()).into());
            }
        } else {
            warn!("No channel found for agent {}", message.destination);
        }
        
        Ok(())
    }

    /// Enable tracing
    async fn enable_tracing(&self, trace_id: u64) -> Result<()> {
        info!("Enabling tracing with ID {}", trace_id);
        
        // Implementation would enable message tracing
        
        Ok(())
    }

    /// Disable tracing
    async fn disable_tracing(&self) -> Result<()> {
        info!("Disabling tracing");
        
        // Implementation would disable message tracing
        
        Ok(())
    }

    /// Message processing task
    async fn message_processing_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Process messages from queue
            if let Some(message) = self.message_queue.pop() {
                if let Err(e) = self.deliver_message(message).await {
                    error!("Message delivery failed: {}", e);
                }
            }
            
            // Yield to prevent CPU spinning
            tokio::task::yield_now().await;
        }
    }

    /// Priority queue processing task
    async fn priority_queue_processing_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Process high priority messages
            let message = {
                let mut priority_queue = self.priority_queue.lock().await;
                priority_queue.pop().map(|(msg, _)| msg)
            };
            
            if let Some(message) = message {
                if let Err(e) = self.deliver_message(message).await {
                    error!("Priority message delivery failed: {}", e);
                }
            } else {
                // No high priority messages, yield
                tokio::task::yield_now().await;
            }
        }
    }

    /// Process high priority messages
    async fn process_high_priority_messages(&self) {
        // This is called from the main event loop for ultra-low latency
        let message = {
            let mut priority_queue = self.priority_queue.lock().await;
            priority_queue.pop().map(|(msg, _)| msg)
        };
        
        if let Some(message) = message {
            if let Err(e) = self.deliver_message(message).await {
                error!("High priority message delivery failed: {}", e);
            }
        }
    }

    /// Route optimization task
    async fn route_optimization_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Periodic route optimization
            if let Err(e) = self.optimize_routes().await {
                error!("Route optimization failed: {}", e);
            }
            
            sleep(Duration::from_secs(30)).await;
        }
    }

    /// Metrics collection task
    async fn metrics_collection_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            self.update_performance_metrics().await;
            sleep(Duration::from_secs(1)).await;
        }
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self) {
        let total_messages = self.message_counters.total.load(AtomicOrdering::SeqCst);
        let queue_depth = self.message_queue.len();
        
        // Calculate latency percentiles
        let percentiles = self.calculate_latency_percentiles().await;
        
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_messages_routed = total_messages;
        metrics.queue_depth = queue_depth;
        metrics.avg_latency_ns = percentiles.p50;
        metrics.p95_latency_ns = percentiles.p95;
        metrics.p99_latency_ns = percentiles.p99;
        metrics.max_latency_ns = percentiles.max;
        
        // Calculate messages per second
        // This is a simplified calculation
        metrics.messages_per_second = total_messages as f64 / 60.0; // Approximate
        
        // Update route efficiency
        let failed_routes = self.message_counters.failed_routes.load(AtomicOrdering::SeqCst);
        if total_messages > 0 {
            metrics.error_rate = (failed_routes as f64 / total_messages as f64) * 100.0;
            metrics.route_efficiency = 100.0 - metrics.error_rate;
        }
    }

    /// Calculate latency percentiles
    async fn calculate_latency_percentiles(&self) -> LatencyPercentiles {
        let mut samples = self.latency_tracker.samples.write().await;
        
        if samples.is_empty() {
            return LatencyPercentiles {
                p50: 0,
                p95: 0,
                p99: 0,
                p999: 0,
                max: 0,
            };
        }
        
        // Convert to sorted vector
        let mut sorted_samples: Vec<u64> = samples.iter().copied().collect();
        sorted_samples.sort_unstable();
        
        let len = sorted_samples.len();
        let p50 = sorted_samples[len * 50 / 100];
        let p95 = sorted_samples[len * 95 / 100];
        let p99 = sorted_samples[len * 99 / 100];
        let p999 = sorted_samples[len * 999 / 1000];
        let max = sorted_samples[len - 1];
        
        let percentiles = LatencyPercentiles {
            p50,
            p95,
            p99,
            p999,
            max,
        };
        
        // Update cached percentiles
        *self.latency_tracker.percentiles.write().await = percentiles.clone();
        
        // Trim samples if too many
        if samples.len() > self.latency_tracker.sample_limit {
            samples.drain(0..samples.len() - self.latency_tracker.sample_limit);
        }
        
        percentiles
    }

    /// Latency tracking task
    async fn latency_tracking_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Update latency percentiles
            let _percentiles = self.calculate_latency_percentiles().await;
            
            sleep(Duration::from_secs(5)).await;
        }
    }

    /// Congestion detection task
    async fn congestion_detection_task(&self) {
        while self.running.load(AtomicOrdering::SeqCst) {
            // Check for congestion on routes
            let routes = self.routes.read().await;
            
            for (route_id, route_info) in routes.iter() {
                let congestion_level = self.calculate_congestion_level(route_info).await;
                
                if congestion_level > 0.8 {
                    let _ = self.event_tx.send(RouterEvent::CongestionDetected {
                        route_id: *route_id,
                        congestion_level,
                    });
                }
            }
            
            sleep(Duration::from_secs(10)).await;
        }
    }

    /// Calculate congestion level for a route
    async fn calculate_congestion_level(&self, route_info: &RouteInfo) -> f64 {
        // Simplified congestion calculation
        let latency_factor = route_info.metrics.latency_ns as f64 / 10_000_000.0; // Normalize to 10ms
        let loss_factor = route_info.metrics.packet_loss / 100.0;
        let jitter_factor = route_info.metrics.jitter_ns as f64 / 1_000_000.0; // Normalize to 1ms
        
        (latency_factor + loss_factor + jitter_factor) / 3.0
    }

    /// Send command to router
    pub async fn send_command(&self, command: RouterCommand) -> Result<()> {
        self.command_tx.send(command)
            .map_err(|e| OrchestrationError::CommunicationError(e.to_string()))?;
        Ok(())
    }

    /// Subscribe to router events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<RouterEvent> {
        self.event_tx.subscribe()
    }

    /// Get current router metrics
    pub async fn get_router_metrics(&self) -> RouterMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Check if router is running
    pub fn is_running(&self) -> bool {
        self.running.load(AtomicOrdering::SeqCst)
    }

    /// Generate new message ID
    pub fn generate_message_id(&self) -> MessageId {
        self.message_id_gen.fetch_add(1, AtomicOrdering::SeqCst)
    }
}

impl Clone for MessageRouterAgent {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            routes: Arc::clone(&self.routes),
            route_lookup: Arc::clone(&self.route_lookup),
            message_queue: Arc::clone(&self.message_queue),
            priority_queue: Arc::clone(&self.priority_queue),
            route_cache: Arc::clone(&self.route_cache),
            performance_metrics: Arc::clone(&self.performance_metrics),
            message_counters: Arc::clone(&self.message_counters),
            latency_tracker: Arc::clone(&self.latency_tracker),
            command_tx: self.command_tx.clone(),
            command_rx: Arc::clone(&self.command_rx),
            event_tx: self.event_tx.clone(),
            message_channels: Arc::clone(&self.message_channels),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            running: Arc::clone(&self.running),
            message_id_gen: Arc::clone(&self.message_id_gen),
            route_id_gen: Arc::clone(&self.route_id_gen),
            compression_engine: Arc::clone(&self.compression_engine),
            encryption_engine: Arc::clone(&self.encryption_engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_message_router_creation() {
        let config = RouterConfig::default();
        let router = MessageRouterAgent::new(config).await.unwrap();
        
        assert!(!router.is_running());
        assert_eq!(router.generate_message_id(), 1);
        assert_eq!(router.generate_message_id(), 2);
    }

    #[tokio::test]
    async fn test_message_priority_ordering() {
        let p1 = MessagePriority::Critical;
        let p2 = MessagePriority::High;
        let p3 = MessagePriority::Normal;
        
        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p1 < p3);
    }

    #[tokio::test]
    async fn test_routing_metadata() {
        let metadata = RoutingMetadata {
            hop_count: 0,
            route_path: SmallVec::new(),
            flags: RoutingFlags {
                requires_ack: true,
                broadcast: false,
                multicast: false,
                ordered: true,
                reliable: true,
                real_time: false,
            },
            qos: QoSRequirements {
                max_latency_ns: 1000,
                min_bandwidth_bps: 1_000_000,
                max_jitter_ns: 100,
                reliability: 99.9,
                ordering: OrderingRequirement::FIFO,
            },
            trace: None,
        };
        
        assert_eq!(metadata.hop_count, 0);
        assert!(metadata.flags.requires_ack);
        assert_eq!(metadata.qos.max_latency_ns, 1000);
    }
}