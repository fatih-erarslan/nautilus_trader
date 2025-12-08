//! Integration module for Neural Forge with existing systems
//! 
//! Provides seamless integration with:
//! - cognition-engine (NHITS time-series forecasting)
//! - ruv_FANN (neural networks and swarm intelligence)
//! - Claude Code/Flow (AI-driven development)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use crate::prelude::*;

pub mod cognition_engine;
pub mod ruv_fann;
pub mod claude_flow;
pub mod ensemble;
pub mod swarm;

pub use cognition_engine::*;
pub use ruv_fann::*;
pub use claude_flow::*;
pub use ensemble::*;
pub use swarm::*;

/// Integration configuration for external systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Cognition Engine integration
    pub cognition_engine: Option<CognitionEngineConfig>,
    
    /// ruv_FANN integration
    pub ruv_fann: Option<RuvFannConfig>,
    
    /// Claude Flow integration
    pub claude_flow: Option<ClaudeFlowConfig>,
    
    /// Ensemble configuration
    pub ensemble: Option<EnsembleConfig>,
    
    /// Swarm coordination
    pub swarm: Option<SwarmConfig>,
    
    /// Performance bridge settings
    pub performance: PerformanceBridgeConfig,
}

/// Cognition Engine integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionEngineConfig {
    /// Enable cognition engine integration
    pub enabled: bool,
    
    /// Cognition engine binary path
    pub binary_path: Option<PathBuf>,
    
    /// NHITS model configuration
    pub nhits: NHITSConfig,
    
    /// Inference settings
    pub inference: InferenceConfig,
    
    /// Monitoring settings
    pub monitoring: MonitoringConfig,
}

/// NHITS model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig {
    /// Input size (sequence length)
    pub input_size: usize,
    
    /// Output size (forecast horizon)
    pub output_size: usize,
    
    /// Number of blocks
    pub n_blocks: usize,
    
    /// MLP units
    pub mlp_units: Vec<usize>,
    
    /// Pooling kernel sizes
    pub pooling_sizes: Vec<usize>,
    
    /// Interpolation mode
    pub interpolation_mode: String,
    
    /// Dropout rate
    pub dropout: f64,
    
    /// Activation function
    pub activation: String,
}

/// ruv_FANN integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvFannConfig {
    /// Enable ruv_FANN integration
    pub enabled: bool,
    
    /// ruv_FANN library path
    pub library_path: Option<PathBuf>,
    
    /// Available models to use
    pub models: Vec<String>,
    
    /// Neuro-divergent configuration
    pub neuro_divergent: NeuroDivergentConfig,
    
    /// Swarm configuration
    pub swarm: RuvSwarmConfig,
    
    /// WASM deployment settings
    pub wasm: WasmConfig,
}

/// Neuro-divergent model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroDivergentConfig {
    /// Enable neuro-divergent models
    pub enabled: bool,
    
    /// Models to include in ensemble
    pub ensemble_models: Vec<NeuroDivergentModel>,
    
    /// Forecasting horizon
    pub horizon: usize,
    
    /// Frequency of the data
    pub frequency: String,
    
    /// Cross-validation folds
    pub cv_folds: usize,
}

/// Neuro-divergent model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuroDivergentModel {
    LSTM { hidden_size: usize, num_layers: usize },
    TCN { kernel_size: usize, num_filters: usize, dilations: Vec<usize> },
    NBEATSx { blocks: Vec<usize>, layers: Vec<usize> },
    NHiTS { n_blocks: usize, mlp_units: Vec<usize> },
    TimeLLM { d_model: usize, num_heads: usize, num_layers: usize },
    Autoformer { d_model: usize, num_heads: usize, factor: usize },
    DeepAR { hidden_size: usize, num_layers: usize },
    Transformer { d_model: usize, num_heads: usize, num_layers: usize },
}

/// ruv_FANN swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvSwarmConfig {
    /// Enable swarm intelligence
    pub enabled: bool,
    
    /// Number of swarm agents
    pub num_agents: usize,
    
    /// Swarm topology
    pub topology: SwarmTopology,
    
    /// Communication protocol
    pub protocol: String,
    
    /// Coordination strategy
    pub strategy: String,
}

/// Swarm topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTopology {
    Ring,
    Mesh,
    Star,
    Hierarchical,
    Adaptive,
}

/// Claude Flow integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeFlowConfig {
    /// Enable Claude Flow integration
    pub enabled: bool,
    
    /// MCP server configuration
    pub mcp_server: McpServerConfig,
    
    /// AI-driven optimization
    pub ai_optimization: AiOptimizationConfig,
    
    /// Automated benchmarking
    pub benchmarking: BenchmarkingConfig,
}

/// MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Available tools
    pub tools: Vec<String>,
    
    /// Authentication token
    pub auth_token: Option<String>,
}

/// AI optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiOptimizationConfig {
    /// Enable hyperparameter optimization
    pub hyperparameter_tuning: bool,
    
    /// Enable architecture search
    pub architecture_search: bool,
    
    /// Enable automated feature engineering
    pub feature_engineering: bool,
    
    /// Optimization budget (iterations)
    pub budget: usize,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Enable ensemble learning
    pub enabled: bool,
    
    /// Ensemble method
    pub method: EnsembleMethod,
    
    /// Model weights
    pub weights: Option<Vec<f64>>,
    
    /// Dynamic weighting
    pub dynamic_weighting: bool,
    
    /// Performance tracking
    pub track_performance: bool,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Simple averaging
    Average,
    
    /// Weighted average
    WeightedAverage,
    
    /// Voting (classification)
    Voting,
    
    /// Stacking with meta-learner
    Stacking { meta_learner: String },
    
    /// Bayesian model averaging
    BayesianAveraging,
    
    /// Dynamic selection
    DynamicSelection,
}

/// Swarm coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Enable swarm coordination
    pub enabled: bool,
    
    /// Coordination protocol
    pub protocol: SwarmProtocol,
    
    /// Agent communication
    pub communication: CommunicationConfig,
    
    /// Task distribution
    pub task_distribution: TaskDistributionConfig,
    
    /// Fault tolerance
    pub fault_tolerance: FaultToleranceConfig,
}

/// Swarm protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmProtocol {
    /// Gossip protocol
    Gossip,
    
    /// Consensus protocol
    Consensus,
    
    /// Leader-follower
    LeaderFollower,
    
    /// Peer-to-peer
    P2P,
    
    /// Hierarchical
    Hierarchical,
}

/// Communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Protocol (TCP, UDP, WebSocket)
    pub protocol: String,
    
    /// Message format (JSON, MessagePack, Protocol Buffers)
    pub format: String,
    
    /// Compression
    pub compression: bool,
    
    /// Encryption
    pub encryption: bool,
    
    /// Heartbeat interval (ms)
    pub heartbeat_interval: u64,
}

/// Task distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDistributionConfig {
    /// Distribution strategy
    pub strategy: DistributionStrategy,
    
    /// Load balancing
    pub load_balancing: bool,
    
    /// Priority queue
    pub priority_queue: bool,
    
    /// Work stealing
    pub work_stealing: bool,
}

/// Distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionStrategy {
    RoundRobin,
    Random,
    LoadBased,
    CapabilityBased,
    Adaptive,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable fault tolerance
    pub enabled: bool,
    
    /// Failure detection timeout (ms)
    pub failure_timeout: u64,
    
    /// Retry attempts
    pub retry_attempts: usize,
    
    /// Backup strategies
    pub backup_strategy: BackupStrategy,
    
    /// Recovery policy
    pub recovery_policy: RecoveryPolicy,
}

/// Backup strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStrategy {
    /// No backup
    None,
    
    /// Hot standby
    HotStandby,
    
    /// Cold backup
    ColdBackup,
    
    /// Distributed backup
    Distributed,
}

/// Recovery policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryPolicy {
    /// Immediate restart
    Restart,
    
    /// Graceful degradation
    Degrade,
    
    /// Failover to backup
    Failover,
    
    /// Manual intervention
    Manual,
}

/// Performance bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBridgeConfig {
    /// Data flow optimization
    pub data_flow: DataFlowConfig,
    
    /// Memory management
    pub memory: MemoryBridgeConfig,
    
    /// Compute orchestration
    pub compute: ComputeBridgeConfig,
    
    /// Performance monitoring
    pub monitoring: PerformanceMonitoringConfig,
}

/// Data flow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowConfig {
    /// Buffer size for data transfers
    pub buffer_size: usize,
    
    /// Batch processing
    pub batch_processing: bool,
    
    /// Streaming mode
    pub streaming: bool,
    
    /// Compression
    pub compression: CompressionConfig,
    
    /// Serialization format
    pub serialization: SerializationFormat,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    
    /// Compression level
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}

/// Serialization formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    Json,
    MessagePack,
    Bincode,
    ProtocolBuffers,
    Arrow,
}

/// Memory bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBridgeConfig {
    /// Shared memory pools
    pub shared_pools: bool,
    
    /// Memory mapping
    pub memory_mapping: bool,
    
    /// Zero-copy transfers
    pub zero_copy: bool,
    
    /// Memory pool size
    pub pool_size: usize,
    
    /// Garbage collection strategy
    pub gc_strategy: GcStrategy,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GcStrategy {
    Aggressive,
    Conservative,
    Adaptive,
    Manual,
}

/// Compute bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBridgeConfig {
    /// Task scheduling
    pub scheduling: SchedulingConfig,
    
    /// Resource allocation
    pub resource_allocation: ResourceAllocationConfig,
    
    /// Pipeline optimization
    pub pipeline_optimization: bool,
    
    /// Compute graph optimization
    pub graph_optimization: bool,
}

/// Scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    
    /// Priority levels
    pub priority_levels: usize,
    
    /// Time slicing
    pub time_slicing: bool,
    
    /// Preemption
    pub preemption: bool,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    FIFO,
    RoundRobin,
    Priority,
    ShortestJobFirst,
    EarliestDeadlineFirst,
    Adaptive,
}

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocationConfig {
    /// CPU allocation strategy
    pub cpu_strategy: AllocationStrategy,
    
    /// Memory allocation strategy
    pub memory_strategy: AllocationStrategy,
    
    /// GPU allocation strategy
    pub gpu_strategy: AllocationStrategy,
    
    /// Dynamic reallocation
    pub dynamic_reallocation: bool,
}

/// Allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    Static,
    Dynamic,
    Proportional,
    Fair,
    Priority,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    
    /// Metrics collection frequency (ms)
    pub collection_frequency: u64,
    
    /// Metrics to collect
    pub metrics: Vec<PerformanceMetric>,
    
    /// Alerting thresholds
    pub thresholds: HashMap<String, f64>,
    
    /// Export configuration
    pub export: MetricsExportConfig,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    CPUUsage,
    MemoryUsage,
    GPUUsage,
    NetworkIO,
    DiskIO,
    Latency,
    Throughput,
    ErrorRate,
    QueueDepth,
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export format
    pub format: MetricsFormat,
    
    /// Export destination
    pub destination: ExportDestination,
    
    /// Export frequency
    pub frequency: u64,
    
    /// Retention period
    pub retention: u64,
}

/// Metrics formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsFormat {
    Prometheus,
    InfluxDB,
    OpenTelemetry,
    Json,
    Csv,
}

/// Export destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportDestination {
    File { path: PathBuf },
    Http { url: String },
    Database { connection_string: String },
    Console,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    
    /// Maximum latency (microseconds)
    pub max_latency_us: u64,
    
    /// Memory limit for inference
    pub memory_limit: usize,
    
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    
    /// Fallback strategy
    pub fallback_strategy: FallbackStrategy,
}

/// Optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Maximum,
}

/// Fallback strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    None,
    SimpleModel,
    LastKnownGood,
    Ensemble,
    HumanIntervention,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Health check frequency (ms)
    pub health_check_frequency: u64,
    
    /// Metrics collection
    pub metrics_collection: bool,
    
    /// Alerting
    pub alerting: AlertingConfig,
    
    /// Logging
    pub logging: LoggingConfig,
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    
    /// Alert channels
    pub channels: Vec<AlertChannel>,
    
    /// Alert rules
    pub rules: Vec<AlertRule>,
}

/// Alert channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    Email { address: String },
    Slack { webhook_url: String },
    Discord { webhook_url: String },
    PagerDuty { service_key: String },
    Http { url: String },
}

/// Alert rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    
    /// Metric to monitor
    pub metric: String,
    
    /// Threshold value
    pub threshold: f64,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Duration before alerting
    pub duration: u64,
    
    /// Severity level
    pub severity: Severity,
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Log format
    pub format: LogFormat,
    
    /// Output destinations
    pub outputs: Vec<LogOutput>,
    
    /// Structured logging
    pub structured: bool,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Plain,
    Json,
    Logfmt,
    Custom { template: String },
}

/// Log outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File { path: PathBuf },
    Syslog,
    Http { url: String },
}

/// WASM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// Enable WASM compilation
    pub enabled: bool,
    
    /// Target platforms
    pub targets: Vec<WasmTarget>,
    
    /// Optimization level
    pub optimization: WasmOptimization,
    
    /// Runtime configuration
    pub runtime: WasmRuntimeConfig,
}

/// WASM target platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmTarget {
    Browser,
    Node,
    Deno,
    CloudflareWorkers,
    Fastly,
    WASI,
}

/// WASM optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmOptimization {
    Debug,
    Release,
    Size,
    Speed,
}

/// WASM runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRuntimeConfig {
    /// Memory limit (bytes)
    pub memory_limit: usize,
    
    /// Stack size (bytes)
    pub stack_size: usize,
    
    /// Enable multi-threading
    pub multi_threading: bool,
    
    /// Enable SIMD
    pub simd: bool,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Enable automated benchmarking
    pub enabled: bool,
    
    /// Benchmark suites
    pub suites: Vec<BenchmarkSuite>,
    
    /// Performance targets
    pub targets: HashMap<String, f64>,
    
    /// Reporting configuration
    pub reporting: BenchmarkReportingConfig,
}

/// Benchmark suites
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkSuite {
    Training,
    Inference,
    Memory,
    Accuracy,
    Latency,
    Throughput,
    Custom { name: String, config: HashMap<String, serde_json::Value> },
}

/// Benchmark reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReportingConfig {
    /// Report format
    pub format: ReportFormat,
    
    /// Output destination
    pub output: PathBuf,
    
    /// Include historical data
    pub historical: bool,
    
    /// Generate plots
    pub plots: bool,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Markdown,
    Pdf,
    Csv,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            cognition_engine: Some(CognitionEngineConfig::default()),
            ruv_fann: Some(RuvFannConfig::default()),
            claude_flow: Some(ClaudeFlowConfig::default()),
            ensemble: Some(EnsembleConfig::default()),
            swarm: Some(SwarmConfig::default()),
            performance: PerformanceBridgeConfig::default(),
        }
    }
}

impl Default for CognitionEngineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            binary_path: None,
            nhits: NHITSConfig::default(),
            inference: InferenceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for NHITSConfig {
    fn default() -> Self {
        Self {
            input_size: 120,
            output_size: 24,
            n_blocks: 3,
            mlp_units: vec![512, 512],
            pooling_sizes: vec![2, 4, 8],
            interpolation_mode: "linear".to_string(),
            dropout: 0.1,
            activation: "relu".to_string(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_latency_us: 100,  // Sub-100μs target
            memory_limit: 1024 * 1024 * 1024,  // 1GB
            optimization_level: OptimizationLevel::Aggressive,
            fallback_strategy: FallbackStrategy::SimpleModel,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            health_check_frequency: 1000,  // 1 second
            metrics_collection: true,
            alerting: AlertingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![],
            rules: vec![],
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Json,
            outputs: vec![LogOutput::Console],
            structured: true,
        }
    }
}

impl Default for RuvFannConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            library_path: None,
            models: vec![
                "LSTM".to_string(),
                "TCN".to_string(),
                "NBEATSx".to_string(),
                "NHiTS".to_string(),
                "Transformer".to_string(),
            ],
            neuro_divergent: NeuroDivergentConfig::default(),
            swarm: RuvSwarmConfig::default(),
            wasm: WasmConfig::default(),
        }
    }
}

impl Default for NeuroDivergentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ensemble_models: vec![
                NeuroDivergentModel::LSTM { hidden_size: 128, num_layers: 3 },
                NeuroDivergentModel::TCN { kernel_size: 3, num_filters: 64, dilations: vec![1, 2, 4, 8] },
                NeuroDivergentModel::NBEATSx { blocks: vec![3, 3, 3], layers: vec![4, 4, 4] },
                NeuroDivergentModel::Transformer { d_model: 256, num_heads: 8, num_layers: 6 },
            ],
            horizon: 24,
            frequency: "H".to_string(),
            cv_folds: 3,
        }
    }
}

impl Default for RuvSwarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_agents: 8,
            topology: SwarmTopology::Adaptive,
            protocol: "consensus".to_string(),
            strategy: "adaptive".to_string(),
        }
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            targets: vec![WasmTarget::Browser, WasmTarget::Node],
            optimization: WasmOptimization::Release,
            runtime: WasmRuntimeConfig::default(),
        }
    }
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        Self {
            memory_limit: 128 * 1024 * 1024,  // 128MB
            stack_size: 1024 * 1024,  // 1MB
            multi_threading: false,
            simd: true,
        }
    }
}

impl Default for ClaudeFlowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mcp_server: McpServerConfig::default(),
            ai_optimization: AiOptimizationConfig::default(),
            benchmarking: BenchmarkingConfig::default(),
        }
    }
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 3000,
            tools: vec![
                "swarm_init".to_string(),
                "agent_spawn".to_string(),
                "task_orchestrate".to_string(),
                "neural_train".to_string(),
                "benchmark_run".to_string(),
            ],
            auth_token: None,
        }
    }
}

impl Default for AiOptimizationConfig {
    fn default() -> Self {
        Self {
            hyperparameter_tuning: true,
            architecture_search: true,
            feature_engineering: true,
            budget: 100,
        }
    }
}

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            suites: vec![
                BenchmarkSuite::Training,
                BenchmarkSuite::Inference,
                BenchmarkSuite::Accuracy,
            ],
            targets: {
                let mut targets = HashMap::new();
                targets.insert("inference_latency_us".to_string(), 100.0);
                targets.insert("training_speedup".to_string(), 10.0);
                targets.insert("accuracy".to_string(), 0.96);
                targets
            },
            reporting: BenchmarkReportingConfig::default(),
        }
    }
}

impl Default for BenchmarkReportingConfig {
    fn default() -> Self {
        Self {
            format: ReportFormat::Html,
            output: PathBuf::from("./benchmark_reports"),
            historical: true,
            plots: true,
        }
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: EnsembleMethod::WeightedAverage,
            weights: None,
            dynamic_weighting: true,
            track_performance: true,
        }
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol: SwarmProtocol::Consensus,
            communication: CommunicationConfig::default(),
            task_distribution: TaskDistributionConfig::default(),
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            protocol: "TCP".to_string(),
            format: "MessagePack".to_string(),
            compression: true,
            encryption: true,
            heartbeat_interval: 1000,
        }
    }
}

impl Default for TaskDistributionConfig {
    fn default() -> Self {
        Self {
            strategy: DistributionStrategy::Adaptive,
            load_balancing: true,
            priority_queue: true,
            work_stealing: true,
        }
    }
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_timeout: 5000,
            retry_attempts: 3,
            backup_strategy: BackupStrategy::HotStandby,
            recovery_policy: RecoveryPolicy::Failover,
        }
    }
}

impl Default for PerformanceBridgeConfig {
    fn default() -> Self {
        Self {
            data_flow: DataFlowConfig::default(),
            memory: MemoryBridgeConfig::default(),
            compute: ComputeBridgeConfig::default(),
            monitoring: PerformanceMonitoringConfig::default(),
        }
    }
}

impl Default for DataFlowConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024 * 1024,  // 1MB
            batch_processing: true,
            streaming: true,
            compression: CompressionConfig::default(),
            serialization: SerializationFormat::MessagePack,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
        }
    }
}

impl Default for MemoryBridgeConfig {
    fn default() -> Self {
        Self {
            shared_pools: true,
            memory_mapping: true,
            zero_copy: true,
            pool_size: 1024 * 1024 * 1024,  // 1GB
            gc_strategy: GcStrategy::Adaptive,
        }
    }
}

impl Default for ComputeBridgeConfig {
    fn default() -> Self {
        Self {
            scheduling: SchedulingConfig::default(),
            resource_allocation: ResourceAllocationConfig::default(),
            pipeline_optimization: true,
            graph_optimization: true,
        }
    }
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            algorithm: SchedulingAlgorithm::Adaptive,
            priority_levels: 3,
            time_slicing: true,
            preemption: true,
        }
    }
}

impl Default for ResourceAllocationConfig {
    fn default() -> Self {
        Self {
            cpu_strategy: AllocationStrategy::Dynamic,
            memory_strategy: AllocationStrategy::Dynamic,
            gpu_strategy: AllocationStrategy::Dynamic,
            dynamic_reallocation: true,
        }
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_frequency: 1000,
            metrics: vec![
                PerformanceMetric::CPUUsage,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::GPUUsage,
                PerformanceMetric::Latency,
                PerformanceMetric::Throughput,
            ],
            thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("cpu_usage".to_string(), 80.0);
                thresholds.insert("memory_usage".to_string(), 85.0);
                thresholds.insert("gpu_usage".to_string(), 90.0);
                thresholds.insert("latency_ms".to_string(), 1.0);
                thresholds
            },
            export: MetricsExportConfig::default(),
        }
    }
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            format: MetricsFormat::Prometheus,
            destination: ExportDestination::Http { url: "http://localhost:9090".to_string() },
            frequency: 10000,  // 10 seconds
            retention: 86400,  // 24 hours
        }
    }
}

impl IntegrationConfig {
    /// Validate integration configuration
    pub fn validate(&self) -> Result<()> {
        // Validate cognition engine config
        if let Some(ref config) = self.cognition_engine {
            config.validate()?;
        }
        
        // Validate ruv_FANN config
        if let Some(ref config) = self.ruv_fann {
            config.validate()?;
        }
        
        // Validate Claude Flow config
        if let Some(ref config) = self.claude_flow {
            config.validate()?;
        }
        
        // Validate ensemble config
        if let Some(ref config) = self.ensemble {
            config.validate()?;
        }
        
        // Validate swarm config
        if let Some(ref config) = self.swarm {
            config.validate()?;
        }
        
        self.performance.validate()?;
        
        Ok(())
    }
}

impl CognitionEngineConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if let Some(ref path) = self.binary_path {
                if !path.exists() {
                    return Err(NeuralForgeError::config("Cognition engine binary not found"));
                }
            }
            
            self.nhits.validate()?;
            self.inference.validate()?;
        }
        
        Ok(())
    }
}

impl NHITSConfig {
    fn validate(&self) -> Result<()> {
        if self.input_size == 0 {
            return Err(NeuralForgeError::config("NHITS input size must be > 0"));
        }
        
        if self.output_size == 0 {
            return Err(NeuralForgeError::config("NHITS output size must be > 0"));
        }
        
        if self.n_blocks == 0 {
            return Err(NeuralForgeError::config("NHITS n_blocks must be > 0"));
        }
        
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(NeuralForgeError::config("NHITS dropout must be in [0, 1)"));
        }
        
        Ok(())
    }
}

impl InferenceConfig {
    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(NeuralForgeError::config("Inference batch size must be > 0"));
        }
        
        if self.max_latency_us == 0 {
            return Err(NeuralForgeError::config("Max latency must be > 0"));
        }
        
        Ok(())
    }
}

impl RuvFannConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.models.is_empty() {
                return Err(NeuralForgeError::config("ruv_FANN models list cannot be empty"));
            }
            
            self.neuro_divergent.validate()?;
            self.swarm.validate()?;
        }
        
        Ok(())
    }
}

impl NeuroDivergentConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.ensemble_models.is_empty() {
                return Err(NeuralForgeError::config("Ensemble models list cannot be empty"));
            }
            
            if self.horizon == 0 {
                return Err(NeuralForgeError::config("Forecast horizon must be > 0"));
            }
        }
        
        Ok(())
    }
}

impl RuvSwarmConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.num_agents == 0 {
                return Err(NeuralForgeError::config("Number of swarm agents must be > 0"));
            }
        }
        
        Ok(())
    }
}

impl ClaudeFlowConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            self.mcp_server.validate()?;
        }
        
        Ok(())
    }
}

impl McpServerConfig {
    fn validate(&self) -> Result<()> {
        if self.port == 0 {
            return Err(NeuralForgeError::config("MCP server port must be > 0"));
        }
        
        Ok(())
    }
}

impl EnsembleConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if let Some(ref weights) = self.weights {
                if weights.is_empty() {
                    return Err(NeuralForgeError::config("Ensemble weights cannot be empty"));
                }
                
                let sum: f64 = weights.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(NeuralForgeError::config("Ensemble weights must sum to 1.0"));
                }
            }
        }
        
        Ok(())
    }
}

impl SwarmConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            self.communication.validate()?;
            self.task_distribution.validate()?;
            self.fault_tolerance.validate()?;
        }
        
        Ok(())
    }
}

impl CommunicationConfig {
    fn validate(&self) -> Result<()> {
        if self.heartbeat_interval == 0 {
            return Err(NeuralForgeError::config("Heartbeat interval must be > 0"));
        }
        
        Ok(())
    }
}

impl TaskDistributionConfig {
    fn validate(&self) -> Result<()> {
        // No specific validation needed for now
        Ok(())
    }
}

impl FaultToleranceConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.failure_timeout == 0 {
                return Err(NeuralForgeError::config("Failure timeout must be > 0"));
            }
            
            if self.retry_attempts == 0 {
                return Err(NeuralForgeError::config("Retry attempts must be > 0"));
            }
        }
        
        Ok(())
    }
}

impl PerformanceBridgeConfig {
    fn validate(&self) -> Result<()> {
        self.data_flow.validate()?;
        self.memory.validate()?;
        self.compute.validate()?;
        self.monitoring.validate()?;
        
        Ok(())
    }
}

impl DataFlowConfig {
    fn validate(&self) -> Result<()> {
        if self.buffer_size == 0 {
            return Err(NeuralForgeError::config("Buffer size must be > 0"));
        }
        
        Ok(())
    }
}

impl MemoryBridgeConfig {
    fn validate(&self) -> Result<()> {
        if self.pool_size == 0 {
            return Err(NeuralForgeError::config("Memory pool size must be > 0"));
        }
        
        Ok(())
    }
}

impl ComputeBridgeConfig {
    fn validate(&self) -> Result<()> {
        self.scheduling.validate()?;
        self.resource_allocation.validate()?;
        
        Ok(())
    }
}

impl SchedulingConfig {
    fn validate(&self) -> Result<()> {
        if self.priority_levels == 0 {
            return Err(NeuralForgeError::config("Priority levels must be > 0"));
        }
        
        Ok(())
    }
}

impl ResourceAllocationConfig {
    fn validate(&self) -> Result<()> {
        // No specific validation needed for now
        Ok(())
    }
}

impl PerformanceMonitoringConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.collection_frequency == 0 {
                return Err(NeuralForgeError::config("Collection frequency must be > 0"));
            }
            
            if self.metrics.is_empty() {
                return Err(NeuralForgeError::config("Metrics list cannot be empty"));
            }
        }
        
        Ok(())
    }
}