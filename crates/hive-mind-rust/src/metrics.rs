//! Performance monitoring and metrics collection system

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};

use crate::{
    config::MetricsConfig,
    error::{HiveMindError, Result},
};

/// Main metrics collection system
#[derive(Debug)]
pub struct MetricsCollector {
    /// Configuration
    config: MetricsConfig,
    
    /// Metrics storage
    storage: Arc<MetricsStorage>,
    
    /// Real-time metrics aggregator
    aggregator: Arc<MetricsAggregator>,
    
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
    
    /// Metrics exporters
    exporters: Arc<RwLock<Vec<Box<dyn MetricsExporter + Send + Sync>>>>,
    
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    
    /// Collection state
    state: Arc<RwLock<CollectionState>>,
    
    /// Metrics channel for async collection
    metrics_tx: mpsc::UnboundedSender<MetricEvent>,
    metrics_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<MetricEvent>>>>,
}

/// Metrics storage system
#[derive(Debug)]
pub struct MetricsStorage {
    /// Time-series database
    time_series: Arc<RwLock<TimeSeriesDB>>,
    
    /// Aggregated metrics cache
    aggregated_cache: Arc<RwLock<HashMap<String, AggregatedMetric>>>,
    
    /// Raw metrics buffer
    raw_buffer: Arc<RwLock<Vec<RawMetric>>>,
    
    /// Storage configuration
    config: StorageConfig,
}

/// Time-series database for metrics
#[derive(Debug)]
pub struct TimeSeriesDB {
    /// Data points organized by metric name
    data_points: HashMap<String, Vec<DataPoint>>,
    
    /// Metric metadata
    metadata: HashMap<String, MetricMetadata>,
    
    /// Retention policies
    retention_policies: HashMap<String, RetentionPolicy>,
    
    /// Index for fast queries
    index: HashMap<String, Vec<usize>>,
}

/// Individual data point in time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    
    /// Metric value
    pub value: MetricValue,
    
    /// Tags for grouping
    pub tags: HashMap<String, String>,
    
    /// Data point metadata
    pub metadata: Option<serde_json::Value>,
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter value (monotonically increasing)
    Counter(u64),
    
    /// Gauge value (can increase/decrease)
    Gauge(f64),
    
    /// Histogram with buckets
    Histogram(HistogramValue),
    
    /// Summary with quantiles
    Summary(SummaryValue),
    
    /// Distribution of values
    Distribution(Vec<f64>),
    
    /// Boolean value
    Boolean(bool),
    
    /// String value
    String(String),
}

/// Histogram value with buckets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramValue {
    /// Sample count
    pub count: u64,
    
    /// Sum of all samples
    pub sum: f64,
    
    /// Bucket counts
    pub buckets: Vec<HistogramBucket>,
}

/// Histogram bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Upper bound of bucket
    pub upper_bound: f64,
    
    /// Count of samples in bucket
    pub count: u64,
}

/// Summary value with quantiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryValue {
    /// Sample count
    pub count: u64,
    
    /// Sum of all samples
    pub sum: f64,
    
    /// Quantile values
    pub quantiles: Vec<Quantile>,
}

/// Quantile definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantile {
    /// Quantile level (0.0 - 1.0)
    pub quantile: f64,
    
    /// Quantile value
    pub value: f64,
}

/// Metric metadata
#[derive(Debug, Clone)]
pub struct MetricMetadata {
    /// Metric name
    pub name: String,
    
    /// Metric type
    pub metric_type: MetricType,
    
    /// Description
    pub description: String,
    
    /// Unit of measurement
    pub unit: String,
    
    /// Labels/dimensions
    pub labels: Vec<String>,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Types of metrics
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Distribution,
    Boolean,
    String,
}

/// Retention policy for metrics
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age of data points
    pub max_age: Duration,
    
    /// Maximum number of data points
    pub max_points: usize,
    
    /// Downsampling strategy
    pub downsampling: DownsamplingStrategy,
    
    /// Compression settings
    pub compression: CompressionSettings,
}

/// Downsampling strategies
#[derive(Debug, Clone, PartialEq)]
pub enum DownsamplingStrategy {
    /// No downsampling
    None,
    
    /// Average values in time windows
    Average(Duration),
    
    /// Maximum values in time windows
    Maximum(Duration),
    
    /// Minimum values in time windows
    Minimum(Duration),
    
    /// Last value in time windows
    Last(Duration),
    
    /// Custom aggregation function
    Custom(String),
}

/// Compression settings for stored metrics
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Compression enabled
    pub enabled: bool,
    
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    
    /// Compression level (1-9)
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
    None,
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,
    
    /// Flush interval for persistence
    pub flush_interval: Duration,
    
    /// Batch size for writes
    pub batch_size: usize,
    
    /// Enable persistence to disk
    pub enable_persistence: bool,
    
    /// Persistence directory
    pub persistence_dir: Option<String>,
}

/// Raw metric before aggregation
#[derive(Debug, Clone)]
pub struct RawMetric {
    /// Metric name
    pub name: String,
    
    /// Metric value
    pub value: MetricValue,
    
    /// Tags
    pub tags: HashMap<String, String>,
    
    /// Timestamp
    pub timestamp: SystemTime,
    
    /// Source component
    pub source: String,
}

/// Aggregated metric result
#[derive(Debug, Clone)]
pub struct AggregatedMetric {
    /// Metric name
    pub name: String,
    
    /// Aggregation type
    pub aggregation_type: AggregationType,
    
    /// Aggregated value
    pub value: f64,
    
    /// Sample count
    pub sample_count: u64,
    
    /// Time window
    pub time_window: Duration,
    
    /// Aggregation timestamp
    pub aggregated_at: SystemTime,
    
    /// Tags used for grouping
    pub group_tags: HashMap<String, String>,
}

/// Types of aggregation
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationType {
    Sum,
    Average,
    Minimum,
    Maximum,
    Count,
    Rate,
    Percentile(f64),
    StandardDeviation,
    Variance,
}

/// Metrics aggregator for real-time processing
#[derive(Debug)]
pub struct MetricsAggregator {
    /// Aggregation rules
    rules: Arc<RwLock<Vec<AggregationRule>>>,
    
    /// Active aggregations
    active_aggregations: Arc<RwLock<HashMap<String, ActiveAggregation>>>,
    
    /// Aggregation workers
    workers: Arc<RwLock<Vec<AggregationWorker>>>,
    
    /// Results channel
    results_tx: mpsc::UnboundedSender<AggregatedMetric>,
    results_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<AggregatedMetric>>>>,
}

/// Aggregation rule definition
#[derive(Debug, Clone)]
pub struct AggregationRule {
    /// Rule name
    pub name: String,
    
    /// Metric pattern to match
    pub metric_pattern: String,
    
    /// Aggregation type
    pub aggregation_type: AggregationType,
    
    /// Time window for aggregation
    pub time_window: Duration,
    
    /// Grouping tags
    pub group_by: Vec<String>,
    
    /// Filters
    pub filters: Vec<MetricFilter>,
    
    /// Rule enabled
    pub enabled: bool,
}

/// Metric filter for aggregation
#[derive(Debug, Clone)]
pub struct MetricFilter {
    /// Filter type
    pub filter_type: FilterType,
    
    /// Filter value
    pub value: String,
    
    /// Filter operation
    pub operation: FilterOperation,
}

/// Filter types
#[derive(Debug, Clone, PartialEq)]
pub enum FilterType {
    Tag,
    MetricName,
    Value,
    Source,
}

/// Filter operations
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOperation {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    GreaterThan,
    LessThan,
    Regex,
}

/// Active aggregation state
#[derive(Debug)]
pub struct ActiveAggregation {
    /// Aggregation ID
    pub id: Uuid,
    
    /// Rule being applied
    pub rule: AggregationRule,
    
    /// Accumulated values
    pub values: Vec<f64>,
    
    /// Window start time
    pub window_start: SystemTime,
    
    /// Last update time
    pub last_updated: SystemTime,
    
    /// Sample count
    pub sample_count: u64,
}

/// Aggregation worker for parallel processing
#[derive(Debug)]
pub struct AggregationWorker {
    /// Worker ID
    pub id: Uuid,
    
    /// Worker state
    pub state: WorkerState,
    
    /// Assigned rules
    pub assigned_rules: Vec<String>,
    
    /// Performance metrics
    pub performance: WorkerPerformance,
}

/// Worker states
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerState {
    Idle,
    Processing,
    Overloaded,
    Error(String),
}

/// Worker performance metrics
#[derive(Debug, Clone)]
pub struct WorkerPerformance {
    /// Metrics processed
    pub metrics_processed: u64,
    
    /// Processing rate (metrics/second)
    pub processing_rate: f64,
    
    /// Average processing time (milliseconds)
    pub avg_processing_time: f64,
    
    /// Error count
    pub error_count: u64,
    
    /// Last update
    pub last_updated: SystemTime,
}

/// Performance monitor for system-wide metrics
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// System metrics collector
    system_collector: Arc<SystemMetricsCollector>,
    
    /// Application metrics collector
    app_collector: Arc<ApplicationMetricsCollector>,
    
    /// Performance profiles
    profiles: Arc<RwLock<HashMap<String, PerformanceProfile>>>,
    
    /// Monitoring state
    state: Arc<RwLock<MonitoringState>>,
}

/// System metrics collector
#[derive(Debug)]
pub struct SystemMetricsCollector {
    /// CPU metrics
    cpu_metrics: Arc<RwLock<CpuMetrics>>,
    
    /// Memory metrics
    memory_metrics: Arc<RwLock<MemoryMetrics>>,
    
    /// Network metrics
    network_metrics: Arc<RwLock<NetworkMetrics>>,
    
    /// Disk metrics
    disk_metrics: Arc<RwLock<DiskMetrics>>,
    
    /// Collection interval
    collection_interval: Duration,
}

/// CPU performance metrics
#[derive(Debug, Clone)]
pub struct CpuMetrics {
    /// CPU utilization percentage (0.0 - 100.0)
    pub utilization: f64,
    
    /// Load average (1, 5, 15 minutes)
    pub load_average: [f64; 3],
    
    /// Context switches per second
    pub context_switches: u64,
    
    /// Interrupts per second
    pub interrupts: u64,
    
    /// CPU frequency (MHz)
    pub frequency: f64,
    
    /// Number of CPU cores
    pub core_count: usize,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Memory performance metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Total memory (bytes)
    pub total_memory: u64,
    
    /// Used memory (bytes)
    pub used_memory: u64,
    
    /// Available memory (bytes)
    pub available_memory: u64,
    
    /// Memory utilization percentage
    pub utilization: f64,
    
    /// Swap total (bytes)
    pub swap_total: u64,
    
    /// Swap used (bytes)
    pub swap_used: u64,
    
    /// Page faults per second
    pub page_faults: u64,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Bytes received per second
    pub bytes_received_per_sec: u64,
    
    /// Bytes sent per second
    pub bytes_sent_per_sec: u64,
    
    /// Packets received per second
    pub packets_received_per_sec: u64,
    
    /// Packets sent per second
    pub packets_sent_per_sec: u64,
    
    /// Network errors per second
    pub errors_per_sec: u64,
    
    /// Network drops per second
    pub drops_per_sec: u64,
    
    /// Active connections
    pub active_connections: u64,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Disk performance metrics
#[derive(Debug, Clone)]
pub struct DiskMetrics {
    /// Disk read bytes per second
    pub read_bytes_per_sec: u64,
    
    /// Disk write bytes per second
    pub write_bytes_per_sec: u64,
    
    /// Disk read operations per second
    pub read_ops_per_sec: u64,
    
    /// Disk write operations per second
    pub write_ops_per_sec: u64,
    
    /// Disk utilization percentage
    pub utilization: f64,
    
    /// Average response time (milliseconds)
    pub avg_response_time: f64,
    
    /// Queue depth
    pub queue_depth: u64,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Application-specific metrics collector
#[derive(Debug)]
pub struct ApplicationMetricsCollector {
    /// Hive mind specific metrics
    hive_metrics: Arc<RwLock<HiveMindMetrics>>,
    
    /// Trading system metrics
    trading_metrics: Arc<RwLock<TradingMetrics>>,
    
    /// Custom metrics registry
    custom_metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
}

/// Hive mind specific metrics
#[derive(Debug, Clone)]
pub struct HiveMindMetrics {
    /// Active agents count
    pub active_agents: u64,
    
    /// Consensus operations per second
    pub consensus_ops_per_sec: f64,
    
    /// Memory operations per second
    pub memory_ops_per_sec: f64,
    
    /// Neural computations per second
    pub neural_ops_per_sec: f64,
    
    /// Network messages per second
    pub network_msgs_per_sec: f64,
    
    /// Agent coordination efficiency
    pub coordination_efficiency: f64,
    
    /// Collective intelligence score
    pub intelligence_score: f64,
    
    /// System resilience score
    pub resilience_score: f64,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Trading system specific metrics
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    /// Orders per second
    pub orders_per_sec: f64,
    
    /// Trades per second
    pub trades_per_sec: f64,
    
    /// Average order latency (microseconds)
    pub avg_order_latency: f64,
    
    /// Market data updates per second
    pub market_data_updates_per_sec: f64,
    
    /// Risk checks per second
    pub risk_checks_per_sec: f64,
    
    /// Portfolio value
    pub portfolio_value: f64,
    
    /// P&L (profit and loss)
    pub pnl: f64,
    
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Last update time
    pub last_updated: SystemTime,
}

/// Custom metric definition
#[derive(Debug, Clone)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    
    /// Metric value
    pub value: MetricValue,
    
    /// Metric metadata
    pub metadata: MetricMetadata,
    
    /// Collection function
    pub collector: Option<String>, // Function name for collection
    
    /// Update interval
    pub update_interval: Duration,
    
    /// Last collected time
    pub last_collected: SystemTime,
}

/// Performance profile for different components
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Profile name
    pub name: String,
    
    /// Component being profiled
    pub component: String,
    
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    
    /// Current performance state
    pub current_state: PerformanceState,
    
    /// Performance history
    pub history: Vec<PerformanceSnapshot>,
    
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetectionSettings,
}

/// Performance thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// CPU utilization thresholds
    pub cpu_warning: f64,
    pub cpu_critical: f64,
    
    /// Memory utilization thresholds
    pub memory_warning: f64,
    pub memory_critical: f64,
    
    /// Response time thresholds (milliseconds)
    pub response_time_warning: f64,
    pub response_time_critical: f64,
    
    /// Throughput thresholds (ops/sec)
    pub throughput_warning: f64,
    pub throughput_critical: f64,
    
    /// Error rate thresholds (errors/sec)
    pub error_rate_warning: f64,
    pub error_rate_critical: f64,
}

/// Performance state
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceState {
    Optimal,
    Warning,
    Critical,
    Degraded,
    Failed,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,
    
    /// Performance metrics at snapshot time
    pub metrics: HashMap<String, f64>,
    
    /// Performance state
    pub state: PerformanceState,
    
    /// Anomalies detected
    pub anomalies: Vec<PerformanceAnomaly>,
}

/// Performance anomaly
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    
    /// Affected metric
    pub metric: String,
    
    /// Anomaly severity
    pub severity: AnomalySeverity,
    
    /// Anomaly description
    pub description: String,
    
    /// Detection timestamp
    pub detected_at: SystemTime,
    
    /// Anomaly duration
    pub duration: Option<Duration>,
}

/// Types of performance anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Value significantly higher than normal
    Spike,
    
    /// Value significantly lower than normal
    Drop,
    
    /// Gradual increase over time
    Trend,
    
    /// Cyclical pattern broken
    Cyclical,
    
    /// Seasonal pattern broken
    Seasonal,
    
    /// Statistical outlier
    Outlier,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly detection settings
#[derive(Debug, Clone)]
pub struct AnomalyDetectionSettings {
    /// Detection enabled
    pub enabled: bool,
    
    /// Sensitivity level (0.0 - 1.0)
    pub sensitivity: f64,
    
    /// Minimum anomaly duration
    pub min_duration: Duration,
    
    /// Detection algorithms
    pub algorithms: Vec<AnomalyDetectionAlgorithm>,
    
    /// Baseline window size
    pub baseline_window: Duration,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical z-score based detection
    ZScore { threshold: f64 },
    
    /// Interquartile range based detection
    IQR { multiplier: f64 },
    
    /// Moving average based detection
    MovingAverage { window: Duration, threshold: f64 },
    
    /// Exponential smoothing based detection
    ExponentialSmoothing { alpha: f64, threshold: f64 },
    
    /// Machine learning based detection
    MachineLearning { model: String },
}

/// Monitoring state
#[derive(Debug, Clone)]
pub struct MonitoringState {
    /// Monitoring enabled
    pub enabled: bool,
    
    /// Collection interval
    pub collection_interval: Duration,
    
    /// Last collection time
    pub last_collection: SystemTime,
    
    /// Metrics collected count
    pub metrics_collected: u64,
    
    /// Anomalies detected count
    pub anomalies_detected: u64,
    
    /// Alerts sent count
    pub alerts_sent: u64,
}

/// Metrics exporter trait
pub trait MetricsExporter: std::fmt::Debug {
    /// Export metrics to external system
    fn export(&self, metrics: &[AggregatedMetric]) -> Result<()>;
    
    /// Get exporter name
    fn name(&self) -> &str;
    
    /// Check if exporter is healthy
    fn is_healthy(&self) -> bool;
}

/// Prometheus metrics exporter
#[derive(Debug)]
pub struct PrometheusExporter {
    /// Export endpoint
    endpoint: String,
    
    /// Authentication token
    token: Option<String>,
    
    /// Export interval
    interval: Duration,
    
    /// Last export time
    last_export: Arc<RwLock<SystemTime>>,
}

/// InfluxDB metrics exporter
#[derive(Debug)]
pub struct InfluxDBExporter {
    /// Database URL
    url: String,
    
    /// Database name
    database: String,
    
    /// Username
    username: Option<String>,
    
    /// Password
    password: Option<String>,
    
    /// Batch size
    batch_size: usize,
}

/// Alert manager for metrics
#[derive(Debug)]
pub struct AlertManager {
    /// Alert rules
    rules: Arc<RwLock<Vec<AlertRule>>>,
    
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    
    /// Alert channels
    channels: Arc<RwLock<Vec<Box<dyn AlertChannel + Send + Sync>>>>,
    
    /// Alert history
    history: Arc<RwLock<Vec<AlertEvent>>>,
    
    /// Alert state
    state: Arc<RwLock<AlertManagerState>>,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    
    /// Rule condition
    pub condition: AlertCondition,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert message template
    pub message_template: String,
    
    /// Rule enabled
    pub enabled: bool,
    
    /// Cooldown period
    pub cooldown: Duration,
    
    /// Tags for routing
    pub tags: HashMap<String, String>,
}

/// Alert condition
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: f64,
        duration: Duration,
    },
    
    /// Rate of change condition
    RateOfChange {
        metric: String,
        rate_threshold: f64,
        window: Duration,
    },
    
    /// Anomaly-based condition
    Anomaly {
        metric: String,
        anomaly_type: AnomalyType,
        sensitivity: f64,
    },
    
    /// Composite condition
    Composite {
        operator: LogicalOperator,
        conditions: Vec<AlertCondition>,
    },
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Logical operators for composite conditions
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Alert ID
    pub id: Uuid,
    
    /// Rule that triggered the alert
    pub rule_name: String,
    
    /// Alert state
    pub state: AlertState,
    
    /// Trigger timestamp
    pub triggered_at: SystemTime,
    
    /// Last notification time
    pub last_notified: SystemTime,
    
    /// Notification count
    pub notification_count: u32,
    
    /// Alert context
    pub context: HashMap<String, serde_json::Value>,
}

/// Alert states
#[derive(Debug, Clone, PartialEq)]
pub enum AlertState {
    Pending,
    Firing,
    Resolved,
    Suppressed,
    Acknowledged,
}

/// Alert channel trait
pub trait AlertChannel: std::fmt::Debug {
    /// Send alert notification
    fn send_alert(&self, alert: &ActiveAlert) -> Result<()>;
    
    /// Get channel name
    fn name(&self) -> &str;
    
    /// Check if channel is healthy
    fn is_healthy(&self) -> bool;
}

/// Email alert channel
#[derive(Debug)]
pub struct EmailAlertChannel {
    /// SMTP server
    smtp_server: String,
    
    /// SMTP port
    smtp_port: u16,
    
    /// Username
    username: String,
    
    /// Password
    password: String,
    
    /// Recipients
    recipients: Vec<String>,
}

/// Slack alert channel
#[derive(Debug)]
pub struct SlackAlertChannel {
    /// Webhook URL
    webhook_url: String,
    
    /// Channel name
    channel: String,
    
    /// Bot username
    username: String,
}

/// Alert event for history
#[derive(Debug, Clone)]
pub struct AlertEvent {
    /// Event ID
    pub id: Uuid,
    
    /// Alert ID
    pub alert_id: Uuid,
    
    /// Event type
    pub event_type: AlertEventType,
    
    /// Event timestamp
    pub timestamp: SystemTime,
    
    /// Event details
    pub details: HashMap<String, serde_json::Value>,
}

/// Alert event types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertEventType {
    Triggered,
    Resolved,
    Acknowledged,
    Suppressed,
    Escalated,
    NotificationSent,
    NotificationFailed,
}

/// Alert manager state
#[derive(Debug, Clone)]
pub struct AlertManagerState {
    /// Manager enabled
    pub enabled: bool,
    
    /// Active alerts count
    pub active_alerts_count: usize,
    
    /// Rules evaluated count
    pub rules_evaluated: u64,
    
    /// Notifications sent count
    pub notifications_sent: u64,
    
    /// Last evaluation time
    pub last_evaluation: SystemTime,
}

/// Collection state
#[derive(Debug, Clone)]
pub struct CollectionState {
    /// Collection enabled
    pub enabled: bool,
    
    /// Collection start time
    pub started_at: SystemTime,
    
    /// Last collection time
    pub last_collection: SystemTime,
    
    /// Metrics collected count
    pub metrics_collected: u64,
    
    /// Collection errors count
    pub collection_errors: u64,
}

/// Metric event for async processing
#[derive(Debug, Clone)]
pub struct MetricEvent {
    /// Event type
    pub event_type: MetricEventType,
    
    /// Event timestamp
    pub timestamp: SystemTime,
    
    /// Event data
    pub data: serde_json::Value,
    
    /// Event source
    pub source: String,
}

/// Metric event types
#[derive(Debug, Clone, PartialEq)]
pub enum MetricEventType {
    MetricCollected,
    MetricAggregated,
    AlertTriggered,
    AlertResolved,
    ExportCompleted,
    ExportFailed,
    SystemStarted,
    SystemStopped,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: &MetricsConfig) -> Result<Self> {
        info!("Initializing metrics collector");
        
        let storage = Arc::new(MetricsStorage::new()?);
        let aggregator = Arc::new(MetricsAggregator::new()?);
        let performance_monitor = Arc::new(PerformanceMonitor::new()?);
        let alert_manager = Arc::new(AlertManager::new()?);
        let exporters = Arc::new(RwLock::new(Vec::new()));
        let state = Arc::new(RwLock::new(CollectionState::default()));
        
        let (metrics_tx, metrics_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            config: config.clone(),
            storage,
            aggregator,
            performance_monitor,
            exporters,
            alert_manager,
            state,
            metrics_tx,
            metrics_rx: Arc::new(RwLock::new(Some(metrics_rx))),
        })
    }
    
    /// Start metrics collection
    pub async fn start(&self) -> Result<()> {
        info!("Starting metrics collector");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.enabled = true;
            state.started_at = SystemTime::now();
            state.last_collection = SystemTime::now();
        }
        
        // Start components
        self.storage.start().await?;
        self.aggregator.start().await?;
        self.performance_monitor.start().await?;
        self.alert_manager.start().await?;
        
        // Start collection loop
        self.start_collection_loop().await?;
        
        info!("Metrics collector started");
        Ok(())
    }
    
    /// Stop metrics collection
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping metrics collector");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.enabled = false;
        }
        
        // Stop components
        self.storage.stop().await?;
        self.aggregator.stop().await?;
        self.performance_monitor.stop().await?;
        self.alert_manager.stop().await?;
        
        info!("Metrics collector stopped");
        Ok(())
    }
    
    /// Record a metric operation
    pub async fn record_metric(&self, name: &str, value: MetricValue, tags: HashMap<String, String>) {
        let metric = RawMetric {
            name: name.to_string(),
            value,
            tags,
            timestamp: SystemTime::now(),
            source: "hive_mind".to_string(),
        };
        
        if let Err(e) = self.metrics_tx.send(MetricEvent {
            event_type: MetricEventType::MetricCollected,
            timestamp: SystemTime::now(),
            data: serde_json::to_value(&metric).unwrap_or_default(),
            source: "metrics_collector".to_string(),
        }) {
            error!("Failed to send metric event: {}", e);
        }
    }
    
    /// Record consensus operation
    pub async fn record_consensus_operation(&self, operation: &str, count: u64) {
        self.record_metric(
            &format!("consensus_{}", operation),
            MetricValue::Counter(count),
            HashMap::new(),
        ).await;
    }
    
    /// Record memory operation
    pub async fn record_memory_operation(&self, operation: &str, count: u64) {
        self.record_metric(
            &format!("memory_{}", operation),
            MetricValue::Counter(count),
            HashMap::new(),
        ).await;
    }
    
    /// Record neural operation
    pub async fn record_neural_operation(&self, operation: &str, count: u64) {
        self.record_metric(
            &format!("neural_{}", operation),
            MetricValue::Counter(count),
            HashMap::new(),
        ).await;
    }
    
    /// Record network operation
    pub async fn record_network_operation(&self, operation: &str, count: u64) {
        self.record_metric(
            &format!("network_{}", operation),
            MetricValue::Counter(count),
            HashMap::new(),
        ).await;
    }
    
    /// Record agent operation
    pub async fn record_agent_operation(&self, operation: &str, count: u64) {
        self.record_metric(
            &format!("agent_{}", operation),
            MetricValue::Counter(count),
            HashMap::new(),
        ).await;
    }
    
    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self, pattern: &str) -> Result<Vec<AggregatedMetric>> {
        self.storage.get_aggregated_metrics(pattern).await
    }
    
    /// Start collection loop
    async fn start_collection_loop(&self) -> Result<()> {
        let mut receiver = {
            let mut rx_guard = self.metrics_rx.write().await;
            rx_guard.take().ok_or_else(|| HiveMindError::InvalidState {
                message: "Metrics receiver already taken".to_string(),
            })?
        };
        
        let storage = self.storage.clone();
        let aggregator = self.aggregator.clone();
        let state = self.state.clone();
        
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                if let Err(e) = Self::process_metric_event(&event, &storage, &aggregator, &state).await {
                    error!("Failed to process metric event: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Process metric event
    async fn process_metric_event(
        event: &MetricEvent,
        storage: &Arc<MetricsStorage>,
        aggregator: &Arc<MetricsAggregator>,
        state: &Arc<RwLock<CollectionState>>,
    ) -> Result<()> {
        match event.event_type {
            MetricEventType::MetricCollected => {
                if let Ok(metric) = serde_json::from_value::<RawMetric>(event.data.clone()) {
                    storage.store_raw_metric(metric.clone()).await?;
                    aggregator.process_metric(metric).await?;
                    
                    // Update collection count
                    let mut state_guard = state.write().await;
                    state_guard.metrics_collected += 1;
                    state_guard.last_collection = SystemTime::now();
                }
            }
            _ => {
                debug!("Unhandled metric event type: {:?}", event.event_type);
            }
        }
        
        Ok(())
    }
}

// Implementation stubs for supporting structures
impl MetricsStorage {
    fn new() -> Result<Self> {
        Ok(Self {
            time_series: Arc::new(RwLock::new(TimeSeriesDB::new())),
            aggregated_cache: Arc::new(RwLock::new(HashMap::new())),
            raw_buffer: Arc::new(RwLock::new(Vec::new())),
            config: StorageConfig::default(),
        })
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting metrics storage");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping metrics storage");
        Ok(())
    }
    
    async fn store_raw_metric(&self, metric: RawMetric) -> Result<()> {
        let mut buffer = self.raw_buffer.write().await;
        buffer.push(metric);
        Ok(())
    }
    
    async fn get_aggregated_metrics(&self, _pattern: &str) -> Result<Vec<AggregatedMetric>> {
        let cache = self.aggregated_cache.read().await;
        Ok(cache.values().cloned().collect())
    }
}

impl TimeSeriesDB {
    fn new() -> Self {
        Self {
            data_points: HashMap::new(),
            metadata: HashMap::new(),
            retention_policies: HashMap::new(),
            index: HashMap::new(),
        }
    }
}

impl MetricsAggregator {
    fn new() -> Result<Self> {
        let (results_tx, results_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            active_aggregations: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(RwLock::new(Vec::new())),
            results_tx,
            results_rx: Arc::new(RwLock::new(Some(results_rx))),
        })
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting metrics aggregator");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping metrics aggregator");
        Ok(())
    }
    
    async fn process_metric(&self, _metric: RawMetric) -> Result<()> {
        // Implementation would process the metric according to aggregation rules
        Ok(())
    }
}

impl PerformanceMonitor {
    fn new() -> Result<Self> {
        Ok(Self {
            system_collector: Arc::new(SystemMetricsCollector::new()?),
            app_collector: Arc::new(ApplicationMetricsCollector::new()?),
            profiles: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(MonitoringState::default())),
        })
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting performance monitor");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping performance monitor");
        Ok(())
    }
}

impl SystemMetricsCollector {
    fn new() -> Result<Self> {
        Ok(Self {
            cpu_metrics: Arc::new(RwLock::new(CpuMetrics::default())),
            memory_metrics: Arc::new(RwLock::new(MemoryMetrics::default())),
            network_metrics: Arc::new(RwLock::new(NetworkMetrics::default())),
            disk_metrics: Arc::new(RwLock::new(DiskMetrics::default())),
            collection_interval: Duration::from_secs(10),
        })
    }
}

impl ApplicationMetricsCollector {
    fn new() -> Result<Self> {
        Ok(Self {
            hive_metrics: Arc::new(RwLock::new(HiveMindMetrics::default())),
            trading_metrics: Arc::new(RwLock::new(TradingMetrics::default())),
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

impl AlertManager {
    fn new() -> Result<Self> {
        Ok(Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            state: Arc::new(RwLock::new(AlertManagerState::default())),
        })
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting alert manager");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping alert manager");
        Ok(())
    }
}

// Default implementations
impl Default for CollectionState {
    fn default() -> Self {
        Self {
            enabled: false,
            started_at: SystemTime::now(),
            last_collection: SystemTime::now(),
            metrics_collected: 0,
            collection_errors: 0,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            flush_interval: Duration::from_secs(60),
            batch_size: 1000,
            enable_persistence: false,
            persistence_dir: None,
        }
    }
}

impl Default for MonitoringState {
    fn default() -> Self {
        Self {
            enabled: false,
            collection_interval: Duration::from_secs(10),
            last_collection: SystemTime::now(),
            metrics_collected: 0,
            anomalies_detected: 0,
            alerts_sent: 0,
        }
    }
}

impl Default for AlertManagerState {
    fn default() -> Self {
        Self {
            enabled: false,
            active_alerts_count: 0,
            rules_evaluated: 0,
            notifications_sent: 0,
            last_evaluation: SystemTime::now(),
        }
    }
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            load_average: [0.0, 0.0, 0.0],
            context_switches: 0,
            interrupts: 0,
            frequency: 0.0,
            core_count: num_cpus::get(),
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_memory: 0,
            used_memory: 0,
            available_memory: 0,
            utilization: 0.0,
            swap_total: 0,
            swap_used: 0,
            page_faults: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_received_per_sec: 0,
            bytes_sent_per_sec: 0,
            packets_received_per_sec: 0,
            packets_sent_per_sec: 0,
            errors_per_sec: 0,
            drops_per_sec: 0,
            active_connections: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for DiskMetrics {
    fn default() -> Self {
        Self {
            read_bytes_per_sec: 0,
            write_bytes_per_sec: 0,
            read_ops_per_sec: 0,
            write_ops_per_sec: 0,
            utilization: 0.0,
            avg_response_time: 0.0,
            queue_depth: 0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for HiveMindMetrics {
    fn default() -> Self {
        Self {
            active_agents: 0,
            consensus_ops_per_sec: 0.0,
            memory_ops_per_sec: 0.0,
            neural_ops_per_sec: 0.0,
            network_msgs_per_sec: 0.0,
            coordination_efficiency: 0.0,
            intelligence_score: 0.0,
            resilience_score: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            orders_per_sec: 0.0,
            trades_per_sec: 0.0,
            avg_order_latency: 0.0,
            market_data_updates_per_sec: 0.0,
            risk_checks_per_sec: 0.0,
            portfolio_value: 0.0,
            pnl: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_types() {
        let counter = MetricValue::Counter(100);
        let gauge = MetricValue::Gauge(75.5);
        
        match counter {
            MetricValue::Counter(value) => assert_eq!(value, 100),
            _ => panic!("Expected counter"),
        }
        
        match gauge {
            MetricValue::Gauge(value) => assert_eq!(value, 75.5),
            _ => panic!("Expected gauge"),
        }
    }
    
    #[test]
    fn test_performance_state_ordering() {
        assert!(PerformanceState::Failed > PerformanceState::Critical);
        assert!(PerformanceState::Critical > PerformanceState::Warning);
        assert!(PerformanceState::Warning > PerformanceState::Optimal);
    }
    
    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }
    
    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(&config);
        assert!(collector.is_ok());
    }
}