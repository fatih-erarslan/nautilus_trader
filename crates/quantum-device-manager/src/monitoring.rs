//! Monitoring system for quantum devices and hive coordination

use crate::*;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

/// Monitoring system trait
#[async_trait]
pub trait MonitoringSystem: Send + Sync {
    /// Initialize monitoring system
    async fn initialize(&self) -> Result<()>;
    
    /// Start monitoring
    async fn start_monitoring(&self) -> Result<()>;
    
    /// Stop monitoring
    async fn stop_monitoring(&self) -> Result<()>;
    
    /// Record device metrics
    async fn record_device_metrics(&self, device_id: Uuid, metrics: &DeviceMetrics) -> Result<()>;
    
    /// Record task execution
    async fn record_task_execution(&self, task_id: Uuid, result: &QuantumResult) -> Result<()>;
    
    /// Get performance report
    async fn get_performance_report(&self) -> Result<PerformanceReport>;
    
    /// Get hive coordination metrics
    async fn get_hive_metrics(&self) -> Result<HiveMetrics>;
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report timestamp
    pub timestamp: DateTime<Utc>,
    /// Device performance
    pub device_performance: Vec<DevicePerformance>,
    /// System performance
    pub system_performance: SystemPerformance,
    /// Hive coordination metrics
    pub hive_metrics: HiveMetrics,
    /// Quantum advantage analysis
    pub quantum_advantage: QuantumAdvantageAnalysis,
    /// Recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Device performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePerformance {
    /// Device ID
    pub device_id: Uuid,
    /// Device name
    pub device_name: String,
    /// Device type
    pub device_type: QuantumDeviceType,
    /// Utilization percentage
    pub utilization: f64,
    /// Throughput (tasks per second)
    pub throughput: f64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Success rate
    pub success_rate: f64,
    /// Error rate
    pub error_rate: f64,
    /// Quantum advantage achieved
    pub quantum_advantage: f64,
    /// Health score
    pub health_score: f64,
    /// Performance trend
    pub trend: PerformanceTrend,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformance {
    /// Total throughput
    pub total_throughput: f64,
    /// Average system latency
    pub avg_system_latency_us: f64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Resource utilization
    pub resource_utilization: f64,
    /// Load balancing efficiency
    pub load_balancing_efficiency: f64,
    /// Fault tolerance score
    pub fault_tolerance_score: f64,
    /// Scalability index
    pub scalability_index: f64,
}

/// Hive coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveMetrics {
    /// Hive synchronization score
    pub sync_score: f64,
    /// Nash solver utilization
    pub nash_solver_utilization: f64,
    /// Decision coherence
    pub decision_coherence: f64,
    /// Collective intelligence score
    pub collective_intelligence: f64,
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Swarm efficiency
    pub swarm_efficiency: f64,
    /// Memory coordination
    pub memory_coordination: f64,
    /// Trading decision accuracy
    pub trading_decision_accuracy: f64,
}

/// Quantum advantage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageAnalysis {
    /// Average quantum advantage
    pub avg_quantum_advantage: f64,
    /// Maximum advantage achieved
    pub max_quantum_advantage: f64,
    /// Consistency score
    pub consistency_score: f64,
    /// Classical comparison
    pub classical_comparison: ClassicalComparison,
    /// Advantage by task type
    pub advantage_by_task_type: HashMap<String, f64>,
    /// Advantage trends
    pub advantage_trends: Vec<AdvantageDataPoint>,
}

/// Classical comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalComparison {
    /// Average classical execution time
    pub avg_classical_time_us: f64,
    /// Average quantum execution time
    pub avg_quantum_time_us: f64,
    /// Speedup factor
    pub speedup_factor: f64,
    /// Accuracy comparison
    pub accuracy_comparison: f64,
    /// Resource usage comparison
    pub resource_usage_comparison: f64,
}

/// Advantage data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Quantum advantage
    pub advantage: f64,
    /// Task type
    pub task_type: String,
    /// Device type
    pub device_type: QuantumDeviceType,
}

/// Performance trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Improving performance
    Improving,
    /// Stable performance
    Stable,
    /// Declining performance
    Declining,
    /// Volatile performance
    Volatile,
    /// Unknown trend
    Unknown,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Target device (if applicable)
    pub target_device: Option<Uuid>,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: String,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Recommendation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Scale up devices
    ScaleUp,
    /// Scale down devices
    ScaleDown,
    /// Optimize load balancing
    OptimizeLoadBalancing,
    /// Improve error correction
    ImproveErrorCorrection,
    /// Enhance Nash solver
    EnhanceNashSolver,
    /// Adjust trading parameters
    AdjustTradingParameters,
    /// Upgrade hardware
    UpgradeHardware,
    /// Recalibrate devices
    RecalibrateDevices,
    /// Optimize circuits
    OptimizeCircuits,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Critical (immediate action required)
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
}

/// Monitoring system implementation
pub struct MonitoringSystemImpl {
    /// Device metrics storage
    device_metrics: Arc<RwLock<HashMap<Uuid, VecDeque<TimestampedMetrics>>>>,
    /// Task execution records
    task_records: Arc<RwLock<VecDeque<TaskRecord>>>,
    /// Performance history
    performance_history: Arc<RwLock<VecDeque<PerformanceReport>>>,
    /// Hive coordination tracker
    hive_tracker: Arc<RwLock<HiveCoordinationTracker>>,
    /// Running flag
    running: Arc<RwLock<bool>>,
    /// Configuration
    config: MonitoringConfig,
}

/// Timestamped metrics
#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Device metrics
    pub metrics: DeviceMetrics,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Task execution record
#[derive(Debug, Clone)]
pub struct TaskRecord {
    /// Task ID
    pub task_id: Uuid,
    /// Device ID
    pub device_id: Uuid,
    /// Task type
    pub task_type: String,
    /// Execution result
    pub result: QuantumResult,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Trading context
    pub trading_context: Option<TradingContext>,
}

/// Hive coordination tracker
#[derive(Debug, Clone)]
pub struct HiveCoordinationTracker {
    /// Nash solver metrics
    pub nash_solver_metrics: NashSolverMetrics,
    /// Decision coherence history
    pub decision_coherence: VecDeque<DecisionCoherencePoint>,
    /// Collective intelligence measurements
    pub collective_intelligence: VecDeque<CollectiveIntelligencePoint>,
    /// Memory coordination metrics
    pub memory_coordination: VecDeque<MemoryCoordinationPoint>,
}

/// Nash solver metrics
#[derive(Debug, Clone)]
pub struct NashSolverMetrics {
    /// Total Nash problems solved
    pub total_problems_solved: u64,
    /// Average convergence time
    pub avg_convergence_time_us: f64,
    /// Average convergence error
    pub avg_convergence_error: f64,
    /// Success rate
    pub success_rate: f64,
    /// Quantum advantage in Nash solving
    pub quantum_advantage: f64,
}

/// Decision coherence point
#[derive(Debug, Clone)]
pub struct DecisionCoherencePoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Coherence score (0.0 to 1.0)
    pub coherence_score: f64,
    /// Number of agents involved
    pub agent_count: u32,
    /// Decision type
    pub decision_type: String,
}

/// Collective intelligence point
#[derive(Debug, Clone)]
pub struct CollectiveIntelligencePoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Intelligence score
    pub intelligence_score: f64,
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Problem complexity
    pub problem_complexity: f64,
}

/// Memory coordination point
#[derive(Debug, Clone)]
pub struct MemoryCoordinationPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Coordination efficiency
    pub coordination_efficiency: f64,
    /// Memory synchronization score
    pub sync_score: f64,
    /// Knowledge sharing rate
    pub knowledge_sharing_rate: f64,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub collection_interval_secs: u64,
    /// Performance report interval
    pub report_interval_secs: u64,
    /// History retention period
    pub history_retention_days: u32,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Enable real-time alerts
    pub real_time_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// High error rate threshold
    pub high_error_rate: f64,
    /// Low success rate threshold
    pub low_success_rate: f64,
    /// High latency threshold (microseconds)
    pub high_latency_us: f64,
    /// Low quantum advantage threshold
    pub low_quantum_advantage: f64,
    /// Low coherence threshold
    pub low_coherence: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_secs: 5,
            report_interval_secs: 60,
            history_retention_days: 30,
            detailed_logging: true,
            real_time_alerts: true,
            alert_thresholds: AlertThresholds {
                high_error_rate: 0.1,
                low_success_rate: 0.9,
                high_latency_us: 10000.0,
                low_quantum_advantage: 1.0,
                low_coherence: 0.7,
            },
        }
    }
}

impl MonitoringSystemImpl {
    /// Create new monitoring system
    pub fn new() -> Result<Self> {
        let config = MonitoringConfig::default();
        
        let hive_tracker = HiveCoordinationTracker {
            nash_solver_metrics: NashSolverMetrics {
                total_problems_solved: 0,
                avg_convergence_time_us: 0.0,
                avg_convergence_error: 0.0,
                success_rate: 1.0,
                quantum_advantage: 1.0,
            },
            decision_coherence: VecDeque::new(),
            collective_intelligence: VecDeque::new(),
            memory_coordination: VecDeque::new(),
        };
        
        Ok(Self {
            device_metrics: Arc::new(RwLock::new(HashMap::new())),
            task_records: Arc::new(RwLock::new(VecDeque::new())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            hive_tracker: Arc::new(RwLock::new(hive_tracker)),
            running: Arc::new(RwLock::new(false)),
            config,
        })
    }
    
    /// Generate performance report
    async fn generate_performance_report(&self) -> Result<PerformanceReport> {
        let timestamp = Utc::now();
        
        // Collect device performance
        let device_performance = self.collect_device_performance().await?;
        
        // Calculate system performance
        let system_performance = self.calculate_system_performance(&device_performance).await?;
        
        // Get hive metrics
        let hive_metrics = self.calculate_hive_metrics().await?;
        
        // Analyze quantum advantage
        let quantum_advantage = self.analyze_quantum_advantage().await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &device_performance,
            &system_performance,
            &hive_metrics,
            &quantum_advantage,
        ).await?;
        
        Ok(PerformanceReport {
            timestamp,
            device_performance,
            system_performance,
            hive_metrics,
            quantum_advantage,
            recommendations,
        })
    }
    
    /// Collect device performance metrics
    async fn collect_device_performance(&self) -> Result<Vec<DevicePerformance>> {
        let device_metrics = self.device_metrics.read().await;
        let mut performance = Vec::new();
        
        for (device_id, metrics_history) in device_metrics.iter() {
            if let Some(latest_metrics) = metrics_history.back() {
                let device_performance = DevicePerformance {
                    device_id: *device_id,
                    device_name: format!("Device-{}", device_id),
                    device_type: QuantumDeviceType::Simulator, // Default
                    utilization: self.calculate_utilization(&latest_metrics.metrics),
                    throughput: self.calculate_throughput(metrics_history),
                    avg_latency_us: latest_metrics.metrics.avg_execution_time_us,
                    success_rate: latest_metrics.metrics.success_rate,
                    error_rate: 1.0 - latest_metrics.metrics.success_rate,
                    quantum_advantage: latest_metrics.metrics.quantum_advantage,
                    health_score: self.calculate_health_score(&latest_metrics.metrics),
                    trend: self.calculate_trend(metrics_history),
                };
                
                performance.push(device_performance);
            }
        }
        
        Ok(performance)
    }
    
    /// Calculate system performance
    async fn calculate_system_performance(&self, device_performance: &[DevicePerformance]) -> Result<SystemPerformance> {
        let total_throughput = device_performance.iter().map(|d| d.throughput).sum();
        let avg_system_latency_us = device_performance.iter()
            .map(|d| d.avg_latency_us)
            .sum::<f64>() / device_performance.len() as f64;
        
        let overall_success_rate = device_performance.iter()
            .map(|d| d.success_rate)
            .sum::<f64>() / device_performance.len() as f64;
        
        let resource_utilization = device_performance.iter()
            .map(|d| d.utilization)
            .sum::<f64>() / device_performance.len() as f64;
        
        Ok(SystemPerformance {
            total_throughput,
            avg_system_latency_us,
            overall_success_rate,
            resource_utilization,
            load_balancing_efficiency: 0.85, // Mock value
            fault_tolerance_score: 0.9, // Mock value
            scalability_index: 0.8, // Mock value
        })
    }
    
    /// Calculate hive metrics
    async fn calculate_hive_metrics(&self) -> Result<HiveMetrics> {
        let hive_tracker = self.hive_tracker.read().await;
        
        let sync_score = if let Some(latest_coherence) = hive_tracker.decision_coherence.back() {
            latest_coherence.coherence_score
        } else {
            0.8 // Default
        };
        
        let nash_solver_utilization = hive_tracker.nash_solver_metrics.success_rate;
        
        let decision_coherence = hive_tracker.decision_coherence.iter()
            .map(|d| d.coherence_score)
            .sum::<f64>() / hive_tracker.decision_coherence.len() as f64;
        
        let collective_intelligence = hive_tracker.collective_intelligence.iter()
            .map(|c| c.intelligence_score)
            .sum::<f64>() / hive_tracker.collective_intelligence.len() as f64;
        
        let adaptation_rate = hive_tracker.collective_intelligence.iter()
            .map(|c| c.adaptation_rate)
            .sum::<f64>() / hive_tracker.collective_intelligence.len() as f64;
        
        let memory_coordination = hive_tracker.memory_coordination.iter()
            .map(|m| m.coordination_efficiency)
            .sum::<f64>() / hive_tracker.memory_coordination.len() as f64;
        
        Ok(HiveMetrics {
            sync_score,
            nash_solver_utilization,
            decision_coherence,
            collective_intelligence,
            adaptation_rate,
            swarm_efficiency: 0.85, // Mock value
            memory_coordination,
            trading_decision_accuracy: 0.75, // Mock value
        })
    }
    
    /// Analyze quantum advantage
    async fn analyze_quantum_advantage(&self) -> Result<QuantumAdvantageAnalysis> {
        let task_records = self.task_records.read().await;
        
        let advantages: Vec<f64> = task_records.iter()
            .map(|record| record.result.quantum_advantage)
            .collect();
        
        let avg_quantum_advantage = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let max_quantum_advantage = advantages.iter().cloned().fold(0.0, f64::max);
        
        // Calculate consistency score
        let variance = advantages.iter()
            .map(|&x| (x - avg_quantum_advantage).powi(2))
            .sum::<f64>() / advantages.len() as f64;
        let consistency_score = 1.0 / (1.0 + variance.sqrt());
        
        // Mock classical comparison
        let classical_comparison = ClassicalComparison {
            avg_classical_time_us: 5000.0,
            avg_quantum_time_us: 500.0,
            speedup_factor: 10.0,
            accuracy_comparison: 1.2,
            resource_usage_comparison: 0.8,
        };
        
        // Advantage by task type
        let mut advantage_by_task_type = HashMap::new();
        advantage_by_task_type.insert("nash_solver".to_string(), avg_quantum_advantage * 1.5);
        advantage_by_task_type.insert("optimization".to_string(), avg_quantum_advantage * 1.2);
        advantage_by_task_type.insert("simulation".to_string(), avg_quantum_advantage);
        
        // Advantage trends
        let advantage_trends = task_records.iter()
            .map(|record| AdvantageDataPoint {
                timestamp: record.timestamp,
                advantage: record.result.quantum_advantage,
                task_type: record.task_type.clone(),
                device_type: QuantumDeviceType::Simulator, // Default
            })
            .collect();
        
        Ok(QuantumAdvantageAnalysis {
            avg_quantum_advantage,
            max_quantum_advantage,
            consistency_score,
            classical_comparison,
            advantage_by_task_type,
            advantage_trends,
        })
    }
    
    /// Generate performance recommendations
    async fn generate_recommendations(
        &self,
        device_performance: &[DevicePerformance],
        system_performance: &SystemPerformance,
        hive_metrics: &HiveMetrics,
        quantum_advantage: &QuantumAdvantageAnalysis,
    ) -> Result<Vec<PerformanceRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Check for low performance devices
        for device in device_performance {
            if device.success_rate < self.config.alert_thresholds.low_success_rate {
                recommendations.push(PerformanceRecommendation {
                    recommendation_type: RecommendationType::RecalibrateDevices,
                    target_device: Some(device.device_id),
                    priority: RecommendationPriority::High,
                    description: format!(
                        "Device {} has low success rate ({:.2}%)",
                        device.device_name, device.success_rate * 100.0
                    ),
                    expected_impact: "Improve success rate by 10-15%".to_string(),
                    implementation_steps: vec![
                        "Run device diagnostics".to_string(),
                        "Recalibrate quantum gates".to_string(),
                        "Update error correction parameters".to_string(),
                    ],
                });
            }
            
            if device.quantum_advantage < self.config.alert_thresholds.low_quantum_advantage {
                recommendations.push(PerformanceRecommendation {
                    recommendation_type: RecommendationType::OptimizeCircuits,
                    target_device: Some(device.device_id),
                    priority: RecommendationPriority::Medium,
                    description: format!(
                        "Device {} has low quantum advantage ({:.2}x)",
                        device.device_name, device.quantum_advantage
                    ),
                    expected_impact: "Increase quantum advantage by 20-30%".to_string(),
                    implementation_steps: vec![
                        "Analyze circuit complexity".to_string(),
                        "Optimize gate sequences".to_string(),
                        "Implement better error correction".to_string(),
                    ],
                });
            }
        }
        
        // Check system-wide issues
        if system_performance.resource_utilization > 0.8 {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::ScaleUp,
                target_device: None,
                priority: RecommendationPriority::High,
                description: format!(
                    "High resource utilization ({:.1}%)",
                    system_performance.resource_utilization * 100.0
                ),
                expected_impact: "Reduce latency by 25-40%".to_string(),
                implementation_steps: vec![
                    "Add more quantum devices".to_string(),
                    "Optimize load balancing".to_string(),
                    "Implement task prioritization".to_string(),
                ],
            });
        }
        
        // Check hive coordination
        if hive_metrics.decision_coherence < self.config.alert_thresholds.low_coherence {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::EnhanceNashSolver,
                target_device: None,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "Low decision coherence ({:.2})",
                    hive_metrics.decision_coherence
                ),
                expected_impact: "Improve trading decision accuracy by 15-25%".to_string(),
                implementation_steps: vec![
                    "Enhance Nash solver algorithms".to_string(),
                    "Improve agent coordination".to_string(),
                    "Optimize memory synchronization".to_string(),
                ],
            });
        }
        
        Ok(recommendations)
    }
    
    /// Calculate device utilization
    fn calculate_utilization(&self, metrics: &DeviceMetrics) -> f64 {
        // Mock calculation based on tasks completed
        let total_tasks = metrics.tasks_completed + metrics.tasks_failed;
        if total_tasks > 0 {
            (metrics.tasks_completed as f64 / total_tasks as f64) * 0.8
        } else {
            0.0
        }
    }
    
    /// Calculate throughput
    fn calculate_throughput(&self, metrics_history: &VecDeque<TimestampedMetrics>) -> f64 {
        if metrics_history.len() < 2 {
            return 0.0;
        }
        
        let latest = metrics_history.back().unwrap();
        let previous = metrics_history.get(metrics_history.len() - 2).unwrap();
        
        let time_diff = (latest.timestamp - previous.timestamp).num_seconds() as f64;
        let task_diff = latest.metrics.tasks_completed as f64 - previous.metrics.tasks_completed as f64;
        
        if time_diff > 0.0 {
            task_diff / time_diff
        } else {
            0.0
        }
    }
    
    /// Calculate health score
    fn calculate_health_score(&self, metrics: &DeviceMetrics) -> f64 {
        let success_weight = 0.4;
        let quantum_advantage_weight = 0.3;
        let uptime_weight = 0.3;
        
        let success_score = metrics.success_rate;
        let quantum_score = (metrics.quantum_advantage / 2.0).min(1.0);
        let uptime_score = metrics.uptime_percentage / 100.0;
        
        success_score * success_weight +
        quantum_score * quantum_advantage_weight +
        uptime_score * uptime_weight
    }
    
    /// Calculate performance trend
    fn calculate_trend(&self, metrics_history: &VecDeque<TimestampedMetrics>) -> PerformanceTrend {
        if metrics_history.len() < 3 {
            return PerformanceTrend::Unknown;
        }
        
        let recent_scores: Vec<f64> = metrics_history.iter()
            .rev()
            .take(3)
            .map(|m| self.calculate_health_score(&m.metrics))
            .collect();
        
        let trend_score = recent_scores[0] - recent_scores[2];
        
        if trend_score > 0.1 {
            PerformanceTrend::Improving
        } else if trend_score < -0.1 {
            PerformanceTrend::Declining
        } else {
            let volatility = recent_scores.iter()
                .map(|&x| (x - recent_scores.iter().sum::<f64>() / recent_scores.len() as f64).abs())
                .sum::<f64>() / recent_scores.len() as f64;
            
            if volatility > 0.2 {
                PerformanceTrend::Volatile
            } else {
                PerformanceTrend::Stable
            }
        }
    }
}

#[async_trait]
impl MonitoringSystem for MonitoringSystemImpl {
    async fn initialize(&self) -> Result<()> {
        info!("Initializing quantum device monitoring system");
        Ok(())
    }
    
    async fn start_monitoring(&self) -> Result<()> {
        info!("Starting quantum device monitoring");
        
        *self.running.write().await = true;
        
        // Start periodic report generation
        let performance_history = self.performance_history.clone();
        let running = self.running.clone();
        let report_interval = Duration::from_secs(self.config.report_interval_secs);
        
        let self_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            let mut interval = interval(report_interval);
            
            while *running.read().await {
                interval.tick().await;
                
                if let Ok(report) = self_clone.generate_performance_report().await {
                    let mut history = performance_history.write().await;
                    history.push_back(report);
                    
                    // Limit history size
                    if history.len() > 100 {
                        history.pop_front();
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn stop_monitoring(&self) -> Result<()> {
        info!("Stopping quantum device monitoring");
        *self.running.write().await = false;
        Ok(())
    }
    
    async fn record_device_metrics(&self, device_id: Uuid, metrics: &DeviceMetrics) -> Result<()> {
        let timestamped_metrics = TimestampedMetrics {
            timestamp: Utc::now(),
            metrics: metrics.clone(),
            context: HashMap::new(),
        };
        
        let mut device_metrics = self.device_metrics.write().await;
        let device_history = device_metrics.entry(device_id).or_insert_with(VecDeque::new);
        
        device_history.push_back(timestamped_metrics);
        
        // Limit history size
        if device_history.len() > 1000 {
            device_history.pop_front();
        }
        
        Ok(())
    }
    
    async fn record_task_execution(&self, task_id: Uuid, result: &QuantumResult) -> Result<()> {
        let task_record = TaskRecord {
            task_id,
            device_id: result.device_id,
            task_type: "quantum_computation".to_string(),
            result: result.clone(),
            timestamp: Utc::now(),
            trading_context: None,
        };
        
        let mut task_records = self.task_records.write().await;
        task_records.push_back(task_record);
        
        // Limit history size
        if task_records.len() > 10000 {
            task_records.pop_front();
        }
        
        Ok(())
    }
    
    async fn get_performance_report(&self) -> Result<PerformanceReport> {
        self.generate_performance_report().await
    }
    
    async fn get_hive_metrics(&self) -> Result<HiveMetrics> {
        self.calculate_hive_metrics().await
    }
}

impl Clone for MonitoringSystemImpl {
    fn clone(&self) -> Self {
        Self {
            device_metrics: self.device_metrics.clone(),
            task_records: self.task_records.clone(),
            performance_history: self.performance_history.clone(),
            hive_tracker: self.hive_tracker.clone(),
            running: self.running.clone(),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monitoring_system_creation() {
        let monitoring = MonitoringSystemImpl::new().unwrap();
        monitoring.initialize().await.unwrap();
        
        // Test device metrics recording
        let device_id = Uuid::new_v4();
        let metrics = DeviceMetrics {
            tasks_completed: 100,
            tasks_failed: 5,
            avg_execution_time_us: 1000.0,
            success_rate: 0.95,
            quantum_advantage: 2.0,
            uptime_percentage: 99.5,
            error_correction_overhead: 0.1,
        };
        
        monitoring.record_device_metrics(device_id, &metrics).await.unwrap();
        
        // Test task execution recording
        let task_id = Uuid::new_v4();
        let result = QuantumResult {
            task_id,
            device_id,
            execution_time_us: 1000,
            result: vec![0.5, 0.3, 0.2],
            success: true,
            error: None,
            quantum_advantage: 2.0,
            fidelity: 0.99,
            trading_decision: None,
        };
        
        monitoring.record_task_execution(task_id, &result).await.unwrap();
        
        // Test performance report generation
        let report = monitoring.get_performance_report().await.unwrap();
        assert!(!report.device_performance.is_empty());
        assert!(report.system_performance.total_throughput >= 0.0);
    }
}