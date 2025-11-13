//! Workload analysis for auto-scaling decisions
//!
//! This module analyzes computational load in real-time to determine
//! optimal scaling configurations from 48 nodes to 1 billion nodes.

use hyperphysics_core::Result;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Workload analyzer for auto-scaling decisions
pub struct WorkloadAnalyzer {
    metrics_history: VecDeque<WorkloadMetrics>,
    max_history_size: usize,
    analysis_window: Duration,
    last_analysis: Instant,
}

/// Current workload metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetrics {
    pub timestamp: std::time::SystemTime,
    pub cpu_usage: f64,           // 0.0 to 1.0
    pub memory_usage: f64,        // 0.0 to 1.0
    pub gpu_usage: f64,           // 0.0 to 1.0
    pub gpu_memory_usage: f64,    // 0.0 to 1.0
    pub active_nodes: usize,      // Current number of nodes
    pub computation_rate: f64,    // Operations per second
    pub phi_calculations: u64,    // Φ calculations per second
    pub energy_calculations: u64, // Energy calculations per second
    pub network_io: f64,          // Network I/O in MB/s
    pub disk_io: f64,             // Disk I/O in MB/s
}

/// Workload analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    pub current_load: LoadLevel,
    pub predicted_load: LoadLevel,
    pub bottleneck: Option<BottleneckType>,
    pub scaling_recommendation: ScalingDecision,
    pub confidence: f64,          // 0.0 to 1.0
    pub analysis_timestamp: std::time::SystemTime,
}

/// Load level classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadLevel {
    Idle,        // < 20% resource usage
    Low,         // 20-40% resource usage
    Medium,      // 40-70% resource usage
    High,        // 70-90% resource usage
    Critical,    // > 90% resource usage
}

/// System bottleneck identification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    GPU,
    GPUMemory,
    NetworkIO,
    DiskIO,
    None,
}

/// Scaling decision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalingDecision {
    ScaleDown(ScalingTarget),
    Maintain,
    ScaleUp(ScalingTarget),
    Emergency(ScalingTarget),
}

/// Scaling target configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalingTarget {
    Micro48,        // 48 nodes (ROI)
    Small16K,       // 16,384 nodes (128x128)
    Medium1M,       // 1,048,576 nodes (1024x1024)
    Large1B,        // 1,000,000,000 nodes (distributed)
}

impl WorkloadAnalyzer {
    /// Create new workload analyzer
    pub fn new() -> Self {
        Self {
            metrics_history: VecDeque::new(),
            max_history_size: 1000, // Keep last 1000 measurements
            analysis_window: Duration::from_secs(60), // 1-minute analysis window
            last_analysis: Instant::now(),
        }
    }
    
    /// Record current workload metrics
    pub fn record_metrics(&mut self, metrics: WorkloadMetrics) {
        self.metrics_history.push_back(metrics);
        
        // Maintain history size limit
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }
    }
    
    /// Analyze current workload and provide scaling recommendation
    pub fn analyze_workload(&mut self) -> Result<WorkloadAnalysis> {
        let now = Instant::now();
        
        // Only analyze if enough time has passed
        if now.duration_since(self.last_analysis) < Duration::from_secs(10) {
            return Err(hyperphysics_core::Error::InvalidArgument(
                "Analysis too frequent - wait at least 10 seconds".to_string()
            ));
        }
        
        self.last_analysis = now;
        
        if self.metrics_history.is_empty() {
            return Err(hyperphysics_core::Error::InvalidArgument(
                "No metrics available for analysis".to_string()
            ));
        }
        
        // Get recent metrics within analysis window
        let cutoff_time = std::time::SystemTime::now() - self.analysis_window;
        let recent_metrics: Vec<&WorkloadMetrics> = self.metrics_history
            .iter()
            .filter(|m| m.timestamp > cutoff_time)
            .collect();
        
        if recent_metrics.is_empty() {
            return Err(hyperphysics_core::Error::InvalidArgument(
                "No recent metrics available".to_string()
            ));
        }
        
        // Analyze current load
        let current_load = self.classify_load_level(&recent_metrics);
        
        // Predict future load based on trends
        let predicted_load = self.predict_load_trend(&recent_metrics);
        
        // Identify bottlenecks
        let bottleneck = self.identify_bottleneck(&recent_metrics);
        
        // Make scaling recommendation
        let scaling_recommendation = self.recommend_scaling(
            &current_load,
            &predicted_load,
            &bottleneck,
            recent_metrics.last().unwrap().active_nodes
        );
        
        // Calculate confidence based on data quality and consistency
        let confidence = self.calculate_confidence(&recent_metrics);
        
        Ok(WorkloadAnalysis {
            current_load,
            predicted_load,
            bottleneck,
            scaling_recommendation,
            confidence,
            analysis_timestamp: std::time::SystemTime::now(),
        })
    }
    
    /// Classify current load level
    fn classify_load_level(&self, metrics: &[&WorkloadMetrics]) -> LoadLevel {
        if metrics.is_empty() {
            return LoadLevel::Idle;
        }
        
        // Calculate average resource usage
        let avg_cpu = metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / metrics.len() as f64;
        let avg_memory = metrics.iter().map(|m| m.memory_usage).sum::<f64>() / metrics.len() as f64;
        let avg_gpu = metrics.iter().map(|m| m.gpu_usage).sum::<f64>() / metrics.len() as f64;
        
        // Use maximum resource usage as primary indicator
        let max_usage = avg_cpu.max(avg_memory).max(avg_gpu);
        
        match max_usage {
            x if x < 0.2 => LoadLevel::Idle,
            x if x < 0.4 => LoadLevel::Low,
            x if x < 0.7 => LoadLevel::Medium,
            x if x < 0.9 => LoadLevel::High,
            _ => LoadLevel::Critical,
        }
    }
    
    /// Predict future load based on recent trends
    fn predict_load_trend(&self, metrics: &[&WorkloadMetrics]) -> LoadLevel {
        if metrics.len() < 3 {
            return self.classify_load_level(metrics);
        }
        
        // Simple linear trend analysis
        let recent_half = &metrics[metrics.len()/2..];
        let earlier_half = &metrics[..metrics.len()/2];
        
        let recent_avg = self.average_resource_usage(recent_half);
        let earlier_avg = self.average_resource_usage(earlier_half);
        
        // Predict based on trend
        let trend = recent_avg - earlier_avg;
        let predicted_usage = recent_avg + trend * 2.0; // Extrapolate trend
        
        match predicted_usage {
            x if x < 0.2 => LoadLevel::Idle,
            x if x < 0.4 => LoadLevel::Low,
            x if x < 0.7 => LoadLevel::Medium,
            x if x < 0.9 => LoadLevel::High,
            _ => LoadLevel::Critical,
        }
    }
    
    /// Calculate average resource usage
    fn average_resource_usage(&self, metrics: &[&WorkloadMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }
        
        let total_usage: f64 = metrics.iter()
            .map(|m| m.cpu_usage.max(m.memory_usage).max(m.gpu_usage))
            .sum();
        
        total_usage / metrics.len() as f64
    }
    
    /// Identify system bottlenecks
    fn identify_bottleneck(&self, metrics: &[&WorkloadMetrics]) -> Option<BottleneckType> {
        if metrics.is_empty() {
            return None;
        }
        
        let latest = metrics.last().unwrap();
        
        // Identify the highest resource usage
        let mut max_usage = 0.0;
        let mut bottleneck = BottleneckType::None;
        
        if latest.cpu_usage > max_usage {
            max_usage = latest.cpu_usage;
            bottleneck = BottleneckType::CPU;
        }
        
        if latest.memory_usage > max_usage {
            max_usage = latest.memory_usage;
            bottleneck = BottleneckType::Memory;
        }
        
        if latest.gpu_usage > max_usage {
            max_usage = latest.gpu_usage;
            bottleneck = BottleneckType::GPU;
        }
        
        if latest.gpu_memory_usage > max_usage {
            max_usage = latest.gpu_memory_usage;
            bottleneck = BottleneckType::GPUMemory;
        }
        
        // Only consider it a bottleneck if usage > 80%
        if max_usage > 0.8 {
            Some(bottleneck)
        } else {
            None
        }
    }
    
    /// Recommend scaling action
    fn recommend_scaling(
        &self,
        current_load: &LoadLevel,
        predicted_load: &LoadLevel,
        bottleneck: &Option<BottleneckType>,
        current_nodes: usize,
    ) -> ScalingDecision {
        // Emergency scaling for critical loads
        if *current_load == LoadLevel::Critical || *predicted_load == LoadLevel::Critical {
            return ScalingDecision::Emergency(self.select_emergency_target(current_nodes));
        }
        
        // Scale up decisions
        if *current_load == LoadLevel::High || *predicted_load == LoadLevel::High {
            return ScalingDecision::ScaleUp(self.select_scale_up_target(current_nodes));
        }
        
        // Scale down decisions
        if *current_load == LoadLevel::Idle && *predicted_load == LoadLevel::Idle {
            if let Some(target) = self.select_scale_down_target(current_nodes) {
                return ScalingDecision::ScaleDown(target);
            }
        }
        
        // Default: maintain current configuration
        ScalingDecision::Maintain
    }
    
    /// Select emergency scaling target
    fn select_emergency_target(&self, current_nodes: usize) -> ScalingTarget {
        match current_nodes {
            n if n <= 48 => ScalingTarget::Small16K,
            n if n <= 16_384 => ScalingTarget::Medium1M,
            n if n <= 1_048_576 => ScalingTarget::Large1B,
            _ => ScalingTarget::Large1B, // Already at maximum
        }
    }
    
    /// Select scale-up target
    fn select_scale_up_target(&self, current_nodes: usize) -> ScalingTarget {
        match current_nodes {
            n if n <= 48 => ScalingTarget::Small16K,
            n if n <= 16_384 => ScalingTarget::Medium1M,
            n if n <= 1_048_576 => ScalingTarget::Large1B,
            _ => ScalingTarget::Large1B, // Already at maximum
        }
    }
    
    /// Select scale-down target
    fn select_scale_down_target(&self, current_nodes: usize) -> Option<ScalingTarget> {
        match current_nodes {
            n if n > 1_048_576 => Some(ScalingTarget::Medium1M),
            n if n > 16_384 => Some(ScalingTarget::Small16K),
            n if n > 48 => Some(ScalingTarget::Micro48),
            _ => None, // Already at minimum
        }
    }
    
    /// Calculate confidence in the analysis
    fn calculate_confidence(&self, metrics: &[&WorkloadMetrics]) -> f64 {
        if metrics.len() < 5 {
            return 0.5; // Low confidence with insufficient data
        }
        
        // Calculate variance in measurements (lower variance = higher confidence)
        let avg_usage = self.average_resource_usage(metrics);
        let variance: f64 = metrics.iter()
            .map(|m| {
                let usage = m.cpu_usage.max(m.memory_usage).max(m.gpu_usage);
                (usage - avg_usage).powi(2)
            })
            .sum::<f64>() / metrics.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Convert to confidence (lower std_dev = higher confidence)
        let confidence = 1.0 - (std_dev * 2.0).min(1.0);
        confidence.max(0.1) // Minimum 10% confidence
    }
    
    /// Get scaling target details
    pub fn get_target_details(&self, target: &ScalingTarget) -> ScalingTargetDetails {
        match target {
            ScalingTarget::Micro48 => ScalingTargetDetails {
                nodes: 48,
                memory_requirement: 10 * 1024 * 1024, // 10 MB
                gpu_requirement: GPURequirement::Integrated,
                update_rate: 10_000, // 10 kHz
                phi_time_limit: Duration::from_millis(1),
            },
            ScalingTarget::Small16K => ScalingTargetDetails {
                nodes: 16_384,
                memory_requirement: 500 * 1024 * 1024, // 500 MB
                gpu_requirement: GPURequirement::MidRange,
                update_rate: 1_000, // 1 kHz
                phi_time_limit: Duration::from_millis(50),
            },
            ScalingTarget::Medium1M => ScalingTargetDetails {
                nodes: 1_048_576,
                memory_requirement: 8 * 1024 * 1024 * 1024, // 8 GB
                gpu_requirement: GPURequirement::HighEnd,
                update_rate: 100, // 100 Hz
                phi_time_limit: Duration::from_millis(500),
            },
            ScalingTarget::Large1B => ScalingTargetDetails {
                nodes: 1_000_000_000,
                memory_requirement: 128 * 1024 * 1024 * 1024, // 128 GB
                gpu_requirement: GPURequirement::MultiGPU,
                update_rate: 10, // 10 Hz
                phi_time_limit: Duration::from_secs(5),
            },
        }
    }
}

/// Scaling target details
#[derive(Debug, Clone)]
pub struct ScalingTargetDetails {
    pub nodes: usize,
    pub memory_requirement: u64, // bytes
    pub gpu_requirement: GPURequirement,
    pub update_rate: u32, // Hz
    pub phi_time_limit: Duration,
}

/// GPU requirement levels
#[derive(Debug, Clone, PartialEq)]
pub enum GPURequirement {
    Integrated,
    MidRange,
    HighEnd,
    MultiGPU,
}

impl Default for WorkloadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
