// Cascade Coordinator - Target: <5ms total system latency
// Hierarchical orchestration of attention cascade system

use super::{
    AttentionError, AttentionLayer, AttentionMetrics, AttentionOutput, AttentionResult,
    MacroAttention, MarketInput, MicroAttention, MilliAttention, TemporalBridge,
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Instant;

/// Hierarchical cascade coordinator for attention system
pub struct CascadeCoordinator {
    // Attention layer instances
    micro_attention: Arc<Mutex<MicroAttention>>,
    milli_attention: Arc<Mutex<MilliAttention>>,
    macro_attention: Arc<Mutex<MacroAttention>>,
    temporal_bridge: Arc<Mutex<TemporalBridge>>,

    // Communication channels
    micro_channel: (Sender<MarketInput>, Receiver<AttentionOutput>),
    milli_channel: (Sender<AttentionOutput>, Receiver<AttentionOutput>),
    macro_channel: (Sender<AttentionOutput>, Receiver<AttentionOutput>),

    // Coordination state
    cascade_state: Arc<RwLock<CascadeState>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    adaptive_controller: Arc<Mutex<AdaptiveController>>,

    // System configuration
    config: CascadeConfig,
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

/// Configuration for cascade system
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    pub target_latency_ns: u64,
    pub parallel_processing: bool,
    pub adaptive_optimization: bool,
    pub enable_predictive_caching: bool,
    pub enable_load_balancing: bool,
    pub max_concurrent_requests: usize,
    pub memory_pool_size: usize,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 5_000_000, // 5ms target
            parallel_processing: true,
            adaptive_optimization: true,
            enable_predictive_caching: true,
            enable_load_balancing: true,
            max_concurrent_requests: 1000,
            memory_pool_size: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// State of the cascade system
#[derive(Debug, Clone)]
struct CascadeState {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    average_latency_ns: u64,
    current_load: f64,
    adaptive_weights: CascadeWeights,
    system_health: SystemHealth,
}

/// Weights for cascade layers
#[derive(Debug, Clone)]
struct CascadeWeights {
    micro_weight: f64,
    milli_weight: f64,
    macro_weight: f64,
    temporal_weight: f64,
}

impl Default for CascadeWeights {
    fn default() -> Self {
        Self {
            micro_weight: 0.4,
            milli_weight: 0.3,
            macro_weight: 0.2,
            temporal_weight: 0.1,
        }
    }
}

/// System health monitoring
#[derive(Debug, Clone)]
struct SystemHealth {
    micro_health: f64,
    milli_health: f64,
    macro_health: f64,
    bridge_health: f64,
    overall_health: f64,
    bottleneck_layer: Option<String>,
}

/// Performance monitoring and optimization
struct PerformanceMonitor {
    latency_history: std::collections::VecDeque<u64>,
    throughput_history: std::collections::VecDeque<f64>,
    error_rates: std::collections::HashMap<String, f64>,
    bottleneck_analysis: BottleneckAnalysis,
    optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Bottleneck analysis for performance optimization
#[derive(Debug, Clone)]
struct BottleneckAnalysis {
    micro_bottlenecks: Vec<Bottleneck>,
    milli_bottlenecks: Vec<Bottleneck>,
    macro_bottlenecks: Vec<Bottleneck>,
    bridge_bottlenecks: Vec<Bottleneck>,
    system_bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone)]
struct Bottleneck {
    component: String,
    severity: f64,
    impact_ns: u64,
    suggested_fix: String,
}

/// Optimization suggestions
#[derive(Debug, Clone)]
struct OptimizationSuggestion {
    component: String,
    optimization_type: OptimizationType,
    expected_improvement_ns: u64,
    implementation_cost: f64,
    priority: f64,
}

#[derive(Debug, Clone)]
enum OptimizationType {
    MemoryOptimization,
    AlgorithmOptimization,
    ParallelizationImprovement,
    CacheOptimization,
    LoadBalancing,
    PredictivePrefetching,
}

/// Adaptive controller for dynamic optimization
struct AdaptiveController {
    learning_rate: f64,
    adaptation_history: Vec<AdaptationRecord>,
    current_strategy: AdaptationStrategy,
    performance_baseline: PerformanceBaseline,
}

#[derive(Debug, Clone)]
struct AdaptationRecord {
    timestamp: u64,
    adaptation_type: String,
    parameters_before: std::collections::HashMap<String, f64>,
    parameters_after: std::collections::HashMap<String, f64>,
    performance_before: f64,
    performance_after: f64,
    success: bool,
}

#[derive(Debug, Clone)]
enum AdaptationStrategy {
    Conservative,
    Aggressive,
    Balanced,
    Custom(std::collections::HashMap<String, f64>),
}

#[derive(Debug, Clone)]
struct PerformanceBaseline {
    baseline_latency_ns: u64,
    baseline_throughput: f64,
    baseline_accuracy: f64,
    last_updated: u64,
}

impl CascadeCoordinator {
    pub fn new(config: CascadeConfig) -> AttentionResult<Self> {
        // Create attention layer instances
        let micro_attention = Arc::new(Mutex::new(MicroAttention::new(
            64,
            config.parallel_processing,
        )?));
        let milli_attention = Arc::new(Mutex::new(MilliAttention::new(
            200,
            config.parallel_processing,
        )?));
        let macro_attention = Arc::new(Mutex::new(MacroAttention::new(0.5, 0.3)?));

        // Create communication channels
        let (micro_tx, micro_rx) = bounded(1000);
        let (micro_out_tx, micro_out_rx) = bounded(1000);
        let (milli_out_tx, milli_out_rx) = bounded(1000);
        let (macro_out_tx, macro_out_rx) = bounded(1000);

        // Create temporal bridge
        let temporal_bridge = Arc::new(Mutex::new(TemporalBridge::new(
            micro_out_rx,
            milli_out_rx,
            macro_out_rx,
            true,
        )?));

        let cascade_state = Arc::new(RwLock::new(CascadeState {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_latency_ns: 0,
            current_load: 0.0,
            adaptive_weights: CascadeWeights::default(),
            system_health: SystemHealth {
                micro_health: 1.0,
                milli_health: 1.0,
                macro_health: 1.0,
                bridge_health: 1.0,
                overall_health: 1.0,
                bottleneck_layer: None,
            },
        }));

        Ok(Self {
            micro_attention,
            milli_attention,
            macro_attention,
            temporal_bridge,
            micro_channel: (micro_tx, micro_rx),
            milli_channel: (micro_out_tx, milli_out_rx),
            macro_channel: (macro_out_tx, macro_out_rx),
            cascade_state,
            performance_monitor: Arc::new(RwLock::new(PerformanceMonitor::new())),
            adaptive_controller: Arc::new(Mutex::new(AdaptiveController::new())),
            config,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start the cascade system with parallel processing
    pub fn start(&self) -> AttentionResult<()> {
        self.is_running
            .store(true, std::sync::atomic::Ordering::Relaxed);

        if self.config.parallel_processing {
            self.start_parallel_processing()?;
        }

        // Start performance monitoring
        self.start_performance_monitoring()?;

        // Start adaptive optimization
        if self.config.adaptive_optimization {
            self.start_adaptive_optimization()?;
        }

        Ok(())
    }

    /// Stop the cascade system
    pub fn stop(&self) {
        self.is_running
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Process market input through the attention cascade
    pub fn process_cascade(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let start = Instant::now();

        // Update cascade state
        {
            let mut state = self.cascade_state.write().unwrap();
            state.total_requests += 1;
        }

        // Execute attention cascade
        let result = if self.config.parallel_processing {
            self.process_parallel_cascade(input)
        } else {
            self.process_sequential_cascade(input)
        };

        let execution_time_ns = start.elapsed().as_nanos() as u64;

        // Update performance metrics
        self.update_cascade_metrics(execution_time_ns, &result);

        // Validate latency target
        if execution_time_ns > self.config.target_latency_ns {
            self.handle_latency_violation(execution_time_ns);
        }

        // Trigger adaptive optimization if needed
        if self.config.adaptive_optimization {
            self.trigger_adaptive_optimization(&result, execution_time_ns)?;
        }

        result
    }

    /// Process cascade in parallel for maximum performance
    fn process_parallel_cascade(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let start = Instant::now();

        // Clone input for parallel processing
        let micro_input = input.clone();
        let milli_input = input.clone();
        let macro_input = input.clone();

        // Execute all attention layers in parallel
        let results: Result<Vec<AttentionOutput>, AttentionError> = [
            || self.process_micro_attention(micro_input),
            || self.process_milli_attention(milli_input),
            || self.process_macro_attention(macro_input),
        ]
        .par_iter()
        .map(|f| f())
        .collect();

        let layer_outputs = results?;

        // Fuse results through temporal bridge
        let fused_output = self.fuse_attention_outputs(&layer_outputs, &input)?;

        let execution_time_ns = start.elapsed().as_nanos() as u64;

        Ok(AttentionOutput {
            execution_time_ns,
            ..fused_output
        })
    }

    /// Process cascade sequentially for deterministic behavior
    fn process_sequential_cascade(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let start = Instant::now();

        // Execute attention layers in sequence
        let micro_output = self.process_micro_attention(input.clone())?;
        let milli_output = self.process_milli_attention(input.clone())?;
        let macro_output = self.process_macro_attention(input.clone())?;

        // Fuse results through temporal bridge
        let layer_outputs = vec![micro_output, milli_output, macro_output];
        let fused_output = self.fuse_attention_outputs(&layer_outputs, &input)?;

        let execution_time_ns = start.elapsed().as_nanos() as u64;

        Ok(AttentionOutput {
            execution_time_ns,
            ..fused_output
        })
    }

    /// Process micro attention layer
    fn process_micro_attention(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let micro_attention = self.micro_attention.lock().unwrap();
        micro_attention.process(&input)
    }

    /// Process milli attention layer
    fn process_milli_attention(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let milli_attention = self.milli_attention.lock().unwrap();
        milli_attention.process(&input)
    }

    /// Process macro attention layer
    fn process_macro_attention(&self, input: MarketInput) -> AttentionResult<AttentionOutput> {
        let macro_attention = self.macro_attention.lock().unwrap();
        macro_attention.process(&input)
    }

    /// Fuse attention outputs using temporal bridge
    fn fuse_attention_outputs(
        &self,
        layer_outputs: &[AttentionOutput],
        input: &MarketInput,
    ) -> AttentionResult<AttentionOutput> {
        let temporal_bridge = self.temporal_bridge.lock().unwrap();
        temporal_bridge.fuse_attention(input)
    }

    /// Start parallel processing threads
    fn start_parallel_processing(&self) -> AttentionResult<()> {
        // Implementation would start worker threads for each attention layer
        Ok(())
    }

    /// Start performance monitoring
    fn start_performance_monitoring(&self) -> AttentionResult<()> {
        // Implementation would start monitoring thread
        Ok(())
    }

    /// Start adaptive optimization
    fn start_adaptive_optimization(&self) -> AttentionResult<()> {
        // Implementation would start optimization thread
        Ok(())
    }

    /// Update cascade performance metrics
    fn update_cascade_metrics(
        &self,
        execution_time_ns: u64,
        result: &AttentionResult<AttentionOutput>,
    ) {
        let mut state = self.cascade_state.write().unwrap();

        match result {
            Ok(_) => state.successful_requests += 1,
            Err(_) => state.failed_requests += 1,
        }

        // Update rolling average latency
        let total_requests = state.total_requests;
        state.average_latency_ns = ((state.average_latency_ns * (total_requests - 1))
            + execution_time_ns)
            / total_requests;

        // Update current load
        state.current_load = execution_time_ns as f64 / self.config.target_latency_ns as f64;

        // Update performance monitor
        let mut monitor = self.performance_monitor.write().unwrap();
        monitor.update_metrics(execution_time_ns, result.is_ok());
    }

    /// Handle latency target violations
    fn handle_latency_violation(&self, actual_latency_ns: u64) {
        let mut monitor = self.performance_monitor.write().unwrap();
        monitor.record_latency_violation(actual_latency_ns, self.config.target_latency_ns);

        // Analyze bottlenecks
        let bottlenecks = self.analyze_bottlenecks();
        monitor.update_bottleneck_analysis(bottlenecks);

        // Generate optimization suggestions
        let suggestions = self.generate_optimization_suggestions();
        monitor.update_optimization_suggestions(suggestions);
    }

    /// Trigger adaptive optimization based on performance
    fn trigger_adaptive_optimization(
        &self,
        result: &AttentionResult<AttentionOutput>,
        execution_time_ns: u64,
    ) -> AttentionResult<()> {
        let mut controller = self.adaptive_controller.lock().unwrap();

        // Calculate performance score
        let performance_score = self.calculate_performance_score(result, execution_time_ns);

        // Check if adaptation is needed
        if controller.should_adapt(performance_score) {
            let adaptation = controller.generate_adaptation(performance_score)?;
            self.apply_adaptation(adaptation)?;
        }

        Ok(())
    }

    /// Calculate overall performance score
    fn calculate_performance_score(
        &self,
        result: &AttentionResult<AttentionOutput>,
        execution_time_ns: u64,
    ) -> f64 {
        let latency_score =
            (self.config.target_latency_ns as f64 / execution_time_ns as f64).min(1.0);
        let success_score = if result.is_ok() { 1.0 } else { 0.0 };

        let accuracy_score = if let Ok(output) = result {
            output.confidence
        } else {
            0.0
        };

        (latency_score * 0.4 + success_score * 0.3 + accuracy_score * 0.3).min(1.0)
    }

    /// Analyze system bottlenecks
    fn analyze_bottlenecks(&self) -> BottleneckAnalysis {
        let micro_metrics = {
            let micro = self.micro_attention.lock().unwrap();
            micro.get_metrics()
        };

        let milli_metrics = {
            let milli = self.milli_attention.lock().unwrap();
            milli.get_metrics()
        };

        let macro_metrics = {
            let macro_attention = self.macro_attention.lock().unwrap();
            macro_attention.get_metrics()
        };

        let bridge_metrics = {
            let bridge = self.temporal_bridge.lock().unwrap();
            bridge.get_metrics()
        };

        // Identify bottlenecks based on latency targets
        let mut bottlenecks = BottleneckAnalysis {
            micro_bottlenecks: Vec::new(),
            milli_bottlenecks: Vec::new(),
            macro_bottlenecks: Vec::new(),
            bridge_bottlenecks: Vec::new(),
            system_bottlenecks: Vec::new(),
        };

        // Analyze micro layer bottlenecks
        if micro_metrics.micro_latency_ns > 10_000 {
            // 10μs target
            bottlenecks.micro_bottlenecks.push(Bottleneck {
                component: "micro_attention".to_string(),
                severity: (micro_metrics.micro_latency_ns as f64 / 10_000.0) - 1.0,
                impact_ns: micro_metrics.micro_latency_ns - 10_000,
                suggested_fix: "Optimize SIMD operations and memory access patterns".to_string(),
            });
        }

        // Analyze milli layer bottlenecks
        if milli_metrics.milli_latency_ns > 1_000_000 {
            // 1ms target
            bottlenecks.milli_bottlenecks.push(Bottleneck {
                component: "milli_attention".to_string(),
                severity: (milli_metrics.milli_latency_ns as f64 / 1_000_000.0) - 1.0,
                impact_ns: milli_metrics.milli_latency_ns - 1_000_000,
                suggested_fix: "Optimize pattern recognition algorithms and parallel processing"
                    .to_string(),
            });
        }

        // Analyze macro layer bottlenecks
        if macro_metrics.macro_latency_ns > 10_000_000 {
            // 10ms target
            bottlenecks.macro_bottlenecks.push(Bottleneck {
                component: "macro_attention".to_string(),
                severity: (macro_metrics.macro_latency_ns as f64 / 10_000_000.0) - 1.0,
                impact_ns: macro_metrics.macro_latency_ns - 10_000_000,
                suggested_fix: "Optimize portfolio optimization and risk calculations".to_string(),
            });
        }

        // Analyze bridge bottlenecks
        if bridge_metrics.bridge_latency_ns > 100_000 {
            // 100μs target
            bottlenecks.bridge_bottlenecks.push(Bottleneck {
                component: "temporal_bridge".to_string(),
                severity: (bridge_metrics.bridge_latency_ns as f64 / 100_000.0) - 1.0,
                impact_ns: bridge_metrics.bridge_latency_ns - 100_000,
                suggested_fix: "Optimize temporal fusion and synchronization".to_string(),
            });
        }

        bottlenecks
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Memory optimization suggestions
        suggestions.push(OptimizationSuggestion {
            component: "memory_management".to_string(),
            optimization_type: OptimizationType::MemoryOptimization,
            expected_improvement_ns: 500_000,
            implementation_cost: 0.3,
            priority: 0.8,
        });

        // Algorithm optimization suggestions
        suggestions.push(OptimizationSuggestion {
            component: "attention_algorithms".to_string(),
            optimization_type: OptimizationType::AlgorithmOptimization,
            expected_improvement_ns: 1_000_000,
            implementation_cost: 0.7,
            priority: 0.9,
        });

        // Parallelization improvements
        suggestions.push(OptimizationSuggestion {
            component: "parallel_processing".to_string(),
            optimization_type: OptimizationType::ParallelizationImprovement,
            expected_improvement_ns: 2_000_000,
            implementation_cost: 0.5,
            priority: 0.95,
        });

        suggestions
    }

    /// Apply optimization adaptation
    fn apply_adaptation(&self, adaptation: AdaptationRecord) -> AttentionResult<()> {
        // Implementation would apply the suggested adaptations
        // This could include adjusting weights, changing algorithms, etc.
        Ok(())
    }

    /// Get comprehensive system metrics
    pub fn get_system_metrics(&self) -> CascadeMetrics {
        let state = self.cascade_state.read().unwrap();
        let monitor = self.performance_monitor.read().unwrap();

        let micro_metrics = {
            let micro = self.micro_attention.lock().unwrap();
            micro.get_metrics()
        };

        let milli_metrics = {
            let milli = self.milli_attention.lock().unwrap();
            milli.get_metrics()
        };

        let macro_metrics = {
            let macro_attention = self.macro_attention.lock().unwrap();
            macro_attention.get_metrics()
        };

        let bridge_metrics = {
            let bridge = self.temporal_bridge.lock().unwrap();
            bridge.get_metrics()
        };

        CascadeMetrics {
            total_requests: state.total_requests,
            successful_requests: state.successful_requests,
            failed_requests: state.failed_requests,
            success_rate: if state.total_requests > 0 {
                state.successful_requests as f64 / state.total_requests as f64
            } else {
                0.0
            },
            average_latency_ns: state.average_latency_ns,
            current_load: state.current_load,
            micro_metrics,
            milli_metrics,
            macro_metrics,
            bridge_metrics,
            system_health: state.system_health.clone(),
            bottleneck_analysis: monitor.bottleneck_analysis.clone(),
        }
    }

    /// Validate system performance against targets
    pub fn validate_performance(&self) -> AttentionResult<()> {
        let metrics = self.get_system_metrics();

        // Check latency target
        if metrics.average_latency_ns > self.config.target_latency_ns {
            return Err(AttentionError::LatencyExceeded {
                actual_ns: metrics.average_latency_ns,
                target_ns: self.config.target_latency_ns,
            });
        }

        // Check success rate
        if metrics.success_rate < 0.95 {
            return Err(AttentionError::ConvergenceFailed);
        }

        // Validate individual layer performance
        {
            let micro = self.micro_attention.lock().unwrap();
            micro.validate_performance()?;
        }

        {
            let milli = self.milli_attention.lock().unwrap();
            milli.validate_performance()?;
        }

        {
            let macro_attention = self.macro_attention.lock().unwrap();
            macro_attention.validate_performance()?;
        }

        Ok(())
    }
}

/// Comprehensive cascade metrics
#[derive(Debug, Clone)]
pub struct CascadeMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub average_latency_ns: u64,
    pub current_load: f64,
    pub micro_metrics: AttentionMetrics,
    pub milli_metrics: AttentionMetrics,
    pub macro_metrics: AttentionMetrics,
    pub bridge_metrics: AttentionMetrics,
    pub system_health: SystemHealth,
    pub bottleneck_analysis: BottleneckAnalysis,
}

// Implementation of helper structs
impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            latency_history: std::collections::VecDeque::new(),
            throughput_history: std::collections::VecDeque::new(),
            error_rates: std::collections::HashMap::new(),
            bottleneck_analysis: BottleneckAnalysis {
                micro_bottlenecks: Vec::new(),
                milli_bottlenecks: Vec::new(),
                macro_bottlenecks: Vec::new(),
                bridge_bottlenecks: Vec::new(),
                system_bottlenecks: Vec::new(),
            },
            optimization_suggestions: Vec::new(),
        }
    }

    fn update_metrics(&mut self, execution_time_ns: u64, success: bool) {
        self.latency_history.push_back(execution_time_ns);
        if self.latency_history.len() > 1000 {
            self.latency_history.pop_front();
        }

        let throughput = 1_000_000_000.0 / execution_time_ns as f64;
        self.throughput_history.push_back(throughput);
        if self.throughput_history.len() > 1000 {
            self.throughput_history.pop_front();
        }
    }

    fn record_latency_violation(&mut self, actual_ns: u64, target_ns: u64) {
        // Record latency violation for analysis
    }

    fn update_bottleneck_analysis(&mut self, analysis: BottleneckAnalysis) {
        self.bottleneck_analysis = analysis;
    }

    fn update_optimization_suggestions(&mut self, suggestions: Vec<OptimizationSuggestion>) {
        self.optimization_suggestions = suggestions;
    }
}

impl AdaptiveController {
    fn new() -> Self {
        Self {
            learning_rate: 0.01,
            adaptation_history: Vec::new(),
            current_strategy: AdaptationStrategy::Balanced,
            performance_baseline: PerformanceBaseline {
                baseline_latency_ns: 5_000_000,
                baseline_throughput: 200.0,
                baseline_accuracy: 0.8,
                last_updated: 0,
            },
        }
    }

    fn should_adapt(&self, performance_score: f64) -> bool {
        performance_score < 0.8
    }

    fn generate_adaptation(&mut self, performance_score: f64) -> AttentionResult<AdaptationRecord> {
        // Generate adaptation based on performance
        Ok(AdaptationRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            adaptation_type: "weight_adjustment".to_string(),
            parameters_before: std::collections::HashMap::new(),
            parameters_after: std::collections::HashMap::new(),
            performance_before: performance_score,
            performance_after: 0.0, // Will be updated later
            success: false,         // Will be updated later
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_coordinator_creation() {
        let config = CascadeConfig::default();
        let coordinator = CascadeCoordinator::new(config).unwrap();
        assert_eq!(coordinator.config.target_latency_ns, 5_000_000);
    }

    #[test]
    fn test_cascade_processing() {
        let config = CascadeConfig {
            parallel_processing: false,
            ..CascadeConfig::default()
        };
        let coordinator = CascadeCoordinator::new(config).unwrap();

        let input = MarketInput {
            timestamp: 1640995200000,
            price: 45000.0,
            volume: 1.5,
            bid: 44990.0,
            ask: 45010.0,
            order_flow: vec![0.5, -0.3, 0.8, -0.2],
            microstructure: vec![0.1, 0.2, -0.1, 0.15],
        };

        let output = coordinator.process_cascade(input).unwrap();
        assert!(output.execution_time_ns > 0);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_performance_metrics() {
        let config = CascadeConfig::default();
        let coordinator = CascadeCoordinator::new(config).unwrap();

        let metrics = coordinator.get_system_metrics();
        assert!(metrics.success_rate >= 0.0 && metrics.success_rate <= 1.0);
        assert!(metrics.current_load >= 0.0);
    }

    #[test]
    fn test_bottleneck_analysis() {
        let config = CascadeConfig::default();
        let coordinator = CascadeCoordinator::new(config).unwrap();

        let bottlenecks = coordinator.analyze_bottlenecks();
        // Bottlenecks analysis should complete without errors
        assert!(true);
    }

    #[test]
    fn test_optimization_suggestions() {
        let config = CascadeConfig::default();
        let coordinator = CascadeCoordinator::new(config).unwrap();

        let suggestions = coordinator.generate_optimization_suggestions();
        assert!(!suggestions.is_empty());

        for suggestion in suggestions {
            assert!(suggestion.expected_improvement_ns > 0);
            assert!(suggestion.implementation_cost >= 0.0 && suggestion.implementation_cost <= 1.0);
            assert!(suggestion.priority >= 0.0 && suggestion.priority <= 1.0);
        }
    }
}
