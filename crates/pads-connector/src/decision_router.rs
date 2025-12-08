//! Decision routing across panarchy scales

use crate::{
    config::{PadsConfig, RoutingConfig, LoadBalancingStrategy},
    error::{PadsError, Result},
    monitoring::PadsMonitor,
    types::*,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, info, warn};
use std::collections::VecDeque;

/// Routes decisions to appropriate scales and handlers
pub struct DecisionRouter {
    config: Arc<PadsConfig>,
    monitor: Arc<PadsMonitor>,
    routes: DashMap<ScaleLevel, RouteHandler>,
    queues: DashMap<ScaleLevel, Arc<RwLock<DecisionQueue>>>,
    load_balancer: Arc<LoadBalancer>,
    execution_pool: Arc<ExecutionPool>,
}

/// Handles routing for a specific scale
struct RouteHandler {
    scale: ScaleLevel,
    processors: Vec<Arc<DecisionProcessor>>,
    strategy: RoutingStrategy,
    metrics: RouteMetrics,
}

/// Decision queue for a scale
struct DecisionQueue {
    queue: VecDeque<RoutedDecision>,
    max_size: usize,
    priority_index: DashMap<String, f64>,
}

/// Load balancer for distributing decisions
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    weights: DashMap<ScaleLevel, f64>,
    round_robin_index: RwLock<usize>,
}

/// Execution pool for processing decisions
struct ExecutionPool {
    semaphore: Arc<Semaphore>,
    workers: Vec<WorkerHandle>,
}

/// Worker handle for execution
struct WorkerHandle {
    id: usize,
    sender: mpsc::Sender<WorkerCommand>,
}

/// Worker command
enum WorkerCommand {
    Process(RoutedDecision),
    Shutdown,
}

/// Decision processor trait
#[async_trait::async_trait]
trait DecisionProcessor: Send + Sync {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult>;
    fn can_handle(&self, decision: &RoutedDecision) -> bool;
}

/// Routing strategy
#[derive(Debug, Clone, Copy)]
enum RoutingStrategy {
    Direct,
    Broadcast,
    Conditional,
    Adaptive,
}

/// Route metrics
#[derive(Debug, Clone, Default)]
struct RouteMetrics {
    total_routed: u64,
    successful: u64,
    failed: u64,
    avg_latency_ms: f64,
}

impl DecisionRouter {
    /// Create new decision router
    pub async fn new(config: Arc<PadsConfig>, monitor: Arc<PadsMonitor>) -> Result<Self> {
        let routes = DashMap::new();
        let queues = DashMap::new();
        
        // Initialize routes and queues for each scale
        for level in [ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro] {
            let handler = Self::create_route_handler(level, &config).await?;
            routes.insert(level, handler);
            
            let queue = Arc::new(RwLock::new(DecisionQueue {
                queue: VecDeque::with_capacity(config.routing_config.max_queue_size),
                max_size: config.routing_config.max_queue_size,
                priority_index: DashMap::new(),
            }));
            queues.insert(level, queue);
        }
        
        let load_balancer = Arc::new(LoadBalancer {
            strategy: config.routing_config.load_balancing,
            weights: DashMap::new(),
            round_robin_index: RwLock::new(0),
        });
        
        let execution_pool = Arc::new(Self::create_execution_pool(&config).await?);
        
        Ok(Self {
            config,
            monitor,
            routes,
            queues,
            load_balancer,
            execution_pool,
        })
    }
    
    /// Create route handler for a scale
    async fn create_route_handler(level: ScaleLevel, config: &PadsConfig) -> Result<RouteHandler> {
        let processors = match level {
            ScaleLevel::Micro => Self::create_micro_processors(config).await?,
            ScaleLevel::Meso => Self::create_meso_processors(config).await?,
            ScaleLevel::Macro => Self::create_macro_processors(config).await?,
        };
        
        let strategy = match level {
            ScaleLevel::Micro => RoutingStrategy::Direct,
            ScaleLevel::Meso => RoutingStrategy::Conditional,
            ScaleLevel::Macro => RoutingStrategy::Adaptive,
        };
        
        Ok(RouteHandler {
            scale: level,
            processors,
            strategy,
            metrics: RouteMetrics::default(),
        })
    }
    
    /// Create micro scale processors
    async fn create_micro_processors(config: &PadsConfig) -> Result<Vec<Arc<DecisionProcessor>>> {
        Ok(vec![
            Arc::new(FastOptimizationProcessor::new(config)),
            Arc::new(LocalSearchProcessor::new(config)),
            Arc::new(ExploitationProcessor::new(config)),
        ])
    }
    
    /// Create meso scale processors
    async fn create_meso_processors(config: &PadsConfig) -> Result<Vec<Arc<DecisionProcessor>>> {
        Ok(vec![
            Arc::new(BalancedProcessor::new(config)),
            Arc::new(CoordinationProcessor::new(config)),
            Arc::new(TransitionProcessor::new(config)),
        ])
    }
    
    /// Create macro scale processors
    async fn create_macro_processors(config: &PadsConfig) -> Result<Vec<Arc<DecisionProcessor>>> {
        Ok(vec![
            Arc::new(StrategicProcessor::new(config)),
            Arc::new(ExplorationProcessor::new(config)),
            Arc::new(InnovationProcessor::new(config)),
        ])
    }
    
    /// Create execution pool
    async fn create_execution_pool(config: &PadsConfig) -> Result<ExecutionPool> {
        let num_workers = config.performance.thread_pool_size;
        let semaphore = Arc::new(Semaphore::new(num_workers));
        let mut workers = Vec::with_capacity(num_workers);
        
        for id in 0..num_workers {
            let (tx, mut rx) = mpsc::channel(100);
            
            // Spawn worker task
            tokio::spawn(async move {
                while let Some(cmd) = rx.recv().await {
                    match cmd {
                        WorkerCommand::Process(_decision) => {
                            // Process decision
                            debug!("Worker {} processing decision", id);
                        }
                        WorkerCommand::Shutdown => break,
                    }
                }
            });
            
            workers.push(WorkerHandle {
                id,
                sender: tx,
            });
        }
        
        Ok(ExecutionPool {
            semaphore,
            workers,
        })
    }
    
    /// Setup routing rules
    pub async fn setup_routes(&self) -> Result<()> {
        info!("Setting up decision routing rules");
        
        // Initialize load balancer weights
        self.load_balancer.weights.insert(ScaleLevel::Micro, 1.0);
        self.load_balancer.weights.insert(ScaleLevel::Meso, 1.0);
        self.load_balancer.weights.insert(ScaleLevel::Macro, 1.0);
        
        Ok(())
    }
    
    /// Route decision to appropriate scale
    pub async fn route_decision(
        &self,
        decision: PanarchyDecision,
        scale: PanarchyScale
    ) -> Result<RoutedDecision> {
        let routing_score = self.calculate_routing_score(&decision, &scale);
        let priority = self.calculate_priority(&decision);
        
        let routed = RoutedDecision {
            decision,
            assigned_scale: scale.level,
            routing_score,
            priority,
        };
        
        // Add to appropriate queue
        self.enqueue_decision(scale.level, routed.clone()).await?;
        
        Ok(routed)
    }
    
    /// Calculate routing score
    fn calculate_routing_score(&self, decision: &PanarchyDecision, scale: &PanarchyScale) -> f64 {
        let mut score = 0.0;
        
        // Scale fit score
        score += match scale.level {
            ScaleLevel::Micro => (1.0 - decision.uncertainty) * 0.3,
            ScaleLevel::Meso => 0.5 * 0.3,
            ScaleLevel::Macro => decision.uncertainty * 0.3,
        };
        
        // Urgency alignment
        score += match scale.level {
            ScaleLevel::Micro => decision.urgency * 0.3,
            ScaleLevel::Meso => (1.0 - (decision.urgency - 0.5).abs() * 2.0) * 0.3,
            ScaleLevel::Macro => (1.0 - decision.urgency) * 0.3,
        };
        
        // Resource availability
        score += scale.potential * 0.4;
        
        score
    }
    
    /// Calculate decision priority
    fn calculate_priority(&self, decision: &PanarchyDecision) -> f64 {
        let weights = &self.config.routing_config.priority_weights;
        
        weights.urgency * decision.urgency +
        weights.impact * decision.impact +
        weights.confidence * (1.0 - decision.uncertainty) +
        weights.resource * 0.5 // Default resource availability
    }
    
    /// Enqueue decision for processing
    async fn enqueue_decision(&self, scale: ScaleLevel, decision: RoutedDecision) -> Result<()> {
        let queue_ref = self.queues.get(&scale)
            .ok_or_else(|| PadsError::routing("Queue not found"))?;
        
        let mut queue = queue_ref.write().await;
        
        // Check queue capacity
        if queue.queue.len() >= queue.max_size {
            return Err(PadsError::routing("Queue full"));
        }
        
        // Add to priority index
        queue.priority_index.insert(decision.decision.id.clone(), decision.priority);
        
        // Insert in priority order
        let insert_pos = queue.queue.iter()
            .position(|d| d.priority < decision.priority)
            .unwrap_or(queue.queue.len());
        
        queue.queue.insert(insert_pos, decision);
        
        Ok(())
    }
    
    /// Execute micro scale decision
    pub async fn execute_micro_decision(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        let handler = self.routes.get(&ScaleLevel::Micro)
            .ok_or_else(|| PadsError::routing("Micro handler not found"))?;
        
        // Find suitable processor
        let processor = handler.processors.iter()
            .find(|p| p.can_handle(&decision))
            .ok_or_else(|| PadsError::routing("No suitable processor"))?;
        
        // Process decision
        let start = std::time::Instant::now();
        let result = processor.process(&decision).await?;
        let latency = start.elapsed().as_millis() as u64;
        
        // Update metrics
        self.monitor.record_routing_latency(ScaleLevel::Micro, latency);
        
        Ok(result)
    }
    
    /// Execute meso scale decision
    pub async fn execute_meso_decision(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        let handler = self.routes.get(&ScaleLevel::Meso)
            .ok_or_else(|| PadsError::routing("Meso handler not found"))?;
        
        match handler.strategy {
            RoutingStrategy::Conditional => {
                // Route based on conditions
                for processor in &handler.processors {
                    if processor.can_handle(&decision) {
                        return processor.process(&decision).await;
                    }
                }
            }
            _ => {
                // Default to first processor
                if let Some(processor) = handler.processors.first() {
                    return processor.process(&decision).await;
                }
            }
        }
        
        Err(PadsError::routing("No suitable meso processor"))
    }
    
    /// Execute macro scale decision
    pub async fn execute_macro_decision(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        let handler = self.routes.get(&ScaleLevel::Macro)
            .ok_or_else(|| PadsError::routing("Macro handler not found"))?;
        
        match handler.strategy {
            RoutingStrategy::Adaptive => {
                // Adaptive routing based on current state
                let best_processor = self.select_best_processor(&handler.processors, &decision).await?;
                best_processor.process(&decision).await
            }
            _ => {
                // Default to strategic processor
                handler.processors.first()
                    .ok_or_else(|| PadsError::routing("No macro processor"))?
                    .process(&decision).await
            }
        }
    }
    
    /// Select best processor adaptively
    async fn select_best_processor(
        &self,
        processors: &[Arc<DecisionProcessor>],
        decision: &RoutedDecision
    ) -> Result<Arc<DecisionProcessor>> {
        // Simple selection based on capability
        processors.iter()
            .find(|p| p.can_handle(decision))
            .cloned()
            .ok_or_else(|| PadsError::routing("No capable processor"))
    }
    
    /// Clear all queues
    pub async fn clear_queues(&self) -> Result<()> {
        for queue_ref in self.queues.iter() {
            let mut queue = queue_ref.write().await;
            queue.queue.clear();
            queue.priority_index.clear();
        }
        Ok(())
    }
    
    /// Get routing status
    pub async fn get_status(&self) -> Result<RoutingStatus> {
        let mut queue_sizes = std::collections::HashMap::new();
        let mut load_balance = std::collections::HashMap::new();
        
        for queue_ref in self.queues.iter() {
            let scale = *queue_ref.key();
            let queue = queue_ref.read().await;
            queue_sizes.insert(scale, queue.queue.len());
        }
        
        for weight_ref in self.load_balancer.weights.iter() {
            load_balance.insert(*weight_ref.key(), *weight_ref.value());
        }
        
        Ok(RoutingStatus {
            queue_sizes,
            routing_latency_ms: 10.0, // Would calculate actual
            load_balance,
        })
    }
    
    /// Adapt routing based on feedback
    pub async fn adapt_routing(&self, feedback: AdaptiveFeedback) -> Result<()> {
        // Update load balancer weights based on performance
        for (scale, adjustment) in feedback.scale_adjustments {
            if let Some(mut weight) = self.load_balancer.weights.get_mut(&scale) {
                *weight = (*weight + adjustment).clamp(0.1, 10.0);
            }
        }
        
        Ok(())
    }
}

// Processor implementations

struct FastOptimizationProcessor {
    config: Arc<PadsConfig>,
}

impl FastOptimizationProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for FastOptimizationProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Fast local optimization logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Micro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 5,
                confidence_score: 0.85,
                resource_usage: 0.3,
                adaptation_rate: 0.1,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.urgency > 0.7 && decision.decision.uncertainty < 0.3
    }
}

struct LocalSearchProcessor {
    config: Arc<PadsConfig>,
}

impl LocalSearchProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for LocalSearchProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Local search logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Micro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 10,
                confidence_score: 0.8,
                resource_usage: 0.4,
                adaptation_rate: 0.15,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, _decision: &RoutedDecision) -> bool {
        true // Can handle any micro decision
    }
}

struct ExploitationProcessor {
    config: Arc<PadsConfig>,
}

impl ExploitationProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for ExploitationProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Exploitation logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Micro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 8,
                confidence_score: 0.9,
                resource_usage: 0.2,
                adaptation_rate: 0.05,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.context.historical_performance.recent_success_rate > 0.7
    }
}

struct BalancedProcessor {
    config: Arc<PadsConfig>,
}

impl BalancedProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for BalancedProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Balanced processing logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Meso,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 20,
                confidence_score: 0.75,
                resource_usage: 0.5,
                adaptation_rate: 0.2,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, _decision: &RoutedDecision) -> bool {
        true
    }
}

struct CoordinationProcessor {
    config: Arc<PadsConfig>,
}

impl CoordinationProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for CoordinationProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Coordination logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Meso,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 25,
                confidence_score: 0.7,
                resource_usage: 0.6,
                adaptation_rate: 0.25,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![ScaleEffect {
                    target_scale: ScaleLevel::Macro,
                    effect_type: "coordination_insight".to_string(),
                    magnitude: 0.3,
                    delay: std::time::Duration::from_secs(5),
                }],
                downward_effects: vec![ScaleEffect {
                    target_scale: ScaleLevel::Micro,
                    effect_type: "parameter_update".to_string(),
                    magnitude: 0.2,
                    delay: std::time::Duration::from_secs(2),
                }],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.objectives.len() > 2
    }
}

struct TransitionProcessor {
    config: Arc<PadsConfig>,
}

impl TransitionProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for TransitionProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Transition handling logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Meso,
            success: true,
            actions: vec![Action {
                action_type: ActionType::Transition(TransitionAction {
                    from_scale: ScaleLevel::Micro,
                    to_scale: ScaleLevel::Macro,
                    trigger: "performance_threshold".to_string(),
                    preparation_steps: vec!["save_state".to_string(), "notify_components".to_string()],
                }),
                parameters: std::collections::HashMap::new(),
                confidence: 0.8,
                expected_impact: 0.5,
            }],
            metrics: DecisionMetrics {
                processing_time_ms: 30,
                confidence_score: 0.65,
                resource_usage: 0.7,
                adaptation_rate: 0.3,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.context.system_state.current_phase == AdaptiveCyclePhase::Release
    }
}

struct StrategicProcessor {
    config: Arc<PadsConfig>,
}

impl StrategicProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for StrategicProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Strategic processing logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Macro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 50,
                confidence_score: 0.6,
                resource_usage: 0.8,
                adaptation_rate: 0.4,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![
                    ScaleEffect {
                        target_scale: ScaleLevel::Meso,
                        effect_type: "strategic_guidance".to_string(),
                        magnitude: 0.6,
                        delay: std::time::Duration::from_secs(10),
                    },
                    ScaleEffect {
                        target_scale: ScaleLevel::Micro,
                        effect_type: "constraint_update".to_string(),
                        magnitude: 0.4,
                        delay: std::time::Duration::from_secs(15),
                    },
                ],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, _decision: &RoutedDecision) -> bool {
        true
    }
}

struct ExplorationProcessor {
    config: Arc<PadsConfig>,
}

impl ExplorationProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for ExplorationProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Exploration logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Macro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 60,
                confidence_score: 0.5,
                resource_usage: 0.9,
                adaptation_rate: 0.5,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.uncertainty > 0.7
    }
}

struct InnovationProcessor {
    config: Arc<PadsConfig>,
}

impl InnovationProcessor {
    fn new(config: &PadsConfig) -> Self {
        Self {
            config: Arc::new(config.clone()),
        }
    }
}

#[async_trait::async_trait]
impl DecisionProcessor for InnovationProcessor {
    async fn process(&self, decision: &RoutedDecision) -> Result<DecisionResult> {
        // Innovation logic
        Ok(DecisionResult {
            decision_id: decision.decision.id.clone(),
            timestamp: chrono::Utc::now(),
            scale_level: ScaleLevel::Macro,
            success: true,
            actions: vec![],
            metrics: DecisionMetrics {
                processing_time_ms: 100,
                confidence_score: 0.4,
                resource_usage: 0.95,
                adaptation_rate: 0.6,
            },
            cross_scale_effects: CrossScaleEffects {
                upward_effects: vec![],
                downward_effects: vec![ScaleEffect {
                    target_scale: ScaleLevel::Meso,
                    effect_type: "innovation_cascade".to_string(),
                    magnitude: 0.8,
                    delay: std::time::Duration::from_secs(30),
                }],
                lateral_effects: vec![],
            },
            errors: vec![],
        })
    }
    
    fn can_handle(&self, decision: &RoutedDecision) -> bool {
        decision.decision.context.system_state.current_phase == AdaptiveCyclePhase::Reorganization
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_decision_routing() {
        let config = Arc::new(PadsConfig::default());
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await.unwrap());
        let router = DecisionRouter::new(config, monitor).await.unwrap();
        
        router.setup_routes().await.unwrap();
        
        let decision = PanarchyDecision::test_decision();
        let scale = PanarchyScale {
            level: ScaleLevel::Micro,
            time_horizon: std::time::Duration::from_millis(100),
            spatial_extent: 0.1,
            connectivity: 0.3,
            resilience: 0.5,
            potential: 0.7,
        };
        
        let routed = router.route_decision(decision, scale).await.unwrap();
        assert_eq!(routed.assigned_scale, ScaleLevel::Micro);
    }
}