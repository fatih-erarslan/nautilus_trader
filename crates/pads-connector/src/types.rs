//! Type definitions for PADS connector

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use async_trait::async_trait;
use crate::Result;

/// Scale level in the panarchy hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaleLevel {
    /// Micro scale - fast, local optimization (exploitation)
    Micro,
    /// Meso scale - balanced transition phase
    Meso,
    /// Macro scale - slow, global exploration
    Macro,
}

/// Panarchy scale information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyScale {
    pub level: ScaleLevel,
    pub time_horizon: std::time::Duration,
    pub spatial_extent: f64,
    pub connectivity: f64,
    pub resilience: f64,
    pub potential: f64,
}

/// Adaptive cycle phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveCyclePhase {
    /// Growth/exploitation phase (r)
    Growth,
    /// Conservation phase (K)
    Conservation,
    /// Release/creative destruction phase (Ω)
    Release,
    /// Reorganization phase (α)
    Reorganization,
}

/// Panarchy decision input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyDecision {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub context: DecisionContext,
    pub objectives: Vec<Objective>,
    pub constraints: Vec<Constraint>,
    pub urgency: f64,
    pub impact: f64,
    pub uncertainty: f64,
}

/// Decision context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    pub market_state: MarketContext,
    pub system_state: SystemContext,
    pub historical_performance: PerformanceContext,
}

/// Market context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    pub volatility: f64,
    pub trend_strength: f64,
    pub liquidity: f64,
    pub regime: String,
}

/// System context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemContext {
    pub resource_utilization: f64,
    pub active_scales: Vec<ScaleLevel>,
    pub current_phase: AdaptiveCyclePhase,
}

/// Performance context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContext {
    pub recent_success_rate: f64,
    pub adaptive_capacity_used: f64,
    pub resilience_score: f64,
}

/// Decision objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub name: String,
    pub weight: f64,
    pub target_value: f64,
    pub optimization_direction: OptimizationDirection,
}

/// Optimization direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Maximize,
    Minimize,
    Target,
}

/// Decision constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub value: f64,
}

/// Constraint type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConstraintType {
    LessThan,
    GreaterThan,
    Equal,
    Range { min: f64, max: f64 },
}

/// Routed decision with scale assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedDecision {
    pub decision: PanarchyDecision,
    pub assigned_scale: ScaleLevel,
    pub routing_score: f64,
    pub priority: f64,
}

/// Decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub decision_id: String,
    pub timestamp: DateTime<Utc>,
    pub scale_level: ScaleLevel,
    pub success: bool,
    pub actions: Vec<Action>,
    pub metrics: DecisionMetrics,
    pub cross_scale_effects: CrossScaleEffects,
    pub errors: Vec<String>,
}

/// Action to be taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: ActionType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub confidence: f64,
    pub expected_impact: f64,
}

/// Action type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Trade(TradeAction),
    Adjust(AdjustmentAction),
    Monitor(MonitorAction),
    Transition(TransitionAction),
}

/// Trade action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAction {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub order_type: OrderType,
}

/// Trade side
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit { price: f64 },
    Stop { price: f64 },
}

/// Adjustment action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentAction {
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub reason: String,
}

/// Monitor action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorAction {
    pub target: String,
    pub metrics: Vec<String>,
    pub duration: std::time::Duration,
    pub alert_thresholds: HashMap<String, f64>,
}

/// Transition action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionAction {
    pub from_scale: ScaleLevel,
    pub to_scale: ScaleLevel,
    pub trigger: String,
    pub preparation_steps: Vec<String>,
}

/// Decision metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetrics {
    pub processing_time_ms: u64,
    pub confidence_score: f64,
    pub resource_usage: f64,
    pub adaptation_rate: f64,
}

/// Cross-scale effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossScaleEffects {
    pub upward_effects: Vec<ScaleEffect>,
    pub downward_effects: Vec<ScaleEffect>,
    pub lateral_effects: Vec<ScaleEffect>,
}

/// Scale effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleEffect {
    pub target_scale: ScaleLevel,
    pub effect_type: String,
    pub magnitude: f64,
    pub delay: std::time::Duration,
}

/// PADS system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsStatus {
    pub scale_status: ScaleStatus,
    pub routing_status: RoutingStatus,
    pub comm_status: CommunicationStatus,
    pub resilience_status: ResilienceStatus,
    pub metrics: SystemMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Scale management status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleStatus {
    pub active_scales: HashMap<ScaleLevel, ScaleInfo>,
    pub current_phase: AdaptiveCyclePhase,
    pub transition_state: Option<TransitionState>,
}

/// Scale information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleInfo {
    pub level: ScaleLevel,
    pub active_decisions: usize,
    pub capacity_used: f64,
    pub performance: f64,
}

/// Transition state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionState {
    pub from: ScaleLevel,
    pub to: ScaleLevel,
    pub progress: f64,
    pub estimated_completion: DateTime<Utc>,
}

/// Routing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStatus {
    pub queue_sizes: HashMap<ScaleLevel, usize>,
    pub routing_latency_ms: f64,
    pub load_balance: HashMap<ScaleLevel, f64>,
}

/// Communication status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStatus {
    pub active_channels: usize,
    pub message_rate: f64,
    pub error_rate: f64,
    pub avg_latency_ms: f64,
}

/// Resilience status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceStatus {
    pub circuit_breaker_state: CircuitState,
    pub health_score: f64,
    pub recovery_capacity: f64,
    pub fault_tolerance_level: f64,
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub decisions_per_second: f64,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
}

/// Panarchy input for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyInput {
    pub data: serde_json::Value,
    pub input_type: InputType,
    pub source: String,
}

/// Input type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    MarketData,
    SystemEvent,
    UserRequest,
    Feedback,
}

/// Panarchy output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyOutput {
    pub result: serde_json::Value,
    pub metadata: OutputMetadata,
}

/// Output metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub scale_used: ScaleLevel,
    pub processing_time_ms: u64,
    pub confidence: f64,
}

/// Adaptive feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFeedback {
    pub performance_score: f64,
    pub adaptation_suggestions: Vec<String>,
    pub scale_adjustments: HashMap<ScaleLevel, f64>,
}

/// Panarchy system trait
#[async_trait]
pub trait PanarchySystem: Send + Sync {
    /// Process input through the panarchy system
    async fn process(&self, input: PanarchyInput) -> Result<PanarchyOutput>;
    
    /// Adapt system based on feedback
    async fn adapt(&self, feedback: AdaptiveFeedback) -> Result<()>;
    
    /// Get current adaptive capacity
    async fn get_adaptive_capacity(&self) -> Result<f64>;
}

impl PanarchyDecision {
    /// Create from input
    pub fn from_input(input: PanarchyInput) -> Result<Self> {
        // Implementation would parse input and create decision
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            context: DecisionContext {
                market_state: MarketContext {
                    volatility: 0.5,
                    trend_strength: 0.3,
                    liquidity: 0.8,
                    regime: "normal".to_string(),
                },
                system_state: SystemContext {
                    resource_utilization: 0.6,
                    active_scales: vec![ScaleLevel::Micro],
                    current_phase: AdaptiveCyclePhase::Growth,
                },
                historical_performance: PerformanceContext {
                    recent_success_rate: 0.75,
                    adaptive_capacity_used: 0.4,
                    resilience_score: 0.8,
                },
            },
            objectives: vec![],
            constraints: vec![],
            urgency: 0.5,
            impact: 0.5,
            uncertainty: 0.3,
        })
    }
    
    /// Test decision for unit tests
    #[cfg(test)]
    pub fn test_decision() -> Self {
        Self::from_input(PanarchyInput {
            data: serde_json::Value::Null,
            input_type: InputType::SystemEvent,
            source: "test".to_string(),
        }).unwrap()
    }
}

impl DecisionResult {
    /// Check if has upward effects
    pub fn has_upward_effects(&self) -> bool {
        !self.cross_scale_effects.upward_effects.is_empty()
    }
    
    /// Check if has downward effects
    pub fn has_downward_effects(&self) -> bool {
        !self.cross_scale_effects.downward_effects.is_empty()
    }
    
    /// Get upward effects
    pub fn get_upward_effects(&self) -> &[ScaleEffect] {
        &self.cross_scale_effects.upward_effects
    }
    
    /// Get downward effects
    pub fn get_downward_effects(&self) -> &[ScaleEffect] {
        &self.cross_scale_effects.downward_effects
    }
    
    /// Get macro insights
    pub fn get_macro_insights(&self) -> Option<MacroInsights> {
        if self.scale_level == ScaleLevel::Macro {
            Some(MacroInsights {
                strategic_direction: "adaptive".to_string(),
                innovation_opportunities: vec![],
                risk_adjustments: HashMap::new(),
            })
        } else {
            None
        }
    }
    
    /// Convert to output
    pub fn into_output(self) -> PanarchyOutput {
        PanarchyOutput {
            result: serde_json::to_value(&self).unwrap_or_default(),
            metadata: OutputMetadata {
                scale_used: self.scale_level,
                processing_time_ms: self.metrics.processing_time_ms,
                confidence: self.metrics.confidence_score,
            },
        }
    }
}

/// Macro-scale insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInsights {
    pub strategic_direction: String,
    pub innovation_opportunities: Vec<String>,
    pub risk_adjustments: HashMap<String, f64>,
}