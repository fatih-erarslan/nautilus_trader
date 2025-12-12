//! # Advanced Integration Framework for Neuro Trader
//!
//! Multi-transport messaging, signal taxonomy, and adaptive coordination
//! Ported from Python PADS with enhancements for Rust performance

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot, broadcast};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use dashmap::DashMap;
use uuid::Uuid;

pub mod messaging;
pub mod adapters;
pub mod signals;
pub mod coordination;
pub mod feedback_loops;

use messaging::*;
use adapters::*;
use signals::*;
use coordination::*;
use feedback_loops::*;

/// Main integration framework coordinator
#[derive(Debug)]
pub struct IntegrationFramework {
    /// Framework configuration
    config: IntegrationConfig,
    
    /// Multi-transport messaging system
    messaging_system: Arc<RwLock<MultiTransportMessaging>>,
    
    /// Component adapters
    adapters: Arc<RwLock<HashMap<String, Box<dyn ComponentAdapter + Send + Sync>>>>,
    
    /// Signal taxonomy processor
    signal_processor: Arc<RwLock<SignalTaxonomyProcessor>>,
    
    /// Coordination manager
    coordination_manager: Arc<RwLock<CoordinationManager>>,
    
    /// Feedback loop system
    feedback_system: Arc<RwLock<FeedbackLoopSystem>>,
    
    /// Integration metrics
    metrics: Arc<RwLock<IntegrationMetrics>>,
    
    /// Framework state
    state: Arc<RwLock<FrameworkState>>,
    
    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
}

/// Integration framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Messaging configuration
    pub messaging: MessagingConfig,
    
    /// Component adapter configurations
    pub adapters: HashMap<String, AdapterConfig>,
    
    /// Signal processing configuration
    pub signals: SignalConfig,
    
    /// Coordination configuration
    pub coordination: CoordinationConfig,
    
    /// Feedback configuration
    pub feedback: FeedbackConfig,
    
    /// Performance monitoring
    pub performance_monitoring: bool,
    
    /// Health check interval
    pub health_check_interval: Duration,
    
    /// Metrics collection interval
    pub metrics_interval: Duration,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            messaging: MessagingConfig::default(),
            adapters: HashMap::new(),
            signals: SignalConfig::default(),
            coordination: CoordinationConfig::default(),
            feedback: FeedbackConfig::default(),
            performance_monitoring: true,
            health_check_interval: Duration::from_secs(30),
            metrics_interval: Duration::from_secs(10),
        }
    }
}

/// Framework state
#[derive(Debug, Clone)]
pub struct FrameworkState {
    /// Current state
    pub state: IntegrationState,
    
    /// Start time
    pub start_time: Instant,
    
    /// Active components
    pub active_components: HashMap<String, ComponentState>,
    
    /// Error count
    pub error_count: u64,
    
    /// Performance score
    pub performance_score: f64,
}

/// Integration state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationState {
    Initializing,
    Running,
    Degraded,
    Stopping,
    Stopped,
    Error,
}

/// Component state
#[derive(Debug, Clone)]
pub struct ComponentState {
    pub component_id: String,
    pub state: ComponentStatus,
    pub last_activity: Instant,
    pub message_count: u64,
    pub error_count: u64,
    pub performance_score: f64,
}

/// Component status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentStatus {
    Active,
    Idle,
    Degraded,
    Failed,
    Reconnecting,
}

/// Integration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total messages processed
    pub total_messages: u64,
    
    /// Messages by type
    pub messages_by_type: HashMap<String, u64>,
    
    /// Messages by component
    pub messages_by_component: HashMap<String, u64>,
    
    /// Processing latency
    pub avg_processing_latency: Duration,
    
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    
    /// Component performance
    pub component_performance: HashMap<String, f64>,
    
    /// Signal classification accuracy
    pub signal_classification_accuracy: f64,
    
    /// Coordination effectiveness
    pub coordination_effectiveness: f64,
    
    /// Feedback loop responsiveness
    pub feedback_responsiveness: f64,
    
    /// Last update
    pub last_updated: Instant,
}

impl Default for IntegrationMetrics {
    fn default() -> Self {
        Self {
            total_messages: 0,
            messages_by_type: HashMap::new(),
            messages_by_component: HashMap::new(),
            avg_processing_latency: Duration::from_millis(0),
            error_rates: HashMap::new(),
            component_performance: HashMap::new(),
            signal_classification_accuracy: 0.0,
            coordination_effectiveness: 0.0,
            feedback_responsiveness: 0.0,
            last_updated: Instant::now(),
        }
    }
}

impl IntegrationFramework {
    /// Create new integration framework
    #[instrument(skip(config))]
    pub async fn new(config: IntegrationConfig) -> Result<Self> {
        info!("Initializing integration framework");
        
        // Initialize messaging system
        let messaging_system = Arc::new(RwLock::new(
            MultiTransportMessaging::new(config.messaging.clone()).await?
        ));
        
        // Initialize adapters
        let adapters = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize signal processor
        let signal_processor = Arc::new(RwLock::new(
            SignalTaxonomyProcessor::new(config.signals.clone()).await?
        ));
        
        // Initialize coordination manager
        let coordination_manager = Arc::new(RwLock::new(
            CoordinationManager::new(config.coordination.clone()).await?
        ));
        
        // Initialize feedback system
        let feedback_system = Arc::new(RwLock::new(
            FeedbackLoopSystem::new(config.feedback.clone()).await?
        ));
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(IntegrationMetrics::default()));
        
        // Initialize state
        let state = Arc::new(RwLock::new(FrameworkState {
            state: IntegrationState::Initializing,
            start_time: Instant::now(),
            active_components: HashMap::new(),
            error_count: 0,
            performance_score: 0.0,
        }));
        
        // Create shutdown channel
        let (shutdown_tx, _) = broadcast::channel(1);
        
        info!("Integration framework initialized successfully");
        
        Ok(Self {
            config,
            messaging_system,
            adapters,
            signal_processor,
            coordination_manager,
            feedback_system,
            metrics,
            state,
            shutdown_tx,
        })
    }
    
    /// Start the integration framework
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting integration framework");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.state = IntegrationState::Running;
        }
        
        // Start messaging system
        self.messaging_system.write().await.start().await?;
        
        // Start signal processor
        self.signal_processor.write().await.start().await?;
        
        // Start coordination manager
        self.coordination_manager.write().await.start().await?;
        
        // Start feedback system
        self.feedback_system.write().await.start().await?;
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        info!("Integration framework started successfully");
        Ok(())
    }
    
    /// Stop the integration framework
    #[instrument(skip(self))]
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping integration framework");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.state = IntegrationState::Stopping;
        }
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Stop all systems
        self.messaging_system.write().await.stop().await?;
        self.signal_processor.write().await.stop().await?;
        self.coordination_manager.write().await.stop().await?;
        self.feedback_system.write().await.stop().await?;
        
        // Stop adapters
        let adapters = self.adapters.read().await;
        for adapter in adapters.values() {
            adapter.stop().await?;
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.state = IntegrationState::Stopped;
        }
        
        info!("Integration framework stopped successfully");
        Ok(())
    }
    
    /// Register a component adapter
    #[instrument(skip(self, adapter))]
    pub async fn register_adapter(&mut self, component_id: String, adapter: Box<dyn ComponentAdapter + Send + Sync>) -> Result<()> {
        info!("Registering adapter for component: {}", component_id);
        
        // Initialize adapter
        adapter.initialize().await?;
        
        // Start adapter
        adapter.start().await?;
        
        // Register with messaging system
        self.messaging_system.write().await.register_component(&component_id).await?;
        
        // Store adapter
        self.adapters.write().await.insert(component_id.clone(), adapter);
        
        // Update component state
        {
            let mut state = self.state.write().await;
            state.active_components.insert(component_id.clone(), ComponentState {
                component_id: component_id.clone(),
                state: ComponentStatus::Active,
                last_activity: Instant::now(),
                message_count: 0,
                error_count: 0,
                performance_score: 1.0,
            });
        }
        
        info!("Adapter registered successfully for component: {}", component_id);
        Ok(())
    }
    
    /// Process integration signal
    #[instrument(skip(self, signal))]
    pub async fn process_signal(&mut self, signal: IntegrationSignal) -> Result<SignalProcessingResult> {
        debug!("Processing integration signal: {:?}", signal.signal_type);
        
        let start_time = Instant::now();
        
        // Classify signal
        let classification = self.signal_processor.write().await.classify_signal(&signal).await?;
        
        // Route signal based on classification
        let routing_decision = self.route_signal(&signal, &classification).await?;
        
        // Process through appropriate adapters
        let mut processing_results = Vec::new();
        for target_component in &routing_decision.target_components {
            if let Some(adapter) = self.adapters.read().await.get(target_component) {
                match adapter.process_signal(&signal).await {
                    Ok(result) => processing_results.push(result),
                    Err(e) => {
                        error!("Error processing signal in component {}: {}", target_component, e);
                        self.record_error(target_component.clone(), e).await;
                    }
                }
            }
        }
        
        // Update coordination state
        self.coordination_manager.write().await.update_coordination_state(&signal).await?;
        
        // Record feedback
        let feedback = SignalFeedback {
            signal_id: signal.id.clone(),
            processing_time: start_time.elapsed(),
            classification_accuracy: classification.confidence,
            routing_effectiveness: routing_decision.confidence,
            component_responses: processing_results.clone(),
        };
        
        self.feedback_system.write().await.record_feedback(feedback).await?;
        
        // Update metrics
        self.update_metrics(&signal, start_time.elapsed()).await;
        
        Ok(SignalProcessingResult {
            signal_id: signal.id,
            classification,
            routing_decision,
            processing_results,
            processing_time: start_time.elapsed(),
        })
    }
    
    /// Route signal to appropriate components
    async fn route_signal(&self, signal: &IntegrationSignal, classification: &SignalClassification) -> Result<RoutingDecision> {
        // Get routing strategy based on signal type
        let routing_strategy = self.get_routing_strategy(&classification.signal_type).await;
        
        // Determine target components
        let target_components = match routing_strategy {
            RoutingStrategy::Broadcast => {
                // Send to all active components
                self.state.read().await.active_components.keys().cloned().collect()
            }
            RoutingStrategy::Selective => {
                // Send to specific components based on signal content
                self.select_target_components(signal, classification).await
            }
            RoutingStrategy::Priority => {
                // Send to highest priority component first
                self.select_priority_components(signal, classification).await
            }
        };
        
        Ok(RoutingDecision {
            target_components,
            routing_strategy,
            confidence: classification.confidence,
            metadata: HashMap::new(),
        })
    }
    
    /// Get routing strategy for signal type
    async fn get_routing_strategy(&self, signal_type: &SignalType) -> RoutingStrategy {
        match signal_type {
            SignalType::MarketData => RoutingStrategy::Broadcast,
            SignalType::TradingSignal => RoutingStrategy::Selective,
            SignalType::RiskAlert => RoutingStrategy::Priority,
            SignalType::SystemEvent => RoutingStrategy::Broadcast,
            SignalType::ConfigUpdate => RoutingStrategy::Broadcast,
            SignalType::PerformanceMetric => RoutingStrategy::Selective,
            _ => RoutingStrategy::Selective,
        }
    }
    
    /// Select target components based on signal content
    async fn select_target_components(&self, signal: &IntegrationSignal, classification: &SignalClassification) -> Vec<String> {
        let mut targets = Vec::new();
        
        // Add logic to select components based on signal content
        // This would be customized based on your specific components
        
        // Example: Route trading signals to trading components
        if matches!(classification.signal_type, SignalType::TradingSignal) {
            targets.extend(vec![
                "trading_engine".to_string(),
                "risk_manager".to_string(),
                "portfolio_manager".to_string(),
            ]);
        }
        
        // Example: Route market data to all analysis components
        if matches!(classification.signal_type, SignalType::MarketData) {
            targets.extend(vec![
                "technical_analyzer".to_string(),
                "fundamental_analyzer".to_string(),
                "sentiment_analyzer".to_string(),
            ]);
        }
        
        // Filter to only include registered components
        let active_components: Vec<String> = self.state.read().await.active_components.keys().cloned().collect();
        targets.retain(|component| active_components.contains(component));
        
        targets
    }
    
    /// Select priority components for critical signals
    async fn select_priority_components(&self, signal: &IntegrationSignal, classification: &SignalClassification) -> Vec<String> {
        let mut targets = Vec::new();
        
        // Add highest priority components first
        if signal.priority > 0.8 {
            targets.push("risk_manager".to_string());
        }
        
        if signal.priority > 0.6 {
            targets.push("trading_engine".to_string());
        }
        
        if signal.priority > 0.4 {
            targets.push("portfolio_manager".to_string());
        }
        
        // Filter to only include registered components
        let active_components: Vec<String> = self.state.read().await.active_components.keys().cloned().collect();
        targets.retain(|component| active_components.contains(component));
        
        targets
    }
    
    /// Start background tasks
    async fn start_background_tasks(&self) -> Result<()> {
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        // Start performance monitoring
        if self.config.performance_monitoring {
            self.start_performance_monitoring().await?;
        }
        
        Ok(())
    }
    
    /// Start health monitoring task
    async fn start_health_monitoring(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let adapters = Arc::clone(&self.adapters);
        let interval = self.config.health_check_interval;
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        // Check component health
                        let adapters_guard = adapters.read().await;
                        for (component_id, adapter) in adapters_guard.iter() {
                            match adapter.health_check().await {
                                Ok(health) => {
                                    // Update component state
                                    let mut state_guard = state.write().await;
                                    if let Some(component_state) = state_guard.active_components.get_mut(component_id) {
                                        component_state.state = if health.is_healthy() {
                                            ComponentStatus::Active
                                        } else {
                                            ComponentStatus::Degraded
                                        };
                                        component_state.performance_score = health.performance_score();
                                    }
                                }
                                Err(e) => {
                                    error!("Health check failed for component {}: {}", component_id, e);
                                    let mut state_guard = state.write().await;
                                    if let Some(component_state) = state_guard.active_components.get_mut(component_id) {
                                        component_state.state = ComponentStatus::Failed;
                                        component_state.error_count += 1;
                                    }
                                }
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start metrics collection task
    async fn start_metrics_collection(&self) -> Result<()> {
        let metrics = Arc::clone(&self.metrics);
        let state = Arc::clone(&self.state);
        let interval = self.config.metrics_interval;
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        // Collect metrics
                        let mut metrics_guard = metrics.write().await;
                        let state_guard = state.read().await;
                        
                        // Update component performance metrics
                        for (component_id, component_state) in &state_guard.active_components {
                            metrics_guard.component_performance.insert(
                                component_id.clone(),
                                component_state.performance_score
                            );
                        }
                        
                        // Calculate overall performance
                        let total_errors = state_guard.error_count;
                        let total_messages = metrics_guard.total_messages;
                        
                        if total_messages > 0 {
                            let error_rate = total_errors as f64 / total_messages as f64;
                            metrics_guard.error_rates.insert("overall".to_string(), error_rate);
                        }
                        
                        metrics_guard.last_updated = Instant::now();
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start performance monitoring task
    async fn start_performance_monitoring(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let coordination_manager = Arc::clone(&self.coordination_manager);
        let feedback_system = Arc::clone(&self.feedback_system);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        // Analyze performance trends
                        let coordination_effectiveness = coordination_manager.read().await.get_effectiveness().await.unwrap_or(0.0);
                        let feedback_responsiveness = feedback_system.read().await.get_responsiveness().await.unwrap_or(0.0);
                        
                        // Update overall performance score
                        let mut state_guard = state.write().await;
                        state_guard.performance_score = (coordination_effectiveness + feedback_responsiveness) / 2.0;
                        
                        // Check for performance degradation
                        if state_guard.performance_score < 0.5 {
                            state_guard.state = IntegrationState::Degraded;
                            warn!("Integration framework performance degraded: {}", state_guard.performance_score);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Update metrics
    async fn update_metrics(&self, signal: &IntegrationSignal, processing_time: Duration) {
        let mut metrics = self.metrics.write().await;
        
        // Update total messages
        metrics.total_messages += 1;
        
        // Update messages by type
        let signal_type_str = format!("{:?}", signal.signal_type);
        *metrics.messages_by_type.entry(signal_type_str).or_insert(0) += 1;
        
        // Update processing latency
        let total_latency = metrics.avg_processing_latency.as_millis() as u64 * (metrics.total_messages - 1) +
                           processing_time.as_millis() as u64;
        metrics.avg_processing_latency = Duration::from_millis(total_latency / metrics.total_messages);
        
        metrics.last_updated = Instant::now();
    }
    
    /// Record error for component
    async fn record_error(&self, component_id: String, error: anyhow::Error) {
        let mut state = self.state.write().await;
        state.error_count += 1;
        
        if let Some(component_state) = state.active_components.get_mut(&component_id) {
            component_state.error_count += 1;
        }
        
        error!("Component {} error: {}", component_id, error);
    }
    
    /// Get framework status
    pub async fn get_status(&self) -> FrameworkStatus {
        let state = self.state.read().await;
        let metrics = self.metrics.read().await;
        
        FrameworkStatus {
            state: state.state,
            uptime: state.start_time.elapsed(),
            active_components: state.active_components.len(),
            total_messages: metrics.total_messages,
            error_count: state.error_count,
            performance_score: state.performance_score,
            avg_processing_latency: metrics.avg_processing_latency,
            last_updated: metrics.last_updated,
        }
    }
    
    /// Get detailed metrics
    pub async fn get_metrics(&self) -> IntegrationMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get component states
    pub async fn get_component_states(&self) -> HashMap<String, ComponentState> {
        self.state.read().await.active_components.clone()
    }
}

/// Framework status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkStatus {
    pub state: IntegrationState,
    pub uptime: Duration,
    pub active_components: usize,
    pub total_messages: u64,
    pub error_count: u64,
    pub performance_score: f64,
    pub avg_processing_latency: Duration,
    pub last_updated: Instant,
}

/// Signal processing result
#[derive(Debug, Clone)]
pub struct SignalProcessingResult {
    pub signal_id: String,
    pub classification: SignalClassification,
    pub routing_decision: RoutingDecision,
    pub processing_results: Vec<ComponentProcessingResult>,
    pub processing_time: Duration,
}

/// Routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub target_components: Vec<String>,
    pub routing_strategy: RoutingStrategy,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

/// Routing strategy
#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    Broadcast,
    Selective,
    Priority,
}

/// Component processing result
#[derive(Debug, Clone)]
pub struct ComponentProcessingResult {
    pub component_id: String,
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub processing_time: Duration,
}

/// Signal feedback
#[derive(Debug, Clone)]
pub struct SignalFeedback {
    pub signal_id: String,
    pub processing_time: Duration,
    pub classification_accuracy: f64,
    pub routing_effectiveness: f64,
    pub component_responses: Vec<ComponentProcessingResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_integration_framework_creation() {
        let config = IntegrationConfig::default();
        let framework = IntegrationFramework::new(config).await;
        assert!(framework.is_ok());
    }
    
    #[tokio::test]
    async fn test_framework_lifecycle() {
        let config = IntegrationConfig::default();
        let mut framework = IntegrationFramework::new(config).await.unwrap();
        
        // Test start
        assert!(framework.start().await.is_ok());
        
        // Check status
        let status = framework.get_status().await;
        assert_eq!(status.state, IntegrationState::Running);
        
        // Test stop
        assert!(framework.stop().await.is_ok());
        
        let status = framework.get_status().await;
        assert_eq!(status.state, IntegrationState::Stopped);
    }
}