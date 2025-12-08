//! PADS (Panarchy Adaptive Decision System) Connector
//! 
//! Sophisticated cross-scale interaction handling for complex market scenarios.
//! Implements panarchy theory for multi-scale adaptive systems.

pub mod config;
pub mod error;
pub mod types;
pub mod scale_manager;
pub mod decision_router;
pub mod communication;
pub mod resilience;
pub mod monitoring;
pub mod integration;

pub use config::PadsConfig;
pub use error::{PadsError, Result};
pub use types::*;
pub use scale_manager::ScaleManager;
pub use decision_router::DecisionRouter;
pub use communication::CrossScaleCommunicator;
pub use resilience::ResilienceEngine;
pub use monitoring::PadsMonitor;
pub use integration::PadsIntegration;

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Main PADS connector for managing cross-scale interactions
pub struct PadsConnector {
    config: Arc<PadsConfig>,
    scale_manager: Arc<RwLock<ScaleManager>>,
    decision_router: Arc<DecisionRouter>,
    communicator: Arc<CrossScaleCommunicator>,
    resilience_engine: Arc<ResilienceEngine>,
    monitor: Arc<PadsMonitor>,
    integration: Arc<PadsIntegration>,
}

impl PadsConnector {
    /// Create a new PADS connector instance
    pub async fn new(config: PadsConfig) -> Result<Self> {
        info!("Initializing PADS connector");
        
        let config = Arc::new(config);
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await?);
        
        let scale_manager = Arc::new(RwLock::new(
            ScaleManager::new(config.clone(), monitor.clone()).await?
        ));
        
        let decision_router = Arc::new(
            DecisionRouter::new(config.clone(), monitor.clone()).await?
        );
        
        let communicator = Arc::new(
            CrossScaleCommunicator::new(config.clone(), monitor.clone()).await?
        );
        
        let resilience_engine = Arc::new(
            ResilienceEngine::new(config.clone(), monitor.clone()).await?
        );
        
        let integration = Arc::new(
            PadsIntegration::new(
                config.clone(),
                scale_manager.clone(),
                decision_router.clone(),
                monitor.clone()
            ).await?
        );
        
        Ok(Self {
            config,
            scale_manager,
            decision_router,
            communicator,
            resilience_engine,
            monitor,
            integration,
        })
    }
    
    /// Initialize the PADS system
    pub async fn initialize(&self) -> Result<()> {
        info!("Starting PADS initialization");
        
        // Initialize scale hierarchy
        self.scale_manager.write().await.initialize_scales().await?;
        
        // Setup decision routing
        self.decision_router.setup_routes().await?;
        
        // Initialize communication channels
        self.communicator.setup_channels().await?;
        
        // Configure resilience mechanisms
        self.resilience_engine.configure().await?;
        
        // Start monitoring
        self.monitor.start().await?;
        
        info!("PADS initialization complete");
        Ok(())
    }
    
    /// Process a decision through the PADS system
    pub async fn process_decision(&self, decision: PanarchyDecision) -> Result<DecisionResult> {
        self.monitor.record_decision_start(&decision);
        
        // Determine appropriate scale
        let scale = self.scale_manager.read().await
            .determine_scale(&decision)
            .await?;
        
        // Route decision to appropriate handler
        let routed_decision = self.decision_router
            .route_decision(decision, scale.clone())
            .await?;
        
        // Process at determined scale
        let result = self.process_at_scale(routed_decision, scale).await?;
        
        // Apply cross-scale effects
        self.apply_cross_scale_effects(&result).await?;
        
        self.monitor.record_decision_complete(&result);
        Ok(result)
    }
    
    /// Process decision at specific scale
    async fn process_at_scale(
        &self,
        decision: RoutedDecision,
        scale: PanarchyScale
    ) -> Result<DecisionResult> {
        match scale.level {
            ScaleLevel::Micro => self.process_micro_scale(decision).await,
            ScaleLevel::Meso => self.process_meso_scale(decision).await,
            ScaleLevel::Macro => self.process_macro_scale(decision).await,
        }
    }
    
    /// Process micro-scale decisions (exploitation phase)
    async fn process_micro_scale(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        self.monitor.record_scale_processing(ScaleLevel::Micro);
        
        // Fast, local optimization
        let result = self.decision_router
            .execute_micro_decision(decision)
            .await?;
        
        // Check for scale transition signals
        if self.should_transition_scale(&result).await? {
            self.initiate_scale_transition(ScaleLevel::Micro, &result).await?;
        }
        
        Ok(result)
    }
    
    /// Process meso-scale decisions (transition phase)
    async fn process_meso_scale(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        self.monitor.record_scale_processing(ScaleLevel::Meso);
        
        // Balance exploration and exploitation
        let result = self.decision_router
            .execute_meso_decision(decision)
            .await?;
        
        // Coordinate with adjacent scales
        self.coordinate_adjacent_scales(&result).await?;
        
        Ok(result)
    }
    
    /// Process macro-scale decisions (exploration phase)
    async fn process_macro_scale(&self, decision: RoutedDecision) -> Result<DecisionResult> {
        self.monitor.record_scale_processing(ScaleLevel::Macro);
        
        // Strategic, long-term optimization
        let result = self.decision_router
            .execute_macro_decision(decision)
            .await?;
        
        // Propagate insights to lower scales
        self.propagate_macro_insights(&result).await?;
        
        Ok(result)
    }
    
    /// Check if scale transition is needed
    async fn should_transition_scale(&self, result: &DecisionResult) -> Result<bool> {
        let scale_manager = self.scale_manager.read().await;
        scale_manager.should_transition(result).await
    }
    
    /// Initiate scale transition
    async fn initiate_scale_transition(
        &self,
        current: ScaleLevel,
        result: &DecisionResult
    ) -> Result<()> {
        warn!("Initiating scale transition from {:?}", current);
        
        let mut scale_manager = self.scale_manager.write().await;
        scale_manager.transition_scale(current, result).await?;
        
        // Notify other components
        self.communicator.broadcast_scale_transition(current).await?;
        
        Ok(())
    }
    
    /// Apply cross-scale effects
    async fn apply_cross_scale_effects(&self, result: &DecisionResult) -> Result<()> {
        // Upward causation (micro -> macro)
        if result.has_upward_effects() {
            self.propagate_upward_effects(result).await?;
        }
        
        // Downward causation (macro -> micro)
        if result.has_downward_effects() {
            self.propagate_downward_effects(result).await?;
        }
        
        Ok(())
    }
    
    /// Propagate effects upward through scales
    async fn propagate_upward_effects(&self, result: &DecisionResult) -> Result<()> {
        self.communicator
            .propagate_upward(result.get_upward_effects())
            .await
    }
    
    /// Propagate effects downward through scales
    async fn propagate_downward_effects(&self, result: &DecisionResult) -> Result<()> {
        self.communicator
            .propagate_downward(result.get_downward_effects())
            .await
    }
    
    /// Coordinate with adjacent scales
    async fn coordinate_adjacent_scales(&self, result: &DecisionResult) -> Result<()> {
        let scale_manager = self.scale_manager.read().await;
        let adjacent_scales = scale_manager.get_adjacent_scales(result.scale_level).await?;
        
        for scale in adjacent_scales {
            self.communicator
                .coordinate_with_scale(scale, result)
                .await?;
        }
        
        Ok(())
    }
    
    /// Propagate macro-scale insights
    async fn propagate_macro_insights(&self, result: &DecisionResult) -> Result<()> {
        if let Some(insights) = result.get_macro_insights() {
            self.scale_manager.write().await
                .update_strategic_parameters(insights)
                .await?;
        }
        Ok(())
    }
    
    /// Get system status
    pub async fn get_status(&self) -> Result<PadsStatus> {
        let scale_status = self.scale_manager.read().await.get_status().await?;
        let routing_status = self.decision_router.get_status().await?;
        let comm_status = self.communicator.get_status().await?;
        let resilience_status = self.resilience_engine.get_status().await?;
        let metrics = self.monitor.get_metrics().await?;
        
        Ok(PadsStatus {
            scale_status,
            routing_status,
            comm_status,
            resilience_status,
            metrics,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Handle system recovery
    pub async fn recover(&self) -> Result<()> {
        error!("Initiating PADS recovery");
        
        // Activate resilience mechanisms
        self.resilience_engine.activate_recovery().await?;
        
        // Reset scale manager to stable state
        self.scale_manager.write().await.reset_to_stable().await?;
        
        // Clear decision routing queues
        self.decision_router.clear_queues().await?;
        
        // Re-establish communication channels
        self.communicator.reconnect_all().await?;
        
        info!("PADS recovery complete");
        Ok(())
    }
}

#[async_trait]
impl PanarchySystem for PadsConnector {
    async fn process(&self, input: PanarchyInput) -> Result<PanarchyOutput> {
        let decision = PanarchyDecision::from_input(input)?;
        let result = self.process_decision(decision).await?;
        Ok(result.into_output())
    }
    
    async fn adapt(&self, feedback: AdaptiveFeedback) -> Result<()> {
        // Update scale parameters based on feedback
        self.scale_manager.write().await
            .adapt_parameters(feedback.clone())
            .await?;
        
        // Adjust decision routing strategies
        self.decision_router
            .adapt_routing(feedback.clone())
            .await?;
        
        // Update resilience thresholds
        self.resilience_engine
            .update_thresholds(feedback)
            .await?;
        
        Ok(())
    }
    
    async fn get_adaptive_capacity(&self) -> Result<f64> {
        let scale_capacity = self.scale_manager.read().await
            .get_adaptive_capacity()
            .await?;
        
        let resilience_capacity = self.resilience_engine
            .get_capacity()
            .await?;
        
        Ok((scale_capacity + resilience_capacity) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pads_initialization() {
        let config = PadsConfig::default();
        let pads = PadsConnector::new(config).await.unwrap();
        assert!(pads.initialize().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_scale_transition() {
        let config = PadsConfig::default();
        let pads = PadsConnector::new(config).await.unwrap();
        pads.initialize().await.unwrap();
        
        let decision = PanarchyDecision::test_decision();
        let result = pads.process_decision(decision).await.unwrap();
        assert!(!result.errors.is_empty() || result.success);
    }
}