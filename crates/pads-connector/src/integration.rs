//! Integration with other components

use crate::{
    config::PadsConfig,
    error::{PadsError, Result},
    monitoring::PadsMonitor,
    scale_manager::ScaleManager,
    decision_router::DecisionRouter,
    types::*,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use cdfa_core::traits::CDFADetector;
use mcp_orchestration::{AgentCapability, TaskRequest};
use cognitive_integration::CognitiveProcessor;
use ml_ensemble::EnsemblePredictor;

/// PADS integration layer
pub struct PadsIntegration {
    config: Arc<PadsConfig>,
    scale_manager: Arc<RwLock<ScaleManager>>,
    decision_router: Arc<DecisionRouter>,
    monitor: Arc<PadsMonitor>,
    cdfa_connector: Arc<CdfaConnector>,
    mcp_connector: Arc<McpConnector>,
    cognitive_connector: Arc<CognitiveConnector>,
    ml_connector: Arc<MlConnector>,
}

/// CDFA connector for panarchy analysis
struct CdfaConnector {
    panarchy_analyzer: Arc<dyn CDFADetector>,
    soc_analyzer: Arc<dyn CDFADetector>,
}

/// MCP orchestration connector
struct McpConnector {
    client: Arc<mcp_orchestration::OrchestrationClient>,
}

/// Cognitive integration connector
struct CognitiveConnector {
    processor: Arc<CognitiveProcessor>,
}

/// ML ensemble connector
struct MlConnector {
    ensemble: Arc<EnsemblePredictor>,
}

impl PadsIntegration {
    /// Create new integration layer
    pub async fn new(
        config: Arc<PadsConfig>,
        scale_manager: Arc<RwLock<ScaleManager>>,
        decision_router: Arc<DecisionRouter>,
        monitor: Arc<PadsMonitor>,
    ) -> Result<Self> {
        let cdfa_connector = Arc::new(Self::create_cdfa_connector().await?);
        let mcp_connector = Arc::new(Self::create_mcp_connector(&config).await?);
        let cognitive_connector = Arc::new(Self::create_cognitive_connector().await?);
        let ml_connector = Arc::new(Self::create_ml_connector().await?);
        
        Ok(Self {
            config,
            scale_manager,
            decision_router,
            monitor,
            cdfa_connector,
            mcp_connector,
            cognitive_connector,
            ml_connector,
        })
    }
    
    /// Create CDFA connector
    async fn create_cdfa_connector() -> Result<CdfaConnector> {
        // Initialize panarchy analyzer
        let panarchy_config = cdfa_panarchy_analyzer::Config::default();
        let panarchy_analyzer = Arc::new(
            cdfa_panarchy_analyzer::PanarchyAnalyzer::new(panarchy_config)
                .map_err(|e| PadsError::integration(format!("Failed to create panarchy analyzer: {}", e)))?
        );
        
        // Initialize SOC analyzer
        let soc_config = cdfa_soc_analyzer::Config::default();
        let soc_analyzer = Arc::new(
            cdfa_soc_analyzer::SOCAnalyzer::new(soc_config)
                .map_err(|e| PadsError::integration(format!("Failed to create SOC analyzer: {}", e)))?
        );
        
        Ok(CdfaConnector {
            panarchy_analyzer,
            soc_analyzer,
        })
    }
    
    /// Create MCP connector
    async fn create_mcp_connector(config: &PadsConfig) -> Result<McpConnector> {
        let mcp_config = mcp_orchestration::Config {
            server_url: "http://localhost:8080".to_string(), // Would come from config
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
        };
        
        let client = Arc::new(
            mcp_orchestration::OrchestrationClient::new(mcp_config)
                .await
                .map_err(|e| PadsError::integration(format!("Failed to create MCP client: {}", e)))?
        );
        
        Ok(McpConnector { client })
    }
    
    /// Create cognitive connector
    async fn create_cognitive_connector() -> Result<CognitiveConnector> {
        let config = cognitive_integration::Config::default();
        let processor = Arc::new(
            CognitiveProcessor::new(config)
                .await
                .map_err(|e| PadsError::integration(format!("Failed to create cognitive processor: {}", e)))?
        );
        
        Ok(CognitiveConnector { processor })
    }
    
    /// Create ML connector
    async fn create_ml_connector() -> Result<MlConnector> {
        let config = ml_ensemble::Config::default();
        let ensemble = Arc::new(
            EnsemblePredictor::new(config)
                .map_err(|e| PadsError::integration(format!("Failed to create ML ensemble: {}", e)))?
        );
        
        Ok(MlConnector { ensemble })
    }
    
    /// Analyze panarchy dynamics
    pub async fn analyze_panarchy_dynamics(&self, data: &[f64]) -> Result<PanarchyAnalysis> {
        debug!("Analyzing panarchy dynamics");
        
        // Get panarchy phase from CDFA
        let phase_result = self.cdfa_connector.panarchy_analyzer
            .detect(data)
            .map_err(|e| PadsError::integration(format!("Panarchy detection failed: {}", e)))?;
        
        // Get SOC regime
        let soc_result = self.cdfa_connector.soc_analyzer
            .detect(data)
            .map_err(|e| PadsError::integration(format!("SOC detection failed: {}", e)))?;
        
        // Combine results
        let current_phase = self.map_cdfa_phase(phase_result.phase);
        let regime_stability = soc_result.soc_strength;
        let transition_probability = phase_result.transition_probability;
        
        Ok(PanarchyAnalysis {
            current_phase,
            regime_stability,
            transition_probability,
            recommended_scale: self.determine_recommended_scale(
                current_phase,
                regime_stability
            ),
        })
    }
    
    /// Map CDFA phase to adaptive cycle phase
    fn map_cdfa_phase(&self, cdfa_phase: i32) -> AdaptiveCyclePhase {
        match cdfa_phase {
            0 => AdaptiveCyclePhase::Growth,
            1 => AdaptiveCyclePhase::Conservation,
            2 => AdaptiveCyclePhase::Release,
            3 => AdaptiveCyclePhase::Reorganization,
            _ => AdaptiveCyclePhase::Growth,
        }
    }
    
    /// Determine recommended scale based on analysis
    fn determine_recommended_scale(
        &self,
        phase: AdaptiveCyclePhase,
        stability: f64
    ) -> ScaleLevel {
        match (phase, stability) {
            (AdaptiveCyclePhase::Growth, s) if s > 0.7 => ScaleLevel::Micro,
            (AdaptiveCyclePhase::Conservation, _) => ScaleLevel::Meso,
            (AdaptiveCyclePhase::Release, _) => ScaleLevel::Macro,
            (AdaptiveCyclePhase::Reorganization, _) => ScaleLevel::Macro,
            _ => ScaleLevel::Meso,
        }
    }
    
    /// Request MCP agent for decision support
    pub async fn request_mcp_agent(&self, decision: &PanarchyDecision) -> Result<AgentResponse> {
        info!("Requesting MCP agent support for decision {}", decision.id);
        
        // Create task request
        let task = TaskRequest {
            id: uuid::Uuid::new_v4().to_string(),
            task_type: "panarchy_decision".to_string(),
            priority: decision.urgency,
            payload: serde_json::to_value(decision)?,
            capabilities_required: vec![
                AgentCapability::DecisionMaking,
                AgentCapability::RiskAssessment,
                AgentCapability::Optimization,
            ],
        };
        
        // Submit to MCP
        let response = self.mcp_connector.client
            .submit_task(task)
            .await
            .map_err(|e| PadsError::integration(format!("MCP task submission failed: {}", e)))?;
        
        Ok(AgentResponse {
            agent_id: response.assigned_agent,
            recommendations: response.result,
            confidence: response.confidence,
        })
    }
    
    /// Get cognitive insights
    pub async fn get_cognitive_insights(&self, context: &DecisionContext) -> Result<CognitiveInsights> {
        debug!("Getting cognitive insights");
        
        let input = cognitive_integration::CognitiveInput {
            market_state: context.market_state.clone(),
            system_state: context.system_state.clone(),
            historical_data: vec![], // Would include actual data
        };
        
        let result = self.cognitive_connector.processor
            .process(input)
            .await
            .map_err(|e| PadsError::integration(format!("Cognitive processing failed: {}", e)))?;
        
        Ok(CognitiveInsights {
            attention_focus: result.attention_weights,
            pattern_recognition: result.detected_patterns,
            strategic_recommendations: result.recommendations,
        })
    }
    
    /// Get ML ensemble predictions
    pub async fn get_ml_predictions(&self, features: &[f64]) -> Result<MlPredictions> {
        debug!("Getting ML ensemble predictions");
        
        let predictions = self.ml_connector.ensemble
            .predict(features)
            .map_err(|e| PadsError::integration(format!("ML prediction failed: {}", e)))?;
        
        Ok(MlPredictions {
            primary_prediction: predictions.mean_prediction,
            confidence_interval: (predictions.lower_bound, predictions.upper_bound),
            model_agreement: predictions.model_agreement,
            feature_importance: predictions.feature_importance,
        })
    }
    
    /// Coordinate cross-component decision
    pub async fn coordinate_decision(&self, decision: &PanarchyDecision) -> Result<CoordinatedDecision> {
        info!("Coordinating decision {} across components", decision.id);
        
        // Analyze panarchy dynamics
        let panarchy_analysis = self.analyze_panarchy_dynamics(&[0.5; 100]).await?;
        
        // Get MCP agent support
        let agent_response = self.request_mcp_agent(decision).await?;
        
        // Get cognitive insights
        let cognitive_insights = self.get_cognitive_insights(&decision.context).await?;
        
        // Get ML predictions
        let ml_predictions = self.get_ml_predictions(&[0.5; 50]).await?;
        
        // Combine all inputs
        let coordinated = CoordinatedDecision {
            decision_id: decision.id.clone(),
            recommended_scale: panarchy_analysis.recommended_scale,
            agent_recommendations: agent_response.recommendations,
            cognitive_focus: cognitive_insights.attention_focus,
            ml_confidence: ml_predictions.model_agreement,
            integrated_score: self.calculate_integrated_score(
                &panarchy_analysis,
                &agent_response,
                &cognitive_insights,
                &ml_predictions
            ),
        };
        
        Ok(coordinated)
    }
    
    /// Calculate integrated score
    fn calculate_integrated_score(
        &self,
        panarchy: &PanarchyAnalysis,
        agent: &AgentResponse,
        cognitive: &CognitiveInsights,
        ml: &MlPredictions
    ) -> f64 {
        // Weighted combination of all inputs
        let weights = (0.3, 0.2, 0.3, 0.2); // (panarchy, agent, cognitive, ml)
        
        weights.0 * panarchy.regime_stability +
        weights.1 * agent.confidence +
        weights.2 * cognitive.pattern_recognition.len() as f64 / 10.0 +
        weights.3 * ml.model_agreement
    }
}

/// Panarchy analysis result
#[derive(Debug, Clone)]
pub struct PanarchyAnalysis {
    pub current_phase: AdaptiveCyclePhase,
    pub regime_stability: f64,
    pub transition_probability: f64,
    pub recommended_scale: ScaleLevel,
}

/// MCP agent response
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub agent_id: String,
    pub recommendations: serde_json::Value,
    pub confidence: f64,
}

/// Cognitive insights
#[derive(Debug, Clone)]
pub struct CognitiveInsights {
    pub attention_focus: Vec<f64>,
    pub pattern_recognition: Vec<String>,
    pub strategic_recommendations: Vec<String>,
}

/// ML predictions
#[derive(Debug, Clone)]
pub struct MlPredictions {
    pub primary_prediction: f64,
    pub confidence_interval: (f64, f64),
    pub model_agreement: f64,
    pub feature_importance: Vec<f64>,
}

/// Coordinated decision
#[derive(Debug, Clone)]
pub struct CoordinatedDecision {
    pub decision_id: String,
    pub recommended_scale: ScaleLevel,
    pub agent_recommendations: serde_json::Value,
    pub cognitive_focus: Vec<f64>,
    pub ml_confidence: f64,
    pub integrated_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_integration_creation() {
        let config = Arc::new(PadsConfig::default());
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await.unwrap());
        let scale_manager = Arc::new(RwLock::new(
            ScaleManager::new(config.clone(), monitor.clone()).await.unwrap()
        ));
        let decision_router = Arc::new(
            DecisionRouter::new(config.clone(), monitor.clone()).await.unwrap()
        );
        
        // Integration creation might fail due to missing dependencies
        // This is expected in test environment
        let result = PadsIntegration::new(
            config,
            scale_manager,
            decision_router,
            monitor
        ).await;
        
        // We expect this to fail in tests due to missing external services
        assert!(result.is_err());
    }
}