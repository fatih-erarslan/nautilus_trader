//! CDFA Integration Module for QAR
//! 
//! Integrates Consensus Data Fusion Algorithms into the Quantum Agentic Reasoning system
//! to enable multi-source data fusion, consensus mechanisms, and cross-scale validation.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

// Import CDFA core functionality
use cdfa_core::{
    CdfaFusion, FusionMethod, FusionParams,
    pearson_correlation, spearman_correlation, kendall_tau,
    jensen_shannon_divergence, dynamic_time_warping
};

// Use our simplified adapters instead of the full crates
use crate::boardroom_adapter::{
    Boardroom, BoardroomConfig, ConsensusProtocol, ConsensusResult
};

use crate::pads_adapter::{
    PadsConnector, IntegrationConfig
};

use crate::{QARError, Result, MarketData, TradingDecision, TradingAction};

/// CDFA Integration Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdfaIntegrationConfig {
    /// Enable multi-source data fusion
    pub enable_fusion: bool,
    
    /// Fusion method to use
    pub fusion_method: String,
    
    /// Minimum consensus threshold (0.0 to 1.0)
    pub consensus_threshold: f64,
    
    /// Enable cross-scale validation via PADS
    pub enable_cross_scale: bool,
    
    /// Number of agents for consensus
    pub num_consensus_agents: usize,
    
    /// Diversity threshold for source acceptance
    pub diversity_threshold: f64,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for CdfaIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            fusion_method: "adaptive".to_string(),
            consensus_threshold: 0.7,
            enable_cross_scale: true,
            num_consensus_agents: 5,
            diversity_threshold: 0.3,
            enable_monitoring: true,
        }
    }
}

/// CDFA Integration Component for QAR
pub struct CdfaIntegration {
    config: CdfaIntegrationConfig,
    boardroom: Arc<Mutex<Boardroom>>,
    pads_connector: Arc<Mutex<PadsConnector>>,
    performance_metrics: Arc<Mutex<CdfaPerformanceMetrics>>,
    data_sources: HashMap<String, DataSource>,
}

/// Data source for CDFA fusion
#[derive(Debug, Clone)]
struct DataSource {
    id: String,
    scores: Array1<f64>,
    confidence: f64,
    timestamp: u64,
}

/// Performance metrics for CDFA integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdfaPerformanceMetrics {
    pub total_fusions: u64,
    pub successful_consensus: u64,
    pub failed_consensus: u64,
    pub average_diversity: f64,
    pub average_fusion_time_us: u64,
    pub cross_scale_validations: u64,
}

impl CdfaIntegration {
    /// Create new CDFA integration component
    pub fn new(config: CdfaIntegrationConfig) -> Result<Self> {
        // Initialize Boardroom for multi-agent consensus
        let boardroom_config = BoardroomConfig {
            consensus_protocol: ConsensusProtocol::Byzantine,
            min_agents: config.num_consensus_agents,
            timeout_ms: 100,
        };
        let boardroom = Arc::new(Mutex::new(
            Boardroom::new(boardroom_config).map_err(|e| QARError::DecisionEngine {
                message: format!("Failed to create boardroom: {}", e)
            })?
        ));
        
        // Initialize PADS connector for cross-scale integration
        let pads_config = IntegrationConfig {
            enable_monitoring: config.enable_monitoring,
            scale_levels: 3,
            decision_timeout_ms: 50,
        };
        let pads_connector = Arc::new(Mutex::new(
            PadsConnector::new(pads_config).map_err(|e| QARError::DecisionEngine {
                message: format!("Failed to create PADS connector: {}", e)
            })?
        ));
        
        Ok(Self {
            config,
            boardroom,
            pads_connector,
            performance_metrics: Arc::new(Mutex::new(CdfaPerformanceMetrics::default())),
            data_sources: HashMap::new(),
        })
    }
    
    /// Add a data source for fusion
    pub fn add_data_source(&mut self, id: String, scores: Vec<f64>, confidence: f64) {
        let source = DataSource {
            id: id.clone(),
            scores: Array1::from_vec(scores),
            confidence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        self.data_sources.insert(id, source);
    }
    
    /// Perform CDFA fusion on available data sources
    pub fn fuse_data_sources(&self) -> Result<FusionResult> {
        let start = std::time::Instant::now();
        
        if self.data_sources.len() < 2 {
            return Err(QARError::DecisionEngine {
                message: "Insufficient data sources for fusion".to_string()
            });
        }
        
        // Convert data sources to matrix for fusion
        let mut scores_matrix = Vec::new();
        let mut weights = Vec::new();
        
        for (_, source) in &self.data_sources {
            scores_matrix.push(source.scores.clone());
            weights.push(source.confidence);
        }
        
        // Check diversity between sources
        let diversity = self.calculate_diversity(&scores_matrix)?;
        if diversity < self.config.diversity_threshold {
            return Err(QARError::DecisionEngine {
                message: format!("Insufficient diversity: {} < {}", diversity, self.config.diversity_threshold)
            });
        }
        
        // Convert to ndarray for CDFA
        let num_sources = scores_matrix.len();
        let num_items = scores_matrix[0].len();
        let mut fusion_input = Array2::zeros((num_sources, num_items));
        
        for (i, scores) in scores_matrix.iter().enumerate() {
            for (j, &score) in scores.iter().enumerate() {
                fusion_input[[i, j]] = score;
            }
        }
        
        // Select fusion method
        let fusion_method = match self.config.fusion_method.as_str() {
            "average" => FusionMethod::Average,
            "weighted" => FusionMethod::WeightedAverage,
            "borda" => FusionMethod::BordaCount,
            "adaptive" => FusionMethod::AdaptiveScore,
            _ => FusionMethod::Average,
        };
        
        // Perform fusion
        let fusion_params = FusionParams {
            weights: Some(Array1::from_vec(weights)),
            normalize: true,
            adaptive_threshold: Some(0.5),
        };
        
        let fused_scores = CdfaFusion::fuse(
            &fusion_input.view(),
            fusion_method,
            Some(fusion_params)
        ).map_err(|e| QARError::DecisionEngine {
            message: format!("CDFA fusion failed: {}", e)
        })?;
        
        // Get consensus via Boardroom
        let consensus = self.get_boardroom_consensus(&fused_scores)?;
        
        // Cross-scale validation via PADS if enabled
        let cross_scale_valid = if self.config.enable_cross_scale {
            self.validate_cross_scale(&fused_scores, &consensus)?
        } else {
            true
        };
        
        // Update metrics
        let elapsed = start.elapsed();
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.total_fusions += 1;
            if consensus.confidence >= self.config.consensus_threshold {
                metrics.successful_consensus += 1;
            } else {
                metrics.failed_consensus += 1;
            }
            metrics.average_diversity = 
                (metrics.average_diversity * (metrics.total_fusions - 1) as f64 + diversity) 
                / metrics.total_fusions as f64;
            metrics.average_fusion_time_us = 
                (metrics.average_fusion_time_us * (metrics.total_fusions - 1) + elapsed.as_micros() as u64) 
                / metrics.total_fusions;
            if cross_scale_valid {
                metrics.cross_scale_validations += 1;
            }
        }
        
        Ok(FusionResult {
            fused_scores,
            consensus,
            diversity_score: diversity,
            cross_scale_valid,
            fusion_time_us: elapsed.as_micros() as u64,
        })
    }
    
    /// Calculate diversity between data sources
    fn calculate_diversity(&self, sources: &[Array1<f64>]) -> Result<f64> {
        if sources.len() < 2 {
            return Ok(1.0);
        }
        
        let mut total_diversity = 0.0;
        let mut pairs = 0;
        
        for i in 0..sources.len() {
            for j in i+1..sources.len() {
                // Use multiple diversity metrics
                let pearson = pearson_correlation(&sources[i], &sources[j])
                    .unwrap_or(0.0);
                let spearman = spearman_correlation(&sources[i], &sources[j])
                    .unwrap_or(0.0);
                let kendall = kendall_tau(&sources[i], &sources[j])
                    .unwrap_or(0.0);
                
                // Diversity is 1 - average correlation
                let avg_correlation = (pearson.abs() + spearman.abs() + kendall.abs()) / 3.0;
                total_diversity += 1.0 - avg_correlation;
                pairs += 1;
            }
        }
        
        Ok(total_diversity / pairs as f64)
    }
    
    /// Get consensus from Boardroom agents
    fn get_boardroom_consensus(&self, fused_scores: &Array1<f64>) -> Result<ConsensusResult> {
        let boardroom = self.boardroom.lock().map_err(|_| QARError::DecisionEngine {
            message: "Failed to lock boardroom".to_string()
        })?;
        
        // Create consensus request
        let consensus_data = HashMap::from([
            ("fused_scores".to_string(), format!("{:?}", fused_scores)),
            ("threshold".to_string(), self.config.consensus_threshold.to_string()),
        ]);
        
        // Get consensus from agents
        boardroom.get_consensus(consensus_data).map_err(|e| QARError::DecisionEngine {
            message: format!("Boardroom consensus failed: {}", e)
        })
    }
    
    /// Validate fusion results across scales using PADS
    fn validate_cross_scale(&self, fused_scores: &Array1<f64>, consensus: &ConsensusResult) -> Result<bool> {
        let pads = self.pads_connector.lock().map_err(|_| QARError::DecisionEngine {
            message: "Failed to lock PADS connector".to_string()
        })?;
        
        // Validate across different time scales
        let validation_data = HashMap::from([
            ("fused_scores".to_string(), format!("{:?}", fused_scores)),
            ("consensus_confidence".to_string(), consensus.confidence.to_string()),
        ]);
        
        pads.validate_across_scales(validation_data).map_err(|e| QARError::DecisionEngine {
            message: format!("Cross-scale validation failed: {}", e)
        })
    }
    
    /// Apply CDFA to trading decision
    pub fn enhance_trading_decision(&mut self, 
                                  market_data: &MarketData,
                                  base_decision: &TradingDecision,
                                  additional_signals: HashMap<String, Vec<f64>>) -> Result<TradingDecision> {
        // Add base decision as a data source
        self.add_data_source(
            "base_qar".to_string(),
            vec![base_decision.confidence, base_decision.prospect_value],
            base_decision.confidence
        );
        
        // Add additional signals as data sources
        for (source_id, scores) in additional_signals {
            self.add_data_source(source_id, scores, 0.8); // Default confidence
        }
        
        // Perform fusion
        let fusion_result = self.fuse_data_sources()?;
        
        // Create enhanced decision
        let mut enhanced_decision = base_decision.clone();
        
        // Adjust confidence based on consensus
        enhanced_decision.confidence = 
            (base_decision.confidence + fusion_result.consensus.confidence) / 2.0;
        
        // Add CDFA reasoning
        enhanced_decision.reasoning.push(format!(
            "CDFA fusion with {} sources, diversity: {:.3}, consensus: {:.3}",
            self.data_sources.len(),
            fusion_result.diversity_score,
            fusion_result.consensus.confidence
        ));
        
        // Clear data sources for next decision
        self.data_sources.clear();
        
        Ok(enhanced_decision)
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> CdfaPerformanceMetrics {
        self.performance_metrics.lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }
}

/// Result of CDFA fusion
#[derive(Debug, Clone)]
pub struct FusionResult {
    pub fused_scores: Array1<f64>,
    pub consensus: ConsensusResult,
    pub diversity_score: f64,
    pub cross_scale_valid: bool,
    pub fusion_time_us: u64,
}

impl Default for CdfaPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_fusions: 0,
            successful_consensus: 0,
            failed_consensus: 0,
            average_diversity: 0.0,
            average_fusion_time_us: 0,
            cross_scale_validations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cdfa_integration_creation() {
        let config = CdfaIntegrationConfig::default();
        let integration = CdfaIntegration::new(config);
        assert!(integration.is_ok());
    }
    
    #[test]
    fn test_add_data_source() {
        let config = CdfaIntegrationConfig::default();
        let mut integration = CdfaIntegration::new(config).unwrap();
        
        integration.add_data_source(
            "test_source".to_string(),
            vec![0.8, 0.6, 0.7, 0.9],
            0.85
        );
        
        assert_eq!(integration.data_sources.len(), 1);
    }
    
    #[test]
    fn test_diversity_calculation() {
        let config = CdfaIntegrationConfig::default();
        let integration = CdfaIntegration::new(config).unwrap();
        
        let sources = vec![
            Array1::from_vec(vec![0.8, 0.6, 0.7, 0.9]),
            Array1::from_vec(vec![0.7, 0.8, 0.6, 0.85]),
            Array1::from_vec(vec![0.2, 0.3, 0.9, 0.1]),
        ];
        
        let diversity = integration.calculate_diversity(&sources).unwrap();
        assert!(diversity > 0.0 && diversity <= 1.0);
    }
}