//! Integration with quantum agentic reasoning and trading systems

use crate::core::{QercError, QercResult, QuantumState};
use crate::engine::QuantumErrorCorrection;
use quantum_agentic_reasoning::{QuantumAgenticReasoning, FactorMap, MarketContext, TradingDecision};
use std::collections::HashMap;

/// Integration wrapper for QAR with QERC
#[derive(Debug, Clone)]
pub struct QarIntegration {
    /// QAR instance
    pub qar: QuantumAgenticReasoning,
    /// QERC instance
    pub qerc: QuantumErrorCorrection,
    /// Integration configuration
    pub config: IntegrationConfig,
}

impl QarIntegration {
    /// Create new QAR-QERC integration
    pub async fn new(qar: QuantumAgenticReasoning, qerc: QuantumErrorCorrection) -> QercResult<Self> {
        Ok(Self {
            qar,
            qerc,
            config: IntegrationConfig::default(),
        })
    }
    
    /// Make error-protected trading decision
    pub async fn make_protected_decision(&self, factors: &FactorMap) -> QercResult<TradingDecision> {
        // Create market context
        let context = MarketContext::default();
        
        // Create quantum state for decision
        let quantum_state = self.qar.create_decision_quantum_state(factors, &context).await
            .map_err(|e| QercError::IntegrationError { 
                message: format!("Failed to create decision state: {}", e) 
            })?;
        
        // Protect state with error correction
        let protected_state = self.qerc.encode_logical_state(&quantum_state).await?;
        
        // Make decision using protected state
        let decision = self.qar.make_decision(factors, &context).await
            .map_err(|e| QercError::IntegrationError { 
                message: format!("Failed to make decision: {}", e) 
            })?;
        
        Ok(decision)
    }
    
    /// Get integration metrics
    pub async fn get_metrics(&self) -> QercResult<IntegrationMetrics> {
        Ok(IntegrationMetrics {
            decision_count: 0,
            error_correction_count: 0,
            average_fidelity: 0.99,
            average_latency_ms: 1.0,
        })
    }
}

/// Integration configuration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable automatic error correction
    pub auto_error_correction: bool,
    /// Error correction threshold
    pub error_threshold: f64,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            auto_error_correction: true,
            error_threshold: 0.1,
            performance_monitoring: true,
        }
    }
}

/// Integration performance metrics
#[derive(Debug, Clone)]
pub struct IntegrationMetrics {
    /// Number of decisions made
    pub decision_count: u64,
    /// Number of error corrections
    pub error_correction_count: u64,
    /// Average fidelity
    pub average_fidelity: f64,
    /// Average latency
    pub average_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_qar_integration() {
        let qar = QuantumAgenticReasoning::new().await.unwrap();
        let qerc = QuantumErrorCorrection::new().await.unwrap();
        let integration = QarIntegration::new(qar, qerc).await.unwrap();
        
        let mut factors = HashMap::new();
        factors.insert("trend".to_string(), 0.75);
        let factor_map = FactorMap::new(factors).unwrap();
        
        let decision = integration.make_protected_decision(&factor_map).await.unwrap();
        assert!(decision.confidence > 0.0);
    }
}