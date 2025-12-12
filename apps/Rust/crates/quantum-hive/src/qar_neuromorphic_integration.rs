//! QAR-Neuromorphic Integration Bridge
//! 
//! This module bridges the gap between neuromorphic modules and QAR,
//! empowering the Quantum Queen with neuromorphic intelligence.

use crate::quantum_queen::QuantumQueen;
use crate::orchestrator::{QuantumHiveOrchestrator, TradingSignal};
use quantum_agentic_reasoning::{QuantumAgenticReasoning as QAR, ExpertOpinion, DecisionContext};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use std::collections::HashMap;

/// Neuromorphic signal that integrates with QAR
#[derive(Debug, Clone)]
pub struct NeuromorphicSignal {
    /// Aggregated prediction from all neuromorphic modules
    pub prediction: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Individual module contributions
    pub module_contributions: HashMap<String, ModuleContribution>,
    /// Spike patterns detected
    pub spike_patterns: Vec<bool>,
    /// Temporal coherence score
    pub temporal_coherence: f64,
    /// Functional optimization score
    pub functional_optimization: f64,
}

/// Individual module contribution to the signal
#[derive(Debug, Clone)]
pub struct ModuleContribution {
    pub module_name: String,
    pub prediction: f64,
    pub confidence: f64,
    pub processing_time_us: u64,
}

impl QuantumQueen {
    /// Integrate neuromorphic signal into QAR decision-making
    /// This is the CRITICAL method that empowers QAR with neuromorphic intelligence
    pub async fn integrate_neuromorphic_signal(&mut self, signal: NeuromorphicSignal) -> Result<()> {
        info!("🧠 Integrating neuromorphic signal into QAR with confidence: {:.2}", signal.confidence);
        
        // Convert neuromorphic signal to QAR expert opinions
        let expert_opinions = self.convert_to_expert_opinions(&signal)?;
        
        // Get QAR instance
        let mut qar = self.qar.write().unwrap();
        
        // Feed expert opinions to QAR
        for opinion in expert_opinions {
            qar.add_expert_opinion(opinion).await?;
        }
        
        // Update QAR's decision context with neuromorphic insights
        let context = self.create_neuromorphic_context(&signal)?;
        qar.update_decision_context(context).await?;
        
        // Store neuromorphic state for future reference
        self.store_neuromorphic_state(&signal).await?;
        
        debug!("✅ Neuromorphic signal successfully integrated into QAR");
        Ok(())
    }
    
    /// Convert neuromorphic signal to QAR expert opinions
    fn convert_to_expert_opinions(&self, signal: &NeuromorphicSignal) -> Result<Vec<ExpertOpinion>> {
        let mut opinions = Vec::new();
        
        // Create expert opinion for each neuromorphic module
        for (module_name, contribution) in &signal.module_contributions {
            let opinion = ExpertOpinion {
                expert_id: format!("neuromorphic_{}", module_name),
                prediction: contribution.prediction,
                confidence: contribution.confidence,
                reasoning: self.generate_neuromorphic_reasoning(module_name, contribution),
                metadata: self.create_module_metadata(module_name, contribution),
            };
            opinions.push(opinion);
        }
        
        // Create aggregated neuromorphic expert opinion
        let aggregated_opinion = ExpertOpinion {
            expert_id: "neuromorphic_ensemble".to_string(),
            prediction: signal.prediction,
            confidence: signal.confidence,
            reasoning: format!(
                "Neuromorphic ensemble prediction based on {} modules. \
                Temporal coherence: {:.2}, Functional optimization: {:.2}",
                signal.module_contributions.len(),
                signal.temporal_coherence,
                signal.functional_optimization
            ),
            metadata: HashMap::from([
                ("temporal_coherence".to_string(), signal.temporal_coherence.to_string()),
                ("functional_optimization".to_string(), signal.functional_optimization.to_string()),
                ("spike_pattern_density".to_string(), 
                 (signal.spike_patterns.iter().filter(|&&x| x).count() as f64 / 
                  signal.spike_patterns.len() as f64).to_string()),
            ]),
        };
        opinions.push(aggregated_opinion);
        
        Ok(opinions)
    }
    
    /// Generate reasoning explanation for a neuromorphic module
    fn generate_neuromorphic_reasoning(&self, module_name: &str, contribution: &ModuleContribution) -> String {
        match module_name {
            "ceflann_elm" => format!(
                "CEFLANN-ELM functional expansion detected {} signal with {:.1}% confidence. \
                Ultra-fast analytical training indicates {} market direction.",
                if contribution.prediction > 0.0 { "bullish" } else { "bearish" },
                contribution.confidence * 100.0,
                if contribution.prediction > 0.0 { "upward" } else { "downward" }
            ),
            "quantum_snn" => format!(
                "Quantum Cerebellar SNN spike patterns suggest {} momentum with {:.1}% confidence. \
                Temporal processing latency: {}μs indicates {} market conditions.",
                if contribution.prediction > 0.0 { "positive" } else { "negative" },
                contribution.confidence * 100.0,
                contribution.processing_time_us,
                if contribution.processing_time_us < 100 { "calm" } else { "volatile" }
            ),
            "cerflann_norse" => format!(
                "CERFLANN Norse neuromorphic processing shows {} trend with {:.1}% confidence. \
                Spiking neural dynamics indicate {} pattern formation.",
                if contribution.prediction > 0.0 { "ascending" } else { "descending" },
                contribution.confidence * 100.0,
                if contribution.confidence > 0.7 { "strong" } else { "weak" }
            ),
            "cerflann_jax" => format!(
                "CERFLANN JAX functional optimization yields {} signal with {:.1}% confidence. \
                JAX-accelerated computation suggests {} opportunity.",
                if contribution.prediction > 0.0 { "long" } else { "short" },
                contribution.confidence * 100.0,
                if contribution.confidence > 0.8 { "high-probability" } else { "moderate" }
            ),
            _ => format!("{} module prediction: {:.4}", module_name, contribution.prediction),
        }
    }
    
    /// Create metadata for module contribution
    fn create_module_metadata(&self, module_name: &str, contribution: &ModuleContribution) -> HashMap<String, String> {
        HashMap::from([
            ("module".to_string(), module_name.to_string()),
            ("latency_us".to_string(), contribution.processing_time_us.to_string()),
            ("signal_strength".to_string(), contribution.prediction.abs().to_string()),
        ])
    }
    
    /// Create neuromorphic context for QAR
    fn create_neuromorphic_context(&self, signal: &NeuromorphicSignal) -> Result<DecisionContext> {
        Ok(DecisionContext {
            market_regime: self.market_regime.clone(),
            volatility_regime: if signal.temporal_coherence < 0.3 { 
                "high_volatility".to_string() 
            } else { 
                "low_volatility".to_string() 
            },
            risk_tolerance: 1.0 - (signal.spike_patterns.iter().filter(|&&x| x).count() as f64 / 
                                  signal.spike_patterns.len().max(1) as f64),
            time_horizon: "short_term".to_string(),
            neuromorphic_insights: Some(HashMap::from([
                ("spike_density".to_string(), 
                 (signal.spike_patterns.iter().filter(|&&x| x).count() as f64).to_string()),
                ("temporal_coherence".to_string(), signal.temporal_coherence.to_string()),
                ("functional_optimization".to_string(), signal.functional_optimization.to_string()),
                ("module_consensus".to_string(), 
                 self.calculate_module_consensus(signal).to_string()),
            ])),
        })
    }
    
    /// Calculate consensus among neuromorphic modules
    fn calculate_module_consensus(&self, signal: &NeuromorphicSignal) -> f64 {
        if signal.module_contributions.is_empty() {
            return 0.0;
        }
        
        let predictions: Vec<f64> = signal.module_contributions.values()
            .map(|c| c.prediction)
            .collect();
        
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        // Lower variance = higher consensus
        1.0 / (1.0 + variance.sqrt())
    }
    
    /// Store neuromorphic state for persistence and analysis
    async fn store_neuromorphic_state(&self, signal: &NeuromorphicSignal) -> Result<()> {
        // In a real implementation, this would store to persistent memory
        // For now, we'll just log it
        info!("Storing neuromorphic state - Prediction: {:.4}, Confidence: {:.2}%", 
              signal.prediction, signal.confidence * 100.0);
        Ok(())
    }
}

/// Enhanced orchestration that feeds into QAR
impl QuantumHiveOrchestrator {
    /// Process market data and empower QAR with neuromorphic intelligence
    pub async fn process_and_empower_qar(
        &self, 
        market_data: MarketData,
        quantum_queen: &mut QuantumQueen
    ) -> Result<TradeAction> {
        // First, get neuromorphic signal
        let trading_signal = self.process_market_data(market_data.clone()).await?;
        
        // Convert trading signal to neuromorphic signal
        let neuromorphic_signal = self.convert_to_neuromorphic_signal(&trading_signal)?;
        
        // Integrate with QAR
        quantum_queen.integrate_neuromorphic_signal(neuromorphic_signal).await?;
        
        // Now QAR makes the final decision with neuromorphic intelligence
        quantum_queen.make_enhanced_decision(&market_data).await
    }
    
    /// Convert trading signal to neuromorphic signal for QAR
    fn convert_to_neuromorphic_signal(&self, signal: &TradingSignal) -> Result<NeuromorphicSignal> {
        let mut module_contributions = HashMap::new();
        
        // Extract individual module contributions
        for contributor in &signal.contributors {
            module_contributions.insert(
                contributor.module.clone(),
                ModuleContribution {
                    module_name: contributor.module.clone(),
                    prediction: contributor.prediction,
                    confidence: contributor.confidence,
                    processing_time_us: contributor.processing_time_us,
                }
            );
        }
        
        // Calculate aggregated metrics
        let prediction = match &signal.signal {
            SignalType::StrongBuy(s) | SignalType::Buy(s) => *s,
            SignalType::StrongSell(s) | SignalType::Sell(s) => -*s,
            SignalType::Hold => 0.0,
            SignalType::EmergencyExit => -1.0,
        };
        
        Ok(NeuromorphicSignal {
            prediction,
            confidence: signal.confidence,
            module_contributions,
            spike_patterns: vec![], // Would be populated from actual spike data
            temporal_coherence: 0.8, // Would be calculated from temporal analysis
            functional_optimization: 0.9, // Would be from JAX optimization metrics
        })
    }
}

/// Enhanced decision making with neuromorphic intelligence
impl QuantumQueen {
    /// Make enhanced decision using both QAR and neuromorphic intelligence
    pub async fn make_enhanced_decision(&self, market_data: &MarketData) -> Result<TradeAction> {
        let qar = self.qar.read().unwrap();
        
        // QAR now has neuromorphic expert opinions integrated
        let quantum_decision = qar.make_sovereign_decision(market_data).await?;
        
        // Apply additional safety checks from whale defense and black swan detection
        let final_decision = self.apply_risk_overlays(quantum_decision).await?;
        
        info!("👑 QAR Sovereign Decision (Neuromorphic-Enhanced): {:?}", final_decision);
        Ok(final_decision)
    }
    
    /// Apply risk overlays from whale defense and black swan detection
    async fn apply_risk_overlays(&self, decision: TradeAction) -> Result<TradeAction> {
        let mut final_decision = decision;
        
        // Check whale defense
        if let Some(ref whale_defense) = *self.whale_defense_core.read().unwrap() {
            if whale_defense.detect_manipulation_risk() > 0.7 {
                warn!("🐋 Whale manipulation detected! Reducing position size");
                final_decision.quantity *= 0.5;
                final_decision.confidence *= 0.7;
            }
        }
        
        // Check black swan risk
        if let Some(ref black_swan) = *self.black_swan_detector.read().unwrap() {
            if black_swan.get_tail_risk_probability() > 0.1 {
                warn!("🦢 Black Swan risk elevated! Applying defensive measures");
                final_decision.risk_factor = final_decision.risk_factor.max(0.8);
                if matches!(final_decision.action_type, ActionType::Buy) {
                    final_decision.action_type = ActionType::Hedge;
                }
            }
        }
        
        Ok(final_decision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neuromorphic_integration() {
        let mut queen = QuantumQueen::new();
        
        let signal = NeuromorphicSignal {
            prediction: 0.75,
            confidence: 0.85,
            module_contributions: HashMap::from([
                ("ceflann_elm".to_string(), ModuleContribution {
                    module_name: "ceflann_elm".to_string(),
                    prediction: 0.8,
                    confidence: 0.9,
                    processing_time_us: 50,
                }),
            ]),
            spike_patterns: vec![true, false, true, true],
            temporal_coherence: 0.7,
            functional_optimization: 0.8,
        };
        
        let result = queen.integrate_neuromorphic_signal(signal).await;
        assert!(result.is_ok());
    }
}