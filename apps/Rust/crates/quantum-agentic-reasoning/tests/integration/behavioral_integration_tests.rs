//! Integration tests for Behavioral Decision Enhancement
//! 
//! Tests the behavioral integration module that combines:
//! - QAR base decisions with Prospect Theory enhancements
//! - Cross-component behavioral consistency
//! - Reference point adaptation and framing effects
//! - Performance under various behavioral scenarios

use quantum_agentic_reasoning::*;
use quantum_agentic_reasoning::behavioral_integration::*;
use prospect_theory::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_behavioral_enhancement_integration() {
    let behavioral_config = BehavioralConfig::default();
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    // Create base QAR decision
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.7,
        expected_return: Some(0.05),
        risk_assessment: Some(0.3),
        urgency_score: 0.6,
        reasoning: "Technical analysis suggests uptrend".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    // Create factor map for behavioral analysis
    let mut factors = HashMap::new();
    factors.insert("trend".to_string(), 0.75);
    factors.insert("volatility".to_string(), 0.4);
    factors.insert("momentum".to_string(), 0.65);
    factors.insert("sentiment".to_string(), 0.8);
    
    let factor_map = FactorMap::new(factors).unwrap();
    
    // Create analysis result
    let analysis_result = AnalysisResult {
        trend_strength: 0.75,
        volatility_score: 0.4,
        momentum_indicator: 0.65,
        confidence_level: 0.8,
        supporting_factors: vec!["Strong trend".to_string(), "High momentum".to_string()],
        risk_factors: vec!["Moderate volatility".to_string()],
    };
    
    let enhancement = enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis_result,
        "BTC/USDT"
    ).unwrap();
    
    // Verify enhancement structure
    assert_eq!(enhancement.original_decision.decision_type, DecisionType::Buy);
    assert!(enhancement.enhanced_decision.confidence >= 0.0 && enhancement.enhanced_decision.confidence <= 1.0);
    assert!(enhancement.prospect_theory_decision.confidence >= 0.0 && enhancement.prospect_theory_decision.confidence <= 1.0);
    
    // Behavioral factors should be present
    assert!(enhancement.behavioral_factors.loss_aversion_impact.abs() <= 3.0);
    assert!(enhancement.behavioral_factors.probability_weighting_bias.abs() <= 1.0);
    assert!(enhancement.behavioral_factors.mental_accounting_bias.abs() <= 1.0);
    
    // Reference point should be set
    assert!(enhancement.reference_point.is_some());
    
    // Framing effect should be calculated
    assert!(enhancement.framing_effect.abs() <= 1.0);
}

#[tokio::test]
async fn test_behavioral_consistency_across_scenarios() {
    let behavioral_config = BehavioralConfig::default();
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    // Test gain scenario
    let gain_decision = TradingDecision {
        decision_type: DecisionType::Sell,
        confidence: 0.8,
        expected_return: Some(0.08),
        risk_assessment: Some(0.2),
        urgency_score: 0.7,
        reasoning: "Taking profits on strong gains".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let mut gain_factors = HashMap::new();
    gain_factors.insert("trend".to_string(), 0.9);
    gain_factors.insert("volatility".to_string(), 0.2);
    gain_factors.insert("momentum".to_string(), 0.8);
    gain_factors.insert("sentiment".to_string(), 0.9); // Very positive
    
    let gain_factor_map = FactorMap::new(gain_factors).unwrap();
    let gain_analysis = AnalysisResult {
        trend_strength: 0.9,
        volatility_score: 0.2,
        momentum_indicator: 0.8,
        confidence_level: 0.9,
        supporting_factors: vec!["Strong uptrend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    let gain_enhancement = enhancer.enhance_decision(
        &gain_decision,
        &gain_factor_map,
        &gain_analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Test loss scenario
    let loss_decision = TradingDecision {
        decision_type: DecisionType::Hold,
        confidence: 0.4,
        expected_return: Some(-0.03),
        risk_assessment: Some(0.7),
        urgency_score: 0.3,
        reasoning: "Waiting for better entry point".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let mut loss_factors = HashMap::new();
    loss_factors.insert("trend".to_string(), 0.2);
    loss_factors.insert("volatility".to_string(), 0.8);
    loss_factors.insert("momentum".to_string(), 0.3);
    loss_factors.insert("sentiment".to_string(), 0.2); // Very negative
    
    let loss_factor_map = FactorMap::new(loss_factors).unwrap();
    let loss_analysis = AnalysisResult {
        trend_strength: 0.2,
        volatility_score: 0.8,
        momentum_indicator: 0.3,
        confidence_level: 0.4,
        supporting_factors: vec!["Wait and see".to_string()],
        risk_factors: vec!["High volatility".to_string(), "Weak trend".to_string()],
    };
    
    let loss_enhancement = enhancer.enhance_decision(
        &loss_decision,
        &loss_factor_map,
        &loss_analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Behavioral differences should be evident
    assert_ne!(gain_enhancement.framing_effect, loss_enhancement.framing_effect);
    assert_ne!(gain_enhancement.behavioral_factors.loss_aversion_impact,
               loss_enhancement.behavioral_factors.loss_aversion_impact);
    
    // Gain frame should be positive, loss frame negative
    assert!(gain_enhancement.framing_effect >= 0.0);
    assert!(loss_enhancement.framing_effect <= 0.0);
    
    // Loss aversion should be stronger in loss scenario
    assert!(loss_enhancement.behavioral_factors.loss_aversion_impact.abs() >=
            gain_enhancement.behavioral_factors.loss_aversion_impact.abs());
}

#[tokio::test]
async fn test_reference_point_adaptation() {
    let behavioral_config = BehavioralConfig {
        reference_adaptation_rate: 0.2, // 20% adaptation
        ..Default::default()
    };
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.6,
        expected_return: Some(0.04),
        risk_assessment: Some(0.4),
        urgency_score: 0.5,
        reasoning: "Base decision".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let mut factors = HashMap::new();
    factors.insert("trend".to_string(), 0.6);
    factors.insert("volatility".to_string(), 0.3);
    factors.insert("momentum".to_string(), 0.6);
    factors.insert("sentiment".to_string(), 0.7);
    
    let factor_map = FactorMap::new(factors).unwrap();
    let analysis = AnalysisResult {
        trend_strength: 0.6,
        volatility_score: 0.3,
        momentum_indicator: 0.6,
        confidence_level: 0.7,
        supporting_factors: vec!["Moderate trend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    // First decision should establish reference point
    let first_enhancement = enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    let first_ref_point = first_enhancement.reference_point.unwrap();
    
    // Simulate price increase over time
    let price_sequence = vec![110.0, 115.0, 120.0, 125.0];
    
    for price in price_sequence {
        // Update factors to reflect new price level
        let mut updated_factors = HashMap::new();
        updated_factors.insert("trend".to_string(), 0.8); // Stronger trend
        updated_factors.insert("volatility".to_string(), 0.25);
        updated_factors.insert("momentum".to_string(), 0.75);
        updated_factors.insert("sentiment".to_string(), 0.8);
        
        let updated_factor_map = FactorMap::new(updated_factors).unwrap();
        
        enhancer.enhance_decision(
            &base_decision,
            &updated_factor_map,
            &analysis,
            "BTC/USDT"
        ).unwrap();
    }
    
    // Final decision to check adapted reference point
    let final_enhancement = enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    let final_ref_point = final_enhancement.reference_point.unwrap();
    
    // Reference point should have adapted upward
    assert!(final_ref_point > first_ref_point);
    
    // But not fully to the highest price (should be adaptive, not reactive)
    assert!(final_ref_point < 125.0);
}

#[tokio::test]
async fn test_framing_effect_variations() {
    let behavioral_config = BehavioralConfig {
        enable_framing: true,
        ..Default::default()
    };
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.6,
        expected_return: Some(0.04),
        risk_assessment: Some(0.4),
        urgency_score: 0.5,
        reasoning: "Base decision".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let analysis = AnalysisResult {
        trend_strength: 0.6,
        volatility_score: 0.3,
        momentum_indicator: 0.6,
        confidence_level: 0.7,
        supporting_factors: vec!["Moderate trend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    // Test gain frame (high sentiment)
    let mut gain_factors = HashMap::new();
    gain_factors.insert("trend".to_string(), 0.6);
    gain_factors.insert("volatility".to_string(), 0.3);
    gain_factors.insert("momentum".to_string(), 0.6);
    gain_factors.insert("sentiment".to_string(), 0.9); // High sentiment -> Gain frame
    
    let gain_factor_map = FactorMap::new(gain_factors).unwrap();
    let gain_enhancement = enhancer.enhance_decision(
        &base_decision,
        &gain_factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Test loss frame (low sentiment)
    let mut loss_factors = HashMap::new();
    loss_factors.insert("trend".to_string(), 0.6);
    loss_factors.insert("volatility".to_string(), 0.3);
    loss_factors.insert("momentum".to_string(), 0.6);
    loss_factors.insert("sentiment".to_string(), 0.1); // Low sentiment -> Loss frame
    
    let loss_factor_map = FactorMap::new(loss_factors).unwrap();
    let loss_enhancement = enhancer.enhance_decision(
        &base_decision,
        &loss_factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Test neutral frame (moderate sentiment)
    let mut neutral_factors = HashMap::new();
    neutral_factors.insert("trend".to_string(), 0.6);
    neutral_factors.insert("volatility".to_string(), 0.3);
    neutral_factors.insert("momentum".to_string(), 0.6);
    neutral_factors.insert("sentiment".to_string(), 0.5); // Moderate sentiment -> Neutral frame
    
    let neutral_factor_map = FactorMap::new(neutral_factors).unwrap();
    let neutral_enhancement = enhancer.enhance_decision(
        &base_decision,
        &neutral_factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Framing effects should differ
    assert!(gain_enhancement.framing_effect > 0.0);
    assert!(loss_enhancement.framing_effect < 0.0);
    assert_eq!(neutral_enhancement.framing_effect, 0.0);
    
    // Loss frame effect should be stronger than gain frame (asymmetry)
    assert!(loss_enhancement.framing_effect.abs() > gain_enhancement.framing_effect.abs());
}

#[tokio::test]
async fn test_prospect_theory_weight_blending() {
    // Test with high PT weight
    let high_pt_config = BehavioralConfig {
        prospect_theory_weight: 0.8,
        ..Default::default()
    };
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config.clone()).unwrap();
    
    let mut high_pt_enhancer = BehavioralDecisionEnhancer::new(high_pt_config, prospect_theory);
    
    // Test with low PT weight
    let low_pt_config = BehavioralConfig {
        prospect_theory_weight: 0.2,
        ..Default::default()
    };
    let prospect_theory_low = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut low_pt_enhancer = BehavioralDecisionEnhancer::new(low_pt_config, prospect_theory_low);
    
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.5, // Moderate base confidence
        expected_return: Some(0.03),
        risk_assessment: Some(0.5),
        urgency_score: 0.5,
        reasoning: "Base decision".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let mut factors = HashMap::new();
    factors.insert("trend".to_string(), 0.8);
    factors.insert("volatility".to_string(), 0.3);
    factors.insert("momentum".to_string(), 0.7);
    factors.insert("sentiment".to_string(), 0.8);
    
    let factor_map = FactorMap::new(factors).unwrap();
    let analysis = AnalysisResult {
        trend_strength: 0.8,
        volatility_score: 0.3,
        momentum_indicator: 0.7,
        confidence_level: 0.8,
        supporting_factors: vec!["Strong trend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    let high_pt_enhancement = high_pt_enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    let low_pt_enhancement = low_pt_enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    // Enhanced decisions should differ based on PT weight
    assert_ne!(high_pt_enhancement.enhanced_decision.confidence,
               low_pt_enhancement.enhanced_decision.confidence);
    
    // High PT weight should be more influenced by prospect theory
    let high_pt_confidence_diff = (high_pt_enhancement.enhanced_decision.confidence - 
                                  high_pt_enhancement.original_decision.confidence).abs();
    let low_pt_confidence_diff = (low_pt_enhancement.enhanced_decision.confidence - 
                                 low_pt_enhancement.original_decision.confidence).abs();
    
    // High PT weight should show more deviation from base decision
    assert!(high_pt_confidence_diff >= low_pt_confidence_diff);
}

#[tokio::test]
async fn test_behavioral_metrics_tracking() {
    let behavioral_config = BehavioralConfig::default();
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.6,
        expected_return: Some(0.04),
        risk_assessment: Some(0.4),
        urgency_score: 0.5,
        reasoning: "Base decision".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let analysis = AnalysisResult {
        trend_strength: 0.6,
        volatility_score: 0.3,
        momentum_indicator: 0.6,
        confidence_level: 0.7,
        supporting_factors: vec!["Moderate trend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    let symbols = vec!["BTC/USDT", "ETH/USDT", "SOL/USDT"];
    let frame_types = vec![
        FrameType::Gain,
        FrameType::Loss,
        FrameType::Neutral,
        FrameType::Gain,
        FrameType::Loss,
    ];
    
    // Process multiple decisions to build metrics
    for (i, symbol) in symbols.iter().cycle().take(5).enumerate() {
        let sentiment = match frame_types[i] {
            FrameType::Gain => 0.8,
            FrameType::Loss => 0.2,
            FrameType::Neutral => 0.5,
        };
        
        let mut factors = HashMap::new();
        factors.insert("trend".to_string(), 0.6);
        factors.insert("volatility".to_string(), 0.3);
        factors.insert("momentum".to_string(), 0.6);
        factors.insert("sentiment".to_string(), sentiment);
        
        let factor_map = FactorMap::new(factors).unwrap();
        
        enhancer.enhance_decision(
            &base_decision,
            &factor_map,
            &analysis,
            symbol
        ).unwrap();
    }
    
    let metrics = enhancer.get_behavioral_metrics();
    
    // Should track reference points
    assert!(metrics.get("reference_points_tracked").unwrap_or(&0.0) >= &3.0);
    
    // Should track framing history
    assert!(metrics.get("framing_history_length").unwrap_or(&0.0) >= &5.0);
    
    // Should have frame distribution
    assert!(metrics.contains_key("gain_frame_ratio"));
    assert!(metrics.contains_key("loss_frame_ratio"));
    assert!(metrics.contains_key("neutral_frame_ratio"));
    
    // Frame ratios should sum to approximately 1.0
    let gain_ratio = metrics.get("gain_frame_ratio").unwrap_or(&0.0);
    let loss_ratio = metrics.get("loss_frame_ratio").unwrap_or(&0.0);
    let neutral_ratio = metrics.get("neutral_frame_ratio").unwrap_or(&0.0);
    
    assert!((gain_ratio + loss_ratio + neutral_ratio - 1.0).abs() < 0.1);
    
    // Should track average reference point
    assert!(metrics.contains_key("average_reference_point"));
    assert!(metrics.get("average_reference_point").unwrap() > &0.0);
}

#[tokio::test]
async fn test_behavioral_state_reset() {
    let behavioral_config = BehavioralConfig::default();
    let pt_config = QuantumProspectTheoryConfig::default();
    let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
    
    let mut enhancer = BehavioralDecisionEnhancer::new(behavioral_config, prospect_theory);
    
    let base_decision = TradingDecision {
        decision_type: DecisionType::Buy,
        confidence: 0.6,
        expected_return: Some(0.04),
        risk_assessment: Some(0.4),
        urgency_score: 0.5,
        reasoning: "Base decision".to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    let mut factors = HashMap::new();
    factors.insert("trend".to_string(), 0.6);
    factors.insert("volatility".to_string(), 0.3);
    factors.insert("momentum".to_string(), 0.6);
    factors.insert("sentiment".to_string(), 0.7);
    
    let factor_map = FactorMap::new(factors).unwrap();
    let analysis = AnalysisResult {
        trend_strength: 0.6,
        volatility_score: 0.3,
        momentum_indicator: 0.6,
        confidence_level: 0.7,
        supporting_factors: vec!["Moderate trend".to_string()],
        risk_factors: vec!["Low volatility".to_string()],
    };
    
    // Build up some state
    for i in 0..5 {
        enhancer.enhance_decision(
            &base_decision,
            &factor_map,
            &analysis,
            &format!("SYM{}", i)
        ).unwrap();
    }
    
    let metrics_before = enhancer.get_behavioral_metrics();
    assert!(metrics_before.get("reference_points_tracked").unwrap_or(&0.0) >= &5.0);
    assert!(metrics_before.get("framing_history_length").unwrap_or(&0.0) >= &5.0);
    
    // Reset state
    enhancer.reset();
    
    let metrics_after = enhancer.get_behavioral_metrics();
    assert_eq!(metrics_after.get("reference_points_tracked").unwrap_or(&0.0), &0.0);
    assert_eq!(metrics_after.get("framing_history_length").unwrap_or(&0.0), &0.0);
    
    // Should still be able to make decisions after reset
    let post_reset_enhancement = enhancer.enhance_decision(
        &base_decision,
        &factor_map,
        &analysis,
        "BTC/USDT"
    ).unwrap();
    
    assert!(post_reset_enhancement.enhanced_decision.confidence >= 0.0 && 
            post_reset_enhancement.enhanced_decision.confidence <= 1.0);
}