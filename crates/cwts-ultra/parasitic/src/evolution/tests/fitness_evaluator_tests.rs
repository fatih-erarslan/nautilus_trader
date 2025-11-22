//! TDD tests for FitnessEvaluator module - ZERO MOCKS policy enforced
//! Tests validate real market performance scoring with sub-millisecond execution

use std::time::Instant;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use super::super::fitness_evaluator::*;
use crate::organisms::{ParasiticOrganism, OrganismGenetics, CuckooOrganism, WaspOrganism, VirusOrganism};

#[tokio::test]
async fn test_fitness_evaluator_initialization() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.4,
        efficiency_weight: 0.2,
        adaptability_weight: 0.2,
        risk_management_weight: 0.2,
        time_decay_factor: 0.95,
        performance_history_length: 100,
        real_time_evaluation: true,
    };
    
    let evaluator = FitnessEvaluator::new(config.clone());
    
    assert_eq!(evaluator.get_config().market_performance_weight, 0.4);
    assert_eq!(evaluator.get_config().efficiency_weight, 0.2);
    assert!(evaluator.get_config().real_time_evaluation);
    assert_eq!(evaluator.get_evaluation_count(), 0);
}

#[tokio::test]
async fn test_fitness_evaluation_sub_millisecond_performance() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.5,
        efficiency_weight: 0.2,
        adaptability_weight: 0.2,
        risk_management_weight: 0.1,
        time_decay_factor: 0.9,
        performance_history_length: 50,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Create test organism with known performance metrics
    let mut organism = CuckooOrganism::new();
    let genetics = OrganismGenetics {
        aggression: 0.7,
        adaptability: 0.8,
        efficiency: 0.9,
        resilience: 0.6,
        reaction_speed: 0.95,
        risk_tolerance: 0.4,
        cooperation: 0.3,
        stealth: 0.8,
    };
    organism.set_genetics(genetics);
    
    // Create market conditions for evaluation
    let market_conditions = MarketConditions {
        volatility: 0.3,
        trend_strength: 0.6,
        liquidity: 0.8,
        correlation_breakdown: false,
        flash_crash_risk: 0.1,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Measure fitness evaluation performance
    let start = Instant::now();
    let fitness_result = evaluator.evaluate_fitness(&organism, &market_conditions).await;
    let duration = start.elapsed();
    
    assert!(fitness_result.is_ok());
    assert!(duration.as_millis() < 1, "Fitness evaluation must complete in under 1ms, took: {:?}", duration);
    
    let fitness_score = fitness_result.unwrap();
    assert!(fitness_score.overall_fitness >= 0.0);
    assert!(fitness_score.overall_fitness <= 1.0);
    assert!(fitness_score.market_performance_score >= 0.0);
    assert!(fitness_score.efficiency_score >= 0.0);
    assert!(fitness_score.adaptability_score >= 0.0);
    assert!(fitness_score.risk_score >= 0.0);
}

#[tokio::test]
async fn test_market_performance_scoring() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 1.0, // Focus entirely on market performance
        efficiency_weight: 0.0,
        adaptability_weight: 0.0,
        risk_management_weight: 0.0,
        time_decay_factor: 1.0, // No decay for this test
        performance_history_length: 10,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Create high-performance organism
    let mut high_perf_organism = WaspOrganism::new();
    let high_perf_genetics = OrganismGenetics {
        aggression: 0.9,
        adaptability: 0.9,
        efficiency: 0.95,
        resilience: 0.8,
        reaction_speed: 0.98,
        risk_tolerance: 0.3, // Controlled risk
        cooperation: 0.4,
        stealth: 0.9,
    };
    high_perf_organism.set_genetics(high_perf_genetics);
    
    // Create low-performance organism
    let mut low_perf_organism = VirusOrganism::new();
    let low_perf_genetics = OrganismGenetics {
        aggression: 0.2,
        adaptability: 0.3,
        efficiency: 0.25,
        resilience: 0.4,
        reaction_speed: 0.3,
        risk_tolerance: 0.9, // High risk
        cooperation: 0.1,
        stealth: 0.2,
    };
    low_perf_organism.set_genetics(low_perf_genetics);
    
    let favorable_market = MarketConditions {
        volatility: 0.2, // Low volatility favors efficiency
        trend_strength: 0.8, // Strong trend
        liquidity: 0.9, // High liquidity
        correlation_breakdown: false,
        flash_crash_risk: 0.05,
        timestamp: std::time::SystemTime::now(),
    };
    
    let high_fitness = evaluator.evaluate_fitness(&high_perf_organism, &favorable_market).await.unwrap();
    let low_fitness = evaluator.evaluate_fitness(&low_perf_organism, &favorable_market).await.unwrap();
    
    assert!(high_fitness.overall_fitness > low_fitness.overall_fitness,
            "High-performance organism should score better: {} vs {}", 
            high_fitness.overall_fitness, low_fitness.overall_fitness);
    assert!(high_fitness.market_performance_score > 0.7, "High-performance organism should score well in market");
    assert!(low_fitness.market_performance_score < 0.5, "Low-performance organism should score poorly in market");
}

#[tokio::test]
async fn test_adaptive_market_conditions_response() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.4,
        efficiency_weight: 0.2,
        adaptability_weight: 0.3, // High weight on adaptability
        risk_management_weight: 0.1,
        time_decay_factor: 0.95,
        performance_history_length: 20,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Create adaptable organism
    let mut adaptable_organism = CuckooOrganism::new();
    let adaptable_genetics = OrganismGenetics {
        aggression: 0.6,
        adaptability: 0.95, // Very high adaptability
        efficiency: 0.7,
        resilience: 0.9,
        reaction_speed: 0.85,
        risk_tolerance: 0.5,
        cooperation: 0.6,
        stealth: 0.7,
    };
    adaptable_organism.set_genetics(adaptable_genetics);
    
    // Test different market conditions
    let volatile_market = MarketConditions {
        volatility: 0.8, // High volatility
        trend_strength: 0.3, // Weak trend
        liquidity: 0.4, // Low liquidity
        correlation_breakdown: true,
        flash_crash_risk: 0.3,
        timestamp: std::time::SystemTime::now(),
    };
    
    let stable_market = MarketConditions {
        volatility: 0.1, // Low volatility
        trend_strength: 0.9, // Strong trend
        liquidity: 0.95, // High liquidity
        correlation_breakdown: false,
        flash_crash_risk: 0.02,
        timestamp: std::time::SystemTime::now(),
    };
    
    let volatile_fitness = evaluator.evaluate_fitness(&adaptable_organism, &volatile_market).await.unwrap();
    let stable_fitness = evaluator.evaluate_fitness(&adaptable_organism, &stable_market).await.unwrap();
    
    // Adaptable organism should perform well in both conditions
    assert!(volatile_fitness.adaptability_score > 0.8, 
            "Adaptable organism should handle volatile markets well");
    assert!(stable_fitness.adaptability_score > 0.8, 
            "Adaptable organism should handle stable markets well");
    
    // Overall fitness should be reasonable in both cases
    assert!(volatile_fitness.overall_fitness > 0.5, "Should handle volatile markets reasonably");
    assert!(stable_fitness.overall_fitness > 0.6, "Should perform well in stable markets");
}

#[tokio::test]
async fn test_risk_management_scoring() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.2,
        efficiency_weight: 0.2,
        adaptability_weight: 0.1,
        risk_management_weight: 0.5, // Focus on risk management
        time_decay_factor: 1.0,
        performance_history_length: 15,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Conservative organism with good risk management
    let mut conservative_organism = WaspOrganism::new();
    let conservative_genetics = OrganismGenetics {
        aggression: 0.3,
        adaptability: 0.7,
        efficiency: 0.6,
        resilience: 0.95, // High resilience
        reaction_speed: 0.8,
        risk_tolerance: 0.2, // Low risk tolerance
        cooperation: 0.8,
        stealth: 0.7,
    };
    conservative_organism.set_genetics(conservative_genetics);
    
    // Aggressive organism with poor risk management
    let mut aggressive_organism = VirusOrganism::new();
    let aggressive_genetics = OrganismGenetics {
        aggression: 0.95,
        adaptability: 0.4,
        efficiency: 0.8,
        resilience: 0.3, // Low resilience
        reaction_speed: 0.9,
        risk_tolerance: 0.9, // High risk tolerance
        cooperation: 0.2,
        stealth: 0.6,
    };
    aggressive_organism.set_genetics(aggressive_genetics);
    
    // High-risk market conditions
    let risky_market = MarketConditions {
        volatility: 0.7,
        trend_strength: 0.2,
        liquidity: 0.3,
        correlation_breakdown: true,
        flash_crash_risk: 0.4, // High crash risk
        timestamp: std::time::SystemTime::now(),
    };
    
    let conservative_fitness = evaluator.evaluate_fitness(&conservative_organism, &risky_market).await.unwrap();
    let aggressive_fitness = evaluator.evaluate_fitness(&aggressive_organism, &risky_market).await.unwrap();
    
    // Conservative organism should have better risk score in risky conditions
    assert!(conservative_fitness.risk_score > aggressive_fitness.risk_score,
            "Conservative organism should have better risk management: {} vs {}", 
            conservative_fitness.risk_score, aggressive_fitness.risk_score);
    
    // Overall fitness should favor conservative approach in risky market
    assert!(conservative_fitness.overall_fitness > aggressive_fitness.overall_fitness,
            "Conservative approach should be favored in risky markets");
}

#[tokio::test]
async fn test_efficiency_scoring_optimization() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.1,
        efficiency_weight: 0.7, // Focus on efficiency
        adaptability_weight: 0.1,
        risk_management_weight: 0.1,
        time_decay_factor: 0.98,
        performance_history_length: 25,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Highly efficient organism
    let mut efficient_organism = CuckooOrganism::new();
    let efficient_genetics = OrganismGenetics {
        aggression: 0.5,
        adaptability: 0.6,
        efficiency: 0.98, // Very high efficiency
        resilience: 0.7,
        reaction_speed: 0.95, // Fast reactions
        risk_tolerance: 0.4,
        cooperation: 0.5,
        stealth: 0.8,
    };
    efficient_organism.set_genetics(efficient_genetics);
    
    // Inefficient organism
    let mut inefficient_organism = VirusOrganism::new();
    let inefficient_genetics = OrganismGenetics {
        aggression: 0.6,
        adaptability: 0.7,
        efficiency: 0.2, // Low efficiency
        resilience: 0.6,
        reaction_speed: 0.3, // Slow reactions
        risk_tolerance: 0.5,
        cooperation: 0.4,
        stealth: 0.5,
    };
    inefficient_organism.set_genetics(inefficient_genetics);
    
    let neutral_market = MarketConditions {
        volatility: 0.4,
        trend_strength: 0.5,
        liquidity: 0.6,
        correlation_breakdown: false,
        flash_crash_risk: 0.1,
        timestamp: std::time::SystemTime::now(),
    };
    
    let efficient_fitness = evaluator.evaluate_fitness(&efficient_organism, &neutral_market).await.unwrap();
    let inefficient_fitness = evaluator.evaluate_fitness(&inefficient_organism, &neutral_market).await.unwrap();
    
    assert!(efficient_fitness.efficiency_score > inefficient_fitness.efficiency_score,
            "Efficient organism should score better on efficiency: {} vs {}", 
            efficient_fitness.efficiency_score, inefficient_fitness.efficiency_score);
    
    assert!(efficient_fitness.efficiency_score > 0.9, 
            "Highly efficient organism should score very well on efficiency");
    assert!(inefficient_fitness.efficiency_score < 0.4, 
            "Inefficient organism should score poorly on efficiency");
}

#[tokio::test]
async fn test_performance_history_tracking() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.25,
        efficiency_weight: 0.25,
        adaptability_weight: 0.25,
        risk_management_weight: 0.25,
        time_decay_factor: 0.9,
        performance_history_length: 5, // Short history for testing
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    let organism = CuckooOrganism::new();
    
    let market_conditions = MarketConditions {
        volatility: 0.3,
        trend_strength: 0.7,
        liquidity: 0.8,
        correlation_breakdown: false,
        flash_crash_risk: 0.05,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Perform multiple evaluations to build history
    let mut fitness_scores = Vec::new();
    for _ in 0..10 {
        let fitness = evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
        fitness_scores.push(fitness.overall_fitness);
    }
    
    assert_eq!(evaluator.get_evaluation_count(), 10);
    
    // Get performance history (should be limited to configured length)
    let history = evaluator.get_performance_history(&organism.id()).await;
    assert!(history.len() <= 5, "History should be limited to configured length");
    
    // Verify that recent evaluations are tracked
    if !history.is_empty() {
        let recent_average = history.iter().sum::<f64>() / history.len() as f64;
        let last_score = fitness_scores.last().unwrap();
        
        // Recent average should be influenced by recent scores
        assert!((recent_average - last_score).abs() < 0.3, 
                "Recent history should reflect recent performance");
    }
}

#[tokio::test]
async fn test_time_decay_factor_application() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.25,
        efficiency_weight: 0.25,
        adaptability_weight: 0.25,
        risk_management_weight: 0.25,
        time_decay_factor: 0.5, // Strong decay for testing
        performance_history_length: 10,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    let organism = WaspOrganism::new();
    
    let market_conditions = MarketConditions {
        volatility: 0.4,
        trend_strength: 0.6,
        liquidity: 0.7,
        correlation_breakdown: false,
        flash_crash_risk: 0.08,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Perform evaluations with time gaps to test decay
    let first_fitness = evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    
    // Simulate time passage and multiple evaluations
    for _ in 0..5 {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let _ = evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    }
    
    // Get historical performance
    let historical_avg = evaluator.get_historical_fitness_average(&organism.id()).await;
    let recent_fitness = evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    
    // With strong decay factor, older evaluations should have less impact
    // Recent performance should dominate the historical average
    assert!(historical_avg.is_some(), "Should have historical data");
    
    let hist_avg = historical_avg.unwrap();
    assert!((hist_avg - recent_fitness.overall_fitness).abs() < 0.2, 
            "Historical average should be close to recent performance due to decay");
}

#[tokio::test]
async fn test_batch_fitness_evaluation_performance() {
    let config = FitnessEvaluationConfig {
        market_performance_weight: 0.25,
        efficiency_weight: 0.25,
        adaptability_weight: 0.25,
        risk_management_weight: 0.25,
        time_decay_factor: 0.95,
        performance_history_length: 50,
        real_time_evaluation: true,
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Create population of organisms
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..100 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions {
        volatility: 0.35,
        trend_strength: 0.65,
        liquidity: 0.75,
        correlation_breakdown: false,
        flash_crash_risk: 0.12,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Measure batch evaluation performance
    let start = Instant::now();
    let batch_results = evaluator.evaluate_population_fitness(&organisms, &market_conditions).await;
    let duration = start.elapsed();
    
    assert!(batch_results.is_ok());
    assert!(duration.as_millis() < 5, "Batch evaluation of 100 organisms should complete in under 5ms, took: {:?}", duration);
    
    let results = batch_results.unwrap();
    assert_eq!(results.len(), 100);
    
    // Verify all organisms were evaluated
    for (organism_id, fitness_score) in &results {
        assert!(organisms.contains_key(organism_id));
        assert!(fitness_score.overall_fitness >= 0.0);
        assert!(fitness_score.overall_fitness <= 1.0);
    }
}

#[tokio::test]
async fn test_real_time_vs_cached_evaluation() {
    let real_time_config = FitnessEvaluationConfig {
        market_performance_weight: 0.25,
        efficiency_weight: 0.25,
        adaptability_weight: 0.25,
        risk_management_weight: 0.25,
        time_decay_factor: 0.95,
        performance_history_length: 30,
        real_time_evaluation: true, // Real-time mode
    };
    
    let cached_config = FitnessEvaluationConfig {
        real_time_evaluation: false, // Cached mode
        ..real_time_config.clone()
    };
    
    let mut real_time_evaluator = FitnessEvaluator::new(real_time_config);
    let mut cached_evaluator = FitnessEvaluator::new(cached_config);
    
    let organism = WaspOrganism::new();
    let market_conditions = MarketConditions {
        volatility: 0.4,
        trend_strength: 0.6,
        liquidity: 0.8,
        correlation_breakdown: false,
        flash_crash_risk: 0.1,
        timestamp: std::time::SystemTime::now(),
    };
    
    // First evaluation - both should compute
    let rt_start = Instant::now();
    let real_time_fitness1 = real_time_evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    let rt_duration1 = rt_start.elapsed();
    
    let cached_start = Instant::now();
    let cached_fitness1 = cached_evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    let cached_duration1 = cached_start.elapsed();
    
    // Second evaluation - cached should be faster
    let rt_start2 = Instant::now();
    let real_time_fitness2 = real_time_evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    let rt_duration2 = rt_start2.elapsed();
    
    let cached_start2 = Instant::now();
    let cached_fitness2 = cached_evaluator.evaluate_fitness(&organism, &market_conditions).await.unwrap();
    let cached_duration2 = cached_start2.elapsed();
    
    // Real-time should always compute (potentially different results)
    // Cached should return same result faster on second call
    assert_eq!(cached_fitness1.overall_fitness, cached_fitness2.overall_fitness, 
               "Cached evaluation should return consistent results");
    
    assert!(cached_duration2 <= cached_duration1, 
            "Second cached evaluation should be faster or equal: {:?} vs {:?}", 
            cached_duration2, cached_duration1);
    
    // Both modes should complete quickly
    assert!(rt_duration1.as_millis() < 1, "Real-time evaluation too slow");
    assert!(cached_duration2.as_millis() < 1, "Cached evaluation too slow");
}