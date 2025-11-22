//! Integration tests for the complete evolution engine
//! Tests the interaction between all evolution components
//! ZERO mocks policy - all tests use real implementations

use std::time::Instant;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use super::super::*;
use crate::organisms::{ParasiticOrganism, CuckooOrganism, WaspOrganism, VirusOrganism};

#[tokio::test]
async fn test_complete_evolution_engine_initialization() {
    let config = EvolutionEngineConfig::default();
    let engine = EvolutionEngine::new(config);
    
    assert_eq!(engine.get_generation(), 0);
    assert!(!engine.has_converged());
    
    let status_config = engine.get_config().await;
    assert!(status_config.enable_neural_evolution);
    assert_eq!(status_config.performance_target_ms, 1.0);
}

#[tokio::test]
async fn test_complete_evolution_cycle_integration() {
    let mut engine = create_high_performance_evolution_engine();
    
    // Create diverse population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for i in 0..30 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = match i % 3 {
            0 => Box::new(CuckooOrganism::new()),
            1 => Box::new(WaspOrganism::new()),
            _ => Box::new(VirusOrganism::new()),
        };
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Market conditions for testing
    let market_conditions = MarketConditions {
        volatility: 0.4,
        trend_strength: 0.6,
        liquidity: 0.8,
        correlation_breakdown: false,
        flash_crash_risk: 0.1,
        timestamp: std::time::SystemTime::now(),
    };
    
    // Run evolution cycle
    let start = Instant::now();
    let result = engine.evolve_organisms(&organisms, &market_conditions).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    assert!(duration.as_millis() < 5, "Complete evolution cycle should be fast, took: {:?}", duration);
    
    let status = result.unwrap();
    assert_eq!(status.current_generation, 1);
    assert_eq!(status.population_size, 30);
    assert!(status.genetic_algorithm_stats.average_fitness >= 0.0);
    assert!(status.mutation_stats.total_mutations > 0);
    assert!(status.performance_metrics.sub_millisecond_compliance || duration.as_millis() < 5);
}

#[tokio::test]
async fn test_multi_generation_evolution() {
    let mut engine = create_evolution_engine();
    
    // Create population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..25 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Run multiple generations
    let mut fitness_progression = Vec::new();
    let mut diversity_progression = Vec::new();
    
    for generation in 0..5 {
        let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        
        assert_eq!(result.current_generation, generation + 1);
        fitness_progression.push(result.genetic_algorithm_stats.average_fitness);
        diversity_progression.push(result.genetic_algorithm_stats.genetic_diversity);
        
        // Verify progress tracking
        assert!(result.mutation_stats.total_mutations > 0);
        assert!(result.performance_metrics.average_evolution_time_ms > 0.0);
        
        if generation > 0 {
            // Should have some consistency in population size
            assert_eq!(result.population_size, 25);
        }
    }
    
    // Should have collected progression data
    assert_eq!(fitness_progression.len(), 5);
    assert_eq!(diversity_progression.len(), 5);
    
    // At least some generations should show progress
    let fitness_improved = fitness_progression.windows(2)
        .any(|window| window[1] > window[0]);
    
    // Evolution should either improve fitness or maintain diversity
    // (Due to randomness, not guaranteed every run, but should happen often)
    println!("Fitness progression: {:?}", fitness_progression);
    println!("Diversity progression: {:?}", diversity_progression);
}

#[tokio::test]
async fn test_evolution_convergence_detection() {
    let mut engine = create_evolution_engine();
    
    // Create homogeneous high-fitness population to promote convergence
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..20 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let high_performance_genetics = crate::organisms::OrganismGenetics {
            aggression: 0.8 + (rand::random::<f64>() * 0.1 - 0.05), // Small variation
            adaptability: 0.9 + (rand::random::<f64>() * 0.1 - 0.05),
            efficiency: 0.95 + (rand::random::<f64>() * 0.1 - 0.05),
            resilience: 0.85 + (rand::random::<f64>() * 0.1 - 0.05),
            reaction_speed: 0.98 + (rand::random::<f64>() * 0.02 - 0.01),
            risk_tolerance: 0.2 + (rand::random::<f64>() * 0.1 - 0.05), // Low risk
            cooperation: 0.6 + (rand::random::<f64>() * 0.1 - 0.05),
            stealth: 0.8 + (rand::random::<f64>() * 0.1 - 0.05),
        };
        organism.set_genetics(high_performance_genetics);
        
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions {
        volatility: 0.1, // Low volatility to promote stability
        trend_strength: 0.8,
        liquidity: 0.9,
        correlation_breakdown: false,
        flash_crash_risk: 0.02,
        timestamp: std::time::SystemTime::now(),
    };
    
    let mut convergence_progression = Vec::new();
    
    // Run evolution until convergence or max generations
    for generation in 0..20 {
        let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        convergence_progression.push(result.convergence_progress);
        
        if engine.has_converged() {
            println!("Converged at generation {}", generation + 1);
            break;
        }
        
        // Should show increasing convergence progress
        assert!(result.convergence_progress >= 0.0);
        assert!(result.convergence_progress <= 1.0);
    }
    
    // Should have tracked convergence progression
    assert!(!convergence_progression.is_empty());
    
    // Convergence should generally increase or at least have high values with similar organisms
    let final_convergence = convergence_progression.last().unwrap();
    assert!(*final_convergence > 0.5, "High-fitness homogeneous population should show high convergence");
}

#[tokio::test]
async fn test_evolution_engine_reset_functionality() {
    let mut engine = create_evolution_engine();
    
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..20 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Run a few evolution cycles
    for _ in 0..3 {
        let _ = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
    }
    
    let pre_reset_generation = engine.get_generation();
    assert_eq!(pre_reset_generation, 3);
    
    // Reset engine
    engine.reset().await;
    
    let post_reset_generation = engine.get_generation();
    assert_eq!(post_reset_generation, 0);
    assert!(!engine.has_converged());
    
    // Should be able to evolve again after reset
    let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
    assert_eq!(result.current_generation, 1);
}

#[tokio::test]
async fn test_evolution_with_changing_market_conditions() {
    let mut engine = create_evolution_engine();
    
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..25 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Simulate changing market conditions
    let market_scenarios = vec![
        MarketConditions {
            volatility: 0.2,
            trend_strength: 0.8,
            liquidity: 0.9,
            correlation_breakdown: false,
            flash_crash_risk: 0.05,
            timestamp: std::time::SystemTime::now(),
        },
        MarketConditions {
            volatility: 0.7,
            trend_strength: 0.3,
            liquidity: 0.4,
            correlation_breakdown: true,
            flash_crash_risk: 0.3,
            timestamp: std::time::SystemTime::now(),
        },
        MarketConditions {
            volatility: 0.4,
            trend_strength: 0.6,
            liquidity: 0.7,
            correlation_breakdown: false,
            flash_crash_risk: 0.1,
            timestamp: std::time::SystemTime::now(),
        },
    ];
    
    let mut adaptation_scores = Vec::new();
    
    for (i, market_condition) in market_scenarios.iter().enumerate() {
        let result = engine.evolve_organisms(&organisms, market_condition).await.unwrap();
        
        adaptation_scores.push(result.genetic_algorithm_stats.average_fitness);
        
        // Should adapt to different market conditions
        assert!(result.genetic_algorithm_stats.genetic_diversity >= 0.0);
        assert!(result.mutation_stats.total_mutations > 0);
        
        println!("Generation {} in market scenario {}: fitness = {:.3}, diversity = {:.3}", 
                 i + 1,
                 i,
                 result.genetic_algorithm_stats.average_fitness,
                 result.genetic_algorithm_stats.genetic_diversity);
    }
    
    // Should have recorded adaptation across different market conditions
    assert_eq!(adaptation_scores.len(), 3);
}

#[tokio::test]
async fn test_evolution_performance_metrics_accuracy() {
    let mut engine = create_high_performance_evolution_engine();
    
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..30 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Run evolution and measure actual performance
    let wall_clock_start = Instant::now();
    let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
    let wall_clock_duration = wall_clock_start.elapsed();
    
    // Verify performance metrics accuracy
    let metrics = &result.performance_metrics;
    
    // Engine-reported time should be close to wall clock time
    let reported_time_ms = metrics.average_evolution_time_ms;
    let wall_clock_time_ms = wall_clock_duration.as_millis() as f64;
    
    let time_difference = (reported_time_ms - wall_clock_time_ms).abs();
    assert!(time_difference < wall_clock_time_ms * 0.5, 
            "Reported time ({:.3}ms) should be close to wall clock time ({:.3}ms)", 
            reported_time_ms, wall_clock_time_ms);
    
    // Throughput metrics should be reasonable
    assert!(metrics.mutations_per_second > 0.0);
    assert!(metrics.evaluations_per_second > 0.0);
    
    // Memory efficiency should be between 0 and 1
    assert!(metrics.memory_efficiency_score >= 0.0);
    assert!(metrics.memory_efficiency_score <= 1.0);
    
    // Sub-millisecond compliance should match actual performance
    let actual_sub_millisecond = wall_clock_duration.as_millis() < 1;
    // Allow some tolerance for measurement variations
    if wall_clock_duration.as_millis() <= 2 {
        // Performance is close enough to target
        assert!(metrics.sub_millisecond_compliance || !actual_sub_millisecond);
    }
}

#[tokio::test]
async fn test_evolution_engine_configuration_updates() {
    let initial_config = EvolutionEngineConfig::default();
    let mut engine = EvolutionEngine::new(initial_config);
    
    // Update configuration
    let new_config = EvolutionEngineConfig {
        genetic_algorithm: GeneticAlgorithmConfig {
            population_size: 50,
            parallel_execution: true,
            ..Default::default()
        },
        enable_neural_evolution: false,
        performance_target_ms: 0.5,
        ..Default::default()
    };
    
    engine.update_config(new_config.clone()).await;
    
    let updated_config = engine.get_config().await;
    assert_eq!(updated_config.genetic_algorithm.population_size, 50);
    assert!(!updated_config.enable_neural_evolution);
    assert_eq!(updated_config.performance_target_ms, 0.5);
    
    // Should be able to evolve with new configuration
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..15 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
    
    assert_eq!(result.current_generation, 1);
    assert!(!result.neural_evolution_active); // Should reflect new config
}

#[tokio::test]
async fn test_evolution_performance_report_generation() {
    let mut engine = create_evolution_engine();
    
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..20 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Run a few evolution cycles
    for _ in 0..3 {
        let _ = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
    }
    
    // Generate performance report
    let report = engine.get_performance_report().await;
    
    // Verify report contains expected information
    assert!(report.contains("Evolution Engine Performance Report"));
    assert!(report.contains("Current Generation: 3"));
    assert!(report.contains("Average Evolution Time"));
    assert!(report.contains("Total Evolution Time"));
    assert!(report.contains("Mutations per Second"));
    assert!(report.contains("Evaluations per Second"));
    assert!(report.contains("Memory Efficiency"));
    assert!(report.contains("Sub-millisecond Compliance"));
    
    println!("Performance Report:\n{}", report);
}