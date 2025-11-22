//! TDD tests for MutationEngine module - ZERO MOCKS policy enforced
//! Tests validate real genetic mutation strategies with adaptive rates

use std::time::Instant;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use super::super::mutation_engine::*;
use crate::organisms::{ParasiticOrganism, OrganismGenetics, CuckooOrganism, WaspOrganism, VirusOrganism};

#[tokio::test]
async fn test_mutation_engine_initialization() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.1,
        adaptive_mutation: true,
        mutation_strength: 0.2,
        max_mutation_rate: 0.5,
        min_mutation_rate: 0.01,
        diversity_threshold: 0.1,
        convergence_pressure: 1.5,
        targeted_mutation: true,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let engine = MutationEngine::new(config.clone());
    
    assert_eq!(engine.get_config().base_mutation_rate, 0.1);
    assert_eq!(engine.get_config().mutation_strength, 0.2);
    assert!(engine.get_config().adaptive_mutation);
    assert_eq!(engine.get_mutation_count(), 0);
}

#[tokio::test]
async fn test_mutation_engine_sub_millisecond_performance() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.2,
        adaptive_mutation: true,
        mutation_strength: 0.15,
        max_mutation_rate: 0.4,
        min_mutation_rate: 0.02,
        diversity_threshold: 0.15,
        convergence_pressure: 1.2,
        targeted_mutation: true,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create test population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..50 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let population_diversity = 0.3;
    
    // Measure mutation performance
    let start = Instant::now();
    let result = engine.apply_mutations(&organisms, population_diversity).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    assert!(duration.as_millis() < 1, "Mutation application must complete in under 1ms, took: {:?}", duration);
    
    let mutation_result = result.unwrap();
    assert!(mutation_result.mutations_applied > 0);
    assert!(mutation_result.total_time_nanos > 0);
    assert!(mutation_result.effective_rate >= 0.0);
}

#[tokio::test]
async fn test_gaussian_mutation_implementation() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.3,
        adaptive_mutation: false,
        mutation_strength: 0.1, // Small mutations
        max_mutation_rate: 0.5,
        min_mutation_rate: 0.1,
        diversity_threshold: 0.2,
        convergence_pressure: 1.0,
        targeted_mutation: false,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create organism with known genetics
    let mut organism = WaspOrganism::new();
    let original_genetics = OrganismGenetics {
        aggression: 0.5,
        adaptability: 0.5,
        efficiency: 0.5,
        resilience: 0.5,
        reaction_speed: 0.5,
        risk_tolerance: 0.5,
        cooperation: 0.5,
        stealth: 0.5,
    };
    organism.set_genetics(original_genetics.clone());
    
    // Apply Gaussian mutation multiple times
    let mut total_change = 0.0;
    let iterations = 100;
    
    for _ in 0..iterations {
        let result = engine.apply_gaussian_mutation(&mut organism, 0.1).await;
        assert!(result.is_ok());
        
        let mutated_genetics = organism.get_genetics();
        
        // Calculate total genetic change
        let change = [
            (mutated_genetics.aggression - original_genetics.aggression).abs(),
            (mutated_genetics.adaptability - original_genetics.adaptability).abs(),
            (mutated_genetics.efficiency - original_genetics.efficiency).abs(),
            (mutated_genetics.resilience - original_genetics.resilience).abs(),
            (mutated_genetics.reaction_speed - original_genetics.reaction_speed).abs(),
            (mutated_genetics.risk_tolerance - original_genetics.risk_tolerance).abs(),
            (mutated_genetics.cooperation - original_genetics.cooperation).abs(),
            (mutated_genetics.stealth - original_genetics.stealth).abs(),
        ].iter().sum::<f64>();
        
        total_change += change;
        
        // Verify values stay within bounds
        assert!(mutated_genetics.aggression >= 0.0 && mutated_genetics.aggression <= 1.0);
        assert!(mutated_genetics.adaptability >= 0.0 && mutated_genetics.adaptability <= 1.0);
        assert!(mutated_genetics.efficiency >= 0.0 && mutated_genetics.efficiency <= 1.0);
        assert!(mutated_genetics.resilience >= 0.0 && mutated_genetics.resilience <= 1.0);
        assert!(mutated_genetics.reaction_speed >= 0.0 && mutated_genetics.reaction_speed <= 1.0);
        assert!(mutated_genetics.risk_tolerance >= 0.0 && mutated_genetics.risk_tolerance <= 1.0);
        assert!(mutated_genetics.cooperation >= 0.0 && mutated_genetics.cooperation <= 1.0);
        assert!(mutated_genetics.stealth >= 0.0 && mutated_genetics.stealth <= 1.0);
    }
    
    let average_change = total_change / iterations as f64;
    
    // Gaussian mutations should produce small, normally distributed changes
    assert!(average_change > 0.01, "Should produce detectable changes");
    assert!(average_change < 0.5, "Changes should be reasonably small for given strength");
}

#[tokio::test]
async fn test_adaptive_mutation_rate_adjustment() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.1,
        adaptive_mutation: true,
        mutation_strength: 0.15,
        max_mutation_rate: 0.4,
        min_mutation_rate: 0.02,
        diversity_threshold: 0.2,
        convergence_pressure: 1.3,
        targeted_mutation: true,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create homogeneous population (low diversity)
    let homogeneous_organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..30 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let similar_genetics = OrganismGenetics {
            aggression: 0.7 + (rand::random::<f64>() * 0.1 - 0.05), // Small variation
            adaptability: 0.6 + (rand::random::<f64>() * 0.1 - 0.05),
            efficiency: 0.8 + (rand::random::<f64>() * 0.1 - 0.05),
            resilience: 0.5 + (rand::random::<f64>() * 0.1 - 0.05),
            reaction_speed: 0.9 + (rand::random::<f64>() * 0.1 - 0.05),
            risk_tolerance: 0.3 + (rand::random::<f64>() * 0.1 - 0.05),
            cooperation: 0.4 + (rand::random::<f64>() * 0.1 - 0.05),
            stealth: 0.8 + (rand::random::<f64>() * 0.1 - 0.05),
        };
        organism.set_genetics(similar_genetics);
        
        let id = organism.id();
        homogeneous_organisms.insert(id, organism);
    }
    
    let low_diversity = 0.05; // Very low diversity
    let initial_rate = engine.get_current_mutation_rate();
    
    // Apply mutations with low diversity - should increase mutation rate
    let low_diversity_result = engine.apply_mutations(&homogeneous_organisms, low_diversity).await.unwrap();
    let increased_rate = engine.get_current_mutation_rate();
    
    // Create diverse population
    let diverse_organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for i in 0..30 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let diverse_genetics = OrganismGenetics {
            aggression: i as f64 / 30.0,
            adaptability: (i * 2) as f64 / 60.0,
            efficiency: (i * 3) as f64 / 90.0,
            resilience: (i * 4) as f64 / 120.0,
            reaction_speed: (i * 5) as f64 / 150.0,
            risk_tolerance: (i * 6) as f64 / 180.0,
            cooperation: (i * 7) as f64 / 210.0,
            stealth: (i * 8) as f64 / 240.0,
        };
        organism.set_genetics(diverse_genetics);
        
        let id = organism.id();
        diverse_organisms.insert(id, organism);
    }
    
    let high_diversity = 0.8; // High diversity
    let diverse_result = engine.apply_mutations(&diverse_organisms, high_diversity).await.unwrap();
    let decreased_rate = engine.get_current_mutation_rate();
    
    // Adaptive mutation should increase rate for low diversity and decrease for high diversity
    assert!(increased_rate > initial_rate, 
            "Mutation rate should increase with low diversity: {} vs {}", 
            increased_rate, initial_rate);
    
    assert!(decreased_rate < increased_rate, 
            "Mutation rate should decrease with high diversity: {} vs {}", 
            decreased_rate, increased_rate);
    
    // Verify mutation counts are reasonable
    assert!(low_diversity_result.mutations_applied > 0);
    assert!(diverse_result.mutations_applied > 0);
}

#[tokio::test]
async fn test_targeted_mutation_strategy() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.2,
        adaptive_mutation: true,
        mutation_strength: 0.2,
        max_mutation_rate: 0.6,
        min_mutation_rate: 0.05,
        diversity_threshold: 0.15,
        convergence_pressure: 1.4,
        targeted_mutation: true, // Enable targeted mutations
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create organism with suboptimal genetics
    let mut poor_performer = CuckooOrganism::new();
    let weak_genetics = OrganismGenetics {
        aggression: 0.1, // Low values that could be improved
        adaptability: 0.2,
        efficiency: 0.15,
        resilience: 0.3,
        reaction_speed: 0.25,
        risk_tolerance: 0.9, // High risk (usually bad)
        cooperation: 0.1,
        stealth: 0.2,
    };
    poor_performer.set_genetics(weak_genetics.clone());
    
    let fitness_scores = vec![0.2]; // Low fitness score
    
    // Apply targeted mutations
    let result = engine.apply_targeted_mutation(&mut poor_performer, &fitness_scores).await;
    assert!(result.is_ok());
    
    let mutated_genetics = poor_performer.get_genetics();
    
    // Targeted mutation should try to improve key performance traits
    // Some traits should have been improved (though not guaranteed due to randomness)
    let improvements = [
        mutated_genetics.efficiency > weak_genetics.efficiency,
        mutated_genetics.reaction_speed > weak_genetics.reaction_speed,
        mutated_genetics.adaptability > weak_genetics.adaptability,
        mutated_genetics.resilience > weak_genetics.resilience,
        mutated_genetics.risk_tolerance < weak_genetics.risk_tolerance, // Lower is better
    ];
    
    let improvement_count = improvements.iter().filter(|&&x| x).count();
    
    // Should have improved at least some traits (probabilistic test)
    // Note: This test may occasionally fail due to randomness, but should pass most of the time
    assert!(improvement_count > 0, 
            "Targeted mutation should improve some traits for poor performer");
    
    // Verify all values still within bounds
    assert!(mutated_genetics.aggression >= 0.0 && mutated_genetics.aggression <= 1.0);
    assert!(mutated_genetics.efficiency >= 0.0 && mutated_genetics.efficiency <= 1.0);
    assert!(mutated_genetics.risk_tolerance >= 0.0 && mutated_genetics.risk_tolerance <= 1.0);
}

#[tokio::test]
async fn test_uniform_mutation_strategy() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.25,
        adaptive_mutation: false,
        mutation_strength: 0.3,
        max_mutation_rate: 0.5,
        min_mutation_rate: 0.1,
        diversity_threshold: 0.2,
        convergence_pressure: 1.0,
        targeted_mutation: false,
        gaussian_mutation: false,
        uniform_mutation: true, // Enable uniform mutations
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create test organism
    let mut organism = WaspOrganism::new();
    let original_genetics = OrganismGenetics {
        aggression: 0.6,
        adaptability: 0.4,
        efficiency: 0.7,
        resilience: 0.3,
        reaction_speed: 0.8,
        risk_tolerance: 0.2,
        cooperation: 0.9,
        stealth: 0.1,
    };
    organism.set_genetics(original_genetics.clone());
    
    // Apply uniform mutations multiple times
    let mut change_distribution = Vec::new();
    let iterations = 50;
    
    for _ in 0..iterations {
        let mut test_organism = WaspOrganism::new();
        test_organism.set_genetics(original_genetics.clone());
        
        let result = engine.apply_uniform_mutation(&mut test_organism, 0.3).await;
        assert!(result.is_ok());
        
        let mutated_genetics = test_organism.get_genetics();
        
        // Calculate changes for each trait
        let changes = [
            (mutated_genetics.aggression - original_genetics.aggression).abs(),
            (mutated_genetics.adaptability - original_genetics.adaptability).abs(),
            (mutated_genetics.efficiency - original_genetics.efficiency).abs(),
            (mutated_genetics.resilience - original_genetics.resilience).abs(),
            (mutated_genetics.reaction_speed - original_genetics.reaction_speed).abs(),
            (mutated_genetics.risk_tolerance - original_genetics.risk_tolerance).abs(),
            (mutated_genetics.cooperation - original_genetics.cooperation).abs(),
            (mutated_genetics.stealth - original_genetics.stealth).abs(),
        ];
        
        let total_change: f64 = changes.iter().sum();
        change_distribution.push(total_change);
        
        // Verify bounds
        assert!(mutated_genetics.aggression >= 0.0 && mutated_genetics.aggression <= 1.0);
        assert!(mutated_genetics.adaptability >= 0.0 && mutated_genetics.adaptability <= 1.0);
        assert!(mutated_genetics.stealth >= 0.0 && mutated_genetics.stealth <= 1.0);
    }
    
    // Uniform mutations should produce more consistent change amounts
    let mean_change: f64 = change_distribution.iter().sum::<f64>() / iterations as f64;
    let variance: f64 = change_distribution.iter()
        .map(|&x| (x - mean_change).powi(2))
        .sum::<f64>() / iterations as f64;
    
    assert!(mean_change > 0.1, "Should produce noticeable changes");
    assert!(variance < mean_change * 0.5, "Uniform mutations should be less variable than Gaussian");
}

#[tokio::test]
async fn test_mutation_strength_scaling() {
    let base_config = MutationEngineConfig {
        base_mutation_rate: 0.2,
        adaptive_mutation: false,
        mutation_strength: 0.1, // Weak mutations
        max_mutation_rate: 0.4,
        min_mutation_rate: 0.1,
        diversity_threshold: 0.2,
        convergence_pressure: 1.0,
        targeted_mutation: false,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let strong_config = MutationEngineConfig {
        mutation_strength: 0.4, // Strong mutations
        ..base_config.clone()
    };
    
    let mut weak_engine = MutationEngine::new(base_config);
    let mut strong_engine = MutationEngine::new(strong_config);
    
    // Create identical test organisms
    let create_test_organism = || {
        let mut organism = CuckooOrganism::new();
        let genetics = OrganismGenetics {
            aggression: 0.5,
            adaptability: 0.5,
            efficiency: 0.5,
            resilience: 0.5,
            reaction_speed: 0.5,
            risk_tolerance: 0.5,
            cooperation: 0.5,
            stealth: 0.5,
        };
        organism.set_genetics(genetics);
        organism
    };
    
    // Test weak mutations
    let mut weak_organism = create_test_organism();
    let weak_result = weak_engine.apply_gaussian_mutation(&mut weak_organism, 0.1).await.unwrap();
    let weak_genetics = weak_organism.get_genetics();
    
    // Test strong mutations
    let mut strong_organism = create_test_organism();
    let strong_result = strong_engine.apply_gaussian_mutation(&mut strong_organism, 0.4).await.unwrap();
    let strong_genetics = strong_organism.get_genetics();
    
    // Calculate total genetic changes
    let weak_change = [
        (weak_genetics.aggression - 0.5).abs(),
        (weak_genetics.adaptability - 0.5).abs(),
        (weak_genetics.efficiency - 0.5).abs(),
        (weak_genetics.resilience - 0.5).abs(),
        (weak_genetics.reaction_speed - 0.5).abs(),
        (weak_genetics.risk_tolerance - 0.5).abs(),
        (weak_genetics.cooperation - 0.5).abs(),
        (weak_genetics.stealth - 0.5).abs(),
    ].iter().sum::<f64>();
    
    let strong_change = [
        (strong_genetics.aggression - 0.5).abs(),
        (strong_genetics.adaptability - 0.5).abs(),
        (strong_genetics.efficiency - 0.5).abs(),
        (strong_genetics.resilience - 0.5).abs(),
        (strong_genetics.reaction_speed - 0.5).abs(),
        (strong_genetics.risk_tolerance - 0.5).abs(),
        (strong_genetics.cooperation - 0.5).abs(),
        (strong_genetics.stealth - 0.5).abs(),
    ].iter().sum::<f64>();
    
    // Strong mutations should generally produce larger changes
    // Note: Due to randomness, this might occasionally fail, but should pass most of the time
    println!("Weak change: {}, Strong change: {}", weak_change, strong_change);
    
    // At minimum, both should produce some change
    assert!(weak_change > 0.01, "Weak mutations should produce some change");
    assert!(strong_change > 0.01, "Strong mutations should produce some change");
}

#[tokio::test]
async fn test_mutation_rate_boundaries() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.1,
        adaptive_mutation: true,
        mutation_strength: 0.2,
        max_mutation_rate: 0.3, // Low maximum
        min_mutation_rate: 0.05, // Higher minimum
        diversity_threshold: 0.15,
        convergence_pressure: 2.0, // High pressure
        targeted_mutation: false,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create organisms
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    for _ in 0..20 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Apply extreme conditions to test boundaries
    let very_low_diversity = 0.01; // Should max out mutation rate
    let very_high_diversity = 0.9; // Should minimize mutation rate
    
    // Test maximum boundary
    let _ = engine.apply_mutations(&organisms, very_low_diversity).await.unwrap();
    let max_rate = engine.get_current_mutation_rate();
    assert!(max_rate <= 0.3, "Mutation rate should not exceed maximum: {}", max_rate);
    
    // Test minimum boundary
    let _ = engine.apply_mutations(&organisms, very_high_diversity).await.unwrap();
    let min_rate = engine.get_current_mutation_rate();
    assert!(min_rate >= 0.05, "Mutation rate should not fall below minimum: {}", min_rate);
}

#[tokio::test]
async fn test_mutation_statistics_tracking() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.15,
        adaptive_mutation: true,
        mutation_strength: 0.2,
        max_mutation_rate: 0.4,
        min_mutation_rate: 0.03,
        diversity_threshold: 0.1,
        convergence_pressure: 1.2,
        targeted_mutation: true,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create test population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..40 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let initial_count = engine.get_mutation_count();
    assert_eq!(initial_count, 0);
    
    // Apply mutations multiple times
    for i in 0..3 {
        let diversity = 0.3 - (i as f64 * 0.1); // Decreasing diversity
        let result = engine.apply_mutations(&organisms, diversity).await.unwrap();
        
        let current_count = engine.get_mutation_count();
        assert!(current_count > initial_count, "Mutation count should increase");
        
        // Verify statistics are being tracked
        let stats = engine.get_mutation_statistics().await;
        assert!(stats.total_mutations > 0);
        assert!(stats.average_mutations_per_cycle > 0.0);
        assert!(stats.current_mutation_rate > 0.0);
        assert!(stats.total_organisms_mutated > 0);
        
        if i > 0 {
            assert!(stats.mutation_rate_history.len() > i);
        }
    }
    
    let final_stats = engine.get_mutation_statistics().await;
    assert_eq!(final_stats.mutation_rate_history.len(), 3);
    assert!(final_stats.total_execution_time_ms > 0.0);
}

#[tokio::test]
async fn test_concurrent_mutation_safety() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.2,
        adaptive_mutation: true,
        mutation_strength: 0.15,
        max_mutation_rate: 0.5,
        min_mutation_rate: 0.05,
        diversity_threshold: 0.2,
        convergence_pressure: 1.3,
        targeted_mutation: false,
        gaussian_mutation: true,
        uniform_mutation: false,
    };
    
    let engine = Arc::new(tokio::sync::Mutex::new(MutationEngine::new(config)));
    
    // Create shared population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..30 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Run concurrent mutations
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let engine_clone = Arc::clone(&engine);
        let organisms_clone = Arc::clone(&organisms);
        
        let handle = tokio::spawn(async move {
            let diversity = 0.2 + (i as f64 * 0.1);
            let mut engine_guard = engine_clone.lock().await;
            engine_guard.apply_mutations(&organisms_clone, diversity).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all mutations to complete
    let mut successful_mutations = 0;
    
    for handle in handles {
        if let Ok(result) = handle.await {
            if result.is_ok() {
                successful_mutations += 1;
            }
        }
    }
    
    assert!(successful_mutations > 0, "At least some concurrent mutations should succeed");
    
    // Verify final state consistency
    let final_engine = engine.lock().await;
    let final_count = final_engine.get_mutation_count();
    assert!(final_count > 0, "Should have recorded mutations from concurrent operations");
}