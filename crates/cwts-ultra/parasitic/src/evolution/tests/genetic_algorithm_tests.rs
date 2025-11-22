//! TDD tests for GeneticAlgorithm module - ZERO MOCKS policy enforced
//! These tests validate real genetic algorithm implementations with sub-millisecond performance

use std::time::Instant;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use super::super::genetic_algorithm::*;
use crate::organisms::{ParasiticOrganism, OrganismGenetics, CuckooOrganism, WaspOrganism, VirusOrganism};

#[tokio::test]
async fn test_genetic_algorithm_initialization() {
    let config = GeneticAlgorithmConfig {
        population_size: 100,
        elite_percentage: 0.1,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.2,
        max_generations: 100,
        convergence_threshold: 0.001,
        parallel_execution: true,
        adaptive_parameters: true,
    };
    
    let algorithm = GeneticAlgorithm::new(config.clone());
    
    assert_eq!(algorithm.get_config().population_size, 100);
    assert_eq!(algorithm.get_config().elite_percentage, 0.1);
    assert_eq!(algorithm.get_generation(), 0);
    assert!(!algorithm.has_converged());
}

#[tokio::test]
async fn test_genetic_algorithm_sub_millisecond_performance() {
    let config = GeneticAlgorithmConfig {
        population_size: 50, // Smaller for performance test
        elite_percentage: 0.2,
        mutation_rate: 0.15,
        crossover_rate: 0.7,
        selection_pressure: 1.1,
        max_generations: 10,
        convergence_threshold: 0.01,
        parallel_execution: true,
        adaptive_parameters: false, // Disable for consistent timing
    };
    
    let mut algorithm = GeneticAlgorithm::new(config);
    
    // Create test population with real organisms
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..50 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = match rand::random::<u8>() % 3 {
            0 => Box::new(CuckooOrganism::new()),
            1 => Box::new(WaspOrganism::new()),
            _ => Box::new(VirusOrganism::new()),
        };
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Measure single evolution cycle performance
    let start = Instant::now();
    let result = algorithm.evolve_population(&organisms).await;
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    assert!(duration.as_millis() < 1, "Evolution cycle must complete in under 1ms, took: {:?}", duration);
    
    let evolution_result = result.unwrap();
    assert!(evolution_result.generation_completed);
    assert!(evolution_result.operations_count > 0);
}

#[tokio::test]
async fn test_tournament_selection_real_implementation() {
    let config = GeneticAlgorithmConfig {
        population_size: 20,
        elite_percentage: 0.3,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.5,
        max_generations: 1,
        convergence_threshold: 0.001,
        parallel_execution: false,
        adaptive_parameters: false,
    };
    
    let algorithm = GeneticAlgorithm::new(config);
    
    // Create organisms with known fitness values
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    let mut organism_ids = Vec::new();
    
    for i in 0..20 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let mut genetics = organism.get_genetics();
        genetics.efficiency = i as f64 / 20.0; // Different fitness levels
        organism.set_genetics(genetics);
        
        let id = organism.id();
        organism_ids.push(id);
        organisms.insert(id, organism);
    }
    
    let fitness_scores: Vec<(Uuid, f64)> = organisms
        .iter()
        .map(|entry| (*entry.key(), entry.value().fitness()))
        .collect();
    
    let selected = algorithm.tournament_selection(&fitness_scores, 5).await.unwrap();
    
    // Tournament selection should prefer higher fitness organisms
    assert!(!selected.is_empty());
    assert!(selected.len() <= 6); // Should select elite percentage
    
    // Verify selected organisms have higher average fitness
    let selected_fitness: f64 = selected.iter()
        .filter_map(|id| organisms.get(id))
        .map(|entry| entry.value().fitness())
        .sum::<f64>() / selected.len() as f64;
    
    let total_average: f64 = fitness_scores.iter().map(|(_, f)| f).sum::<f64>() / fitness_scores.len() as f64;
    
    assert!(selected_fitness >= total_average, "Selected organisms should have higher fitness");
}

#[tokio::test]
async fn test_roulette_wheel_selection_real_implementation() {
    let config = GeneticAlgorithmConfig {
        population_size: 30,
        elite_percentage: 0.2,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.0,
        max_generations: 1,
        convergence_threshold: 0.001,
        parallel_execution: false,
        adaptive_parameters: false,
    };
    
    let algorithm = GeneticAlgorithm::new(config);
    
    // Create organisms with varying fitness
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for i in 0..30 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let mut genetics = organism.get_genetics();
        genetics.efficiency = (i as f64 / 30.0).powf(2.0); // Quadratic distribution
        organism.set_genetics(genetics);
        
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let fitness_scores: Vec<(Uuid, f64)> = organisms
        .iter()
        .map(|entry| (*entry.key(), entry.value().fitness()))
        .collect();
    
    let selected = algorithm.roulette_wheel_selection(&fitness_scores).await.unwrap();
    
    // Roulette wheel should select organisms probabilistically based on fitness
    assert!(!selected.is_empty());
    assert!(selected.len() <= 6); // Elite percentage of 30
    
    // Higher fitness organisms should have higher probability of selection
    let high_fitness_count = selected.iter()
        .filter_map(|id| organisms.get(id))
        .filter(|entry| entry.value().fitness() > 0.5)
        .count();
    
    let total_high_fitness = organisms.iter()
        .filter(|entry| entry.value().fitness() > 0.5)
        .count();
    
    // Should have reasonable representation of high-fitness organisms
    if total_high_fitness > 0 {
        let selection_ratio = high_fitness_count as f64 / selected.len() as f64;
        let population_ratio = total_high_fitness as f64 / organisms.len() as f64;
        assert!(selection_ratio >= population_ratio * 0.8, "High fitness organisms should be well represented");
    }
}

#[tokio::test]
async fn test_adaptive_parameter_adjustment() {
    let config = GeneticAlgorithmConfig {
        population_size: 40,
        elite_percentage: 0.15,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.2,
        max_generations: 50,
        convergence_threshold: 0.005,
        parallel_execution: true,
        adaptive_parameters: true,
    };
    
    let mut algorithm = GeneticAlgorithm::new(config);
    
    // Create diverse population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for i in 0..40 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = match i % 3 {
            0 => Box::new(CuckooOrganism::new()),
            1 => Box::new(WaspOrganism::new()),
            _ => Box::new(VirusOrganism::new()),
        };
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let initial_mutation_rate = algorithm.get_config().mutation_rate;
    let initial_selection_pressure = algorithm.get_config().selection_pressure;
    
    // Run multiple evolution cycles to trigger adaptation
    for _ in 0..5 {
        let result = algorithm.evolve_population(&organisms).await;
        assert!(result.is_ok());
    }
    
    let final_config = algorithm.get_config();
    
    // Adaptive parameters should have changed based on population dynamics
    assert!(
        final_config.mutation_rate != initial_mutation_rate ||
        final_config.selection_pressure != initial_selection_pressure,
        "Adaptive parameters should adjust during evolution"
    );
}

#[tokio::test]
async fn test_convergence_detection() {
    let config = GeneticAlgorithmConfig {
        population_size: 20,
        elite_percentage: 0.3,
        mutation_rate: 0.05, // Low mutation to promote convergence
        crossover_rate: 0.9,
        selection_pressure: 2.0, // High selection pressure
        max_generations: 100,
        convergence_threshold: 0.01,
        parallel_execution: false,
        adaptive_parameters: false,
    };
    
    let mut algorithm = GeneticAlgorithm::new(config);
    
    // Create population with similar high-fitness organisms to promote convergence
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..20 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let mut genetics = organism.get_genetics();
        genetics.efficiency = 0.9 + (rand::random::<f64>() * 0.1); // High, similar fitness
        genetics.adaptability = 0.85 + (rand::random::<f64>() * 0.15);
        organism.set_genetics(genetics);
        
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let mut generations = 0;
    
    // Run evolution until convergence or max generations
    while !algorithm.has_converged() && generations < 50 {
        let result = algorithm.evolve_population(&organisms).await;
        assert!(result.is_ok());
        generations += 1;
    }
    
    // Should converge due to similar high fitness
    assert!(algorithm.has_converged(), "Algorithm should converge with similar high-fitness population");
    assert!(generations > 0, "Should take at least one generation");
}

#[tokio::test]
async fn test_parallel_vs_sequential_performance() {
    // Test parallel execution
    let parallel_config = GeneticAlgorithmConfig {
        population_size: 100,
        elite_percentage: 0.1,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.2,
        max_generations: 5,
        convergence_threshold: 0.001,
        parallel_execution: true,
        adaptive_parameters: false,
    };
    
    let mut parallel_algorithm = GeneticAlgorithm::new(parallel_config);
    
    // Test sequential execution
    let sequential_config = GeneticAlgorithmConfig {
        population_size: 100,
        elite_percentage: 0.1,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.2,
        max_generations: 5,
        convergence_threshold: 0.001,
        parallel_execution: false,
        adaptive_parameters: false,
    };
    
    let mut sequential_algorithm = GeneticAlgorithm::new(sequential_config);
    
    // Create identical populations for fair comparison
    let create_population = || {
        let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
        for i in 0..100 {
            let organism: Box<dyn ParasiticOrganism + Send + Sync> = match i % 3 {
                0 => Box::new(CuckooOrganism::new()),
                1 => Box::new(WaspOrganism::new()),
                _ => Box::new(VirusOrganism::new()),
            };
            let id = organism.id();
            organisms.insert(id, organism);
        }
        organisms
    };
    
    let parallel_organisms = create_population();
    let sequential_organisms = create_population();
    
    // Measure parallel execution time
    let parallel_start = Instant::now();
    let parallel_result = parallel_algorithm.evolve_population(&parallel_organisms).await;
    let parallel_duration = parallel_start.elapsed();
    
    // Measure sequential execution time
    let sequential_start = Instant::now();
    let sequential_result = sequential_algorithm.evolve_population(&sequential_organisms).await;
    let sequential_duration = sequential_start.elapsed();
    
    assert!(parallel_result.is_ok());
    assert!(sequential_result.is_ok());
    
    // Both should complete in under 1ms for this population size
    assert!(parallel_duration.as_millis() < 1, "Parallel execution too slow: {:?}", parallel_duration);
    assert!(sequential_duration.as_millis() < 1, "Sequential execution too slow: {:?}", sequential_duration);
    
    // Parallel should generally be faster for larger populations (though not always measurable at this scale)
    println!("Parallel: {:?}, Sequential: {:?}", parallel_duration, sequential_duration);
}

#[tokio::test]
async fn test_genetic_diversity_calculation() {
    let config = GeneticAlgorithmConfig {
        population_size: 30,
        elite_percentage: 0.2,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        selection_pressure: 1.0,
        max_generations: 1,
        convergence_threshold: 0.001,
        parallel_execution: false,
        adaptive_parameters: false,
    };
    
    let algorithm = GeneticAlgorithm::new(config);
    
    // Create diverse population
    let diverse_organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for i in 0..30 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let genetics = OrganismGenetics {
            aggression: (i as f64 / 30.0),
            adaptability: ((i * 2) as f64 / 60.0),
            efficiency: ((i * 3) as f64 / 90.0),
            resilience: ((i * 4) as f64 / 120.0),
            reaction_speed: ((i * 5) as f64 / 150.0),
            risk_tolerance: ((i * 6) as f64 / 180.0),
            cooperation: ((i * 7) as f64 / 210.0),
            stealth: ((i * 8) as f64 / 240.0),
        };
        organism.set_genetics(genetics);
        
        let id = organism.id();
        diverse_organisms.insert(id, organism);
    }
    
    // Create homogeneous population
    let homogeneous_organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..30 {
        let mut organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
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
        
        let id = organism.id();
        homogeneous_organisms.insert(id, organism);
    }
    
    let diverse_diversity = algorithm.calculate_genetic_diversity(&diverse_organisms).await;
    let homogeneous_diversity = algorithm.calculate_genetic_diversity(&homogeneous_organisms).await;
    
    assert!(diverse_diversity > homogeneous_diversity, 
            "Diverse population should have higher genetic diversity: {} vs {}", 
            diverse_diversity, homogeneous_diversity);
    assert!(diverse_diversity > 0.1, "Diverse population should have significant diversity");
    assert!(homogeneous_diversity < 0.1, "Homogeneous population should have low diversity");
}

#[tokio::test]
async fn test_evolution_statistics_tracking() {
    let config = GeneticAlgorithmConfig {
        population_size: 25,
        elite_percentage: 0.2,
        mutation_rate: 0.15,
        crossover_rate: 0.75,
        selection_pressure: 1.3,
        max_generations: 10,
        convergence_threshold: 0.001,
        parallel_execution: true,
        adaptive_parameters: true,
    };
    
    let mut algorithm = GeneticAlgorithm::new(config);
    
    // Create population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..25 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Run several evolution cycles and track statistics
    let mut statistics = Vec::new();
    
    for generation in 0..5 {
        let result = algorithm.evolve_population(&organisms).await.unwrap();
        
        let stats = algorithm.get_evolution_statistics().await;
        statistics.push(stats.clone());
        
        // Verify statistics are being tracked correctly
        assert_eq!(stats.current_generation, generation + 1);
        assert!(stats.population_size > 0);
        assert!(stats.average_fitness >= 0.0);
        assert!(stats.best_fitness >= stats.average_fitness);
        assert!(stats.genetic_diversity >= 0.0);
        assert!(stats.operations_performed > 0);
        
        if generation > 0 {
            // Compare with previous generation
            let prev_stats = &statistics[generation - 1];
            assert!(stats.current_generation == prev_stats.current_generation + 1);
        }
    }
    
    // Verify progression over generations
    assert!(statistics.len() == 5);
    assert!(statistics.last().unwrap().current_generation == 5);
}