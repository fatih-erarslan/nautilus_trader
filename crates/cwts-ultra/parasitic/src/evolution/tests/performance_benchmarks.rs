//! Performance benchmarks for evolution engine components
//! Validates sub-millisecond performance requirements
//! Real implementation testing with no mocks

use std::time::Instant;
use std::sync::Arc;
use dashmap::DashMap;
use uuid::Uuid;
use super::super::*;
use crate::organisms::{ParasiticOrganism, CuckooOrganism, WaspOrganism, VirusOrganism};

/// Performance target constants
const SUB_MILLISECOND_TARGET: u128 = 1; // 1 millisecond
const ULTRA_FAST_TARGET: u128 = 500; // 500 microseconds

#[tokio::test]
async fn benchmark_genetic_algorithm_performance() {
    let config = GeneticAlgorithmConfig {
        population_size: 100,
        parallel_execution: true,
        adaptive_parameters: false, // Disable for consistent timing
        ..Default::default()
    };
    
    let mut algorithm = GeneticAlgorithm::new(config);
    
    // Create test population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..100 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    // Warm-up run
    let _ = algorithm.evolve_population(&organisms).await.unwrap();
    
    // Benchmark multiple runs
    let mut times = Vec::new();
    const BENCHMARK_RUNS: usize = 10;
    
    for _ in 0..BENCHMARK_RUNS {
        let start = Instant::now();
        let result = algorithm.evolve_population(&organisms).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        times.push(duration.as_micros());
    }
    
    let average_time = times.iter().sum::<u128>() / times.len() as u128;
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();
    
    println!("Genetic Algorithm Performance (100 organisms):");
    println!("  Average: {}µs", average_time);
    println!("  Min: {}µs", min_time);
    println!("  Max: {}µs", max_time);
    println!("  Target: <{}µs ({}ms)", SUB_MILLISECOND_TARGET * 1000, SUB_MILLISECOND_TARGET);
    
    assert!(average_time < SUB_MILLISECOND_TARGET * 1000, 
            "Genetic algorithm should complete in under {}ms, average: {}µs", 
            SUB_MILLISECOND_TARGET, average_time);
}

#[tokio::test]
async fn benchmark_fitness_evaluator_performance() {
    let config = FitnessEvaluationConfig {
        real_time_evaluation: true,
        ..Default::default()
    };
    
    let mut evaluator = FitnessEvaluator::new(config);
    
    // Create test population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..100 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Warm-up run
    let _ = evaluator.evaluate_population_fitness(&organisms, &market_conditions).await.unwrap();
    
    // Benchmark batch evaluation
    let mut batch_times = Vec::new();
    const BATCH_RUNS: usize = 10;
    
    for _ in 0..BATCH_RUNS {
        let start = Instant::now();
        let result = evaluator.evaluate_population_fitness(&organisms, &market_conditions).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        batch_times.push(duration.as_micros());
    }
    
    let average_batch_time = batch_times.iter().sum::<u128>() / batch_times.len() as u128;
    let throughput = (100.0 * 1_000_000.0) / average_batch_time as f64; // evaluations per second
    
    println!("Fitness Evaluator Performance (100 organisms):");
    println!("  Average batch time: {}µs", average_batch_time);
    println!("  Throughput: {:.0} evaluations/second", throughput);
    println!("  Per organism: {:.1}µs", average_batch_time as f64 / 100.0);
    
    assert!(average_batch_time < SUB_MILLISECOND_TARGET * 1000, 
            "Batch fitness evaluation should complete in under {}ms", SUB_MILLISECOND_TARGET);
    
    // Individual evaluation benchmark
    let organism = CuckooOrganism::new();
    let mut individual_times = Vec::new();
    
    for _ in 0..100 {
        let start = Instant::now();
        let result = evaluator.evaluate_fitness(&organism, &market_conditions).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        individual_times.push(duration.as_nanos());
    }
    
    let average_individual_time = individual_times.iter().sum::<u128>() / individual_times.len() as u128;
    
    println!("  Individual evaluation: {}ns", average_individual_time);
    assert!(average_individual_time < 100_000, // 100 microseconds
            "Individual fitness evaluation should be very fast: {}ns", average_individual_time);
}

#[tokio::test]
async fn benchmark_mutation_engine_performance() {
    let config = MutationEngineConfig {
        base_mutation_rate: 0.2,
        adaptive_mutation: true,
        gaussian_mutation: true,
        ..Default::default()
    };
    
    let mut engine = MutationEngine::new(config);
    
    // Create test population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..100 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(VirusOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let population_diversity = 0.3;
    
    // Warm-up run
    let _ = engine.apply_mutations(&organisms, population_diversity).await.unwrap();
    
    // Benchmark mutation application
    let mut times = Vec::new();
    const MUTATION_RUNS: usize = 10;
    
    for _ in 0..MUTATION_RUNS {
        let start = Instant::now();
        let result = engine.apply_mutations(&organisms, population_diversity).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        times.push(duration.as_micros());
    }
    
    let average_time = times.iter().sum::<u128>() / times.len() as u128;
    let mutation_rate = 100.0 * 1_000_000.0 / average_time as f64; // mutations per second estimate
    
    println!("Mutation Engine Performance (100 organisms):");
    println!("  Average time: {}µs", average_time);
    println!("  Estimated mutation rate: {:.0} operations/second", mutation_rate);
    
    assert!(average_time < SUB_MILLISECOND_TARGET * 1000, 
            "Mutation application should complete in under {}ms", SUB_MILLISECOND_TARGET);
}

#[tokio::test]
async fn benchmark_complete_evolution_cycle() {
    let mut engine = create_high_performance_evolution_engine();
    
    // Test different population sizes
    let population_sizes = vec![25, 50, 100];
    
    for &pop_size in &population_sizes {
        println!("\nBenchmarking population size: {}", pop_size);
        
        // Create population
        let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
        
        for i in 0..pop_size {
            let organism: Box<dyn ParasiticOrganism + Send + Sync> = match i % 3 {
                0 => Box::new(CuckooOrganism::new()),
                1 => Box::new(WaspOrganism::new()),
                _ => Box::new(VirusOrganism::new()),
            };
            let id = organism.id();
            organisms.insert(id, organism);
        }
        
        let market_conditions = MarketConditions::default();
        
        // Warm-up
        let _ = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        engine.reset().await;
        
        // Benchmark runs
        let mut times = Vec::new();
        const CYCLE_RUNS: usize = 5;
        
        for _ in 0..CYCLE_RUNS {
            let start = Instant::now();
            let result = engine.evolve_organisms(&organisms, &market_conditions).await;
            let duration = start.elapsed();
            
            assert!(result.is_ok());
            times.push(duration.as_micros());
            
            engine.reset().await; // Reset for next run
        }
        
        let average_time = times.iter().sum::<u128>() / times.len() as u128;
        let min_time = *times.iter().min().unwrap();
        let max_time = *times.iter().max().unwrap();
        let throughput = (pop_size as f64 * 1_000_000.0) / average_time as f64;
        
        println!("  Average: {}µs", average_time);
        println!("  Min: {}µs", min_time);
        println!("  Max: {}µs", max_time);
        println!("  Throughput: {:.0} organisms/second", throughput);
        
        // Performance requirements scale with population size
        let target_time = if pop_size <= 50 {
            SUB_MILLISECOND_TARGET * 1000 // 1ms for smaller populations
        } else {
            SUB_MILLISECOND_TARGET * 2000 // 2ms for larger populations
        };
        
        assert!(average_time < target_time, 
                "Complete evolution cycle for {} organisms should complete in under {}µs, got {}µs", 
                pop_size, target_time, average_time);
    }
}

#[tokio::test]
async fn benchmark_scaling_performance() {
    let mut engine = create_high_performance_evolution_engine();
    let market_conditions = MarketConditions::default();
    
    // Test performance scaling with population size
    let population_sizes = vec![10, 25, 50, 100, 150];
    let mut scaling_results = Vec::new();
    
    for &pop_size in &population_sizes {
        // Create population
        let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
        
        for _ in 0..pop_size {
            let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
            let id = organism.id();
            organisms.insert(id, organism);
        }
        
        // Warm-up
        let _ = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        engine.reset().await;
        
        // Single benchmark run
        let start = Instant::now();
        let result = engine.evolve_organisms(&organisms, &market_conditions).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        
        let time_per_organism = duration.as_nanos() as f64 / pop_size as f64;
        scaling_results.push((pop_size, duration.as_micros(), time_per_organism));
        
        engine.reset().await;
    }
    
    println!("\nScaling Performance Analysis:");
    println!("Pop Size | Total Time (µs) | Time per Organism (ns)");
    println!("---------|------------------|----------------------");
    
    for (pop_size, total_time, time_per_organism) in &scaling_results {
        println!("{:8} | {:15} | {:20.1}", pop_size, total_time, time_per_organism);
    }
    
    // Analyze scaling efficiency
    let base_throughput = scaling_results[0].2; // Time per organism for smallest population
    let largest_throughput = scaling_results.last().unwrap().2;
    
    let efficiency_ratio = base_throughput / largest_throughput;
    println!("\nScaling Efficiency: {:.2}x", efficiency_ratio);
    
    // Should scale reasonably well (within 3x degradation)
    assert!(efficiency_ratio > 0.33, 
            "Performance should not degrade by more than 3x with population scaling, got {:.2}x", 
            efficiency_ratio);
}

#[tokio::test]
async fn benchmark_memory_efficiency() {
    let mut engine = create_evolution_engine();
    
    // Create large population to test memory usage
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..200 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(CuckooOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Measure memory usage over multiple generations
    let mut memory_measurements = Vec::new();
    
    for generation in 0..10 {
        let start = Instant::now();
        let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        let duration = start.elapsed();
        
        // Approximate memory efficiency from performance metrics
        let efficiency = result.performance_metrics.memory_efficiency_score;
        memory_measurements.push((generation + 1, efficiency, duration.as_micros()));
        
        println!("Generation {}: Memory Efficiency = {:.3}, Time = {}µs", 
                 generation + 1, efficiency, duration.as_micros());
    }
    
    // Memory efficiency should remain reasonable
    let average_efficiency: f64 = memory_measurements.iter()
        .map(|(_, eff, _)| *eff)
        .sum::<f64>() / memory_measurements.len() as f64;
    
    println!("Average Memory Efficiency: {:.3}", average_efficiency);
    
    assert!(average_efficiency > 0.3, 
            "Average memory efficiency should be reasonable: {:.3}", average_efficiency);
    
    // Performance should remain consistent across generations
    let times: Vec<u128> = memory_measurements.iter().map(|(_, _, time)| *time).collect();
    let time_variance = calculate_variance(&times);
    let average_time = times.iter().sum::<u128>() / times.len() as u128;
    let coefficient_of_variation = (time_variance.sqrt() / average_time as f64) * 100.0;
    
    println!("Performance Consistency (CV): {:.1}%", coefficient_of_variation);
    
    assert!(coefficient_of_variation < 50.0, 
            "Performance should be consistent across generations: {:.1}% CV", 
            coefficient_of_variation);
}

#[tokio::test]
async fn stress_test_continuous_evolution() {
    let mut engine = create_high_performance_evolution_engine();
    
    // Create moderate population
    let organisms: Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>> = Arc::new(DashMap::new());
    
    for _ in 0..50 {
        let organism: Box<dyn ParasiticOrganism + Send + Sync> = Box::new(WaspOrganism::new());
        let id = organism.id();
        organisms.insert(id, organism);
    }
    
    let market_conditions = MarketConditions::default();
    
    // Run continuous evolution for extended period
    const STRESS_GENERATIONS: usize = 100;
    let mut performance_history = Vec::new();
    let overall_start = Instant::now();
    
    for generation in 0..STRESS_GENERATIONS {
        let start = Instant::now();
        let result = engine.evolve_organisms(&organisms, &market_conditions).await.unwrap();
        let duration = start.elapsed();
        
        performance_history.push(duration.as_micros());
        
        // Check for performance degradation
        if generation > 0 && generation % 10 == 0 {
            let recent_avg: u128 = performance_history.iter()
                .rev()
                .take(10)
                .sum::<u128>() / 10;
            
            let initial_avg: u128 = performance_history.iter()
                .take(10)
                .sum::<u128>() / 10.min(performance_history.len());
            
            let degradation_ratio = recent_avg as f64 / initial_avg as f64;
            
            assert!(degradation_ratio < 2.0, 
                    "Performance should not degrade significantly over time: {:.2}x at generation {}", 
                    degradation_ratio, generation);
        }
    }
    
    let total_duration = overall_start.elapsed();
    let average_time = performance_history.iter().sum::<u128>() / performance_history.len() as u128;
    let total_operations = STRESS_GENERATIONS as f64;
    let throughput = total_operations / total_duration.as_secs_f64();
    
    println!("Stress Test Results ({} generations):", STRESS_GENERATIONS);
    println!("  Total time: {:.2}s", total_duration.as_secs_f64());
    println!("  Average per generation: {}µs", average_time);
    println!("  Throughput: {:.1} generations/second", throughput);
    
    // Should complete stress test in reasonable time
    assert!(total_duration.as_secs() < 30, 
            "Stress test should complete in under 30 seconds, took {:.2}s", 
            total_duration.as_secs_f64());
    
    // Average performance should still meet targets
    assert!(average_time < SUB_MILLISECOND_TARGET * 2000, // Allow 2ms under stress
            "Average performance under stress should be reasonable: {}µs", average_time);
}

/// Helper function to calculate variance
fn calculate_variance(values: &[u128]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<u128>() as f64 / values.len() as f64;
    let variance = values.iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance
}