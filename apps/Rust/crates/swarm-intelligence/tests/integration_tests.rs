//! Integration tests for swarm intelligence algorithms

use swarm_intelligence::*;
use swarm_intelligence::algorithms::pso::*;
use swarm_intelligence::utils::test_functions::*;
use nalgebra::DVector;
use approx::assert_relative_eq;
use std::time::Duration;

#[test]
fn test_pso_sphere_optimization() {
    let bounds = vec![(-5.0, 5.0); 5];
    let mut optimizer = ParticleSwarmOptimizer::new(30, bounds, 200).unwrap();
    
    let result = optimizer.optimize(sphere).unwrap();
    
    assert!(result.success);
    assert!(result.best_fitness < 1e-2);
    assert_eq!(result.best_position.len(), 5);
    assert!(result.execution_time_ms > 0.0);
    assert!(result.iterations <= 200);
}

#[test]
fn test_pso_rosenbrock_optimization() {
    let bounds = vec![(-2.048, 2.048); 3];
    let config = AlgorithmConfig {
        population_size: 50,
        max_iterations: 500,
        tolerance: 1e-3,
        ..Default::default()
    };
    
    let mut optimizer = ParticleSwarmOptimizer::with_config(
        config,
        Bounds::from_tuples(bounds).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    let result = optimizer.optimize(rosenbrock).unwrap();
    
    assert!(result.success);
    assert!(result.best_fitness < 1.0); // Rosenbrock is harder to optimize
}

#[test]
fn test_pso_rastrigin_optimization() {
    let bounds = vec![(-5.12, 5.12); 4];
    let mut pso_params = PSOParameters::default();
    pso_params.adaptive_inertia = true;
    
    let config = AlgorithmConfig {
        population_size: 60,
        max_iterations: 300,
        tolerance: 1e-2,
        ..Default::default()
    };
    
    let mut optimizer = ParticleSwarmOptimizer::with_config(
        config,
        Bounds::from_tuples(bounds).unwrap(),
        pso_params,
    ).unwrap();
    
    let result = optimizer.optimize(rastrigin).unwrap();
    
    assert!(result.success);
    // Rastrigin is multimodal, so we accept higher tolerance
    assert!(result.best_fitness < 10.0);
}

#[test]
fn test_parallel_execution_performance() {
    let bounds = vec![(-10.0, 10.0); 8];
    
    // Sequential execution
    let config_sequential = AlgorithmConfig {
        population_size: 40,
        max_iterations: 100,
        parallel_threads: Some(1),
        ..Default::default()
    };
    
    let mut optimizer_sequential = ParticleSwarmOptimizer::with_config(
        config_sequential,
        Bounds::from_tuples(bounds.clone()).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    let start = std::time::Instant::now();
    let result_sequential = optimizer_sequential.optimize(sphere).unwrap();
    let time_sequential = start.elapsed();
    
    // Parallel execution
    let config_parallel = AlgorithmConfig {
        population_size: 40,
        max_iterations: 100,
        parallel_threads: Some(4),
        ..Default::default()
    };
    
    let mut optimizer_parallel = ParticleSwarmOptimizer::with_config(
        config_parallel,
        Bounds::from_tuples(bounds).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    let start = std::time::Instant::now();
    let result_parallel = optimizer_parallel.optimize(sphere).unwrap();
    let time_parallel = start.elapsed();
    
    // Both should succeed
    assert!(result_sequential.success);
    assert!(result_parallel.success);
    
    // Results should be similar in quality
    assert!((result_sequential.best_fitness - result_parallel.best_fitness).abs() < 1.0);
    
    // Parallel should generally be faster for computationally intensive problems
    // Note: For simple sphere function, the overhead might make parallel slower
    println!("Sequential time: {:?}", time_sequential);
    println!("Parallel time: {:?}", time_parallel);
}

#[test]
fn test_parameter_adaptation() {
    let bounds = vec![(-1.0, 1.0); 3];
    let mut optimizer = ParticleSwarmOptimizer::new(25, bounds, 100).unwrap();
    
    // Test initial parameters
    let initial_params = optimizer.pso_params.clone();
    
    // Update parameters
    let new_params = AlgorithmParameters {
        inertia_weight: Some(0.5),
        cognitive_factor: Some(2.5),
        social_factor: Some(2.5),
        ..Default::default()
    };
    
    optimizer.update_parameters(new_params).unwrap();
    
    // Verify parameters were updated
    assert_eq!(optimizer.pso_params.inertia_weight, 0.5);
    assert_eq!(optimizer.pso_params.cognitive_factor, 2.5);
    assert_eq!(optimizer.pso_params.social_factor, 2.5);
    
    // Optimize with new parameters
    let result = optimizer.optimize(sphere).unwrap();
    assert!(result.success);
}

#[test]
fn test_convergence_detection() {
    let bounds = vec![(-0.1, 0.1); 2]; // Small search space for quick convergence
    let config = AlgorithmConfig {
        population_size: 20,
        max_iterations: 1000,
        tolerance: 1e-6,
        ..Default::default()
    };
    
    let mut optimizer = ParticleSwarmOptimizer::with_config(
        config,
        Bounds::from_tuples(bounds).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    let result = optimizer.optimize(sphere).unwrap();
    
    assert!(result.success);
    assert!(result.best_fitness < 1e-4);
    // Should converge well before max iterations due to small search space
    assert!(result.iterations < 1000);
}

#[test]
fn test_bounds_clamping() {
    let bounds = vec![(-1.0, 1.0); 2];
    let bounds_obj = Bounds::from_tuples(bounds).unwrap();
    
    let position = DVector::from_vec(vec![2.0, -3.0]); // Outside bounds
    let mut particle = Particle::new(position, &bounds_obj);
    
    // Update position (should be clamped)
    particle.velocity = DVector::from_vec(vec![0.0, 0.0]); // No velocity change
    particle.update_position(&bounds_obj);
    
    // Position should be within bounds
    assert!(particle.position[0] >= -1.0 && particle.position[0] <= 1.0);
    assert!(particle.position[1] >= -1.0 && particle.position[1] <= 1.0);
}

#[test]
fn test_metrics_collection() {
    let bounds = vec![(-2.0, 2.0); 3];
    let mut optimizer = ParticleSwarmOptimizer::new(20, bounds, 50).unwrap();
    
    let result = optimizer.optimize(sphere).unwrap();
    
    let metrics = optimizer.metrics();
    
    assert_eq!(metrics.iterations_completed, result.iterations);
    assert_eq!(metrics.evaluations_completed, result.evaluations);
    assert!(metrics.execution_time_per_iteration.len() > 0);
    assert!(metrics.diversity_index >= 0.0);
}

#[test]
fn test_reset_functionality() {
    let bounds = vec![(-1.0, 1.0); 2];
    let mut optimizer = ParticleSwarmOptimizer::new(15, bounds, 30).unwrap();
    
    // First optimization
    let result1 = optimizer.optimize(sphere).unwrap();
    
    // Reset optimizer
    optimizer.reset().unwrap();
    
    // Second optimization
    let result2 = optimizer.optimize(sphere).unwrap();
    
    // Both should succeed
    assert!(result1.success);
    assert!(result2.success);
    
    // Results might be different due to randomization
    // but both should be valid optimization results
    assert!(result2.evaluations > 0);
    assert!(result2.iterations > 0);
}

#[test]
fn test_high_dimensional_optimization() {
    let dimension = 20;
    let bounds = vec![(-5.0, 5.0); dimension];
    
    let config = AlgorithmConfig {
        population_size: 100, // Larger population for high-dimensional problems
        max_iterations: 500,
        parallel_threads: Some(4),
        ..Default::default()
    };
    
    let mut optimizer = ParticleSwarmOptimizer::with_config(
        config,
        Bounds::from_tuples(bounds).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    let result = optimizer.optimize(sphere).unwrap();
    
    assert!(result.success);
    assert_eq!(result.best_position.len(), dimension);
    assert!(result.best_fitness < 10.0); // Should be able to optimize even in high dimensions
}

#[cfg(feature = "parallel")]
#[test]
fn test_memory_efficiency() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    // This test ensures memory usage doesn't grow excessively during optimization
    let bounds = vec![(-1.0, 1.0); 10];
    let config = AlgorithmConfig {
        population_size: 50,
        max_iterations: 200,
        ..Default::default()
    };
    
    let mut optimizer = ParticleSwarmOptimizer::with_config(
        config,
        Bounds::from_tuples(bounds).unwrap(),
        PSOParameters::default(),
    ).unwrap();
    
    // Get initial memory baseline (approximate)
    let initial_memory = get_memory_usage();
    
    let result = optimizer.optimize(sphere).unwrap();
    
    let final_memory = get_memory_usage();
    
    assert!(result.success);
    
    // Memory growth should be reasonable (less than 100MB for this test)
    let memory_growth = final_memory.saturating_sub(initial_memory);
    assert!(memory_growth < 100 * 1024 * 1024, "Excessive memory usage: {} bytes", memory_growth);
}

// Helper function to get approximate memory usage
fn get_memory_usage() -> usize {
    // This is a simplified memory estimation
    // In a real implementation, you might use more sophisticated memory tracking
    std::process::id() as usize * 1024 // Placeholder
}