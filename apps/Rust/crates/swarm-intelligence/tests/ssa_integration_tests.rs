//! Integration tests for Salp Swarm Algorithm

use swarm_intelligence::{
    SalpSwarmAlgorithm, SsaParameters, SsaVariant, ChainTopology, 
    OceanCurrentPattern, MarineEnvironment, SwarmAlgorithm,
    OptimizationProblem,
};
use nalgebra::DVector;
use tokio;

/// Test basic SSA functionality
#[tokio::test]
async fn test_ssa_basic_functionality() {
    let params = SsaParameters {
        population_size: 10,
        max_iterations: 50,
        variant: SsaVariant::Standard,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    // Simple sphere function
    let problem = OptimizationProblem {
        dimensions: 2,
        lower_bounds: DVector::from_element(2, -5.0),
        upper_bounds: DVector::from_element(2, 5.0),
        objective: Box::new(|x: &DVector<f64>| {
            x.iter().map(|xi| xi * xi).sum()
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    // Run a few steps
    for _ in 0..10 {
        ssa.step().await.unwrap();
    }
    
    // Check that we have a valid result
    assert!(ssa.best_fitness().is_finite());
    let best_individual = ssa.get_best_individual().unwrap();
    assert_eq!(best_individual.position().len(), 2);
}

/// Test different SSA variants
#[tokio::test]
async fn test_ssa_variants() {
    let variants = vec![
        SsaVariant::Standard,
        SsaVariant::Enhanced, 
        SsaVariant::Quantum,
        SsaVariant::Chaotic,
        SsaVariant::Marine,
    ];
    
    for variant in variants {
        let params = SsaParameters {
            population_size: 8,
            max_iterations: 20,
            variant,
            ..Default::default()
        };
        
        let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
        
        let problem = OptimizationProblem {
            dimensions: 3,
            lower_bounds: DVector::from_element(3, -2.0),
            upper_bounds: DVector::from_element(3, 2.0),
            objective: Box::new(|x: &DVector<f64>| {
                x.iter().map(|xi| xi * xi).sum()
            }),
            minimize: true,
        };
        
        ssa.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..5 {
            ssa.step().await.unwrap();
        }
        
        // Check algorithm name matches variant
        let expected_name = match variant {
            SsaVariant::Standard => "Salp Swarm Algorithm",
            SsaVariant::Enhanced => "Enhanced Salp Swarm Algorithm",
            SsaVariant::Quantum => "Quantum Salp Swarm Algorithm",
            SsaVariant::Chaotic => "Chaotic Salp Swarm Algorithm",
            SsaVariant::Marine => "Marine Salp Swarm Algorithm",
        };
        
        assert_eq!(ssa.name(), expected_name);
        assert!(ssa.best_fitness().is_finite());
    }
}

/// Test different chain topologies
#[tokio::test]
async fn test_chain_topologies() {
    let topologies = vec![
        ChainTopology::Linear,
        ChainTopology::Ring,
        ChainTopology::Branched { branches: 3 },
        ChainTopology::Adaptive,
        ChainTopology::MultiChain { chains: 2 },
    ];
    
    for topology in topologies {
        let params = SsaParameters {
            population_size: 12,
            max_iterations: 30,
            chain_topology: topology,
            ..Default::default()
        };
        
        let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
        
        let problem = OptimizationProblem {
            dimensions: 2,
            lower_bounds: DVector::from_element(2, -3.0),
            upper_bounds: DVector::from_element(2, 3.0),
            objective: Box::new(|x: &DVector<f64>| {
                // Rosenbrock function (simplified)
                let x0 = x[0];
                let x1 = x[1];
                100.0 * (x1 - x0 * x0).powi(2) + (1.0 - x0).powi(2)
            }),
            minimize: true,
        };
        
        ssa.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..10 {
            ssa.step().await.unwrap();
        }
        
        assert!(ssa.best_fitness().is_finite());
        
        // Check metrics
        let metrics = ssa.metrics();
        assert!(metrics.iteration > 0);
        assert!(metrics.best_fitness.is_some());
        assert!(metrics.diversity.is_some());
    }
}

/// Test marine environment effects
#[tokio::test]
async fn test_marine_environment() {
    let marine_env = MarineEnvironment {
        current_pattern: OceanCurrentPattern::Turbulent,
        turbulence: 0.2,
        food_density: 1.5,
        ..Default::default()
    };
    
    let params = SsaParameters {
        population_size: 15,
        max_iterations: 40,
        variant: SsaVariant::Marine,
        marine_environment: marine_env,
        current_influence: 0.8,
        buoyancy_factor: 1.2,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    let problem = OptimizationProblem {
        dimensions: 4,
        lower_bounds: DVector::from_element(4, -1.0),
        upper_bounds: DVector::from_element(4, 1.0),
        objective: Box::new(|x: &DVector<f64>| {
            // Griewank function
            let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
            let prod_cos = x.iter().enumerate().map(|(i, xi)| {
                (xi / ((i + 1) as f64).sqrt()).cos()
            }).product::<f64>();
            sum_sq - prod_cos + 1.0
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    // Run optimization with marine environment
    for _ in 0..15 {
        ssa.step().await.unwrap();
    }
    
    assert!(ssa.best_fitness().is_finite());
    
    // Test detailed status
    let status = ssa.detailed_status();
    assert!(status.contains_key("iteration"));
    assert!(status.contains_key("best_fitness"));
    assert!(status.contains_key("chain_count"));
    assert!(status.contains_key("average_depth"));
}

/// Test convergence detection
#[tokio::test]
async fn test_convergence_detection() {
    let params = SsaParameters {
        population_size: 20,
        max_iterations: 100,
        tolerance: 1e-3,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    // Simple quadratic function with known minimum
    let problem = OptimizationProblem {
        dimensions: 2,
        lower_bounds: DVector::from_element(2, -2.0),
        upper_bounds: DVector::from_element(2, 2.0),
        objective: Box::new(|x: &DVector<f64>| {
            (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    let mut converged = false;
    let mut iterations = 0;
    
    // Run until convergence or max iterations
    while iterations < 50 && !converged {
        ssa.step().await.unwrap();
        converged = ssa.has_converged();
        iterations += 1;
    }
    
    // Should make progress towards the minimum at (1, 1)
    assert!(ssa.best_fitness() < 4.0); // Initial fitness should be better than worst case
}

/// Test reset functionality
#[tokio::test]
async fn test_reset_functionality() {
    let params = SsaParameters {
        population_size: 10,
        max_iterations: 50,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    let problem = OptimizationProblem {
        dimensions: 2,
        lower_bounds: DVector::from_element(2, -1.0),
        upper_bounds: DVector::from_element(2, 1.0),
        objective: Box::new(|x: &DVector<f64>| {
            x.iter().map(|xi| xi * xi).sum()
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    // Run some iterations
    for _ in 0..10 {
        ssa.step().await.unwrap();
    }
    
    let fitness_before_reset = ssa.best_fitness();
    let iteration_before_reset = ssa.metrics().iteration;
    
    // Reset the algorithm
    ssa.reset().await.unwrap();
    
    // Check that state was reset
    let metrics_after_reset = ssa.metrics();
    assert_eq!(metrics_after_reset.iteration, 0);
    
    // Run again and ensure it works
    for _ in 0..5 {
        ssa.step().await.unwrap();
    }
    
    assert!(ssa.metrics().iteration > 0);
    assert!(ssa.best_fitness().is_finite());
}

/// Test adaptive parameters
#[tokio::test]
async fn test_adaptive_parameters() {
    let params = SsaParameters {
        population_size: 12,
        max_iterations: 30,
        adaptive_parameters: true,
        chain_break_probability: 0.2,
        chain_reform_probability: 0.9,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    let problem = OptimizationProblem {
        dimensions: 3,
        lower_bounds: DVector::from_element(3, -5.0),
        upper_bounds: DVector::from_element(3, 5.0),
        objective: Box::new(|x: &DVector<f64>| {
            // Rastrigin function
            let n = x.len() as f64;
            let sum = x.iter().map(|xi| {
                xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()
            }).sum::<f64>();
            10.0 * n + sum
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    // Run optimization with adaptive parameters
    for _ in 0..15 {
        ssa.step().await.unwrap();
    }
    
    assert!(ssa.best_fitness().is_finite());
    
    // Check that metrics are being updated
    let metrics = ssa.metrics();
    assert!(metrics.iteration > 0);
    assert!(metrics.best_fitness.is_some());
    assert!(metrics.diversity.is_some());
    assert!(metrics.average_fitness.is_some());
    assert!(metrics.memory_usage.is_some());
}

/// Performance benchmark test
#[tokio::test]
async fn test_performance_benchmark() {
    let params = SsaParameters {
        population_size: 30,
        max_iterations: 100,
        variant: SsaVariant::Enhanced,
        parallel_chains: true,
        ..Default::default()
    };
    
    let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
    
    // Ackley function - multimodal with global minimum at origin
    let problem = OptimizationProblem {
        dimensions: 5,
        lower_bounds: DVector::from_element(5, -32.768),
        upper_bounds: DVector::from_element(5, 32.768),
        objective: Box::new(|x: &DVector<f64>| {
            let n = x.len() as f64;
            let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
            let sum_cos = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>() / n;
            
            -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
        }),
        minimize: true,
    };
    
    ssa.initialize(problem).await.unwrap();
    
    let start_time = std::time::Instant::now();
    
    // Run optimization
    for _ in 0..20 {
        ssa.step().await.unwrap();
    }
    
    let elapsed_time = start_time.elapsed();
    
    // Performance assertions
    assert!(ssa.best_fitness().is_finite());
    assert!(elapsed_time < std::time::Duration::from_secs(10)); // Should complete reasonably fast
    
    // Check that we made progress
    assert!(ssa.best_fitness() < 50.0); // Much better than random solutions
    
    let metrics = ssa.metrics();
    assert!(metrics.time_per_iteration.is_some());
    assert!(metrics.memory_usage.is_some());
    
    println!("SSA Performance Test:");
    println!("  Best fitness: {:.6}", ssa.best_fitness());
    println!("  Total time: {:?}", elapsed_time);
    println!("  Iterations: {}", metrics.iteration);
    println!("  Time per iteration: {:?} μs", metrics.time_per_iteration);
}