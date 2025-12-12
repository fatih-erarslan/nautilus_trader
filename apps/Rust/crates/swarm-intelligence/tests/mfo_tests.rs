//! Comprehensive tests for Moth-Flame Optimization (MFO) implementation

use swarm_intelligence::algorithms::mfo::{
    MothFlameOptimization, MfoParameters, MfoVariant, Moth, Flame,
    NavigationStrategy, FlameUpdateStrategy, SpiralPattern
};
use swarm_intelligence::core::{CommonParameters, OptimizationProblem, SwarmAlgorithm};
use nalgebra::DVector;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use approx::assert_relative_eq;

type Float = f64;

/// Test basic MFO creation and initialization
#[tokio::test]
async fn test_mfo_creation_and_initialization() {
    let params = MfoParameters::default();
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    
    assert_eq!(mfo.name(), "Moth-Flame Optimization");
    
    // Test initialization with sphere function
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    mfo.initialize(problem).await.unwrap();
}

/// Test different MFO variants
#[tokio::test]
async fn test_mfo_variants() {
    let variants = vec![
        (MfoVariant::Standard, "Moth-Flame Optimization"),
        (MfoVariant::Enhanced, "Enhanced Moth-Flame Optimization"),
        (MfoVariant::Chaotic, "Chaotic Moth-Flame Optimization"),
        (MfoVariant::Quantum, "Quantum Moth-Flame Optimization"),
        (MfoVariant::LevyFlight, "Levy Flight Moth-Flame Optimization"),
        (MfoVariant::Binary, "Binary Moth-Flame Optimization"),
    ];
    
    for (variant, expected_name) in variants {
        let params = MfoParameters {
            variant,
            ..Default::default()
        };
        
        let mfo = MothFlameOptimization::new(params).unwrap();
        assert_eq!(mfo.name(), expected_name);
    }
}

/// Test MFO on sphere function
#[tokio::test]
async fn test_mfo_sphere_function() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 1000,
            tolerance: 1e-3,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    
    mfo.initialize(problem).await.unwrap();
    
    // Run optimization
    for _ in 0..50 {
        mfo.step().await.unwrap();
        if mfo.has_converged() {
            break;
        }
    }
    
    // Should find a reasonably good solution
    let best_fitness = mfo.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < 10.0);
}

/// Test different navigation strategies
#[tokio::test]
async fn test_navigation_strategies() {
    let strategies = vec![
        NavigationStrategy::LogarithmicSpiral,
        NavigationStrategy::AdaptiveSpiral,
        NavigationStrategy::MultiSpiral,
        NavigationStrategy::Phototaxis,
        NavigationStrategy::RandomWalk,
    ];
    
    for strategy in strategies {
        let params = MfoParameters {
            common: CommonParameters {
                population_size: 15,
                max_evaluations: 300,
                seed: Some(789),
                ..Default::default()
            },
            navigation_strategy: strategy,
            ..Default::default()
        };
        
        let mut mfo = MothFlameOptimization::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        mfo.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            mfo.step().await.unwrap();
        }
    }
}

/// Test different flame update strategies
#[tokio::test]
async fn test_flame_update_strategies() {
    let strategies = vec![
        FlameUpdateStrategy::BestMoths,
        FlameUpdateStrategy::DiversityBased,
        FlameUpdateStrategy::ClusterBased,
        FlameUpdateStrategy::AdaptiveRedistribution,
        FlameUpdateStrategy::EliteBased,
    ];
    
    for strategy in strategies {
        let params = MfoParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 400,
                seed: Some(456),
                ..Default::default()
            },
            flame_update_strategy: strategy,
            ..Default::default()
        };
        
        let mut mfo = MothFlameOptimization::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        mfo.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            mfo.step().await.unwrap();
        }
    }
}

/// Test different spiral patterns
#[tokio::test]
async fn test_spiral_patterns() {
    let patterns = vec![
        SpiralPattern::Logarithmic,
        SpiralPattern::Archimedean,
        SpiralPattern::Golden,
        SpiralPattern::Fibonacci,
        SpiralPattern::Adaptive,
    ];
    
    for pattern in patterns {
        let params = MfoParameters {
            common: CommonParameters {
                population_size: 12,
                max_evaluations: 240,
                seed: Some(321),
                ..Default::default()
            },
            spiral_pattern: pattern,
            navigation_strategy: NavigationStrategy::MultiSpiral,
            ..Default::default()
        };
        
        let mut mfo = MothFlameOptimization::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        mfo.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            mfo.step().await.unwrap();
        }
    }
}

/// Test moth creation and properties
#[test]
fn test_moth_creation() {
    let params = MfoParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
    
    let moth = Moth::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(moth.id, 0);
    assert_eq!(moth.position.len(), 2);
    assert_eq!(moth.velocity.len(), 2);
    assert!(moth.position[0] >= -10.0 && moth.position[0] <= 10.0);
    assert!(moth.position[1] >= -5.0 && moth.position[1] <= 5.0);
    assert!(moth.navigation_angle >= 0.0 && moth.navigation_angle <= 2.0 * std::f64::consts::PI);
    assert!(moth.wing_frequency > 0.0);
    assert!(moth.energy_level > 0.0);
}

/// Test flame creation and properties
#[test]
fn test_flame_creation() {
    let position = DVector::from_vec(vec![1.0, 2.0]);
    let fitness = 0.5;
    let flame = Flame::new(0, position.clone(), fitness);
    
    assert_eq!(flame.id, 0);
    assert_eq!(flame.position, position);
    assert_eq!(flame.fitness, fitness);
    assert!(flame.intensity > 0.0);
    assert_eq!(flame.age, 0);
    assert_eq!(flame.moth_count, 0);
}

/// Test flame intensity update
#[test]
fn test_flame_intensity_update() {
    let position = DVector::from_vec(vec![0.0, 0.0]);
    let mut flame = Flame::new(0, position, 2.0);
    
    let initial_intensity = flame.intensity;
    let initial_age = flame.age;
    
    flame.update_intensity();
    
    assert_eq!(flame.age, initial_age + 1);
    // Intensity calculation: 1.0 / (1.0 + fitness)
    let expected_intensity = 1.0 / (1.0 + 2.0);
    assert_relative_eq!(flame.intensity, expected_intensity, epsilon = 1e-10);
}

/// Test moth navigation mechanisms
#[test]
fn test_moth_navigation() {
    let params = MfoParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 3];
    let mut moth = Moth::new(0, 3, &bounds, &params, &mut rng);
    
    let flame_position = DVector::from_vec(vec![2.0, -1.0, 1.5]);
    let flame = Flame::new(0, flame_position, 1.0);
    let flames = vec![flame];
    
    let initial_position = moth.position.clone();
    
    moth.update_position(&flames, &params, 10, &mut rng).unwrap();
    
    // Position should have been updated due to navigation
    assert_ne!(moth.position, initial_position);
}

/// Test spiral navigation patterns
#[test]
fn test_spiral_navigation_patterns() {
    let params = MfoParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut moth = Moth::new(0, 2, &bounds, &params, &mut rng);
    
    let flame_position = DVector::from_vec(vec![3.0, 4.0]);
    let flame = Flame::new(0, flame_position, 0.5);
    
    let initial_position = moth.position.clone();
    
    // Test logarithmic spiral
    moth.logarithmic_spiral_navigation(&flame, &params, 5, &mut rng).unwrap();
    assert_ne!(moth.position, initial_position);
    
    // Reset and test Archimedean spiral
    moth.position = initial_position.clone();
    moth.archimedean_spiral_navigation(&flame, &params, &mut rng).unwrap();
    assert_ne!(moth.position, initial_position);
    
    // Reset and test golden spiral
    moth.position = initial_position.clone();
    moth.golden_spiral_navigation(&flame, &params, &mut rng).unwrap();
    assert_ne!(moth.position, initial_position);
}

/// Test Levy flight generation
#[test]
fn test_levy_flight_generation() {
    let params = MfoParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let moth = Moth::new(0, 2, &bounds, &params, &mut rng);
    
    let levy_step = moth.generate_levy_step(1.5, &mut rng);
    
    // Levy step should be finite
    assert!(levy_step.is_finite());
}

/// Test boundary handling
#[test]
fn test_boundary_handling() {
    let params = MfoParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-1.0, 1.0); 2];
    let mut moth = Moth::new(0, 2, &bounds, &params, &mut rng);
    
    // Set moth outside bounds
    moth.position = DVector::from_vec(vec![2.0, -2.0]);
    
    moth.apply_boundaries(&bounds);
    
    // Position should be within bounds
    assert!(moth.position[0] >= -1.0 && moth.position[0] <= 1.0);
    assert!(moth.position[1] >= -1.0 && moth.position[1] <= 1.0);
}

/// Test flame memory update
#[test]
fn test_flame_memory_update() {
    let params = MfoParameters {
        flame_memory_duration: 3,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut moth = Moth::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(moth.flame_memory.len(), 0);
    
    // Add flames to memory
    for i in 0..5 {
        let flame_pos = DVector::from_vec(vec![i as Float, (i + 1) as Float]);
        let flame = Flame::new(i, flame_pos, i as Float);
        moth.update_flame_memory(&flame, &params);
    }
    
    // Memory should be limited to flame_memory_duration
    assert_eq!(moth.flame_memory.len(), params.flame_memory_duration);
}

/// Test convergence detection
#[tokio::test]
async fn test_convergence_detection() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 10,
            max_evaluations: 1000,
            tolerance: 1e-6,
            seed: Some(42),
            ..Default::default()
        },
        convergence_threshold: 1e-6,
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_simple_problem();
    
    mfo.initialize(problem).await.unwrap();
    
    // Should eventually converge or run without error
    for _ in 0..50 {
        mfo.step().await.unwrap();
        if mfo.has_converged() {
            break;
        }
    }
    
    // Test passes if no errors occur
}

/// Test parallel evaluation
#[tokio::test]
async fn test_parallel_evaluation() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 400,
            parallel_evaluation: true,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-5.0, 5.0); 3]);
    
    mfo.initialize(problem).await.unwrap();
    
    // Should work with parallel evaluation
    for _ in 0..10 {
        mfo.step().await.unwrap();
    }
    
    let best_fitness = mfo.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < Float::INFINITY);
}

/// Test algorithm metrics
#[tokio::test]
async fn test_algorithm_metrics() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 15,
            max_evaluations: 300,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
    
    mfo.initialize(problem).await.unwrap();
    
    // Run a few iterations and check metrics
    for _ in 0..5 {
        mfo.step().await.unwrap();
    }
    
    let metrics = mfo.metrics();
    assert!(metrics.total_iterations > 0);
    assert!(metrics.last_step_duration.as_nanos() > 0);
}

/// Test flame count reduction over iterations
#[tokio::test]
async fn test_flame_count_reduction() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 400,
            seed: Some(42),
            ..Default::default()
        },
        flame_reduction_factor: 0.8,
        min_flame_count: 2,
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
    
    mfo.initialize(problem).await.unwrap();
    
    let initial_flame_count = mfo.current_flame_count;
    
    // Run several iterations
    for _ in 0..20 {
        mfo.step().await.unwrap();
    }
    
    // Flame count should have decreased but not below minimum
    assert!(mfo.current_flame_count <= initial_flame_count);
    assert!(mfo.current_flame_count >= params.min_flame_count);
}

/// Test population diversity calculation
#[tokio::test]
async fn test_population_diversity() {
    let params = MfoParameters {
        common: CommonParameters {
            population_size: 10,
            max_evaluations: 200,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut mfo = MothFlameOptimization::new(params).unwrap();
    let problem = create_sphere_problem(2, vec![(-5.0, 5.0); 2]);
    
    mfo.initialize(problem).await.unwrap();
    
    // Run a few iterations
    for _ in 0..5 {
        mfo.step().await.unwrap();
    }
    
    // Diversity should be positive for a non-converged population
    let metrics = mfo.metrics();
    assert!(metrics.diversity >= 0.0);
}

// Helper functions for creating test problems

fn create_sphere_problem(dimensions: usize, bounds: Vec<(Float, Float)>) -> OptimizationProblem {
    OptimizationProblem::new(
        dimensions,
        bounds,
        Box::new(|x: &DVector<Float>| {
            Ok(x.iter().map(|xi| xi * xi).sum())
        }),
    )
}

fn create_simple_problem() -> OptimizationProblem {
    OptimizationProblem::new(
        2,
        vec![(-1.0, 1.0); 2],
        Box::new(|x: &DVector<Float>| {
            Ok(x[0] * x[0] + x[1] * x[1])
        }),
    )
}