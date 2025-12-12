//! Comprehensive tests for Whale Optimization Algorithm (WOA) implementation

use swarm_intelligence::algorithms::woa::{
    WhaleOptimizationAlgorithm, WoaParameters, WoaVariant, Whale,
    HuntingStrategy, PodFormation, BubbleNetPattern, HuntingRole
};
use swarm_intelligence::core::{CommonParameters, OptimizationProblem, SwarmAlgorithm};
use nalgebra::DVector;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use approx::assert_relative_eq;

type Float = f64;

/// Test basic WOA creation and initialization
#[tokio::test]
async fn test_woa_creation_and_initialization() {
    let params = WoaParameters::default();
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    
    assert_eq!(woa.name(), "Whale Optimization Algorithm");
    
    // Test initialization with sphere function
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    woa.initialize(problem).await.unwrap();
}

/// Test different WOA variants
#[tokio::test]
async fn test_woa_variants() {
    let variants = vec![
        (WoaVariant::Standard, "Whale Optimization Algorithm"),
        (WoaVariant::Enhanced, "Enhanced Whale Optimization Algorithm"),
        (WoaVariant::Chaotic, "Chaotic Whale Optimization Algorithm"),
        (WoaVariant::Quantum, "Quantum Whale Optimization Algorithm"),
        (WoaVariant::Binary, "Binary Whale Optimization Algorithm"),
        (WoaVariant::MultiObjective, "Multi-Objective Whale Optimization Algorithm"),
    ];
    
    for (variant, expected_name) in variants {
        let params = WoaParameters {
            variant,
            ..Default::default()
        };
        
        let woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        assert_eq!(woa.name(), expected_name);
    }
}

/// Test WOA on sphere function
#[tokio::test]
async fn test_woa_sphere_function() {
    let params = WoaParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 1000,
            tolerance: 1e-3,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    
    woa.initialize(problem).await.unwrap();
    
    // Run optimization
    for _ in 0..50 {
        woa.step().await.unwrap();
        if woa.has_converged() {
            break;
        }
    }
    
    // Should find a reasonably good solution
    let best_fitness = woa.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < 10.0);
}

/// Test WOA on Rastrigin function
#[tokio::test]
async fn test_woa_rastrigin_function() {
    let params = WoaParameters {
        common: CommonParameters {
            population_size: 30,
            max_evaluations: 2000,
            tolerance: 1e-2,
            seed: Some(123),
            ..Default::default()
        },
        variant: WoaVariant::Enhanced,
        ..Default::default()
    };
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_rastrigin_problem(3, vec![(-5.12, 5.12); 3]);
    
    woa.initialize(problem).await.unwrap();
    
    // Run optimization
    for _ in 0..100 {
        woa.step().await.unwrap();
        if woa.has_converged() {
            break;
        }
    }
    
    let best_fitness = woa.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < 50.0); // Rastrigin is challenging
}

/// Test different hunting strategies
#[tokio::test]
async fn test_hunting_strategies() {
    let strategies = vec![
        HuntingStrategy::BubbleNet,
        HuntingStrategy::Encircling,
        HuntingStrategy::RandomSearch,
        HuntingStrategy::Cooperative,
        HuntingStrategy::DeepDive,
    ];
    
    for strategy in strategies {
        let params = WoaParameters {
            common: CommonParameters {
                population_size: 15,
                max_evaluations: 300,
                seed: Some(789),
                ..Default::default()
            },
            hunting_strategy: strategy,
            ..Default::default()
        };
        
        let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        woa.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            woa.step().await.unwrap();
        }
    }
}

/// Test different pod formations
#[tokio::test]
async fn test_pod_formations() {
    let formations = vec![
        PodFormation::Single,
        PodFormation::Multiple,
        PodFormation::Linear,
        PodFormation::Circular,
        PodFormation::VFormation,
    ];
    
    for formation in formations {
        let params = WoaParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 400,
                seed: Some(456),
                ..Default::default()
            },
            pod_formation: formation,
            ..Default::default()
        };
        
        let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        woa.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            woa.step().await.unwrap();
        }
    }
}

/// Test different bubble-net patterns
#[tokio::test]
async fn test_bubble_net_patterns() {
    let patterns = vec![
        BubbleNetPattern::UpwardSpiral,
        BubbleNetPattern::ShrinkingCircle,
        BubbleNetPattern::Figure8,
        BubbleNetPattern::DoubleHelix,
        BubbleNetPattern::Random,
    ];
    
    for pattern in patterns {
        let params = WoaParameters {
            common: CommonParameters {
                population_size: 12,
                max_evaluations: 240,
                seed: Some(321),
                ..Default::default()
            },
            bubble_net_pattern: pattern,
            ..Default::default()
        };
        
        let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        woa.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            woa.step().await.unwrap();
        }
    }
}

/// Test whale creation and properties
#[test]
fn test_whale_creation() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
    
    let whale = Whale::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(whale.id, 0);
    assert_eq!(whale.position.len(), 2);
    assert_eq!(whale.velocity.len(), 2);
    assert!(whale.position[0] >= -10.0 && whale.position[0] <= 10.0);
    assert!(whale.position[1] >= -5.0 && whale.position[1] <= 5.0);
    assert!(whale.energy_level > 0.0);
    assert!(whale.dive_depth > 0.0);
    assert!(whale.communication_range > 0.0);
}

/// Test whale position update mechanisms
#[test]
fn test_whale_position_update() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 3];
    let mut whale = Whale::new(0, 3, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![1.0, 2.0, -1.0]);
    let random_whale = DVector::from_vec(vec![0.5, -0.5, 0.8]);
    let initial_position = whale.position.clone();
    
    whale.update_position(&global_best, &random_whale, &params, 5, 100, &mut rng).unwrap();
    
    // Position should have been updated
    assert_ne!(whale.position, initial_position);
}

/// Test bubble-net attack patterns
#[test]
fn test_bubble_net_attack() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 3];
    let mut whale = Whale::new(0, 3, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![2.0, -1.0, 1.5]);
    let initial_position = whale.position.clone();
    
    whale.bubble_net_attack(&global_best, &params, 10, &mut rng).unwrap();
    
    // Position should have changed due to bubble-net attack
    assert_ne!(whale.position, initial_position);
}

/// Test Levy flight generation
#[test]
fn test_levy_flight_generation() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let whale = Whale::new(0, 2, &bounds, &params, &mut rng);
    
    let levy_step = whale.generate_levy_step(1.5, &mut rng);
    
    // Levy step should be finite
    assert!(levy_step.is_finite());
}

/// Test boundary constraints
#[test]
fn test_boundary_handling() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-1.0, 1.0); 2];
    let mut whale = Whale::new(0, 2, &bounds, &params, &mut rng);
    
    // Set whale outside bounds
    whale.position = DVector::from_vec(vec![2.0, -2.0]);
    
    whale.apply_boundaries(&bounds);
    
    // Position should be within bounds
    assert!(whale.position[0] >= -1.0 && whale.position[0] <= 1.0);
    assert!(whale.position[1] >= -1.0 && whale.position[1] <= 1.0);
}

/// Test whale migration between pods
#[test]
fn test_whale_migration() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut whale = Whale::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(whale.pod_id, None);
    assert_eq!(whale.migration_count, 0);
    
    whale.migrate_pod(1, &params);
    
    assert_eq!(whale.pod_id, Some(1));
    assert_eq!(whale.migration_count, 1);
    
    // Energy should decrease due to migration cost
    assert!(whale.energy_level < 1.0);
}

/// Test hunting role assignments
#[test]
fn test_hunting_roles() {
    let params = WoaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    
    // Create multiple whales and check role diversity
    let mut roles = std::collections::HashSet::new();
    for i in 0..20 {
        let whale = Whale::new(i, 2, &bounds, &params, &mut rng);
        roles.insert(whale.hunting_role);
    }
    
    // Should have multiple different roles
    assert!(roles.len() > 1);
}

/// Test convergence detection
#[tokio::test]
async fn test_convergence_detection() {
    let params = WoaParameters {
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
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_simple_problem();
    
    woa.initialize(problem).await.unwrap();
    
    // Should eventually converge or run without error
    for _ in 0..50 {
        woa.step().await.unwrap();
        if woa.has_converged() {
            break;
        }
    }
    
    // Test passes if no errors occur
}

/// Test parallel evaluation
#[tokio::test]
async fn test_parallel_evaluation() {
    let params = WoaParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 400,
            parallel_evaluation: true,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-5.0, 5.0); 3]);
    
    woa.initialize(problem).await.unwrap();
    
    // Should work with parallel evaluation
    for _ in 0..10 {
        woa.step().await.unwrap();
    }
    
    let best_fitness = woa.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < Float::INFINITY);
}

/// Test algorithm metrics
#[tokio::test]
async fn test_algorithm_metrics() {
    let params = WoaParameters {
        common: CommonParameters {
            population_size: 15,
            max_evaluations: 300,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
    
    woa.initialize(problem).await.unwrap();
    
    // Run a few iterations and check metrics
    for _ in 0..5 {
        woa.step().await.unwrap();
    }
    
    let metrics = woa.metrics();
    assert!(metrics.total_iterations > 0);
    assert!(metrics.last_step_duration.as_nanos() > 0);
}

/// Test population diversity calculation
#[tokio::test]
async fn test_population_diversity() {
    let params = WoaParameters {
        common: CommonParameters {
            population_size: 10,
            max_evaluations: 200,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(2, vec![(-5.0, 5.0); 2]);
    
    woa.initialize(problem).await.unwrap();
    
    // Run a few iterations
    for _ in 0..5 {
        woa.step().await.unwrap();
    }
    
    // Diversity should be positive for a non-converged population
    let metrics = woa.metrics();
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

fn create_rastrigin_problem(dimensions: usize, bounds: Vec<(Float, Float)>) -> OptimizationProblem {
    OptimizationProblem::new(
        dimensions,
        bounds,
        Box::new(|x: &DVector<Float>| {
            let a = 10.0;
            let n = x.len() as Float;
            let sum: Float = x.iter().map(|xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos()).sum();
            Ok(a * n + sum)
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