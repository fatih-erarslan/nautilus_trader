//! Comprehensive tests for Sine Cosine Algorithm (SCA) implementation

use swarm_intelligence::algorithms::sca::{
    SineCosineAlgorithm, ScaParameters, ScaVariant, ScaAgent,
    OscillationStrategy, WavePattern, ParameterUpdateStrategy
};
use swarm_intelligence::core::{CommonParameters, OptimizationProblem, SwarmAlgorithm};
use nalgebra::DVector;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use approx::assert_relative_eq;

type Float = f64;

/// Test basic SCA creation and initialization
#[tokio::test]
async fn test_sca_creation_and_initialization() {
    let params = ScaParameters::default();
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    
    assert_eq!(sca.name(), "Sine Cosine Algorithm");
    
    // Test initialization with sphere function
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    sca.initialize(problem).await.unwrap();
}

/// Test different SCA variants
#[tokio::test]
async fn test_sca_variants() {
    let variants = vec![
        (ScaVariant::Standard, "Sine Cosine Algorithm"),
        (ScaVariant::Enhanced, "Enhanced Sine Cosine Algorithm"),
        (ScaVariant::Chaotic, "Chaotic Sine Cosine Algorithm"),
        (ScaVariant::Quantum, "Quantum Sine Cosine Algorithm"),
        (ScaVariant::LevyFlight, "Levy Flight Sine Cosine Algorithm"),
        (ScaVariant::Binary, "Binary Sine Cosine Algorithm"),
    ];
    
    for (variant, expected_name) in variants {
        let params = ScaParameters {
            variant,
            ..Default::default()
        };
        
        let sca = SineCosineAlgorithm::new(params).unwrap();
        assert_eq!(sca.name(), expected_name);
    }
}

/// Test SCA on sphere function
#[tokio::test]
async fn test_sca_sphere_function() {
    let params = ScaParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 1000,
            tolerance: 1e-3,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(5, vec![(-5.0, 5.0); 5]);
    
    sca.initialize(problem).await.unwrap();
    
    // Run optimization
    for _ in 0..50 {
        sca.step().await.unwrap();
        if sca.has_converged() {
            break;
        }
    }
    
    // Should find a reasonably good solution
    let best_fitness = sca.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < 10.0);
}

/// Test different oscillation strategies
#[tokio::test]
async fn test_oscillation_strategies() {
    let strategies = vec![
        OscillationStrategy::Standard,
        OscillationStrategy::AdaptiveFrequency,
        OscillationStrategy::MultiFrequency,
        OscillationStrategy::Harmonic,
        OscillationStrategy::Damped,
    ];
    
    for strategy in strategies {
        let params = ScaParameters {
            common: CommonParameters {
                population_size: 15,
                max_evaluations: 300,
                seed: Some(789),
                ..Default::default()
            },
            oscillation_strategy: strategy,
            ..Default::default()
        };
        
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        sca.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            sca.step().await.unwrap();
        }
    }
}

/// Test different wave patterns
#[tokio::test]
async fn test_wave_patterns() {
    let patterns = vec![
        WavePattern::Sine,
        WavePattern::Cosine,
        WavePattern::Mixed,
        WavePattern::Square,
        WavePattern::Triangular,
        WavePattern::Sawtooth,
    ];
    
    for pattern in patterns {
        let params = ScaParameters {
            common: CommonParameters {
                population_size: 12,
                max_evaluations: 240,
                seed: Some(321),
                ..Default::default()
            },
            wave_pattern: pattern,
            oscillation_strategy: OscillationStrategy::AdaptiveFrequency,
            ..Default::default()
        };
        
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        sca.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..5 {
            sca.step().await.unwrap();
        }
    }
}

/// Test different parameter update strategies
#[tokio::test]
async fn test_parameter_update_strategies() {
    let strategies = vec![
        ParameterUpdateStrategy::Linear,
        ParameterUpdateStrategy::Exponential,
        ParameterUpdateStrategy::Adaptive,
        ParameterUpdateStrategy::Piecewise,
        ParameterUpdateStrategy::Sinusoidal,
    ];
    
    for strategy in strategies {
        let params = ScaParameters {
            common: CommonParameters {
                population_size: 10,
                max_evaluations: 200,
                seed: Some(456),
                ..Default::default()
            },
            parameter_update_strategy: strategy,
            ..Default::default()
        };
        
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
        
        sca.initialize(problem).await.unwrap();
        
        // Should be able to run a few iterations without error
        for _ in 0..10 {
            sca.step().await.unwrap();
        }
    }
}

/// Test agent creation and properties
#[test]
fn test_agent_creation() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
    
    let agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(agent.id, 0);
    assert_eq!(agent.position.len(), 2);
    assert_eq!(agent.velocity.len(), 2);
    assert!(agent.position[0] >= -10.0 && agent.position[0] <= 10.0);
    assert!(agent.position[1] >= -5.0 && agent.position[1] <= 5.0);
    assert!(agent.oscillation_phase >= 0.0 && agent.oscillation_phase <= 2.0 * std::f64::consts::PI);
    assert!(agent.personal_frequency > 0.0);
    assert!(agent.energy_level > 0.0);
}

/// Test position update mechanisms
#[test]
fn test_agent_position_update() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 3];
    let mut agent = ScaAgent::new(0, 3, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![1.0, 2.0, -1.0]);
    let current_a = 1.5;
    let initial_position = agent.position.clone();
    
    agent.update_position(&global_best, current_a, &params, 5, &mut rng).unwrap();
    
    // Position should have been updated
    assert_ne!(agent.position, initial_position);
}

/// Test standard oscillation update
#[test]
fn test_standard_oscillation_update() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![2.0, -1.0]);
    let initial_position = agent.position.clone();
    
    // Generate random parameters
    let r1 = 1.5;
    let r2 = 0.8;
    let r3 = 1.2;
    let r4 = 0.3;
    let a = 2.0;
    
    agent.standard_oscillation_update(&global_best, a, r1, r2, r3, r4, &params).unwrap();
    
    // Position should have changed
    assert_ne!(agent.position, initial_position);
}

/// Test adaptive frequency update
#[test]
fn test_adaptive_frequency_update() {
    let params = ScaParameters {
        wave_pattern: WavePattern::Mixed,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![1.0, 1.0]);
    let initial_position = agent.position.clone();
    let initial_phase = agent.oscillation_phase;
    
    let r1 = 1.0;
    let r2 = 1.0;
    let r3 = 1.0;
    let r4 = 0.6;
    let a = 1.0;
    
    agent.adaptive_frequency_update(&global_best, a, r1, r2, r3, r4, &params, 10).unwrap();
    
    // Position and phase should have changed
    assert_ne!(agent.position, initial_position);
    assert_ne!(agent.oscillation_phase, initial_phase);
}

/// Test harmonic oscillation update
#[test]
fn test_harmonic_oscillation_update() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 3];
    let mut agent = ScaAgent::new(0, 3, &bounds, &params, &mut rng);
    
    let global_best = DVector::from_vec(vec![0.5, -0.5, 1.0]);
    let initial_position = agent.position.clone();
    
    let r1 = 0.8;
    let r2 = 1.2;
    let r3 = 0.7;
    let r4 = 0.4;
    let a = 1.5;
    
    agent.harmonic_oscillation_update(&global_best, a, r1, r2, r3, r4, &params, 5).unwrap();
    
    // Position should have changed due to harmonic oscillation
    assert_ne!(agent.position, initial_position);
}

/// Test parameter a updates
#[test]
fn test_parameter_a_updates() {
    let mut sca = SineCosineAlgorithm::new(ScaParameters::default()).unwrap();
    
    // Test linear update
    sca.params.parameter_update_strategy = ParameterUpdateStrategy::Linear;
    let initial_a = sca.current_a;
    sca.iteration = 25;
    sca.update_parameter_a(100);
    assert!(sca.current_a < initial_a);
    
    // Test exponential update
    sca.params.parameter_update_strategy = ParameterUpdateStrategy::Exponential;
    sca.current_a = sca.params.initial_a;
    sca.iteration = 50;
    sca.update_parameter_a(100);
    assert!(sca.current_a < sca.params.initial_a);
    
    // Test adaptive update
    sca.params.parameter_update_strategy = ParameterUpdateStrategy::Adaptive;
    sca.current_a = 1.0;
    sca.update_parameter_a(100);
    assert!(sca.current_a > 0.0);
}

/// Test Levy flight generation
#[test]
fn test_levy_flight_generation() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    let levy_step = agent.generate_levy_step(1.5, &mut rng);
    
    // Levy step should be finite
    assert!(levy_step.is_finite());
}

/// Test boundary handling
#[test]
fn test_boundary_handling() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-1.0, 1.0); 2];
    let mut agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    // Set agent outside bounds
    agent.position = DVector::from_vec(vec![2.0, -2.0]);
    
    agent.apply_boundaries(&bounds);
    
    // Position should be within bounds
    assert!(agent.position[0] >= -1.0 && agent.position[0] <= 1.0);
    assert!(agent.position[1] >= -1.0 && agent.position[1] <= 1.0);
}

/// Test position memory update
#[test]
fn test_position_memory_update() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    let mut agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
    
    assert_eq!(agent.position_memory.len(), 0);
    
    // Update memory several times
    for i in 0..15 {
        agent.position = DVector::from_vec(vec![i as Float, (i + 1) as Float]);
        agent.update_position_memory(&params);
    }
    
    // Memory should be limited to 10 positions
    assert_eq!(agent.position_memory.len(), 10);
}

/// Test wave pattern preferences
#[test]
fn test_wave_pattern_preferences() {
    let params = ScaParameters::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let bounds = vec![(-10.0, 10.0); 2];
    
    // Create multiple agents and check pattern diversity
    let mut patterns = std::collections::HashSet::new();
    for i in 0..30 {
        let agent = ScaAgent::new(i, 2, &bounds, &params, &mut rng);
        patterns.insert(agent.pattern_preference);
    }
    
    // Should have multiple different pattern preferences
    assert!(patterns.len() > 1);
}

/// Test convergence detection
#[tokio::test]
async fn test_convergence_detection() {
    let params = ScaParameters {
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
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_simple_problem();
    
    sca.initialize(problem).await.unwrap();
    
    // Should eventually converge or run without error
    for _ in 0..50 {
        sca.step().await.unwrap();
        if sca.has_converged() {
            break;
        }
    }
    
    // Test passes if no errors occur
}

/// Test parallel evaluation
#[tokio::test]
async fn test_parallel_evaluation() {
    let params = ScaParameters {
        common: CommonParameters {
            population_size: 20,
            max_evaluations: 400,
            parallel_evaluation: true,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-5.0, 5.0); 3]);
    
    sca.initialize(problem).await.unwrap();
    
    // Should work with parallel evaluation
    for _ in 0..10 {
        sca.step().await.unwrap();
    }
    
    let best_fitness = sca.get_best_individual().unwrap().fitness();
    assert!(*best_fitness < Float::INFINITY);
}

/// Test algorithm metrics
#[tokio::test]
async fn test_algorithm_metrics() {
    let params = ScaParameters {
        common: CommonParameters {
            population_size: 15,
            max_evaluations: 300,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(3, vec![(-2.0, 2.0); 3]);
    
    sca.initialize(problem).await.unwrap();
    
    // Run a few iterations and check metrics
    for _ in 0..5 {
        sca.step().await.unwrap();
    }
    
    let metrics = sca.metrics();
    assert!(metrics.total_iterations > 0);
    assert!(metrics.last_step_duration.as_nanos() > 0);
}

/// Test oscillation frequency adaptation
#[tokio::test]
async fn test_oscillation_frequency_adaptation() {
    let params = ScaParameters {
        common: CommonParameters {
            population_size: 10,
            max_evaluations: 200,
            seed: Some(42),
            ..Default::default()
        },
        variant: ScaVariant::Enhanced,
        oscillation_strategy: OscillationStrategy::AdaptiveFrequency,
        ..Default::default()
    };
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(2, vec![(-2.0, 2.0); 2]);
    
    sca.initialize(problem).await.unwrap();
    
    // Store initial frequencies
    let initial_frequencies: Vec<Float> = sca.agents.iter()
        .map(|agent| agent.personal_frequency)
        .collect();
    
    // Run several iterations
    for _ in 0..20 {
        sca.step().await.unwrap();
    }
    
    // At least some frequencies should have changed
    let final_frequencies: Vec<Float> = sca.agents.iter()
        .map(|agent| agent.personal_frequency)
        .collect();
    
    let changed_count = initial_frequencies.iter()
        .zip(final_frequencies.iter())
        .filter(|(initial, final)| (initial - final).abs() > 1e-10)
        .count();
    
    assert!(changed_count > 0);
}

/// Test population diversity calculation
#[tokio::test]
async fn test_population_diversity() {
    let params = ScaParameters {
        common: CommonParameters {
            population_size: 10,
            max_evaluations: 200,
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut sca = SineCosineAlgorithm::new(params).unwrap();
    let problem = create_sphere_problem(2, vec![(-5.0, 5.0); 2]);
    
    sca.initialize(problem).await.unwrap();
    
    // Run a few iterations
    for _ in 0..5 {
        sca.step().await.unwrap();
    }
    
    // Diversity should be positive for a non-converged population
    let metrics = sca.metrics();
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