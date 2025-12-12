//! Complete CDFA Framework Demonstration
//!
//! This example demonstrates the full capabilities of the Combinatorial Diversity
//! Fusion Analysis framework, including algorithm fusion, performance tracking,
//! adaptive parameter tuning, and algorithm enhancement.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use tokio;

use swarm_intelligence::{
    // Core types
    OptimizationProblem, ParticleSwarmOptimization, GreyWolfOptimizer,
    DifferentialEvolution, ArtificialBeeColony,
    
    // CDFA components
    CombinatorialDiversityFusionAnalyzer, FusionStrategy, DiversityMetrics,
    PerformanceTracker, AdaptiveParameterTuning, EnhancementFramework,
    
    // Configuration and types
    DiversityConfig, DiversityType, AdaptiveTuningConfig, EnhancementConfig,
    AlgorithmInfo, AlgorithmType, AlgorithmCharacteristics, EnhancementType,
    ConvergenceSpeed, ExplorationCapability, ExploitationCapability,
    ScalabilityProfile, ScalingBehavior, ProblemType, ComplexityClass,
    ParallelizationPotential, ParallelismLevel, OverheadLevel,
    
    // Parameter tuning
    ParameterSpace, ParameterDefinition, ParameterType, ParameterRange,
    SensitivityLevel, UpdateFrequency, GridSearchStrategy, RandomSearchStrategy,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🚀 CDFA Framework Complete Demonstration");
    println!("=========================================\n");
    
    // 1. Set up the optimization problem
    let problem = create_test_problem()?;
    println!("✅ Created optimization problem: 10D Rastrigin function\n");
    
    // 2. Initialize CDFA components
    let (cdfa_analyzer, performance_tracker, diversity_metrics, adaptive_tuner, enhancement_framework) = 
        initialize_cdfa_framework().await?;
    println!("✅ Initialized CDFA framework components\n");
    
    // 3. Register algorithms
    let algorithm_ids = register_algorithms(&enhancement_framework).await?;
    println!("✅ Registered {} algorithms\n", algorithm_ids.len());
    
    // 4. Demonstrate diversity analysis
    demonstrate_diversity_analysis(&diversity_metrics).await?;
    
    // 5. Demonstrate performance tracking
    demonstrate_performance_tracking(&performance_tracker).await?;
    
    // 6. Demonstrate parameter tuning
    demonstrate_adaptive_tuning(&adaptive_tuner).await?;
    
    // 7. Demonstrate algorithm fusion
    demonstrate_algorithm_fusion(&cdfa_analyzer, &problem).await?;
    
    // 8. Demonstrate algorithm enhancement
    demonstrate_algorithm_enhancement(&enhancement_framework, &problem).await?;
    
    // 9. Run comprehensive benchmark
    run_comprehensive_benchmark(&cdfa_analyzer, &performance_tracker, &problem).await?;
    
    println!("\n🎉 CDFA Framework demonstration completed successfully!");
    println!("   All components working together for optimal performance.");
    
    Ok(())
}

/// Create a test optimization problem (10D Rastrigin function)
fn create_test_problem() -> Result<OptimizationProblem> {
    let problem = OptimizationProblem::new()
        .dimensions(10)
        .bounds(-5.12, 5.12)
        .objective(|x| {
            // Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
            let a = 10.0;
            let n = x.len() as f64;
            a * n + x.iter().map(|&xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
        })
        .build()?;
    
    Ok(problem)
}

/// Initialize all CDFA framework components
async fn initialize_cdfa_framework() -> Result<(
    CombinatorialDiversityFusionAnalyzer,
    PerformanceTracker,
    DiversityMetrics,
    AdaptiveParameterTuning,
    EnhancementFramework,
)> {
    // Initialize swarm intelligence framework
    swarm_intelligence::initialize(Some(8), true, true).await?;
    
    // Create CDFA analyzer with optimal settings
    let cdfa_analyzer = CombinatorialDiversityFusionAnalyzer::new();
    
    // Create performance tracker
    let performance_tracker = PerformanceTracker::new();
    
    // Create diversity metrics calculator
    let diversity_metrics = DiversityMetrics::new();
    
    // Create adaptive parameter tuner
    let tuning_config = AdaptiveTuningConfig {
        max_iterations: 30,
        convergence_threshold: 1e-6,
        exploration_factor: 0.1,
        learning_rate: 0.01,
        parallel_tuning: true,
        ..Default::default()
    };
    let adaptive_tuner = AdaptiveParameterTuning::with_config(tuning_config);
    
    // Create enhancement framework
    let enhancement_config = EnhancementConfig {
        max_enhancement_iterations: 20,
        improvement_threshold: 0.05,
        parallel_enhancement: true,
        enable_meta_learning: true,
        ..Default::default()
    };
    let enhancement_framework = EnhancementFramework::with_config(enhancement_config);
    
    Ok((cdfa_analyzer, performance_tracker, diversity_metrics, adaptive_tuner, enhancement_framework))
}

/// Register test algorithms in the enhancement framework
async fn register_algorithms(framework: &EnhancementFramework) -> Result<Vec<String>> {
    let mut algorithm_ids = Vec::new();
    
    // PSO Algorithm
    let pso_info = AlgorithmInfo {
        id: "pso_standard".to_string(),
        name: "Particle Swarm Optimization".to_string(),
        algorithm_type: AlgorithmType::SwarmIntelligence,
        characteristics: AlgorithmCharacteristics {
            convergence_speed: ConvergenceSpeed::Fast,
            exploration_capability: ExplorationCapability::Good,
            exploitation_capability: ExploitationCapability::Average,
            scalability: ScalabilityProfile {
                dimension_scaling: ScalingBehavior::Linear,
                population_scaling: ScalingBehavior::Linear,
                iteration_scaling: ScalingBehavior::Linear,
                parallel_scaling: ScalingBehavior::Linear,
            },
            problem_suitability: vec![ProblemType::Continuous, ProblemType::Multimodal],
            time_complexity: ComplexityClass::Linear,
            space_complexity: ComplexityClass::Linear,
            parallelization: ParallelizationPotential {
                inherent_parallelism: ParallelismLevel::High,
                data_parallelism: ParallelismLevel::Excellent,
                task_parallelism: ParallelismLevel::Medium,
                communication_overhead: OverheadLevel::Low,
            },
        },
        compatible_enhancements: vec![
            EnhancementType::ParameterTuning,
            EnhancementType::ParallelizationEnhancement,
            EnhancementType::DiversityMaintenance,
        ],
        enhancement_count: 0,
        best_parameters: None,
        baseline_performance: Some(0.75),
    };
    
    framework.register_algorithm(pso_info)?;
    algorithm_ids.push("pso_standard".to_string());
    
    // Grey Wolf Optimizer
    let gwo_info = AlgorithmInfo {
        id: "gwo_standard".to_string(),
        name: "Grey Wolf Optimizer".to_string(),
        algorithm_type: AlgorithmType::NatureInspired,
        characteristics: AlgorithmCharacteristics {
            convergence_speed: ConvergenceSpeed::Medium,
            exploration_capability: ExplorationCapability::Excellent,
            exploitation_capability: ExploitationCapability::Good,
            scalability: ScalabilityProfile {
                dimension_scaling: ScalingBehavior::Linear,
                population_scaling: ScalingBehavior::Linear,
                iteration_scaling: ScalingBehavior::Linear,
                parallel_scaling: ScalingBehavior::Linear,
            },
            problem_suitability: vec![ProblemType::Continuous, ProblemType::Multimodal],
            time_complexity: ComplexityClass::Quadratic,
            space_complexity: ComplexityClass::Linear,
            parallelization: ParallelizationPotential {
                inherent_parallelism: ParallelismLevel::Medium,
                data_parallelism: ParallelismLevel::High,
                task_parallelism: ParallelismLevel::Low,
                communication_overhead: OverheadLevel::Medium,
            },
        },
        compatible_enhancements: vec![
            EnhancementType::ParameterTuning,
            EnhancementType::ConvergenceAcceleration,
            EnhancementType::Hybridization,
        ],
        enhancement_count: 0,
        best_parameters: None,
        baseline_performance: Some(0.68),
    };
    
    framework.register_algorithm(gwo_info)?;
    algorithm_ids.push("gwo_standard".to_string());
    
    // Differential Evolution
    let de_info = AlgorithmInfo {
        id: "de_standard".to_string(),
        name: "Differential Evolution".to_string(),
        algorithm_type: AlgorithmType::EvolutionaryComputation,
        characteristics: AlgorithmCharacteristics {
            convergence_speed: ConvergenceSpeed::Slow,
            exploration_capability: ExplorationCapability::Average,
            exploitation_capability: ExploitationCapability::Excellent,
            scalability: ScalabilityProfile {
                dimension_scaling: ScalingBehavior::Linear,
                population_scaling: ScalingBehavior::Linear,
                iteration_scaling: ScalingBehavior::Linear,
                parallel_scaling: ScalingBehavior::Linear,
            },
            problem_suitability: vec![ProblemType::Continuous, ProblemType::Unimodal],
            time_complexity: ComplexityClass::Linear,
            space_complexity: ComplexityClass::Linear,
            parallelization: ParallelizationPotential {
                inherent_parallelism: ParallelismLevel::High,
                data_parallelism: ParallelismLevel::High,
                task_parallelism: ParallelismLevel::High,
                communication_overhead: OverheadLevel::Low,
            },
        },
        compatible_enhancements: vec![
            EnhancementType::ParameterTuning,
            EnhancementType::ConvergenceAcceleration,
            EnhancementType::AdaptiveStrategies,
        ],
        enhancement_count: 0,
        best_parameters: None,
        baseline_performance: Some(0.82),
    };
    
    framework.register_algorithm(de_info)?;
    algorithm_ids.push("de_standard".to_string());
    
    // Artificial Bee Colony
    let abc_info = AlgorithmInfo {
        id: "abc_standard".to_string(),
        name: "Artificial Bee Colony".to_string(),
        algorithm_type: AlgorithmType::SwarmIntelligence,
        characteristics: AlgorithmCharacteristics {
            convergence_speed: ConvergenceSpeed::Medium,
            exploration_capability: ExplorationCapability::Good,
            exploitation_capability: ExploitationCapability::Good,
            scalability: ScalabilityProfile {
                dimension_scaling: ScalingBehavior::Linear,
                population_scaling: ScalingBehavior::Linear,
                iteration_scaling: ScalingBehavior::Linear,
                parallel_scaling: ScalingBehavior::Linear,
            },
            problem_suitability: vec![ProblemType::Continuous, ProblemType::Multimodal],
            time_complexity: ComplexityClass::Linear,
            space_complexity: ComplexityClass::Linear,
            parallelization: ParallelizationPotential {
                inherent_parallelism: ParallelismLevel::Medium,
                data_parallelism: ParallelismLevel::Medium,
                task_parallelism: ParallelismLevel::Medium,
                communication_overhead: OverheadLevel::Medium,
            },
        },
        compatible_enhancements: vec![
            EnhancementType::ParameterTuning,
            EnhancementType::DiversityMaintenance,
            EnhancementType::LocalSearchIntegration,
        ],
        enhancement_count: 0,
        best_parameters: None,
        baseline_performance: Some(0.70),
    };
    
    framework.register_algorithm(abc_info)?;
    algorithm_ids.push("abc_standard".to_string());
    
    println!("📋 Registered algorithms:");
    for id in &algorithm_ids {
        println!("   • {}", id);
    }
    
    Ok(algorithm_ids)
}

/// Demonstrate diversity analysis capabilities
async fn demonstrate_diversity_analysis(diversity_metrics: &DiversityMetrics) -> Result<()> {
    println!("🔍 Diversity Analysis Demonstration");
    println!("-----------------------------------");
    
    use swarm_intelligence::{Population, BasicIndividual, Position};
    
    // Create test populations with different diversity characteristics
    let mut low_diversity_pop = Population::new();
    let mut high_diversity_pop = Population::new();
    
    // Low diversity population (clustered)
    for i in 0..20 {
        let pos = Position::from_vec(vec![
            0.1 + (i as f64) * 0.01,
            0.1 + (i as f64) * 0.01,
        ]);
        low_diversity_pop.add(BasicIndividual::new(pos));
    }
    
    // High diversity population (spread out)
    for i in 0..20 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 20.0;
        let radius = 1.0 + (i as f64) * 0.1;
        let pos = Position::from_vec(vec![
            radius * angle.cos(),
            radius * angle.sin(),
        ]);
        high_diversity_pop.add(BasicIndividual::new(pos));
    }
    
    // Calculate diversity measures
    let config = DiversityConfig {
        measures: vec![DiversityType::Combined],
        use_parallel: true,
        enable_cache: true,
        ..Default::default()
    };
    
    let low_div_result = diversity_metrics.calculate_diversity(&low_diversity_pop, &config)?;
    let high_div_result = diversity_metrics.calculate_diversity(&high_diversity_pop, &config)?;
    
    println!("📊 Diversity Analysis Results:");
    
    match (&low_div_result, &high_div_result) {
        (swarm_intelligence::DiversityMeasure::Combined { composite_score: low_score, .. },
         swarm_intelligence::DiversityMeasure::Combined { composite_score: high_score, .. }) => {
            println!("   • Low diversity population score:  {:.4}", low_score);
            println!("   • High diversity population score: {:.4}", high_score);
            println!("   • Diversity ratio: {:.2}x", high_score / low_score.max(1e-10));
        }
        _ => println!("   • Diversity measures calculated successfully"),
    }
    
    println!("✅ Diversity analysis completed\n");
    Ok(())
}

/// Demonstrate performance tracking capabilities
async fn demonstrate_performance_tracking(tracker: &PerformanceTracker) -> Result<()> {
    println!("📈 Performance Tracking Demonstration");
    println!("------------------------------------");
    
    use swarm_intelligence::{Population, BasicIndividual, AlgorithmMetrics};
    
    // Start monitoring
    tracker.start_monitoring(
        "demo_algorithm".to_string(),
        vec![swarm_intelligence::MetricType::All],
    )?;
    
    // Simulate algorithm execution with metrics
    let mut population = Population::new();
    for i in 0..50 {
        let pos = swarm_intelligence::Position::from_vec(vec![
            (i as f64) * 0.1,
            (i as f64) * 0.05,
        ]);
        population.add(BasicIndividual::new(pos));
    }
    
    // Record metrics for several iterations
    for iteration in 1..=10 {
        let metrics = AlgorithmMetrics {
            iteration,
            best_fitness: Some(1.0 / (iteration as f64)), // Improving fitness
            average_fitness: Some(2.0 / (iteration as f64)),
            diversity: Some(0.8 - (iteration as f64) * 0.05), // Decreasing diversity
            convergence_rate: Some(0.1),
            evaluations: iteration * 50,
            time_per_iteration: Some(10 + iteration * 2),
            memory_usage: Some(1024 * iteration),
        };
        
        tracker.record_metrics(
            "demo_algorithm".to_string(),
            metrics,
            &population,
            Duration::from_millis(10 + iteration * 2),
        )?;
        
        // Small delay to simulate real execution
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Get performance history
    let history = tracker.get_performance_history("demo_algorithm");
    if let Some(history) = history {
        println!("📊 Performance Tracking Results:");
        println!("   • Total measurements: {}", history.metrics_history.len());
        println!("   • Algorithm stability: {:.2}", history.summary.stability_score);
        println!("   • Trend direction: {:?}", history.trends.trend_direction);
    }
    
    // Run a benchmark
    let benchmark_config = swarm_intelligence::BenchmarkConfig {
        dimensions: 5,
        population_size: 30,
        max_iterations: 50,
        runs: 3,
        problem_type: "rastrigin".to_string(),
        parameters: HashMap::new(),
    };
    
    let benchmark_result = tracker.run_benchmark("demo_algorithm".to_string(), benchmark_config)?;
    
    println!("🏆 Benchmark Results:");
    println!("   • Success rate: {:.1}%", benchmark_result.results.success_rate * 100.0);
    println!("   • Reliability: {:.3}", benchmark_result.results.reliability);
    println!("   • Avg evaluations/sec: {:.0}", benchmark_result.results.avg_performance.efficiency.evaluations_per_second);
    
    // Stop monitoring
    tracker.stop_monitoring("demo_algorithm")?;
    
    println!("✅ Performance tracking completed\n");
    Ok(())
}

/// Demonstrate adaptive parameter tuning
async fn demonstrate_adaptive_tuning(tuner: &mut AdaptiveParameterTuning) -> Result<()> {
    println!("🎛️  Adaptive Parameter Tuning Demonstration");
    println!("-------------------------------------------");
    
    // Set up parameter space
    let mut parameter_space = ParameterSpace::default();
    
    // Add learning rate parameter
    let learning_rate_param = ParameterDefinition {
        name: "learning_rate".to_string(),
        param_type: ParameterType::Continuous,
        range: ParameterRange::Continuous { min: 0.001, max: 0.1 },
        description: "Learning rate for the algorithm".to_string(),
        sensitivity: SensitivityLevel::High,
        update_frequency: UpdateFrequency::Adaptive,
    };
    parameter_space.parameters.insert("learning_rate".to_string(), learning_rate_param);
    
    // Add population size parameter
    let pop_size_param = ParameterDefinition {
        name: "population_size".to_string(),
        param_type: ParameterType::Discrete,
        range: ParameterRange::Discrete { min: 20, max: 100, step: 10 },
        description: "Population size for the algorithm".to_string(),
        sensitivity: SensitivityLevel::Medium,
        update_frequency: UpdateFrequency::Periodic { every_n_iterations: 10 },
    };
    parameter_space.parameters.insert("population_size".to_string(), pop_size_param);
    
    tuner.set_parameter_space(parameter_space);
    
    // Add tuning strategies
    tuner.add_strategy(Box::new(GridSearchStrategy::new(5)));
    tuner.add_strategy(Box::new(RandomSearchStrategy::new(10)));
    
    // Define a simple evaluation function (minimize Rosenbrock function)
    let evaluation_fn = |params: &swarm_intelligence::ParameterSet| -> Result<f64, swarm_intelligence::SwarmError> {
        let lr = params.values.get("learning_rate").copied().unwrap_or(0.01);
        let pop_size = params.values.get("population_size").copied().unwrap_or(50.0);
        
        // Simulated performance function (higher is better)
        let performance = -((lr - 0.02).powi(2) + (pop_size - 40.0).powi(2) / 1000.0);
        Ok(performance)
    };
    
    // Run parameter tuning
    let result = tuner.tune_parameters("demo_tuning".to_string(), evaluation_fn).await?;
    
    println!("🎯 Parameter Tuning Results:");
    println!("   • Best learning rate: {:.4}", result.values.get("learning_rate").unwrap_or(&0.02));
    println!("   • Best population size: {:.0}", result.values.get("population_size").unwrap_or(&40.0));
    println!("   • Parameter source: {:?}", result.source);
    println!("   • Confidence: {:.3}", result.confidence);
    
    // Get tuning statistics
    let stats = tuner.get_tuning_statistics("demo_tuning");
    println!("📈 Tuning Statistics:");
    println!("   • Total evaluations: {}", stats.total_evaluations);
    println!("   • Success rate: {:.1}%", stats.success_rate * 100.0);
    println!("   • Best performance: {:.6}", stats.best_performance);
    
    println!("✅ Adaptive parameter tuning completed\n");
    Ok(())
}

/// Demonstrate algorithm fusion capabilities
async fn demonstrate_algorithm_fusion(
    analyzer: &CombinatorialDiversityFusionAnalyzer,
    problem: &OptimizationProblem,
) -> Result<()> {
    println!("🔀 Algorithm Fusion Demonstration");
    println!("--------------------------------");
    
    // Add algorithms to the analyzer
    let pso = ParticleSwarmOptimization::new();
    let gwo = GreyWolfOptimizer::new();
    let de = DifferentialEvolution::new();
    
    analyzer.add_algorithm(pso, "PSO".to_string())?;
    analyzer.add_algorithm(gwo, "GWO".to_string())?;
    analyzer.add_algorithm(de, "DE".to_string())?;
    
    // Generate algorithm combinations
    let combinations = analyzer.generate_combinations(2)?;
    println!("🧮 Generated {} 2-algorithm combinations:", combinations.len());
    for (i, combo) in combinations.iter().enumerate() {
        println!("   {}. {:?}", i + 1, combo);
    }
    
    // Test different fusion strategies
    let strategies = vec![
        ("Parallel Fusion", FusionStrategy::Parallel { weights: vec![0.4, 0.6] }),
        ("Sequential Fusion", FusionStrategy::Sequential),
        ("Adaptive Fusion", FusionStrategy::Adaptive { switch_threshold: 0.1 }),
    ];
    
    for (strategy_name, strategy) in strategies {
        println!("\n🚀 Testing {}", strategy_name);
        
        let algorithms = vec!["PSO".to_string(), "GWO".to_string()];
        let fusion_result = analyzer.analyze_fusion(
            algorithms,
            problem.clone(),
            strategy,
            50, // iterations
        ).await?;
        
        println!("   • Best fitness: {:.6}", fusion_result.best_solution.best_fitness);
        println!("   • Synergy score: {:.4}", fusion_result.synergy_score);
        println!("   • Algorithm count: {}", fusion_result.combination_metrics.algorithm_count);
        println!("   • Diversity score: {:.4}", fusion_result.combination_metrics.diversity_score);
        println!("   • Improvement factor: {:.4}", fusion_result.combination_metrics.improvement_factor);
        println!("   • Total time: {:?}", fusion_result.timing.total_time);
    }
    
    println!("✅ Algorithm fusion demonstration completed\n");
    Ok(())
}

/// Demonstrate algorithm enhancement capabilities
async fn demonstrate_algorithm_enhancement(
    framework: &EnhancementFramework,
    problem: &OptimizationProblem,
) -> Result<()> {
    println!("⚡ Algorithm Enhancement Demonstration");
    println!("-------------------------------------");
    
    // Get enhancement recommendations for each algorithm
    let algorithms = vec!["pso_standard", "gwo_standard", "de_standard", "abc_standard"];
    
    for algorithm_id in &algorithms {
        let recommendations = framework.get_enhancement_recommendations(algorithm_id)?;
        
        println!("🎯 Enhancement recommendations for {}:", algorithm_id);
        for (i, rec) in recommendations.iter().enumerate() {
            println!("   {}. {:?} - Priority: {:?}", i + 1, rec.enhancement_type, rec.priority);
            println!("      Expected improvement: {:.1}%", rec.expected_improvement * 100.0);
            println!("      Rationale: {}", rec.rationale);
        }
        println!();
    }
    
    // Create hybrid algorithm blueprint
    let component_algorithms = vec!["pso_standard".to_string(), "de_standard".to_string()];
    let hybrid_blueprint = framework.create_hybrid_algorithm(
        component_algorithms,
        "ParallelExecution".to_string(),
        Some(vec![0.6, 0.4]),
    )?;
    
    println!("🔗 Hybrid Algorithm Blueprint:");
    println!("   • Components: {:?}", hybrid_blueprint.components);
    println!("   • Strategy: {}", hybrid_blueprint.strategy);
    println!("   • Weights: {:?}", hybrid_blueprint.weights);
    println!("   • Estimated performance: {:.4}", hybrid_blueprint.estimated_performance);
    println!("   • Interactions: {} rules", hybrid_blueprint.interactions.len());
    
    println!("✅ Algorithm enhancement demonstration completed\n");
    Ok(())
}

/// Run comprehensive benchmark comparing all approaches
async fn run_comprehensive_benchmark(
    analyzer: &CombinatorialDiversityFusionAnalyzer,
    tracker: &PerformanceTracker,
    problem: &OptimizationProblem,
) -> Result<()> {
    println!("🏁 Comprehensive Benchmark");
    println!("=========================");
    
    // Define test scenarios
    let scenarios = vec![
        ("Single Algorithm (PSO)", vec!["PSO".to_string()]),
        ("Single Algorithm (GWO)", vec!["GWO".to_string()]),
        ("Parallel Fusion (PSO+GWO)", vec!["PSO".to_string(), "GWO".to_string()]),
        ("Sequential Fusion (PSO+DE)", vec!["PSO".to_string(), "DE".to_string()]),
        ("Triple Fusion (PSO+GWO+DE)", vec!["PSO".to_string(), "GWO".to_string(), "DE".to_string()]),
    ];
    
    let mut results = Vec::new();
    
    for (scenario_name, algorithms) in scenarios {
        println!("\n🧪 Running scenario: {}", scenario_name);
        
        let start_time = std::time::Instant::now();
        
        let strategy = if algorithms.len() == 1 {
            FusionStrategy::Sequential
        } else {
            FusionStrategy::Parallel { 
                weights: vec![1.0 / algorithms.len() as f64; algorithms.len()] 
            }
        };
        
        let fusion_result = analyzer.analyze_fusion(
            algorithms.clone(),
            problem.clone(),
            strategy,
            100, // More iterations for benchmark
        ).await?;
        
        let elapsed = start_time.elapsed();
        
        results.push((
            scenario_name,
            fusion_result.best_solution.best_fitness,
            fusion_result.synergy_score,
            elapsed,
        ));
        
        println!("   ✓ Best fitness: {:.6}", fusion_result.best_solution.best_fitness);
        println!("   ✓ Synergy score: {:.4}", fusion_result.synergy_score);
        println!("   ✓ Execution time: {:?}", elapsed);
    }
    
    // Print benchmark summary
    println!("\n📊 Benchmark Summary");
    println!("━━━━━━━━━━━━━━━━━━━━");
    println!("{:<30} {:>12} {:>12} {:>12}", "Scenario", "Best Fitness", "Synergy", "Time (ms)");
    println!("{}", "─".repeat(70));
    
    for (name, fitness, synergy, time) in &results {
        println!("{:<30} {:>12.6} {:>12.4} {:>12}", 
                name, fitness, synergy, time.as_millis());
    }
    
    // Find best performing scenario
    if let Some((best_scenario, best_fitness, _, _)) = results.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!("\n🏆 Best performing scenario: {}", best_scenario);
        println!("   Best fitness achieved: {:.6}", best_fitness);
    }
    
    println!("\n✅ Comprehensive benchmark completed\n");
    Ok(())
}