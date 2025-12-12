//! Performance Benchmarks for Bio-Inspired Swarm Algorithms
//! 
//! This module provides comprehensive performance benchmarking for all bio-inspired
//! optimization algorithms across different problem types and market conditions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use dynamic_swarm_selector::*;
use market_regime_detector::MarketRegime;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark configuration
struct BenchmarkConfig {
    pub dimensions: usize,
    pub population_size: usize,
    pub max_iterations: u32,
    pub bounds: Vec<(f64, f64)>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dimensions: 10,
            population_size: 50,
            max_iterations: 1000,
            bounds: vec![(-10.0, 10.0); 10],
        }
    }
}

/// Benchmark objective function
struct BenchmarkObjective {
    pub function_type: BenchmarkFunction,
    pub dimensions: usize,
    pub bounds: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
enum BenchmarkFunction {
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
}

impl BenchmarkFunction {
    fn evaluate(&self, x: &[f64]) -> f64 {
        match self {
            BenchmarkFunction::Sphere => {
                x.iter().map(|xi| xi.powi(2)).sum()
            }
            BenchmarkFunction::Rosenbrock => {
                x.windows(2)
                    .map(|w| 100.0 * (w[1] - w[0].powi(2)).powi(2) + (1.0 - w[0]).powi(2))
                    .sum()
            }
            BenchmarkFunction::Rastrigin => {
                let a = 10.0;
                a * x.len() as f64 + x.iter()
                    .map(|xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<f64>()
            }
            BenchmarkFunction::Ackley => {
                let n = x.len() as f64;
                let sum1 = x.iter().map(|xi| xi.powi(2)).sum::<f64>() / n;
                let sum2 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>() / n;
                -20.0 * (-0.2 * sum1.sqrt()).exp() - sum2.exp() + 20.0 + std::f64::consts::E
            }
        }
    }
}

#[async_trait::async_trait]
impl ObjectiveFunction for BenchmarkObjective {
    async fn evaluate(&self, solution: &Solution) -> Result<f64, SwarmSelectionError> {
        let fitness = self.function_type.evaluate(&solution.parameters);
        Ok(fitness)
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }
    
    fn get_dimension(&self) -> usize {
        self.dimensions
    }
    
    fn is_maximization(&self) -> bool {
        false
    }
}

/// Benchmark individual algorithm performance
fn benchmark_algorithm_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let algorithms = vec![
        SwarmAlgorithm::ParticleSwarm,
        SwarmAlgorithm::GeneticAlgorithm,
        SwarmAlgorithm::DifferentialEvolution,
        SwarmAlgorithm::GreyWolf,
        SwarmAlgorithm::WhaleOptimization,
        SwarmAlgorithm::QuantumParticleSwarm,
        SwarmAlgorithm::AdaptiveHybrid,
    ];
    
    let functions = vec![
        BenchmarkFunction::Sphere,
        BenchmarkFunction::Rosenbrock,
        BenchmarkFunction::Rastrigin,
        BenchmarkFunction::Ackley,
    ];
    
    let mut group = c.benchmark_group("algorithm_performance");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    
    for algorithm in algorithms {
        for function in functions {
            let parameters = OptimizationParameters {
                population_size: config.population_size,
                max_iterations: config.max_iterations,
                tolerance: 1e-6,
                bounds: config.bounds.clone(),
                constraints: vec![],
                initialization_strategy: InitializationStrategy::LatinHypercube,
            };
            
            let objective = BenchmarkObjective {
                function_type: function,
                dimensions: config.dimensions,
                bounds: config.bounds.clone(),
            };
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", algorithm), format!("{:?}", function)),
                &(algorithm, objective, parameters),
                |b, (alg, obj, params)| {
                    b.iter(|| {
                        rt.block_on(async {
                            let mut optimizer = AlgorithmFactory::create_optimizer(*alg, params);
                            let result = optimizer.optimize(obj, params).await;
                            black_box(result)
                        })
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark scalability with different dimensions
fn benchmark_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let dimensions = vec![5, 10, 20, 50, 100];
    
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    
    for dim in dimensions {
        let bounds = vec![(-10.0, 10.0); dim];
        let parameters = OptimizationParameters {
            population_size: 50,
            max_iterations: 500,
            tolerance: 1e-6,
            bounds: bounds.clone(),
            constraints: vec![],
            initialization_strategy: InitializationStrategy::LatinHypercube,
        };
        
        let objective = BenchmarkObjective {
            function_type: BenchmarkFunction::Sphere,
            dimensions: dim,
            bounds: bounds.clone(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("ParticleSwarm", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut optimizer = AlgorithmFactory::create_optimizer(
                            SwarmAlgorithm::ParticleSwarm,
                            &parameters
                        );
                        let result = optimizer.optimize(&objective, &parameters).await;
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark convergence speed
fn benchmark_convergence_speed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let iterations = vec![100, 500, 1000, 2000];
    
    let mut group = c.benchmark_group("convergence_speed");
    group.sample_size(10);
    
    for max_iter in iterations {
        let bounds = vec![(-10.0, 10.0); 10];
        let parameters = OptimizationParameters {
            population_size: 50,
            max_iterations: max_iter,
            tolerance: 1e-6,
            bounds: bounds.clone(),
            constraints: vec![],
            initialization_strategy: InitializationStrategy::LatinHypercube,
        };
        
        let objective = BenchmarkObjective {
            function_type: BenchmarkFunction::Rosenbrock,
            dimensions: 10,
            bounds: bounds.clone(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("GreyWolf", max_iter),
            &max_iter,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut optimizer = AlgorithmFactory::create_optimizer(
                            SwarmAlgorithm::GreyWolf,
                            &parameters
                        );
                        let result = optimizer.optimize(&objective, &parameters).await;
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark population size effects
fn benchmark_population_size(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let population_sizes = vec![20, 50, 100, 200];
    
    let mut group = c.benchmark_group("population_size");
    group.sample_size(10);
    
    for pop_size in population_sizes {
        let bounds = vec![(-10.0, 10.0); 10];
        let parameters = OptimizationParameters {
            population_size: pop_size,
            max_iterations: 1000,
            tolerance: 1e-6,
            bounds: bounds.clone(),
            constraints: vec![],
            initialization_strategy: InitializationStrategy::LatinHypercube,
        };
        
        let objective = BenchmarkObjective {
            function_type: BenchmarkFunction::Ackley,
            dimensions: 10,
            bounds: bounds.clone(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("WhaleOptimization", pop_size),
            &pop_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut optimizer = AlgorithmFactory::create_optimizer(
                            SwarmAlgorithm::WhaleOptimization,
                            &parameters
                        );
                        let result = optimizer.optimize(&objective, &parameters).await;
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark market regime compatibility
fn benchmark_regime_compatibility(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let regimes = vec![
        MarketRegime::LowVolatility,
        MarketRegime::HighVolatility,
        MarketRegime::StrongUptrend,
        MarketRegime::QuantumCoherent,
    ];
    
    let mut group = c.benchmark_group("regime_compatibility");
    group.sample_size(10);
    
    for regime in regimes {
        let bounds = vec![(-10.0, 10.0); 10];
        let parameters = OptimizationParameters {
            population_size: 50,
            max_iterations: 500,
            tolerance: 1e-6,
            bounds: bounds.clone(),
            constraints: vec![],
            initialization_strategy: InitializationStrategy::LatinHypercube,
        };
        
        let objective = BenchmarkObjective {
            function_type: BenchmarkFunction::Rastrigin,
            dimensions: 10,
            bounds: bounds.clone(),
        };
        
        // Find compatible algorithm
        let algorithm = if SwarmAlgorithm::ParticleSwarm.is_regime_compatible(&regime) {
            SwarmAlgorithm::ParticleSwarm
        } else if SwarmAlgorithm::GreyWolf.is_regime_compatible(&regime) {
            SwarmAlgorithm::GreyWolf
        } else {
            SwarmAlgorithm::AdaptiveHybrid
        };
        
        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", algorithm), format!("{:?}", regime)),
            &regime,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut optimizer = AlgorithmFactory::create_optimizer(algorithm, &parameters);
                        let result = optimizer.optimize(&objective, &parameters).await;
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark quantum enhancement effects
fn benchmark_quantum_enhancement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quantum_enhancement");
    group.sample_size(10);
    
    let bounds = vec![(-10.0, 10.0); 10];
    let parameters = OptimizationParameters {
        population_size: 50,
        max_iterations: 1000,
        tolerance: 1e-6,
        bounds: bounds.clone(),
        constraints: vec![],
        initialization_strategy: InitializationStrategy::LatinHypercube,
    };
    
    let objective = BenchmarkObjective {
        function_type: BenchmarkFunction::Sphere,
        dimensions: 10,
        bounds: bounds.clone(),
    };
    
    // Classical PSO
    group.bench_function("ClassicalPSO", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = AlgorithmFactory::create_optimizer(
                    SwarmAlgorithm::ParticleSwarm,
                    &parameters
                );
                let result = optimizer.optimize(&objective, &parameters).await;
                black_box(result)
            })
        })
    });
    
    // Quantum PSO
    group.bench_function("QuantumPSO", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = AlgorithmFactory::create_optimizer(
                    SwarmAlgorithm::QuantumParticleSwarm,
                    &parameters
                );
                let result = optimizer.optimize(&objective, &parameters).await;
                black_box(result)
            })
        })
    });
    
    group.finish();
}

/// Benchmark dynamic swarm selector performance
fn benchmark_dynamic_selector(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("dynamic_selector");
    group.sample_size(10);
    
    let bounds = vec![(-10.0, 10.0); 10];
    let parameters = OptimizationParameters {
        population_size: 50,
        max_iterations: 500,
        tolerance: 1e-6,
        bounds: bounds.clone(),
        constraints: vec![],
        initialization_strategy: InitializationStrategy::LatinHypercube,
    };
    
    let objective = BenchmarkObjective {
        function_type: BenchmarkFunction::Rosenbrock,
        dimensions: 10,
        bounds: bounds.clone(),
    };
    
    // Single algorithm
    group.bench_function("SingleAlgorithm", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = AlgorithmFactory::create_optimizer(
                    SwarmAlgorithm::ParticleSwarm,
                    &parameters
                );
                let result = optimizer.optimize(&objective, &parameters).await;
                black_box(result)
            })
        })
    });
    
    // Adaptive hybrid
    group.bench_function("AdaptiveHybrid", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = AlgorithmFactory::create_optimizer(
                    SwarmAlgorithm::AdaptiveHybrid,
                    &parameters
                );
                let result = optimizer.optimize(&objective, &parameters).await;
                black_box(result)
            })
        })
    });
    
    group.finish();
}

/// Benchmark memory usage and efficiency
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10);
    
    let bounds = vec![(-10.0, 10.0); 10];
    let parameters = OptimizationParameters {
        population_size: 100,
        max_iterations: 1000,
        tolerance: 1e-6,
        bounds: bounds.clone(),
        constraints: vec![],
        initialization_strategy: InitializationStrategy::LatinHypercube,
    };
    
    let objective = BenchmarkObjective {
        function_type: BenchmarkFunction::Ackley,
        dimensions: 10,
        bounds: bounds.clone(),
    };
    
    let algorithms = vec![
        SwarmAlgorithm::ParticleSwarm,
        SwarmAlgorithm::GeneticAlgorithm,
        SwarmAlgorithm::AntColony,
        SwarmAlgorithm::GreyWolf,
    ];
    
    for algorithm in algorithms {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", format!("{:?}", algorithm)),
            &algorithm,
            |b, alg| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut optimizer = AlgorithmFactory::create_optimizer(*alg, &parameters);
                        let result = optimizer.optimize(&objective, &parameters).await;
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential execution
fn benchmark_parallel_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("parallel_execution");
    group.sample_size(10);
    
    let bounds = vec![(-10.0, 10.0); 10];
    let parameters = OptimizationParameters {
        population_size: 50,
        max_iterations: 500,
        tolerance: 1e-6,
        bounds: bounds.clone(),
        constraints: vec![],
        initialization_strategy: InitializationStrategy::LatinHypercube,
    };
    
    let objective = BenchmarkObjective {
        function_type: BenchmarkFunction::Griewank,
        dimensions: 10,
        bounds: bounds.clone(),
    };
    
    // Sequential execution (simulated)
    group.bench_function("Sequential", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = AlgorithmFactory::create_optimizer(
                    SwarmAlgorithm::ParticleSwarm,
                    &parameters
                );
                let result = optimizer.optimize(&objective, &parameters).await;
                black_box(result)
            })
        })
    });
    
    // Parallel execution with multiple algorithms
    group.bench_function("Parallel", |b| {
        b.iter(|| {
            rt.block_on(async {
                let algorithms = vec![
                    SwarmAlgorithm::ParticleSwarm,
                    SwarmAlgorithm::GeneticAlgorithm,
                    SwarmAlgorithm::GreyWolf,
                ];
                
                let mut futures = Vec::new();
                for algorithm in algorithms {
                    let mut optimizer = AlgorithmFactory::create_optimizer(algorithm, &parameters);
                    let future = optimizer.optimize(&objective, &parameters);
                    futures.push(future);
                }
                
                // Wait for all to complete
                for future in futures {
                    let result = future.await;
                    black_box(result);
                }
            })
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_algorithm_performance,
    benchmark_scalability,
    benchmark_convergence_speed,
    benchmark_population_size,
    benchmark_regime_compatibility,
    benchmark_quantum_enhancement,
    benchmark_dynamic_selector,
    benchmark_memory_efficiency,
    benchmark_parallel_execution
);

criterion_main!(benches);