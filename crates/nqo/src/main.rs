//! NQO Command-line Interface
//! 
//! Demonstrates the Neuromorphic Quantum Optimizer with various optimization problems.

use nqo::{NeuromorphicQuantumOptimizer, NqoConfig, OptimizationProblem};
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let problem_type = args.get(1).map(String::as_str).unwrap_or("sphere");
    
    println!("🧠 Neuromorphic Quantum Optimizer v{}", env!("CARGO_PKG_VERSION"));
    println!("Problem: {}", problem_type);
    println!();
    
    // Create configuration
    let config = NqoConfig {
        neurons: 64,
        qubits: 4,
        adaptivity: 0.7,
        learning_rate: 0.01,
        epochs: 20,
        use_gpu: false, // Set to false for CPU-only demonstration
        use_simd: true,
        ..Default::default()
    };
    
    // Initialize optimizer
    println!("Initializing NQO...");
    let optimizer = NeuromorphicQuantumOptimizer::new(config).await?;
    
    // Define optimization problems
    let problem = match problem_type {
        "sphere" => {
            println!("Optimizing sphere function: f(x) = Σx²");
            OptimizationProblem {
                objective: Box::new(|x: &[f64]| x.iter().map(|xi| xi * xi).sum()),
                initial_params: vec![5.0; 5],
                bounds: Some(vec![(-10.0, 10.0); 5]),
                dimension: 5,
            }
        }
        "rosenbrock" => {
            println!("Optimizing Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²");
            OptimizationProblem {
                objective: Box::new(|x: &[f64]| {
                    let a = 1.0 - x[0];
                    let b = x[1] - x[0] * x[0];
                    a * a + 100.0 * b * b
                }),
                initial_params: vec![-1.0, 1.0],
                bounds: Some(vec![(-5.0, 5.0); 2]),
                dimension: 2,
            }
        }
        "rastrigin" => {
            println!("Optimizing Rastrigin function: f(x) = 10n + Σ[x² - 10cos(2πx)]");
            OptimizationProblem {
                objective: Box::new(|x: &[f64]| {
                    let a = 10.0;
                    let n = x.len() as f64;
                    a * n + x.iter()
                        .map(|xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                        .sum::<f64>()
                }),
                initial_params: vec![2.0; 4],
                bounds: Some(vec![(-5.12, 5.12); 4]),
                dimension: 4,
            }
        }
        "portfolio" => {
            println!("Optimizing portfolio allocation (maximize return, minimize risk)");
            OptimizationProblem {
                objective: Box::new(|x: &[f64]| {
                    // Ensure weights sum to 1
                    let sum: f64 = x.iter().sum();
                    let normalized: Vec<f64> = x.iter().map(|&xi| xi / sum).collect();
                    
                    // Expected returns for each asset
                    let returns = vec![0.12, 0.10, 0.15, 0.08, 0.13];
                    let expected_return: f64 = normalized.iter()
                        .zip(returns.iter())
                        .map(|(w, r)| w * r)
                        .sum();
                    
                    // Risk (simplified as sum of squared weights)
                    let risk: f64 = normalized.iter().map(|w| w * w).sum::<f64>().sqrt();
                    
                    // Objective: maximize return while minimizing risk
                    // We minimize negative return + risk penalty
                    -expected_return + 0.5 * risk
                }),
                initial_params: vec![0.2; 5], // Equal weights initially
                bounds: Some(vec![(0.0, 1.0); 5]), // Weights between 0 and 1
                dimension: 5,
            }
        }
        _ => {
            println!("Unknown problem type. Using sphere function.");
            OptimizationProblem {
                objective: Box::new(|x: &[f64]| x.iter().map(|xi| xi * xi).sum()),
                initial_params: vec![5.0; 5],
                bounds: Some(vec![(-10.0, 10.0); 5]),
                dimension: 5,
            }
        }
    };
    
    // Run optimization
    println!("\nStarting optimization...");
    println!("Initial parameters: {:?}", problem.initial_params);
    println!("Initial value: {:.6}", (problem.objective)(&problem.initial_params));
    println!();
    
    let start = std::time::Instant::now();
    let result = optimizer.optimize(&problem).await?;
    let elapsed = start.elapsed();
    
    // Display results
    println!("✅ Optimization complete!");
    println!("Final parameters: {:?}", result.params);
    println!("Final value: {:.6}", result.value);
    println!("Improvement: {:.2}%", 
        (result.initial_value - result.value) / result.initial_value.abs() * 100.0);
    println!("Iterations: {}", result.iterations);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Time: {:.2}ms", elapsed.as_millis());
    
    // Show optimization trajectory
    if !result.history.is_empty() {
        println!("\nOptimization trajectory:");
        let step = result.history.len() / 10;
        for (i, &value) in result.history.iter().step_by(step.max(1)).enumerate() {
            let progress = i * step.max(1);
            println!("  Step {:3}: {:.6}", progress, value);
        }
    }
    
    // Get performance metrics
    let metrics = optimizer.get_performance_metrics();
    let stats = optimizer.get_execution_stats();
    
    println!("\nPerformance Metrics:");
    println!("  Mean improvement: {:.2}%", metrics.mean_improvement * 100.0);
    println!("  Success rate: {:.2}%", metrics.success_rate * 100.0);
    println!("  Samples: {}", metrics.sample_size);
    
    if stats.count > 0 {
        println!("\nExecution Statistics:");
        println!("  Average time: {:.2}ms", stats.avg_time_ms);
        println!("  Min time: {:.2}ms", stats.min_time_ms);
        println!("  Max time: {:.2}ms", stats.max_time_ms);
    }
    
    println!("\nAvailable problems: sphere, rosenbrock, rastrigin, portfolio");
    
    Ok(())
}