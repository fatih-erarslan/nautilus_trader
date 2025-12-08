//! Simple optimization example
//!
//! Run with: cargo run --example optimize

use hyperphysics_plugin::prelude::*;

fn main() {
    println!("=== HyperPhysics Plugin - Optimization Example ===\n");
    
    // Define test functions
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    
    let rastrigin = |x: &[f64]| {
        let n = x.len() as f64;
        10.0 * n + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
    };
    
    let rosenbrock = |x: &[f64]| {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    };
    
    // Test 1: Quick optimization
    println!("1. Quick Optimization (Sphere, 10D)");
    let result = HyperPhysics::quick_optimize(10, sphere).unwrap();
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Iterations: {}", result.metrics.iterations);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 2: Builder pattern with Grey Wolf
    println!("2. Grey Wolf Optimization (Rastrigin, 10D)");
    let result = HyperPhysics::optimize()
        .dimensions(10)
        .bounds(-5.12, 5.12)
        .strategy(Strategy::GreyWolf)
        .population(50)
        .iterations(500)
        .minimize(rastrigin)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Strategy: {:?}", result.strategy);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 3: Whale Optimization
    println!("3. Whale Optimization (Rosenbrock, 5D)");
    let result = HyperPhysics::optimize()
        .dimensions(5)
        .bounds(-5.0, 10.0)
        .strategy(Strategy::Whale)
        .iterations(1000)
        .minimize(rosenbrock)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Best solution: {:?}", result.solution.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 4: Adaptive strategy
    println!("4. Adaptive Strategy (Sphere, 20D)");
    let result = HyperPhysics::optimize()
        .dimensions(20)
        .bounds(-100.0, 100.0)
        .strategy(Strategy::Adaptive)
        .population(100)
        .iterations(1000)
        .minimize(sphere)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Evaluations: {}", result.metrics.evaluations);
    println!("   Diversity: {:.3}", result.metrics.diversity);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 5: Benchmark multiple strategies
    println!("5. Strategy Benchmark (Sphere, 10D)");
    let results = HyperPhysics::benchmark(10, sphere).unwrap();
    
    for (strategy, result) in results {
        println!("   {:?}: {:.6} ({}ms)", strategy, result.fitness, result.metrics.time_ms);
    }
    
    println!("\n=== Complete ===");
}
