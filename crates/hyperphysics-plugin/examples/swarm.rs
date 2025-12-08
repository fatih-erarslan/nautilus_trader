//! Multi-strategy swarm example
//!
//! Run with: cargo run --example swarm

use hyperphysics_plugin::prelude::*;

fn main() {
    println!("=== HyperPhysics Plugin - Multi-Strategy Swarm Example ===\n");
    
    // Define objective functions
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    
    let ackley = |x: &[f64]| {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
        let sum_cos: f64 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>() / n;
        -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
    };
    
    // Test 1: Multi-strategy swarm
    println!("1. Multi-Strategy Swarm (PSO + Grey Wolf + Whale)");
    let result = SwarmBuilder::new()
        .agents(60)
        .dimensions(10)
        .bounds(-5.0, 5.0)
        .strategies(vec![Strategy::ParticleSwarm, Strategy::GreyWolf, Strategy::Whale])
        .topology(Topology::Mesh)
        .iterations(500)
        .minimize(sphere)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Iterations: {}", result.metrics.iterations);
    println!("   Diversity: {:.3}", result.metrics.diversity);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    println!("   Strategy performance:");
    for (strategy, perf) in &result.strategy_performance {
        println!("     {:?}: {:.6}", strategy, perf);
    }
    println!();
    
    // Test 2: Hyperbolic topology
    println!("2. Hyperbolic Topology Swarm (Ackley function)");
    let result = SwarmBuilder::new()
        .agents(50)
        .dimensions(10)
        .bounds(-32.768, 32.768)
        .strategies(vec![Strategy::Cuckoo, Strategy::DifferentialEvolution])
        .topology(Topology::Hyperbolic)
        .iterations(500)
        .minimize(ackley)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 3: Ring topology
    println!("3. Ring Topology Swarm");
    let result = SwarmBuilder::new()
        .agents(30)
        .dimensions(5)
        .bounds(-10.0, 10.0)
        .strategies(vec![Strategy::ParticleSwarm])
        .topology(Topology::Ring)
        .iterations(300)
        .minimize(sphere)
        .unwrap();
    
    println!("   Best fitness: {:.6}", result.fitness);
    println!("   Time: {}ms\n", result.metrics.time_ms);
    
    // Test 4: pBit Lattice
    println!("4. pBit Lattice Simulation");
    let mut lattice = LatticeBuilder::new()
        .dimensions_2d(16, 16)
        .temperature(2.0)
        .coupling(1.0)
        .build()
        .unwrap();
    
    println!("   Initial state:");
    let state = lattice.state();
    println!("     Magnetization: {:.3}", state.magnetization);
    println!("     Energy: {:.3}", state.energy);
    
    // Anneal
    lattice.anneal(0.1, 200);
    
    println!("   After annealing:");
    let state = lattice.state();
    println!("     Magnetization: {:.3}", state.magnetization);
    println!("     Energy: {:.3}", state.energy);
    println!("     Temperature: {:.3}", lattice.temperature());
    
    println!("\n=== Complete ===");
}
