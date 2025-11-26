//! Natural Drift Optimizer Demonstration
//!
//! This example demonstrates the key concepts of the Natural Drift Optimizer
//! based on Maturana & Varela's autopoiesis theory.
//!
//! Run with: cargo run --example natural_drift_demo -p hyperphysics-optimization

use hyperphysics_optimization::natural_drift::NaturalDriftOptimizer;
use nalgebra::DVector;

fn main() {
    println!("=== Natural Drift Optimizer Demo ===");
    println!("Based on Maturana & Varela's Autopoiesis Theory (1987)\n");

    // Example 1: Basic drift in 2D space
    println!("Example 1: Basic Natural Drift (2D)");
    println!("{}", "-".repeat(50));
    basic_drift_2d();

    println!("\n");

    // Example 2: Satisficing behavior (not optimizing)
    println!("Example 2: Satisficing vs Optimizing");
    println!("{}", "-".repeat(50));
    demonstrate_satisficing();

    println!("\n");

    // Example 3: Finding viable paths
    println!("Example 3: Finding Viable Trajectories");
    println!("{}", "-".repeat(50));
    demonstrate_path_finding();

    println!("\n");

    // Example 4: Viability boundaries
    println!("Example 4: Viability Score Analysis");
    println!("{}", "-".repeat(50));
    demonstrate_viability_scores();
}

fn basic_drift_2d() {
    let initial_state = DVector::from_vec(vec![0.0, 0.0]);
    let viability_bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];

    let mut optimizer = NaturalDriftOptimizer::with_seed(
        initial_state,
        viability_bounds,
        42,
    ).expect("Failed to create optimizer");

    println!("Initial state: [0.0, 0.0]");
    println!("Viability bounds: [-1.0, 1.0] × [-1.0, 1.0]");
    println!("\nExecuting 10 drift steps...\n");

    for i in 1..=10 {
        let result = optimizer.drift_step();
        println!(
            "Step {}: [{:6.3}, {:6.3}] - viable: {}, score: {:.4}",
            i,
            result.new_state[0],
            result.new_state[1],
            result.is_viable,
            result.viability_score
        );
    }

    println!("\nAll states remained within viable region!");
}

fn demonstrate_satisficing() {
    // Create a 1D system with large viable region
    let initial_state = DVector::from_vec(vec![0.0]);
    let viability_bounds = vec![(-10.0, 10.0)];

    let mut optimizer = NaturalDriftOptimizer::with_seed(
        initial_state,
        viability_bounds,
        42,
    ).expect("Failed to create optimizer");

    println!("System demonstrates SATISFICING (not optimizing):");
    println!("- Wanders randomly within viable region");
    println!("- Does NOT maximize/minimize position");
    println!("- ANY viable state is acceptable\n");

    let mut positions = Vec::new();
    for _ in 0..100 {
        optimizer.drift_step();
        positions.push(optimizer.current_state()[0]);
    }

    let min_pos = positions.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_pos = positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_pos: f64 = positions.iter().sum::<f64>() / positions.len() as f64;

    println!("After 100 steps:");
    println!("  Minimum position: {:.3}", min_pos);
    println!("  Maximum position: {:.3}", max_pos);
    println!("  Average position: {:.3}", avg_pos);
    println!("  Range explored:   {:.3}", max_pos - min_pos);
    println!("\nSystem explores viable space without optimizing!");
}

fn demonstrate_path_finding() {
    let initial_state = DVector::from_vec(vec![0.0, 0.0]);
    let viability_bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];

    let mut optimizer = NaturalDriftOptimizer::with_seed(
        initial_state,
        viability_bounds,
        42,
    ).expect("Failed to create optimizer");

    let target = DVector::from_vec(vec![0.7, 0.7]);
    println!("Finding viable path from [0.0, 0.0] to [0.7, 0.7]...");

    match optimizer.find_viable_path(&target, 1000) {
        Some(path) => {
            println!("\nViable path found with {} steps!", path.len());
            println!("\nFirst 5 steps:");
            for (i, state) in path.iter().take(5).enumerate() {
                let score = optimizer.viability_score(state);
                println!(
                    "  Step {}: [{:6.3}, {:6.3}] - viability: {:.4}",
                    i, state[0], state[1], score
                );
            }
            if path.len() > 5 {
                println!("  ...");
                let last = path.last().unwrap();
                let score = optimizer.viability_score(last);
                println!(
                    "  Step {}: [{:6.3}, {:6.3}] - viability: {:.4}",
                    path.len() - 1, last[0], last[1], score
                );
            }
            println!("\nKey insight: ANY viable path is acceptable (satisficing)");
        }
        None => {
            println!("\nNo viable path found within step limit");
        }
    }
}

fn demonstrate_viability_scores() {
    let initial_state = DVector::from_vec(vec![0.0, 0.0]);
    let viability_bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];

    let optimizer = NaturalDriftOptimizer::new(
        initial_state,
        viability_bounds,
    ).expect("Failed to create optimizer");

    println!("Viability Score = min normalized distance to boundaries");
    println!("  1.0 = center of region");
    println!("  0.0 = at boundary");
    println!("  <0  = outside viable region\n");

    let test_states = vec![
        (vec![0.0, 0.0], "Center"),
        (vec![0.5, 0.0], "Halfway to boundary"),
        (vec![1.0, 0.0], "At boundary"),
        (vec![1.5, 0.0], "Outside (non-viable)"),
        (vec![0.25, 0.25], "Near center"),
        (vec![-0.9, 0.9], "Near corner"),
    ];

    println!("State Analysis:");
    for (coords, description) in test_states {
        let state = DVector::from_vec(coords.clone());
        let score = optimizer.viability_score(&state);
        let viable = optimizer.is_viable(&state);
        println!(
            "  [{:5.2}, {:5.2}] {:20} - score: {:6.3}, viable: {}",
            coords[0], coords[1], description, score, viable
        );
    }

    println!("\nViability score guides system toward stable regions");
    println!("while accepting ANY viable state (satisficing principle)");
}
