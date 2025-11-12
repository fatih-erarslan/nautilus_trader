//! # Hello HyperPhysics
//!
//! Basic example demonstrating the complete HyperPhysics engine

use hyperphysics_core::HyperPhysicsEngine;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════╗");
    println!("║   HyperPhysics Engine v1.0            ║");
    println!("║   pBit Dynamics on Hyperbolic Lattice ║");
    println!("╚═══════════════════════════════════════╝\n");

    // Create ROI system with 48 nodes
    println!("Creating 48-node hyperbolic lattice ({{3,7,2}} tessellation)...");
    let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0)?;

    println!("  ✓ Hyperbolic geometry (H³, K=-1)");
    println!("  ✓ pBit dynamics (Gillespie algorithm)");
    println!("  ✓ Coupling network (exponential decay)");
    println!("  ✓ Thermodynamics (Landauer principle)");
    println!("  ✓ Consciousness metrics (Φ and CI)\n");

    // Display initial state
    let initial_metrics = engine.metrics();
    println!("Initial State:");
    println!("  Nodes: {}", initial_metrics.state.num_pbits);
    println!("  Energy: {:.2e} J", initial_metrics.energy);
    println!("  Entropy: {:.2e} J/K", initial_metrics.entropy);
    println!("  Magnetization: {:.3}", initial_metrics.magnetization);
    println!();

    // Run simulation
    println!("Running simulation (100 steps)...");
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for step in 0..100 {
        engine.step_with_rng(&mut rng)?;

        if (step + 1) % 20 == 0 {
            let m = engine.metrics();
            println!("  Step {:3}: E={:.2e} J, S={:.2e} J/K, M={:.3}",
                     step + 1,
                     m.energy,
                     m.entropy,
                     m.magnetization);
        }
    }

    println!();

    // Calculate consciousness metrics
    println!("Calculating consciousness metrics...");
    let phi = engine.integrated_information()?;
    let ci = engine.resonance_complexity()?;

    println!("  Φ (Integrated Information): {:.3}", phi);
    println!("  CI (Resonance Complexity): {:.3}", ci);
    println!();

    // Final metrics
    let final_metrics = engine.metrics();
    println!("Final State:");
    println!("  Energy: {:.2e} J", final_metrics.energy);
    println!("  Entropy: {:.2e} J/K", final_metrics.entropy);
    println!("  Negentropy: {:.2e} J/K", final_metrics.negentropy);
    println!("  Magnetization: {:.3}", final_metrics.magnetization);
    println!("  Causal Density: {:.3}", final_metrics.causal_density);
    println!();

    // Thermodynamic verification
    println!("Thermodynamic Verification:");
    println!("  Second Law: {}", if final_metrics.second_law_satisfied {
        "✓ SATISFIED (ΔS ≥ 0)"
    } else {
        "✗ VIOLATED"
    });
    println!("  Landauer Bound: {}", if final_metrics.landauer_bound_satisfied {
        "✓ SATISFIED (E ≥ kT ln 2)"
    } else {
        "✗ VIOLATED"
    });
    println!();

    println!("Simulation complete!");
    println!("Total events: {}", final_metrics.state.events);
    println!("Simulation time: {:.3e} s", final_metrics.state.time);

    Ok(())
}
