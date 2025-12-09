//! # Tengri Holographic Cortex Demo
//!
//! Demonstrates the 4-engine pBit topology with hyperbolic embedding.
//!
//! ## Run
//! ```bash
//! cargo run --example cortex_demo --release
//! ```

use tengri_holographic_cortex::*;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     TENGRI HOLOGRAPHIC CORTEX - 11D Hyperbolic Lattice       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Print verified constants
    println!("📐 Wolfram-Verified Constants:");
    println!("   • Ising T_c = {:.15}", ISING_CRITICAL_TEMP);
    println!("   • STDP τ₊ = {} ms", STDP_TAU_PLUS);
    println!("   • Hyperbolic dim = {} (Lorentz {})", HYPERBOLIC_DIM, LORENTZ_DIM);
    println!();
    
    // Create 4-engine cortex
    let config = TopologyConfig {
        engine_config: EngineConfig {
            num_pbits: 1024,
            temperature: ISING_CRITICAL_TEMP,
            seed: Some(42),
            ..Default::default()
        },
        base_temperature: ISING_CRITICAL_TEMP,
        enable_stdp: true,
        coupling_scale: 0.1,
    };
    
    let mut cortex = Cortex4::new(config);
    
    println!("🧠 Cortex4 Configuration:");
    println!("   • Engines: {}", NUM_ENGINES);
    println!("   • pBits per engine: {}", cortex.engines[0].num_pbits());
    println!("   • Total pBits: {}", NUM_ENGINES * cortex.engines[0].num_pbits());
    println!("   • Base temperature: {:.6}", ISING_CRITICAL_TEMP);
    println!();
    
    // Run simulation
    let num_steps = 1000;
    println!("⚡ Running {} simulation steps...", num_steps);
    
    let start = Instant::now();
    for _ in 0..num_steps {
        cortex.step();
    }
    let elapsed = start.elapsed();
    
    let steps_per_sec = num_steps as f64 / elapsed.as_secs_f64();
    
    println!("   • Time: {:.2?}", elapsed);
    println!("   • Steps/sec: {:.0}", steps_per_sec);
    println!("   • µs/step: {:.1}", elapsed.as_micros() as f64 / num_steps as f64);
    println!();
    
    // Print spike rates
    let rates = cortex.spike_rates();
    println!("📊 Engine Spike Rates:");
    for (i, rate) in rates.iter().enumerate() {
        let bar_len = (rate * 40.0) as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(40 - bar_len);
        println!("   • Engine {}: {:.3} [{}]", i, rate, bar);
    }
    println!();
    
    // Print coupling matrix
    println!("🔗 4-Engine Coupling Matrix (Wolfram-verified eigenvalues: [2.5, -1.5, -0.5, -0.5]):");
    for i in 0..4 {
        print!("   [");
        for j in 0..4 {
            print!(" {:.1}", cortex.couplings.coupling(i, j));
        }
        println!(" ]");
    }
    println!();
    
    // Print global hyperbolic embedding
    if let Some(embedding) = cortex.global_embedding() {
        println!("🌀 Global Hyperbolic Embedding (Lorentz H¹¹):");
        println!("   • x₀ (time): {:.6}", embedding.time());
        println!("   • Spatial: {:?}", &embedding.spatial()[..5]);
        println!("   • Constraint ⟨x,x⟩_L = {:.10} (should be -1)", embedding.lorentz_constraint());
        
        // Convert to Poincaré ball
        let poincare = embedding.to_poincare();
        let norm: f64 = poincare.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("   • Poincaré norm: {:.6} (must be < 1)", norm);
    }
    println!();
    
    // Memory fabric demo
    println!("💾 Memory Fabric (HNSW + LSH):");
    let mut fabric = MemoryFabric::default();
    
    // Insert some memories
    let start = Instant::now();
    for i in 0..1000 {
        let v: Vec<f64> = (0..11).map(|j| (i * 11 + j) as f64 / 10000.0).collect();
        fabric.insert(v);
    }
    println!("   • Inserted 1000 memories in {:.2?}", start.elapsed());
    
    // Query
    let query = vec![0.5; 11];
    let start = Instant::now();
    let results = fabric.query(&query, 5);
    println!("   • Query k=5 in {:.2?}", start.elapsed());
    println!("   • Top results: {:?}", results.iter().map(|(id, d)| (*id, format!("{:.4}", d))).collect::<Vec<_>>());
    println!();
    
    // GPU performance estimate
    println!("🎮 GPU Performance Estimate (AMD Radeon 6800XT):");
    let gpu_config = GpuConfig {
        num_nodes: 1_000_000,
        num_edges: 10_000_000,
        ..Default::default()
    };
    
    let mem = MemoryEstimate::from_config(&gpu_config);
    let perf = PerfEstimate::radeon_6800xt(&gpu_config);
    
    println!("   • Nodes: 1M, Edges: 10M");
    println!("   • Memory: {}", mem.format());
    println!("   • Message pass: {:.3} ms", perf.message_pass_ms);
    println!("   • Möbius aggregate: {:.3} ms", perf.mobius_aggregate_ms);
    println!("   • Estimated throughput: {:.0} ticks/sec", perf.ticks_per_second);
    println!();
    
    // SIMD performance demo
    println!("⚡ SIMD Performance (AVX2):");
    let fields: Vec<f32> = (0..10000).map(|i| (i as f32 - 5000.0) * 0.001).collect();
    let biases = vec![0.0f32; 10000];
    let mut probs = vec![0.0f32; 10000];
    
    let start = Instant::now();
    for _ in 0..1000 {
        simd::pbit_probabilities_batch(&fields, &biases, 1.0, &mut probs);
    }
    let elapsed = start.elapsed();
    
    println!("   • 10K pBit probabilities × 1000 iterations: {:.2?}", elapsed);
    println!("   • Throughput: {:.1}M samples/sec", 10.0 * 1000.0 / elapsed.as_secs_f64());
    println!();
    
    println!("✅ Demo complete!");
}
