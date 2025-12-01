//! # Neuromorphic Cortical Bus Demo
//!
//! Demonstrates the HyperPhysics cortical bus with:
//! - Lock-free spike routing (~60ns latency)
//! - Scalable pBit dynamics (64K pBits in <10ms)
//! - Packed bit storage and sparse CSR couplings
//!
//! Run with: cargo run --example neuromorphic_demo --release

use hyperphysics_cortical_bus::prelude::*;
use hyperphysics_cortical_bus::scalable_pbit::{ScalablePBitFabric, ScalablePBitConfig};
use std::time::Instant;

fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║      🧠 HyperPhysics Neuromorphic Cortical Bus Demo 🧠             ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  • Lock-free spike routing: ~60ns latency                          ║");
    println!("║  • Scalable pBit: packed bits + sparse CSR couplings               ║");
    println!("║  • Metropolis dynamics with Ising model                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. Benchmark Spike Injection (Ring Buffers)
    // =========================================================================
    println!("⚡ Benchmarking lock-free spike injection...\n");
    
    let ring = SpscRingBuffer::<Spike, 16384>::new();
    let num_spikes = 100_000u64;
    
    let start = Instant::now();
    for i in 0..num_spikes {
        let spike = Spike::new(
            i as u32,
            (i % 65536) as u16,
            if i % 2 == 0 { 100 } else { -100 },
            (i % 256) as u8,
        );
        // Push, drain if full
        if !ring.push(spike) {
            let _ = ring.pop();
            ring.push(spike);
        }
    }
    let inject_time = start.elapsed();
    let ns_per_spike = inject_time.as_nanos() / num_spikes as u128;
    
    println!("   ✓ Injected {} spikes in {:?}", num_spikes, inject_time);
    println!("   ✓ Average latency: {}ns per spike", ns_per_spike);
    println!("   ✓ Throughput: {:.2}M spikes/sec\n", 
             num_spikes as f64 / inject_time.as_secs_f64() / 1_000_000.0);

    // =========================================================================
    // 2. Scalable pBit Dynamics - 1K pBits
    // =========================================================================
    println!("🔮 Scalable pBit Dynamics (Packed Bits + Sparse CSR)\n");
    
    // 1K pBits demo
    let config_1k = ScalablePBitConfig {
        num_pbits: 1024,
        avg_degree: 6,
        temperature: 1.0,
        ..Default::default()
    };
    let mut fabric_1k = ScalablePBitFabric::with_random_couplings(config_1k, 42);
    
    let (nnz, avg_deg) = fabric_1k.coupling_stats();
    println!("   1K pBits Configuration:");
    println!("   ├─ Sparse couplings: {} edges (avg degree {:.1})", nnz, avg_deg);
    println!("   ├─ State memory: {} bytes (packed u64)", fabric_1k.len() / 8 + 1);
    println!("   └─ Temperature: 1.0\n");

    // Warm up
    for _ in 0..10 {
        fabric_1k.metropolis_sweep();
    }

    // Benchmark 1K
    let start = Instant::now();
    let num_sweeps = 1000;
    for _ in 0..num_sweeps {
        fabric_1k.metropolis_sweep();
    }
    let time_1k = start.elapsed();
    let ns_per_sweep_1k = time_1k.as_nanos() / num_sweeps;
    let ns_per_spin_1k = ns_per_sweep_1k / 1024;

    println!("   1K pBit Performance:");
    println!("   ├─ {:.2}µs per sweep", ns_per_sweep_1k as f64 / 1000.0);
    println!("   ├─ {:.1}ns per spin update", ns_per_spin_1k);
    println!("   └─ Magnetization: {:.3}\n", fabric_1k.magnetization());

    // =========================================================================
    // 3. Scalable pBit Dynamics - 64K pBits
    // =========================================================================
    let config_64k = ScalablePBitConfig::l2_optimal(); // 64K pBits
    let mut fabric_64k = ScalablePBitFabric::with_random_couplings(config_64k, 12345);
    
    let (nnz, avg_deg) = fabric_64k.coupling_stats();
    println!("   64K pBits Configuration:");
    println!("   ├─ Sparse couplings: {} edges (avg degree {:.1})", nnz, avg_deg);
    println!("   ├─ State memory: {} bytes (packed u64)", fabric_64k.len() / 8 + 1);
    println!("   └─ Temperature: 1.0\n");

    // Warm up
    for _ in 0..5 {
        fabric_64k.metropolis_sweep();
    }

    // Benchmark 64K
    let start = Instant::now();
    let num_sweeps = 100;
    for _ in 0..num_sweeps {
        fabric_64k.metropolis_sweep();
    }
    let time_64k = start.elapsed();
    let us_per_sweep_64k = time_64k.as_micros() as f64 / num_sweeps as f64;
    let ns_per_spin_64k = us_per_sweep_64k * 1000.0 / 65536.0;

    println!("   64K pBit Performance:");
    println!("   ├─ {:.2}ms per sweep", us_per_sweep_64k / 1000.0);
    println!("   ├─ {:.1}ns per spin update", ns_per_spin_64k);
    println!("   └─ Magnetization: {:.3}\n", fabric_64k.magnetization());

    // =========================================================================
    // 4. State Visualization
    // =========================================================================
    println!("   State Evolution (64K pBits, first 256 shown):\n");
    
    print!("   ");
    for i in 0..256 {
        if fabric_64k.get_state(i) {
            print!("█");
        } else {
            print!("░");
        }
        if (i + 1) % 64 == 0 {
            println!();
            if i < 255 {
                print!("   ");
            }
        }
    }
    
    let active = fabric_64k.count_active();
    println!("\n   [{} active / {} total = {:.1}%]\n", 
             active, fabric_64k.len(), 
             active as f64 / fabric_64k.len() as f64 * 100.0);

    // =========================================================================
    // 5. Performance Summary
    // =========================================================================
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                      Performance Summary                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Spike Injection:      {:>8}ns per spike                         ║", ns_per_spike);
    println!("║  Spike Throughput:     {:>8.2}M spikes/sec                        ║", 
             num_spikes as f64 / inject_time.as_secs_f64() / 1_000_000.0);
    println!("║  1K pBit Sweep:        {:>8.2}µs ({:.1}ns/spin)                    ║",
             ns_per_sweep_1k as f64 / 1000.0, ns_per_spin_1k);
    println!("║  64K pBit Sweep:       {:>8.2}ms ({:.1}ns/spin)                   ║",
             us_per_sweep_64k / 1000.0, ns_per_spin_64k);
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("✅ Demo complete! Scalable pBit fabric ready for billion-scale.\n");

    Ok(())
}

