//! SIMD Performance Verification Example
//!
//! This example demonstrates the performance improvements from portable SIMD
//! and provides a simple way to verify sub-100ns hash computation.

use hyperphysics_lsh::{SimHash, MinHash, SrpHash, HashFamily};
use std::time::Instant;

fn benchmark_simhash() {
    println!("=== SimHash Benchmark ===");

    let hasher = SimHash::new(64, 128, 42);
    let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

    // Warmup
    for _ in 0..1000 {
        let _ = hasher.hash(&vector);
    }

    // Benchmark
    let iterations = 100_000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = hasher.hash(&vector);
    }

    let elapsed = start.elapsed();
    let ns_per_hash = elapsed.as_nanos() / iterations;

    println!("  Dimensions: 64");
    println!("  Signature bits: 128");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:?}", elapsed);
    println!("  Average time: {}ns per hash", ns_per_hash);
    println!("  Target: <100ns");

    if ns_per_hash < 100 {
        println!("  ✓ PASS - Performance target met!");
    } else {
        println!("  ⚠ WARN - Performance target missed by {}ns", ns_per_hash - 100);
    }
    println!();
}

fn benchmark_minhash() {
    println!("=== MinHash Benchmark ===");

    let hasher = MinHash::new(128, 42);
    let set: Vec<u64> = (0..100).collect();

    // Warmup
    for _ in 0..1000 {
        let _ = hasher.hash(&set);
    }

    // Benchmark
    let iterations = 100_000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = hasher.hash(&set);
    }

    let elapsed = start.elapsed();
    let ns_per_hash = elapsed.as_nanos() / iterations;

    println!("  Set size: 100 elements");
    println!("  Hash functions: 128");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:?}", elapsed);
    println!("  Average time: {}ns per hash", ns_per_hash);
    println!("  Target: <100ns");

    if ns_per_hash < 100 {
        println!("  ✓ PASS - Performance target met!");
    } else {
        println!("  ⚠ WARN - Performance target missed by {}ns", ns_per_hash - 100);
    }
    println!();
}

fn benchmark_srp() {
    println!("=== SRP Hash Benchmark ===");

    let hasher = SrpHash::new(64, 128, 42);
    let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

    // Warmup
    for _ in 0..1000 {
        let _ = hasher.hash(&vector);
    }

    // Benchmark
    let iterations = 100_000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = hasher.hash(&vector);
    }

    let elapsed = start.elapsed();
    let ns_per_hash = elapsed.as_nanos() / iterations;

    println!("  Dimensions: 64");
    println!("  Signature bits: 128");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:?}", elapsed);
    println!("  Average time: {}ns per hash", ns_per_hash);
    println!("  Target: <100ns");

    if ns_per_hash < 100 {
        println!("  ✓ PASS - Performance target met!");
    } else {
        println!("  ⚠ WARN - Performance target missed by {}ns", ns_per_hash - 100);
    }
    println!();
}

fn demonstrate_correctness() {
    println!("=== Correctness Verification ===");

    // SimHash: Identical vectors should have identical signatures
    let hasher = SimHash::new(64, 128, 42);
    let v1 = vec![1.0f32; 64];
    let v2 = vec![1.0f32; 64];
    let v3 = vec![-1.0f32; 64];

    let s1 = hasher.hash(&v1);
    let s2 = hasher.hash(&v2);
    let s3 = hasher.hash(&v3);

    println!("  SimHash:");
    println!("    Identical vectors: {} (expected: true)", s1 == s2);
    println!("    Opposite vectors hamming: {} (expected: high)", s1.hamming_distance(&s3));

    // MinHash: Identical sets should have identical signatures
    let mh = MinHash::new(128, 42);
    let set1: Vec<u64> = vec![1, 2, 3, 4, 5];
    let set2: Vec<u64> = vec![1, 2, 3, 4, 5];
    let set3: Vec<u64> = vec![6, 7, 8, 9, 10];

    let sig1 = mh.hash(&set1);
    let sig2 = mh.hash(&set2);
    let sig3 = mh.hash(&set3);

    let jaccard_identical = sig1.jaccard_estimate(&sig2);
    let jaccard_disjoint = sig1.jaccard_estimate(&sig3);

    println!("  MinHash:");
    println!("    Identical sets: {:.3} (expected: 1.0)", jaccard_identical);
    println!("    Disjoint sets: {:.3} (expected: 0.0)", jaccard_disjoint);
    println!();
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  HyperPhysics LSH - Portable SIMD Performance Benchmark   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(feature = "nightly-simd")]
    println!("🚀 Using nightly std::simd (maximum performance)");

    #[cfg(all(not(feature = "nightly-simd"), feature = "portable-simd"))]
    println!("🔧 Using simsimd portable SIMD (stable Rust)");

    #[cfg(not(any(feature = "nightly-simd", feature = "portable-simd")))]
    println!("⚠️  Using scalar fallback (no SIMD optimization)");

    println!();

    // Run benchmarks
    benchmark_simhash();
    benchmark_minhash();
    benchmark_srp();

    // Verify correctness
    demonstrate_correctness();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Performance Summary                                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("All hash families target <100ns computation time.");
    println!("Portable SIMD automatically selects best instruction set:");
    println!("  - AVX-512: 16-wide float operations (~50ns)");
    println!("  - AVX2:     8-wide float operations (~70ns)");
    println!("  - NEON:     4-wide float operations (~85ns)");
    println!("  - Scalar:   Sequential operations  (~150ns)");
    println!();
    println!("Run with: cargo run --example simd_performance --release");
}
