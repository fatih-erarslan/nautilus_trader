#!/usr/bin/env rust-script
//! Simple SOC validation without external dependencies

use std::time::Instant;

// Minimal SOC implementation for validation
struct SimpleSOC {
    min_points: usize,
    tolerance_factor: f64,
}

impl SimpleSOC {
    fn new() -> Self {
        Self {
            min_points: 20,
            tolerance_factor: 0.2,
        }
    }
    
    fn sample_entropy(&self, data: &[f64]) -> (f64, u64) {
        let start = Instant::now();
        
        let n = data.len();
        if n < self.min_points {
            return (0.5, start.elapsed().as_nanos() as u64);
        }
        
        // Calculate tolerance
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-12 {
            return (0.5, start.elapsed().as_nanos() as u64);
        }
        
        let tolerance = self.tolerance_factor * std_dev;
        
        // Count template matches for m=2 and m=3
        let count_2 = self.count_matches(data, 2, tolerance);
        let count_3 = self.count_matches(data, 3, tolerance);
        
        let entropy = if count_2 > 0 && count_3 > 0 {
            -(count_3 as f64 / count_2 as f64).ln()
        } else {
            0.5
        };
        
        let duration = start.elapsed().as_nanos() as u64;
        (entropy, duration)
    }
    
    fn count_matches(&self, data: &[f64], m: usize, tolerance: f64) -> u32 {
        let n = data.len();
        if n < m + 1 {
            return 0;
        }
        
        let mut count = 0u32;
        
        for i in 0..=(n - m) {
            for j in (i + 1)..=(n - m) {
                let mut matches = true;
                for k in 0..m {
                    if (data[i + k] - data[j + k]).abs() > tolerance {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }
        }
        
        count
    }
    
    fn full_analysis(&self, data: &[f64]) -> (f64, f64, u64) {
        let start = Instant::now();
        
        let (entropy, entropy_time) = self.sample_entropy(data);
        let entropy_rate = self.simple_entropy_rate(data);
        
        let total_time = start.elapsed().as_nanos() as u64;
        (entropy, entropy_rate, total_time)
    }
    
    fn simple_entropy_rate(&self, data: &[f64]) -> f64 {
        let n = data.len();
        if n < 5 {
            return 0.0;
        }
        
        // Simple predictability measure
        let mut errors = 0.0;
        for i in 2..n {
            let predicted = 2.0 * data[i-1] - data[i-2];
            let error = (data[i] - predicted).abs();
            errors += error;
        }
        
        let avg_error = errors / (n - 2) as f64;
        (1.0 + avg_error).ln()
    }
}

// Test data generators
fn generate_sine_wave(n: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
    (0..n).map(|i| {
        amplitude * (2.0 * std::f64::consts::PI * frequency * i as f64 / n as f64).sin()
    }).collect()
}

fn generate_critical_system(n: usize) -> Vec<f64> {
    let mut data = vec![1.0];
    
    for i in 1..n {
        let base = data[i-1];
        
        if i % 20 == 0 {
            // Avalanche event
            data.push(base + 5.0);
        } else if i % 5 == 0 {
            // Medium fluctuation
            data.push(base + (i as f64 * 0.1).sin());
        } else {
            // Small fluctuation
            data.push(base + 0.1 * (i as f64 * 0.1).sin());
        }
    }
    
    data
}

fn main() {
    println!("SOC Analyzer Performance Validation");
    println!("===================================");
    
    let soc = SimpleSOC::new();
    
    // Test cases
    let test_cases = vec![
        ("Sine Wave (100 points)", generate_sine_wave(100, 1.0, 1.0)),
        ("Critical System (100 points)", generate_critical_system(100)),
        ("Sine Wave (50 points)", generate_sine_wave(50, 2.0, 1.0)),
        ("Critical System (200 points)", generate_critical_system(200)),
    ];
    
    for (name, data) in test_cases {
        println!("\nTesting: {}", name);
        
        // Run multiple iterations for accurate timing
        let iterations = 100;
        let mut entropy_times = Vec::new();
        let mut total_times = Vec::new();
        let mut entropies = Vec::new();
        let mut entropy_rates = Vec::new();
        
        for _ in 0..iterations {
            let (entropy, entropy_rate, total_time) = soc.full_analysis(&data);
            let (_, entropy_time) = soc.sample_entropy(&data);
            
            entropy_times.push(entropy_time);
            total_times.push(total_time);
            entropies.push(entropy);
            entropy_rates.push(entropy_rate);
        }
        
        // Calculate statistics
        let avg_entropy_time = entropy_times.iter().sum::<u64>() / iterations;
        let avg_total_time = total_times.iter().sum::<u64>() / iterations;
        let min_entropy_time = *entropy_times.iter().min().unwrap();
        let min_total_time = *total_times.iter().min().unwrap();
        let max_entropy_time = *entropy_times.iter().max().unwrap();
        let max_total_time = *total_times.iter().max().unwrap();
        
        let avg_entropy = entropies.iter().sum::<f64>() / iterations as f64;
        let avg_entropy_rate = entropy_rates.iter().sum::<f64>() / iterations as f64;
        
        println!("  Results over {} iterations:", iterations);
        println!("    Sample Entropy: {:.3} (avg)", avg_entropy);
        println!("    Entropy Rate: {:.3} (avg)", avg_entropy_rate);
        println!();
        println!("  Performance (Sample Entropy):");
        println!("    Average: {} ns", avg_entropy_time);
        println!("    Min: {} ns", min_entropy_time);
        println!("    Max: {} ns", max_entropy_time);
        println!("    Target: 500 ns");
        println!("    Meets target: {}", avg_entropy_time <= 500);
        println!();
        println!("  Performance (Total Analysis):");
        println!("    Average: {} ns", avg_total_time);
        println!("    Min: {} ns", min_total_time);
        println!("    Max: {} ns", max_total_time);
        println!("    Target: 800 ns");
        println!("    Meets target: {}", avg_total_time <= 800);
        
        // Performance rating
        let entropy_score = if avg_entropy_time <= 500 { 
            "EXCELLENT" 
        } else if avg_entropy_time <= 1000 { 
            "GOOD" 
        } else if avg_entropy_time <= 5000 { 
            "FAIR" 
        } else { 
            "NEEDS_OPTIMIZATION" 
        };
        
        let total_score = if avg_total_time <= 800 { 
            "EXCELLENT" 
        } else if avg_total_time <= 1600 { 
            "GOOD" 
        } else if avg_total_time <= 8000 { 
            "FAIR" 
        } else { 
            "NEEDS_OPTIMIZATION" 
        };
        
        println!("    Entropy Performance: {}", entropy_score);
        println!("    Total Performance: {}", total_score);
    }
    
    // Performance summary
    println!("\n\nPerformance Summary");
    println!("==================");
    println!("Targets:");
    println!("  - Sample Entropy: ~500 ns");
    println!("  - Total SOC Index: ~800 ns");
    println!();
    println!("Implementation Features:");
    println!("  ✓ Sample entropy calculation");
    println!("  ✓ Entropy rate estimation");
    println!("  ✓ Template matching algorithm");
    println!("  ✓ Performance measurement");
    println!("  ✓ Multiple test cases");
    println!();
    println!("Optimization notes:");
    println!("  - SIMD can provide 2-4x speedup for larger datasets");
    println!("  - Cache-aligned memory access improves performance");
    println!("  - Lookup tables can accelerate template matching");
    println!("  - Parallel processing helps with batch analysis");
    
    println!("\nValidation complete!");
}